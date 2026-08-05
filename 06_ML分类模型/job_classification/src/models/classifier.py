"""
岗位分类器 — 迁移自人工智能挑战赛 v3/v4 关系提取模型

核心模型:
- JobClassifier: DeBERTa/RoBERTa encoder + 线性分类头（支持原型矩阵初始化）
- PrototypeClassifier: 基于岗位描述原型的余弦相似度分类
- HierarchicalJobClassifier: 岗位族 → 岗位方向 → 具体岗位 层级分类

源码来源: 人工智能挑战赛 v3/utils_v3/model.py + v3&v4/utils_v3/model.py
适配: (Subject, Object) → (简历文本/skill_text, 岗位描述/job_title)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class JobClassifier(nn.Module):
    """
    Encoder + 线性分类头 — 岗位分类核心模型

    支持: BERT, RoBERTa, DeBERTa-v3, BGE 等 HuggingFace 模型
    分类头权重可选 岗位描述原型矩阵初始化

    Args:
        model_name: HuggingFace 模型名
        num_labels: 岗位类别数
        dropout: dropout 比例
        prototype_matrix: 可选 (num_labels, hidden_size) 原型矩阵
        proto_init_scale: 原型初始化缩放因子
    """

    def __init__(self, model_name, num_labels, dropout=0.15,
                 prototype_matrix=None, proto_init_scale=1.0):
        super().__init__()
        self.encoder = _create_encoder(model_name)
        self.hidden_size = _infer_hidden_size(self.encoder)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, num_labels)

        if prototype_matrix is not None:
            proto = torch.from_numpy(prototype_matrix).float() * proto_init_scale
            if proto.shape == self.classifier.weight.data.shape:
                self.classifier.weight.data.copy_(proto)
                self.classifier.bias.data.zero_()
            elif proto.shape[::-1] == self.classifier.weight.data.shape:
                self.classifier.weight.data.copy_(proto.T)
                self.classifier.bias.data.zero_()
            else:
                raise ValueError(
                    f'Prototype shape {proto.shape} != classifier weight '
                    f'{self.classifier.weight.data.shape}'
                )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        if token_type_ids is not None:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        else:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        cls_embedding = outputs.last_hidden_state[:, 0, :]
        if self.training:
            cls_embedding = self.dropout(cls_embedding)
        logits = self.classifier(cls_embedding)
        return logits, cls_embedding, outputs.last_hidden_state

    def get_embeddings(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        if token_type_ids is not None:
            kwargs['token_type_ids'] = token_type_ids
        outputs = self.encoder(**kwargs)
        return outputs.last_hidden_state[:, 0, :]


class PrototypeClassifier(nn.Module):
    """
    原型分类器 — 基于 [CLS] 嵌入与岗位原型矩阵的余弦相似度分类

    用途:
    - Stage 0 岗位描述匹配预训练：让 encoder 学习岗位描述语义
    - 推理时辅助：为冷门岗位提供语义检索级别的先验

    Args:
        prototype_matrix: (num_labels, hidden_size) 岗位描述嵌入矩阵
        temperature: softmax 温度，越大越"锐利"
    """

    def __init__(self, prototype_matrix, temperature=0.1):
        super().__init__()
        self.register_buffer(
            'prototype', torch.from_numpy(prototype_matrix).float()
        )
        self.temperature = temperature

    def forward(self, cls_embedding):
        cls_norm = F.normalize(cls_embedding, dim=-1, eps=1e-8)
        proto_norm = F.normalize(self.prototype, dim=-1, eps=1e-8)
        sim = torch.matmul(cls_norm, proto_norm.t()) / self.temperature
        return sim


class HierarchicalJobClassifier(nn.Module):
    """
    层级岗位分类器 — 岗位族 → 岗位方向 → 具体岗位

    两级分类：粗粒度岗位族 → 细粒度具体岗位
    粗粒度预测对细粒度施加结构性先验，缩小候选范围

    Args:
        hidden_size: encoder 输出维度
        num_family: 岗位族数量 (~6-10)
        num_job: 具体岗位数量 (~80)
        family_to_job_mask: (num_family, num_job) 归属矩阵
        dropout: dropout 比例
    """

    def __init__(self, hidden_size, num_family, num_job,
                 family_to_job_mask=None, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.family_head = nn.Linear(hidden_size, num_family)
        self.job_head = nn.Linear(hidden_size, num_job)

        if family_to_job_mask is not None:
            self.register_buffer(
                'family_to_job_mask',
                torch.from_numpy(family_to_job_mask).float()
            )
        else:
            self.family_to_job_mask = None

    def forward(self, h):
        h = self.dropout(h)
        family_logits = self.family_head(h)
        job_logits = self.job_head(h)

        if self.family_to_job_mask is not None:
            family_probs = F.softmax(family_logits, dim=-1)
            job_from_family = torch.matmul(family_probs, self.family_to_job_mask)
            job_logits = job_logits + torch.log(job_from_family + 1e-8)

        return job_logits, family_logits


# ═══════════════════════════════════════════════════════════════
#  Helper Functions
# ═══════════════════════════════════════════════════════════════

def _create_encoder(model_name):
    """创建 HuggingFace encoder 并配置 dropout"""
    from transformers import AutoModel
    model = AutoModel.from_pretrained(model_name)

    cfg = model.config
    for attr in ['classifier_dropout', 'pooler_dropout',
                  'hidden_dropout_prob', 'attention_probs_dropout_prob']:
        if hasattr(cfg, attr):
            setattr(cfg, attr, 0.1)

    return model


def _infer_hidden_size(encoder):
    """推断 encoder 的 hidden_size"""
    cfg = encoder.config
    if hasattr(cfg, 'hidden_size'):
        return cfg.hidden_size
    if hasattr(cfg, 'd_model'):
        return cfg.d_model
    if hasattr(encoder, 'embeddings') and \
       hasattr(encoder.embeddings, 'word_embeddings'):
        return encoder.embeddings.word_embeddings.weight.shape[-1]
    return 768


def build_job_family_mask(job_names, family_keywords=None):
    """
    从岗位名自动推断岗位族归属，构建 family → job 映射矩阵

    Args:
        job_names: 岗位名列表
        family_keywords: 可选，自定义岗位族关键词字典
                         格式: {family_id: [keyword1, keyword2, ...]}

    Returns:
        (num_family, len(job_names)) 的 0/1 矩阵
    """
    if family_keywords is None:
        family_keywords = {
            0: ['算法', 'AI', '机器', '深度', 'NLP', '图像', '推荐',
                '搜索', '语音', '风控', 'SLAM', 'AIGC', '大模型',
                '自然语言', '规控', '自动驾驶'],
            1: ['后端', 'Java', 'Go', 'Python', 'C++', 'C语言', 'Node',
                'Golang', '全栈', '服务端', '驱动', '高性能'],
            2: ['前端', 'Android', 'iOS', '鸿蒙', 'Web', 'H5', '移动端',
                'React', 'Vue'],
            3: ['数据', '分析', 'ETL', '仓库', '采集', '挖掘', '标注',
                '治理', '架构师(数据)', '开发(数据)'],
            4: ['测试', '运维', 'DevOps', 'SRE', '自动化测试', '测试开发',
                '运维开发'],
            5: ['产品', '项目经理', '技术文档', '需求分析', '售前', '售后',
                '销售技术', '数字化管理'],
            6: ['硬件', '嵌入式', 'FPGA', 'DSP', '芯片', '单片机', '射频',
                '天线', '电子', '光电子', '传感', '物联网'],
            7: ['通信', '核心网', '光网络', '无线', '数据通信',
                '云', '系统集成', '系统工程师'],
        }

    num_family = max(family_keywords.keys()) + 1
    mask = np.zeros((num_family, len(job_names)), dtype='float32')

    for i, job_name in enumerate(job_names):
        assigned = False
        for fid, keywords in family_keywords.items():
            if any(kw in job_name for kw in keywords):
                mask[fid, i] = 1.0
                assigned = True
                break
        if not assigned:
            # 默认归入"产品/管理"族
            mask[5, i] = 1.0

    return mask
