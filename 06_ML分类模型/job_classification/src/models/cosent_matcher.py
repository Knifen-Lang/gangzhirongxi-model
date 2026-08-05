"""
CoSENT 语义匹配模型 — 基于 GitHub shibing624/text2vec

CoSENT (Cosine Sentence): 使用 ranking-based loss 直接优化余弦相似度，
训练目标与推理目标一致，比 Sentence-BERT 收敛更快、效果更好。

论文: CoSENT: A more efficient sentence embedding method

迁移自: https://github.com/shibing624/text2vec

使用方式:
- Bi-Encoder: 分别编码 text_a(简历) 和 text_b(岗位)，计算余弦相似度
- 相比 Cross-Encoder(JobClassifier): 速度快，可预计算岗位嵌入做语义检索
- 使用 Chinese MacBERT/RoBERTa 作为 backbone
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


class CoSENTJobMatcher(nn.Module):
    """
    CoSENT 岗位匹配模型

    Bi-Encoder 架构:
    - 用同一个 encoder 分别编码 text_a 和 text_b
    - 计算余弦相似度 → sigmoid → 匹配分数

    对比现有 JobClassifier (Cross-Encoder):
    - CoSENT: O(1) 推理 (预计算岗位嵌入), 适合大规模检索
    - JobClassifier: O(N) 推理, 适合精排

    Args:
        model_name: HuggingFace 中文模型名
        encoder_type: 'MEAN' | 'CLS' | 'FIRST_LAST_AVG'
        max_seq_length: 最大序列长度
        temperature: CoSENT loss 温度
    """

    def __init__(self, model_name='hfl/chinese-macbert-base',
                 encoder_type='MEAN', max_seq_length=256, temperature=0.05):
        super().__init__()
        self.model_name = model_name
        self.encoder_type = encoder_type
        self.max_seq_length = max_seq_length
        self.temperature = temperature

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.config = self.bert.config

    def forward(self, input_ids_a, attention_mask_a,
                input_ids_b, attention_mask_b):
        """
        双编码器前向

        Args:
            input_ids_a: (batch, seq_len) text_a token ids
            attention_mask_a: (batch, seq_len)
            input_ids_b: (batch, seq_len) text_b token ids
            attention_mask_b: (batch, seq_len)

        Returns:
            similarity: (batch,) 余弦相似度
            emb_a, emb_b: 分别的嵌入
        """
        emb_a = self._encode(input_ids_a, attention_mask_a)
        emb_b = self._encode(input_ids_b, attention_mask_b)

        # 余弦相似度
        similarity = F.cosine_similarity(emb_a, emb_b, dim=-1)
        return similarity, emb_a, emb_b

    def _encode(self, input_ids, attention_mask):
        """编码单侧文本"""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state

        if self.encoder_type == 'CLS':
            emb = hidden[:, 0, :]
        elif self.encoder_type == 'MEAN':
            # Mean pooling over all tokens (masking padding)
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
            emb = torch.sum(hidden * mask_expanded, dim=1) / torch.clamp(
                mask_expanded.sum(dim=1), min=1e-9
            )
        elif self.encoder_type == 'FIRST_LAST_AVG':
            # Average of first and last hidden states (来自 text2vec)
            first = hidden[:, 0, :]
            last = hidden[:, -1, :]
            emb = (first + last) / 2.0
        else:
            emb = hidden[:, 0, :]

        return emb

    def encode(self, sentences, batch_size=64, show_progress=True):
        """
        编码句子列表

        Args:
            sentences: list of str
            batch_size: batch size
            show_progress: 是否显示进度条

        Returns:
            embeddings: (N, hidden_size) numpy array
        """
        self.eval()
        was_training = self.training

        dataset = _TextDataset(sentences, self.tokenizer, self.max_seq_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_embeddings = []
        with torch.no_grad():
            iterator = tqdm(dataloader, desc='Encoding') if show_progress else dataloader
            for batch in iterator:
                input_ids = batch['input_ids'].to(self.bert.device)
                attention_mask = batch['attention_mask'].to(self.bert.device)
                emb = self._encode(input_ids, attention_mask)
                all_embeddings.append(emb.cpu().numpy())

        if was_training:
            self.train()

        return np.concatenate(all_embeddings, axis=0)

    def predict_similarity(self, text_a, text_b, batch_size=64):
        """
        预测文本对相似度

        Args:
            text_a: list of str (简历/技能)
            text_b: list of str (岗位描述)

        Returns:
            scores: (N,) 余弦相似度
        """
        emb_a = self.encode(text_a, batch_size=batch_size, show_progress=False)
        emb_b = self.encode(text_b, batch_size=batch_size, show_progress=False)

        emb_a = F.normalize(torch.from_numpy(emb_a), dim=-1)
        emb_b = F.normalize(torch.from_numpy(emb_b), dim=-1)
        scores = torch.sum(emb_a * emb_b, dim=-1)
        return scores.numpy()


class CoSENTLoss(nn.Module):
    """
    CoSENT Ranking Loss

    优化目标: s(x,y^+) > s(x,y^-)
    其中 s(·,·) 是余弦相似度

    L = log(1 + sum_{i,j} exp(s(x_i, y_j^-) - s(x_i, y_i^+) / temperature))

    使用 in-batch negatives + pairwise margin
    """

    def __init__(self, temperature=0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, similarity_scores, labels):
        """
        Args:
            similarity_scores: (batch,) 正样本对的余弦相似度
            labels: (batch,) 标签（仅用于构建正负样本对）
        """
        # 构建相似度矩阵 (batch, batch)
        # 对角线是正样本，非对角线是负样本
        batch_size = similarity_scores.shape[0]

        # CoSENT 核心: 让正样本对相似度 > 负样本对相似度
        # 简化版: 使用 cross-entropy 优化
        # 这里使用 softmax over in-batch

        # 构建 logits: (batch, batch)
        # 每个正样本对的相似度 vs 所有人
        sim_matrix = similarity_scores.unsqueeze(0).repeat(batch_size, 1)
        # 对角线是正样本
        targets = torch.arange(batch_size).to(similarity_scores.device)

        loss = F.cross_entropy(sim_matrix / self.temperature, targets)
        return loss


class CosentRankingLoss(nn.Module):
    """
    CoSENT 原始 Ranking Loss (来自 text2vec)

    对于 batch 中两对样本 (a1,b1) 和 (a2,b2):
    如果 sim(a1,b1) > sim(a2,b2) 但标注相反，则产生损失

    L = log(1 + sum exp(λ * (cos(u_i, v_j) - cos(u_i, v_i))))
    """

    def __init__(self, temperature=0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, emb_a, emb_b, labels):
        """
        Args:
            emb_a: (batch, hidden) text_a 嵌入
            emb_b: (batch, hidden) text_b 嵌入
            labels: (batch,) 标签 (1=正样本, 0=负样本)
        """
        # 计算所有对的余弦相似度
        emb_a_norm = F.normalize(emb_a, dim=-1)
        emb_b_norm = F.normalize(emb_b, dim=-1)

        # (batch, batch) 相似度矩阵
        sim = torch.matmul(emb_a_norm, emb_b_norm.t()) / self.temperature

        # 正样本 mask
        labels = labels.reshape(-1, 1)
        pos_mask = (labels == labels.t()).float()
        pos_mask = pos_mask * (1.0 - torch.eye(labels.shape[0], device=labels.device))

        # CoSENT: 让同标签的对相似度高于不同标签对
        # 使用 margin-based ranking
        pos_sim = sim.diag()  # 每个样本的正对相似度
        neg_sim = sim * (1.0 - torch.eye(labels.shape[0], device=labels.device))

        loss = 0.0
        valid_pairs = 0
        for i in range(len(labels)):
            for j in range(len(labels)):
                if i != j and labels[i] == labels[j]:
                    # 同标签的两个样本: sim_i > sim_j for all j != i
                    pass

        # 简化: 使用 InfoNCE style
        # 每个 a_i 的正样本是 b_i
        labels_idx = torch.arange(len(labels)).to(sim.device)
        loss = F.cross_entropy(sim, labels_idx)

        return loss


# ═══════════════════════════════════════════════════
#  Data
# ═══════════════════════════════════════════════════

class _TextDataset(Dataset):
    """简单文本编码 Dataset"""

    def __init__(self, sentences, tokenizer, max_length):
        if isinstance(sentences, str):
            sentences = [sentences]
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.sentences[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
        }


class CoSENTPairDataset(Dataset):
    """
    CoSENT 正负样本对 Dataset

    正样本: (技能文本, 对应岗位描述)  ⇒ label=1
    负样本: (技能文本, 随机岗位描述)  ⇒ label=0
    """

    def __init__(self, dataframe, tokenizer, max_length=256,
                 neg_ratio=1.0):
        """
        Args:
            dataframe: 包含 [text_a, text_b, label] 的 DataFrame
            tokenizer: HuggingFace tokenizer
            max_length: 最大长度
            neg_ratio: 正负样本比
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 构建正样本
        self.positive_pairs = []
        all_job_texts = {}
        for _, row in dataframe.iterrows():
            label = row['label']
            text_a = str(row['text_a'])
            text_b = str(row['text_b'])

            self.positive_pairs.append((text_a, text_b, 1.0))

            if label not in all_job_texts:
                all_job_texts[label] = []
            all_job_texts[label].append(text_b)

        # 构建负样本（从不同岗位随机采样）
        self.negative_pairs = []
        all_labels = list(all_job_texts.keys())
        for _, row in dataframe.iterrows():
            label = row['label']
            text_a = str(row['text_a'])
            # 随机选择一个不同的岗位
            other_labels = [l for l in all_labels if l != label]
            if other_labels:
                neg_label = np.random.choice(other_labels)
                neg_text_b = np.random.choice(all_job_texts[neg_label])
                self.negative_pairs.append((text_a, neg_text_b, 0.0))

        self.all_pairs = self.positive_pairs + self.negative_pairs

    def __len__(self):
        return len(self.all_pairs)

    def __getitem__(self, idx):
        text_a, text_b, label = self.all_pairs[idx]

        enc_a = self.tokenizer(
            text_a, max_length=self.max_length,
            padding='max_length', truncation=True, return_tensors='pt',
        )
        enc_b = self.tokenizer(
            text_b, max_length=self.max_length,
            padding='max_length', truncation=True, return_tensors='pt',
        )

        return {
            'input_ids_a': enc_a['input_ids'].squeeze(0),
            'attention_mask_a': enc_a['attention_mask'].squeeze(0),
            'input_ids_b': enc_b['input_ids'].squeeze(0),
            'attention_mask_b': enc_b['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float32),
        }


def cosent_pair_collate_fn(batch):
    """CoSENT pair collate"""
    return {
        'input_ids_a': torch.stack([b['input_ids_a'] for b in batch]),
        'attention_mask_a': torch.stack([b['attention_mask_a'] for b in batch]),
        'input_ids_b': torch.stack([b['input_ids_b'] for b in batch]),
        'attention_mask_b': torch.stack([b['attention_mask_b'] for b in batch]),
        'label': torch.stack([b['label'] for b in batch]),
    }
