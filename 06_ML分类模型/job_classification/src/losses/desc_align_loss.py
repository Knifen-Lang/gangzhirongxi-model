"""
岗位描述对齐损失 (Description Alignment Loss)

直接迁移自 人工智能挑战赛 v3&v4/utils_v3/losses.py

让 [CLS] 嵌入与正确岗位描述的原型向量对齐。
相对于原型矩阵做余弦相似度 → CrossEntropy。

适配: 关系描述 → 岗位描述
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DescAlignLoss(nn.Module):
    """
    描述对齐损失

    强制 encoder 的 [CLS] 输出与正确岗位的描述原型对齐。
    temperature=0.1 防止多类 softmax 梯度稀释。

    Args:
        prototype_matrix: (num_jobs, hidden_size) 岗位描述原型矩阵
        temperature: softmax 温度
    """

    def __init__(self, prototype_matrix, temperature=0.1):
        super().__init__()
        self.register_buffer(
            'prototype', torch.from_numpy(prototype_matrix).float()
        )
        self.temperature = temperature

    def forward(self, cls_embedding, labels):
        cls_norm = F.normalize(cls_embedding, dim=-1, eps=1e-8)
        proto_norm = F.normalize(
            self.prototype.to(cls_embedding.device), dim=-1, eps=1e-8
        )
        sim = torch.matmul(cls_norm, proto_norm.t()) / self.temperature
        loss = F.cross_entropy(sim, labels)
        return loss
