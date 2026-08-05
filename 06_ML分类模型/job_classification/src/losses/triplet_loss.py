"""
Large-Margin Triplet Loss — 混淆岗位簇分离

直接迁移自 人工智能挑战赛 v3&v4/utils_v3/losses.py

对每个 anchor 样本，拉近同类样本，推远异类样本。
特别适用于相似岗位的细粒度区分（如 Java后端 vs Go后端）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletMarginLoss(nn.Module):
    """
    大间隔三元组损失 (LM-ProtoNet, CIKM 2019)

    选择 hardest negative（最相似的异类样本）和 random easy positive。

    Args:
        margin: 正负样本间的最小间隔（默认 0.3）
    """

    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, dim=-1, eps=1e-8)
        sim = torch.matmul(embeddings, embeddings.t())

        labels = labels.reshape(-1, 1)
        pos_mask = (labels == labels.t()).float()
        pos_mask = pos_mask * (1.0 - torch.eye(
            labels.shape[0], device=labels.device
        ))

        # Hardest negative: 最相似的异类样本
        all_neg_sim = sim * (1.0 - pos_mask) - 1e8 * pos_mask
        hardest_neg_sim = all_neg_sim.max(dim=1).values

        # Random easy positive: 随机选择 50% 的正样本
        rand = torch.rand(pos_mask.shape, device=pos_mask.device)
        sampled_pos_mask = pos_mask * (rand > 0.5)
        easy_pos_sim = (sim * sampled_pos_mask).max(dim=1).values
        easy_pos_sim[sampled_pos_mask.sum(dim=1) == 0] = 0.0

        has_pos = pos_mask.sum(dim=1) > 0
        has_neg = (1.0 - pos_mask).sum(dim=1) > 0
        valid = has_pos & has_neg

        if valid.float().sum() == 0:
            return torch.tensor(0.0, device=embeddings.device)

        loss = F.relu(
            hardest_neg_sim[valid] - easy_pos_sim[valid] + self.margin
        )
        return loss.mean()
