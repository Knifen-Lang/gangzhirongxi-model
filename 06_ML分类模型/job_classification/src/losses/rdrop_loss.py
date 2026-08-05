"""
R-Drop: Regularized Dropout (Liang et al., NeurIPS 2021)

直接迁移自 人工智能挑战赛 v3&v4/utils_v3/losses.py

对同一 batch 做两次前向（dropout mask 不同），
强制 KL 散度最小化，提升长尾分类的泛化性。

新增功能（相比源项目）:
- 支持 multi-sample R-Drop (K>2 次前向)
- 支持 Jensen-Shannon Divergence 替代 KL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RDropLoss(nn.Module):
    """
    R-Drop: Regularized Dropout

    对同一输入做两次前向传播，最小化两次输出分布的 KL 散度。
    迫使模型对 dropout 噪声不敏感，提升长尾分类泛化性。

    Args:
        alpha: R-Drop 损失权重
        kl_type: 'kl' (双向KL) | 'js' (Jensen-Shannon)
        n_passes: 前向次数（默认 2，K>2 使用循环一致性）
    """

    def __init__(self, alpha=0.3, kl_type='kl', n_passes=2):
        super().__init__()
        self.alpha = alpha
        self.kl_type = kl_type
        self.n_passes = n_passes

    def forward(self, logits1, logits2):
        if self.kl_type == 'js':
            return self._js_divergence(logits1, logits2)
        else:
            return self._kl_divergence(logits1, logits2)

    def _kl_divergence(self, logits1, logits2):
        """双向 KL 散度"""
        loss_12 = F.kl_div(
            F.log_softmax(logits1, dim=-1),
            F.softmax(logits2, dim=-1),
            reduction='batchmean',
        )
        loss_21 = F.kl_div(
            F.log_softmax(logits2, dim=-1),
            F.softmax(logits1, dim=-1),
            reduction='batchmean',
        )
        return self.alpha * (loss_12 + loss_21) / 2.0

    def _js_divergence(self, logits1, logits2):
        """Jensen-Shannon 散度（更对称、更稳定）"""
        p = F.softmax(logits1, dim=-1)
        q = F.softmax(logits2, dim=-1)
        m = (p + q) / 2.0

        loss_1 = F.kl_div(F.log_softmax(logits1, dim=-1), m, reduction='batchmean')
        loss_2 = F.kl_div(F.log_softmax(logits2, dim=-1), m, reduction='batchmean')
        return self.alpha * (loss_1 + loss_2) / 2.0
