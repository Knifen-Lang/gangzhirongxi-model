"""
CWBSLoss — Competition-Weighted Balanced Softmax

直接迁移自 人工智能挑战赛 v3&v4/utils_v3/losses.py

公式:
  L = -w_y * log[exp(f_y + tau * log pi_y) / sum_j exp(f_j + tau * log pi_j)]

  w_y: 比赛权重（样本越少权重越大）
  log pi_y: 类别先验频率对数（Balanced Softmax 频率偏置校正）
  tau: 控制频率偏置强度

V4 改进: 支持自适应 tau（per-class tau）+ weight_min 防止梯度消失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CWBSLoss(nn.Module):
    """
    Competition-Weighted Balanced Softmax Loss

    Args:
        class_counts: 每个类的真实样本数列表
        counts_max: 最大样本数
        counts_min: 最小样本数
        tau: 固定 tau 值（不使用自适应时）
        tau_per_class: 可选，每个类的自适应 tau 值数组
        weight_min: 最小权重（防止头类权重→0 导致梯度消失）
    """

    def __init__(self, class_counts, counts_max, counts_min, tau=0.5,
                 tau_per_class=None, weight_min=0.1):
        super().__init__()
        self.num_classes = len(class_counts)
        self.tau = tau
        self.use_adaptive_tau = tau_per_class is not None

        # 比赛权重: 样本少 → 权重大
        raw_weights = [
            (counts_max - c + counts_min * 0.1) / (counts_max + counts_min * 0.1)
            for c in class_counts
        ]
        weights = [max(w, weight_min) for w in raw_weights]
        self.register_buffer(
            'weights', torch.tensor(weights, dtype=torch.float32)
        )

        # 类先验（对数频率）
        total = sum(class_counts)
        log_pi = [np.log(max(c / total, 1e-8)) for c in class_counts]
        self.register_buffer(
            'log_pi', torch.tensor(log_pi, dtype=torch.float32)
        )

        if tau_per_class is not None:
            self.register_buffer(
                'tau_vector',
                torch.tensor(tau_per_class, dtype=torch.float32)
            )
        else:
            self.tau_vector = None

    def forward(self, logits, labels):
        if self.use_adaptive_tau:
            tau_batch = self.tau_vector.to(labels.device)[labels].unsqueeze(1)
            balanced_logits = (
                logits +
                tau_batch * self.log_pi.to(logits.device).unsqueeze(0)
            )
        else:
            balanced_logits = (
                logits + self.tau * self.log_pi.to(logits.device)
            )

        log_probs = F.log_softmax(balanced_logits, dim=-1)
        loss_per_sample = -log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        weighted_loss = (
            loss_per_sample * self.weights.to(labels.device)[labels]
        )
        return weighted_loss.mean()
