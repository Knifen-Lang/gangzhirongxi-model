"""
Focal Loss — 长尾分类核心损失函数

直接迁移自 人工智能挑战赛 v3&v4/utils_v3/losses.py

FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

- alpha_t: 比赛权重，对少样本类放大梯度
- gamma=2: 标准设置，降低易分类样本的损失贡献
- 非对称 Label Smoothing: 头类弱平滑，尾类强平滑

新增功能（相比源项目）:
- 支持 Bi-Tempered Loss (https://github.com/google/bi-tempered-loss)
  当 t1 < 1.0 时使用 tempered softmax，对噪声标签更鲁棒
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    """
    Focal Loss for long-tail classification

    Args:
        class_counts: 每个类的样本数列表
        counts_max: 最大样本数
        counts_min: 最小样本数
        gamma: focal gamma 参数（默认 2.0）
        use_asymmetric_ls: 是否使用非对称 Label Smoothing
        ls_per_class: 每类的 label smoothing 值
        ls_weight: LS 辅助损失权重
        alpha_min: 最小 alpha 值（防止头类 alpha→0）
    """

    def __init__(self, class_counts, counts_max, counts_min, gamma=2.0,
                 use_asymmetric_ls=False, ls_per_class=None, ls_weight=0.05,
                 alpha_min=0.1):
        super().__init__()
        self.gamma = gamma
        self.use_asymmetric_ls = use_asymmetric_ls and ls_per_class is not None
        self.ls_per_class = ls_per_class
        self.ls_weight = ls_weight

        # 比赛权重作为 alpha
        alpha = [
            (counts_max - c + counts_min * 0.1) / (counts_max + counts_min * 0.1)
            for c in class_counts
        ]
        alpha = [max(a, alpha_min) for a in alpha]
        self.register_buffer(
            'alpha', torch.tensor(alpha, dtype=torch.float32)
        )

    def forward(self, logits, labels):
        if self.use_asymmetric_ls:
            ls_smoothed = cross_entropy_with_asymmetric_ls(
                logits, labels, self.ls_per_class
            )
            ce_raw = F.cross_entropy(logits, labels, reduction='none')
            ce_loss = ce_raw + self.ls_weight * (ls_smoothed - ce_raw.detach())
        else:
            ce_loss = F.cross_entropy(logits, labels, reduction='none')

        ce_loss = torch.clamp(ce_loss, max=50.0)
        pt = torch.exp(-ce_loss)
        pt = torch.clamp(pt, min=1e-7, max=1.0)

        alpha_t = self.alpha.to(labels.device)[labels]
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        loss = focal_weight * ce_loss
        return loss.mean()


def asymmetric_label_smoothing(class_counts, ls_min=0.02, ls_max=0.12):
    """
    非对称 Label Smoothing

    头类 (样本多) → ls_min: 几乎不模糊，保持精度
    尾类 (样本少) → ls_max: 强平滑，防止对少量样本过拟合

    Args:
        class_counts: 每类样本数列表
        ls_min: 最小值（头类）
        ls_max: 最大值（尾类）

    Returns:
        ls_per_class: 每类的 label smoothing 值
    """
    counts = np.array(class_counts)
    c_min, c_max = counts.min(), counts.max()
    normalized = (counts - c_min) / max(c_max - c_min, 1)
    ls = ls_max - (ls_max - ls_min) * normalized
    return ls


def cross_entropy_with_asymmetric_ls(logits, labels, ls_per_class):
    """
    非对称 Label Smoothing 交叉熵损失

    Args:
        logits: (batch_size, num_classes) 模型输出
        labels: (batch_size,) 真实标签
        ls_per_class: 每类的 label smoothing 值

    Returns:
        loss: 标量损失
    """
    n_classes = logits.shape[-1]
    ls_per_class_t = torch.tensor(
        ls_per_class, dtype=torch.float32, device=logits.device
    )
    ls_batch = ls_per_class_t[labels].unsqueeze(1)

    # 构建平滑目标
    smooth_targets = ls_batch.expand(-1, n_classes) / (n_classes - 1)
    smooth_targets.scatter_(1, labels.unsqueeze(1), 1.0 - ls_batch)

    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(smooth_targets * log_probs).sum(dim=-1)
    return loss.mean()


class BiTemperedLoss(nn.Module):
    """
    Bi-Tempered Logistic Loss (Google, 2019)

    对噪声标签和异常值更鲁棒：
    - t1 < 1.0: tempered softmax → 重尾分布，对错误标签更宽容
    - t2 > 1.0: tempered logistic → 对离群点更鲁棒

    Ref: https://github.com/google/bi-tempered-loss

    Args:
        t1: softmax 温度 (< 1.0 = tempered, = 1.0 = standard)
        t2: logistic 温度 (> 1.0 = tempered, = 1.0 = standard)
        label_smoothing: 标准 label smoothing
    """

    def __init__(self, t1=1.0, t2=1.0, label_smoothing=0.0):
        super().__init__()
        self.t1 = t1
        self.t2 = t2
        self.label_smoothing = label_smoothing

    def forward(self, logits, labels):
        return bi_tempered_logistic_loss(
            logits, labels, self.t1, self.t2, self.label_smoothing
        )


def log_t(u, t):
    """Compute log_t(u) = (u^(1-t) - 1) / (1-t) for t != 1"""
    if abs(t - 1.0) < 1e-8:
        return torch.log(u + 1e-8)
    return (u ** (1.0 - t) - 1.0) / (1.0 - t)


def exp_t(u, t):
    """Compute exp_t(u) for t != 1 (2D function inverse of log_t)"""
    if abs(t - 1.0) < 1e-8:
        return torch.exp(u)
    return torch.clamp(1.0 + (1.0 - t) * u, min=0.0) ** (1.0 / max(1.0 - t, 1e-8))


def bi_tempered_logistic_loss(logits, labels, t1, t2, label_smoothing=0.0):
    """
    Bi-Tempered Logistic Loss

    Args:
        logits: (batch_size, num_classes)
        labels: (batch_size,)
        t1: softmax temperature
        t2: logistic temperature
        label_smoothing: label smoothing factor
    """
    n_classes = logits.shape[-1]
    labels_onehot = F.one_hot(labels, n_classes).float()

    if label_smoothing > 0:
        labels_onehot = (
            labels_onehot * (1.0 - label_smoothing) +
            label_smoothing / n_classes
        )

    # Tempered softmax probabilities
    # Numerically stable: compute in log space
    logits_max = logits.max(dim=-1, keepdim=True).values
    shifted = logits - logits_max

    # Use numerical stable tempered softmax
    if abs(t1 - 1.0) < 1e-8:
        probs = F.softmax(logits, dim=-1)
    else:
        exp_vals = exp_t(shifted, t1)
        probs = exp_vals / exp_vals.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    # Tempered logistic loss
    loss = 0.0
    for j in range(n_classes):
        p_j = probs[:, j]
        y_j = labels_onehot[:, j]
        inner = y_j * log_t(p_j, t2) + (1.0 - y_j) * log_t(1.0 - p_j, t2)
        inner = torch.where(y_j > 0, log_t(p_j, t2), log_t(1.0 - p_j, t2))
        loss = loss - inner

    # Average the inner term
    return loss / n_classes
