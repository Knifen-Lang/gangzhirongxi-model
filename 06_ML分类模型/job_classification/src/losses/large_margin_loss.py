"""
Large Margin Softmax Loss (LM-ProtoNet style)
=============================================
Adds cosine additive margin to ground-truth class logit before
cross-entropy, increasing inter-class separation for confusion clusters.

Migrated from: 人工智能挑战赛/relation-extraction-new/src/losses/large_margin_loss.py
Original: PaddlePaddle → Converted to: PyTorch

Formula:
  L = -log( exp(s * (cos_theta_y - m)) / (exp(s * (cos_theta_y - m)) + sum_{j!=y} exp(s * cos_theta_j)) )

Where:
  - cos_theta = logits normalized to [-1, 1] range
  - s = scale factor (default 30.0)
  - m = margin (default 0.35, range 0.2-0.5)

Use case: Fine-grained separation e.g. Java后端 vs Go后端, 数据分析 vs 数据科学
           Works in LOGIT space (unlike TripletMarginLoss which works in EMBEDDING space).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LargeMarginLoss(nn.Module):
    """
    Large Margin Softmax Loss.

    Args:
        scale: Scaling factor for logits (default 30.0).
               Higher = sharper distribution, better separation but harder to train.
        margin: Additive cosine margin for ground-truth class (default 0.35).
                Higher = more inter-class separation.
                - 0.2: mild separation, suitable for early training
                - 0.35: moderate (recommended default)
                - 0.5: aggressive, may hurt convergence on small datasets
    """

    def __init__(self, scale: float = 30.0, margin: float = 0.35):
        super().__init__()
        self.scale = scale
        self.margin = margin

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch_size, num_classes) raw classifier logits
            labels: (batch_size,) ground-truth class indices

        Returns:
            scalar loss
        """
        # Normalize logits to cosine range [-1, 1]
        max_abs = logits.abs().max() + 1e-8
        cos_theta = logits / max_abs

        # Get cosine value for ground-truth class
        one_hot = F.one_hot(labels, num_classes=cos_theta.shape[-1]).float()
        cos_theta_y = (one_hot * cos_theta).sum(dim=-1)  # (batch_size,)

        # Apply additive margin to ground-truth class
        cos_theta_y_margin = cos_theta_y - self.margin

        # Reconstruct logits: margin for GT class, unchanged for others
        logits_margin = cos_theta * (1 - one_hot) + cos_theta_y_margin.unsqueeze(-1) * one_hot

        # Scale and compute cross-entropy
        logits_margin = logits_margin * self.scale

        return F.cross_entropy(logits_margin, labels)


class LargeMarginLossV2(nn.Module):
    """
    Enhanced variant with adaptive margin based on class frequency.

    Rare classes get larger margin (harder to classify → need more separation).
    Frequent classes get smaller margin (already well-separated).

    Args:
        scale: Scaling factor (default 30.0)
        margin_min: Margin for most frequent class (default 0.2)
        margin_max: Margin for rarest class (default 0.5)
    """

    def __init__(self, scale: float = 30.0, margin_min: float = 0.2, margin_max: float = 0.5):
        super().__init__()
        self.scale = scale
        self.margin_min = margin_min
        self.margin_max = margin_max

    def forward(self, logits, labels, class_counts=None):
        """
        Args:
            logits: (batch_size, num_classes)
            labels: (batch_size,)
            class_counts: (num_classes,) optional per-class sample counts.
                          If None, uses uniform margin = (min+max)/2.
        """
        max_abs = logits.abs().max() + 1e-8
        cos_theta = logits / max_abs

        one_hot = F.one_hot(labels, num_classes=cos_theta.shape[-1]).float()
        cos_theta_y = (one_hot * cos_theta).sum(dim=-1)

        # Compute per-class margins
        if class_counts is not None:
            # Normalize: rare=1.0, frequent=0.0
            counts = class_counts.float()
            rarity = 1.0 - (counts / counts.max())  # (num_classes,)
            per_class_margin = self.margin_min + (self.margin_max - self.margin_min) * rarity
        else:
            per_class_margin = torch.full(
                (cos_theta.shape[-1],),
                (self.margin_min + self.margin_max) / 2,
                device=logits.device,
            )

        # Get margin for each sample's true class
        sample_margins = per_class_margin[labels]  # (batch_size,)

        cos_theta_y_margin = cos_theta_y - sample_margins
        logits_margin = cos_theta * (1 - one_hot) + cos_theta_y_margin.unsqueeze(-1) * one_hot
        logits_margin = logits_margin * self.scale

        return F.cross_entropy(logits_margin, labels)
