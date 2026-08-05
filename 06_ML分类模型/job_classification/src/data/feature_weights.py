"""
Feature Weight Loader — 将 calibrated_burst 转为训练样本权重

用法:
    from src.data.feature_weights import load_sample_weights
    weights = load_sample_weights(df, "calibrated_features_v2.json")
    sampler = WeightedRandomSampler(weights, len(df), replacement=True)
"""

import json
import numpy as np

KEYWORDS = [
    "agent", "rag", "mcp", "function calling", "moe", "rlhf",
    "MLLM", "multi-agent systems", "diffusion transformers", "world models",
    "self-evolving", "ai-agent", "self-improving", "training-free",
    "knowledge distillation", "test-time", "test-time adaptation",
    "long-context", "llm-guided", "llm-driven", "deep-research",
    "synthetic data", "godot-mcp", "hermes-agent",
]


def load_calibrated_features(path="calibrated_features_v2.json"):
    """加载校准特征，返回 {keyword: calibrated_burst}"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        feat["keyword"]: feat["calibrated_burst_v2"]
        for feat in data["calibrated_features"]
    }


def compute_sample_weights(df, feature_path="calibrated_features_v2.json",
                           base_weight=1.0, max_boost=3.0):
    """
    为每条 JD 计算基于 tech 信号的样本权重。

    逻辑: JD 中包含的高 burst 关键词越多 → 权重越高
    权重范围: [base_weight, base_weight * max_boost]

    Args:
        df: DataFrame with 'skill_requirements' column
        feature_path: path to calibrated_features_v2.json
        base_weight: minimum sample weight
        max_boost: maximum weight multiplier

    Returns:
        numpy array of sample weights
    """
    calib = load_calibrated_features(feature_path)

    weights = np.ones(len(df)) * base_weight

    for i, row in df.iterrows():
        text = str(row.get("skill_requirements", "")).lower()
        if not text:
            continue

        # Sum calibrated burst of all matching keywords
        total_burst = 0.0
        for kw in KEYWORDS:
            if kw.lower() in text:
                total_burst += calib.get(kw, 0.1)

        # Boost: up to max_boost for high-signal JDs
        boost = 1.0 + min(total_burst, 1.0) * (max_boost - 1.0)
        weights[i] = base_weight * boost

    # Normalize
    weights = weights / weights.sum()

    return weights


def compute_class_weights(label_encoder, df, feature_path="calibrated_features_v2.json"):
    """
    计算每个岗位类别的 tech 信号权重。

    对每个类别，取该类别 JDs 中最常出现的 tech 关键词的 calibrated_burst 均值。
    """
    calib = load_calibrated_features(feature_path)

    class_bursts = {c: [] for c in range(len(label_encoder.classes_))}

    for i, row in df.iterrows():
        label = row.get("label")
        if label is None:
            continue
        text = str(row.get("skill_requirements", "")).lower()

        class_burst = []
        for kw in KEYWORDS:
            if kw.lower() in text:
                class_burst.append(calib.get(kw, 0.1))

        if class_burst:
            class_bursts[label].append(np.mean(class_burst))

    # Average per class
    class_weights = np.ones(len(label_encoder.classes_))
    for c in range(len(class_weights)):
        if class_bursts[c]:
            class_weights[c] = 1.0 + np.mean(class_bursts[c])
        else:
            class_weights[c] = 1.0

    class_weights = class_weights / class_weights.mean()
    return class_weights


def compute_keyword_match_matrix(df, feature_path="calibrated_features_v2.json"):
    """
    构建 (N_samples × 24_keywords) 的匹配矩阵。
    可直接作为辅助特征输入模型。
    """
    calib = load_calibrated_features(feature_path)
    matrix = np.zeros((len(df), len(KEYWORDS)))

    for i, row in df.iterrows():
        text = str(row.get("skill_requirements", "")).lower()
        for j, kw in enumerate(KEYWORDS):
            if kw.lower() in text:
                matrix[i, j] = calib.get(kw, 0.1)

    return matrix, KEYWORDS
