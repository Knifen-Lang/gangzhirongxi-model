"""
Bayesian kNN 非参数分类器 — 冷门岗位推理增强

直接迁移自 人工智能挑战赛 v3&v4/utils_v3/bayesian_knn.py

推理时用训练集嵌入做非参数投票，不依赖 softmax 分类器参数。
对 ≤5 条的极端少样本类特别有效。

P(job | x) ∝ sum_{i in N_k(x)} exp(-d(x, x_i)^2 / sigma^2) * I(y_i == job)

新增功能（相比源项目）:
- 支持 FAISS 加速 kNN 检索
- 支持自适应 lambda 融合
"""

import numpy as np
import torch
from scipy.spatial.distance import cdist


class BayesianKNNClassifier:
    """
    Bayesian kNN 非参数分类器 (SCoRE, 2025)

    用训练集嵌入做非参数投票:
    P(job | x) ∝ sum_{i∈N_k(x)} exp(-d(x,x_i)²/σ²) × I(y_i==job)

    Args:
        train_embeddings: (N_train, hidden_size) 训练集 [CLS] 嵌入
        train_labels: (N_train,) 训练集标签 ID
        label_encoder: sklearn LabelEncoder
        k: 近邻数
        sigma: 'adaptive' | 'per_point' | float
    """

    def __init__(self, train_embeddings, train_labels, label_encoder,
                 k=20, sigma='adaptive', use_faiss=False):
        self.train_embeddings = train_embeddings.astype('float32')
        self.train_labels = np.array(train_labels)
        self.label_encoder = label_encoder
        self.k = int(k)
        self.use_faiss = use_faiss

        # 预计算每个训练点的第 k 近邻距离
        pairwise_dists = cdist(
            self.train_embeddings, self.train_embeddings
        )
        pairwise_dists.sort(axis=1)
        self.kth_dists = pairwise_dists[
            :, min(k, pairwise_dists.shape[1] - 1)
        ]

        if sigma == 'adaptive':
            self.sigma = np.median(self.kth_dists)
        elif sigma == 'per_point':
            self.sigma = self.kth_dists
        else:
            self.sigma = float(sigma)

        # FAISS 索引（可选）
        self.faiss_index = None
        if use_faiss:
            try:
                import faiss
                d = train_embeddings.shape[1]
                self.faiss_index = faiss.IndexFlatL2(d)
                self.faiss_index.add(train_embeddings)
                print(f'  FAISS 索引构建完成: {d} 维, '
                      f'{len(train_embeddings)} 条')
            except ImportError:
                print('  FAISS 未安装，使用 brute-force kNN')

    def predict_proba(self, query_embeddings):
        """
        预测概率分布

        Args:
            query_embeddings: (N_query, hidden_size)

        Returns:
            probs: (N_query, num_classes) 概率分布
        """
        query_embeddings = query_embeddings.astype('float32')

        if self.faiss_index is not None:
            distances, top_k_idx = self.faiss_index.search(
                query_embeddings, self.k
            )
            top_k_dists = distances
        else:
            distances = cdist(query_embeddings, self.train_embeddings)
            top_k_idx = np.argpartition(
                distances, self.k - 1, axis=1
            )[:, :self.k]
            top_k_dists = np.take_along_axis(
                distances, top_k_idx, axis=1
            )

        top_k_labels = self.train_labels[top_k_idx]

        # 高斯核权重
        if isinstance(self.sigma, np.ndarray):
            sigma2 = (self.sigma[top_k_idx] ** 2)
        else:
            sigma2 = self.sigma ** 2

        weights = np.exp(-top_k_dists ** 2 / sigma2)
        weights_sum = weights.sum(axis=1, keepdims=True) + 1e-8
        weights = weights / weights_sum

        n_classes = len(self.label_encoder.classes_)
        probs = np.zeros(
            (query_embeddings.shape[0], n_classes), dtype='float32'
        )

        for i in range(query_embeddings.shape[0]):
            for j in range(self.k):
                lid = top_k_labels[i, j]
                probs[i, lid] += weights[i, j]

        return probs

    def predict(self, query_embeddings):
        """预测类别"""
        probs = self.predict_proba(query_embeddings)
        pred_labels = np.argmax(probs, axis=1)
        return self.label_encoder.inverse_transform(pred_labels)


# ═══════════════════════════════════════════════════════════════
#  Helper Functions
# ═══════════════════════════════════════════════════════════════

def compute_train_embeddings(model, train_loader, device='cuda'):
    """
    计算训练集全部嵌入向量

    Args:
        model: 训练好的模型
        train_loader: 训练数据 loader
        device: 设备

    Returns:
        embeddings: (N, hidden_size)
        labels: (N,) label IDs
    """
    model.eval()
    all_embeddings = []
    all_labels = []

    dev = torch.device(device if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        for batch in train_loader:
            if batch is None:
                continue
            input_ids = batch['data'].to(dev)
            mask = batch['cls_mask'].to(dev)
            labels = batch['label']

            _, cls_emb, _ = model(input_ids, mask)
            all_embeddings.append(cls_emb.cpu().numpy())
            all_labels.extend(labels.cpu().numpy().tolist())

    return (
        np.concatenate(all_embeddings, axis=0),
        np.array(all_labels),
    )


def bayesian_knn_ensemble(softmax_probs, knn_probs, alpha=0.15):
    """
    Bayesian kNN + Softmax 概率融合

    Args:
        softmax_probs: (N, C) 模型 softmax 输出
        knn_probs: (N, C) kNN 概率输出
        alpha: kNN 权重（冷门岗位多用 kNN）

    Returns:
        fused_probs: (N, C) 融合后概率
    """
    return alpha * knn_probs + (1 - alpha) * softmax_probs


def dynamic_entropy_ensemble(probs_list, base_weights=None, entropy_scale=2.0):
    """
    双层动态权重集成

    Layer 1: base_weights 来自 CV 分数（固定）
    Layer 2: 逐样本熵权重（动态）

    Args:
        probs_list: list of (N, C) 概率矩阵
        base_weights: (N_models,) 基础权重
        entropy_scale: 熵缩放因子

    Returns:
        ensemble_probs: (N, C)
    """
    N_models = len(probs_list)
    N_samples = probs_list[0].shape[0]

    probs_list = [np.asarray(p, dtype='float64') for p in probs_list]

    if base_weights is None:
        base_weights = np.ones(N_models) / N_models
    base_weights = np.asarray(base_weights, dtype='float64')

    entropies = []
    for probs in probs_list:
        entropy = -np.sum(probs * np.log(probs + 1e-8), axis=-1)
        entropies.append(entropy)
    entropies = np.stack(entropies, axis=0)

    confidence = 1.0 / (entropies + 0.1)
    log_weights = np.log(base_weights.reshape(-1, 1) + 1e-8)
    log_weights = log_weights + entropy_scale * np.log(confidence + 1e-8)

    weights = np.exp(log_weights - log_weights.max(axis=0, keepdims=True))
    weights = weights / weights.sum(axis=0, keepdims=True)

    ensemble_probs = sum(
        weights[i, :, None] * probs_list[i] for i in range(N_models)
    )
    return ensemble_probs
