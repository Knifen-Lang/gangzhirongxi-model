"""
模型对比评测 — 岗位匹配任务

对比以下方法:
1. Cross-Encoder (JobClassifier / DeBERTa) — 最高精度, 慢
2. CoSENT Bi-Encoder (text2vec) — 快速, 语义匹配
3. Bayesian kNN — 非参数冷门推理
4. TF-IDF + Cosine — 传统baseline
5. Ensemble (Cross + CoSENT + kNN) — 融合

输出: 每个模型在各类别区间 (≤5, 6-10, 11-20, 21-50, >50) 的准确率

用法:
  python scripts/benchmark.py --data_dir ../zhilian_direct/zhilian_direct
"""

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
import torch

# 设置 HuggingFace 镜像（国内加速）
if not os.environ.get('HF_ENDPOINT'):
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.data.data_utils import (
    load_zhilian_data, get_class_counts, stratified_kfold_split,
    load_tokenizer, save_label_classes,
)
from src.models.classifier import JobClassifier
from src.models.cosent_matcher import CoSENTJobMatcher
from src.inference.bayesian_knn import (
    BayesianKNNClassifier, compute_train_embeddings, dynamic_entropy_ensemble,
)
from src.utils.evaluation import (
    compute_competition_score, compute_interval_scores, compute_fewshot_score,
)


class TFIDFBaseline:
    """TF-IDF + 余弦相似度 baseline"""

    def __init__(self):
        self.vectorizer = None
        self.job_vectors = None
        self.job_labels = None

    def fit(self, texts_a, texts_b, labels):
        """构建岗位文本的 TF-IDF 向量"""
        self.vectorizer = TfidfVectorizer(
            max_features=5000, ngram_range=(1, 2), analyzer='char_wb',
        )
        all_texts = texts_a + texts_b
        self.vectorizer.fit(all_texts)

        # 为每个岗位构建平均向量
        label_set = sorted(set(labels))
        self.job_labels = label_set
        self.job_vectors = []
        for l in label_set:
            mask = [i for i, lb in enumerate(labels) if lb == l]
            texts = [texts_b[i] for i in mask] + [texts_a[i] for i in mask[:2]]
            vecs = self.vectorizer.transform(texts)
            self.job_vectors.append(vecs.mean(axis=0))

    def predict(self, texts_a):
        """预测岗位"""
        vecs_a = self.vectorizer.transform(texts_a)
        job_vecs = np.asarray(np.vstack(self.job_vectors))
        scores = cosine_similarity(vecs_a, job_vecs)
        pred_indices = np.argmax(scores, axis=1)
        return [self.job_labels[i] for i in pred_indices], scores


class CoSENTBaseline:
    """CoSENT 双编码器匹配"""

    def __init__(self, model_name='shibing624/text2vec-base-chinese', device='cpu'):
        self.device = torch.device(device)
        self.matcher = CoSENTJobMatcher(
            model_name=model_name, encoder_type='MEAN', max_seq_length=256,
        )
        self.matcher.to(self.device)
        self.matcher.eval()
        self.job_embeddings = None
        self.job_labels = None

    def fit(self, texts_b, labels):
        """预计算岗位文本嵌入"""
        self.job_embeddings = self.matcher.encode(texts_b, show_progress=True)
        self.job_labels = np.array(labels)

    def predict(self, texts_a):
        """预测岗位"""
        query_emb = self.matcher.encode(texts_a, show_progress=True)
        query_emb = F.normalize(torch.from_numpy(query_emb), dim=-1)
        job_emb = F.normalize(torch.from_numpy(self.job_embeddings), dim=-1)
        scores = torch.matmul(query_emb, job_emb.t()).numpy()
        pred_indices = np.argmax(scores, axis=1)
        return [self.job_labels[i] for i in pred_indices], scores


class CrossEncoderBaseline:
    """Cross-Encoder (JobClassifier) 匹配"""

    def __init__(self, model_name='microsoft/deberta-v3-base',
                 num_classes=87, device='cpu'):
        self.device = torch.device(device)
        self.model = JobClassifier(model_name, num_classes, dropout=0.15)
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = load_tokenizer(model_name)
        self.label_encoder = None

    def load_weights(self, model_path):
        state_dict = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(state_dict, strict=False)

    def predict(self, texts_a, texts_b, batch_size=32):
        """逐对预测（因为 cross-encoder 需要拼接）"""
        from src.data.data_utils import encode_job_pair

        all_preds = []
        all_probs = []

        for i in tqdm(range(0, len(texts_a), batch_size), desc='Cross-Encoder'):
            batch_a = texts_a[i:i+batch_size]
            batch_b = texts_b[i:i+batch_size]

            input_ids_list = []
            attn_list = []
            for ta, tb in zip(batch_a, batch_b):
                ids, attn = encode_job_pair(
                    self.tokenizer, ta, tb, 256, encoding_format='A',
                )
                input_ids_list.append(ids)
                attn_list.append(attn)

            input_ids = torch.from_numpy(
                np.stack(input_ids_list)
            ).long().to(self.device)
            attention_mask = torch.from_numpy(
                np.stack(attn_list)
            ).long().to(self.device)

            with torch.no_grad():
                logits, _, _ = self.model(input_ids, attention_mask)
                probs = F.softmax(logits, dim=-1)

            pred_indices = torch.argmax(logits, dim=1)
            if self.label_encoder:
                preds = self.label_encoder.inverse_transform(
                    pred_indices.cpu().numpy()
                )
            else:
                preds = pred_indices.cpu().numpy()

            all_preds.extend(preds)
            all_probs.append(probs.cpu().numpy())

        return all_preds, np.concatenate(all_probs, axis=0)


def run_benchmark(data_dir, output_dir='./outputs',
                  cross_encoder_model=None, small_test=False):
    """
    运行完整 benchmark

    Args:
        data_dir: 数据目录
        cross_encoder_model: 训练好的 cross-encoder 路径
        small_test: True=只用少量数据测试
    """
    print('=' * 60)
    print('  岗位匹配模型对比评测')
    print('=' * 60)

    # ════ 加载数据 ════
    print('\n>>> 加载数据...')
    df = load_zhilian_data(data_dir)

    if small_test:
        top_labels = df['label'].value_counts().head(30).index.tolist()
        df = df[df['label'].isin(top_labels)].copy()
        print(f'  缩减到 {len(top_labels)} 类, {len(df)} 条')

    # 划分 train/val
    folds = stratified_kfold_split(df, n_folds=1, random_state=42, val_ratio=0.15)
    train_df, val_df = folds[0]
    print(f'  训练: {len(train_df)}, 验证: {len(val_df)}')

    class_counts = get_class_counts(df)
    counts_max = max(class_counts.values())
    counts_min = min(class_counts.values())

    # 准备数据
    texts_a_train = train_df['text_a'].astype(str).tolist()
    texts_b_train = train_df['text_b'].astype(str).tolist()
    labels_train = train_df['label'].astype(str).tolist()

    texts_a_val = val_df['text_a'].astype(str).tolist()
    texts_b_val = val_df['text_b'].astype(str).tolist()
    labels_val = val_df['label'].astype(str).tolist()

    le = LabelEncoder()
    le.fit(df['label'].unique())
    num_classes = len(le.classes_)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'  设备: {device}')

    results = {}

    # ════ Model 1: TF-IDF Baseline ════
    print('\n>>> [1/5] TF-IDF Baseline...')
    t0 = time.time()
    tfidf = TFIDFBaseline()
    tfidf.fit(texts_a_train, texts_b_train, labels_train)
    preds_tfidf, scores_tfidf = tfidf.predict(texts_a_val)
    tfidf_time = time.time() - t0

    results['TF-IDF'] = _evaluate(
        labels_val, preds_tfidf, class_counts, counts_max, counts_min,
        infer_time=tfidf_time, num_samples=len(texts_a_val),
    )

    # ════ Model 2: CoSENT Bi-Encoder ════
    print('\n>>> [2/5] CoSENT Bi-Encoder (bert-base-chinese)...')
    t0 = time.time()
    cosent = CoSENTBaseline(
        model_name='bert-base-chinese', device=device
    )

    # 用所有岗位构建嵌入库
    unique_jobs = train_df.groupby('label')['text_b'].first()
    job_labels_list = unique_jobs.index.tolist()
    job_texts_list = unique_jobs.values.tolist()

    cosent.fit(job_texts_list, job_labels_list)

    # 预测: 每个 val 样本匹配到最相似岗位
    val_unique_jobs = val_df['label'].unique()
    preds_cosent = []
    for i in tqdm(range(0, len(texts_a_val), 64), desc='CoSENT predict'):
        batch_a = texts_a_val[i:i+64]
        emb_a = cosent.matcher.encode(batch_a, show_progress=False)
        emb_a = F.normalize(torch.from_numpy(emb_a), dim=-1)
        job_emb = F.normalize(torch.from_numpy(cosent.job_embeddings), dim=-1)
        scores = torch.matmul(emb_a, job_emb.t()).numpy()
        pred_idx = np.argmax(scores, axis=1)
        preds_cosent.extend([cosent.job_labels[i] for i in pred_idx])

    cosent_time = time.time() - t0

    results['CoSENT (bi-encoder)'] = _evaluate(
        labels_val, preds_cosent, class_counts, counts_max, counts_min,
        infer_time=cosent_time, num_samples=len(texts_a_val),
    )

    # ════ Model 3: Bayesian kNN ════
    print('\n>>> [3/5] Bayesian kNN...')
    t0 = time.time()

    # 用 CoSENT 嵌入做 kNN (因为 cross-encoder 可能需要训练)
    cosent.matcher.eval()
    train_emb = cosent.matcher.encode(texts_a_train, show_progress=True)
    train_lbls = le.transform(labels_train)

    knn = BayesianKNNClassifier(
        train_emb, train_lbls, le, k=20, sigma='adaptive',
    )
    val_emb = cosent.matcher.encode(texts_a_val, show_progress=True)
    knn_preds = knn.predict(val_emb)
    knn_time = time.time() - t0

    results['Bayesian kNN'] = _evaluate(
        labels_val, knn_preds, class_counts, counts_max, counts_min,
        infer_time=knn_time, num_samples=len(texts_a_val),
    )

    # ════ Model 4: Cross-Encoder (如果提供了权重) ════
    if cross_encoder_model and os.path.exists(cross_encoder_model):
        print('\n>>> [4/5] Cross-Encoder (DeBERTa-v3)...')
        t0 = time.time()
        ce = CrossEncoderBaseline(
            model_name='microsoft/deberta-v3-base',
            num_classes=num_classes, device=device,
        )
        ce.label_encoder = le
        ce.load_weights(cross_encoder_model)

        # 对验证集预测
        preds_ce, probs_ce = ce.predict(texts_a_val, texts_b_val, batch_size=8)
        ce_time = time.time() - t0

        results['Cross-Encoder (DeBERTa)'] = _evaluate(
            labels_val, preds_ce, class_counts, counts_max, counts_min,
            infer_time=ce_time, num_samples=len(texts_a_val),
        )
    else:
        print(f'\n>>> [4/5] Cross-Encoder — 跳过（需先训练, 模型路径不存在）')

    # ════ Model 5: Ensemble (CoSENT + kNN) ════
    print('\n>>> [5/5] Ensemble (CoSENT + kNN)...')
    t0 = time.time()

    # 融合 CoSENT 和 kNN 结果
    # 获取 CoSENT 概率
    val_emb_norm = F.normalize(torch.from_numpy(val_emb), dim=-1)
    job_emb_norm = F.normalize(torch.from_numpy(cosent.job_embeddings), dim=-1)
    cosent_scores = torch.matmul(val_emb_norm, job_emb_norm.t()).numpy()
    cosent_probs = np.exp(cosent_scores) / np.exp(cosent_scores).sum(axis=1, keepdims=True)

    knn_probs = knn.predict_proba(val_emb)

    # 动态权重融合
    ens_probs = dynamic_entropy_ensemble(
        [cosent_probs, knn_probs], base_weights=np.array([0.6, 0.4]),
    )
    ens_preds = le.inverse_transform(np.argmax(ens_probs, axis=1))
    ens_time = time.time() - t0

    results['Ensemble (CoSENT+kNN)'] = _evaluate(
        labels_val, ens_preds, class_counts, counts_max, counts_min,
        infer_time=ens_time, num_samples=len(texts_a_val),
    )

    # ════ 汇总 ════
    _print_summary(results)

    return results


def _evaluate(y_true, y_pred, class_counts, counts_max, counts_min,
              infer_time=0, num_samples=1):
    """评估并返回指标"""
    y_true = np.asarray(y_true, dtype=str)
    y_pred = np.asarray(y_pred, dtype=str)

    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    score, _ = compute_competition_score(
        y_true, y_pred, class_counts, counts_max, counts_min,
    )
    intervals = compute_interval_scores(y_true, y_pred, class_counts)
    fewshot = compute_fewshot_score(y_true, y_pred, class_counts, max_count=5)

    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'competition_score': score,
        'fewshot_score': fewshot,
        'intervals': intervals,
        'infer_time': infer_time,
        'num_samples': num_samples,
        'qps': num_samples / max(infer_time, 0.001),
    }


def _print_summary(results):
    """打印对比汇总"""
    print('\n' + '=' * 70)
    print('  模型对比汇总')
    print('=' * 70)

    # 表头
    print(f'{"模型":<30} {"准确率":>8} {"F1-macro":>8} {"比赛分数":>8} '
          f'{"FewShot":>8} {"QPS":>8}')
    print('-' * 70)

    for model_name, metrics in results.items():
        print(f'{model_name:<30} '
              f'{metrics["accuracy"]:>8.4f} '
              f'{metrics["f1_macro"]:>8.4f} '
              f'{metrics["competition_score"]:>8.4f} '
              f'{metrics["fewshot_score"]:>8.4f} '
              f'{metrics["qps"]:>8.0f}')

    # 区间分数对比
    print('\n--- 按样本数区间分数 ---')
    intervals = ['<=5', '6-10', '11-20', '21-50', '51-100', '>100']
    header = f'{"模型":<30} '
    for name in intervals:
        header += f'{name:>8} '
    print(header)
    print('-' * (30 + 9 * len(intervals)))

    for model_name, metrics in results.items():
        row = f'{model_name:<30} '
        for name in intervals:
            v = metrics['intervals'].get(name, 0.0)
            row += f'{v:>8.4f} '
        print(row)

    # 推理速度对比
    print('\n--- 推理速度对比 ---')
    for model_name, metrics in sorted(
        results.items(), key=lambda x: x[1]['infer_time']
    ):
        print(f'  {model_name:<30}: '
              f'{metrics["infer_time"]:.2f}s / '
              f'{metrics.get("num_samples", "N/A")} 样本 '
              f'({metrics["qps"]:.0f} QPS)')


def main():
    parser = argparse.ArgumentParser(description='岗位匹配模型对比评测')
    parser.add_argument('--data_dir', type=str,
                        default='../zhilian_direct/zhilian_direct')
    parser.add_argument('--output_dir', type=str, default='../outputs')
    parser.add_argument('--cross_encoder_model', type=str, default=None,
                        help='训练好的 cross-encoder 模型路径 (.pt)')
    parser.add_argument('--small_test', action='store_true', default=True,
                        help='用小规模数据快速测试')

    args = parser.parse_args()
    run_benchmark(
        args.data_dir, args.output_dir,
        args.cross_encoder_model, args.small_test,
    )


if __name__ == '__main__':
    main()
