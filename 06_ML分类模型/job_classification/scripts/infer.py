"""
岗位分类系统 — 推理脚本

支持:
- 单模型推理 (MC Dropout TTA)
- 多模型集成推理
- Bayesian kNN 增强推理
- 温度缩放校准

用法:
  # 单模型推理
  python infer.py --model_path ./outputs/fold_0/best_model.pt \
    --model_name microsoft/deberta-v3-base \
    --input_text "Python,Django,MySQL" --job_title "Python开发工程师"

  # 批量推理
  python infer.py --model_path ./outputs/fold_0/best_model.pt \
    --input_csv ./test_jobs.csv --output_file ./predictions.csv
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.models.classifier import JobClassifier
from src.data.data_utils import (
    load_tokenizer, encode_job_pair, load_zhilian_data,
)
from src.inference.bayesian_knn import (
    BayesianKNNClassifier, compute_train_embeddings,
    bayesian_knn_ensemble, dynamic_entropy_ensemble,
)


def load_model(model_path, model_name, num_classes, prototype_matrix=None,
               device='cuda', proto_init_scale=0.05):
    """加载训练好的模型"""
    model = JobClassifier(
        model_name, num_classes, dropout=0.15,
        prototype_matrix=prototype_matrix,
        proto_init_scale=proto_init_scale,
    ).to(device)

    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def predict_single_text(model, tokenizer, text_a, text_b, label_encoder,
                        device='cuda', mc_samples=5, tta_formats=None,
                        temperature=1.5):
    """
    单条文本预测 — 带 MC Dropout TTA

    Args:
        model: 模型
        tokenizer: tokenizer
        text_a: 候选人文本 (skills/resume)
        text_b: 岗位文本 (title+desc)
        label_encoder: sklearn LabelEncoder
        mc_samples: MC Dropout 前向次数
        tta_formats: TTA 编码格式列表
        temperature: softmax 温度

    Returns:
        probs: (num_classes,) 概率分布
    """
    if tta_formats is None:
        tta_formats = ['A']

    all_probs = []
    model.train()  # Enable dropout for MC

    for fmt in tta_formats:
        input_ids, attention_mask = encode_job_pair(
            tokenizer, text_a, text_b, max_length=256,
            encoding_format=fmt,
        )

        input_ids = torch.from_numpy(input_ids).unsqueeze(0).long().to(device)
        attention_mask = torch.from_numpy(attention_mask).unsqueeze(0).long().to(device)

        fmt_probs = []
        with torch.no_grad():
            for _ in range(mc_samples):
                logits, _, _ = model(input_ids, attention_mask)
                probs = F.softmax(logits / temperature, dim=-1)
                fmt_probs.append(probs.cpu().numpy())

        all_probs.append(np.mean(fmt_probs, axis=0))

    probs = np.mean(all_probs, axis=0)
    return probs[0]  # squeeze batch


def predict_batch(model, tokenizer, texts_a, texts_b, label_encoder,
                  device='cuda', batch_size=32, mc_samples=3,
                  tta_formats=None, temperature=1.5):
    """
    批量预测

    Args:
        texts_a: list of 候选人文本
        texts_b: list of 岗位文本

    Returns:
        probs: (N, num_classes)
        predictions: list of 预测岗位名
    """
    from torch.utils.data import Dataset as TDataset, DataLoader as TDataLoader

    if tta_formats is None:
        tta_formats = ['A']

    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.to(dev)
    model.eval()

    class PredDataset(TDataset):
        def __init__(self, texts_a, texts_b, tokenizer, max_length, fmt):
            self.samples = []
            for ta, tb in zip(texts_a, texts_b):
                ids, attn = encode_job_pair(
                    tokenizer, ta, tb, max_length, encoding_format=fmt,
                )
                self.samples.append((ids, attn))
            self.max_length = max_length

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            ids, attn = self.samples[idx]
            return {
                'input_ids': torch.from_numpy(ids).long(),
                'attention_mask': torch.from_numpy(attn).long(),
            }

    def collate(batch):
        return {
            'input_ids': torch.stack([b['input_ids'] for b in batch]),
            'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        }

    all_probs_list = []

    for fmt in tta_formats:
        dataset = PredDataset(texts_a, texts_b, tokenizer, 256, fmt)
        dataloader = TDataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, collate_fn=collate,
        )

        fmt_probs = []
        # MC Dropout
        model.train()
        for _ in range(mc_samples):
            probs_list = []
            with torch.no_grad():
                for batch in tqdm(dataloader, desc=f'TTA {fmt}', leave=False):
                    input_ids = batch['input_ids'].to(dev)
                    attention_mask = batch['attention_mask'].to(dev)
                    logits, _, _ = model(input_ids, attention_mask)
                    probs = F.softmax(logits / temperature, dim=-1)
                    probs_list.append(probs.cpu().numpy())
            fmt_probs.append(np.concatenate(probs_list, axis=0))
        model.eval()

        all_probs_list.append(np.mean(fmt_probs, axis=0))

    probs = np.mean(all_probs_list, axis=0)
    pred_indices = np.argmax(probs, axis=1)
    predictions = label_encoder.inverse_transform(pred_indices)

    return probs, predictions


def infer_with_knn(model, knn_classifier, tokenizer, text_a, text_b,
                   label_encoder, device='cuda', alpha=0.15):
    """
    推理 + Bayesian kNN 增强
    """
    # Softmax 预测
    softmax_probs = predict_single_text(
        model, tokenizer, text_a, text_b, label_encoder, device=device,
    )

    # kNN 预测需要先编码输入
    input_ids, attention_mask = encode_job_pair(
        tokenizer, text_a, text_b, max_length=256, encoding_format='A',
    )
    input_ids = torch.from_numpy(input_ids).unsqueeze(0).long().to(device)
    attention_mask = torch.from_numpy(attention_mask).unsqueeze(0).long().to(device)

    model.eval()
    with torch.no_grad():
        _, cls_emb, _ = model(input_ids, attention_mask)
    query_emb = cls_emb.cpu().numpy()

    knn_probs = knn_classifier.predict_proba(query_emb)

    # 融合
    fused_probs = bayesian_knn_ensemble(
        softmax_probs.reshape(1, -1), knn_probs, alpha=alpha,
    )
    return fused_probs[0]


def main():
    parser = argparse.ArgumentParser(description='岗位分类推理')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--model_name', type=str,
                        default='microsoft/deberta-v3-base')
    parser.add_argument('--labels_path', type=str, default=None,
                        help='label_classes.txt 路径')
    parser.add_argument('--prototype_path', type=str, default=None)
    parser.add_argument('--proto_init_scale', type=float, default=0.05)

    # 输入
    parser.add_argument('--input_text', type=str, default=None,
                        help='单条候选人文本 (逗号分隔的技能)')
    parser.add_argument('--job_title', type=str, default=None,
                        help='目标岗位名')
    parser.add_argument('--input_csv', type=str, default=None,
                        help='批量输入 CSV')
    parser.add_argument('--output_file', type=str, default='./predictions.csv')

    # 推理参数
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--mc_samples', type=int, default=5)
    parser.add_argument('--tta_formats', type=str, nargs='+',
                        default=['A', 'B', 'C'])
    parser.add_argument('--temperature', type=float, default=1.5)
    parser.add_argument('--use_knn', action='store_true', default=False)
    parser.add_argument('--knn_k', type=int, default=20)
    parser.add_argument('--knn_alpha', type=float, default=0.15)
    parser.add_argument('--train_data_dir', type=str, default=None,
                        help='kNN 需要训练数据路径')

    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--top_k', type=int, default=5,
                        help='输出 Top-K 预测')

    args = parser.parse_args()

    device = torch.device(
        args.device if args.device
        else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    print(f'设备: {device}')

    # 加载标签
    if args.labels_path:
        with open(args.labels_path, 'r', encoding='utf-8') as f:
            labels = [line.strip() for line in f if line.strip()]
    else:
        # 尝试从 model_path 推断
        parent = os.path.dirname(os.path.dirname(args.model_path))
        lp = os.path.join(parent, 'label_classes.txt')
        if os.path.exists(lp):
            with open(lp, 'r', encoding='utf-8') as f:
                labels = [line.strip() for line in f if line.strip()]
        else:
            raise ValueError('需要 --labels_path')

    num_classes = len(labels)
    print(f'岗位类别数: {num_classes}')

    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)

    # 加载原型矩阵
    prototype_matrix = None
    if args.prototype_path and os.path.exists(args.prototype_path):
        prototype_matrix = np.load(args.prototype_path)

    # 加载模型
    tokenizer = load_tokenizer(args.model_name)
    model = load_model(
        args.model_path, args.model_name, num_classes,
        prototype_matrix=prototype_matrix, device=device,
        proto_init_scale=args.proto_init_scale,
    )
    print(f'模型加载完成')

    # kNN 初始化
    knn_classifier = None
    if args.use_knn and args.train_data_dir:
        print('初始化 Bayesian kNN...')
        # 需要训练数据来构建 kNN 索引
        train_df = load_zhilian_data(args.train_data_dir)
        from src.data.data_utils import JobDataset
        from torch.utils.data import DataLoader

        train_dataset = JobDataset(
            train_df, tokenizer, label_encoder, max_length=args.max_length,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=dynamic_collate_fn, num_workers=0,
        )

        train_emb, train_labels = compute_train_embeddings(
            model, train_loader, device=device,
        )
        knn_classifier = BayesianKNNClassifier(
            train_emb, train_labels, label_encoder,
            k=args.knn_k, sigma='adaptive',
        )
        print(f'kNN 初始化完成 (k={args.knn_k})')

    # ═══════════════════════════════════════════
    # 推理
    # ═══════════════════════════════════════════

    if args.input_text:
        # 单条推理
        text_a = args.input_text
        text_b = args.job_title if args.job_title else '岗位匹配'

        if knn_classifier:
            probs = infer_with_knn(
                model, knn_classifier, tokenizer, text_a, text_b,
                label_encoder, device=device, alpha=args.knn_alpha,
            )
        else:
            probs = predict_single_text(
                model, tokenizer, text_a, text_b, label_encoder,
                device=device, mc_samples=args.mc_samples,
                tta_formats=args.tta_formats,
                temperature=args.temperature,
            )

        # Top-K
        top_k_indices = np.argsort(probs)[::-1][:args.top_k]
        print(f'\n候选人: {text_a[:100]}...')
        print(f'目标岗位: {text_b}')
        print(f'Top-{args.top_k} 预测:')
        for rank, idx in enumerate(top_k_indices):
            print(f'  {rank+1}. {label_encoder.classes_[idx]} '
                  f'(置信度: {probs[idx]:.4f})')

    elif args.input_csv:
        # 批量推理
        df = pd.read_csv(args.input_csv, encoding='utf-8-sig')
        df.columns = [str(c).strip() for c in df.columns]

        text_a_col = next(
            (c for c in df.columns if c.lower() in
             ['text_a', 'skill_requirements', 'resume', 'skills']),
            df.columns[0]
        )
        text_b_col = next(
            (c for c in df.columns if c.lower() in
             ['text_b', 'job_name', 'job_title', 'job_desc']),
            None
        )

        texts_a = df[text_a_col].astype(str).tolist()
        texts_b = (
            df[text_b_col].astype(str).tolist() if text_b_col
            else ['岗位匹配'] * len(texts_a)
        )

        probs, predictions = predict_batch(
            model, tokenizer, texts_a, texts_b, label_encoder,
            device=device, batch_size=args.batch_size,
            mc_samples=args.mc_samples,
            tta_formats=args.tta_formats,
            temperature=args.temperature,
        )

        # 保存结果
        result_df = df.copy()
        result_df['predicted_job'] = predictions
        result_df['confidence'] = probs.max(axis=1)

        # Top-3 预测
        top3_indices = np.argsort(probs, axis=1)[:, -3:][:, ::-1]
        for rank in range(3):
            result_df[f'top_{rank+1}'] = [
                label_encoder.classes_[idx] for idx in top3_indices[:, rank]
            ]
            result_df[f'top_{rank+1}_conf'] = [
                probs[i, idx] for i, idx in enumerate(top3_indices[:, rank])
            ]

        result_df.to_csv(args.output_file, index=False, encoding='utf-8-sig')
        print(f'结果保存至: {args.output_file}')
        print(f'预测分布:')
        print(result_df['predicted_job'].value_counts().head(10))
    else:
        print('请提供 --input_text 或 --input_csv')


if __name__ == '__main__':
    main()
