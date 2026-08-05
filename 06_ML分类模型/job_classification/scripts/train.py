"""
岗位分类系统 — 两阶段训练脚本

迁移自 人工智能挑战赛 v4训练/train_v4_plus.py

核心设计:
  1. 原型矩阵初始化分类器权重 (proto_init_scale=0.05)
  2. Stage 1: FocalLoss + SupCon + SpanCL + 辅助损失 + IB/Uniform混合采样 + R-Drop
  3. Stage 2: cRT CWBS校准 (冻结encoder + 自适应tau)
  4. Early Stopping (patience=5)
  5. 5-Fold Cross Validation

用法:
  # 快速实验 (1 fold, deberta-v3-base)
  python train.py --model_name microsoft/deberta-v3-base \
    --data_dir ../zhilian_direct/zhilian_direct \
    --n_folds 1 --batch_size 16 --epochs_stage1 8 --epochs_stage2 3

  # 完整训练 (5 fold, deberta-v3-large)
  python train.py --model_name microsoft/deberta-v3-large \
    --data_dir ../zhilian_direct/zhilian_direct \
    --n_folds 5 --batch_size 8 --epochs_stage1 12 --epochs_stage2 5

  # 跳过 Stage 1 (从已有模型出发)
  python train.py --skip_s1 --resume_s1_path ./outputs/fold_0/best_model_s1.pt
"""

import argparse
import math
import os
import sys
import random
import logging
import json
from datetime import datetime

# 设置 HuggingFace 镜像（国内加速）
if not os.environ.get('HF_ENDPOINT'):
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

# 添加 src 路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.models.classifier import JobClassifier
from src.losses import (
    CWBSLoss, FocalLoss, SupConLoss, SpanContrastiveLoss,
    TripletMarginLoss, DescAlignLoss, RDropLoss,
    asymmetric_label_smoothing,
)
from src.data.data_utils import (
    load_zhilian_data, load_synthetic_data, get_class_counts,
    CWBSWeightedSampler, JobDataset, dynamic_collate_fn, load_tokenizer,
    stratified_kfold_split, adaptive_tau_per_class, save_label_classes,
)
from src.utils.evaluation import validate_with_intervals


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    root_logger = logging.getLogger()
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
    root_logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(os.path.join(save_dir, 'train.log'),
                             mode='w', encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    root_logger.addHandler(fh)
    root_logger.addHandler(ch)


def resolve_device(device_arg):
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def run_training(args):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.output_dir, f'job_cls_{timestamp}')
    setup_logging(save_dir)
    set_seed(args.random_seed)
    device = resolve_device(args.device)

    logging.info(f'设备: {device}')
    logging.info(f'参数: {vars(args)}')

    # ═══════════════════════════════════════════════════
    # 1. 加载数据
    # ═══════════════════════════════════════════════════
    logging.info('=' * 60)
    logging.info('加载数据...')
    logging.info('=' * 60)

    raw_train_df = load_zhilian_data(args.data_dir)
    logging.info(f'原始训练样本: {len(raw_train_df)}')

    synthetic_df = load_synthetic_data(args.synthetic_data_path)
    logging.info(f'合成样本: {len(synthetic_df)}')

    combined_df = pd.concat([raw_train_df, synthetic_df], ignore_index=True)
    logging.info(f'合并后总样本: {len(combined_df)}')

    raw_class_counts = get_class_counts(raw_train_df)
    all_class_counts = {
        k: v for k, v in sorted(raw_class_counts.items(), key=lambda x: -x[1])
    }
    counts_max = max(all_class_counts.values())
    counts_min = min(all_class_counts.values())
    logging.info(
        f'类别数: {len(all_class_counts)}, '
        f'最多: {counts_max}, 最少: {counts_min}'
    )

    label_encoder = LabelEncoder()
    label_encoder.fit(combined_df['label'].unique())
    num_classes = len(label_encoder.classes_)
    save_label_classes(label_encoder, save_dir)

    class_counts_list = [
        all_class_counts.get(c, 0) for c in label_encoder.classes_
    ]

    tokenizer = load_tokenizer(args.model_name)
    encoding_format = getattr(args, 'encoding_format', 'A')

    # ═══════════════════════════════════════════════════
    # 2. 加载/构建原型矩阵
    # ═══════════════════════════════════════════════════
    prototype_matrix = None
    if args.prototype_path and os.path.exists(args.prototype_path):
        prototype_matrix = np.load(args.prototype_path)
        proto_has_nan = bool(np.any(np.isnan(prototype_matrix)))
        if proto_has_nan:
            logging.error('原型矩阵包含 NaN! 请重新生成。')
            raise ValueError('原型矩阵损坏')

        # 重新排序以匹配 label_encoder
        prototype_names_path = args.prototype_path.replace(
            '.npy', '_names.json'
        )
        if os.path.exists(prototype_names_path):
            with open(prototype_names_path, 'r', encoding='utf-8') as f:
                proto_names = json.load(f)
            encoder_names = list(label_encoder.classes_)
            if proto_names != encoder_names:
                logging.info('重新排序原型矩阵...')
                name_to_idx = {name: i for i, name in enumerate(proto_names)}
                reordered = np.zeros_like(prototype_matrix)
                missing_count = 0
                for j, name in enumerate(encoder_names):
                    if name in name_to_idx:
                        reordered[j] = prototype_matrix[name_to_idx[name]]
                    else:
                        reordered[j] = (
                            np.random.randn(prototype_matrix.shape[1]).astype(
                                'float32') * 0.02
                        )
                        missing_count += 1
                if missing_count > 0:
                    logging.warning(
                        f'  {missing_count} 类缺失原型，使用随机初始化'
                    )
                prototype_matrix = reordered

        logging.info(f'原型矩阵: {prototype_matrix.shape}, NaN={proto_has_nan}')
    else:
        logging.info('未找到原型矩阵，使用随机初始化分类头')
        if args.prototype_path:
            logging.warning(
                f'原型矩阵不存在: {args.prototype_path}。'
                f'运行: python precompute_prototypes.py'
            )

    # ═══════════════════════════════════════════════════
    # 3. 划分 Folds
    # ═══════════════════════════════════════════════════
    raw_folds = stratified_kfold_split(
        raw_train_df, n_folds=args.n_folds, random_state=args.random_seed
    )
    folds = []
    for raw_train_part, val_df in raw_folds:
        if len(synthetic_df) > 0:
            train_df = pd.concat(
                [raw_train_part, synthetic_df], ignore_index=True
            )
        else:
            train_df = raw_train_part
        folds.append((train_df, val_df))
    logging.info(
        f'创建 {len(folds)} 个分层 fold (验证集仅来自原始数据)'
    )

    # ═══════════════════════════════════════════════════
    # 4. 训练每个 Fold
    # ═══════════════════════════════════════════════════
    all_fold_results = []

    for fold_idx in range(args.n_folds):
        logging.info(f'\n{"="*60}')
        logging.info(f'  Fold {fold_idx + 1}/{args.n_folds}')
        logging.info(f'{"="*60}')

        train_df, val_df = folds[fold_idx]
        fold_save_dir = os.path.join(save_dir, f'fold_{fold_idx}')
        os.makedirs(fold_save_dir, exist_ok=True)

        # 构建 DataLoader
        train_dataset = JobDataset(
            train_df, tokenizer, label_encoder, args.max_length,
            encoding_format=encoding_format,
        )
        val_dataset = JobDataset(
            val_df, tokenizer, label_encoder, args.max_length,
            encoding_format=encoding_format,
        )

        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=dynamic_collate_fn, num_workers=args.num_workers,
        )

        # 模型初始化
        model = JobClassifier(
            args.model_name, num_classes, dropout=args.dropout,
            prototype_matrix=prototype_matrix,
            proto_init_scale=args.proto_init_scale,
        ).to(device)
        logging.info(
            f'分类器初始化 (prototype_scale={args.proto_init_scale})'
        )

        scaler = torch.cuda.amp.GradScaler() if (
            device.type == 'cuda' and args.use_amp
        ) else None

        # ═══════════════════════════════════════════════
        # Stage 1: 表示学习
        # ═══════════════════════════════════════════════
        if not args.skip_s1 and args.epochs_stage1 > 0:
            _run_stage1(
                args, model, train_dataset, train_df, val_loader,
                label_encoder, all_class_counts, counts_max, counts_min,
                device, scaler, fold_save_dir, class_counts_list,
                prototype_matrix,
            )

        # ═══════════════════════════════════════════════
        # Stage 2: CWBS 校准
        # ═══════════════════════════════════════════════
        if not args.no_s2:
            _run_stage2(
                args, model, train_dataset, train_df, val_loader,
                label_encoder, all_class_counts, counts_max, counts_min,
                device, scaler, fold_save_dir, class_counts_list,
            )

        all_fold_results.append({'fold': fold_idx})
        logging.info(f'Fold {fold_idx} 完成')

    # ═══════════════════════════════════════════════════
    # 5. CV 汇总
    # ═══════════════════════════════════════════════════
    logging.info(f'\n{"="*60}')
    logging.info(f'  训练完成. 模型保存至 {save_dir}')
    logging.info(f'{"="*60}')


def _run_stage1(args, model, train_dataset, train_df, val_loader,
                label_encoder, all_class_counts, counts_max, counts_min,
                device, scaler, fold_save_dir, class_counts_list,
                prototype_matrix):
    """Stage 1: 表示学习"""

    logging.info('========== STAGE 1: 表示学习 ==========')
    logging.info(
        f'  损失: Focal(γ={args.focal_gamma}) + '
        f'LS({args.label_smoothing}) + R-Drop({args.rdrop_weight})'
        f' + SupCon({args.supcon_weight}) + SpanCL({args.spancl_weight})'
    )

    # IB+Uniform 混合采样
    ib_class_counts = train_df['label'].value_counts().to_dict()
    ib_weights = np.array([
        1.0 / max(ib_class_counts.get(train_df.iloc[i]['label'], 1), 1)
        for i in range(len(train_df))
    ])
    ib_weights = ib_weights / ib_weights.sum()
    uniform_weights = np.ones(len(train_df)) / len(train_df)
    mixed_weights = (
        args.ib_ratio * ib_weights +
        (1 - args.ib_ratio) * uniform_weights
    )
    mixed_weights = mixed_weights / mixed_weights.sum()

    mixed_sampler = WeightedRandomSampler(
        mixed_weights, num_samples=len(train_df), replacement=True,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=mixed_sampler,
        collate_fn=dynamic_collate_fn, num_workers=args.num_workers,
    )

    total_steps_s1 = len(train_loader) * args.epochs_stage1
    opt_s1 = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': args.lr_stage1},
        {'params': model.classifier.parameters(),
         'lr': args.lr_classifier_s1},
    ], weight_decay=args.weight_decay)
    warmup_steps = int(args.warmup_ratio * total_steps_s1)

    def lr_lambda_s1(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(
            max(1, total_steps_s1 - warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    sched_s1 = torch.optim.lr_scheduler.LambdaLR(opt_s1, lr_lambda_s1)

    # 损失函数
    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    focal_loss_fn = FocalLoss(
        class_counts_list, counts_max, counts_min,
        gamma=args.focal_gamma,
    )
    rdrop_loss = RDropLoss(alpha=args.rdrop_weight) \
        if args.rdrop_weight > 0 else None
    supcon_loss = SupConLoss(temperature=0.07) \
        if args.supcon_weight > 0 else None
    spancl_loss = SpanContrastiveLoss() \
        if args.spancl_weight > 0 else None
    triplet_loss = TripletMarginLoss(margin=0.3) \
        if args.triplet_weight > 0 else None
    desc_align_loss = DescAlignLoss(prototype_matrix) \
        if (args.desc_align_weight > 0 and prototype_matrix is not None) \
        else None

    best_score_s1 = 0.0
    patience_s1_counter = 0

    for epoch in range(args.epochs_stage1):
        model.train()
        epoch_loss = 0.0
        train_steps = 0

        pbar = tqdm(train_loader,
                     desc=f'S1 Epoch {epoch+1}/{args.epochs_stage1}')
        for batch in pbar:
            if batch is None:
                continue
            input_ids = batch['data'].to(device)
            mask = batch['cls_mask'].to(device)
            labels = batch['label'].to(device)

            opt_s1.zero_grad()

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    loss = _compute_stage1_loss(
                        model, input_ids, mask, labels,
                        ce_loss_fn, focal_loss_fn, rdrop_loss,
                        supcon_loss, spancl_loss, triplet_loss,
                        desc_align_loss, args, device,
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(opt_s1)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=args.grad_clip
                )
                scaler.step(opt_s1)
                scaler.update()
            else:
                loss = _compute_stage1_loss(
                    model, input_ids, mask, labels,
                    ce_loss_fn, focal_loss_fn, rdrop_loss,
                    supcon_loss, spancl_loss, triplet_loss,
                    desc_align_loss, args, device,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=args.grad_clip
                )
                opt_s1.step()

            sched_s1.step()

            if torch.isnan(loss):
                logging.warning(f'NaN at S1 step {train_steps}')
                continue

            epoch_loss += loss.item()
            train_steps += 1
            pbar.set_postfix(
                {'loss': f'{epoch_loss / train_steps:.4f}'}
            )

        # 验证
        avg_loss = epoch_loss / max(train_steps, 1)
        result = validate_with_intervals(
            model, val_loader, label_encoder,
            all_class_counts, counts_max, counts_min, device,
        )
        score = result['score']
        interval_str = ' | '.join(
            f'{k}:{v:.4f}' for k, v in result['intervals'].items()
        )

        logging.info(
            f'  S1 Epoch {epoch+1}: loss={avg_loss:.4f}, '
            f'score={score:.4f} '
            f'({result["val_classes"]}/{len(all_class_counts)} classes), '
            f'acc={result["accuracy"]:.4f}, '
            f'fewshot={result["fewshot_score"]:.4f}'
        )
        logging.info(f'    区间: {interval_str}')

        if score > best_score_s1:
            best_score_s1 = score
            patience_s1_counter = 0
            torch.save(
                model.state_dict(),
                os.path.join(fold_save_dir, 'best_model_s1.pt'),
            )
            logging.info(f'    最优S1模型已保存 (score={score:.4f})')
        else:
            patience_s1_counter += 1
            logging.info(
                f'    未提升 ({patience_s1_counter}/{args.patience})'
            )
            if patience_s1_counter >= args.patience:
                logging.info(f'    S1 Early Stopping at epoch {epoch+1}')
                break

    # 加载最优 S1 模型
    s1_path = os.path.join(fold_save_dir, 'best_model_s1.pt')
    if os.path.exists(s1_path):
        state_dict = torch.load(s1_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        logging.info(f'加载最优S1模型, score={best_score_s1:.4f}')


def _compute_stage1_loss(model, input_ids, mask, labels,
                         ce_loss_fn, focal_loss_fn, rdrop_loss,
                         supcon_loss, spancl_loss, triplet_loss,
                         desc_align_loss, args, device):
    """计算 Stage 1 的复合损失"""
    loss_total = 0.0

    if rdrop_loss is not None:
        logits1, cls1, hidden1 = model(input_ids, mask)
        logits2, cls2, hidden2 = model(input_ids, mask)

        # 主损失: Focal + CE（两次前向取平均）
        loss_main = (
            focal_loss_fn(logits1, labels) +
            focal_loss_fn(logits2, labels) +
            ce_loss_fn(logits1, labels) +
            ce_loss_fn(logits2, labels)
        ) / 4.0

        # R-Drop
        loss_rdrop = rdrop_loss(logits1, logits2)

        loss_total = loss_main + loss_rdrop

        # 使用 cls1/hidden1 做辅助损失
        cls_emb = cls1
        hidden_states = hidden1
    else:
        logits1, cls_emb, hidden_states = model(input_ids, mask)

        # 主损失: Focal + CE
        loss_main = (
            focal_loss_fn(logits1, labels) +
            ce_loss_fn(logits1, labels)
        ) / 2.0

        loss_total = loss_main

    # SupCon
    if supcon_loss is not None and args.supcon_weight > 0:
        loss_supcon = supcon_loss(cls_emb, labels)
        loss_total = loss_total + args.supcon_weight * loss_supcon

    # SpanCL
    if spancl_loss is not None and args.spancl_weight > 0:
        loss_spancl = spancl_loss(hidden_states, input_ids, labels)
        loss_total = loss_total + args.spancl_weight * loss_spancl

    # TripletMargin
    if triplet_loss is not None and args.triplet_weight > 0:
        loss_triplet = triplet_loss(cls_emb, labels)
        loss_total = loss_total + args.triplet_weight * loss_triplet

    # Description Alignment
    if desc_align_loss is not None and args.desc_align_weight > 0:
        loss_align = desc_align_loss(cls_emb, labels)
        loss_total = loss_total + args.desc_align_weight * loss_align

    return loss_total


def _run_stage2(args, model, train_dataset, train_df, val_loader,
                label_encoder, all_class_counts, counts_max, counts_min,
                device, scaler, fold_save_dir, class_counts_list):
    """Stage 2: CWBS 校准 (cRT: freeze encoder, train classifier)"""

    logging.info('========== STAGE 2: CWBS 校准 ==========')
    logging.info('  冻结 encoder，仅训练分类头')

    # 冻结 encoder
    for param in model.encoder.parameters():
        param.requires_grad = False

    # CWBS 损失 + 自适应 tau
    tau_per_class = adaptive_tau_per_class(
        class_counts_list, counts_max, counts_min,
        tau_min=args.tau_min, tau_max=args.tau_max,
    ) if args.adaptive_tau else None

    cwbs_loss_fn = CWBSLoss(
        class_counts_list, counts_max, counts_min,
        tau_per_class=tau_per_class if args.adaptive_tau else None,
        tau=args.tau_fixed,
        weight_min=args.cwbs_weight_min,
    )

    # CWBS 采样器
    cwbs_weights = {
        c: (counts_max - all_class_counts.get(c, 0) + counts_min * 0.1) /
           (counts_max + counts_min * 0.1)
        for c in all_class_counts
    }
    cwbs_sampler = CWBSWeightedSampler(
        train_df, cwbs_weights, args.batch_size,
        cwbs_ratio=args.cwbs_ratio,
        samples_per_epoch=len(train_df) * 2,
    )

    total_steps_s2 = (
        (len(train_df) * 2 // args.batch_size) * args.epochs_stage2
    )
    opt_s2 = torch.optim.AdamW(
        [p for n, p in model.named_parameters() if p.requires_grad],
        lr=args.lr_stage2, weight_decay=0.0,
    )
    sched_s2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_s2, T_max=total_steps_s2,
    )

    best_score_s2 = 0.0

    for epoch in range(args.epochs_stage2):
        model.train()
        epoch_loss = 0.0
        train_steps = 0

        pbar = tqdm(cwbs_sampler,
                     desc=f'S2 Epoch {epoch+1}/{args.epochs_stage2}')
        for batch_indices in pbar:
            batch_samples = [
                train_dataset[int(idx)] for idx in batch_indices
            ]
            batch = dynamic_collate_fn(batch_samples)
            if batch is None:
                continue

            input_ids = batch['data'].to(device)
            mask = batch['cls_mask'].to(device)
            labels = batch['label'].to(device)

            opt_s2.zero_grad()

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    logits, _, _ = model(input_ids, mask)
                    loss = cwbs_loss_fn(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(opt_s2)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=args.grad_clip
                )
                scaler.step(opt_s2)
                scaler.update()
            else:
                logits, _, _ = model(input_ids, mask)
                loss = cwbs_loss_fn(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=args.grad_clip
                )
                opt_s2.step()

            sched_s2.step()

            if torch.isnan(loss):
                continue

            epoch_loss += loss.item()
            train_steps += 1
            pbar.set_postfix(
                {'loss': f'{epoch_loss / max(train_steps, 1):.4f}'}
            )

        avg_loss = epoch_loss / max(train_steps, 1)
        result = validate_with_intervals(
            model, val_loader, label_encoder,
            all_class_counts, counts_max, counts_min, device,
        )
        score = result['score']
        interval_str = ' | '.join(
            f'{k}:{v:.4f}' for k, v in result['intervals'].items()
        )

        logging.info(
            f'  S2 Epoch {epoch+1}: loss={avg_loss:.4f}, '
            f'score={score:.4f} '
            f'({result["val_classes"]}/{len(all_class_counts)} classes), '
            f'acc={result["accuracy"]:.4f}, '
            f'fewshot={result["fewshot_score"]:.4f}'
        )
        logging.info(f'    区间: {interval_str}')

        if score > best_score_s2:
            best_score_s2 = score
            torch.save(
                model.state_dict(),
                os.path.join(fold_save_dir, 'best_model.pt'),
            )
            logging.info(f'    最优模型已保存 (score={score:.4f})')

    torch.save(
        model.state_dict(),
        os.path.join(fold_save_dir, 'model_final.pt'),
    )
    logging.info(f'S2完成, best score={best_score_s2:.4f}')


def main():
    parser = argparse.ArgumentParser(
        description='岗位分类系统 — 两阶段训练'
    )

    # ─── 基础参数 ───
    parser.add_argument('--model_name', type=str,
                        default='microsoft/deberta-v3-base')
    parser.add_argument('--data_dir', type=str,
                        default='../zhilian_direct')
    parser.add_argument('--synthetic_data_path', type=str,
                        default='./outputs/synthetic_data.csv')
    parser.add_argument('--prototype_path', type=str,
                        default='./outputs/prototype_matrix.npy')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--n_folds', type=int, default=5)

    # ─── Stage 1 ───
    parser.add_argument('--epochs_stage1', type=int, default=15)
    parser.add_argument('--lr_stage1', type=float, default=2e-5)
    parser.add_argument('--lr_classifier_s1', type=float, default=5e-5)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--rdrop_weight', type=float, default=0.1)
    parser.add_argument('--supcon_weight', type=float, default=0.2)
    parser.add_argument('--spancl_weight', type=float, default=0.1)
    parser.add_argument('--triplet_weight', type=float, default=0.05)
    parser.add_argument('--desc_align_weight', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--ib_ratio', type=float, default=0.5)
    parser.add_argument('--no_s2', action='store_true', default=False)
    parser.add_argument('--skip_s1', action='store_true', default=False)
    parser.add_argument('--resume_s1_path', type=str, default=None)
    parser.add_argument('--patience', type=int, default=5)

    # ─── Stage 2 ───
    parser.add_argument('--epochs_stage2', type=int, default=8)
    parser.add_argument('--lr_stage2', type=float, default=1e-4)
    parser.add_argument('--tau_fixed', type=float, default=1.0)
    parser.add_argument('--adaptive_tau', action='store_true', default=True)
    parser.add_argument('--tau_min', type=float, default=0.3)
    parser.add_argument('--tau_max', type=float, default=1.5)
    parser.add_argument('--cwbs_ratio', type=float, default=0.8)
    parser.add_argument('--cwbs_weight_min', type=float, default=0.1)

    # ─── 通用 ───
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--proto_init_scale', type=float, default=0.05)
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--encoding_format', type=str, default='A')

    args = parser.parse_args()
    run_training(args)


if __name__ == '__main__':
    main()
