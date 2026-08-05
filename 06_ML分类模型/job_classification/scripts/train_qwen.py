"""
Qwen2.5 + LoRA 微调训练脚本

核心设计:
- 复用现有的全部损失函数 (CWBS, Focal, SupCon, SpanCL, RDrop, Triplet)
- 复用现有的评估体系 (validate_with_intervals, competition_score)
- 复用现有的数据处理 (load_zhilian_data, CWBSWeightedSampler)
- 三阶段训练: S1=LoRA+分类头联合训练, S2=CWBS分类头校准

8GB 显存优化:
- batch_size=4, gradient_accumulation=4 → effective_batch=16
- 4-bit QLoRA + gradient_checkpointing
- max_length=512 (Qwen 支持更长输入)

用法:
  python scripts/train_qwen.py
  python scripts/train_qwen.py --n_folds 1 --epochs_s1 5 --epochs_s2 3
"""

# ═══ 必须在导入 transformers 之前设置 HF 镜像 ═══
import os as _os
if not _os.environ.get('HF_ENDPOINT'):
    _os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import argparse
import math
import os
import sys
import random
import logging
import json
from datetime import datetime
from typing import Optional, Dict, List, Tuple

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

from src.models.qwen_classifier import (
    QwenJobClassifier, load_qwen_tokenizer, encode_for_qwen,
)
from src.losses import (
    CWBSLoss, FocalLoss, SupConLoss, SpanContrastiveLoss,
    TripletMarginLoss, RDropLoss, asymmetric_label_smoothing,
)
from src.data.data_utils import (
    load_zhilian_data, load_synthetic_data, get_class_counts,
    CWBSWeightedSampler, dynamic_collate_fn,
    stratified_kfold_split, adaptive_tau_per_class, save_label_classes,
)
from src.utils.evaluation import validate_with_intervals


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    root_logger = logging.getLogger()
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
    root_logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    fh = logging.FileHandler(
        os.path.join(save_dir, 'train.log'), mode='w', encoding='utf-8'
    )
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    root_logger.addHandler(fh)
    root_logger.addHandler(ch)


class QwenJobDataset(torch.utils.data.Dataset):
    """
    适配 Qwen tokenizer 的岗位分类 Dataset

    与现有 JobDataset 保持相同的接口，内部使用 encode_for_qwen()
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer,
        label_encoder: LabelEncoder,
        max_length: int = 512,
        encoding_format: str = "B",
    ):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.le = label_encoder
        self.max_length = max_length
        self.encoding_format = encoding_format

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        row = self.data.iloc[idx]

        input_ids, attention_mask = encode_for_qwen(
            self.tokenizer,
            str(row['text_a']),
            str(row['text_b']),
            self.max_length,
            encoding_format=self.encoding_format,
        )

        label_id = self.le.transform([row['label']])[0]
        is_synthetic = row.get('is_synthetic', False)
        sample_weight = 0.75 if is_synthetic else 1.0

        return {
            'valid': True,
            'token_ids': input_ids,
            'cls_mask': attention_mask,
            'label_id': np.int64(label_id),
            'sample_weight': np.float32(sample_weight),
        }


def resolve_device(device_arg: str) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


# ═══════════════════════════════════════════════════════════════
#  Stage 1: LoRA + 分类头联合训练
# ═══════════════════════════════════════════════════════════════

def run_stage1(
    args, model, train_df, val_loader, label_encoder,
    all_class_counts, counts_max, counts_min, device,
    scaler, fold_save_dir, class_counts_list,
):
    """Stage 1: LoRA + 分类头联合表示学习"""

    logging.info('========== STAGE 1: LoRA + 表示学习 ==========')
    logging.info(
        f'  损失: Focal(γ={args.focal_gamma}) + LS({args.label_smoothing}) '
        f'+ R-Drop({args.rdrop_weight})'
        f' + SupCon({args.supcon_weight}) + SpanCL({args.spancl_weight})'
    )

    # 确保 LoRA + 分类头可训练
    model.freeze_base_model()  # 冻结基座，只训练 LoRA
    for param in model.classifier.parameters():
        param.requires_grad = True

    # ── Tokenizer ──
    tokenizer = load_qwen_tokenizer(args.model_name)

    # ── IB + Uniform 混合采样 ──
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

    train_dataset = QwenJobDataset(
        train_df, tokenizer, label_encoder,
        max_length=args.max_length,
        encoding_format=args.encoding_format,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=mixed_sampler,
        collate_fn=dynamic_collate_fn,
        num_workers=args.num_workers,
    )

    # ── 优化器 ──
    total_steps_s1 = len(train_loader) * args.epochs_stage1 // args.grad_accum
    opt_s1 = torch.optim.AdamW([
        {
            'params': [p for n, p in model.named_parameters()
                      if 'lora' in n.lower() and p.requires_grad],
            'lr': args.lr_stage1, 'weight_decay': 0.01,
        },
        {
            'params': model.classifier.parameters(),
            'lr': args.lr_classifier_s1, 'weight_decay': 0.01,
        },
    ])

    warmup_steps = int(args.warmup_ratio * total_steps_s1)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(
            max(1, total_steps_s1 - warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    sched_s1 = torch.optim.lr_scheduler.LambdaLR(opt_s1, lr_lambda)

    # ── 损失函数 ──
    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    focal_loss_fn = FocalLoss(
        class_counts_list, counts_max, counts_min,
        gamma=args.focal_gamma,
    )
    rdrop_loss = RDropLoss(alpha=args.rdrop_weight) if args.rdrop_weight > 0 else None
    supcon_loss = SupConLoss(temperature=0.07) if args.supcon_weight > 0 else None
    spancl_loss = SpanContrastiveLoss() if args.spancl_weight > 0 else None
    triplet_loss = TripletMarginLoss(margin=0.3) if args.triplet_weight > 0 else None

    best_score = 0.0
    patience_counter = 0

    for epoch in range(args.epochs_stage1):
        model.train()
        epoch_loss = 0.0
        train_steps = 0
        opt_s1.zero_grad()

        pbar = tqdm(train_loader,
                     desc=f'S1 Epoch {epoch+1}/{args.epochs_stage1}')
        for batch in pbar:
            if batch is None:
                continue

            input_ids = batch['data'].to(device)
            mask = batch['cls_mask'].to(device)
            labels = batch['label'].to(device)

            # ── 损失计算 ──
            if rdrop_loss is not None:
                # R-Drop: 两次前向
                logits1, cls1, hidden1 = model(input_ids, mask)
                logits2, cls2, hidden2 = model(input_ids, mask)

                loss_main = (
                    focal_loss_fn(logits1, labels) +
                    focal_loss_fn(logits2, labels) +
                    ce_loss_fn(logits1, labels) +
                    ce_loss_fn(logits2, labels)
                ) / 4.0
                loss_rd = rdrop_loss(logits1, logits2)
                loss_total = loss_main + loss_rd
                cls_emb = cls1
                hidden_states = hidden1
            else:
                logits1, cls_emb, hidden_states = model(input_ids, mask)
                loss_total = (
                    focal_loss_fn(logits1, labels) +
                    ce_loss_fn(logits1, labels)
                ) / 2.0

            # 辅助损失
            if supcon_loss is not None and args.supcon_weight > 0:
                loss_total = loss_total + args.supcon_weight * supcon_loss(cls_emb, labels)

            if spancl_loss is not None and args.spancl_weight > 0:
                loss_total = loss_total + args.spancl_weight * spancl_loss(
                    hidden_states, input_ids, labels
                )

            if triplet_loss is not None and args.triplet_weight > 0:
                loss_total = loss_total + args.triplet_weight * triplet_loss(cls_emb, labels)

            # ── 梯度累积 ──
            loss_total = loss_total / args.grad_accum

            if scaler is not None:
                scaler.scale(loss_total).backward()
            else:
                loss_total.backward()

            train_steps += 1

            if train_steps % args.grad_accum == 0:
                if scaler is not None:
                    scaler.unscale_(opt_s1)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip
                    )
                    scaler.step(opt_s1)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip
                    )
                    opt_s1.step()

                sched_s1.step()
                opt_s1.zero_grad()

            if not torch.isnan(loss_total):
                epoch_loss += loss_total.item() * args.grad_accum

            pbar.set_postfix({
                'loss': f'{epoch_loss / max(train_steps, 1):.4f}',
                'lr': f'{sched_s1.get_last_lr()[0]:.2e}',
            })

        # ── 验证 ──
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
            f'  S1 E{epoch+1}: loss={avg_loss:.4f}, score={score:.4f}, '
            f'acc={result["accuracy"]:.4f}, '
            f'fewshot={result["fewshot_score"]:.4f}'
        )
        logging.info(f'    区间: {interval_str}')

        if score > best_score:
            best_score = score
            patience_counter = 0
            model.save_pretrained(
                os.path.join(fold_save_dir, 'best_model_s1')
            )
            logging.info(f'    ★ 最优S1模型已保存 (score={score:.4f})')
        else:
            patience_counter += 1
            logging.info(f'    未提升 ({patience_counter}/{args.patience})')
            if patience_counter >= args.patience:
                logging.info(f'    S1 Early Stopping at epoch {epoch+1}')
                break

    # 加载最优 S1 模型
    s1_path = os.path.join(fold_save_dir, 'best_model_s1')
    if os.path.exists(os.path.join(s1_path, 'lora_adapter')):
        model.load_pretrained(s1_path)
        logging.info(f'  S1完成, best_score={best_score:.4f}')


# ═══════════════════════════════════════════════════════════════
#  Stage 2: CWBS 分类头校准
# ═══════════════════════════════════════════════════════════════

def run_stage2(
    args, model, train_df, val_loader, label_encoder,
    all_class_counts, counts_max, counts_min, device,
    scaler, fold_save_dir, class_counts_list,
):
    """Stage 2: 冻结 LoRA + 基座，仅训练分类头 + CWBS 校准"""

    logging.info('========== STAGE 2: CWBS 分类头校准 ==========')
    logging.info('  冻结 LoRA+基座, 仅训练分类头')

    # 冻结所有，只训练分类头
    model.freeze_all_but_classifier()

    tokenizer = load_qwen_tokenizer(args.model_name)

    # ── CWBS 损失 ──
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

    # ── CWBS 采样器 ──
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

    train_dataset = QwenJobDataset(
        train_df, tokenizer, label_encoder,
        max_length=args.max_length,
        encoding_format=args.encoding_format,
    )

    # ── 优化器: 仅分类头 ──
    opt_s2 = torch.optim.AdamW(
        [p for p in model.classifier.parameters() if p.requires_grad],
        lr=args.lr_stage2,
        weight_decay=0.0,
    )

    total_steps_s2 = (
        (len(train_df) * 2 // args.batch_size) * args.epochs_stage2
    )
    sched_s2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_s2, T_max=total_steps_s2,
    )

    best_score = 0.0

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
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(opt_s2)
                scaler.update()
            else:
                logits, _, _ = model(input_ids, mask)
                loss = cwbs_loss_fn(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt_s2.step()

            sched_s2.step()

            if not torch.isnan(loss):
                epoch_loss += loss.item()
                train_steps += 1
                pbar.set_postfix({'loss': f'{epoch_loss / max(train_steps,1):.4f}'})

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
            f'  S2 E{epoch+1}: loss={avg_loss:.4f}, score={score:.4f}, '
            f'acc={result["accuracy"]:.4f}, '
            f'fewshot={result["fewshot_score"]:.4f}'
        )
        logging.info(f'    区间: {interval_str}')

        if score > best_score:
            best_score = score
            model.save_pretrained(
                os.path.join(fold_save_dir, 'best_model')
            )
            logging.info(f'    ★ 最优模型已保存 (score={score:.4f})')

    model.save_pretrained(os.path.join(fold_save_dir, 'model_final'))
    logging.info(f'  S2完成, best_score={best_score:.4f}')


# ═══════════════════════════════════════════════════════════════
#  主训练函数
# ═══════════════════════════════════════════════════════════════

def run_training(args):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.output_dir, f'qwen_finetune_{timestamp}')
    setup_logging(save_dir)
    set_seed(args.random_seed)
    device = resolve_device(args.device)

    logging.info(f'设备: {device}')
    logging.info(f'模型: {args.model_name}')
    logging.info(f'参数: {vars(args)}')

    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logging.info(f'GPU 显存: {vram:.1f}GB')
        logging.info(f'Effective batch: {args.batch_size}×{args.grad_accum}='
                     f'{args.batch_size * args.grad_accum}')

    # ═══════════════════════════════════════════════════════
    # 1. 加载数据
    # ═══════════════════════════════════════════════════════
    logging.info('=' * 60)
    logging.info('加载数据...')

    raw_train_df = load_zhilian_data(args.data_dir)
    logging.info(f'原始训练样本: {len(raw_train_df)}')

    synthetic_df = load_synthetic_data(args.synthetic_data_path)
    logging.info(f'合成样本: {len(synthetic_df)}')

    combined_df = pd.concat([raw_train_df, synthetic_df], ignore_index=True)
    logging.info(f'合并后: {len(combined_df)}')

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

    # ═══════════════════════════════════════════════════════
    # 2. 加载 Tokenizer
    # ═══════════════════════════════════════════════════════
    logging.info('加载 tokenizer...')
    tokenizer = load_qwen_tokenizer(args.model_name)

    # ═══════════════════════════════════════════════════════
    # 3. 划分 Folds
    # ═══════════════════════════════════════════════════════
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
    logging.info(f'创建 {len(folds)} 个分层 fold')

    # ═══════════════════════════════════════════════════════
    # 4. 加载原型矩阵 (可选, 用于分类头初始化)
    # ═══════════════════════════════════════════════════════
    prototype_matrix = None
    if args.prototype_path and os.path.exists(args.prototype_path):
        prototype_matrix = np.load(args.prototype_path)
        logging.info(f'原型矩阵: {prototype_matrix.shape}')
    else:
        logging.info('无原型矩阵, 分类头随机初始化')

    # ═══════════════════════════════════════════════════════
    # 5. 训练每个 Fold
    # ═══════════════════════════════════════════════════════
    all_fold_results = []

    for fold_idx in range(args.n_folds):
        logging.info(f'\n{"="*60}')
        logging.info(f'  Fold {fold_idx + 1}/{args.n_folds}')
        logging.info(f'{"="*60}')

        train_df, val_df = folds[fold_idx]
        fold_save_dir = os.path.join(save_dir, f'fold_{fold_idx}')
        os.makedirs(fold_save_dir, exist_ok=True)

        # ── 创建模型 ──
        lora_cfg = {
            "r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            "bias": "none",
        }

        logging.info(f'初始化 Qwen2.5-7B + LoRA(r={args.lora_r})...')
        model = QwenJobClassifier(
            model_name=args.model_name,
            num_labels=num_classes,
            lora_config=lora_cfg,
            use_4bit=True,
            dropout=args.dropout,
            prototype_matrix=prototype_matrix,
            proto_init_scale=args.proto_init_scale,
        )

        # ── 验证集 DataLoader ──
        val_dataset = QwenJobDataset(
            val_df, tokenizer, label_encoder,
            max_length=args.max_length,
            encoding_format=args.encoding_format,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=dynamic_collate_fn, num_workers=args.num_workers,
        )

        scaler = torch.amp.GradScaler('cuda') if (
            device.type == 'cuda' and args.use_amp
        ) else None

        # ── Stage 1: LoRA + 分类头表示学习 ──
        if not args.skip_s1 and args.epochs_stage1 > 0:
            run_stage1(
                args, model, train_df, val_loader, label_encoder,
                all_class_counts, counts_max, counts_min, device,
                scaler, fold_save_dir, class_counts_list,
            )
        elif args.skip_s1:
            # 跳过 S1, 从指定目录加载已保存的 best_model_s1
            resume_dir = args.resume_from_dir if args.resume_from_dir else save_dir
            s1_path = os.path.join(resume_dir, f'fold_{fold_idx}', 'best_model_s1')
            if os.path.exists(os.path.join(s1_path, 'lora_adapter')):
                model.load_pretrained(s1_path)
                logging.info(f'  已加载 S1 模型: {s1_path}')
            else:
                raise FileNotFoundError(
                    f'S1 模型不存在: {s1_path}. 请指定 --resume_from_dir 或先跑完 Stage 1.'
                )

        # ── Stage 2: CWBS 分类头校准 ──
        if not args.no_s2:
            run_stage2(
                args, model, train_df, val_loader, label_encoder,
                all_class_counts, counts_max, counts_min, device,
                scaler, fold_save_dir, class_counts_list,
            )

        all_fold_results.append({'fold': fold_idx})
        logging.info(f'Fold {fold_idx} 完成')

    logging.info(f'\n{"="*60}')
    logging.info(f'  全部训练完成! 模型保存至 {save_dir}')
    logging.info(f'{"="*60}')


def main():
    parser = argparse.ArgumentParser(
        description='Qwen2.5 + LoRA 微调 — 岗位分类'
    )

    # ─── 模型参数 ───
    parser.add_argument('--model_name', type=str,
                        default='Qwen/Qwen2.5-3B-Instruct')
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

    # ─── LoRA 参数 ───
    parser.add_argument('--lora_r', type=int, default=64)
    parser.add_argument('--lora_alpha', type=int, default=128)
    parser.add_argument('--lora_dropout', type=float, default=0.05)

    # ─── Stage 1 ───
    parser.add_argument('--epochs_stage1', type=int, default=6)
    parser.add_argument('--lr_stage1', type=float, default=5e-5)
    parser.add_argument('--lr_classifier_s1', type=float, default=2e-4)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--rdrop_weight', type=float, default=0.1)
    parser.add_argument('--supcon_weight', type=float, default=0.15)
    parser.add_argument('--spancl_weight', type=float, default=0.05)
    parser.add_argument('--triplet_weight', type=float, default=0.05)
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--ib_ratio', type=float, default=0.5)
    parser.add_argument('--no_s2', action='store_true', default=False)
    parser.add_argument('--skip_s1', action='store_true', default=False)
    parser.add_argument('--resume_from_dir', type=str, default='',
                        help='从已有目录加载 S1 模型 (配合 --skip_s1 使用)')
    parser.add_argument('--patience', type=int, default=3)

    # ─── Stage 2 ───
    parser.add_argument('--epochs_stage2', type=int, default=3)
    parser.add_argument('--lr_stage2', type=float, default=1e-4)
    parser.add_argument('--tau_fixed', type=float, default=1.0)
    parser.add_argument('--adaptive_tau', action='store_true', default=True)
    parser.add_argument('--tau_min', type=float, default=0.3)
    parser.add_argument('--tau_max', type=float, default=1.5)
    parser.add_argument('--cwbs_ratio', type=float, default=0.8)
    parser.add_argument('--cwbs_weight_min', type=float, default=0.1)

    # ─── 训练参数 ───
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--grad_accum', type=int, default=4,
                        help='梯度累积步数 (effective_batch = batch_size × grad_accum)')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--proto_init_scale', type=float, default=0.05)
    parser.add_argument('--use_amp', action='store_true', default=False,
                        help='使用 AMP 混合精度 (bfloat16 模型通常不需要)')
    parser.add_argument('--no_amp', action='store_false', dest='use_amp')
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--encoding_format', type=str, default='B')

    args = parser.parse_args()
    run_training(args)


if __name__ == '__main__':
    main()
