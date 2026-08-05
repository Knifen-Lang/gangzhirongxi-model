"""
评估指标 — 岗位分类

直接迁移自 人工智能挑战赛 v3&v4/utils_v3/evaluation.py

核心指标:
- Competition Score: 比赛加权分数（少样本类权重大）
- Interval Scores: 按样本数分区间评估
- Few-shot Score: ≤5条样本类的准确率
- Temperature Search: 最优温度缩放
- Confusion Analysis: 混淆矩阵分析
"""

import numpy as np
import torch


def compute_competition_score(y_true, y_pred, class_counts,
                               counts_max, counts_min):
    """
    比赛加权分数

    每个类的权重 = 1 - normalized_count
    少样本类 → 高权重，多样本类 → 低权重
    """
    y_true = np.asarray(y_true, dtype=str)
    y_pred = np.asarray(y_pred, dtype=str)
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))

    m_weights = {}
    m_scores = {}
    for m in unique_classes:
        count_m = class_counts.get(m, 0)
        m_weights[m] = (
            (counts_max - count_m + counts_min * 0.1) /
            (counts_max + counts_min * 0.1)
        )
        mask = (y_true == m)
        m_total = mask.sum()
        m_scores[m] = (
            (y_pred[mask] == m).sum() / m_total if m_total > 0 else 0.0
        )

    numerator = sum(m_weights[m] * m_scores[m] for m in unique_classes)
    denominator = sum(m_weights[m] for m in unique_classes)
    score = numerator / denominator if denominator > 0 else 0.0
    return score, m_scores


def compute_interval_scores(y_true, y_pred, class_counts):
    """按样本数区间评估"""
    y_true = np.asarray(y_true, dtype=str)
    y_pred = np.asarray(y_pred, dtype=str)

    intervals = {
        '<=5': (1, 5),
        '6-10': (6, 10),
        '11-20': (11, 20),
        '21-50': (21, 50),
        '51-100': (51, 100),
        '>100': (101, 999999),
    }
    results = {}
    for name, (lo, hi) in intervals.items():
        classes_in_range = [
            c for c, cnt in class_counts.items() if lo <= cnt <= hi
        ]
        if not classes_in_range:
            results[name] = 0.0
            continue
        correct = 0
        total = 0
        for c in classes_in_range:
            mask = (y_true == c)
            c_total = int(mask.sum())
            if c_total > 0:
                correct += int((y_pred[mask] == c).sum())
                total += c_total
        results[name] = correct / total if total > 0 else 0.0
    return results


def compute_fewshot_score(y_true, y_pred, class_counts, max_count=5):
    """少样本类准确率"""
    y_true = np.asarray(y_true, dtype=str)
    y_pred = np.asarray(y_pred, dtype=str)
    fewshot_classes = [
        c for c, cnt in class_counts.items() if cnt <= max_count
    ]
    if not fewshot_classes:
        return 0.0
    correct = 0
    total = 0
    for c in fewshot_classes:
        mask = (y_true == c)
        c_total = mask.sum()
        if c_total > 0:
            correct += (y_pred[mask] == c).sum()
            total += c_total
    return correct / total if total > 0 else 0.0


def validate(model, val_loader, label_encoder, class_counts,
             counts_max, counts_min, device='cuda'):
    """基本验证"""
    model.eval()
    all_preds, all_labels = [], []
    total_correct, total_samples = 0, 0

    dev = torch.device(device if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            input_ids = batch['data'].to(dev)
            mask = batch['cls_mask'].to(dev)
            labels = batch['label'].to(dev)

            logits, _, _ = model(input_ids, mask)
            preds = torch.argmax(logits, dim=1)

            total_correct += int((preds == labels).sum().item())
            total_samples += int(labels.shape[0])
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    pred_names = label_encoder.inverse_transform(np.array(all_preds))
    label_names = label_encoder.inverse_transform(np.array(all_labels))

    score, _ = compute_competition_score(
        label_names, pred_names, class_counts, counts_max, counts_min
    )
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    return {
        'score': score, 'accuracy': accuracy,
        'y_true': label_names, 'y_pred': pred_names,
    }


def validate_with_intervals(model, val_loader, label_encoder,
                            class_counts, counts_max, counts_min, device='cuda'):
    """完整验证 + 区间分数"""
    model.eval()
    all_preds, all_labels = [], []
    total_correct, total_samples = 0, 0

    dev = torch.device(device if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            input_ids = batch['data'].to(dev)
            mask = batch['cls_mask'].to(dev)
            labels = batch['label'].to(dev)

            logits, _, _ = model(input_ids, mask)
            preds = torch.argmax(logits, dim=1)

            total_correct += int((preds == labels).sum().item())
            total_samples += int(labels.shape[0])
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    pred_names = label_encoder.inverse_transform(np.array(all_preds))
    label_names = label_encoder.inverse_transform(np.array(all_labels))

    val_classes = set(label_names)
    total_classes = len(class_counts)
    missing_classes = total_classes - len(val_classes)

    score, m_scores = compute_competition_score(
        label_names, pred_names, class_counts, counts_max, counts_min
    )
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    interval_scores = compute_interval_scores(label_names, pred_names, class_counts)
    fewshot_score = compute_fewshot_score(label_names, pred_names, class_counts, max_count=5)

    return {
        'score': score,
        'accuracy': accuracy,
        'intervals': interval_scores,
        'fewshot_score': fewshot_score,
        'val_classes': len(val_classes),
        'missing_classes': missing_classes,
        'y_true': label_names,
        'y_pred': pred_names,
    }


def search_temperature(model, val_loader, label_encoder, class_counts,
                       counts_max, counts_min, device='cuda', temps=None):
    """搜索最优温度缩放参数"""
    if temps is None:
        temps = [1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]

    model.eval()
    all_logits_list = []
    all_labels_list = []

    dev = torch.device(device if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            input_ids = batch['data'].to(dev)
            mask = batch['cls_mask'].to(dev)
            labels = batch['label'].to(dev)
            logits, _, _ = model(input_ids, mask)
            all_logits_list.append(logits.cpu().numpy())
            all_labels_list.append(labels.cpu().numpy())

    all_logits = np.concatenate(all_logits_list, axis=0)
    all_labels = label_encoder.inverse_transform(
        np.concatenate(all_labels_list).astype(int)
    )

    best_temp, best_score = 1.0, 0.0
    print(f'{"Temp":>8}  {"Score":>8}  {"Accuracy":>8}  {"Few-shot":>8}')
    print('-' * 42)
    for temp in temps:
        scaled = all_logits / temp
        preds = label_encoder.inverse_transform(
            np.argmax(scaled, axis=1).astype(int)
        )
        score, _ = compute_competition_score(
            all_labels, preds, class_counts, counts_max, counts_min
        )
        accuracy = (preds == all_labels).mean()
        fewshot = compute_fewshot_score(all_labels, preds, class_counts, max_count=5)
        marker = ' *' if score > best_score else ''
        print(
            f'{temp:>8.2f}  {score:>8.4f}  {accuracy:>8.4f}  {fewshot:>8.4f}{marker}'
        )
        if score > best_score:
            best_score = score
            best_temp = temp

    print(f'\n最优温度: {best_temp:.2f} (score={best_score:.4f})')
    return best_temp, best_score


def compute_confusion_by_interval(y_true, y_pred, label_names, class_counts):
    """按区间分析混淆"""
    intervals = {
        '<=5': 5, '6-10': 10, '11-20': 20,
        '21-50': 50, '51-100': 100, '>100': 999999,
    }
    y_true = np.asarray(y_true, dtype=str)
    y_pred = np.asarray(y_pred, dtype=str)

    results = {}
    for name, max_count in intervals.items():
        if name == '>100':
            few_classes = [
                c for c, cnt in class_counts.items() if cnt > 100
            ]
        else:
            few_classes = [
                c for c, cnt in class_counts.items() if cnt <= max_count
            ]

        mask = np.isin(
            y_true, [c for c in label_names if c in few_classes]
        )
        if mask.sum() == 0:
            results[name] = {'total': 0, 'correct': 0,
                             'top_misclassified': []}
            continue

        sub_true = y_true[mask]
        sub_pred = y_pred[mask]
        correct = (sub_true == sub_pred).sum()
        total = mask.sum()

        errors = {}
        for i in range(total):
            if sub_true[i] != sub_pred[i]:
                key = f"{sub_true[i]} → {sub_pred[i]}"
                errors[key] = errors.get(key, 0) + 1

        top_errors = sorted(errors.items(), key=lambda x: -x[1])[:10]
        results[name] = {
            'total': total, 'correct': correct,
            'ratio': correct / total if total > 0 else 0,
            'top_misclassified': top_errors,
        }

    return results
