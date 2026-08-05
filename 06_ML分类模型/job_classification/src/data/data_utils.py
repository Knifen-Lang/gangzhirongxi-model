"""
数据处理工具 — 适配智联招聘 zhilian_direct 数据

迁移自 人工智能挑战赛 v3&v4/utils_v3/data_utils.py

核心适配:
- 输入: (简历文本/skill_requirements, 岗位名称) 代替 (Subject, Object)
- 数据源: zhilian_direct CSV 文件夹，每文件=一个岗位类
- 支持多格式编码: A(标准), B(描述增强), C(指令格式)
"""

import os
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  Tokenizer
# ═══════════════════════════════════════════════════════════════

def load_tokenizer(model_name):
    """加载 HuggingFace tokenizer"""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name)


# ═══════════════════════════════════════════════════════════════
#  Data Loading — 适配 zhilian_direct
# ═══════════════════════════════════════════════════════════════

def load_zhilian_data(dir_path, use_skill_requirements=True,
                      use_full_description=False):
    """
    从 zhilian_direct 文件夹加载所有岗位 CSV 数据

    CSV 格式:
      job_name, company_name, salary, work_area, city, education,
      work_year, issue_date, source, skill_requirements, tech_tags, job_url

    文件命名: zhilian_direct_{岗位名}.csv

    Args:
        dir_path: zhilian_direct CSV 文件夹路径
        use_skill_requirements: True=使用skill_requirements字段, False=使用文本描述
        use_full_description: True=拼接所有文本字段

    Returns:
        DataFrame: 包含 [text_a, text_b, label, is_synthetic] 列
          - text_a: 简历/技能文本（作为"候选人"侧）
          - text_b: 岗位名称+描述（作为"岗位"侧）
          - label: 岗位标签 = 文件名中的岗位名
    """
    all_data = []
    if not os.path.exists(dir_path):
        raise ValueError(f"找不到目录: {dir_path}")

    # 查找 CSV 文件（支持 zhilian_direct 双层嵌套）
    csv_files = []
    for root, dirs, files in os.walk(dir_path):
        for f in files:
            if f.endswith('.csv'):
                csv_files.append(os.path.join(root, f))

    print(f"从 {dir_path} 加载 {len(csv_files)} 个CSV文件...")

    for file_path in tqdm(csv_files, desc="加载数据"):
        filename = os.path.basename(file_path)
        # 从文件名提取岗位标签
        # zhilian_direct_Python.csv → Python
        # zhilian_direct_算法工程师.csv → 算法工程师
        label_name = filename[:-4].strip()  # 去掉 .csv
        # 去掉可能的 zhilian_direct_ 或 zhilian_direct 前缀
        for prefix in ['zhilian_direct_', 'zhilian_direct']:
            if label_name.startswith(prefix):
                label_name = label_name[len(prefix):]
                break

        try:
            df = pd.read_csv(file_path, low_memory=False, encoding='utf-8-sig')
            if df.empty:
                continue

            df.columns = [str(col).strip() for col in df.columns]

            # 构建 text_a (候选人侧) 和 text_b (岗位侧)
            text_a_list = []
            text_b_list = []

            for _, row in df.iterrows():
                # text_a: 候选人信息 = skill_requirements + tech_tags
                if use_skill_requirements:
                    skills = str(row.get('skill_requirements', ''))
                    tech = str(row.get('tech_tags', ''))

                    # 从 skill_requirements 中提取文本（去除网址等噪声）
                    if skills and skills != 'nan':
                        text_a = skills
                    elif tech and tech != 'nan':
                        text_a = tech
                    else:
                        text_a = label_name  # fallback

                    # 添加教育/经验信息
                    edu = str(row.get('education', ''))
                    exp = str(row.get('work_year', ''))
                    extra = []
                    if edu and edu != 'nan':
                        extra.append(f"学历:{edu}")
                    if exp and exp != 'nan':
                        extra.append(f"经验:{exp}")
                    if extra:
                        text_a = text_a + '。' + '，'.join(extra)
                else:
                    text_a = str(row.get('skill_requirements', label_name))

                # text_b: 岗位侧 = 岗位名 + 补充描述
                job_name = str(row.get('job_name', label_name))
                company = str(row.get('company_name', ''))
                area = str(row.get('work_area', ''))

                text_b_parts = [f"岗位:{job_name}"]
                if company and company != 'nan':
                    text_b_parts.append(f"公司:{company}")
                if area and area != 'nan':
                    text_b_parts.append(f"地点:{area}")

                text_b = '。'.join(text_b_parts)

                text_a_list.append(text_a)
                text_b_list.append(text_b)

            # 构建DataFrame行
            for ta, tb in zip(text_a_list, text_b_list):
                all_data.append({
                    'text_a': ta,
                    'text_b': tb,
                    'label': label_name,
                    'is_synthetic': False,
                })

        except Exception as e:
            print(f"警告: {filename} 加载出错: {e}")

    if not all_data:
        raise ValueError(f"{dir_path} 没有有效数据")

    full_df = pd.DataFrame(all_data)
    full_df['text_a'] = full_df['text_a'].astype(str)
    full_df['text_b'] = full_df['text_b'].astype(str)

    print(f"加载完成: {len(full_df)} 条数据, "
          f"{full_df['label'].nunique()} 个岗位类别")

    # 打印类别分布
    counts = full_df['label'].value_counts()
    print(f"  样本数范围: {counts.min()} ~ {counts.max()}")
    print(f"  ≤10条的类别: {(counts <= 10).sum()}")
    print(f"  ≤20条的类别: {(counts <= 20).sum()}")

    return full_df


def load_synthetic_data(path):
    """加载 LLM 合成数据"""
    if not path or not os.path.exists(path):
        print(f"未找到合成数据: {path}, 跳过.")
        return pd.DataFrame(columns=['text_a', 'text_b', 'label', 'is_synthetic'])

    df = pd.read_csv(path, low_memory=False, encoding='utf-8-sig')
    df.columns = [str(col).strip() for col in df.columns]

    # 适配列名
    text_a_col = next(
        (c for c in df.columns if c.lower() in ['text_a', 'resume', 'skill']), None
    )
    text_b_col = next(
        (c for c in df.columns if c.lower() in ['text_b', 'job', 'job_desc']), None
    )
    label_col = next(
        (c for c in df.columns if c.lower() in ['label', 'job_title', 'job_name']), None
    )

    if not all([text_a_col, text_b_col, label_col]):
        raise ValueError("合成数据必须包含 text_a/resume, text_b/job, label 列")

    out = pd.DataFrame()
    out['text_a'] = df[text_a_col].astype(str)
    out['text_b'] = df[text_b_col].astype(str)
    out['label'] = df[label_col].astype(str)
    out['is_synthetic'] = True
    out = out.dropna(subset=['text_a', 'text_b', 'label'])
    print(f"加载 {len(out)} 条合成数据 from {path}")
    return out


# ═══════════════════════════════════════════════════════════════
#  Class Statistics
# ═══════════════════════════════════════════════════════════════

def get_class_counts(df):
    """获取每类真实样本数（排除合成数据）"""
    if 'is_synthetic' in df.columns and df['is_synthetic'].any():
        real_df = df[~df['is_synthetic']]
    else:
        real_df = df
    return real_df['label'].value_counts().to_dict()


def adaptive_tau_per_class(class_counts, counts_max, counts_min,
                           tau_min=0.3, tau_max=1.5):
    """
    每类自适应 tau

    样本少 → tau 大（强频率校正）
    样本多 → tau 小（弱校正，保留模型判断力）
    """
    counts = np.array(class_counts)
    normalized = (counts - counts_min) / max(counts_max - counts_min, 1)
    tau = tau_min + (tau_max - tau_min) * (1 - normalized)
    return tau


# ═══════════════════════════════════════════════════════════════
#  Text Encoding
# ═══════════════════════════════════════════════════════════════

def encode_job_pair(tokenizer, text_a, text_b, max_length, encoding_format='A'):
    """
    编码 (候选人文本, 岗位文本) 对

    支持多种编码格式用于 TTA (Test-Time Augmentation):

    格式 A (标准): [CLS] text_a [SEP] text_b [SEP]
      适合: 标准双句分类

    格式 B (描述增强): [CLS] 岗位需求: text_b。候选人背景: text_a [SEP]
      适合: 强调岗位需求侧的匹配

    格式 C (指令格式): [CLS] 判断候选人是否适合岗位。岗位: text_b。简历: text_a [SEP]
      适合: GLiREL 风格的指令微调

    Args:
        tokenizer: HuggingFace tokenizer
        text_a: 候选人文本 (skills/resume)
        text_b: 岗位文本 (title + description)
        max_length: 最大长度
        encoding_format: 'A', 'B', 或 'C'

    Returns:
        input_ids, attention_mask (numpy arrays)
    """
    if encoding_format == 'A':
        encoding = tokenizer(
            text=text_a,
            text_pair=text_b,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
        )
    elif encoding_format == 'B':
        full_text = f"岗位需求: {text_b}。\n候选人背景: {text_a}"
        encoding = tokenizer(
            text=full_text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
        )
    elif encoding_format == 'C':
        full_text = f"判断以下候选人是否适合该岗位。\n{text_b}\n候选人简历: {text_a}"
        encoding = tokenizer(
            text=full_text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
        )
    else:
        raise ValueError(f"不支持的编码格式: {encoding_format}")

    input_ids = encoding['input_ids']
    attention_mask = encoding.get(
        'attention_mask', [1] * len(input_ids)
    )

    return np.array(input_ids, dtype='int64'), np.array(attention_mask, dtype='int64')


# ═══════════════════════════════════════════════════════════════
#  Dataset
# ═══════════════════════════════════════════════════════════════

class JobDataset(Dataset):
    """
    岗位分类 Dataset

    Args:
        dataframe: 包含 [text_a, text_b, label, is_synthetic?] 的 DataFrame
        tokenizer: HuggingFace tokenizer
        label_encoder: sklearn LabelEncoder
        max_length: 最大 token 长度
        encoding_format: 'A' | 'B' | 'C'
    """

    def __init__(self, dataframe, tokenizer, label_encoder, max_length=128,
                 encoding_format='A'):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.le = label_encoder
        self.max_length = max_length
        self.encoding_format = encoding_format

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        input_ids, attention_mask = encode_job_pair(
            self.tokenizer, row['text_a'], row['text_b'],
            self.max_length, encoding_format=self.encoding_format
        )

        label_id = self.le.transform([row['label']])[0]

        # 合成数据降权
        is_synthetic = row.get('is_synthetic', False)
        sample_weight = 0.75 if is_synthetic else 1.0

        return {
            'valid': True,
            'token_ids': input_ids,
            'cls_mask': attention_mask,
            'label_id': np.int64(label_id),
            'sample_weight': np.float32(sample_weight),
        }


def dynamic_collate_fn(samples):
    """动态 collate — 处理 batch 中的可变长度"""
    valid_samples = [s for s in samples if s.get('valid', False)]
    if not valid_samples:
        return None
    return {
        'data': torch.from_numpy(
            np.stack([s['token_ids'] for s in valid_samples])
        ).long(),
        'label': torch.tensor(
            [s['label_id'] for s in valid_samples], dtype=torch.long
        ),
        'cls_mask': torch.from_numpy(
            np.stack([s['cls_mask'] for s in valid_samples])
        ).long(),
    }


# ═══════════════════════════════════════════════════════════════
#  Samplers
# ═══════════════════════════════════════════════════════════════

class CWBSWeightedSampler:
    """
    CWBS 混合采样器

    70% CWBS 采样（少样本类权重大）+ 30% 均匀采样（保持头类多样性）

    Args:
        dataframe: 训练 DataFrame
        weights_dict: {label: weight} 比赛权重字典
        batch_size: batch 大小
        cwbs_ratio: CWBS 采样比例
        samples_per_epoch: 每 epoch 采样数
    """

    def __init__(self, dataframe, weights_dict, batch_size,
                 cwbs_ratio=0.7, samples_per_epoch=None):
        self.indices = np.arange(len(dataframe))
        self.batch_size = batch_size
        self.cwbs_ratio = cwbs_ratio

        labels = dataframe['label'].values
        sqrt_w = np.array(
            [np.sqrt(weights_dict.get(l, 1.0)) for l in labels]
        )
        sqrt_w_sum = sqrt_w.sum()
        self.cwbs_probs = (
            sqrt_w / sqrt_w_sum if sqrt_w_sum > 0
            else np.ones(len(labels)) / len(labels)
        )
        self.uniform_probs = np.ones(len(labels)) / len(labels)
        self.n_samples = samples_per_epoch or len(dataframe)

    def __iter__(self):
        n_cwbs = int(self.n_samples * self.cwbs_ratio)
        n_uniform = self.n_samples - n_cwbs

        cwbs_idx = (
            np.random.choice(self.indices, size=n_cwbs,
                             p=self.cwbs_probs, replace=True)
            if n_cwbs > 0 else np.array([], dtype=np.int64)
        )
        uniform_idx = (
            np.random.choice(self.indices, size=n_uniform,
                             p=self.uniform_probs, replace=True)
            if n_uniform > 0 else np.array([], dtype=np.int64)
        )

        all_idx = np.concatenate([cwbs_idx, uniform_idx])
        np.random.shuffle(all_idx)

        for i in range(0, len(all_idx), self.batch_size):
            yield all_idx[i:i + self.batch_size]

    def __len__(self):
        return max(1, self.n_samples // self.batch_size)


class ClassBalancedSampler:
    """
    Class-Balanced Sampling (Cui et al., CVPR 2019)

    每个类的采样概率 ∝ (1-β)/(1-β^n_y)
    β=0.999: 接近 Instance-Balanced
    """

    def __init__(self, dataframe, class_counts, beta=0.999):
        self.indices = np.arange(len(dataframe))
        labels = dataframe['label'].values
        effective_num = (
            1.0 - beta ** np.array([class_counts.get(l, 1) for l in labels])
        )
        class_weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
        self.probs = class_weights / class_weights.sum()
        self.probs = np.clip(self.probs, 0, None)
        self.probs = self.probs / (self.probs.sum() + 1e-12)
        self.n_samples = len(dataframe)

    def __iter__(self):
        idx = np.random.choice(
            self.indices, size=self.n_samples, p=self.probs, replace=True
        )
        return iter(idx)

    def __len__(self):
        return self.n_samples


# ═══════════════════════════════════════════════════════════════
#  Stratified K-Fold Split
# ═══════════════════════════════════════════════════════════════

def stratified_kfold_split(df, n_folds=5, random_state=42, val_ratio=0.1):
    """
    分层 K-Fold 划分

    保证每个 fold 中每类至少有 1 个样本。
    极端少样本类 (1条): 全部放入 train
    少于 n_folds 条的类: LOOCV 风格
    """
    np.random.seed(random_state)
    counts = df['label'].value_counts()

    if n_folds == 1:
        return _single_split(df, counts, val_ratio, random_state)

    n_total = len(counts)
    n_count1 = int((counts == 1).sum())
    logger.info(
        f'  Val split: {n_total} classes, '
        f'{n_count1} count=1 classes → train only'
    )

    fold_train_indices = [[] for _ in range(n_folds)]
    fold_val_indices = [[] for _ in range(n_folds)]

    for label, count in counts.items():
        label_df = df[df['label'] == label]
        idx = label_df.index.tolist()
        np.random.shuffle(idx)

        if count == 1:
            for f in range(n_folds):
                fold_train_indices[f].extend(idx)
        elif count < n_folds:
            for f in range(count):
                fold_val_indices[f].append(idx[f])
                fold_train_indices[f].extend(
                    [i for i in idx if i != idx[f]]
                )
            for f in range(count, n_folds):
                fold_train_indices[f].extend(idx)
        else:
            fold_size = max(1, count // n_folds)
            for f in range(n_folds):
                start = f * fold_size
                end = (f + 1) * fold_size if f < n_folds - 1 else count
                val_idx = idx[start:end]
                train_idx = [i for i in idx if i not in set(val_idx)]
                fold_val_indices[f].extend(val_idx)
                fold_train_indices[f].extend(train_idx)

    folds = []
    for f in range(n_folds):
        train_df = df.loc[fold_train_indices[f]].sample(
            frac=1, random_state=random_state
        ).reset_index(drop=True)
        val_df = df.loc[fold_val_indices[f]].reset_index(drop=True)
        folds.append((train_df, val_df))

    return folds


def _single_split(df, counts, val_ratio, random_state):
    """单 fold 划分"""
    np.random.seed(random_state)
    train_indices = []
    val_indices = []

    for label, count in counts.items():
        label_df = df[df['label'] == label]
        idx = label_df.index.tolist()
        np.random.shuffle(idx)

        if count == 1:
            train_indices.extend(idx)
        elif count < 10:
            train_indices.extend(idx[:-1])
            val_indices.append(idx[-1])
        else:
            n_val = max(1, int(count * val_ratio))
            val_idx = idx[:n_val]
            train_idx = idx[n_val:]
            train_indices.extend(train_idx)
            val_indices.extend(val_idx)

    val_classes = len(df.loc[val_indices]['label'].unique())
    logger.info(
        f'  Val set: {len(val_indices)} samples, '
        f'{val_classes} classes '
        f'(avg {len(val_indices)/max(val_classes,1):.1f} samples/class)'
    )

    train_df = df.loc[train_indices].sample(
        frac=1, random_state=random_state
    ).reset_index(drop=True)
    val_df = df.loc[val_indices].reset_index(drop=True)
    return [(train_df, val_df)]


# ═══════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════

def save_label_classes(label_encoder, save_dir):
    """保存标签类别列表"""
    path = os.path.join(save_dir, 'label_classes.txt')
    with open(path, 'w', encoding='utf-8') as f:
        for label in label_encoder.classes_:
            f.write(f'{label}\n')


def create_fold_dataloaders(folds, tokenizer, label_encoder, batch_size,
                            max_length, class_counts, encoding_format='A',
                            use_cbs=False, num_workers=0):
    """创建所有 fold 的 dataloader"""
    dataloaders = []

    for fold_idx, (train_df, val_df) in enumerate(folds):
        train_dataset = JobDataset(
            train_df, tokenizer, label_encoder, max_length,
            encoding_format=encoding_format
        )
        val_dataset = JobDataset(
            val_df, tokenizer, label_encoder, max_length,
            encoding_format=encoding_format
        )

        if use_cbs:
            sampler = ClassBalancedSampler(train_df, class_counts, beta=0.999)
            sampler_iter = iter(sampler)
            indices = list(sampler_iter)
            from torch.utils.data import BatchSampler, SequentialSampler
            batch_sampler = BatchSampler(
                indices, batch_size=batch_size, drop_last=False
            )
            train_loader = DataLoader(
                train_dataset, batch_sampler=batch_sampler,
                collate_fn=dynamic_collate_fn, num_workers=num_workers,
            )
        else:
            ib_class_counts = train_df['label'].value_counts().to_dict()
            ib_weights = np.array([
                1.0 / max(ib_class_counts.get(train_df.iloc[i]['label'], 1), 1)
                for i in range(len(train_df))
            ])
            ib_weights = ib_weights / ib_weights.sum()
            from torch.utils.data import WeightedRandomSampler, BatchSampler
            ib_sampler = WeightedRandomSampler(
                ib_weights, num_samples=len(train_df), replacement=True
            )
            batch_sampler = BatchSampler(
                ib_sampler, batch_size=batch_size, drop_last=False
            )
            train_loader = DataLoader(
                train_dataset, batch_sampler=batch_sampler,
                collate_fn=dynamic_collate_fn, num_workers=num_workers,
            )

        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=dynamic_collate_fn, num_workers=num_workers,
        )

        dataloaders.append((train_loader, val_loader, train_df, val_df))

    return dataloaders
