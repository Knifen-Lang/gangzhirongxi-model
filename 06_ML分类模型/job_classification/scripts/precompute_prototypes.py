"""
岗位描述原型矩阵预计算

迁移自 人工智能挑战赛 v4训练/v3&v4/precompute_prototypes_v3.py

对每个岗位用 DeBERTa/BGE 编码其描述文本 → 原型向量。
原型矩阵用于:
1. 分类头权重初始化（提供语义先验）
2. PrototypeClassifier（Stage 0 描述匹配预训练）
3. 推理时辅助分类

用法:
  python precompute_prototypes.py --model_name microsoft/deberta-v3-base \
    --data_dir ../zhilian_direct/zhilian_direct --output_dir ../outputs
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


def get_job_names_from_dir(data_dir):
    """从 zhilian_direct CSV 目录获取岗位名列表"""
    job_names = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.csv'):
                filename = os.path.basename(f)
                label = filename[:-4].strip()
                for prefix in ['zhilian_direct_', 'zhilian_direct']:
                    if label.startswith(prefix):
                        label = label[len(prefix):]
                        break
                job_names.append(label)
    job_names = sorted(set(job_names))
    print(f"找到 {len(job_names)} 个岗位类别")
    return job_names


def build_job_descriptions(data_dir, job_names):
    """
    为每个岗位构建描述文本

    策略：从该岗位的所有样本中提取常见关键词，构建原型描述
    """
    descriptions = {}

    for job_name in tqdm(job_names, desc="构建岗位描述"):
        # 在 data_dir 中查找对应 CSV
        found = False
        for root, dirs, files in os.walk(data_dir):
            for f in files:
                if f.endswith('.csv'):
                    label = os.path.basename(f)[:-4].strip()
                    for prefix in ['zhilian_direct_', 'zhilian_direct']:
                        if label.startswith(prefix):
                            label = label[len(prefix):]
                            break
                    if label == job_name:
                        file_path = os.path.join(root, f)
                        try:
                            df = pd.read_csv(file_path, encoding='utf-8-sig')
                            df.columns = [str(c).strip() for c in df.columns]

                            # 收集 skill_requirements 和 tech_tags
                            skills_all = []
                            for _, row in df.iterrows():
                                skills = str(row.get('skill_requirements', ''))
                                if skills and skills != 'nan':
                                    skills_all.append(skills)
                                tech = str(row.get('tech_tags', ''))
                                if tech and tech != 'nan':
                                    skills_all.append(tech)

                            # 取前 3 个有代表性的描述拼接
                            if skills_all:
                                desc_samples = skills_all[:3]
                                desc = ' | '.join(desc_samples)[:500]
                            else:
                                desc = f"{job_name}岗位"

                            descriptions[job_name] = desc
                            found = True
                        except Exception as e:
                            print(f"  警告: {f} 读取失败: {e}")
                        break
                if found:
                    break
            if found:
                break

        if not found:
            descriptions[job_name] = f"{job_name}相关岗位"

    return descriptions


def precompute_prototypes(model_name, data_dir, output_dir, descriptions_path=None):
    """预计算岗位原型矩阵"""
    job_names = get_job_names_from_dir(data_dir)
    num_labels = len(job_names)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    # 推断 hidden_size
    dummy = tokenizer("test", max_length=10, padding='max_length',
                      truncation=True, return_tensors='pt')
    dummy = {k: v.to(device) for k, v in dummy.items()}
    with torch.no_grad():
        out = model(**dummy)
    hidden_size = out.last_hidden_state.shape[-1]
    print(f"Hidden size: {hidden_size}")

    # 加载或构建岗位描述
    if descriptions_path and os.path.exists(descriptions_path):
        with open(descriptions_path, 'r', encoding='utf-8') as f:
            descriptions = json.load(f)
        print(f"加载 {len(descriptions)} 条岗位描述")
    else:
        descriptions = build_job_descriptions(data_dir, job_names)

    # 编码原型
    prototype_matrix = np.zeros((num_labels, hidden_size), dtype='float32')

    for i, job_name in enumerate(tqdm(job_names, desc="计算原型")):
        desc = descriptions.get(job_name, job_name)
        text = f"岗位: {job_name}。技能要求: {desc}"

        enc = tokenizer(text, max_length=256, padding='max_length',
                        truncation=True, return_tensors='pt')
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = model(**enc)
        cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        prototype_matrix[i] = cls_emb

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    npy_path = os.path.join(output_dir, 'prototype_matrix.npy')
    names_path = os.path.join(output_dir, 'prototype_matrix_names.json')
    desc_path = os.path.join(output_dir, 'job_descriptions.json')

    np.save(npy_path, prototype_matrix)
    with open(names_path, 'w', encoding='utf-8') as f:
        json.dump(job_names, f, ensure_ascii=False)
    with open(desc_path, 'w', encoding='utf-8') as f:
        json.dump(descriptions, f, ensure_ascii=False, indent=2)

    print(f"\n保存完成:")
    print(f"  prototype_matrix.npy: {prototype_matrix.shape}")
    print(f"  prototype_matrix_names.json: {len(job_names)} 个岗位名")


def main():
    parser = argparse.ArgumentParser(description='岗位原型矩阵预计算')
    parser.add_argument('--model_name', type=str,
                        default='microsoft/deberta-v3-base')
    parser.add_argument('--data_dir', type=str,
                        default='../zhilian_direct/zhilian_direct')
    parser.add_argument('--descriptions_path', type=str, default=None,
                        help='可选，手动编写的岗位描述JSON')
    parser.add_argument('--output_dir', type=str, default='../outputs')
    args = parser.parse_args()

    precompute_prototypes(
        args.model_name, args.data_dir,
        args.output_dir, args.descriptions_path
    )


if __name__ == '__main__':
    main()
