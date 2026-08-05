"""
LLM 合成数据生成 — 冷门岗位数据增强

迁移自 人工智能挑战赛 v3/synthesize_data_v3.py 框架

使用 DeepSeek API 为少样本岗位生成额外的训练数据。
核心策略:
- L1 (≤10条): 生成 40 条/类
- L2 (11-15条): 生成 25 条/类
- L3 (16-20条): 生成 15 条/类

用法:
  python synthesize_data.py --api_key sk-xxxx --data_dir ../zhilian_direct/zhilian_direct

环境变量:
  export DEEPSEEK_API_KEY=sk-xxxx
"""

import argparse
import csv
import json
import os
import re
import ssl
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

_SSL_CONTEXT = ssl.create_default_context()
_SSL_CONTEXT.check_hostname = False
_SSL_CONTEXT.verify_mode = ssl.CERT_NONE

# ═══════════════════════════════════════════════════
# LLM API 配置
# ═══════════════════════════════════════════════════

DEEPSEEK_CONFIG = {
    'api_base': 'https://api.deepseek.com/v1',
    'model': 'deepseek-v4-pro',
    'fallback_model': 'deepseek-v4-flash',
    'env_key': 'DEEPSEEK_API_KEY',
    'max_tokens': 4096,
}

# ═══════════════════════════════════════════════════
# 岗位混淆簇 — 生成负样本
# ═══════════════════════════════════════════════════

JOB_CONFUSION_CLUSTERS = {
    "backend_dev": [
        "Java", "Golang", "Python", "C++", "C语言", "Node.js",
        "全栈工程师",
    ],
    "frontend_dev": [
        "前端开发工程师", "Android", "iOS开发", "鸿蒙开发工程师",
    ],
    "data_ai": [
        "数据分析师", "数据工程师", "数据科学家", "数据开发",
        "数据挖掘", "数据仓库", "数据架构师",
        "机器学习", "深度学习", "大模型算法", "人工智能工程师",
        "NLP", "图像算法", "推荐算法", "搜索算法",
        "算法工程师", "算法研究员", "AIGC算法",
    ],
    "devops_test": [
        "测试开发", "自动化测试", "运维开发工程师",
    ],
    "hardware_iot": [
        "嵌入式软件工程师", "硬件工程师", "电子工程师",
        "FPGA开发", "DSP开发", "单片机", "芯片工程师",
    ],
    "product_mgmt": [
        "产品经理", "AI产品经理", "数据产品经理", "硬件产品经理",
        "项目经理", "软件项目经理",
    ],
}


def get_api_key(config, args_key):
    """获取API密钥"""
    if args_key:
        return args_key
    key = os.environ.get(config['env_key'])
    if key:
        return key
    raise ValueError(
        f"需要API密钥: 设置 {config['env_key']} 环境变量 "
        f"或使用 --api_key"
    )


def call_llm_api(prompt, config, api_key, temperature=0.8):
    """调用 LLM API"""
    import urllib.request
    import urllib.error

    api_base = config['api_base'].rstrip('/')
    model = config['model']
    max_tokens = config['max_tokens']

    payload = {
        'model': model,
        'messages': [
            {'role': 'system',
             'content': '你是一位专业的HR和招聘专家，擅长生成真实的岗位描述和候选人技能。'},
            {'role': 'user', 'content': prompt},
        ],
        'temperature': temperature,
        'max_tokens': max_tokens,
    }

    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        f'{api_base}/chat/completions',
        data=data,
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}',
        },
    )

    max_retries = 5
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, context=_SSL_CONTEXT, timeout=120) as resp:
                result = json.loads(resp.read().decode('utf-8'))
                content = result['choices'][0]['message']['content']
                return content
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8')
            if e.code == 429:
                wait = 2 ** attempt
                print(f'  限流，{wait}s后重试...')
                time.sleep(wait)
            elif e.code >= 500:
                wait = 2 ** attempt
                print(f'  服务器错误 {e.code}，{wait}s后重试...')
                time.sleep(wait)
            else:
                print(f'  API错误 {e.code}: {error_body[:200]}')
                return None
        except Exception as e:
            print(f'  网络错误: {e}')
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None

    return None


def build_synthesis_prompt(job_name, job_descriptions, n_target,
                           confusion_jobs=None, few_shot_samples=None):
    """
    构建岗位合成 Prompt

    迁移自 synthesize_data_v3.py 的单步强制理解法
    """
    desc = job_descriptions.get(job_name, job_name)
    confusion_text = ''
    if confusion_jobs:
        confusion_text = (
            f'注意区分以下易混淆岗位: {", ".join(confusion_jobs)}。'
        )

    few_shot_text = ''
    if few_shot_samples and len(few_shot_samples) > 0:
        few_shot_text = '\n参考示例:\n'
        for i, sample in enumerate(few_shot_samples[:3]):
            ta = sample.get('text_a', sample.get('skill_requirements', ''))
            few_shot_text += f'{i+1}. 技能/职责: {ta[:200]}\n'

    prompt = f"""
你是专业HR和招聘专家。请为以下岗位生成真实的候选人技能/职责描述。

岗位名称: {job_name}
岗位描述: {desc}
{confusion_text}
需生成: {n_target} 条不同的候选人描述

要求:
1. 每条描述列出该岗位候选人应具备的技能、工具、经验
2. 覆盖不同经验级别（初/中/高级）
3. 覆盖不同行业场景（如互联网、金融、制造业等）
4. 使用真实的技术栈和工具名
5. 避免重复和模板化

{few_shot_text}

请按以下格式输出:
UNDERSTANDING: [你对{job_name}岗位的理解]
PATTERNS: [常见技能模式]

然后每行一条:
技能描述<TAB>{job_name}<TAB>1

开始：
"""
    return prompt


def parse_synthesis_response(response, job_name, target_n):
    """
    解析 LLM 响应，提取生成的数据

    返回: list of {text_a, label} dicts
    """
    if not response:
        return []

    generated = []
    lines = response.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('UNDERSTANDING') or line.startswith('PATTERNS'):
            continue
        if line.startswith('#') or line.startswith('//'):
            continue

        # 尝试 TAB 分隔: text_a \t label \t 1
        parts = line.split('\t')
        if len(parts) >= 2:
            text_a = parts[0].strip()
            label = parts[1].strip()

            # 质量过滤
            if len(text_a) < 20:
                continue
            if len(text_a) > 2000:
                continue
            # 跳过只有编号的行
            if re.match(r'^\d+[\.\)、]?\s*$', text_a):
                continue

            # 优先用 job_name 作为标签
            generated.append({
                'text_a': text_a,
                'text_b': f'岗位:{job_name}',
                'label': job_name,
                'is_synthetic': True,
            })
        else:
            # 没有 TAB 的行，如果够长就当做技能描述
            if len(line) > 30 and job_name not in line:
                generated.append({
                    'text_a': line,
                    'text_b': f'岗位:{job_name}',
                    'label': job_name,
                    'is_synthetic': True,
                })

        if len(generated) >= target_n * 3:  # Overshoot，后续过滤
            break

    return generated[:target_n]


def filter_generated(generated, existing_data, labels_set):
    """
    5层启发式过滤

    迁移自 synthesize_data_v3.py
    """
    filtered = []

    existing_texts = set()
    if existing_data is not None and len(existing_data) > 0:
        for t in existing_data['text_a'].astype(str):
            existing_texts.add(t[:100])

    for item in generated:
        text_a = item['text_a']
        label = item['label']

        # L1: 基础有效性
        if not text_a or len(text_a) < 20:
            continue

        # L2: 标签有效性
        if label not in labels_set:
            continue

        # L3: 去重
        key = text_a[:100]
        if key in existing_texts:
            continue
        existing_texts.add(key)

        # L4: 包含合理内容（至少有一些中文字符或技术词）
        has_chinese = bool(re.search(r'[一-鿿]', text_a))
        has_tech = bool(re.search(
            r'[A-Za-z+#.]{2,}', text_a
        ))
        if not has_chinese and not has_tech:
            continue

        filtered.append(item)

    return filtered


def synthesize_data(data_dir, output_dir, api_key=None, mode='all',
                    api_config=None):
    """
    主合成函数

    Args:
        data_dir: zhilian_direct 数据目录
        output_dir: 输出目录
        api_key: API 密钥
        mode: 'all' | 'L1_only'
        api_config: LLM 配置
    """
    if api_config is None:
        api_config = DEEPSEEK_CONFIG

    # 加载现有数据
    print('加载现有数据...')
    all_files = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.csv'):
                all_files.append(os.path.join(root, f))

    job_data = {}
    for file_path in tqdm(all_files, desc='加载'):
        filename = os.path.basename(file_path)
        label = filename[:-4].strip()
        for prefix in ['zhilian_direct_', 'zhilian_direct']:
            if label.startswith(prefix):
                label = label[len(prefix):]
                break

        try:
            df = pd.read_csv(file_path, encoding='utf-8-sig')
            df.columns = [str(c).strip() for c in df.columns]
            job_data[label] = df
        except Exception as e:
            print(f'  {filename} 加载失败: {e}')

    print(f'加载 {len(job_data)} 个岗位类别')

    # 统计每类样本数
    class_counts = {
        name: len(df) for name, df in job_data.items()
    }
    counts = sorted(class_counts.items(), key=lambda x: x[1])

    # 分级
    levels = {
        'L1': {'max': 10, 'generate': 40},
        'L2': {'max': 15, 'generate': 25},
        'L3': {'max': 20, 'generate': 15},
        'L4': {'max': 25, 'generate': 0},
    }

    plan = defaultdict(list)
    for job_name, cnt in counts:
        if cnt <= levels['L1']['max']:
            plan['L1'].append(job_name)
        elif cnt <= levels['L2']['max']:
            plan['L2'].append(job_name)
        elif cnt <= levels['L3']['max']:
            plan['L3'].append(job_name)
        else:
            plan['L4'].append(job_name)

    print(f'\n合成计划:')
    print(f'  L1 (≤{levels["L1"]["max"]}条): {len(plan["L1"])} 类, '
          f'每类生成 {levels["L1"]["generate"]} 条')
    print(f'  L2 (≤{levels["L2"]["max"]}条): {len(plan["L2"])} 类, '
          f'每类生成 {levels["L2"]["generate"]} 条')
    print(f'  L3 (≤{levels["L3"]["max"]}条): {len(plan["L3"])} 类, '
          f'每类生成 {levels["L3"]["generate"]} 条')
    print(f'  L4 (>25条): {len(plan["L4"])} 类, 不合成')

    if mode == 'L1_only':
        target_levels = ['L1']
    else:
        target_levels = ['L1', 'L2', 'L3']

    # 构建岗位描述
    job_descriptions = {
        name: f'{name}岗位，典型技能包括相关技术栈和工具'  # placeholder
        for name in class_counts
    }

    # 为每个岗位分配混淆簇
    confusion_map = {}
    for job_name in class_counts:
        for cluster_name, jobs in JOB_CONFUSION_CLUSTERS.items():
            if job_name in jobs:
                confusion_map[job_name] = [j for j in jobs if j != job_name]
                break
        if job_name not in confusion_map:
            confusion_map[job_name] = []

    all_labels = set(class_counts.keys())

    # 开始合成
    api_key = get_api_key(api_config, api_key)
    all_generated = []

    for level in target_levels:
        if level not in plan or not plan[level]:
            continue

        n_gen = levels[level]['generate']
        if n_gen <= 0:
            continue

        jobs = plan[level]
        print(f'\n{"="*60}')
        print(f'  {level}: 合成 {len(jobs)} 类, 每类 {n_gen} 条')
        print(f'{"="*60}')

        for job_name in tqdm(jobs, desc=f'合成 {level}'):
            existing_df = job_data.get(job_name)

            # 构建 few-shot
            few_shot = []
            if existing_df is not None and len(existing_df) > 0:
                for _, row in existing_df.head(3).iterrows():
                    skills = str(row.get('skill_requirements', ''))
                    if skills and skills != 'nan':
                        few_shot.append({'text_a': skills})

            confusion = confusion_map.get(job_name, [])[:3]

            prompt = build_synthesis_prompt(
                job_name, job_descriptions, n_gen * 2,  # overshoot
                confusion_jobs=confusion,
                few_shot_samples=few_shot,
            )

            response = call_llm_api(prompt, api_config, api_key)
            if response is None:
                print(f'  {job_name}: API调用失败，跳过')
                continue

            generated = parse_synthesis_response(
                response, job_name, n_gen * 2
            )
            filtered = filter_generated(
                generated, existing_df, all_labels
            )

            all_generated.extend(filtered[:n_gen])
            print(f'  {job_name}: 生成 {len(filtered[:n_gen])}/{n_gen} 条')
            time.sleep(0.5)  # 避免限流

    # 保存
    if all_generated:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, 'synthetic_data.csv')

        out_df = pd.DataFrame(all_generated)
        out_df.to_csv(out_path, index=False, encoding='utf-8-sig')
        print(f'\n合成完成: {len(all_generated)} 条')
        print(f'保存至: {out_path}')
    else:
        print('\n无数据生成')


def main():
    parser = argparse.ArgumentParser(
        description='LLM合成数据 — 冷门岗位增强'
    )
    parser.add_argument('--data_dir', type=str,
                        default='../zhilian_direct/zhilian_direct')
    parser.add_argument('--output_dir', type=str, default='../outputs')
    parser.add_argument('--api_key', type=str, default=None)
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'L1_only'])
    args = parser.parse_args()

    synthesize_data(
        args.data_dir, args.output_dir, args.api_key, args.mode,
    )


if __name__ == '__main__':
    main()
