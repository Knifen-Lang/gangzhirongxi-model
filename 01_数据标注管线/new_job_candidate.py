#!/usr/bin/env python3
"""
new_job_candidate.py — 新岗位候选判定

基于两条标准判定一个 JD 是否属于"新岗位候选"：
  标准一：AI 是主业而非辅助（AI术语 ≥ 3个 + AI职责占比 > 50%）
  标准二：存在真实业务招聘需求（同类岗位 ≥ 5个 + 分布在 ≥ 3家公司 + 时间跨度 ≥ 6个月）
  排除规则：岗位名称包含传统底座关键词（鸿蒙/Android/SLAM等）→ 直接判定为"否"

Usage:
    python new_job_candidate.py --jd_id JD_001 --input debug_20.csv --output candidate_report.json
    python new_job_candidate.py --input debug_20.csv --output all_candidates.json
"""

import os, sys, json, argparse, csv, re
from collections import defaultdict, Counter
from datetime import datetime, timedelta


# ============================================================
#  配置
# ============================================================

# AI 相关术语词表（用于信号提取）
AI_TERMS = [
    'LLM', '大语言模型', '大模型', 'AIGC', 'RAG', 'Agent', '智能体',
    '多模态', 'Prompt', '提示词', '提示工程', 'LangChain', 'LlamaIndex',
    '深度学习', '机器学习', '神经网络', 'Transformer', 'GPT', 'ChatGPT',
    'Stable Diffusion', '扩散模型', 'GAN', '生成式AI', '生成式人工智能',
    '自然语言处理', 'NLP', '计算机视觉', 'CV', '语音识别', '语音合成',
    '强化学习', 'RLHF', 'SFT', '微调', 'Fine-tuning', '预训练',
    'ComfyUI', 'LoRA', 'Embedding', '向量数据库', '知识图谱',
    'Copilot', 'AI编程', 'AI Agent', 'AI应用',
]

# 传统底座关键词（排他规则）
EXCLUDE_KEYWORDS = [
    '鸿蒙', 'Android', '安卓', 'iOS', 'SLAM', '嵌入式', '单片机',
    'FPGA', '芯片', '驱动开发', 'DSP', '射频', '天线',
    '普通后端', '传统开发',
]


# ============================================================
#  信号提取
# ============================================================

def extract_ai_signals(jd_text):
    """从 JD 文本中提取 AI 相关信号"""
    if not jd_text:
        return {'terms': [], 'count': 0, 'ratio': 0.0}

    jd_lower = jd_text.lower()
    found_terms = []
    for term in AI_TERMS:
        if term.lower() in jd_lower or term in jd_text:
            found_terms.append(term)

    # 估算 AI 相关职责占比（基于 AI 术语出现的句子数 / 总句子数）
    sentences = re.split(r'[。；;.\n]', jd_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    if not sentences:
        return {'terms': found_terms, 'count': len(found_terms), 'ratio': 0.0}

    ai_sentences = 0
    for sent in sentences:
        for term in AI_TERMS:
            if term.lower() in sent.lower():
                ai_sentences += 1
                break

    ratio = ai_sentences / len(sentences) if sentences else 0.0

    return {
        'terms': found_terms,
        'count': len(found_terms),
        'ratio': round(ratio, 3),
        'total_sentences': len(sentences),
        'ai_sentences': ai_sentences,
    }


def check_exclusion(job_name):
    """检查是否命中排他规则"""
    if not job_name:
        return {'excluded': False, 'keywords': []}
    found = [kw for kw in EXCLUDE_KEYWORDS if kw in job_name]
    return {'excluded': len(found) > 0, 'keywords': found}


# ============================================================
#  标准判断
# ============================================================

def check_criterion_1(signals, jd_text, job_name):
    """
    标准一：AI 是主业而非辅助
    条件：
      - AI 术语 ≥ 3 个
      - AI 相关职责占比 > 50%（或 AI 术语 ≥ 5 个则放宽到 30%）
      - 岗位名称不包含传统底座关键词
    """
    exclusion = check_exclusion(job_name)
    if exclusion['excluded']:
        return {
            'passed': False,
            'reason': f'命中排他关键词: {exclusion["keywords"]}',
            'ai_terms_found': signals['terms'],
            'ai_term_count': signals['count'],
            'ai_sentence_ratio': signals['ratio'],
            'exclusion': exclusion,
        }

    term_count = signals['count']
    ratio = signals['ratio']

    # 核心判断
    if term_count >= 5:
        passed_core = ratio >= 0.30  # 术语多则放宽
    elif term_count >= 3:
        passed_core = ratio >= 0.50
    else:
        passed_core = False

    return {
        'passed': passed_core,
        'reason': f'AI术语{term_count}个，AI占比{ratio:.0%}' if passed_core else
                  f'不满足：AI术语{term_count}个（需≥3），AI占比{ratio:.0%}（需>50%）',
        'ai_terms_found': signals['terms'],
        'ai_term_count': term_count,
        'ai_sentence_ratio': ratio,
        'exclusion': exclusion,
    }


def check_criterion_2(jd_row, all_rows):
    """
    标准二：存在真实业务招聘需求
    条件：
      - 同类岗位（技能相似度 > 0.5）≥ 5 个
      - 分布在 ≥ 3 家公司
      - 时间跨度 ≥ 6 个月
    """
    ref_skills = set()
    for col in ['essential_skills', 'preferred_skills', 'tech_tags']:
        val = jd_row.get(col, '')
        ref_skills |= set(s.strip() for s in val.replace(';', ',').split(',') if s.strip())

    if not ref_skills:
        return {'passed': False, 'reason': '无参考技能可比较'}

    # 查找同类岗位
    similar_jobs = []
    companies = set()
    dates = []

    for row in all_rows:
        if row is jd_row:
            continue
        row_skills = set()
        for col in ['essential_skills', 'preferred_skills', 'tech_tags']:
            val = row.get(col, '')
            row_skills |= set(s.strip() for s in val.replace(';', ',').split(',') if s.strip())

        if not row_skills:
            continue

        # Jaccard 相似度
        intersection = len(ref_skills & row_skills)
        union = len(ref_skills | row_skills)
        similarity = intersection / union if union > 0 else 0

        if similarity > 0.5:
            similar_jobs.append(row)
            company = row.get('company_name', '').strip()
            if company:
                companies.add(company)
            date_str = row.get('issue_date', '')
            if date_str:
                dates.append(date_str)

    # 时间跨度计算
    time_span_str = '未知'
    time_span_months = 0
    if dates:
        parsed_dates = []
        for d in dates:
            for fmt in ['%Y-%m-%d', '%Y/%m/%d']:
                try:
                    parsed_dates.append(datetime.strptime(d.strip(), fmt))
                    break
                except ValueError:
                    continue
        if len(parsed_dates) >= 2:
            min_date = min(parsed_dates)
            max_date = max(parsed_dates)
            time_span_months = (max_date.year - min_date.year) * 12 + (max_date.month - min_date.month)
            time_span_str = f'{time_span_months}个月'

    passed = (
        len(similar_jobs) >= 5 and
        len(companies) >= 3 and
        time_span_months >= 6
    )

    return {
        'passed': passed,
        'reason': f'同类岗位{len(similar_jobs)}个，{len(companies)}家公司，时间跨度{time_span_str}' if passed else
                  f'不满足：同类岗位{len(similar_jobs)}个（需≥5），{len(companies)}家公司（需≥3），时间跨度{time_span_str}（需≥6个月）',
        'similar_jobs_count': len(similar_jobs),
        'companies': list(companies)[:10],
        'companies_count': len(companies),
        'time_span': time_span_str,
        'time_span_months': time_span_months,
    }


# ============================================================
#  主判定函数
# ============================================================

def judge_job_candidate(jd_row, all_rows):
    """综合判定一个 JD 是否为新岗位候选"""
    jd_id = jd_row.get('job_url', '')[-20:] if jd_row.get('job_url') else 'unknown'
    job_name = jd_row.get('job_name', '未知岗位')
    jd_text = jd_row.get('skill_requirements', '')
    if not jd_text:
        jd_text = jd_row.get('tech_tags', '')

    # 提取 AI 信号
    signals = extract_ai_signals(jd_text)

    # 标准一
    c1 = check_criterion_1(signals, jd_text, job_name)

    # 标准二
    c2 = check_criterion_2(jd_row, all_rows)

    # 综合判定
    is_candidate = c1['passed'] and c2['passed']

    # 置信度
    if is_candidate:
        confidence = min(0.95, 0.5 + 0.15 * signals['count'] + 0.1 * (c2['similar_jobs_count'] / 20))
    else:
        confidence = max(0.1, 0.3 + 0.05 * signals['count'])

    return {
        'jd_id': jd_id,
        'job_name': job_name,
        'company_name': jd_row.get('company_name', ''),
        'is_new_job_candidate': is_candidate,
        'confidence': round(confidence, 2),
        'candidate_job_name': job_name if is_candidate else None,
        'reasons': {
            'criterion_1_ai_core': c1,
            'criterion_2_real_demand': c2,
        },
        'evidence': [
            f'AI术语: {", ".join(signals["terms"][:8])}' + (f'等共{signals["count"]}个' if signals['count'] > 8 else ''),
            f'同类岗位数: {c2.get("similar_jobs_count", 0)}',
            f'覆盖公司数: {c2.get("companies_count", 0)}',
            f'时间跨度: {c2.get("time_span", "未知")}',
        ],
    }


def batch_judge(rows):
    """批量判定"""
    results = []
    for row in rows:
        result = judge_job_candidate(row, rows)
        results.append(result)
    return results


# ============================================================
#  主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='新岗位候选判定')
    parser.add_argument('--input', type=str, required=True, help='输入 CSV 文件')
    parser.add_argument('--jd_id', type=str, default='', help='指定 JD 标识（job_url 尾部片段）')
    parser.add_argument('--output', type=str, default='candidate_report.json', help='输出 JSON 路径')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f'[ERROR] 输入文件不存在: {args.input}')
        sys.exit(1)

    # 加载数据
    with open(args.input, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f'[INFO] 加载 {len(rows)} 条 JD 记录')

    if args.jd_id:
        # 单条判定
        target = None
        for row in rows:
            url = row.get('job_url', '')
            if args.jd_id in url:
                target = row
                break
        if not target:
            print(f'[ERROR] 未找到 jd_id={args.jd_id}')
            sys.exit(1)

        result = judge_job_candidate(target, rows)
        results = [result]
    else:
        # 批量判定
        results = batch_judge(rows)

    # 输出
    candidates = [r for r in results if r['is_new_job_candidate']]
    output = {
        'generated_at': datetime.now().isoformat(),
        'total_jds': len(results),
        'candidates_found': len(candidates),
        'candidates': candidates,
        'all_results': results,
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f'\n[INFO] 判定完成: {len(results)} 条 JD, 发现 {len(candidates)} 个新岗位候选')
    print(f'[INFO] 报告保存至: {args.output}')

    # 打印候选摘要
    for c in candidates:
        print(f'\n✅ {c["job_name"]} ({c["confidence"]:.0%})')
        print(f'   AI术语: {c["reasons"]["criterion_1_ai_core"]["ai_terms_found"][:5]}')


if __name__ == '__main__':
    main()
