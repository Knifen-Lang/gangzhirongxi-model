#!/usr/bin/env python3
"""
change_detector_v2.py — 岗位技能时序变化检测（三层过滤版）

复用 annotate_v3.py 的技能字典和规则引擎，在已标注数据基础上增加：
  第1层：统计显著性过滤（Fisher精确检验）  → 排除采样噪声
  第2层：抄袭簇检测 + 通胀校准            → 排除技能通胀
  第3层：知识库校验（过时技能/基础技能）    → 排除已知旧技能误标

输入：已标注 CSV 目录（87个文件）
输出：可信技能变化报告

Usage:
    python change_detector_v2.py \
      --annotated_dir "./【已标注v3】/【已标注" \
      --output change_report_v2.json
"""

import os, sys, json, argparse, csv, re, math, glob
from collections import defaultdict, Counter
from datetime import datetime
from difflib import SequenceMatcher

# ── 复用 annotate_v3 的字典 ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '【已标注v3】'))
from annotate_v3 import (
    AI_SKILLS, TRADITIONAL_SKILLS, SOFT_SKILLS, OUTDATED_SKILLS,
    CORE_AI_SIGNALS, TRADITIONAL_TECH_BASE,
)


# ============================================================
#  工具函数
# ============================================================

def parse_month(date_str):
    m = re.search(r'(\d+)月', str(date_str))
    return int(m.group(1)) if m else 0


def quarter_label(month):
    if month == 0: return 'Unknown'
    return f'2026Q{(month-1)//3+1}'


def parse_skills(skill_str):
    """从标注格式解析技能列表: 【技能名｜分类｜级别】"""
    if not skill_str or not skill_str.strip():
        return []
    skills = []
    for part in skill_str.split('；'):
        part = part.strip()
        m = re.match(r'【(.+?)｜', part)
        if m:
            skills.append(m.group(1))
        elif part and part != '（JD文本过短）':
            skills.append(part)
    return skills


def text_similarity(a, b):
    """计算两段文本的 Jaccard 相似度"""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


# ============================================================
#  第1层：统计显著性（Fisher精确检验）
# ============================================================

def fisher_exact_test(a, b, c, d):
    """
    Fisher精确检验（2x2列联表）
      窗口A: 出现 a 次 / 未出现 b 次  (total_a = a+b)
      窗口B: 出现 c 次 / 未出现 d 次  (total_b = c+d)
    H0: 两个窗口中技能出现的概率相同
    返回 p-value
    """
    if a + b + c + d == 0:
        return 1.0

    # 使用 scipy 的近似（如果不可用则用简单超几何近似）
    n = a + b + c + d
    k = a + c  # 总出现次数
    N_minus_K = b + d

    # 超几何分布: P(X >= a) = sum_{i=a}^{min(k, a+b)} C(k,i) * C(N-k, a+b-i) / C(N, a+b)
    def log_comb(n, k):
        if k < 0 or k > n: return -float('inf')
        return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)

    log_total = log_comb(n, a + b)
    p_value = 0.0

    for i in range(a, min(k, a + b) + 1):
        log_num = log_comb(k, i) + log_comb(n - k, a + b - i)
        p_value += math.exp(log_num - log_total)

    return min(p_value, 1.0)


def compute_effect_size(a, b, c, d):
    """计算 Cohen's h（效应量）"""
    total_a = a + b
    total_b = c + d
    if total_a == 0 or total_b == 0:
        return 0.0
    p1 = a / total_a
    p2 = c / total_b
    h = 2 * abs(math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))
    return h


def statistical_filter(skill, window_a_rows, window_b_rows, alpha=0.05):
    """第1层：统计显著性过滤"""
    # 计算2x2列联表
    a = sum(1 for r in window_b_rows if skill in parse_skills(r.get('必备技能', '')) or
            skill in parse_skills(r.get('加分技能', '')))
    b = len(window_b_rows) - a
    c = sum(1 for r in window_a_rows if skill in parse_skills(r.get('必备技能', '')) or
            skill in parse_skills(r.get('加分技能', '')))
    d = len(window_a_rows) - c

    p_value = fisher_exact_test(a, b, c, d)
    effect_size = compute_effect_size(a, b, c, d)

    # 效应量解读
    if effect_size < 0.2:
        effect_label = 'small'
    elif effect_size < 0.5:
        effect_label = 'medium'
    else:
        effect_label = 'large'

    return {
        'p_value': round(p_value, 4),
        'effect_size': round(effect_size, 3),
        'effect_label': effect_label,
        'significant': p_value < alpha and effect_size >= 0.2,
        'contingency_table': {'window_a_present': c, 'window_a_absent': d,
                              'window_b_present': a, 'window_b_absent': b},
    }


# ============================================================
#  第2层：抄袭簇检测 + 通胀校准
# ============================================================

def detect_plagiarism_clusters(rows, threshold=0.8):
    """检测抄袭簇：返回每行的通胀系数"""
    jd_texts = [r.get('skill_requirements', '') for r in rows]
    n = len(rows)
    visited = [False] * n
    clusters = []

    for i in range(n):
        if visited[i]: continue
        cluster = [i]
        for j in range(i + 1, n):
            if visited[j]: continue
            if text_similarity(jd_texts[i], jd_texts[j]) > threshold:
                cluster.append(j)
                visited[j] = True
        clusters.append(cluster)

    # 计算通胀系数
    inflation_factors = [1.0] * n
    for cluster in clusters:
        factor = len(cluster)
        for idx in cluster:
            inflation_factors[idx] = factor

    return inflation_factors


def inflation_aware_count(skill, rows, inflation_factors):
    """通胀校准后的技能频次"""
    raw_count = 0
    effective_count = 0.0
    for i, r in enumerate(rows):
        all_skills = set()
        all_skills.update(parse_skills(r.get('必备技能', '')))
        all_skills.update(parse_skills(r.get('加分技能', '')))
        if skill in all_skills:
            raw_count += 1
            effective_count += 1.0 / inflation_factors[i]

    return {
        'raw_count': raw_count,
        'effective_count': round(effective_count, 2),
        'is_inflated': effective_count < raw_count * 0.5 and raw_count >= 3,
    }


# ============================================================
#  第3层：知识库校验
# ============================================================

def knowledge_check(skill):
    """第3层：知识库校验"""
    checks = {}

    # 检查是否为过时技能
    for outdated_skill, reason in OUTDATED_SKILLS:
        if skill == outdated_skill:
            checks['obsolete'] = {'is_obsolete': True, 'reason': reason}
            break
    else:
        checks['obsolete'] = {'is_obsolete': False}

    # 检查是否为软技能
    checks['is_soft_skill'] = skill in SOFT_SKILLS

    # 检查技能类型
    if skill in AI_SKILLS:
        checks['skill_type'] = 'AI新兴技能'
    elif skill in TRADITIONAL_SKILLS:
        checks['skill_type'] = '传统技术'
    else:
        checks['skill_type'] = '未分类'

    # 基础技能判定（传统技术 + 高频 → 基础）
    checks['is_foundational'] = (checks['skill_type'] == '传统技术')

    return checks


# ============================================================
#  主检测逻辑
# ============================================================

def load_all_annotated(annotated_dir):
    """加载所有已标注CSV，按岗位名称分组"""
    files = glob.glob(os.path.join(annotated_dir, '*.csv'))
    print(f'[INFO] 加载 {len(files)} 个标注文件...')

    by_job = defaultdict(list)
    for fpath in files:
        with open(fpath, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                job = row.get('job_name', '未知岗位').strip()[:40]
                by_job[job].append(row)

    total = sum(len(v) for v in by_job.values())
    print(f'[INFO] {total} 条JD, {len(by_job)} 个去重岗位')
    return by_job


def detect_changes_for_job(job_name, rows, min_jd=3):
    """对单个岗位执行三层过滤检测"""
    # 按季度分组
    by_window = defaultdict(list)
    for r in rows:
        month = parse_month(r.get('issue_date', ''))
        w = quarter_label(month)
        by_window[w].append(r)

    sorted_windows = sorted(by_window.keys())
    valid_windows = {w: rs for w, rs in by_window.items() if len(rs) >= min_jd}

    if len(valid_windows) < 2:
        return None

    sorted_windows = sorted(valid_windows.keys())

    # 构建窗口摘要
    window_summaries = []
    window_skillsets = {}
    for w in sorted_windows:
        rows_w = valid_windows[w]
        # 抄袭簇检测
        inf_factors = detect_plagiarism_clusters(rows_w)

        essential = Counter()
        preferred = Counter()
        for i, r in enumerate(rows_w):
            wf = 1.0 / inf_factors[i]
            for s in parse_skills(r.get('必备技能', '')):
                essential[s] += wf
            for s in parse_skills(r.get('加分技能', '')):
                preferred[s] += wf

        window_skillsets[w] = {
            'rows': rows_w,
            'inflation_factors': inf_factors,
            'essential': essential,
            'preferred': preferred,
        }

        window_summaries.append({
            'window': w,
            'jd_count': len(rows_w),
            'skills': {
                'essential': sorted([s for s, c in essential.most_common(10)], key=lambda s: -essential[s]),
                'preferred': sorted([s for s, c in preferred.most_common(5)], key=lambda s: -preferred[s]),
            },
            'warning': None if len(rows_w) >= 3 else f'数据不足（仅{len(rows_w)}条JD）',
        })

    # 三层过滤检测变化
    changes = []
    warnings = []

    for i in range(1, len(sorted_windows)):
        prev_w = sorted_windows[i - 1]
        curr_w = sorted_windows[i]
        prev = window_skillsets[prev_w]
        curr = window_skillsets[curr_w]

        prev_all_skills = set(prev['essential'].keys()) | set(prev['preferred'].keys())
        curr_all_skills = set(curr['essential'].keys()) | set(curr['preferred'].keys())

        # 候选技能变化
        all_candidate_skills = prev_all_skills | curr_all_skills

        for skill in all_candidate_skills:
            in_prev = skill in prev_all_skills
            in_curr = skill in curr_all_skills

            if in_prev == in_curr:
                # 两个窗口都有 → 检查级别变化
                was_ess = skill in prev['essential']
                is_ess = skill in curr['essential']
                if was_ess == is_ess:
                    continue  # 无变化
                change_type = '升级（加分→必备）' if is_ess else '降级（必备→加分）'
            elif in_curr and not in_prev:
                change_type = '新增'
            else:
                change_type = '删除'

            # ── 第1层：统计过滤 ──
            stat_result = statistical_filter(skill, prev['rows'], curr['rows'])
            if change_type in ('新增', '删除') and not stat_result['significant']:
                continue  # 不显著，跳过

            # ── 第2层：通胀检测 ──
            if change_type == '新增' and in_curr:
                inf_result = inflation_aware_count(skill, curr['rows'], curr['inflation_factors'])
            elif change_type == '删除' and in_prev:
                inf_result = inflation_aware_count(skill, prev['rows'], prev['inflation_factors'])
            else:
                inf_result = {'raw_count': 0, 'effective_count': 0, 'is_inflated': False}

            if inf_result.get('is_inflated'):
                change_type += '（通胀嫌疑）'
                warnings.append(f'技能"{skill}"在{curr_w}中疑似通胀（raw={inf_result["raw_count"]}, effective={inf_result["effective_count"]}）')

            # ── 第3层：知识库校验 ──
            kc = knowledge_check(skill)
            if kc['obsolete']['is_obsolete']:
                change_type += '（过时技能）'
            if kc['is_foundational'] and change_type.startswith('新增'):
                change_type += '（基础技能，疑似采样偏差）'
            if kc['is_soft_skill']:
                continue  # 软技能，跳过

            # ── 找证据 ──
            evidence = []
            target_rows = curr['rows'] if in_curr else prev['rows']
            for r in target_rows[:3]:
                text = r.get('skill_requirements', '')
                if skill in text:
                    idx = text.find(skill)
                    evidence.append(text[max(0, idx-30):min(len(text), idx+len(skill)+30)].strip())

            # ── 置信度 ──
            if stat_result['significant'] and stat_result['effect_label'] == 'large' and not inf_result.get('is_inflated') and not kc['obsolete']['is_obsolete']:
                confidence = '高'
            elif stat_result['significant'] and not inf_result.get('is_inflated'):
                confidence = '中'
            else:
                confidence = '低'

            changes.append({
                'skill': skill,
                'type': change_type,
                'confidence': confidence,
                'from_window': prev_w if in_prev else None,
                'to_window': curr_w if in_curr else None,
                'statistical_test': stat_result,
                'inflation_check': inf_result,
                'knowledge_check': kc,
                'evidence': evidence,
                'recommendation': (
                    '确认为真实变化，建议更新岗位画像' if confidence == '高'
                    else '建议人工审核确认' if confidence == '中'
                    else '疑似误报，需人工核实'
                ),
            })

    return {
        'job': job_name,
        'total_jds': len(rows),
        'windows': window_summaries,
        'changes': sorted(changes, key=lambda c: (
            0 if c['confidence'] == '高' else 1 if c['confidence'] == '中' else 2,
            0 if c['type'].startswith('新增') else 1
        )),
        'warnings': warnings,
    }


# ============================================================
#  主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='岗位技能变化检测 v2（三层过滤）')
    parser.add_argument('--annotated_dir', type=str, required=True, help='已标注CSV目录')
    parser.add_argument('--job', type=str, default='', help='指定岗位（默认分析所有）')
    parser.add_argument('--output', type=str, default='change_report_v2.json')
    parser.add_argument('--min_jd', type=int, default=2, help='每窗口最小JD数')
    parser.add_argument('--top_n', type=int, default=10, help='分析前N个最大岗位')
    args = parser.parse_args()

    by_job = load_all_annotated(args.annotated_dir)

    if args.job:
        if args.job in by_job:
            targets = {args.job: by_job[args.job]}
        else:
            matched = {k: v for k, v in by_job.items() if args.job in k}
            targets = matched if matched else {}
            if not targets:
                print(f'[WARN] 未找到岗位: {args.job}')
                sys.exit(1)
    else:
        # 选 JD 数量最多的 top_n 岗位
        top_jobs = sorted(by_job.items(), key=lambda x: -len(x[1]))[:min(args.top_n, len(by_job))]
        targets = dict(top_jobs)

    reports = []
    for job, rows in targets.items():
        print(f'[INFO] 分析: {job} ({len(rows)}条JD)...')
        report = detect_changes_for_job(job, rows, args.min_jd)
        if report and report['changes']:
            reports.append(report)
            print(f'  {len(report["windows"])}个窗口, {len(report["changes"])}个变化')

    # 输出
    # 将 ndarray 转换为 list 以便 JSON 序列化
    def convert(obj):
        if isinstance(obj, (set,)):
            return list(obj)
        if isinstance(obj, (float,)):
            if math.isnan(obj) or math.isinf(obj):
                return None
        return obj

    output = {
        'generated_at': datetime.now().isoformat(),
        'total_jobs_analyzed': len(reports),
        'total_changes': sum(len(r['changes']) for r in reports),
        'reports': reports,
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=convert)

    print(f'\n[INFO] 报告保存至: {args.output}')
    print(f'  岗位数: {len(reports)}')
    print(f'  总变化数: {output["total_changes"]}')
    print(f'  高置信度: {sum(1 for r in reports for c in r["changes"] if c["confidence"]=="高")}')
    print(f'  中置信度: {sum(1 for r in reports for c in r["changes"] if c["confidence"]=="中")}')
    print(f'  低置信度: {sum(1 for r in reports for c in r["changes"] if c["confidence"]=="低")}')


if __name__ == '__main__':
    main()
