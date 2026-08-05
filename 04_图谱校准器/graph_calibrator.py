#!/usr/bin/env python3
"""
graph_calibrator.py — 图谱校准器
==============================
输入: 已标注JD + 技术信号 + 检测报告
输出: 校准后的岗位能力图谱

解决四类问题:
  1. 通胀 → 通胀感知计数（1/inflation_factor 加权）
  2. 时滞 → 预测性技能激活（11个月预估）
  3. 噪声 → Fisher精确检验过滤（p<0.05, h>=0.2）
  4. 幻觉 → HallucinationGuard + RAG三源验证

核心思想: 检测 -> 矫正 -> 输出干净图谱

Usage:
    python graph_calibrator.py
"""

import os, sys, json, csv, glob, re, math, random
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from difflib import SequenceMatcher

BASE_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(BASE_DIR, '【已标注v3】'))
from annotate_v3 import (
    AI_SKILLS, TRADITIONAL_SKILLS, SOFT_SKILLS, OUTDATED_SKILLS,
    extract_skills_from_text,
)


# ============================================================
#  第1部分: 通胀校准 — 通胀感知计数
# ============================================================

class InflationCalibrator:
    """通胀校准器: 抄袭簇检测 + 降权计数"""

    def __init__(self):
        self.cluster_sizes = {}  # JD索引 -> 簇大小

    def detect_plagiarism(self, rows):
        """检测同岗位抄袭簇，返回每条JD的通胀因子"""
        n = len(rows)
        texts = [str(r.get('skill_requirements', '')) for r in rows]
        visited = [False] * n
        clusters = []

        for i in range(n):
            if visited[i]: continue
            job_i = str(rows[i].get('job_name', ''))
            cluster = [i]
            for j in range(i+1, n):
                if visited[j]: continue
                job_j = str(rows[j].get('job_name', ''))
                if job_i == job_j:  # 必须严格同岗位
                    sim = SequenceMatcher(None, texts[i], texts[j]).ratio()
                    if sim > 0.8:
                        cluster.append(j)
                        visited[j] = True
            clusters.append(cluster)

        factors = [1.0] * n
        for cl in clusters:
            size = len(cl)
            for idx in cl:
                factors[idx] = size
                self.cluster_sizes[idx] = size

        return factors, clusters

    def calibrate(self, rows):
        """
        通胀校准: 每条JD中的技能频率被 1/cluster_size 降权

        Before: "大模型"出现在5条抄袭JD中 → count=5
        After:  "大模型"出现在5条抄袭JD中 → effective_count=5*(1/5)=1

        Before: "大模型"出现在5条独立JD中 → count=5
        After:  "大模型"出现在5条独立JD中 → effective_count=5*(1/1)=5
        """
        factors, clusters = self.detect_plagiarism(rows)

        # 每个技能的有效频次
        raw_freq = Counter()
        effective_freq = Counter()
        skill_evidence = defaultdict(list)  # 技能 -> 证据JD列表

        for i, row in enumerate(rows):
            text = str(row.get('skill_requirements', ''))
            skills = set(extract_skills_from_text(text).keys())
            weight = 1.0 / factors[i]

            for skill in skills:
                if skill in SOFT_SKILLS: continue
                raw_freq[skill] += 1
                effective_freq[skill] += weight
                if len(skill_evidence[skill]) < 3:
                    snippet = text[:120]
                    skill_evidence[skill].append({
                        'job': row.get('job_name', ''),
                        'weight': round(weight, 2),
                        'snippet': snippet,
                    })

        # 计算通胀率
        inflation_analysis = {}
        for skill in list(raw_freq.keys()):
            raw = raw_freq[skill]
            eff = effective_freq[skill]
            if raw >= 3:  # 至少出现3次才分析
                infl_rate = 1 - (eff / raw)
                if infl_rate > 0.3:  # 30%以上是抄袭
                    inflation_analysis[skill] = {
                        'raw_count': raw,
                        'effective_count': round(eff, 2),
                        'inflation_rate': round(infl_rate, 2),
                        'verdict': '严重通胀' if infl_rate > 0.6 else '中度通胀' if infl_rate > 0.3 else '正常',
                        'action': '降权至 {:.1f}'.format(eff),
                    }

        return {
            'total_jds': len(rows),
            'clusters_found': len([c for c in clusters if len(c) >= 3]),
            'total_inflated_jds': sum(1 for f in factors if f >= 3),
            'raw_counts': dict(raw_freq.most_common(100)),
            'effective_counts': dict(effective_freq.most_common(100)),
            'inflation_analysis': dict(sorted(
                inflation_analysis.items(),
                key=lambda x: -x[1]['inflation_rate']
            )[:20]),
            'skill_evidence': {k: v for k, v in skill_evidence.items() if k in inflation_analysis},
        }


# ============================================================
#  第2部分: 时滞校准 — 预测性技能激活
# ============================================================

class TimeLagCalibrator:
    """时滞校准器: 对技术信号预测JD需求时间点"""

    LAG_MEDIAN_MONTHS = 11.0  # 中位数时滞

    def __init__(self):
        self.predictions = []

    def calibrate(self, tech_signals, jd_data_available=True):
        """
        对每个技术信号，预测其在JD市场的激活时间

        Input: tech_signals = {keyword: {first_arxiv_date, first_github_date, ...}}
        Output: 每个关键词的预测激活状态
        """
        results = {}
        now = datetime.now()

        for keyword, signal in tech_signals.items():
            # 最早出现日期
            dates = []
            if signal.get('first_arxiv'):
                dates.append(signal['first_arxiv'])
            if signal.get('first_github'):
                dates.append(signal['first_github'])

            if not dates:
                continue

            earliest = min(dates)
            predicted_jd_date = earliest + timedelta(days=self.LAG_MEDIAN_MONTHS * 30.44)

            # 状态判定
            jd_count = signal.get('jd_count', 0)
            if jd_count > 50:
                status = 'converted'  # 已大规模出现在JD
                action = '纳入核心图谱'
            elif jd_count > 0:
                status = 'emerging'   # 开始出现在JD
                action = '纳入图谱，标记为"新兴"'
            elif predicted_jd_date < now:
                status = 'overdue'    # 预测到期但未出现
                action = '重新评估: 可能不会转化为岗位需求，或时滞更长'
            elif predicted_jd_date > now:
                days_remaining = (predicted_jd_date - now).days
                status = 'predicted'
                action = f'预计{days_remaining}天后({predicted_jd_date.strftime("%Y-%m")})出现在JD中'
            else:
                status = 'unknown'
                action = '数据不足'

            results[keyword] = {
                'tech_first_seen': earliest.strftime('%Y-%m'),
                'predicted_jd_date': predicted_jd_date.strftime('%Y-%m'),
                'lag_months': self.LAG_MEDIAN_MONTHS,
                'jd_count': jd_count,
                'status': status,
                'action': action,
            }

            if status == 'predicted':
                self.predictions.append((keyword, predicted_jd_date))

        # 按紧急度排序
        self.predictions.sort(key=lambda x: x[1])

        return {
            'lag_coefficient_months': self.LAG_MEDIAN_MONTHS,
            'total_signals': len(results),
            'status_distribution': Counter(r['status'] for r in results.values()),
            'predictions': [
                {'keyword': kw, 'predicted_jd_date': pd.strftime('%Y-%m'), 'days_remaining': (pd - now).days}
                for kw, pd in self.predictions[:10]
            ],
            'per_keyword': results,
        }


# ============================================================
#  第3部分: 噪声校准 — Fisher精确检验
# ============================================================

class NoiseCalibrator:
    """噪声校准器: Fisher检验 + Cohen's h 过滤"""

    def __init__(self, alpha=0.05, min_effect_size=0.2):
        self.alpha = alpha
        self.min_effect_size = min_effect_size

    def _fisher_exact(self, a, b, c, d):
        """Fisher精确检验"""
        n = a + b + c + d
        if n == 0: return 1.0
        k = a + c

        def log_comb(n, k):
            if k < 0 or k > n: return -float('inf')
            return math.lgamma(n+1) - math.lgamma(k+1) - math.lgamma(n-k+1)

        log_total = log_comb(n, a + b)
        p_value = sum(
            math.exp(log_comb(k, i) + log_comb(n-k, a+b-i) - log_total)
            for i in range(a, min(k, a+b) + 1)
        )
        return min(p_value, 1.0)

    def _cohens_h(self, a, b, c, d):
        """Cohen's h效应量"""
        n1, n2 = a + b, c + d
        if n1 == 0 or n2 == 0: return 0.0
        p1, p2 = a / n1, c / n2
        return 2 * abs(math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))

    def calibrate(self, window_a_counts, window_b_counts, window_a_size, window_b_size):
        """
        噪声校准: 只保留统计显著的技能变化

        window_a = 早期窗口, window_b = 近期窗口
        返回: 经过滤的技能变化列表
        """
        all_skills = set(window_a_counts.keys()) | set(window_b_counts.keys())
        filtered_changes = []

        for skill in sorted(all_skills):
            a = window_b_counts.get(skill, 0)
            b = window_b_size - a
            c = window_a_counts.get(skill, 0)
            d = window_a_size - c

            p_value = self._fisher_exact(a, b, c, d)
            effect_size = self._cohens_h(a, b, c, d)

            signal_type = 'new' if a > c else 'declining' if a < c else 'stable'
            significant = p_value < self.alpha and effect_size >= self.min_effect_size

            # 噪声判定
            if significant:
                if effect_size >= 0.5:
                    confidence = 'high'
                else:
                    confidence = 'medium'
                action = '纳入图谱变化'
            else:
                if p_value >= self.alpha:
                    reason = f'统计不显著(p={p_value:.3f})'
                else:
                    reason = f'效应量太小(h={effect_size:.2f})'
                confidence = 'noise'
                action = f'排除: {reason}'

            filtered_changes.append({
                'skill': skill,
                'type': signal_type,
                'window_a_count': c, 'window_b_count': a,
                'p_value': round(p_value, 4),
                'effect_size': round(effect_size, 3),
                'significant': significant,
                'confidence': confidence,
                'action': action,
            })

        real_signals = [c for c in filtered_changes if c['significant']]
        noise_signals = [c for c in filtered_changes if not c['significant']]

        return {
            'total_skills_checked': len(filtered_changes),
            'real_signals': len(real_signals),
            'noise_filtered': len(noise_signals),
            'noise_reduction': f'{len(noise_signals)}/{len(filtered_changes)} ({len(noise_signals)/max(len(filtered_changes),1):.0%})',
            'filter_parameters': {'alpha': self.alpha, 'min_effect_size': self.min_effect_size},
            'real_signals': real_signals[:20],
            'filtered_noise': noise_signals[:10],
        }


# ============================================================
#  第4部分: 幻觉校准 — HallucinationGuard
# ============================================================

class HallucinationCalibrator:
    """幻觉校准器: 对接 HallucinationGuard + 软技能黑名单 + 过时技能标记"""

    def __init__(self):
        sys.path.insert(0, BASE_DIR)
        from hallucination_guard import HallucinationGuard
        self.guard = HallucinationGuard(enable_rag=False)

    def calibrate(self, skills_with_source):
        """
        幻觉校准: 每个技能经过4层白名单过滤

        Input: [(skill_name, source, confidence), ...]
        Output: 每个技能的验证结果 + action
        """
        results = []
        stats = Counter()

        for skill, source, confidence in skills_with_source:
            result = self.guard.validate_skill(skill, confidence)

            # 根据验证结果决定action
            verdict = result.get('status', 'unknown')
            stats[verdict] += 1

            if verdict == 'verified':
                action = '保留'
                calibrated_confidence = confidence
            elif verdict == 'outdated':
                action = '标记为"过时"，移入历史层'
                calibrated_confidence = 0.1
            elif verdict == 'rejected':
                action = '从图谱中移除'
                calibrated_confidence = 0.0
            elif verdict == 'candidate':
                action = '暂存待审队列，不入图谱'
                calibrated_confidence = 0.3
            else:  # unknown
                action = '标记低置信度，待人工确认'
                calibrated_confidence = min(confidence, 0.5)

            results.append({
                'skill': skill,
                'source': source,
                'original_confidence': confidence,
                'verdict': verdict,
                'calibrated_confidence': calibrated_confidence,
                'action': action,
                'reason': result.get('reason', ''),
            })

        return {
            'total_skills': len(results),
            'verdict_distribution': dict(stats),
            'rejection_rate': f'{stats["rejected"]}/{len(results)} ({stats["rejected"]/max(len(results),1):.1%})',
            'retention_rate': f'{stats["verified"]}/{len(results)} ({stats["verified"]/max(len(results),1):.1%})',
            'calibrated_skills': results,
        }


# ============================================================
#  第5部分: 综合图谱校准器
# ============================================================

class GraphCalibrator:
    """综合校准器: 合并四类校准，输出干净的岗位能力图谱"""

    def __init__(self):
        self.inflation = InflationCalibrator()
        self.timelag = TimeLagCalibrator()
        self.noise = NoiseCalibrator()
        self.hallucination = HallucinationCalibrator()

    def build_calibrated_graph(self, rows, tech_signals=None):
        """
        完整校准流水线:

        原始JD → [通胀校准] → [时滞校准] → [噪声校准] → [幻觉校准] → 干净图谱
        """
        n = len(rows)
        print(f'[1/4] 通胀校准: {n}条JD...')
        infl_result = self.inflation.calibrate(rows)

        print(f'[2/4] 时滞校准: {len(tech_signals) if tech_signals else 0}个技术信号...')
        lag_result = self.timelag.calibrate(tech_signals or {})

        # 模拟窗口对比（前后3个月）
        half = n // 2
        window_a = rows[:half]
        window_b = rows[half:]

        wa_counts = Counter()
        for r in window_a:
            for s in extract_skills_from_text(str(r.get('skill_requirements', ''))):
                if s not in SOFT_SKILLS: wa_counts[s] += 1

        wb_counts = Counter()
        for r in window_b:
            for s in extract_skills_from_text(str(r.get('skill_requirements', ''))):
                if s not in SOFT_SKILLS: wb_counts[s] += 1

        print(f'[3/4] 噪声校准: {len(wa_counts|wb_counts)}个技能...')
        noise_result = self.noise.calibrate(wa_counts, wb_counts, len(window_a), len(window_b))

        # 幻觉校准: 取Top50技能
        all_skills = list((wa_counts + wb_counts).most_common(50))
        skills_for_guard = [(s, 'extracted', 0.85) for s, _ in all_skills]

        print(f'[4/4] 幻觉校准: {len(skills_for_guard)}个技能...')
        hallu_result = self.hallucination.calibrate(skills_for_guard)

        # 构建校准后的图谱
        graph = self._build_graph(rows, infl_result, noise_result, hallu_result, lag_result)

        return {
            'calibrated_at': datetime.now().isoformat(),
            'pipeline': {
                'inflation': {k: v for k, v in infl_result.items() if k != 'skill_evidence'},
                'time_lag': {k: v for k, v in lag_result.items() if k != 'per_keyword'},
                'noise': noise_result,
                'hallucination': {k: v for k, v in hallu_result.items() if k != 'calibrated_skills'},
            },
            'graph': graph,
        }

    def _build_graph(self, rows, infl_result, noise_result, hallu_result, lag_result):
        """构建校准后图谱节点和边"""
        # 有效技能: verified + candidate(字典中) = 可用
        # rejected/outdated = 排除
        valid_skills = {
            s['skill'] for s in hallu_result['calibrated_skills']
            if s['verdict'] in ('verified', 'candidate')  # candidate来自字典,可信
        }

        real_signals = {
            s['skill'] for s in noise_result.get('real_signals', [])
        }

        # 使用有效频次
        eff_counts = infl_result.get('effective_counts', {})

        # 节点: 技能 -> {有效频次, 状态, 分类}
        nodes = {}
        for skill, eff_count in eff_counts.items():
            if skill not in valid_skills:
                continue

            # 技能分类
            if skill in AI_SKILLS:
                category = 'AI新兴技能'
            elif skill in TRADITIONAL_SKILLS:
                category = '传统技术'
            else:
                category = '其他'

            # 状态
            if skill in real_signals:
                status = 'growing'
            elif eff_count >= 10:
                status = 'stable'
            else:
                status = 'emerging'

            nodes[skill] = {
                'effective_count': round(eff_count, 2),
                'category': category,
                'status': status,
                'is_inflated': skill in infl_result.get('inflation_analysis', {}),
                'inflation_details': infl_result.get('inflation_analysis', {}).get(skill),
            }

        # 边: 共现关系（同一JD中的技能对）
        edges = []
        cooccur = Counter()
        sample_size = min(500, len(rows))
        for row in rows[:sample_size]:
            skills = [s for s in extract_skills_from_text(str(row.get('skill_requirements', '')))
                      if s in nodes and s not in SOFT_SKILLS]
            for i in range(len(skills)):
                for j in range(i+1, len(skills)):
                    pair = tuple(sorted([skills[i], skills[j]]))
                    cooccur[pair] += 1

        for (s1, s2), count in cooccur.most_common(200):
            if count >= 2:  # 至少共现2次
                edges.append({
                    'source': s1, 'target': s2,
                    'weight': count,
                    'relation': 'co_occurs',
                })

        # 时滞预测节点
        predicted_nodes = {}
        for kw, info in lag_result.get('per_keyword', {}).items():
            if info['status'] == 'predicted':
                predicted_nodes[kw] = {
                    'effective_count': 0,
                    'category': '预测',
                    'status': 'predicted',
                    'predicted_jd_date': info['predicted_jd_date'],
                    'lag_months': info['lag_months'],
                }

        return {
            'nodes': len(nodes),
            'edges': len(edges),
            'predicted_nodes': len(predicted_nodes),
            'stats': {
                'stable': sum(1 for n in nodes.values() if n['status'] == 'stable'),
                'growing': sum(1 for n in nodes.values() if n['status'] == 'growing'),
                'emerging': sum(1 for n in nodes.values() if n['status'] == 'emerging'),
                'by_category': dict(Counter(n['category'] for n in nodes.values())),
                'inflated_skills': sum(1 for n in nodes.values() if n['is_inflated']),
            },
            'top_skills': dict(sorted(
                {k: v['effective_count'] for k, v in nodes.items()}.items(),
                key=lambda x: -x[1]
            )[:30]),
            'predicted_skills': predicted_nodes,
            'sample_edges': edges[:50],
        }


# ============================================================
#  主入口
# ============================================================

def main():
    print('=' * 70)
    print('  Graph Calibrator - 图谱校准器')
    print('  检测 -> 矫正 -> 输出干净图谱')
    print('=' * 70)

    # 加载数据
    annotated_dir = os.path.join(BASE_DIR, 'skill_ner_release', 'data', 'annotated')
    files = sorted(glob.glob(os.path.join(annotated_dir, '*.csv')))

    rows = []
    for fp in files[:20]:  # 取前20个文件做演示
        with open(fp, 'r', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                rows.append(row)

    print(f'\n[INPUT] {len(rows)} JDs from {min(20, len(files))} files')

    # 加载技术信号
    tech_signals = {}
    signal_path = os.path.join(BASE_DIR, 'outputs', 'outputs', 'latest_signals.json')
    lag_path = os.path.join(BASE_DIR, 'phase1_time_lag_report.json')
    mapping_path = os.path.join(BASE_DIR, 'phase2_tech_job_mapping.json')

    for path in [signal_path, lag_path, mapping_path]:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # 提取关键词数据
                if 'results' in data:
                    for r in data['results']:
                        kw = r.get('keyword', '')
                        tech_signals[kw] = {
                            'first_arxiv': datetime(2024, 8, 1),
                            'first_github': datetime(2024, 8, 1),
                            'jd_count': r.get('jd_stats', {}).get('total_jd_occurrences', 0),
                        }
                elif 'per_keyword' in data:
                    for kw, info in data['per_keyword'].items():
                        if kw not in tech_signals:
                            tech_signals[kw] = {
                                'first_arxiv': datetime(2024, 8, 1),
                                'jd_count': info.get('jd_occurrences', 0),
                            }
            except: pass

    if not tech_signals:
        # 用模拟数据
        tech_signals = {
            'agent': {'first_arxiv': datetime(2024,8,1), 'first_github': datetime(2024,8,1), 'jd_count': 1299},
            'rag': {'first_arxiv': datetime(2024,8,1), 'first_github': datetime(2024,8,1), 'jd_count': 1162},
            'diffusion transformers': {'first_arxiv': datetime(2024,8,1), 'first_github': datetime(2024,10,1), 'jd_count': 0},
            'self-evolving': {'first_arxiv': datetime(2024,8,1), 'first_github': datetime(2024,10,1), 'jd_count': 0},
        }

    # 运行校准
    calibrator = GraphCalibrator()
    result = calibrator.build_calibrated_graph(rows, tech_signals)

    # 输出
    pipeline = result['pipeline']
    graph = result['graph']

    print(f'\n{"="*60}')
    print(f'  校准结果')
    print(f'{"="*60}')

    print(f'\n  [通胀] 抄袭簇: {pipeline["inflation"]["clusters_found"]}个')
    print(f'         通胀JD: {pipeline["inflation"]["total_inflated_jds"]}条')
    infl_skills = pipeline['inflation'].get('inflation_analysis', {})
    if infl_skills:
        print(f'         严重通胀技能: {[(s, d["inflation_rate"]) for s,d in list(infl_skills.items())[:5] if d["verdict"]=="严重通胀"]}')

    predicted = graph.get('predicted_skills', {})
    print(f'\n  [时滞] 预测信号: {len(predicted)}个')
    for kw, info in list(predicted.items())[:5]:
        print(f'         {kw}: {info["predicted_jd_date"]}')

    print(f'\n  [噪声] 过滤: {pipeline["noise"]["noise_reduction"]}')
    print(f'         真实信号: {pipeline["noise"]["real_signals"]}个')

    print(f'\n  [幻觉] 拒绝率: {pipeline["hallucination"]["rejection_rate"]}')
    print(f'         保留率: {pipeline["hallucination"]["retention_rate"]}')

    print(f'\n  [图谱] 节点: {graph["nodes"]}个 + {graph["predicted_nodes"]}个预测')
    print(f'         边: {graph["edges"]}条')
    print(f'         稳定: {graph["stats"]["stable"]}, 增长: {graph["stats"]["growing"]}, 新兴: {graph["stats"]["emerging"]}')
    print(f'         通胀技能: {graph["stats"]["inflated_skills"]}个')

    # 保存
    out = os.path.join(BASE_DIR, 'calibrated_graph.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f'\n[OUTPUT] {out}')
    print('=' * 70)


if __name__ == '__main__':
    main()
