#!/usr/bin/env python3
"""
capability_updater.py — 既有岗位能力动态更新引擎

核心思路：
  不是逐条JD做静态规则匹配（check_ability_update_v2的问题），
  也不是按日历季度硬切窗口，
  而是按JD发布顺序做滑动窗口采样，输出技能随时间连续变化的趋势曲线。

窗口策略（v4 — 滑动窗口）：
  每个岗位的JD按发布时间排序 → 固定大小的滑动窗口扫过 →
  每个窗口采样一次技能分布 → 输出 N 个连续时间点的技能快照。
  前端拿到后可以直接渲染为平滑渐变的时间轴，不再有"Q1跳到Q2"的断点。

三层过滤：
  第1层：Fisher精确检验 + Cohen's h效应量 → 排除采样噪声
  第2层：Jaccard抄袭簇检测 + 通胀系数校准 → 排除技能通胀
  第3层：过时技能知识库 + 软技能黑名单 → 排除已知旧技能和伪技能

输出：
  1. 每个岗位的"能力更新"摘要（可写回标注数据）
  2. 每个岗位的 timeline 数组（前端时间轴渲染的原始数据）
  3. 结构化变化报告（可接入 graph_calibrator.py）

Usage:
    python capability_updater.py --annotated_dir "./【已标注】filtered/jd_v2/" --output capability_update_report.json
"""

import os, sys, json, argparse, csv, re, math, glob
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from difflib import SequenceMatcher

# ── 复用 annotate_v3 的字典 ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '【已标注v3】'))
try:
    from annotate_v3 import (
        AI_SKILLS, TRADITIONAL_SKILLS, SOFT_SKILLS, OUTDATED_SKILLS,
        CORE_AI_SIGNALS, TRADITIONAL_TECH_BASE,
    )
except ImportError:
    # 回退：从 annotate_jd_v2 导入
    sys.path.insert(0, os.path.dirname(__file__))
    from annotate_jd_v2 import (
        AI_SKILLS, TRADITIONAL_SKILLS, SOFT_SKILLS,
    )
    OUTDATED_SKILLS = []
    CORE_AI_SIGNALS = list(AI_SKILLS)
    TRADITIONAL_TECH_BASE = list(TRADITIONAL_SKILLS)


# ============================================================
#  日期解析
# ============================================================

def parse_date(date_str):
    """解析多种日期格式 → (year, month, day) 或 None"""
    if not date_str or not str(date_str).strip():
        return None
    s = str(date_str).strip()

    # 去掉干扰后缀: "更新", "发布", "刷新" 等
    s = re.sub(r'(更新|发布|刷新|录入|采集)$', '', s).strip()

    # "7月25日" → (2026, 7, 25)
    m = re.match(r'(\d{1,2})月(\d{1,2})日', s)
    if m:
        return (2026, int(m.group(1)), int(m.group(2)))

    # "2025-07-15" / "2025/07/15"
    m = re.match(r'(20\d{2})[-/](\d{1,2})[-/](\d{1,2})', s)
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))

    # "2025-07" / "2025/07"
    m = re.match(r'(20\d{2})[-/](\d{1,2})$', s)
    if m:
        return (int(m.group(1)), int(m.group(2)), 1)

    # "90天前" → 推算
    m = re.match(r'(\d+)天前', s)
    if m:
        days_ago = int(m.group(1))
        d = datetime(2026, 8, 6) - timedelta(days=days_ago)
        return (d.year, d.month, d.day)

    # "今天" / "昨天"
    if '今天' in s:
        return (2026, 8)
    if '昨天' in s:
        return (2026, 8)

    return None


def quarter_key(year_month):
    """(year, month) → '2026Q1'"""
    if year_month is None:
        return 'Unknown'
    y, m = year_month
    return f'{y}Q{(m-1)//3+1}'


def month_index(year_month):
    """转为连续月份索引，用于排序。支持 (y,m) 或 (y,m,d)"""
    if year_month is None:
        return -1
    if len(year_month) == 3:
        y, m, d = year_month
        return y * 12 + m + d / 31.0  # 天级别精度
    y, m = year_month
    return y * 12 + m


# ============================================================
#  技能解析
# ============================================================

def parse_skills(skill_str):
    """从标注格式解析技能: 【技能名｜分类｜级别】 → set of skill names"""
    if not skill_str or not str(skill_str).strip():
        return set()
    s = str(skill_str).strip()
    if s in ('（JD文本过短）', '（未识别）', '（JD中未识别到明确的必备硬性技能）',
             '（未识别到明确的加分技能）'):
        return set()
    skills = set()
    for part in re.split(r'[；;]', s):
        part = part.strip()
        m = re.match(r'【(.+?)[｜|]', part)
        if m:
            sk = m.group(1).strip()
            if sk and '未识别' not in sk:
                skills.add(sk)
        elif part and '未识别' not in part and not part.startswith('（'):
            skills.add(part)
    return skills


def classify_skill_type(skill):
    """判断技能类型"""
    if skill in AI_SKILLS:
        return 'AI新兴技能'
    elif skill in TRADITIONAL_SKILLS:
        return '传统技术'
    elif skill in SOFT_SKILLS:
        return '软技能'
    else:
        return '未分类'


# ============================================================
#  第1层：统计显著性
# ============================================================

def fisher_exact_test(a, b, c, d):
    """
    Fisher精确检验（2x2列联表，双尾）
      窗口A: 出现a次 / 未出现b次
      窗口B: 出现c次 / 未出现d次
    H0: 两个窗口中技能出现概率相同
    返回双尾 p-value
    """
    if a + b + c + d == 0:
        return 1.0

    n = a + b + c + d
    k = a + c
    a_plus_b = a + b

    def log_comb(n_val, k_val):
        if k_val < 0 or k_val > n_val:
            return -float('inf')
        return math.lgamma(n_val + 1) - math.lgamma(k_val + 1) - math.lgamma(n_val - k_val + 1)

    log_total = log_comb(n, a_plus_b)

    # 观测表的概率
    obs_log_prob = log_comb(k, a) + log_comb(n - k, a_plus_b - a) - log_total
    obs_prob = math.exp(obs_log_prob)

    # 双尾：累加所有概率 <= 观测概率的表格
    p_value = 0.0
    for i in range(0, min(k, a_plus_b) + 1):
        log_num = log_comb(k, i) + log_comb(n - k, a_plus_b - i)
        prob = math.exp(log_num - log_total)
        if prob <= obs_prob * 1.0001:  # 容差
            p_value += prob

    return min(p_value, 1.0)


def cohens_h(p1, p2):
    """Cohen's h 效应量"""
    if p1 < 0 or p2 < 0 or p1 > 1 or p2 > 1:
        return 0.0
    # 防止 arcsin 定义域溢出
    p1 = max(0.0, min(1.0, p1))
    p2 = max(0.0, min(1.0, p2))
    return 2 * abs(math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))


# ============================================================
#  第2层：抄袭簇检测 + 通胀校准
# ============================================================

def text_similarity(a, b):
    """Jaccard相似度"""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def detect_plagiarism_clusters(rows, threshold=0.8):
    """检测抄袭簇，返回每行的通胀系数"""
    jd_texts = [r.get('skill_requirements', '') for r in rows]
    n = len(rows)
    visited = [False] * n
    clusters = []

    for i in range(n):
        if visited[i]:
            continue
        cluster = [i]
        for j in range(i + 1, n):
            if visited[j]:
                continue
            if text_similarity(jd_texts[i], jd_texts[j]) > threshold:
                cluster.append(j)
                visited[j] = True
        clusters.append(cluster)

    inflation_factors = [1.0] * n
    for cluster in clusters:
        factor = len(cluster)
        for idx in cluster:
            inflation_factors[idx] = factor

    return inflation_factors


# ============================================================
#  第3层：知识库校验
# ============================================================

def knowledge_check(skill):
    """检查技能是否过时/软技能/基础技能"""
    checks = {}

    # 过时技能
    for outdated_skill, reason in OUTDATED_SKILLS:
        if skill == outdated_skill:
            checks['obsolete'] = {'is_obsolete': True, 'reason': reason}
            break
    else:
        checks['obsolete'] = {'is_obsolete': False}

    checks['is_soft_skill'] = skill in SOFT_SKILLS
    checks['skill_type'] = classify_skill_type(skill)
    checks['is_foundational'] = (checks['skill_type'] == '传统技术')

    return checks


# ============================================================
#  传统岗位判定
# ============================================================

TRADITIONAL_JOB_KEYWORDS = [
    '产品经理', '前端', '后端', '测试', '运维', '爬虫',
    'Java', 'Python', 'C++', 'Android', 'iOS', '嵌入式',
    '项目经理', '数据分析', '硬件', '网络', '通信',
    '运营', '设计', '销售', '客服', '技术支持', '实施',
    '架构师', 'DBA', '系统管理员', '安全', '质量',
    '电子', '电气', '自动化', '机械', '结构',
    '.NET', 'PHP', 'Golang', 'Node.js', 'Web', 'H5',
    'Flutter', 'React Native', '小程序',
]

PURE_AI_JOB_SIGNALS = [
    '大模型', 'LLM', 'Agent', 'AIGC', '算法', '深度学习',
    'NLP', '自然语言', '计算机视觉', '人工智能', 'AI',
    '机器学习', '强化学习', '图像识别', '语音识别',
    '生成式', 'Prompt', 'RAG', '模型训练', '模型推理',
]


def is_traditional_job(job_name):
    """判断是否为传统岗位（非纯AI原生岗位）"""
    # 有纯AI信号 → 不是传统岗位
    if any(sig in job_name for sig in PURE_AI_JOB_SIGNALS):
        return False
    # 有传统关键词 → 是传统岗位
    if any(t in job_name for t in TRADITIONAL_JOB_KEYWORDS):
        return True
    # 默认视为传统岗位
    return True


# ============================================================
#  核心：时序能力更新检测
# ============================================================

class CapabilityUpdater:
    """
    既有岗位能力动态更新引擎。

    对每个岗位：
    1. 按时序窗口（季度）分组
    2. 统计每个窗口的技能分布
    3. 对比相邻窗口/首尾窗口，检测：
       - 新兴技能（emerging）：早期窗口无，后期窗口有
       - 衰退技能（declining）：早期窗口有，后期窗口无
       - 升级技能（upgraded）：加分→必备
    4. 三层过滤后输出可信变化
    """

    def __init__(self, alpha=0.05, min_effect_size=0.2, min_jd_per_window=2,
                 plagiarism_threshold=0.8, min_prevalence=0.05,
                 window_size=10, step=3):
        self.alpha = alpha
        self.min_effect_size = min_effect_size
        self.min_jd_per_window = min_jd_per_window
        self.plagiarism_threshold = plagiarism_threshold
        self.min_prevalence = min_prevalence  # 技能在窗口中至少出现的最低比例
        self.window_size = window_size        # 滑动窗口大小（条JD）
        self.step = step                      # 窗口滑动步长（条JD）

    def load_data(self, annotated_dir):
        """加载所有已标注CSV，按岗位名称分组"""
        files = glob.glob(os.path.join(annotated_dir, '*.csv'))
        print(f'[CapabilityUpdater] 加载 {len(files)} 个标注文件...')

        by_job = defaultdict(list)
        total = 0
        skipped = 0
        for fpath in files:
            with open(fpath, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    total += 1
                    job = row.get('job_name', '').strip()
                    if not job:
                        skipped += 1
                        continue
                    # 截断过长的岗位名
                    job_key = job[:60]
                    by_job[job_key].append(row)

        print(f'[CapabilityUpdater] {total} 条JD → {len(by_job)} 个去重岗位 (跳过{skipped}条无岗位名)')
        return by_job

    def build_sliding_windows(self, rows):
        """
        按JD发布时间排序后，用滑动窗口采样。
        不再依赖日历季度 — 纯按JD顺序切窗，技能演化渐变可见。

        自适应窗口大小：
          - JD充足（≥window_size*2）→ 标准滑动窗口
          - JD较少（≥10条）       → 缩小窗口到 n//2，确保至少前后2个采样点
          - JD太少（<10条）       → 整个作为一个窗口（不足以做时序分析）

        返回:
          windows: [(window_index, window_rows, date_range_str), ...]
          undated: 无日期的JD列表
        """
        dated = []
        undated = []
        for r in rows:
            date_str = r.get('issue_date', '')
            parsed = parse_date(date_str)
            if parsed:
                dated.append((parsed, r))
            else:
                undated.append(r)

        # 按时间排序
        dated.sort(key=lambda x: month_index(x[0]))

        n = len(dated)
        MIN_TOTAL_JD = 4  # 至少4条JD（前后对半分各2条，够做Fisher检验）

        if n < MIN_TOTAL_JD:
            return [], undated

        # 自适应窗口大小
        effective_window = min(self.window_size, max(self.min_jd_per_window, n // 2))
        effective_step = max(1, effective_window // 3)  # 约2/3重叠

        if n < effective_window * 2:
            # 不够两个完整窗口 → 前后对半分
            mid = n // 2
            first_rows = [r for _, r in dated[:mid]]
            second_rows = [r for _, r in dated[mid:]]
            w1 = (0, first_rows, self._date_range_str(first_rows))
            w2 = (1, second_rows, self._date_range_str(second_rows))
            return [w1, w2], undated

        windows = []
        win_idx = 0
        start = 0
        while start + effective_window <= n:
            end = start + effective_window
            win_rows = [r for _, r in dated[start:end]]
            date_range = self._date_range_str(win_rows)
            windows.append((win_idx, win_rows, date_range))
            win_idx += 1
            start += effective_step

        # 最后一个窗口：如果剩余JD >= min_jd_per_window 且跟上一个窗口不完全重叠就保留
        if start < n:
            remaining = [r for _, r in dated[start:]]
            if len(remaining) >= self.min_jd_per_window:
                date_range = self._date_range_str(remaining)
                windows.append((win_idx, remaining, date_range))

        return windows, undated

    def _date_range_str(self, rows):
        """从一组row中提取日期范围字符串"""
        dates = []
        for r in rows:
            d = r.get('issue_date', '')
            if d:
                dates.append(str(d).strip())
        if not dates:
            return 'Unknown'
        if len(dates) == 1:
            return dates[0]
        return f'{dates[0]} → {dates[-1]}'

    # 旧版兼容（不再使用）
    def build_time_windows(self, rows):
        """已废弃 — 请使用 build_sliding_windows"""
        return self.build_sliding_windows(rows)

    def window_skill_stats(self, rows):
        """
        统计一个窗口内的技能分布。
        返回 {skill: {essential_count, bonus_count, effective_essential, effective_bonus, ...}}
        """
        inf_factors = detect_plagiarism_clusters(rows, self.plagiarism_threshold)

        stats = defaultdict(lambda: {
            'essential_raw': 0, 'bonus_raw': 0,
            'essential_effective': 0.0, 'bonus_effective': 0.0,
            'jd_count': 0,
        })

        for i, r in enumerate(rows):
            wf = 1.0 / inf_factors[i]
            ess = parse_skills(r.get('必备技能', ''))
            bon = parse_skills(r.get('加分技能', ''))

            for s in ess:
                stats[s]['essential_raw'] += 1
                stats[s]['essential_effective'] += wf
                stats[s]['jd_count'] += 1
            for s in bon:
                stats[s]['bonus_raw'] += 1
                stats[s]['bonus_effective'] += wf
                stats[s]['jd_count'] += 1

        return stats, inf_factors

    def detect_emerging_skills(self, early_stats, late_stats, n_early, n_late):
        """检测新兴技能：流行度显著增长（不限于从无到有）"""
        emerging = []
        all_skills = set(list(early_stats.keys()) + list(late_stats.keys()))

        for skill in all_skills:
            kc = knowledge_check(skill)
            if kc['is_soft_skill'] or kc['obsolete']['is_obsolete']:
                continue

            early_ess = early_stats[skill]['essential_effective'] if skill in early_stats else 0.0
            early_bon = early_stats[skill]['bonus_effective'] if skill in early_stats else 0.0
            late_ess = late_stats[skill]['essential_effective'] if skill in late_stats else 0.0
            late_bon = late_stats[skill]['bonus_effective'] if skill in late_stats else 0.0

            early_total = early_ess + early_bon
            late_total = late_ess + late_bon

            # 后期至少要有实质性的存在
            if late_total < 2.0:
                continue

            p_early = early_total / max(n_early, 1)
            p_late = late_total / max(n_late, 1)

            # 必须增长（后期 > 早期）
            if p_late <= p_early:
                continue

            # 流行度增长至少 1.5x 或从零到有
            if p_early > 0 and p_late / max(p_early, 0.001) < 1.5:
                continue

            # 最低流行度
            if p_late < self.min_prevalence:
                continue

            # Fisher检验
            a = int(round(late_total))
            b = n_late - a
            c = int(round(early_total))
            d = n_early - c
            p_value = fisher_exact_test(a, b, c, d)
            h = cohens_h(p_late, p_early)

            # 双条件判定：统计显著 OR 效应量够大
            stat_sig = p_value < self.alpha and h >= self.min_effect_size
            large_effect = h >= 0.5  # 中等效应量，即使p值不够也值得关注
            if not stat_sig and not large_effect:
                continue

            # 通胀检查
            late_raw = (late_stats[skill]['essential_raw'] + late_stats[skill]['bonus_raw']) if skill in late_stats else 0
            is_inflated = (late_total < late_raw * 0.5 and late_raw >= 3)
            if is_inflated:
                continue

            # 分类：区分真正新增 vs 权重上升
            is_truly_new = (p_early < 0.01)  # 早期几乎不存在
            if is_truly_new:
                if kc['skill_type'] == 'AI新兴技能':
                    change_category = 'AI技能新增'
                else:
                    change_category = '技能新增'
            else:
                change_category = '技能权重上升'

            if kc['is_foundational']:
                confidence = '高' if (stat_sig and h >= 0.5) else '中' if h >= 0.5 else '低'
            else:
                confidence = '高' if (stat_sig and h >= 0.5) else '中' if (stat_sig or h >= 0.5) else '低'

            emerging.append({
                'skill': skill,
                'category': change_category,
                'confidence': confidence,
                'effect_size': round(h, 3),
                'p_value': round(p_value, 4),
                'statistically_significant': stat_sig,
                'early_prevalence': round(p_early, 4),
                'late_prevalence': round(p_late, 4),
                'late_essential_count': round(late_ess, 2),
                'late_bonus_count': round(late_bon, 2),
                'skill_type': kc['skill_type'],
            })

        return sorted(emerging, key=lambda x: (-x['effect_size'], x['confidence'] == '高'))

    def detect_declining_skills(self, early_stats, late_stats, n_early, n_late):
        """检测衰退技能：流行度显著下降（不限于从有到无）"""
        declining = []
        all_skills = set(list(early_stats.keys()) + list(late_stats.keys()))

        for skill in all_skills:
            kc = knowledge_check(skill)
            if kc['is_soft_skill']:
                continue

            early_ess = early_stats[skill]['essential_effective'] if skill in early_stats else 0.0
            early_bon = early_stats[skill]['bonus_effective'] if skill in early_stats else 0.0
            late_ess = late_stats[skill]['essential_effective'] if skill in late_stats else 0.0
            late_bon = late_stats[skill]['bonus_effective'] if skill in late_stats else 0.0

            early_total = early_ess + early_bon
            late_total = late_ess + late_bon

            # 早期至少要有实质性的存在
            if early_total < 2.0:
                continue

            p_early = early_total / max(n_early, 1)
            p_late = late_total / max(n_late, 1)

            # 必须下降（后期 < 早期）
            if p_late >= p_early:
                continue

            # 流行度下降至少 1/3 或降到零
            if p_late > 0 and p_early / max(p_late, 0.001) < 1.5:
                continue

            # 早期必须足够常见才有"衰退"意义
            if p_early < self.min_prevalence:
                continue

            # Fisher检验
            a = int(round(late_total))
            b = n_late - a
            c = int(round(early_total))
            d = n_early - c
            p_value = fisher_exact_test(a, b, c, d)
            h = cohens_h(p_early, p_late)

            # 双条件判定：统计显著 OR 效应量够大
            stat_sig = p_value < self.alpha and h >= self.min_effect_size
            large_effect = h >= 0.5
            if not stat_sig and not large_effect:
                continue

            # 通胀检查
            early_raw = (early_stats[skill]['essential_raw'] + early_stats[skill]['bonus_raw']) if skill in early_stats else 0
            is_inflated = (early_total < early_raw * 0.5 and early_raw >= 3)
            if is_inflated:
                continue

            # 分类：区分真正衰退（消失）vs 权重下降
            is_truly_gone = (p_late < 0.01)  # 后期几乎不存在
            if is_truly_gone:
                if kc['obsolete']['is_obsolete']:
                    change_category = '过时技能淘汰'
                else:
                    change_category = '技能衰退'
            else:
                change_category = '技能权重下降'

            if kc['is_foundational']:
                confidence = '高' if (stat_sig and h >= 0.5) else '中' if h >= 0.5 else '低'
            elif kc['obsolete']['is_obsolete']:
                confidence = '高' if (stat_sig or h >= 0.5) else '中'
            else:
                confidence = '高' if (stat_sig and h >= 0.5) else '中' if (stat_sig or h >= 0.5) else '低'

            declining.append({
                'skill': skill,
                'category': change_category,
                'confidence': confidence,
                'effect_size': round(h, 3),
                'p_value': round(p_value, 4),
                'statistically_significant': stat_sig,
                'early_prevalence': round(p_early, 4),
                'late_prevalence': round(p_late, 4),
                'early_essential_count': round(early_ess, 2),
                'late_essential_count': round(late_ess, 2),
                'skill_type': kc['skill_type'],
                'is_obsolete': kc['obsolete']['is_obsolete'],
            })

        return sorted(declining, key=lambda x: (-x['effect_size'], x['confidence'] == '高'))

    def detect_skill_upgrades(self, early_stats, late_stats, n_early, n_late):
        """检测技能升级：从加分变为必备"""
        upgrades = []
        common_skills = set(early_stats.keys()) & set(late_stats.keys())

        for skill in common_skills:
            kc = knowledge_check(skill)
            if kc['is_soft_skill'] or kc['obsolete']['is_obsolete']:
                continue

            # 早期：主要是加分
            early_ess = early_stats[skill]['essential_effective']
            early_bon = early_stats[skill]['bonus_effective']
            # 后期：主要是必备
            late_ess = late_stats[skill]['essential_effective']
            late_bon = late_stats[skill]['bonus_effective']

            early_total = early_ess + early_bon
            late_total = late_ess + late_bon

            # 必须两个窗口都有足够样本
            if early_total < 3.0 or late_total < 3.0:
                continue

            # 早期加分占比 > 70%，后期必备占比 > 70%
            early_bonus_ratio = early_bon / max(early_total, 0.01)
            late_essential_ratio = late_ess / max(late_total, 0.01)

            # 更严格：需要实质性变化，不是微弱波动
            if early_bonus_ratio >= 0.7 and late_essential_ratio >= 0.7:
                # 检查变化幅度是否足够大（比率变化≥0.4）
                shift = late_essential_ratio - (1.0 - early_bonus_ratio)
                if shift < 0.3:
                    continue

                upgrades.append({
                    'skill': skill,
                    'category': '升级（加分→必备）',
                    'confidence': '高' if shift >= 0.5 else '中' if shift >= 0.3 else '低',
                    'early_bonus_ratio': round(early_bonus_ratio, 2),
                    'late_essential_ratio': round(late_essential_ratio, 2),
                    'shift_magnitude': round(shift, 2),
                    'skill_type': kc['skill_type'],
                })

        return sorted(upgrades, key=lambda x: -x['late_essential_ratio'])

    def detect_skill_downgrades(self, early_stats, late_stats, n_early, n_late):
        """检测技能降级：从必备变为加分。要求变化幅度大、两侧样本充足。"""
        downgrades = []
        common_skills = set(early_stats.keys()) & set(late_stats.keys())

        for skill in common_skills:
            kc = knowledge_check(skill)
            if kc['is_soft_skill'] or kc['obsolete']['is_obsolete']:
                continue

            early_ess = early_stats[skill]['essential_effective']
            early_bon = early_stats[skill]['bonus_effective']
            late_ess = late_stats[skill]['essential_effective']
            late_bon = late_stats[skill]['bonus_effective']

            early_total = early_ess + early_bon
            late_total = late_ess + late_bon

            # 两侧至少各有3条有效JD
            if early_total < 3.0 or late_total < 3.0:
                continue

            early_essential_ratio = early_ess / max(early_total, 0.01)
            late_bonus_ratio = late_bon / max(late_total, 0.01)

            # 早期必备占比 >= 70%，后期加分占比 >= 70%
            # 变化幅度 >= 0.4（如 80%必备 → 20%必备 = 0.6 shift）
            if early_essential_ratio >= 0.7 and late_bonus_ratio >= 0.7:
                shift = early_essential_ratio - (1.0 - late_bonus_ratio)
                if shift < 0.4:
                    continue

                downgrades.append({
                    'skill': skill,
                    'category': '降级（必备→加分）',
                    'confidence': '高' if shift >= 0.6 else '中',
                    'early_essential_ratio': round(early_essential_ratio, 2),
                    'late_bonus_ratio': round(late_bonus_ratio, 2),
                    'shift_magnitude': round(shift, 2),
                    'effect_size': round(shift, 3),
                    'skill_type': kc['skill_type'],
                })

        return downgrades

    def detect_weight_changes(self, early_stats, late_stats, n_early, n_late):
        """检测技能权重变化：技能存在于两个窗口，但重要性显著上升或下降"""
        weight_changes = []
        common = set(early_stats.keys()) & set(late_stats.keys())

        for skill in common:
            kc = knowledge_check(skill)
            if kc['is_soft_skill'] or kc['obsolete']['is_obsolete']:
                continue

            early_ess = early_stats[skill]['essential_effective']
            early_bon = early_stats[skill]['bonus_effective']
            late_ess = late_stats[skill]['essential_effective']
            late_bon = late_stats[skill]['bonus_effective']

            early_total = early_ess + early_bon
            late_total = late_ess + late_bon

            if early_total < 2.0 or late_total < 2.0:
                continue

            p_early = early_total / max(n_early, 1)
            p_late = late_total / max(n_late, 1)

            # 至少 1.5 倍变化
            ratio = p_late / max(p_early, 0.001) if p_late > p_early else p_early / max(p_late, 0.001)
            if ratio < 1.5:
                continue

            a = int(round(late_total))
            b = n_late - a
            c = int(round(early_total))
            d = n_early - c
            p_value = fisher_exact_test(a, b, c, d)
            h = cohens_h(p_early, p_late)

            stat_sig = p_value < self.alpha and h >= self.min_effect_size
            large_effect = h >= 0.5
            if not stat_sig and not large_effect:
                continue

            direction = '↑' if p_late > p_early else '↓'
            if p_late > p_early:
                change_category = '技能权重上升'
            else:
                change_category = '技能权重下降'

            if stat_sig and h >= 0.5:
                confidence = '高'
            elif stat_sig or h >= 0.5:
                confidence = '中'
            else:
                confidence = '低'

            weight_changes.append({
                'skill': skill,
                'category': change_category,
                'confidence': confidence,
                'effect_size': round(h, 3),
                'p_value': round(p_value, 4),
                'statistically_significant': stat_sig,
                'early_prevalence': round(p_early, 4),
                'late_prevalence': round(p_late, 4),
                'direction': direction,
                'early_essential': round(early_ess, 2),
                'late_essential': round(late_ess, 2),
                'skill_type': kc['skill_type'],
            })

        return sorted(weight_changes, key=lambda x: -x['effect_size'])

    def analyze_job(self, job_name, rows):
        """对单个岗位执行滑动窗口时序分析，输出前端可渲染的 timeline 数据"""
        windows, undated = self.build_sliding_windows(rows)

        if len(windows) < 1:
            return None

        # 构建每个窗口的技能统计
        window_snapshots = []
        for win_idx, win_rows, date_range in windows:
            if len(win_rows) < self.min_jd_per_window:
                continue
            stats, inf_factors = self.window_skill_stats(win_rows)
            top_skills = sorted(
                [(s, d['essential_effective'] + d['bonus_effective'],
                  d['essential_effective'], d['bonus_effective'])
                 for s, d in stats.items()],
                key=lambda x: -x[1]
            )[:20]
            window_snapshots.append({
                'window_index': win_idx,
                'date_range': date_range,
                'jd_count': len(win_rows),
                'total_unique_skills': len(stats),
                'top_skills': [
                    {'skill': s, 'effective_count': round(tc, 2),
                     'essential': round(es, 2), 'bonus': round(bn, 2)}
                    for s, tc, es, bn in top_skills
                ],
                '_stats': stats,          # 内部使用
                '_inf_factors': inf_factors,
                '_rows': win_rows,
            })

        if len(window_snapshots) < 2:
            return None  # 至少2个有效窗口才能检测变化

        # ── 变化检测策略 ──
        # 相邻窗口重叠70%，几乎一样，不宜做主力检测。
        # 主力：每个窗口 vs 第一个窗口（累积趋势），越往后差距越大，自然捕获渐变。
        # 辅助：首尾对比（最大跨度）+ 隔窗对比（中等跨度）。
        all_changes = []
        first = window_snapshots[0]
        last = window_snapshots[-1]

        # 1) 累积趋势：窗口[0] vs 窗口[i]，i 从后半段开始（跨度够大才有意义）
        half_point = max(2, len(window_snapshots) // 2)
        for i in range(half_point, len(window_snapshots)):
            curr = window_snapshots[i]
            emerging = self.detect_emerging_skills(
                first['_stats'], curr['_stats'], first['jd_count'], curr['jd_count'])
            declining = self.detect_declining_skills(
                first['_stats'], curr['_stats'], first['jd_count'], curr['jd_count'])
            upgrades = self.detect_skill_upgrades(
                first['_stats'], curr['_stats'], first['jd_count'], curr['jd_count'])
            downgrades = self.detect_skill_downgrades(
                first['_stats'], curr['_stats'], first['jd_count'], curr['jd_count'])
            for ch in emerging + declining + upgrades + downgrades:
                ch['_from_window'] = 0
                ch['_to_window'] = i
                ch['_from_date'] = first['date_range']
                ch['_to_date'] = curr['date_range']
            all_changes.extend(emerging + declining + upgrades + downgrades)

        # 2) 隔窗对比：每隔 step_compare 个窗口比较一次（中等跨度，捕获中期变化）
        step_compare = max(3, len(window_snapshots) // 8)
        for i in range(step_compare, len(window_snapshots), step_compare):
            prev = window_snapshots[i - step_compare]
            curr = window_snapshots[i]
            emerging = self.detect_emerging_skills(
                prev['_stats'], curr['_stats'], prev['jd_count'], curr['jd_count'])
            declining = self.detect_declining_skills(
                prev['_stats'], curr['_stats'], prev['jd_count'], curr['jd_count'])
            upgrades = self.detect_skill_upgrades(
                prev['_stats'], curr['_stats'], prev['jd_count'], curr['jd_count'])
            downgrades = self.detect_skill_downgrades(
                prev['_stats'], curr['_stats'], prev['jd_count'], curr['jd_count'])
            for ch in emerging + declining + upgrades + downgrades:
                ch['_from_window'] = prev['window_index']
                ch['_to_window'] = curr['window_index']
                ch['_from_date'] = prev['date_range']
                ch['_to_date'] = curr['date_range']
            all_changes.extend(emerging + declining + upgrades + downgrades)

        # 3) 首尾对比（最大跨度），置信度加权
        first_last_emerging = self.detect_emerging_skills(
            first['_stats'], last['_stats'], first['jd_count'], last['jd_count'])
        first_last_declining = self.detect_declining_skills(
            first['_stats'], last['_stats'], first['jd_count'], last['jd_count'])
        first_last_upgrades = self.detect_skill_upgrades(
            first['_stats'], last['_stats'], first['jd_count'], last['jd_count'])
        first_last_downgrades = self.detect_skill_downgrades(
            first['_stats'], last['_stats'], first['jd_count'], last['jd_count'])
        for ch in (first_last_emerging + first_last_declining +
                   first_last_upgrades + first_last_downgrades):
            ch['_from_window'] = 0
            ch['_to_window'] = len(window_snapshots) - 1
            ch['_from_date'] = first['date_range']
            ch['_to_date'] = last['date_range']
            if ch['confidence'] == '中':
                ch['confidence'] = '高'
            elif ch['confidence'] == '低':
                ch['confidence'] = '中'
        all_changes.extend(first_last_emerging + first_last_declining +
                          first_last_upgrades + first_last_downgrades)

        # 去重：同一技能 + 同一变化类型 → 保留效应量最大 + 置信度最高的
        deduped = {}
        for ch in all_changes:
            key = (ch['skill'], ch['category'])
            if key not in deduped:
                deduped[key] = ch
            else:
                old = deduped[key]
                new_score = (ch.get('effect_size', 0),
                            {'高': 3, '中': 2, '低': 1}.get(ch['confidence'], 0))
                old_score = (old.get('effect_size', 0),
                            {'高': 3, '中': 2, '低': 1}.get(old['confidence'], 0))
                if new_score > old_score:
                    deduped[key] = ch

        changes = sorted(deduped.values(),
                        key=lambda x: (
            {'AI技能新增': 0, '技能新增': 1, '技能权重上升': 2,
             '技能衰退': 3, '技能权重下降': 4,
             '过时技能淘汰': 5, '升级': 6, '降级': 7}
                .get(x.get('category', '').replace('（加分→必备）', '').replace('（必备→加分）', ''), 8),
            -x.get('effect_size', 0)
        ))

        # 分类汇总
        emerging = [c for c in changes if '新增' in c.get('category', '')]
        declining = [c for c in changes if '衰退' in c.get('category', '') or '淘汰' in c.get('category', '')]
        weight_up = [c for c in changes if '权重上升' in c.get('category', '')]
        weight_down = [c for c in changes if '权重下降' in c.get('category', '')]
        upgrades = [c for c in changes if '升级' in c.get('category', '')]
        downgrades = [c for c in changes if '降级' in c.get('category', '')]
        ai_emerging = [c for c in emerging if c['skill_type'] == 'AI新兴技能']
        ai_declining_obsolete = [c for c in declining if c.get('is_obsolete')]
        traditional = is_traditional_job(job_name)

        # 生成摘要
        update_summary = self._generate_update_summary(
            job_name, traditional, ai_emerging, ai_declining_obsolete,
            emerging, declining, upgrades, weight_up, weight_down)

        # 移除内部字段，保留前端可用的干净数据
        clean_snapshots = []
        for ws in window_snapshots:
            clean_snapshots.append({
                'window_index': ws['window_index'],
                'date_range': ws['date_range'],
                'jd_count': ws['jd_count'],
                'total_unique_skills': ws['total_unique_skills'],
                'top_skills': ws['top_skills'],
            })

        return {
            'job': job_name,
            'is_traditional': traditional,
            'total_jds': len(rows),
            'dated_jds': sum(ws['jd_count'] for ws in window_snapshots),
            'undated_jds': len(undated),
            'time_range': f'{window_snapshots[0]["date_range"]} → {window_snapshots[-1]["date_range"]}',
            'timeline': clean_snapshots,           # ← 前端时间轴渲染数据
            'window_count': len(clean_snapshots),
            'window_size': self.window_size,
            'step': self.step,
            'changes': changes,
            'change_summary': {
                'emerging_count': len(emerging),
                'declining_count': len(declining),
                'weight_up_count': len(weight_up),
                'weight_down_count': len(weight_down),
                'upgrade_count': len(upgrades),
                'downgrade_count': len(downgrades),
                'ai_emerging': [c['skill'] for c in ai_emerging],
                'obsolete_declining': [c['skill'] for c in ai_declining_obsolete],
                'high_confidence': len([c for c in changes if c['confidence'] == '高']),
                'medium_confidence': len([c for c in changes if c['confidence'] == '中']),
                'low_confidence': len([c for c in changes if c['confidence'] == '低']),
            },
            'update_summary': update_summary,
        }

    def _generate_update_summary(self, job_name, traditional, ai_emerging,
                                  ai_declining_obsolete, emerging, declining, upgrades,
                                  weight_up=None, weight_down=None):
        """生成人类可读的能力更新摘要"""
        parts = []
        weight_up = weight_up or []
        weight_down = weight_down or []

        if ai_emerging:
            skills_str = '、'.join(c['skill'] for c in ai_emerging[:5])
            if traditional:
                parts.append(f'传统岗位新增AI技能要求：{skills_str}')
            else:
                parts.append(f'岗位新增AI技能：{skills_str}')

        if ai_declining_obsolete:
            skills_str = '、'.join(c['skill'] for c in ai_declining_obsolete[:5])
            parts.append(f'过时技能被市场淘汰：{skills_str}')

        # 非AI但有统计显著变化
        other_emerging = [c for c in emerging
                          if c['skill_type'] != 'AI新兴技能' and c['confidence'] in ('高', '中')]
        if other_emerging:
            skills_str = '、'.join(c['skill'] for c in other_emerging[:5])
            parts.append(f'新增技能需求：{skills_str}')

        other_declining = [c for c in declining
                           if not c.get('is_obsolete') and c['confidence'] in ('高', '中')]
        if other_declining:
            skills_str = '、'.join(c['skill'] for c in other_declining[:5])
            parts.append(f'技能需求减少：{skills_str}')

        if upgrades:
            skills_str = '、'.join(c['skill'] for c in upgrades[:5])
            parts.append(f'技能重要性提升（加分→必备）：{skills_str}')

        if weight_up:
            skills_str = '、'.join(c['skill'] for c in weight_up[:5])
            parts.append(f'技能权重上升：{skills_str}')

        if weight_down:
            skills_str = '、'.join(c['skill'] for c in weight_down[:5])
            parts.append(f'技能权重下降：{skills_str}')

        if not parts:
            return '无显著能力更新'

        return '；'.join(parts)

    def run(self, annotated_dir, top_n=None):
        """全量运行，返回所有岗位的分析结果"""
        by_job = self.load_data(annotated_dir)

        # 按JD数量排序（优先分析数据充足的大岗位）
        sorted_jobs = sorted(by_job.items(), key=lambda x: -len(x[1]))

        if top_n:
            sorted_jobs = sorted_jobs[:top_n]

        results = []
        skipped_insufficient = 0
        skipped_no_changes = 0

        for job_name, rows in sorted_jobs:
            result = self.analyze_job(job_name, rows)
            if result is None:
                skipped_insufficient += 1
                continue
            if not result['changes']:
                skipped_no_changes += 1
            results.append(result)

        print(f'\n[CapabilityUpdater] 滑动窗口分析完成:')
        print(f'  窗口配置: size={self.window_size}条/窗, step={self.step}条')
        print(f'  分析岗位: {len(results)}')
        print(f'  数据不足跳过: {skipped_insufficient}')
        print(f'  无显著变化: {skipped_no_changes}')
        print(f'  有变化的岗位: {len([r for r in results if r["changes"]])}')

        total_changes = sum(len(r['changes']) for r in results)
        high = sum(1 for r in results for c in r['changes'] if c['confidence'] == '高')
        mid = sum(1 for r in results for c in r['changes'] if c['confidence'] == '中')
        low = sum(1 for r in results for c in r['changes'] if c['confidence'] == '低')
        print(f'  总变化数: {total_changes} (高:{high} 中:{mid} 低:{low})')

        # 时间轴覆盖统计
        windows_per_job = Counter(len(r['timeline']) for r in results)
        print(f'  时间轴覆盖:')
        for nw, cnt in sorted(windows_per_job.items()):
            print(f'    {nw}个采样点: {cnt}个岗位')

        return results

    def build_update_index(self, results):
        """
        构建 {岗位名 → 能力更新文本} 的索引，
        供 annotate_jd_v2.py 写回标注数据使用。
        """
        index = {}
        for r in results:
            job = r['job']
            summary = r['update_summary']
            index[job] = summary

            # 也按子串匹配（因为岗位名可能有细微差异）
            # 例如 "Java后端开发工程师" 应该匹配 "Java后端开发"
            short_job = job[:20]
            if short_job != job:
                if short_job not in index:
                    index[short_job] = summary

        return index

    def export_graph_input(self, results):
        """
        导出为 graph_calibrator.py 可用的格式。
        每个变化映射为图谱中的节点状态变更。
        """
        graph_updates = []
        for r in results:
            for c in r['changes']:
                if c['confidence'] not in ('高', '中'):
                    continue

                graph_updates.append({
                    'job': r['job'],
                    'skill': c['skill'],
                    'change_type': c['category'],
                    'confidence': c['confidence'],
                    'effect_size': c.get('effect_size', 0),
                    'action': self._map_to_graph_action(c),
                })

        return graph_updates

    def _map_to_graph_action(self, change):
        """将变化类型映射为图谱操作"""
        cat = change['category']
        if '新增' in cat:
            return 'add_edge' if change['confidence'] == '高' else 'suggest_edge'
        elif '淘汰' in cat or '衰退' in cat:
            return 'mark_deprecated'
        elif '升级' in cat:
            return 'upgrade_weight'
        elif '降级' in cat:
            return 'downgrade_weight'
        return 'no_action'


# ============================================================
#  主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='既有岗位能力动态更新引擎（滑动窗口v4）')
    parser.add_argument('--annotated_dir', type=str, required=True,
                        help='已标注CSV目录')
    parser.add_argument('--output', type=str, default='capability_update_report.json')
    parser.add_argument('--graph_output', type=str, default='capability_graph_updates.json')
    parser.add_argument('--top_n', type=int, default=0,
                        help='限制分析前N个岗位（0=全部）')
    parser.add_argument('--window_size', type=int, default=10,
                        help='滑动窗口大小（条JD），默认10')
    parser.add_argument('--step', type=int, default=3,
                        help='窗口滑动步长（条JD），默认3')
    parser.add_argument('--min_jd', type=int, default=2,
                        help='单窗口最小JD数，默认2')
    args = parser.parse_args()

    print('=' * 60)
    print('  既有岗位能力动态更新引擎（滑动窗口v4）')
    print('=' * 60)

    updater = CapabilityUpdater(
        alpha=0.05,
        min_effect_size=0.2,
        min_jd_per_window=args.min_jd,
        window_size=args.window_size,
        step=args.step,
    )

    top_n = args.top_n if args.top_n > 0 else None
    results = updater.run(args.annotated_dir, top_n=top_n)

    # 构建更新索引
    update_index = updater.build_update_index(results)

    # 导出图谱更新
    graph_updates = updater.export_graph_input(results)

    # 汇总报告
    report = {
        'generated_at': datetime.now().isoformat(),
        'total_jobs_analyzed': len(results),
        'total_changes': sum(len(r['changes']) for r in results),
        'total_graph_updates': len(graph_updates),
        'update_index_size': len(update_index),
        'config': {
            'alpha': updater.alpha,
            'min_effect_size': updater.min_effect_size,
            'min_jd_per_window': updater.min_jd_per_window,
            'window_size': updater.window_size,
            'step': updater.step,
        },
        'summary': {
            'jobs_with_changes': len([r for r in results if r['changes']]),
            'jobs_without_changes': len([r for r in results if not r['changes']]),
            'ai_emerging_jobs': len([r for r in results
                                     if any(c['skill_type'] == 'AI新兴技能'
                                            for c in r['changes'] if '新增' in c.get('category', ''))]),
            'obsolete_declining_jobs': len([r for r in results
                                            if any(c.get('is_obsolete')
                                                   for c in r['changes'])]),
        },
        'reports': results,
        'update_index': update_index,
        'graph_updates': graph_updates,
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f'\n[INFO] 报告: {args.output}')
    print(f'  更新索引: {len(update_index)} 条')
    print(f'  图谱更新: {len(graph_updates)} 条')

    # 单独输出图谱更新
    with open(args.graph_output, 'w', encoding='utf-8') as f:
        json.dump(graph_updates, f, ensure_ascii=False, indent=2)
    print(f'  图谱输入: {args.graph_output}')


if __name__ == '__main__':
    main()
