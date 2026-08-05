"""
Model Feature Generator
=======================
Generates features for the ML training pipeline (Subsystem B) from
tech signal data + JD annotations. Outputs a feature matrix ready
for model consumption.

Features generated:
1. BURST_SCORE      — multi-source weighted burst probability (0-1)
2. TREND_VECTOR     — 3-month rolling trend (up/flat/down)
3. LAG_COEFFICIENT   — personalized tech→JD lag in months
4. JD_PENETRATION    — keyword penetration rate in JD corpus
5. COHEN_H          — effect size of skill frequency change

Output:
- model_features.json   — per-keyword feature vectors
- model_features.csv    — flat table for pandas/training
"""

import sqlite3
import json
import csv
import math
import glob
import os
from collections import defaultdict
from datetime import datetime

# ============================================================
# Config
# ============================================================
SQLITE_PATH = "outputs/outputs/tech_signals.sqlite"
JD_DIR = "【已标注】jd_v2"
PHASE1_PATH = "phase1_time_lag_report.json"

KEYWORDS = [
    "agent", "rag", "mcp", "function calling", "moe", "rlhf",
    "MLLM", "multi-agent systems", "diffusion transformers", "world models",
    "self-evolving", "ai-agent", "self-improving", "training-free",
    "knowledge distillation", "test-time", "test-time adaptation",
    "long-context", "llm-guided", "llm-driven", "deep-research",
    "synthetic data", "godot-mcp", "hermes-agent",
]

# Source weights for burst formula
W_ARXIV = 0.5   # Research papers
W_GITHUB = 0.3  # Open-source activity
W_JD = 0.2      # Job market demand


# ============================================================
# 1. Data Loading
# ============================================================
def load_monthly_series():
    """Load keyword_monthly_counts into {keyword: {month: {arxiv, github}}}"""
    conn = sqlite3.connect(SQLITE_PATH)
    cur = conn.cursor()
    series = defaultdict(lambda: defaultdict(lambda: {"arxiv": 0, "github": 0}))
    cur.execute("SELECT keyword, month, source, count FROM keyword_monthly_counts")
    for kw, month, src, cnt in cur.fetchall():
        series[kw][month][src] = cnt
    conn.close()
    return series


def load_jd_counts():
    """Count keyword occurrences in JD corpus. Returns {keyword: count}"""
    counts = defaultdict(int)
    csv_files = glob.glob(os.path.join(JD_DIR, "*.csv"))
    for fpath in csv_files:
        try:
            with open(fpath, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    text = (row.get("skill_requirements", "") or "").lower()
                    for kw in KEYWORDS:
                        if kw.lower() in text:
                            counts[kw] += 1
        except Exception:
            pass
    return dict(counts)


def load_lag_data():
    """Load Phase 1 lag coefficients"""
    with open(PHASE1_PATH, "r", encoding="utf-8") as f:
        p1 = json.load(f)
    return {r["keyword"]: r for r in p1["keyword_details"]}


# ============================================================
# 2. Burst Score Formula
# ============================================================
def compute_burst_score(keyword, series, jd_count, total_jds=193553):
    """
    Multi-source weighted burst score.

    Formula:
      burst = tanh(normalized_growth) × persistence × source_coherence

    Components:
      - normalized_growth: (recent_3m_avg / max(baseline_avg, epsilon)) - 1
      - persistence: min(months_active / 6, 1.0)  ← at least 6 months
      - source_coherence: 1.0 if all sources agree, <1 if divergent

    Returns float in [0, 1], higher = stronger burst signal.
    """
    months = sorted(series.keys())
    if len(months) < 4:
        return 0.0, {"error": "insufficient_data"}

    # Recent 3 months vs baseline (everything before)
    recent_months = months[-3:]
    baseline_months = months[:-3] if len(months) > 3 else months[:1]

    def monthly_total(m):
        return series[m]["arxiv"] + series[m]["github"]

    recent_vals = [monthly_total(m) for m in recent_months]
    baseline_vals = [monthly_total(m) for m in baseline_months]

    recent_avg = sum(recent_vals) / len(recent_vals)
    baseline_avg = sum(baseline_vals) / len(baseline_vals) if baseline_vals else 1.0
    epsilon = 1.0

    # --- Component 1: Normalized Growth ---
    # Avoid divide-by-zero and handle baseline=0 case
    if baseline_avg < epsilon:
        # New term: baseline was ~0, now emerging
        normalized_growth = math.log2(max(recent_avg, 1) + 1)  # log-scale for new terms
    else:
        raw_growth = recent_avg / baseline_avg
        normalized_growth = raw_growth - 1.0  # e.g. 2.0x → 1.0 growth

    # Squash to [0, 3] range via tanh
    growth_score = math.tanh(max(normalized_growth, 0) / 2.0)  # /2 to spread sigmoid

    # --- Component 2: Persistence ---
    active_months = sum(1 for m in months if monthly_total(m) > 0)
    persistence = min(active_months / 6.0, 1.0)

    # --- Component 3: Source Divergence ---
    # arXiv-heavy vs GitHub-heavy signals mean different things
    arxiv_recent = sum(series[m]["arxiv"] for m in recent_months)
    github_recent = sum(series[m]["github"] for m in recent_months)
    total_recent = arxiv_recent + github_recent + 1

    arxiv_share = arxiv_recent / total_recent
    github_share = github_recent / total_recent

    # JD penetration rate
    jd_penetration = min(jd_count / max(total_jds * 0.001, 1), 1.0)

    # Multi-source weighted burst
    weighted_burst = (
        W_ARXIV * arxiv_share * growth_score +
        W_GITHUB * github_share * growth_score +
        W_JD * jd_penetration
    )

    # Final burst score: weighted × persistence
    burst_score = round(weighted_burst * persistence, 4)

    # Clamp to [0, 1]
    burst_score = max(0.0, min(1.0, burst_score))

    components = {
        "recent_3m_avg": round(recent_avg, 1),
        "baseline_avg": round(baseline_avg, 1),
        "growth_ratio": round(recent_avg / max(baseline_avg, 1), 2),
        "normalized_growth": round(normalized_growth, 2),
        "growth_score": round(growth_score, 3),
        "persistence": round(persistence, 2),
        "arxiv_share": round(arxiv_share, 2),
        "github_share": round(github_share, 2),
        "jd_penetration": round(jd_penetration, 4),
        "weighted_burst": round(weighted_burst, 3),
        "active_months": active_months,
    }

    return burst_score, components


# ============================================================
# 3. Dynamic Evolution Features
# ============================================================
def compute_trend_vector(keyword, series):
    """
    Compute 3-month rolling trend vector for dynamic evolution analysis.

    Returns:
    - trend_direction: "strong_rise" | "rise" | "stable" | "decline" | "strong_decline"
    - monthly_series: list of (month, total) for time-series modeling
    - acceleration: 2nd derivative (is growth accelerating or decelerating?)
    """
    months = sorted(series.keys())
    totals = [series[m]["arxiv"] + series[m]["github"] for m in months]

    if len(totals) < 4:
        return {"trend_direction": "insufficient_data", "monthly_series": list(zip(months, totals))}

    # 3-month rolling averages
    rolling = []
    for i in range(2, len(totals)):
        rolling.append(sum(totals[i - 2:i + 1]) / 3.0)

    # Linear regression on rolling averages to get slope
    n = len(rolling)
    if n < 2:
        return {"trend_direction": "insufficient_data", "monthly_series": list(zip(months, totals))}

    x_mean = (n - 1) / 2.0
    y_mean = sum(rolling) / n
    numerator = sum((i - x_mean) * (rolling[i] - y_mean) for i in range(n))
    denominator = sum((i - x_mean) ** 2 for i in range(n))

    if denominator == 0:
        slope = 0
    else:
        slope = numerator / denominator

    # Normalize slope relative to mean
    relative_slope = slope / max(y_mean, 1.0)

    # Acceleration (2nd derivative)
    if n >= 3:
        first_diff = [rolling[i + 1] - rolling[i] for i in range(n - 1)]
        accel = first_diff[-1] - first_diff[0] if len(first_diff) >= 2 else 0
        accel_norm = accel / max(y_mean, 1.0)
    else:
        accel_norm = 0.0

    # Classify
    if relative_slope > 0.15:
        direction = "strong_rise"
    elif relative_slope > 0.05:
        direction = "rise"
    elif relative_slope > -0.05:
        direction = "stable"
    elif relative_slope > -0.15:
        direction = "decline"
    else:
        direction = "strong_decline"

    return {
        "trend_direction": direction,
        "relative_slope": round(relative_slope, 4),
        "acceleration": round(accel_norm, 4),
        "rolling_averages": [round(v, 1) for v in rolling],
        "latest_3m_avg": round(rolling[-1], 1) if rolling else 0,
        "monthly_series": [(m, t) for m, t in zip(months, totals)],
    }


# ============================================================
# 4. Cohen's h Effect Size (Quarter-over-Quarter)
# ============================================================
def compute_cohens_h(series, q1_months, q2_months):
    """Compute Cohen's h between two quarter periods for a keyword."""
    q1_total = sum(series[m]["arxiv"] + series[m]["github"]
                   for m in q1_months if m in series)
    q2_total = sum(series[m]["arxiv"] + series[m]["github"]
                   for m in q2_months if m in series)

    # Use total as denominator for proportion
    total = q1_total + q2_total + 1

    p1 = q1_total / total
    p2 = q2_total / total

    h = 2 * abs(math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))

    label = "small" if h < 0.2 else ("medium" if h < 0.5 else "large")
    return round(h, 3), label


# ============================================================
# 5. Full Feature Matrix
# ============================================================
def build_feature_matrix():
    """Generate complete feature matrix for all 24 keywords."""
    print("Loading data...")
    series = load_monthly_series()
    jd_counts = load_jd_counts()
    lag_data = load_lag_data()

    all_months = set()
    for kw_series in series.values():
        all_months.update(kw_series.keys())
    all_months = sorted(all_months)

    # Split into quarters
    q_months = defaultdict(list)
    for m in all_months:
        yr = int(m[:4])
        mo = int(m[5:7])
        q = f"{yr}Q{(mo - 1) // 3 + 1}"
        q_months[q].append(m)
    quarters = sorted(q_months.keys())

    features = []

    for kw in KEYWORDS:
        kw_series = series.get(kw, {})
        jd_count = jd_counts.get(kw, 0)
        lag = lag_data.get(kw, {})

        # 1. BURST_SCORE
        burst_score, burst_components = compute_burst_score(kw, kw_series, jd_count)

        # 2. TREND_VECTOR
        trend = compute_trend_vector(kw, kw_series)

        # 3. LAG_COEFFICIENT
        lag_months = lag.get("lag_months")
        lag_status = lag.get("lag_status", "unknown")

        # 4. JD_PENETRATION
        jd_penetration = round(jd_count / 193553, 6)

        # 5. COHEN_H (latest 2 quarters if available)
        cohens_h = None
        if len(quarters) >= 2:
            q1, q2 = quarters[-2], quarters[-1]
            cohens_h, h_label = compute_cohens_h(kw_series, q_months[q1], q_months[q2])
        else:
            h_label = "n/a"

        # 6. Classification
        if jd_count > 50:
            market_status = "converted"
        elif jd_count > 0:
            market_status = "emerging"
        else:
            market_status = "lag_window"

        # Build feature vector
        feat = {
            # Identifier
            "keyword": kw,

            # Target features for model training
            "burst_score": burst_score,                      # [0,1] continuous
            "trend_direction": trend["trend_direction"],      # categorical: strong_rise/rise/stable/decline/strong_decline
            "trend_slope": trend["relative_slope"],           # continuous
            "trend_acceleration": trend["acceleration"],      # continuous
            "lag_months": lag_months or -1,                   # continuous, -1 = unknown
            "jd_penetration": jd_penetration,                 # [0,1] continuous
            "cohens_h": cohens_h or 0,                        # [0, π] continuous
            "market_status": market_status,                   # label: converted/emerging/lag_window

            # Auxiliary info
            "jd_count": jd_count,
            "growth_ratio": burst_components.get("growth_ratio", 0),
            "active_months": burst_components.get("active_months", 0),
            "latest_3m_avg": trend["latest_3m_avg"],

            # Decomposition
            "burst_components": burst_components,
            "trend_details": trend,
            "lag_details": {
                "status": lag_status,
                "arxiv_first": lag.get("tech_arxiv_first_month"),
                "github_first": lag.get("tech_github_first_month"),
                "jd_first": lag.get("jd_first_date"),
            },
        }
        features.append(feat)

    # Sort by burst_score descending
    features.sort(key=lambda f: f["burst_score"], reverse=True)

    return features


# ============================================================
# 6. Output
# ============================================================
def export_json(features):
    output = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "feature_schema": {
            "burst_score": "float [0,1] — multi-source weighted burst probability",
            "trend_direction": "categorical — strong_rise/rise/stable/decline/strong_decline",
            "trend_slope": "float — normalized slope of 3-month rolling avg",
            "trend_acceleration": "float — 2nd derivative (growth accelerating?)",
            "lag_months": "float — estimated tech→JD lag in months, -1=unknown",
            "jd_penetration": "float [0,1] — fraction of JDs containing keyword",
            "cohens_h": "float [0,pi] — effect size between latest 2 quarters",
            "market_status": "label — converted/emerging/lag_window",
        },
        "summary": {
            "total_keywords": len(features),
            "mean_burst": round(sum(f["burst_score"] for f in features) / len(features), 3),
            "top5_by_burst": [f["keyword"] for f in features[:5]],
            "market_distribution": {
                s: sum(1 for f in features if f["market_status"] == s)
                for s in ["converted", "emerging", "lag_window"]
            },
        },
        "features": features,
    }

    with open("model_features.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print("  model_features.json")


def export_csv(features):
    """Flat CSV for pandas training."""
    rows = []
    for feat in features:
        rows.append({
            "keyword": feat["keyword"],
            "burst_score": feat["burst_score"],
            "trend_direction": feat["trend_direction"],
            "trend_slope": feat["trend_slope"],
            "trend_acceleration": feat["trend_acceleration"],
            "lag_months": feat["lag_months"],
            "jd_penetration": feat["jd_penetration"],
            "cohens_h": feat["cohens_h"],
            "market_status": feat["market_status"],
            "jd_count": feat["jd_count"],
            "growth_ratio": feat["growth_ratio"],
        })

    with open("model_features.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print("  model_features.csv")


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("Model Feature Generator")
    print("=" * 60)

    features = build_feature_matrix()

    export_json(features)
    export_csv(features)

    print(f"\n  Keywords: {len(features)}")
    print(f"  Mean burst: {sum(f['burst_score'] for f in features)/len(features):.3f}")
    print(f"  Top 5 by burst:")
    for f in features[:5]:
        direction_mark = {"strong_rise": "[UP]", "rise": "[up]", "stable": "[--]", "decline": "[dn]", "strong_decline": "[DN]"}
        mark = direction_mark.get(f["trend_direction"], "[??]")
        print(f"    {mark} {f['keyword']:30s} burst={f['burst_score']:.3f}  trend={f['trend_direction']:15s}  status={f['market_status']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
