"""
Phase 1: Time Lag Verification
===============================
Compare tech keyword first appearances in arXiv/GitHub vs JD data.
Calculate personalized lag coefficients.
"""

import sqlite3
import csv
import json
import os
import glob
import re
from datetime import datetime, timedelta
from collections import defaultdict

# ============================================================
# Config
# ============================================================
SQLITE_PATH = "outputs/outputs/tech_signals.sqlite"
JD_DIR     = "【已标注】jd_v2"
OUTPUT_JSON = "phase1_time_lag_report.json"
OUTPUT_MD   = "phase1_time_lag_report.md"

KEYWORDS = [
    "agent", "rag", "mcp", "function calling", "moe", "rlhf",
    "MLLM", "multi-agent systems", "diffusion transformers", "world models",
    "self-evolving", "ai-agent", "self-improving", "training-free",
    "knowledge distillation", "test-time", "test-time adaptation",
    "long-context", "llm-guided", "llm-driven", "deep-research",
    "synthetic data", "godot-mcp", "hermes-agent",
]


def parse_jd_date(date_str):
    """
    Parse JD dates in various formats:
    - "5月22日更新" (Zhilian format, "month-day updated")
    - "2026-03-15" (standard)
    - "2025/07/01" (slashed)
    - "2025年7月" (Chinese year-month)
    Infers year from context (2025-2026 data collection window).
    """
    if not date_str:
        return None

    date_str = date_str.strip()

    # 1. Standard formats with explicit year
    for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%Y-%m", "%Y/%m"]:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    # 2. Chinese year-month-day: "2025年7月15日" or "2025年07月"
    m = re.match(r"(\d{4})年(\d{1,2})月(\d{1,2})日?", date_str)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3) or 1)
        return datetime(y, mo, d)

    # 3. Zhilian "M月D日更新" format (no year, need to infer)
    m = re.match(r"(\d{1,2})月(\d{1,2})日(?:更新)?", date_str)
    if m:
        mo, d = int(m.group(1)), int(m.group(2))
        # Infer year: data was collected 2025-2026
        # If month is 1-6, likely 2026; if 7-12, could be 2025
        # But we need a conservative approach: assign based on month
        if mo >= 7:
            y = 2025  # July+ = earliest collection (2025 H2)
        else:
            y = 2026  # Jan-Jun = later collection (2026 H1)
        return datetime(y, mo, d)

    # 4. "90天前发布" / "30天前发布" (Liepin relative format)
    m = re.match(r"(\d+)天前(?:发布|更新)", date_str)
    if m:
        days_ago = int(m.group(1))
        return datetime.now() - timedelta(days=days_ago)

    # 5. "今天更新" / "今天发布"
    if "今天" in date_str:
        return datetime.now()

    # 6. Just "M月更新" (no day)
    m = re.match(r"(\d{1,2})月(?:更新)?", date_str)
    if m:
        mo = int(m.group(1))
        y = 2025 if mo >= 7 else 2026
        return datetime(y, mo, 1)

    return None


# ============================================================
# Step 1: Tech signal first appearances from SQLite
# ============================================================
def get_tech_first_appearances():
    conn = sqlite3.connect(SQLITE_PATH)
    cur = conn.cursor()
    first_appear = {}
    for kw in KEYWORDS:
        first_appear[kw] = {"arxiv": None, "github": None}
        for source in ["arxiv", "github"]:
            cur.execute(
                """SELECT month FROM keyword_monthly_counts
                   WHERE keyword=? AND source=? AND count > 0
                   ORDER BY month ASC LIMIT 1""",
                (kw, source),
            )
            row = cur.fetchone()
            if row:
                first_appear[kw][source] = row[0]
    conn.close()
    return first_appear


# ============================================================
# Step 2: JD first appearances from CSV
# ============================================================
def get_jd_first_appearances():
    csv_files = glob.glob(os.path.join(JD_DIR, "*.csv"))
    print(f"  Scanning {len(csv_files)} JD CSV files...")

    jd_first = {}
    for kw in KEYWORDS:
        jd_first[kw] = {"first_date": None, "first_job": "", "first_source": "",
                         "total_occurrences": 0, "months_seen": set()}

    for fi, fpath in enumerate(csv_files):
        if (fi + 1) % 30 == 0:
            print(f"    Progress: {fi + 1}/{len(csv_files)}")

        try:
            with open(fpath, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    skills_text = (row.get("skill_requirements", "") or "").lower()
                    if not skills_text:
                        continue

                    issue_date = row.get("issue_date", "") or ""
                    job_name = row.get("job_name", "") or ""
                    source = row.get("source", "") or ""

                    for kw in KEYWORDS:
                        if kw.lower() in skills_text:
                            jd_first[kw]["total_occurrences"] += 1

                            if issue_date:
                                dt = parse_jd_date(issue_date.strip())
                                if dt:
                                    month_key = dt.strftime("%Y-%m")
                                    jd_first[kw]["months_seen"].add(month_key)
                                    date_key = dt.strftime("%Y-%m-%d")
                                    if (jd_first[kw]["first_date"] is None or
                                            date_key < jd_first[kw]["first_date"]):
                                        jd_first[kw]["first_date"] = date_key
                                        jd_first[kw]["first_job"] = job_name
                                        jd_first[kw]["first_source"] = source
        except Exception as e:
            print(f"    Warning: {os.path.basename(fpath)}: {e}")

    for kw in KEYWORDS:
        jd_first[kw]["months_seen"] = sorted(jd_first[kw]["months_seen"])

    return jd_first


# ============================================================
# Step 3: Compute lag
# ============================================================
def compute_lag(tech_first, jd_first):
    results = []
    for kw in KEYWORDS:
        tf = tech_first[kw]
        jf = jd_first[kw]

        tech_dates = []
        if tf["arxiv"]:
            tech_dates.append(("arxiv", tf["arxiv"]))
        if tf["github"]:
            tech_dates.append(("github", tf["github"]))
        earliest_tech = min(tech_dates, key=lambda x: x[1]) if tech_dates else (None, None)

        entry = {
            "keyword": kw,
            "tech_arxiv_first_month": tf["arxiv"],
            "tech_github_first_month": tf["github"],
            "tech_earliest_month": earliest_tech[1],
            "tech_earliest_source": earliest_tech[0],
            "jd_first_date": jf["first_date"],
            "jd_first_job": jf["first_job"],
            "jd_first_source": jf["first_source"],
            "jd_total_occurrences": jf["total_occurrences"],
            "jd_months_seen": len(jf["months_seen"]),
            "jd_month_span": jf["months_seen"][-1] if jf["months_seen"] else None,
        }

        lag_months = None
        lag_status = "no_jd_data"

        if jf["first_date"] and earliest_tech[1]:
            try:
                tech_dt = datetime.strptime(earliest_tech[1] + "-01", "%Y-%m-%d")
                jd_dt = datetime.strptime(jf["first_date"], "%Y-%m-%d")
                lag_months = round((jd_dt - tech_dt).days / 30.44, 1)
            except:
                pass
        elif jf["first_date"] and not earliest_tech[1]:
            lag_status = "jd_only"
        elif not jf["first_date"] and earliest_tech[1]:
            lag_status = "tech_only"
        else:
            lag_status = "no_data"

        entry["lag_months"] = lag_months
        entry["lag_status"] = lag_status

        # arXiv-specific lag
        if tf["arxiv"] and jf["first_date"]:
            try:
                td = datetime.strptime(tf["arxiv"] + "-01", "%Y-%m-%d")
                jd = datetime.strptime(jf["first_date"], "%Y-%m-%d")
                entry["lag_arxiv_to_jd_months"] = round((jd - td).days / 30.44, 1)
            except:
                entry["lag_arxiv_to_jd_months"] = None
        else:
            entry["lag_arxiv_to_jd_months"] = None

        # GitHub-specific lag
        if tf["github"] and jf["first_date"]:
            try:
                td = datetime.strptime(tf["github"] + "-01", "%Y-%m-%d")
                jd = datetime.strptime(jf["first_date"], "%Y-%m-%d")
                entry["lag_github_to_jd_months"] = round((jd - td).days / 30.44, 1)
            except:
                entry["lag_github_to_jd_months"] = None
        else:
            entry["lag_github_to_jd_months"] = None

        results.append(entry)

    return results


# ============================================================
# Step 4: Statistics
# ============================================================
def compute_stats(results):
    valid_lags = [r["lag_months"] for r in results
                   if r["lag_months"] is not None and r["lag_months"] >= 0]
    arxiv_lags = [r["lag_arxiv_to_jd_months"] for r in results
                   if r.get("lag_arxiv_to_jd_months") is not None and r["lag_arxiv_to_jd_months"] >= 0]
    github_lags = [r["lag_github_to_jd_months"] for r in results
                    if r.get("lag_github_to_jd_months") is not None and r["lag_github_to_jd_months"] >= 0]

    def pct(vals, p):
        if not vals:
            return None
        s = sorted(vals)
        idx = int(len(s) * p / 100)
        return s[min(idx, len(s) - 1)]

    return {
        "total_keywords": len(results),
        "keywords_with_jd_data": sum(1 for r in results if r["jd_first_date"]),
        "keywords_with_tech_data": sum(1 for r in results
                                        if r["tech_arxiv_first_month"] or r["tech_github_first_month"]),
        "keywords_with_both": sum(1 for r in results
                                   if r["jd_first_date"] and (r["tech_arxiv_first_month"] or r["tech_github_first_month"])),
        "tech_only_keywords": [r["keyword"] for r in results if r["lag_status"] == "tech_only"],
        "jd_only_keywords": [r["keyword"] for r in results if r["lag_status"] == "jd_only"],
        "lag_overall": {
            "mean_months": round(sum(valid_lags) / len(valid_lags), 1) if valid_lags else None,
            "median_months": pct(valid_lags, 50),
            "p25_months": pct(valid_lags, 25),
            "p75_months": pct(valid_lags, 75),
            "min_months": min(valid_lags) if valid_lags else None,
            "max_months": max(valid_lags) if valid_lags else None,
            "sample_size": len(valid_lags),
        },
        "lag_arxiv": {
            "mean_months": round(sum(arxiv_lags) / len(arxiv_lags), 1) if arxiv_lags else None,
            "median_months": pct(arxiv_lags, 50),
            "sample_size": len(arxiv_lags),
        },
        "lag_github": {
            "mean_months": round(sum(github_lags) / len(github_lags), 1) if github_lags else None,
            "median_months": pct(github_lags, 50),
            "sample_size": len(github_lags),
        },
    }


# ============================================================
# Markdown Report
# ============================================================
def generate_md(output):
    stats = output["summary_statistics"]
    results = output["keyword_details"]
    sorted_r = sorted(
        [r for r in results if r["lag_months"] is not None and r["lag_months"] >= 0],
        key=lambda r: r["lag_months"], reverse=True
    )

    md = f"""# Tech Keyword -> JD Time Lag Report

> Generated: {output['generated_at']}
> Sources: tech_signals.sqlite + 【已标注】jd_v2 (193,553 JDs)

---

## 1. Overall Lag Statistics

| Metric | arXiv->JD | GitHub->JD | Combined (earliest signal->JD) |
|--------|-----------|------------|-------------------------------|
| Sample size | {stats['lag_arxiv']['sample_size']} | {stats['lag_github']['sample_size']} | {stats['lag_overall']['sample_size']} |
| Mean (months) | {stats['lag_arxiv']['mean_months']} | {stats['lag_github']['mean_months']} | {stats['lag_overall']['mean_months']} |
| Median (months) | {stats['lag_arxiv']['median_months']} | {stats['lag_github']['median_months']} | {stats['lag_overall']['median_months']} |
| P25 (months) | - | - | {stats['lag_overall']['p25_months']} |
| P75 (months) | - | - | {stats['lag_overall']['p75_months']} |
| Range (months) | - | - | {stats['lag_overall']['min_months']} ~ {stats['lag_overall']['max_months']} |

> **Key finding**: Tech signals lead JD demand by approximately **{stats['lag_overall']['median_months']} months** (median).
> This is your personalized lag coefficient for future predictions.

---

## 2. Per-Keyword Details

| Keyword | arXiv First | GitHub First | JD First | Lag (mo) | JD Occurrences |
|---------|------------|-------------|---------|---------|---------------|
"""
    for r in sorted_r:
        lag = f"{r['lag_months']}mo"
        arx = r['tech_arxiv_first_month'] or '-'
        gh = r['tech_github_first_month'] or '-'
        jd = r['jd_first_date'] or '-'
        md += f"| {r['keyword']} | {arx} | {gh} | {jd} | {lag} | {r['jd_total_occurrences']} |\n"

    # Tech-only (lag window)
    md += "\n---\n\n## 3. Lag Windows (Tech signal exists, JD not yet)\n\n"
    tech_only = [r for r in results if r["lag_status"] == "tech_only"]
    if tech_only:
        for r in tech_only:
            md += f"- **{r['keyword']}**: arXiv {r['tech_arxiv_first_month'] or '-'}, GitHub {r['tech_github_first_month'] or '-'}\n"
    else:
        md += "_(none)_\n"

    # JD-only
    md += "\n## 4. JD-Only Signals (not from arXiv/GitHub)\n\n"
    jd_only = [r for r in results if r["lag_status"] == "jd_only"]
    if jd_only:
        for r in jd_only:
            md += f"- **{r['keyword']}**: JD first {r['jd_first_date']} ({r['jd_first_job']})\n"
    else:
        md += "_(none)_\n"

    # Top lag keywords
    md += "\n## 5. Highest Lag (Best Early Warning)\n\n"
    for r in sorted_r[:5]:
        md += f"- **{r['keyword']}**: {r['lag_months']}mo -- signal {r['tech_earliest_month']} -> JD {r['jd_first_date']}\n"

    md += f"""
---

## 6. Recommended Lag Coefficients

Based on {stats['lag_overall']['sample_size']} keywords with complete data:

| Scenario | Coefficient |
|----------|------------|
| **Conservative** (P75) | {stats['lag_overall']['p75_months']} months |
| **Median** (best estimate) | {stats['lag_overall']['median_months']} months |
| **Aggressive** (P25) | {stats['lag_overall']['p25_months']} months |

> When arXiv/GitHub first detects a tech keyword inflection point,
> expect observable JD market demand in ~**{stats['lag_overall']['median_months']} months**.
"""
    return md


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("Phase 1: Tech Keyword Time Lag Verification")
    print("=" * 60)

    print("\n[1/4] Extracting tech signal first appearances from SQLite...")
    tech_first = get_tech_first_appearances()
    kw_tech = sum(1 for k in KEYWORDS if tech_first[k]["arxiv"] or tech_first[k]["github"])
    print(f"  Done. {kw_tech}/{len(KEYWORDS)} keywords have tech signal data.")

    print("\n[2/4] Scanning JD CSVs for keyword first appearances...")
    jd_first = get_jd_first_appearances()
    kw_jd = sum(1 for k in KEYWORDS if jd_first[k]["first_date"])
    total_occ = sum(jd_first[k]["total_occurrences"] for k in KEYWORDS)
    print(f"  Done. {kw_jd}/{len(KEYWORDS)} keywords found in JDs, {total_occ} total occurrences.")

    print("\n[3/4] Computing time lags...")
    results = compute_lag(tech_first, jd_first)
    stats = compute_stats(results)

    print("\n[4/4] Generating reports...")
    output = {
        "report": "Tech Keyword -> JD Time Lag Verification",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "methodology": {
            "tech_data_source": "outputs/outputs/tech_signals.sqlite (keyword_monthly_counts)",
            "jd_data_source": "【已标注】jd_v2/ (193,553 annotated JDs)",
            "keywords_tracked": len(KEYWORDS),
            "lag_calculation": "days between earliest tech month-end and JD first-date, divided by 30.44",
        },
        "summary_statistics": stats,
        "keyword_details": results,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"  JSON: {OUTPUT_JSON}")

    md = generate_md(output)
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"  Markdown: {OUTPUT_MD}")

    print("\n" + "=" * 60)
    print("KEY FINDINGS:")
    print(f"  Overall lag median:  {stats['lag_overall']['median_months']} months")
    print(f"  Overall lag mean:    {stats['lag_overall']['mean_months']} months")
    print(f"  Sample size:         {stats['lag_overall']['sample_size']} keywords")
    print(f"  Tech-only (lag window): {stats['tech_only_keywords']}")
    print(f"  JD-only (other channels): {stats['jd_only_keywords']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
