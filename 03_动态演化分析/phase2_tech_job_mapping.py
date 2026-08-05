"""
Phase 2: Tech -> Job Candidate Mapping (Co-occurrence Analysis)
================================================================
For each tech keyword, find:
1. Co-occurring required skills in JDs
2. Associated job categories
3. Which tech signals are already "converted" to JD demand vs still in lag window
"""

import csv
import json
import glob
import os
import sqlite3
import re
from collections import defaultdict, Counter
from datetime import datetime

# ============================================================
# Config
# ============================================================
JD_DIR = "【已标注】jd_v2"
SQLITE_PATH = "outputs/outputs/tech_signals.sqlite"
OUTPUT_JSON = "phase2_tech_job_mapping.json"
OUTPUT_MD = "phase2_tech_job_mapping.md"

KEYWORDS = [
    "agent", "rag", "mcp", "function calling", "moe", "rlhf",
    "MLLM", "multi-agent systems", "diffusion transformers", "world models",
    "self-evolving", "ai-agent", "self-improving", "training-free",
    "knowledge distillation", "test-time", "test-time adaptation",
    "long-context", "llm-guided", "llm-driven", "deep-research",
    "synthetic data", "godot-mcp", "hermes-agent",
]


def parse_skills(skill_str):
    """Parse 'skill1|cat|level; skill2|cat|level' into list of skill names."""
    if not skill_str or skill_str in ("无", "[]", ""):
        return []
    skills = []
    for part in skill_str.split(";"):
        part = part.strip().strip("【】")
        if not part:
            continue
        pieces = part.split("|")
        if pieces:
            name = pieces[0].strip()
            if name and name != "无":
                skills.append(name)
    return skills


def classify_job_category(job_name):
    """Simple keyword-based job category classification."""
    jn = (job_name or "").lower()
    rules = [
        ("AI/ML Engineer", ["算法", "机器学习", "深度学习", "自然语言", "nlp", "cv", "推荐算法",
                            "ai", "人工智能", "大模型", "llm", "aigc", "agent"]),
        ("Data Engineer", ["数据工程", "etl", "数据仓库", "数据管道", "data engineer"]),
        ("Data Analyst", ["数据分析", "商业分析", "bi", "数据运营", "数据产品"]),
        ("Data Scientist", ["数据科学", "数据挖掘", "风控模型", "量化"]),
        ("Backend Developer", ["后端", "java", "golang", "服务端", "后台开发"]),
        ("Frontend Developer", ["前端", "web", "h5", "小程序", "flutter", "react", "vue"]),
        ("Full Stack", ["全栈", "full stack", "fullstack"]),
        ("DevOps/SRE", ["运维", "devops", "sre", "云原生", "k8s", "kubernetes"]),
        ("Embedded/Hardware", ["嵌入式", "硬件", "驱动", "单片机", "mcu", "fpga"]),
        ("Security", ["安全", "渗透", "soc", "security"]),
        ("QA/Test", ["测试", "qa", "质量"]),
        ("Product Manager", ["产品经理", "产品总监", "pm"]),
        ("Tech Lead/Manager", ["技术经理", "技术总监", "cto", "架构师", "tech lead"]),
        ("LLM/Agent Specialist", ["大模型", "llm", "agent", "rag", "prompt", "langchain"]),
        ("Robotics/Autonomous", ["机器人", "自动驾驶", "感知", "规划控制", "slam", "ros"]),
        ("Other", []),
    ]
    scores = []
    for cat, keywords in rules:
        score = sum(1 for k in keywords if k in jn)
        if score > 0:
            scores.append((cat, score))
    if scores:
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0]
    return "Other"


# ============================================================
# Main Analysis
# ============================================================
def main():
    print("=" * 60)
    print("Phase 2: Tech -> Job Candidate Mapping")
    print("=" * 60)

    # ---- Load tech signal growth data ----
    print("\n[1/4] Loading tech signal trends from SQLite...")
    conn = sqlite3.connect(SQLITE_PATH)
    cur = conn.cursor()

    # Get latest 3-month average and previous baseline for each keyword
    tech_trends = {}
    for kw in KEYWORDS:
        cur.execute("""
            SELECT month, source, count FROM keyword_monthly_counts
            WHERE keyword=? ORDER BY month ASC
        """, (kw,))
        rows = cur.fetchall()
        months_data = defaultdict(lambda: {"arxiv": 0, "github": 0})
        for month, source, count in rows:
            months_data[month][source] = count

        sorted_months = sorted(months_data.keys())
        if len(sorted_months) >= 6:
            recent_3 = sorted_months[-3:]
            baseline = sorted_months[:6]  # first 6 months
            recent_total = sum(months_data[m]["arxiv"] + months_data[m]["github"] for m in recent_3)
            baseline_total = sum(months_data[m]["arxiv"] + months_data[m]["github"] for m in baseline)
            baseline_avg = baseline_total / len(baseline) if baseline else 1
            growth_ratio = round(recent_total / baseline_avg, 2) if baseline_avg > 0 else None
        else:
            growth_ratio = None

        total_arxiv = sum(v["arxiv"] for v in months_data.values())
        total_github = sum(v["github"] for v in months_data.values())

        tech_trends[kw] = {
            "total_arxiv": total_arxiv,
            "total_github": total_github,
            "growth_ratio": growth_ratio,
            "months_tracked": len(sorted_months),
        }
    conn.close()

    # ---- Scan JDs for co-occurrence ----
    print("\n[2/4] Scanning JDs for keyword-skill co-occurrence...")
    csv_files = glob.glob(os.path.join(JD_DIR, "*.csv"))

    # Per-keyword aggregates
    kw_data = {}
    for kw in KEYWORDS:
        kw_data[kw] = {
            "jd_count": 0,
            "co_skills": Counter(),        # co-occurring required skills
            "co_bonus_skills": Counter(),  # co-occurring bonus skills
            "co_sig_skills": Counter(),    # co-occurring signature skills
            "job_categories": Counter(),
            "job_names": Counter(),
            "new_job_candidates": 0,
            "capability_updates": 0,
            "companies": Counter(),
        }

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

                    job_name = row.get("job_name", "") or ""
                    company = row.get("company_name", "") or ""
                    req_skills = parse_skills(row.get("必备技能", "") or "")
                    bonus_skills = parse_skills(row.get("加分技能", "") or "")
                    sig_skills = parse_skills(row.get("特征技能", "") or "")
                    is_new = (row.get("是否新岗位候选", "") or "").strip() == "是"
                    cap_update = "新增AI技能" in (row.get("能力更新", "") or "")

                    for kw in KEYWORDS:
                        if kw.lower() not in skills_text:
                            continue

                        kd = kw_data[kw]
                        kd["jd_count"] += 1
                        for s in req_skills:
                            kd["co_skills"][s] += 1
                        for s in bonus_skills:
                            kd["co_bonus_skills"][s] += 1
                        for s in sig_skills:
                            kd["co_sig_skills"][s] += 1
                        if is_new:
                            kd["new_job_candidates"] += 1
                        if cap_update:
                            kd["capability_updates"] += 1

                        cat = classify_job_category(job_name)
                        kd["job_categories"][cat] += 1
                        kd["job_names"][job_name] += 1
                        if company:
                            kd["companies"][company] += 1
        except Exception as e:
            print(f"    Warning: {os.path.basename(fpath)}: {e}")

    # ---- Consolidate results ----
    print("\n[3/4] Consolidating results...")
    results = []
    for kw in KEYWORDS:
        kd = kw_data[kw]
        tt = tech_trends[kw]

        top_skills = kd["co_skills"].most_common(10)
        top_cats = kd["job_categories"].most_common(5)
        top_jobs = kd["job_names"].most_common(5)
        top_companies = kd["companies"].most_common(5)

        # Categorize status
        if kd["jd_count"] > 50:
            status = "converted"  # Already strong JD demand
        elif kd["jd_count"] > 0:
            status = "emerging"   # Just starting to appear
        else:
            status = "lag_window" # Not yet in JDs

        results.append({
            "keyword": kw,
            "status": status,
            "tech_trend": tt,
            "jd_stats": {
                "total_jd_occurrences": kd["jd_count"],
                "new_job_candidates": kd["new_job_candidates"],
                "capability_updates": kd["capability_updates"],
            },
            "top_co_skills": [{"skill": s, "count": c} for s, c in top_skills],
            "top_co_bonus_skills": [{"skill": s, "count": c} for s, c in kd["co_bonus_skills"].most_common(10)],
            "top_co_sig_skills": [{"skill": s, "count": c} for s, c in kd["co_sig_skills"].most_common(5)],
            "job_categories": [{"category": c, "count": n} for c, n in top_cats],
            "top_job_titles": [{"title": t, "count": n} for t, n in top_jobs],
            "top_companies": [{"company": c, "count": n} for c, n in top_companies],
        })

    # ---- Generate output ----
    print("\n[4/4] Generating reports...")
    output = {
        "report": "Tech -> Job Candidate Mapping",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "total_keywords": len(KEYWORDS),
            "converted": sum(1 for r in results if r["status"] == "converted"),
            "emerging": sum(1 for r in results if r["status"] == "emerging"),
            "lag_window": sum(1 for r in results if r["status"] == "lag_window"),
        },
        "results": results,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # Markdown report
    md = generate_md(output)
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(md)

    # Print summary
    s = output["summary"]
    print(f"\n  JSON: {OUTPUT_JSON}")
    print(f"  MD:   {OUTPUT_MD}")
    print(f"\n  Converted:  {s['converted']} keywords (already in JD demand)")
    print(f"  Emerging:   {s['emerging']} keywords (starting to appear)")
    print(f"  Lag window: {s['lag_window']} keywords (tech-only, prediction opportunity)")


def generate_md(output):
    md = f"""# Tech -> Job Candidate Mapping Report

> Generated: {output['generated_at']}

---

## Summary

| Status | Count | Keywords |
|--------|-------|---------|
"""
    converted = [r for r in output["results"] if r["status"] == "converted"]
    emerging = [r for r in output["results"] if r["status"] == "emerging"]
    lag = [r for r in output["results"] if r["status"] == "lag_window"]

    md += f"| Converted | {len(converted)} | {', '.join(r['keyword'] for r in converted)} |\n"
    md += f"| Emerging | {len(emerging)} | {', '.join(r['keyword'] for r in emerging)} |\n"
    md += f"| Lag Window | {len(lag)} | {', '.join(r['keyword'] for r in lag)} |\n"

    # Detail for lag window keywords
    md += "\n---\n\n## Lag Window (Prediction Opportunity)\n\n"
    md += "These keywords appear in arXiv/GitHub but have **zero JD mentions**. They represent the strongest forward-looking signals.\n\n"
    md += "| Keyword | Tech Trend (arXiv total) | Growth Ratio |\n"
    md += "|---------|------------------------|-------------|\n"
    for r in lag:
        tt = r["tech_trend"]
        gr = f"{tt['growth_ratio']}x" if tt['growth_ratio'] else '-'
        md += f"| {r['keyword']} | {tt['total_arxiv']} | {gr} |\n"

    # Detail for converted keywords
    md += "\n---\n\n## Converted (Active JD Demand)\n\n"
    for r in converted:
        md += f"### {r['keyword']} ({r['jd_stats']['total_jd_occurrences']} JD mentions)\n\n"
        md += f"- New job candidates: {r['jd_stats']['new_job_candidates']}\n"
        md += f"- Capability updates: {r['jd_stats']['capability_updates']}\n"

        if r["top_co_skills"]:
            skills_str = ', '.join(f'{s["skill"]}({s["count"]})' for s in r['top_co_skills'][:5])
            md += f"- Top co-skills: {skills_str}\n"
        if r["job_categories"]:
            cats_str = ', '.join(f'{c["category"]}({c["count"]})' for c in r['job_categories'][:3])
            md += f"- Job categories: {cats_str}\n"
        if r["top_job_titles"]:
            titles_str = ', '.join(f'{t["title"]}({t["count"]})' for t in r['top_job_titles'][:3])
            md += f"- Top titles: {titles_str}\n"
        md += "\n"

    # Detail for emerging keywords
    md += "\n---\n\n## Emerging (Starting to Appear)\n\n"
    for r in emerging:
        md += f"### {r['keyword']} ({r['jd_stats']['total_jd_occurrences']} JD mentions)\n\n"
        if r["top_job_titles"]:
            etitles = ', '.join(f'{t["title"]}({t["count"]})' for t in r['top_job_titles'][:3])
            md += f"- Top titles: {etitles}\n"
        if r["job_categories"]:
            ecats = ', '.join(f'{c["category"]}({c["count"]})' for c in r['job_categories'][:3])
            md += f"- Categories: {ecats}\n"
        md += "\n"

    return md


if __name__ == "__main__":
    main()
