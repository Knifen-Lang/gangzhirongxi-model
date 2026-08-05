"""
Phase 3: Human Audit Panel + Graph Update Logic
================================================
For each new job candidate (lag window + emerging keywords),
generate an audit entry following the "four things" standard:
1. How discovered (data pipeline trace)
2. Why judged (scoring criteria)
3. Who confirms (expert domain assignment)
4. How graph changes after confirmation (what gets added/updated)
"""

import json
import sqlite3
from datetime import datetime
from collections import defaultdict

SQLITE_PATH = "outputs/outputs/tech_signals.sqlite"
PHASE1_JSON = "phase1_time_lag_report.json"
PHASE2_JSON = "phase2_tech_job_mapping.json"
OUTPUT_JSON = "phase3_audit_panel.json"
OUTPUT_MD = "phase3_audit_panel.md"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Expert domain mapping for each keyword
EXPERT_DOMAINS = {
    "diffusion transformers": "图像/视频生成、多模态模型",
    "self-evolving": "强化学习、自主智能体",
    "self-improving": "LLM训练优化、对齐",
    "training-free": "推理优化、prompt工程",
    "knowledge distillation": "模型压缩、边缘部署",
    "test-time adaptation": "推理时自适应、领域泛化",
    "llm-guided": "LLM辅助设计、自动化开发",
    "deep-research": "AI辅助科研、知识发现",
    "synthetic data": "数据增强、隐私保护",
    "godot-mcp": "游戏引擎MCP集成、工具链",
    "hermes-agent": "开源Agent框架、工具调用",
    "long-context": "长上下文处理、文档理解",
    "world models": "世界模型、强化学习",
    "ai-agent": "AI Agent框架、自主系统",
    "multi-agent systems": "多Agent协作、分布式AI",
    "MLLM": "多模态大模型、视觉语言",
    "moe": "混合专家架构、模型路由",
    "llm-driven": "LLM驱动开发、代码生成",
    "test-time": "推理时计算、动态推理",
}


def get_signal_evidence(keyword):
    """Pull sample evidence items from the signal_events table."""
    conn = sqlite3.connect(SQLITE_PATH)
    cur = conn.cursor()
    cur.execute(
        """SELECT title, url, source, published FROM signal_events
           WHERE keyword=? ORDER BY published DESC LIMIT 5""",
        (keyword,),
    )
    rows = cur.fetchall()
    conn.close()
    return [{"title": r[0], "url": r[1], "source": r[2], "published": r[3]} for r in rows]


def main():
    print("=" * 60)
    print("Phase 3: Human Audit Panel Generator")
    print("=" * 60)

    p1 = load_json(PHASE1_JSON)
    p2 = load_json(PHASE2_JSON)

    # Build combined view
    p1_map = {r["keyword"]: r for r in p1["keyword_details"]}
    p2_map = {r["keyword"]: r for r in p2["results"]}

    audit_entries = []

    for r2 in p2["results"]:
        kw = r2["keyword"]
        r1 = p1_map.get(kw, {})
        status = r2["status"]

        # Only audit non-converted keywords
        if status == "converted":
            continue

        # Get evidence
        evidence = get_signal_evidence(kw)

        # Build "four things" entry
        domain = EXPERT_DOMAINS.get(kw, "通用AI技术")

        # 1. How discovered
        discovery = build_discovery(kw, r1, r2)

        # 2. Why judged
        judgment = build_judgment(kw, r2, status)

        # 3. Who confirms
        reviewer = build_reviewer(kw, domain, r2)

        # 4. How graph changes
        graph_change = build_graph_change(kw, domain, r2, status)

        audit_entries.append({
            "keyword": kw,
            "status": status,
            "priority": "high" if status == "lag_window" else "medium",
            "domain": domain,
            "discovery": discovery,
            "judgment": judgment,
            "reviewer": reviewer,
            "graph_change": graph_change,
            "evidence": evidence[:3],  # Top 3 evidence items
        })

    # Sort: lag window first, then by growth ratio desc
    audit_entries.sort(key=lambda e: (
        0 if e["status"] == "lag_window" else 1,
        -(p2_map[e["keyword"]]["tech_trend"]["growth_ratio"] or 0)
    ))

    output = {
        "report": "Human Audit Panel for New Job Candidates",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "methodology": {
            "data_sources": [
                "phase1_time_lag_report.json (time lag verification)",
                "phase2_tech_job_mapping.json (co-occurrence analysis)",
                "tech_signals.sqlite → signal_events (evidence samples)",
            ],
            "audit_criteria": "Four-Things Standard",
        },
        "total_for_review": len(audit_entries),
        "high_priority": sum(1 for e in audit_entries if e["priority"] == "high"),
        "medium_priority": sum(1 for e in audit_entries if e["priority"] == "medium"),
        "entries": audit_entries,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"  JSON: {OUTPUT_JSON}")

    md = generate_md(output)
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"  MD:   {OUTPUT_MD}")

    print(f"\n  Total for review: {len(audit_entries)} entries")
    print(f"  High priority: {output['high_priority']} (lag window)")
    print(f"  Medium priority: {output['medium_priority']} (emerging)")


def build_discovery(kw, r1, r2):
    """How was this keyword discovered as a new-job signal?"""
    arxiv_first = r1.get("tech_arxiv_first_month", "-")
    github_first = r1.get("tech_github_first_month", "-")
    jd_count = r2.get("jd_stats", {}).get("total_jd_occurrences", 0)

    parts = [f"通过技术信号采集管道在 arXiv 和 GitHub 中追踪到关键词 '{kw}'"]

    if arxiv_first and github_first:
        parts.append(f"arXiv 首现 {arxiv_first}，GitHub 首现 {github_first}")
    elif arxiv_first:
        parts.append(f"arXiv 首现 {arxiv_first}")

    if jd_count == 0:
        parts.append("JD 数据库中尚未出现该词，判定为时滞窗口信号")
    else:
        parts.append(f"JD 数据库中出现 {jd_count} 次，判定为新兴信号")

    return "；".join(parts)


def build_judgment(kw, r2, status):
    """Why is this judged as a new-job signal?"""
    tt = r2.get("tech_trend", {})
    growth = tt.get("growth_ratio", 0) or 0
    arxiv_total = tt.get("total_arxiv", 0)

    reasons = []

    if status == "lag_window":
        reasons.append("JD 中零出现，但 arXiv/GitHub 存在持续活动")
    else:
        reasons.append(f"JD 中已出现 {r2.get('jd_stats', {}).get('total_jd_occurrences', 0)} 次，但尚未达到主流需求规模")

    if growth > 5:
        reasons.append(f"增长比率 {growth}x（近3月 vs 基线），呈爆发趋势")
    elif growth > 2:
        reasons.append(f"增长比率 {growth}x，呈上升趋势")

    if arxiv_total > 500:
        reasons.append(f"arXiv 累计 {arxiv_total} 篇论文，学术热度高")

    if r2.get("jd_stats", {}).get("new_job_candidates", 0) > 0:
        nc = r2["jd_stats"]["new_job_candidates"]
        reasons.append(f"已有 {nc} 个岗位被标注为'新岗位候选'")

    return "；".join(reasons) if reasons else "待人工判断"


def build_reviewer(kw, domain, r2):
    """Who should confirm this?"""
    cats = [c["category"] for c in (r2.get("job_categories") or [])[:3]]
    cat_str = "/".join(cats) if cats else "通用"

    return {
        "domain": domain,
        "suggested_reviewer_type": "AI/ML技术专家" if "AI" in cat_str else "行业技术专家",
        "related_job_categories": cat_str,
    }


def build_graph_change(kw, domain, r2, status):
    """What happens to the graph after confirmation?"""
    changes = {
        "add_tech_node": {
            "node_type": "Technology",
            "name": kw,
            "domain": domain,
            "source": "tech_signal_pipeline",
        },
        "actions": [],
    }

    if status == "lag_window":
        changes["actions"].append({
            "action": "create_predicted_job_node",
            "description": f"创建预测性岗位节点: '{kw}相关工程师'，标记为 'predicted' 状态",
            "activation_condition": f"当 JD 中 {kw} 的出现频率连续3个月 > 阈值时自动转为 'confirmed'",
        })
    else:
        top_jobs = (r2.get("top_job_titles") or [])[:3]
        for j in top_jobs:
            changes["actions"].append({
                "action": "link_to_existing_job",
                "description": f"关联已有岗位 '{j['title']}' ({j['count']}次)",
            })

    # Always add relationship edges
    changes["actions"].append({
        "action": "add_tech_job_edges",
        "description": f"创建 Technology->Job 关系边，权重 = arXiv计数×0.5 + GitHub计数×0.3 + JD计数×0.2",
    })

    return changes


def generate_md(output):
    md = f"""# Human Audit Panel: New Job Candidates

> Generated: {output['generated_at']}
> Total for review: **{output['total_for_review']}** entries
> High priority: **{output['high_priority']}** | Medium priority: **{output['medium_priority']}**

---

## Audit Standard: "Four Things" (四件事标准)

Each new job candidate must pass human review covering:

1. **怎么发现** — Data pipeline trace (which source, when first seen)
2. **为什么判断** — Scoring criteria (growth ratio, JD co-occurrence, burst signal)
3. **谁来确认** — Expert domain assignment (who has authority to confirm)
4. **确认后图谱怎么变** — Graph update actions (what nodes/edges get created)

---

## Priority Queue

"""
    for i, entry in enumerate(output["entries"]):
        prio_emoji = "🔴" if entry["priority"] == "high" else "🟡"
        status_label = "时滞窗口" if entry["status"] == "lag_window" else "新兴信号"

        md += f"""### {i + 1}. {prio_emoji} [{status_label}] {entry['keyword']}

**领域**: {entry['domain']}

**1. 怎么发现**:
{entry['discovery']}

**2. 为什么判断**:
{entry['judgment']}

**3. 谁来确认**:
- 建议审核人类型: {entry['reviewer']['suggested_reviewer_type']}
- 技术领域: {entry['reviewer']['domain']}
- 相关岗位类别: {entry['reviewer']['related_job_categories']}

**4. 确认后图谱变化**:
"""
        for action in entry['graph_change']['actions']:
            md += f"- **{action['action']}**: {action['description']}\n"

        if entry.get('evidence'):
            md += "\n**证据样本**:\n"
            for ev in entry['evidence']:
                md += f"- [{ev['source']}] {ev['title'][:100]}\n"

        md += "\n---\n\n"

    return md


if __name__ == "__main__":
    main()
