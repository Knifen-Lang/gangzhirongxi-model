"""
通胀 + 幻觉 深度修复
=====================

通胀: 薪资分层验证 — 真实需求词在高薪JD中比例更高
幻觉: 对接现有 HallucinationGuard + RAG 三源验证
"""

import re
import json
import glob
import csv
import sqlite3
import sys
import os
from collections import defaultdict, Counter
from datetime import datetime

# 接入现有管线
sys.path.insert(0, os.path.dirname(__file__))
from hallucination_guard import HallucinationGuard, TECH_KNOWLEDGE_BASE
from rag_retriever import RAGRetriever

# ============================================================
# 1. 通胀修复 —— 薪资分层验证
# ============================================================

def parse_salary(salary_str):
    """
    解析薪资字符串，返回 (min_k, max_k) 千元/月。
    支持格式: "15k-25k", "15000-25000", "1.5万-2.5万", "15-25K",
              "20-40K·15薪", "面议"
    """
    if not salary_str or salary_str == '面议':
        return None

    s = str(salary_str).strip().lower().replace('k', '000').replace('K', '000')

    # "1.5万-2.5万"
    m = re.search(r'([\d.]+)万\s*[-~至]\s*([\d.]+)万', salary_str)
    if m:
        lo = float(m.group(1)) * 10000
        hi = float(m.group(2)) * 10000
        return (lo / 1000, hi / 1000)

    # 纯数字范围
    m = re.search(r'([\d.]+)\s*[-~至]\s*([\d.]+)', s)
    if m:
        lo = float(m.group(1))
        hi = float(m.group(2))
        # 判断单位: 如果 > 100 则为元/月, 除以1000
        if lo > 100:
            lo /= 1000
            hi /= 1000
        # 如果 < 3 则为 万/月 或 万元/年, 除以12
        if lo < 3 and hi < 10:
            pass  # 可能就是千元范围，不动
        return (lo, hi)

    # 单一数字
    m = re.search(r'([\d.]+)', s)
    if m:
        v = float(m.group(1))
        if v > 100:
            v /= 1000
        return (v, v)

    return None


def salary_level(salary_str):
    """返回薪资层级: 'high' / 'mid' / 'low' / 'unknown'"""
    parsed = parse_salary(salary_str)
    if parsed is None:
        return 'unknown'
    mid = (parsed[0] + parsed[1]) / 2
    if mid >= 25:   # >= 25K/月
        return 'high'
    elif mid >= 12:  # 12-25K/月
        return 'mid'
    else:
        return 'low'


def compute_salary_premium(keyword, salary_dist):
    """
    计算关键词的"薪资溢价"：
      premium = P(high|keyword) / P(high|all)
    值 > 1.0 表示该词在高薪JD中过度出现（真实需求信号）
    值 ≈ 1.0 表示薪资分布正常
    值 < 1.0 表示该词更多出现在低薪JD中（通胀/buzzword信号）
    """
    total_high = sum(dist.get('high', 0) for dist in salary_dist.values())
    total_all = sum(sum(dist.values()) for dist in salary_dist.values())

    if total_all == 0:
        return 1.0, salary_dist

    kw_dist = salary_dist.get(keyword, {'high': 0, 'mid': 0, 'low': 0, 'unknown': 0})
    kw_total = sum(kw_dist.values())

    if kw_total < 5:
        return 1.0, kw_dist  # 样本太小，不判定

    baseline_high_ratio = total_high / total_all if total_all > 0 else 0
    kw_high_ratio = kw_dist.get('high', 0) / kw_total if kw_total > 0 else 0

    # 溢价 = 关键词高薪比 / 全局高薪比
    if baseline_high_ratio > 0:
        premium = kw_high_ratio / baseline_high_ratio
    else:
        premium = 1.0

    # 通胀因子：溢价越低，通胀越严重
    # premium > 1.2: 真实需求（0%通胀惩罚）
    # premium 0.8-1.2: 正常
    # premium < 0.8: buzzword嫌疑
    if premium >= 1.2:
        infl_factor = 1.0
        verdict = "genuine_demand"
    elif premium >= 0.8:
        infl_factor = 0.85
        verdict = "normal"
    elif premium >= 0.5:
        infl_factor = 0.7
        verdict = "mild_inflation"
    else:
        infl_factor = 0.5
        verdict = "severe_inflation"

    return round(infl_factor, 3), {
        "premium": round(premium, 2),
        "kw_high_ratio": round(kw_high_ratio, 3),
        "baseline_high_ratio": round(baseline_high_ratio, 3),
        "kw_total": kw_total,
        "verdict": verdict,
        "distribution": {k: v for k, v in kw_dist.items()},
    }


def scan_salary_inflation(jd_dir="【已标注】jd_v2"):
    """扫描 JD 薪资数据，建立关键词→薪资分布。"""
    csv_files = glob.glob(f"{jd_dir}/*.csv")
    kw_salary = defaultdict(lambda: Counter())
    global_salary = Counter()

    keywords = [
        "agent", "rag", "mcp", "function calling", "moe", "rlhf",
        "MLLM", "multi-agent", "diffusion transformer", "world model",
        "self-evolving", "ai-agent", "self-improving", "training-free",
        "knowledge distillation", "test-time", "long-context",
        "llm-guided", "llm-driven", "deep-research",
        "synthetic data", "godot-mcp", "hermes-agent",
    ]

    for fp in csv_files:
        try:
            with open(fp, "r", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    text = (row.get("skill_requirements", "") or "").lower()
                    salary = row.get("salary", "") or ""
                    level = salary_level(salary)
                    global_salary[level] += 1

                    for kw in keywords:
                        if kw.lower() in text:
                            kw_salary[kw][level] += 1
        except Exception:
            pass

    return dict(kw_salary), dict(global_salary)


# ============================================================
# 2. 幻觉修复 —— 对接 HallucinationGuard
# ============================================================

def build_enhanced_knowledge_base():
    """从现有 TECH_KNOWLEDGE_BASE + 验证技能库 构建增强知识库。"""
    kb = dict(TECH_KNOWLEDGE_BASE)

    # 添加 24 个关键词的上下文
    keyword_contexts = {
        "agent": {"description": "AI智能体，基于LLM的自主决策系统", "category": "AI Agent"},
        "rag": {"description": "检索增强生成，结合外部知识库的生成方法", "category": "RAG"},
        "mcp": {"description": "Model Context Protocol，AI模型上下文协议", "category": "AI协议"},
        "function calling": {"description": "LLM函数调用能力，使模型能调用外部工具", "category": "LLM能力"},
        "moe": {"description": "混合专家架构，动态路由到不同子模型", "category": "模型架构"},
        "rlhf": {"description": "基于人类反馈的强化学习，对齐训练方法", "category": "训练方法"},
        "MLLM": {"description": "多模态大语言模型，支持图文音视频输入", "category": "多模态"},
        "multi-agent systems": {"description": "多智能体协作系统", "category": "AI Agent"},
        "diffusion transformers": {"description": "扩散Transformer，图像/视频生成架构", "category": "生成模型"},
        "world models": {"description": "世界模型，学习环境动态的预测模型", "category": "强化学习"},
        "self-evolving": {"description": "自进化模型，持续学习自我改进", "category": "训练方法"},
        "ai-agent": {"description": "AI Agent框架与工具", "category": "AI Agent"},
        "self-improving": {"description": "自改进系统，模型自我优化能力", "category": "训练方法"},
        "training-free": {"description": "免训练方法，zero-shot/few-shot推理", "category": "推理方法"},
        "knowledge distillation": {"description": "知识蒸馏，大模型到小模型的知识迁移", "category": "模型压缩"},
        "test-time": {"description": "测试时计算，推理阶段动态调整", "category": "推理方法"},
        "test-time adaptation": {"description": "测试时自适应，推理时动态适应新领域", "category": "推理方法"},
        "long-context": {"description": "长上下文处理，扩展LLM上下文窗口", "category": "LLM能力"},
        "llm-guided": {"description": "LLM引导的方法，用LLM辅助设计/开发", "category": "AI辅助"},
        "llm-driven": {"description": "LLM驱动的方法，以LLM为核心引擎", "category": "AI辅助"},
        "deep-research": {"description": "深度研究，AI自主搜索+综合分析能力", "category": "AI应用"},
        "synthetic data": {"description": "合成数据生成，用于训练数据增强", "category": "数据"},
        "godot-mcp": {"description": "Godot游戏引擎的MCP协议集成", "category": "工具链"},
        "hermes-agent": {"description": "开源Agent框架，工具调用与推理", "category": "AI Agent"},
    }

    for kw, info in keyword_contexts.items():
        if kw not in kb:
            kb[kw] = {
                "description": info["description"],
                "category": info["category"],
                "sources": ["tech_signal_pipeline", "keyword_monthly_counts"],
            }

    return kb


def hallucination_check_with_guard(keyword, co_skills_data, jd_count):
    """
    使用现有 HallucinationGuard 进行幻觉检查。

    Args:
        keyword: 技术关键词
        co_skills_data: 共现技能列表 [(skill, count), ...]
        jd_count: JD中出现次数

    Returns:
        verdict dict
    """
    # 初始化 Guard
    guard = HallucinationGuard(build_enhanced_knowledge_base())

    if jd_count == 0:
        return {
            "hallucination_score": 0.0,
            "risk_level": "no_data",
            "verdict": "时滞窗口，无JD数据需要验证",
            "details": [],
        }

    results = []
    if isinstance(co_skills_data, list) and len(co_skills_data) > 0:
        for item in co_skills_data[:10]:
            if isinstance(item, dict):
                skill_name = item.get("skill", "")
                skill_count = item.get("count", 0)
            elif isinstance(item, (list, tuple)):
                skill_name = item[0]
                skill_count = item[1] if len(item) > 1 else 1
            else:
                continue

            if not skill_name:
                continue

            # 清理技能名（去掉标注格式残留）
            clean_name = re.sub(r'\|.+', '', skill_name).strip()

            # HallucinationGuard 验证
            result = guard.validate_skill(clean_name, confidence=0.6)
            results.append({
                "skill": clean_name,
                "count": skill_count,
                "verdict": result.get("verdict", "unknown"),
                "confidence": result.get("confidence", 0),
            })
    else:
        return {
            "hallucination_score": 0.5,
            "risk_level": "medium",
            "verdict": f"共现数据缺失 (jd_count={jd_count})",
            "details": [],
        }

    # 统计幻觉比例
    total = len(results)
    if total == 0:
        return {"hallucination_score": 0.5, "risk_level": "medium",
                "verdict": "无可验证的共现技能", "details": []}

    verified = sum(1 for r in results if r["verdict"] == "verified")
    rejected = sum(1 for r in results if r["verdict"] == "rejected")
    needs_review = sum(1 for r in results if r["verdict"] == "needs_review")

    rejected_ratio = rejected / total
    verified_ratio = verified / total

    # 判定
    if verified_ratio >= 0.7:
        score = 0.1
        risk = "low"
        verdict = f"共现技能{verified}/{total}通过验证"
    elif rejected_ratio >= 0.5:
        score = 0.8
        risk = "high"
        verdict = f"共现技能{rejected}/{total}被拒绝，高度可疑"
    elif rejected_ratio >= 0.2:
        score = 0.4
        risk = "medium"
        verdict = f"共现技能{rejected}/{total}被拒绝，建议抽查"
    else:
        score = 0.2
        risk = "low"
        verdict = f"共现技能{verified}/{total}通过，{needs_review}/{total}待审"

    return {
        "hallucination_score": round(score, 3),
        "risk_level": risk,
        "verdict": verdict,
        "details": results,
    }


# ============================================================
# 3. 综合执行
# ============================================================

def main():
    print("=" * 60)
    print("通胀(薪资分层) + 幻觉(HallucinationGuard) 修复")
    print("=" * 60)

    # ── 通胀分析 ──
    print("\n[1/3] 薪资分层通胀分析...")
    kw_salary, global_salary = scan_salary_inflation()
    print(f"  全局薪资分布: high={global_salary.get('high',0)}, "
          f"mid={global_salary.get('mid',0)}, low={global_salary.get('low',0)}, "
          f"unknown={global_salary.get('unknown',0)}")

    inflation_results = {}
    for kw in kw_salary:
        infl_factor, details = compute_salary_premium(kw, kw_salary)
        inflation_results[kw] = {"infl_factor": infl_factor, **details}

    # 展示通胀发现
    print(f"\n  关键词薪资溢价分析:")
    genuine = [(kw, d) for kw, d in inflation_results.items() if d.get("verdict") == "genuine_demand"]
    inflated = [(kw, d) for kw, d in inflation_results.items() if d.get("verdict") in ("mild_inflation", "severe_inflation")]

    if genuine:
        print(f"    真实需求 ({len(genuine)}):")
        for kw, d in sorted(genuine, key=lambda x: -x[1]["premium"])[:5]:
            print(f"      {kw:30s} premium={d['premium']:.2f}x  high_ratio={d['kw_high_ratio']:.1%}")
    if inflated:
        print(f"    Buzzword嫌疑 ({len(inflated)}):")
        for kw, d in sorted(inflated, key=lambda x: x[1]["premium"])[:5]:
            print(f"      {kw:30s} premium={d['premium']:.2f}x  high_ratio={d['kw_high_ratio']:.1%}  verdict={d['verdict']}")

    # ── 幻觉检查 ──
    print("\n[2/3] HallucinationGuard 幻觉检查...")
    with open("phase2_tech_job_mapping.json", "r", encoding="utf-8") as f:
        p2 = json.load(f)

    hallucination_results = {}
    for r in p2["results"]:
        kw = r["keyword"]
        co_skills = r.get("top_co_skills", [])
        jd_count = r.get("jd_stats", {}).get("total_jd_occurrences", 0)
        result = hallucination_check_with_guard(kw, co_skills, jd_count)
        hallucination_results[kw] = result

    # 展示幻觉发现
    high_risk = [(kw, d) for kw, d in hallucination_results.items() if d["risk_level"] == "high"]
    medium_risk = [(kw, d) for kw, d in hallucination_results.items() if d["risk_level"] == "medium"]
    low_risk = [(kw, d) for kw, d in hallucination_results.items() if d["risk_level"] == "low"]
    no_data = [(kw, d) for kw, d in hallucination_results.items() if d["risk_level"] == "no_data"]

    print(f"    高风险: {len(high_risk)}, 中风险: {len(medium_risk)}, "
          f"低风险: {len(low_risk)}, 无数据: {len(no_data)}")

    if high_risk:
        print(f"    ⚠ 高风险关键词:")
        for kw, d in high_risk:
            print(f"      {kw}: {d['verdict']}")
    if medium_risk:
        print(f"    ⚡ 中风险关键词:")
        for kw, d in medium_risk[:5]:
            print(f"      {kw}: {d['verdict']}")

    # ── 更新模型特征 ──
    print("\n[3/3] 更新 calibrated_features.json...")
    with open("model_features.json", "r", encoding="utf-8") as f:
        model_data = json.load(f)

    updated = []
    for feat in model_data["features"]:
        kw = feat["keyword"]

        # 通胀修复
        infl_info = inflation_results.get(kw, {"infl_factor": 1.0, "verdict": "no_data"})
        infl_factor = infl_info.get("infl_factor", 1.0)

        # 幻觉修复
        hallu_info = hallucination_results.get(kw, {"hallucination_score": 0.5, "risk_level": "unknown"})

        # 重算校准 burst = 原始 burst × infl_factor × (1 - hallucination_score * 0.5)
        original_burst = feat["burst_score"]
        hallu_penalty = 1.0 - hallu_info["hallucination_score"] * 0.5
        calibrated_burst = round(original_burst * infl_factor * hallu_penalty, 4)

        updated_feat = {
            **feat,
            "calibrated_burst_v2": calibrated_burst,
            "inflation_v2": {
                "factor": infl_factor,
                "method": "salary_stratification",
                "verdict": infl_info.get("verdict", "unknown"),
                "premium": infl_info.get("premium"),
            },
            "hallucination_v2": {
                "score": hallu_info["hallucination_score"],
                "risk": hallu_info["risk_level"],
                "verdict": hallu_info["verdict"],
                "method": "HallucinationGuard + TECH_KNOWLEDGE_BASE",
            },
        }
        updated.append(updated_feat)

    output = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "method": {
            "inflation": "salary_stratification — 薪资溢价 = P(high_salary|keyword) / P(high_salary|all)",
            "hallucination": "HallucinationGuard.validate_skill() + TECH_KNOWLEDGE_BASE 60+ verified skills",
        },
        "calibrated_features": updated,
    }

    with open("calibrated_features_v2.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # 对比表
    print("\n  Burst 校准对比 (薪资通胀 + HallucinationGuard):")
    print(f"  {'Keyword':25s} {'原始':>6s} {'v1':>6s} {'v2':>6s} {'变化'}")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*6} {'-'*15}")

    with open("calibrated_features.json", "r", encoding="utf-8") as f:
        v1_data = json.load(f)
    v1_map = {c["keyword"]: c["calibrated_burst"] for c in v1_data["calibrated_features"]}

    for f in sorted(updated, key=lambda x: x["calibrated_burst_v2"], reverse=True)[:10]:
        v1 = v1_map.get(f["keyword"], f["burst_score"])
        v2 = f["calibrated_burst_v2"]
        diff = v2 - v1
        arrow = "↑" if diff > 0.01 else ("↓" if diff < -0.01 else "→")
        print(f"  {f['keyword']:25s} {f['burst_score']:6.3f} {v1:6.3f} {v2:6.3f} {arrow} {diff:+.3f}")

    print(f"\n  输出: calibrated_features_v2.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
