# 03 — 动态演化分析

> 时序变化检测 + 技术信号采集 + 岗位映射

---

## 文件说明

| 文件 | 功能 | 输出 |
|------|------|------|
| `change_detector_v2.py` | 三层过滤检测岗位技能变化 | `change_report_v2.json` |
| `phase1_time_lag.py` | arXiv/GitHub→JD时滞验证 | `phase1_time_lag_report.json` |
| `phase2_tech_job_mapping.py` | 技术关键词→岗位映射(共现分析) | `phase2_tech_job_mapping.json` |
| `phase3_audit_panel.py` | 人工审核面板(19条待审) | `phase3_audit_panel.json` |

## 运行

```bash
# 时序变化检测
python change_detector_v2.py --annotated_dir "../08_最终数据/jd_v2/" --output change_report.json

# 技术信号采集（需arXiv/GitHub API）
python phase1_time_lag.py
python phase2_tech_job_mapping.py
python phase3_audit_panel.py
```

## 三层过滤 (change_detector_v2.py)

```
3.5万条JD → 按岗位+季度分组
  │
  ├─ 第1层: Fisher精确检验 + Cohen's h  → 排除采样噪声(p<0.05, h>=0.2)
  ├─ 第2层: Jaccard抄袭簇检测 + 通胀校准 → 排除技能通胀(相似度>0.8)
  └─ 第3层: 过时技能+基础技能知识库     → 排除已知旧技能
      │
      ▼
  可信变化: 高置信度2个, 中置信度3个, 低置信度51个(被过滤)
```

## 时滞分析 (phase1/2/3)

| 状态 | 数量 | 关键词示例 | JD出现次数 |
|------|------|-----------|-----------|
| Converted(已转化) | 5 | agent, rag, mcp, function calling, rlhf | 50-1299 |
| Emerging(新兴) | 8 | moe, MLLM, world models, ai-agent | 1-48 |
| Lag Window(时滞窗口) | 11 | diffusion transformers, self-evolving... | 0 |

**中位数时滞：11个月**

## 技术信号来源

- arXiv API: cs.AI / cs.CL / cs.LG
- GitHub API: 仓库名关键词匹配
- 追踪关键词: 24个(6基础 + 18新兴)
- 数据窗口: 2025-07 ~ 2026-08(14个月)
