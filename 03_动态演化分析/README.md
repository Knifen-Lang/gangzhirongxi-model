# 03 — 动态演化分析

> v4滑动窗口能力更新引擎 + 技术信号采集 + 岗位映射

---

## 文件说明

| 文件 | 功能 | 输出 |
|------|------|------|
| `capability_updater.py` | **v4滑动窗口引擎**：JD排序→滑动窗口→双阈值检测→5维变化分类 | `capability_update_report.json` (760岗位×4,007变化) |
| `change_detector_v2.py` | 三层过滤检测岗位技能变化（v3季度版，已升级为v4） | `change_report_v2.json` |
| `phase1_time_lag.py` | arXiv/GitHub→JD时滞验证 | `phase1_time_lag_report.json` |
| `phase2_tech_job_mapping.py` | 技术关键词→岗位映射(共现分析) | `phase2_tech_job_mapping.json` |
| `phase3_audit_panel.py` | 人工审核面板(19条待审) | `phase3_audit_panel.json` |

## 运行

```bash
# v4 能力动态更新（推荐）
python capability_updater.py \
    --annotated_dir "../08_最终数据/jd_v2/" \
    --output capability_update_report.json \
    --graph_output capability_graph_updates.json \
    --window_size 10 --step 3

# v3 时序变化检测（旧版）
python change_detector_v2.py --annotated_dir "../08_最终数据/jd_v2/" --output change_report.json

# 技术信号采集（需arXiv/GitHub API）
python phase1_time_lag.py
python phase2_tech_job_mapping.py
python phase3_audit_panel.py
```

## v4 滑动窗口引擎 (capability_updater.py)

```
22,884条JD → 按岗位+JD发布时间排序 → 滑动窗口采样 → 双阈值检测 → 三层过滤 → 5维分类
  │
  ├─ 窗口策略: 窗口大小=10条JD, 步长=3条, 自适应小岗位(4-9条→前后两半对比)
  ├─ 对比策略: 累积对比(window[0] vs window[i]) + 间隔对比 + 首尾对比
  ├─ 第1层: Fisher双尾精确检验 + Cohen's h双阈值(p<0.05+h≥0.2 OR h≥0.5)
  ├─ 第2层: Jaccard抄袭簇检测 + 通胀校准(相似度>0.8视为抄袭)
  └─ 第3层: 过时技能知识库 + 最小出现率过滤(5%)
      │
      ▼
  760岗位, 524有变化, 4,007个可信变化
  高置信度1,430项, 中置信度2,577项, 零低级噪声

  变化分类:
  技能新增(847) + AI技能新增(170) + 技能衰退(1,045) + 
  技能权重上升(991) + 技能权重下降(917) + 
  升级加分→必备(13) + 降级必备→加分(22) + 过时淘汰(2)
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
