# 01 — 数据标注管线

> 核心模块：从原始JD文本到结构化标注输出的全流程

---

## 文件说明

| 文件 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `annotate_v3.py` | 基础引擎：260+技能词典 + 分类函数 | — | — |
| `annotate_jd_v2.py` | 主标注脚本：清洗 + 抽取 + 6项标注 | 原始JD CSV | 已标注CSV(+7列) |
| `llm_skill_extractor.py` | 五级兜底管线 | 字典未命中的JD文本 | 推断技能(含置信度) |
| `hallucination_guard.py` | 四层白名单幻觉防控 | 抽取的技能列表 | verified/rejected/outdated |
| `rag_retriever.py` | RAG三源验证 | 字典外候选技能 | 验证结果+证据 |

## 运行

```bash
python annotate_jd_v2.py
```

## 数据流

```
原始CSV (job_name, skill_requirements, ...)
  │
  ├─ is_it_industry()           ← 清洗：剔除非IT
  ├─ extract_skills_from_text() ← 词典抽取
  │   └─ 未命中 → llm_skill_extractor (五级管线)
  ├─ annotate_row()             ← 六项标注
  │   ├─ 必备技能 (1-6项) → 【技能名｜类别｜掌握级别】
  │   ├─ 加分技能 (0-4项)
  │   ├─ 技能通胀 (无/轻/中/重)
  │   ├─ 过时技能 (无/有(列表))
  │   ├─ 是否新岗位候选 (是/否)
  │   └─ 能力更新 (无/新增(技能列表))
  └─ 输出CSV (原始列 + 7标注列)
```

## 五级抽取管线

| 级别 | 方法 | 覆盖率 |
|------|------|--------|
| Tier 1 | 字典精确匹配(260+词) | ~83% JD |
| Tier 2 | 语义推断(165条模糊正则) | RAG验证 |
| Tier 3 | 极简JD跳过(<60字无岗位名) | 直接跳过 |
| Tier 4 | 职责→技能隐式推断(15条正则) | RAG验证 |
| Tier 5 | 同类岗位协同过滤(Top5高频) | RAG验证 |

## 幻觉防控

```
NER/字典输出
  ├─ 第1关: 软技能黑名单(40项) → rejected
  ├─ 第2关: 过时技能字典(39项) → outdated
  ├─ 第3关: 描述性短语模式(4类) → rejected
  ├─ 第4关: LLM约束输出 → 禁止编造
  └─ 第5关: RAG三源验证 → verified/hallucinated/needs_review
```

## 依赖

`【已标注v3】/annotate_v3.py` 必须在 Python path 中。
