# 04 — 新岗位发现与定义

> 新岗位候选判定 + 结构化岗位定义生成 + 人工审核工作流
> 对标题目要求：识别新兴岗位并生成岗位名称、核心职责、必备技能、加分技能、典型行业应用场景，支持人工优化与动态更新

---

## 文件清单

| 文件 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `new_job_definition.py` | 筛选→归一化→聚合→合成→(可选LLM精炼)→导出 | 已标注CSV（164文件） | `new_job_definitions.json` + `audit_definitions.xlsx` |
| `new_job_candidate.py` | 双标准新岗位候选判定 | JD原文 | CSV"是否新岗位候选"字段 |

---

## 数据管线

```
已标注JD(22,884条, 164 CSV)
    │
    ├─ new_job_candidate.py
    │   条件① AI是主业: ≥3个AI术语 + 占比>50% + 无传统底座关键词
    │   条件② 真实需求: 同类≥5条 + ≥3家公司 + 时间跨度≥6个月
    │   └─→ 2,392条新岗位候选 (P=100%, R=100%, F1=100%)
    │
    └─ new_job_definition.py
         │
         ├─ 1. 岗位名归一化 (778个原始组)
         │     去括号/职级/营销标签/中英混合取中文
         │     实习生→工程师 / 研发→开发统一 / 复合名取主段
         │
         ├─ 2. 同归一化名聚合 (≥2条JD的281个簇)
         │
         ├─ 3. 逐簇合成定义
         │     核心职责: 正则提取职责段→AI信号密度+簇内频率双因子排序→Top 8
         │     必备技能: Counter统计→≥20%支持率→Top 8
         │     加分技能: Counter统计→≥10%支持率→Top 6 (去重必备)
         │     行业场景: company_name正则匹配13类关键词
         │
         ├─ 4. [可选] LLM精炼 (--use_llm)
         │     规则初稿→DeepSeek API凝练职责+场景→幻觉校验→择优采纳
         │
         └─→ new_job_definitions.json (281个定义, 1,895条JD)
              audit_definitions.xlsx (待审核工作簿)
```

---

## 输出字段

### JSON (`new_job_definitions.json`)

```json
{
  "definitions": [{
    "cluster_id": 5,
    "岗位名称": "大模型算法工程师",
    "别称": ["LLM算法工程师", "大语言模型算法工程师"],
    "归一化名": "大模型算法工程师",
    "核心职责": [
      "大模型预训练与微调(SFT/RLHF/DPO)",
      "大模型推理优化与部署"
    ],
    "必备技能": [
      {"skill": "大模型", "category": "AI新兴技能", "proficiency": "熟悉", "support_pct": 0.55}
    ],
    "加分技能": [
      {"skill": "Python", "category": "传统技术", "proficiency": "精通", "support_pct": 0.20}
    ],
    "典型行业应用场景": [
      {"industry": "通用信息技术", "support_pct": 0.83}
    ],
    "source_traceability": {
      "total_jds": 95,
      "unique_companies": 74,
      "top_companies": ["公司A", "公司B"],
      "top_cities": ["上海-长宁区", "郑州-郑东新区"],
      "inflation_ratio": 0.12
    },
    "人工优化": {
      "status": "pending_review",
      "audit_id": "newjob_005"
    },
    "动态更新": {
      "linked_timeline": false
    }
  }],
  "statistics": {
    "total_definitions": 281,
    "total_jds_covered": 1895,
    "avg_jds_per_definition": 6.7,
    "engine": "rule"
  }
}
```

### Excel (`audit_definitions.xlsx`)

3个工作表：

| 工作表 | 内容 | 用途 |
|--------|------|------|
| **待审核定义** | 281行×16列，每行一个定义，含完整职责/技能/行业/溯源 | 人工逐条review，修改"审核状态"和"审核意见"列 |
| **统计概览** | 汇总指标 + Top 20岗位排名 | 快速了解全局 |
| **行业分布** | 13行业×关联JD数/定义数 | 评审展示用 |

Excel 列：审核编号 | 岗位名称 | 别称 | JD数 | 公司数 | 通胀比例 | 核心职责(换行) | 必备技能(支持率) | 加分技能(支持率) | 行业场景 | 代表公司 | 代表城市 | LLM精炼 | 审核状态 | 审核意见

---

## 岗位名归一化规则

| # | 处理 | 示例 |
|---|------|------|
| 1 | 去【】[] 营销标签 | `【急聘】AI工程师` → `AI工程师` |
| 2 | 去括号（含嵌套，最多3轮） | `AI算法工程师(大模型方向)(J12345)` → `AI算法工程师` |
| 3 | 去职级前缀（高级/资深/实习/应届等） | `高级AI工程师` → `AI工程师` |
| 4 | 实习生→工程师（技术岗） | `大模型算法实习生` → `大模型算法工程师` |
| 5 | 研发→开发统一 | `AI Agent研发工程师` → `AI Agent开发工程师` |
| 6 | 英文关键词后"开发"可选合并 | `AI Agent开发工程师` → `AI Agent工程师` |
| 7 | 中英混合名取中文段 | `AI Edge Engineer – AI边缘工程师` → `AI边缘工程师` |
| 8 | "/" 复合名取含角色后缀最段 | `AI算法工程师/研究员` → `AI算法工程师` |
| 9 | 去 `---` 公司/城市后缀 | `大模型算法专家-----荣耀---深圳` → `大模型算法专家` |
| 10 | 去开头公司/项目名前缀 | `华为-AI算法工程师` → `AI算法工程师` |

---

## CLI 参考

### 基本用法

```bash
# 纯规则模式（默认，零外部依赖）
python new_job_definition.py \
    --input_dir "../09_最终数据/jd_v2/" \
    --output new_job_definitions.json \
    --audit_excel audit_definitions.xlsx

# LLM精炼模式（需 DEEPSEEK_API_KEY 环境变量）
python new_job_definition.py \
    --input_dir "../09_最终数据/jd_v2/" \
    --output new_job_definitions.json \
    --audit_excel audit_definitions.xlsx \
    --use_llm --llm_max 30
```

### 参数一览

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--input_dir` | str | **必填** | 已标注CSV目录 |
| `--output` | str | `new_job_definitions.json` | 输出JSON路径 |
| `--audit_excel` | str | 空（不导出） | 审核Excel路径，如 `audit_definitions.xlsx` |
| `--timeline_report` | str | 空 | 能力动态更新报告，关联趋势数据 |
| `--use_llm` | flag | False | 启用LLM精炼（需 `DEEPSEEK_API_KEY`） |
| `--llm_max` | int | 30 | LLM精炼最多处理前N个岗位（仅JD≥5的） |

### 纯规则 vs LLM精炼

| 维度 | 纯规则 (默认) | LLM精炼 (--use_llm) |
|------|--------------|---------------------|
| 依赖 | Python stdlib + pandas | + DeepSeek API |
| 确定性 | 100%可复现 | 受temperature影响 |
| 核心职责 | 从JD原文截取+排序 | LLM凝练≤30字/条 |
| 行业场景 | 公司名正则匹配 | LLM从JD描述推断 |
| 幻觉风险 | 零（只统计不生成） | 有→3-gram溯源校验+自动回退 |
| 覆盖范围 | 全量281个 | JD≥5的大簇（Top N个） |
| 速度 | ~5秒 | ~5秒 + LLM调用时间 |

---

## LLM精炼详情

### 流程

```
规则初稿 → build_llm_prompt() → DeepSeek API → 解析JSON
    → verify_llm_duties() → 3-gram回溯命中率检查
        → ≥15%命中: 采纳LLM输出
        → <15%命中: 告警，回退规则版
```

### Prompt内容

发送给LLM的数据全部来自源统计，不掺杂模型知识：
- 岗位名称 + 别称
- 样本量 / 公司数
- 原始职责句（规则提取的 Top 10）
- 高频技能 + 支持率
- 规则推断的行业分布
- **明确约束：** 严禁编造源数据中不存在的技能、公司、数字

### 幻觉校验

| 检查项 | 方法 | 阈值 |
|--------|------|------|
| 职责-源数据关联 | 3-gram命中源JD文本 | ≥15% |
| 职责条数合理性 | 5-8条 | — |
| 行业可溯源 | 行业名与源公司名匹配 | — |
| 整体可信度 | 告警数 ≤ 50%职责数 | 超过则全部回退 |

### 容错

- API Key未设置 → 自动退纯规则，打印提示
- API调用失败（5次重试后） → 该条定义保留规则版
- JSON解析失败 → 尝试正则提取，仍失败则回退
- 幻觉校验不通过 → 打印告警数，回退规则版

---

## 人工审核工作流

### Step 1: 生成待审数据

```bash
python new_job_definition.py \
    --input_dir "./【已标注】filtered/jd_v2/" \
    --output new_job_definitions.json \
    --audit_excel audit_definitions.xlsx
```

### Step 2: 打开 Excel 逐条审核

`audit_definitions.xlsx` → 工作表"待审核定义"

对每条定义：
1. 看"核心职责"列 — 是否准确？缺什么？
2. 看"必备技能"列 — 支持率是否合理？是否需要人工补充？
3. 看"行业场景"列 — 是否遗漏了重要行业？
4. 修改"审核状态"列：`待审核` → `已通过` / `已修订` / `已驳回`
5. 在"审核意见"列填写修改建议

### Step 3: 后端同步审核结果

审核后端通过 `audit_id`（如 `newjob_005`）匹配JSON和Excel，将审核结果写回数据库。下次重新生成时，已审核的定义可保留人工修订。

---

## Top 15 岗位

| 排名 | 岗位名称 | JD数 | 公司数 | 核心技能 |
|------|---------|------|--------|---------|
| 1 | AI应用开发工程师 | 196 | 120 | 大模型, Agent, Python, RAG |
| 2 | AI算法工程师 | 164 | 78 | 大模型, Python, PyTorch, 多模态 |
| 3 | 算法工程师 | 138 | 70 | 大模型, Python, 深度学习, 机器学习 |
| 4 | AI产品经理 | 109 | 83 | 大模型, Agent, RAG, PRD |
| 5 | 大模型算法工程师 | 95 | 74 | 大模型, 微调, 预训练, Agent |
| 6 | AIGC算法工程师 | 64 | 38 | AIGC, 图像生成, Stable Diffusion, 微调 |
| 7 | AI Agent工程师 | 50 | 28 | Agent, AI Agent, 大模型, RAG |
| 8 | ai大模型算法工程师 | 36 | 24 | 大模型, 多模态, 微调, 模型训练 |
| 9 | 人工智能工程师 | 28 | 15 | 大模型, 智能体, RAG, Python |
| 10 | 大模型应用开发工程师 | 25 | 16 | Python, 大模型, 深度学习, LangChain |
| 11 | NLP算法工程师 | 24 | 22 | 模型训练, 大模型, LoRA, LLM |
| 12 | 算法研究员 | 21 | 14 | 深度学习, 大模型, LLM, CV |
| 13 | Python开发工程师 | 17 | 13 | Python, 大模型, GPT, Agent |
| 14 | AIGC应用开发工程师 | 16 | 14 | AIGC, Python, Stable Diffusion, 大模型 |
| 15 | 数据分析师 | 15 | 12 | 数据分析, Python, SQL, 大模型 |

---

## 集成指南（后端用）

### 目录结构

```
04_新岗位发现与定义/
├── README.md                  # 本文件
├── new_job_definition.py      # 主程序
├── new_job_candidate.py       # 候选判定（已有）
├── new_job_definitions.json   # 输出JSON
└── audit_definitions.xlsx     # 输出审核Excel
```

### 输入依赖

- `../09_最终数据/jd_v2/【已标注】*.csv` — 164个已标注CSV文件
- 每个CSV必须包含列：`job_name`, `是否新岗位候选`, `必备技能`, `加分技能`, `skill_requirements`, `company_name`, `work_area`, `技能通胀`
- （可选）`../03_动态演化分析/capability_update_report.json` — 用于关联时间轴

### 输出文件

| 文件 | 大小（估） | 用途 |
|------|-----------|------|
| `new_job_definitions.json` | ~500KB | 前端渲染 / API返回 |
| `audit_definitions.xlsx` | ~100KB | 人工审核 |

### 对前端的数据接口

参见项目根目录的 `前端数据接口说明.md`。`new_job_definitions.json` 的字段路径示例：

- 定义列表：`.definitions[]`
- 岗位名：`.[].岗位名称`
- 核心职责：`.[].核心职责[]`
- 技能+支持率：`.[].必备技能[].skill` / `.support_pct`
- 行业场景：`.[].典型行业应用场景[].industry` / `.support_pct`
