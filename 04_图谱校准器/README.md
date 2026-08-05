# 04 — 图谱校准器

> 检测→矫正闭环。将通胀/时滞/噪声/幻觉的检测结果转化为矫正动作，输出干净的岗位能力图谱。

---

## 文件说明

| 文件 | 功能 |
|------|------|
| `graph_calibrator.py` | 四合一校准器，输入标注JD+技术信号，输出`calibrated_graph.json` |

## 运行

```bash
python graph_calibrator.py
```

## 四个校准器

### 1. InflationCalibrator（通胀校准）

**检测**：同岗位JD Jaccard相似度>0.8 → 抄袭簇
**矫正**：每条JD权重 = 1/簇大小

```
Before: 5条抄袭JD都写"大模型" → count=5
After:  5条抄袭JD → effective_count = 5×(1/5) = 1
```

### 2. TimeLagCalibrator（时滞校准）

**检测**：arXiv/GitHub vs JD首次出现时间差（中位数11个月）
**矫正**：预测性技能激活时间线

```
状态机: predicted → emerging → converted
        predicted(N天后激活) → overdue(到期未出现→重新评估)
```

### 3. NoiseCalibrator（噪声校准）

**检测**：Fisher精确检验 + Cohen's h效应量
**矫正**：只保留p<0.05且h>=0.2的信号（95%噪声被过滤）

### 4. HallucinationCalibrator（幻觉校准）

**检测**：4层白名单（软技能/过时/描述短语/已验证）
**矫正**：verified→保留 / rejected→移除 / outdated→历史层 / candidate→待审

## 输出

`calibrated_graph.json`:
```json
{
  "graph": {
    "nodes": 50,      // 技能节点（含有效频次+状态+分类）
    "edges": 200,     // 共现关系边
    "predicted_nodes": 0,  // 预测中的技能
    "stats": {
      "stable": 46, "growing": 4, "emerging": 0,
      "inflated_skills": 0
    }
  }
}
```

## 关键创新

不同于常见的"检测完写报告就结束"，本模块将四类问题的检测结果**直接转化为矫正动作**反馈到图谱构建中。最终图谱中每个节点都经过校准，不再包含通胀虚高、噪声虚报或幻觉技能。
