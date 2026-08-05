# 06 — ML分类模型

> 岗位多分类引擎：DeBERTa-v3 + CWBS长尾损失 + 异构集成

---

## 架构

```
输入: 简历文本 + 岗位描述
  │
  ├─ Encoder: DeBERTa-v3-large (304M, 1024-dim)
  │     ├─ RTD (Replaced Token Detection)
  │     └─ GDES (Gradient Disentangled Embedding Sharing)
  │
  ├─ 分类器:
  │     ├─ JobClassifier: Linear(1024, N_jobs)
  │     ├─ PrototypeClassifier: Cosine(h, proto_matrix)
  │     └─ HierarchicalClassifier: 岗位族(8) → 具体岗位(~80)
  │
  ├─ 训练: Two-Stage
  │     ├─ Stage 1: Focal + SupCon + SpanCL (15 epochs)
  │     └─ Stage 2: CWBS + 自适应tau (8 epochs)
  │
  └─ 推理: 异构集成 + Bayesian kNN + TTA
```

## 核心技术

| 技术 | 用途 | 来源 |
|------|------|------|
| DeBERTa-v3 | 文本编码 | Microsoft 2021 |
| Focal Loss (gamma=2.0) | 聚焦难样本/冷门岗 | ICCV 2017 |
| CWBS Loss | 频率偏置校正 | ICLR 2021 |
| SupCon | 同类嵌入聚集 | NeurIPS 2020 |
| SpanCL | 技能span对比 | IJCAI 2025 |
| Bayesian kNN | 冷门岗非参数推理 | SCoRE 2025 |
| 异构集成 | 4模型熵感知动态权重 | 自研 |

## 8个岗位族

算法AI / 后端开发 / 前端移动 / 数据 / 测试运维 / 产品管理 / 硬件嵌入式 / 通信云

## 训练

```bash
cd job_classification/scripts
python train.py
```

## 推理

```bash
python infer_ensemble.py --resume "resume text here"
```

## 技术复用率

12项核心技术中10项直接迁移自Wikidata关系提取项目(563类长尾分类, V3 Score 0.80+)，复用率>80%。

详细设计见 `../07_文档/技术迁移方案_岗位识别.md`
