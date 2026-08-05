"""
对比学习损失 — SupCon + SpanCL

直接迁移自 人工智能挑战赛 v3&v4/utils_v3/losses.py

- SupConLoss: Supervised Contrastive Loss (Khosla et al., NeurIPS 2020)
  同类嵌入靠近，异类嵌入远离

- SpanContrastiveLoss: Span级对比损失 (TKRE, IJCAI 2025)
  对技能词/公司名 span 做对比学习
  适配岗位分类: 用 [SEP] 分割 岗位描述 + 简历文本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., NeurIPS 2020)

    同类样本在嵌入空间聚集，异类样本远离。
    使用 in-batch negatives 扩展负样本数。

    Args:
        temperature: softmax 温度（默认 0.07）
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        features = F.normalize(features, dim=-1, eps=1e-8)
        sim = torch.matmul(features, features.t()) / self.temperature

        labels = labels.reshape(-1, 1)
        pos_mask = (labels == labels.t()).float()
        pos_mask = pos_mask * (1.0 - torch.eye(
            labels.shape[0], device=labels.device
        ))

        n_pos = pos_mask.sum(dim=1)
        has_pos = (n_pos > 0).float()

        # 数值稳定
        sim_max = sim.max(dim=1, keepdim=True).values
        exp_sim = torch.exp(torch.clamp(sim - sim_max, max=30.0))
        exp_sim = exp_sim * (1.0 - torch.eye(
            labels.shape[0], device=labels.device
        ))

        neg_sum = exp_sim.sum(dim=1)
        pos_exp_sum = (exp_sim * pos_mask).sum(dim=1)

        valid = has_pos > 0
        if valid.float().sum() == 0:
            return torch.tensor(0.0, device=features.device)

        pos_exp_sum = pos_exp_sum[valid]
        neg_sum = neg_sum[valid]
        loss = -torch.log(
            torch.clamp(pos_exp_sum / (neg_sum + 1e-8), min=1e-8)
        )
        return torch.clamp(loss.mean(), max=10.0)


class SpanContrastiveLoss(nn.Module):
    """
    Span 级对比损失 — 适配岗位文本

    对 岗位描述 span 和 简历文本 span 分别做对比学习:
    - 正样本: 同岗位下不同样本的 岗位描述 span 嵌入 / 简历 span 嵌入
    - 负样本: 不同岗位的 span 嵌入

    适配: 使用 [SEP] token 分割 岗位描述 [SEP] 简历文本 [SEP]

    Args:
        temperature: softmax 温度
        sep_token_id: [SEP] token 的 id (BERT/DeBERTa 默认为 2)
    """

    def __init__(self, temperature=0.07, sep_token_id=2):
        super().__init__()
        self.temperature = temperature
        self.sep_token_id = int(sep_token_id)

    def forward(self, hidden_states, input_ids, labels):
        batch_size = hidden_states.shape[0]

        job_embeds = []
        resume_embeds = []
        for i in range(batch_size):
            ids = input_ids[i]
            sep_positions = (ids == self.sep_token_id).nonzero(as_tuple=True)[0]

            if len(sep_positions) >= 2:
                # job_desc: [CLS+1 : sep1]
                job_start = 1
                job_end = sep_positions[0].item()
                # resume: [sep1+1 : sep2]
                resume_start = job_end + 1
                resume_end = sep_positions[1].item()

                j_emb = hidden_states[i, job_start:job_end].mean(dim=0) \
                    if job_end > job_start else hidden_states[i, 0, :]
                r_emb = hidden_states[i, resume_start:resume_end].mean(dim=0) \
                    if resume_end > resume_start else hidden_states[i, 0, :]
            else:
                # fallback: 前半段=岗位描述, 后半段=简历
                half = max(1, hidden_states.shape[1] // 2)
                j_emb = hidden_states[i, 1:half].mean(dim=0)
                r_emb = hidden_states[i, half:].mean(dim=0)

            job_embeds.append(j_emb)
            resume_embeds.append(r_emb)

        job_embeds = torch.stack(job_embeds, dim=0)
        resume_embeds = torch.stack(resume_embeds, dim=0)

        loss_j = self._contrastive_loss(job_embeds, labels)
        loss_r = self._contrastive_loss(resume_embeds, labels)
        return (loss_j + loss_r) / 2.0

    def _contrastive_loss(self, embeddings, labels):
        embeddings = F.normalize(embeddings, dim=-1, eps=1e-8)
        sim = torch.matmul(embeddings, embeddings.t()) / self.temperature

        labels = labels.reshape(-1, 1)
        pos_mask = (labels == labels.t()).float()
        pos_mask = pos_mask * (1.0 - torch.eye(
            labels.shape[0], device=labels.device
        ))

        n_pos = pos_mask.sum(dim=1)
        has_pos = n_pos > 0

        sim_max = sim.max(dim=1, keepdim=True).values
        exp_sim = torch.exp(torch.clamp(sim - sim_max, max=30.0))
        exp_sim = exp_sim * (1.0 - torch.eye(
            labels.shape[0], device=labels.device
        ))

        neg_sum = exp_sim.sum(dim=1)
        pos_exp_sum = (exp_sim * pos_mask).sum(dim=1)

        valid = has_pos.float().sum() > 0
        if not valid:
            return torch.tensor(0.0, device=embeddings.device)

        valid_idx = has_pos.nonzero().squeeze(1)
        pos_exp_sum = pos_exp_sum[valid_idx]
        neg_sum = neg_sum[valid_idx]
        loss = -torch.log(pos_exp_sum / (neg_sum + 1e-8))
        return loss.mean()
