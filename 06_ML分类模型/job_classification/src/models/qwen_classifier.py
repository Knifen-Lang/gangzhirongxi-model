"""
Qwen2.5 + LoRA 岗位分类器

核心设计:
- Qwen2.5-7B-Instruct 作为 backbone (4-bit QLoRA)
- LoRA 适配器注入所有 attention + FFN 线性层
- 取最后一个 token 的 hidden state → Linear 分类头
- 完全兼容现有的损失函数 (CWBS, Focal, SupCon, SpanCL, RDrop, Triplet)

8GB 显存适配:
- 4-bit NF4 量化: 模型 ~4-5GB
- LoRA rank=64: ~50-80MB
- batch_size=4 + gradient_accumulation=4 → effective_batch=16
- gradient_checkpointing=True
"""

# ═══ 必须在导入 transformers 之前设置 HF 镜像 ═══
import os as _os
if not _os.environ.get('HF_ENDPOINT'):
    _os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple


class QwenJobClassifier(nn.Module):
    """
    Qwen2.5 + LoRA + 分类头 的岗位分类器

    与现有 JobClassifier 保持相同的 forward 接口:
        forward(input_ids, attention_mask) → (logits, cls_embedding, hidden_states)

    Args:
        model_name: Qwen2.5 模型名 (如 "Qwen/Qwen2.5-7B-Instruct")
        num_labels: 岗位类别数
        lora_config: LoRA 参数字典
        use_4bit: 是否使用 4-bit 量化
        dropout: 分类头 dropout
        prototype_matrix: 可选原型矩阵用于初始化分类头权重
        proto_init_scale: 原型初始化缩放
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        num_labels: int = 87,
        lora_config: Optional[Dict] = None,
        use_4bit: Optional[bool] = None,
        dropout: float = 0.15,
        prototype_matrix: Optional[np.ndarray] = None,
        proto_init_scale: float = 1.0,
    ):
        super().__init__()

        self.model_name = model_name
        self.num_labels = num_labels

        # ── 自动判断是否用 4-bit 量化 ──
        # 7B+ 模型: 4-bit 省显存; 3B以下: 全精度 LoRA
        if use_4bit is None:
            use_4bit = ("7b" in model_name.lower() or "14b" in model_name.lower())

        from transformers import BitsAndBytesConfig

        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            bnb_config = None

        # ── 加载 Qwen 模型 ──
        from transformers import AutoModelForCausalLM

        device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else {"": "cpu"}

        self.qwen = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            attn_implementation="eager",
        )

        # 启用梯度检查点 (省显存)
        self.qwen.gradient_checkpointing_enable()

        # ── 配置 LoRA ──
        if lora_config is None:
            lora_config = {
                "r": 64,
                "lora_alpha": 128,
                "lora_dropout": 0.05,
                "target_modules": [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
                "bias": "none",
                "task_type": "CAUSAL_LM",
            }

        from peft import get_peft_model, LoraConfig, TaskType

        peft_config = LoraConfig(
            r=lora_config.get("r", 64),
            lora_alpha=lora_config.get("lora_alpha", 128),
            lora_dropout=lora_config.get("lora_dropout", 0.05),
            target_modules=lora_config.get("target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"]),
            bias=lora_config.get("bias", "none"),
            task_type=TaskType.CAUSAL_LM,
        )

        self.qwen = get_peft_model(self.qwen, peft_config)
        self.qwen.print_trainable_parameters()

        # ── 获取 hidden_size ──
        base_model = self.qwen.base_model.model
        self.hidden_size = base_model.config.hidden_size

        # ── 分类头 (用 bfloat16 匹配 QLoRA 输出, 节省显存) ──
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, num_labels, bias=True,
                                     dtype=torch.bfloat16)
        # 确保分类头在正确的设备上 (QLoRA 模型在 cuda)
        if torch.cuda.is_available():
            self.classifier = self.classifier.to(torch.cuda.current_device())

        # 原型矩阵初始化分类头
        if prototype_matrix is not None:
            proto = torch.from_numpy(prototype_matrix).float() * proto_init_scale
            if proto.shape == (num_labels, self.hidden_size):
                self.classifier.weight.data.copy_(proto)
                self.classifier.bias.data.zero_()
                print(f"[QwenJobClassifier] 分类头用原型矩阵初始化 "
                      f"shape={proto.shape}")
            else:
                # 维度不匹配 (如 Qwen hidden_size ≠ DeBERTa hidden_size)
                # 不初始化，保留随机权重
                print(f"[QwenJobClassifier] 原型矩阵维度不匹配 "
                      f"({proto.shape} vs ({num_labels},{self.hidden_size}))，"
                      f"使用随机初始化")

        # ── 可训练参数统计 ──
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[QwenJobClassifier] 总参数: {total/1e6:.1f}M, "
              f"可训练: {trainable/1e6:.1f}M ({100*trainable/total:.1f}%)")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)

        Returns:
            logits: (batch, num_labels) 分类 logits
            cls_embedding: (batch, hidden_size) 句子表示 (最后一个 token)
            hidden_states: (batch, seq_len, hidden_size) 完整隐藏状态
        """
        # Qwen forward
        outputs = self.qwen(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # 取最后一个有效 token 的 hidden state（作为句子表示）
        hidden_states = outputs.hidden_states[-1]  # (batch, seq_len, hidden_size)

        # 用 attention_mask 找到每个样本最后一个有效 token 的位置
        # sequence_lengths = attention_mask.sum(dim=1) - 1  # (batch,)
        # batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        # cls_embedding = hidden_states[batch_indices, sequence_lengths]  # (batch, hidden)

        # 更安全的取法: 用最后一个非padding位置
        batch_size = input_ids.size(0)
        seq_lens = attention_mask.sum(dim=1) - 1  # 最后一个有效token的索引
        seq_lens = torch.clamp(seq_lens, min=0)
        batch_idx = torch.arange(batch_size, device=input_ids.device)
        cls_embedding = hidden_states[batch_idx, seq_lens]

        # Dropout + 分类
        # classifier 是 bfloat16，输出转 float32 供 loss 函数使用
        if self.training:
            cls_embedding = self.dropout(cls_embedding)

        logits = self.classifier(cls_embedding).float()

        # 统一转 float32 供 loss 函数使用 (SupCon, SpanCL 等需要 float32)
        return logits, cls_embedding.float(), hidden_states.float()

    def get_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """获取句子嵌入（用于 kNN 推理、原型匹配）"""
        with torch.no_grad():
            outputs = self.qwen(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states[-1]
            seq_lens = attention_mask.sum(dim=1) - 1
            seq_lens = torch.clamp(seq_lens, min=0)
            batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
            return hidden_states[batch_idx, seq_lens]

    def save_pretrained(self, save_dir: str):
        """保存 LoRA 权重 + 分类头"""
        import os
        os.makedirs(save_dir, exist_ok=True)

        # 保存 LoRA adapter
        self.qwen.save_pretrained(os.path.join(save_dir, "lora_adapter"))

        # 保存分类头
        torch.save(
            self.classifier.state_dict(),
            os.path.join(save_dir, "classifier_head.pt"),
        )
        print(f"[QwenJobClassifier] 模型已保存至 {save_dir}")

    def load_pretrained(self, load_dir: str):
        """加载 LoRA 权重 + 分类头"""
        import os

        # 加载 LoRA adapter
        from peft import PeftModel
        lora_path = os.path.join(load_dir, "lora_adapter")
        if os.path.exists(lora_path):
            self.qwen = PeftModel.from_pretrained(
                self.qwen.base_model.model if hasattr(self.qwen, 'base_model')
                else self.qwen,
                lora_path,
            )

        # 加载分类头
        head_path = os.path.join(load_dir, "classifier_head.pt")
        if os.path.exists(head_path):
            self.classifier.load_state_dict(torch.load(head_path, map_location="cpu"))
            print(f"[QwenJobClassifier] 分类头已加载 from {head_path}")

    def gradient_checkpointing_enable(self):
        """开启梯度检查点"""
        self.qwen.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        """关闭梯度检查点"""
        self.qwen.gradient_checkpointing_disable()

    def enable_adapter_layers(self):
        """启用 LoRA 层训练"""
        for name, param in self.qwen.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = True

    def freeze_base_model(self):
        """冻结 Qwen 基座参数（仅训练 LoRA + 分类头）"""
        for name, param in self.qwen.named_parameters():
            if "lora" not in name.lower():
                param.requires_grad = False
            else:
                param.requires_grad = True

    def unfreeze_all(self):
        """解冻所有参数"""
        for param in self.qwen.parameters():
            param.requires_grad = True

    def freeze_all_but_classifier(self):
        """冻结所有（包括LoRA），仅训练分类头（用于 Stage 2 CWBS 校准）"""
        for param in self.qwen.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)
        # 确保 LoRA 在训练模式下 dropout 正常工作
        return self


# ═══════════════════════════════════════════════════════════════
#  辅助函数
# ═══════════════════════════════════════════════════════════════

def load_qwen_tokenizer(model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
    """加载 Qwen tokenizer"""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",  # 重要: decoder-only 模型用 left padding
    )

    # Qwen tokenizer 通常不需要 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def encode_for_qwen(
    tokenizer,
    text_a: str,
    text_b: str,
    max_length: int = 512,
    encoding_format: str = "A",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    编码 (候选人文本, 岗位文本) 对 → Qwen 的 input_ids + attention_mask

    支持多种格式 (与现有 encode_job_pair 保持一致):
      A: [text_a] + [SEP] + [text_b]
      B: 岗位需求: [text_b]。\n候选人背景: [text_a]
      C: 判断以下候选人是否适合该岗位。\n[text_b]\n候选人简历: [text_a]
    """
    if encoding_format == "A":
        full_text = f"岗位需求: {text_b}\n候选人技能: {text_a}"
    elif encoding_format == "B":
        full_text = f"请判断以下候选人最适合哪个岗位。\n\n岗位信息: {text_b}\n候选人背景: {text_a}\n\n请从87个岗位中选择最匹配的一个。"
    elif encoding_format == "C":
        full_text = f"岗位名称: {text_b}\n技能要求: {text_a}\n\n任务: 该候选人属于上述岗位吗?"
    else:
        full_text = f"{text_a}\n{text_b}"

    # Tokenize with left padding (important for decoder-only models)
    encoding = tokenizer(
        full_text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors=None,  # return numpy
    )

    return (
        np.array(encoding["input_ids"], dtype="int64"),
        np.array(encoding["attention_mask"], dtype="int64"),
    )
