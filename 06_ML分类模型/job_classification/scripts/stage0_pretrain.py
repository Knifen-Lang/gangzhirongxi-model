"""
Stage 0: Description Matching Pretraining
==========================================
Warm-start the encoder by training it to match job descriptions to
prototype embeddings via cosine similarity. This gives the encoder
semantic understanding of job categories before fine-grained classification.

Migrated from: 人工智能挑战赛/v3/v3/train_v3.py (lines 187-229)
Original: PaddlePaddle → Converted to: PyTorch

Usage:
    from stage0_pretrain import run_stage0
    run_stage0(model, prototype_matrix, job_descriptions, label_encoder,
               tokenizer, device, epochs=5, lr=2e-5, batch_size=16)
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


class DescriptionDataset(Dataset):
    """
    Dataset for Stage 0: (job_description_text, label_index) pairs.

    Each item encodes "岗位描述: {description}" and maps to the class label.
    This teaches the encoder to align [CLS] embeddings with prototype vectors.
    """

    def __init__(self, descriptions, labels, tokenizer, max_length=256):
        self.descriptions = descriptions
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        text = str(self.descriptions[idx])
        label = self.labels[idx]

        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long),
        }


class PrototypeClassifier(nn.Module):
    """
    Cosine-similarity classifier for Stage 0.
    Matches [CLS] embedding against prototype matrix without a learned head.
    """

    def __init__(self, prototype_matrix: torch.Tensor):
        super().__init__()
        self.register_buffer('prototypes', prototype_matrix)  # (num_classes, hidden_dim)

    def forward(self, cls_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cls_embeddings: (batch_size, hidden_dim) — normalized [CLS]
        Returns:
            logits: (batch_size, num_classes) — cosine similarity scores
        """
        cls_norm = F.normalize(cls_embeddings, p=2, dim=-1)
        proto_norm = F.normalize(self.prototypes, p=2, dim=-1)
        logits = torch.matmul(cls_norm, proto_norm.T)  # (B, N)
        # Temperature scaling (learned in LM-ProtoNet, fixed 0.05 here)
        return logits / 0.05


def create_desc_dataset(label_encoder, tokenizer, max_length=256,
                        descriptions_dict=None):
    """
    Build DescriptionDataset from label encoder's class names.

    If descriptions_dict is provided, use it. Otherwise, use class names
    formatted as "岗位描述: {class_name}工程师岗位" as fallback descriptions.
    """
    class_names = label_encoder.classes_
    descriptions = []
    labels = list(range(len(class_names)))

    for i, name in enumerate(class_names):
        if descriptions_dict and name in descriptions_dict:
            desc = descriptions_dict[name]
        else:
            # Fallback: use class name as description
            desc = f"岗位描述: {name}工程师岗位，负责相关技术方向的设计、开发与维护工作。"

        descriptions.append(desc)

    return DescriptionDataset(descriptions, labels, tokenizer, max_length)


def run_stage0(model, prototype_matrix, label_encoder, tokenizer,
               device='cuda', epochs=5, lr=2e-5, batch_size=16,
               max_length=256, descriptions_dict=None,
               weight_decay=0.01, grad_clip=1.0, save_path=None):
    """
    Stage 0: Description Matching Pretraining.

    Freezes the classification head, trains only the encoder to align
    [CLS] embeddings with prototype vectors via cosine similarity.

    Args:
        model: JobClassifier instance
        prototype_matrix: (num_classes, hidden_dim) precomputed prototype embeddings
        label_encoder: sklearn LabelEncoder with .classes_ attribute
        tokenizer: HuggingFace tokenizer
        device: 'cuda' or 'cpu'
        epochs: number of pretraining epochs (default 5)
        lr: learning rate (default 2e-5)
        batch_size: batch size (default 16)
        max_length: max token length (default 256)
        descriptions_dict: optional {class_name: description_text} mapping
        weight_decay: AdamW weight decay
        grad_clip: max gradient norm
        save_path: optional path to save Stage 0 encoder weights

    Returns:
        model with pretrained encoder
    """
    logging.info('=' * 60)
    logging.info('STAGE 0: Description Matching Pretraining')
    logging.info('=' * 60)

    # Build dataset
    desc_dataset = create_desc_dataset(
        label_encoder, tokenizer, max_length, descriptions_dict
    )
    desc_loader = DataLoader(
        desc_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0,
    )

    logging.info(f'  Description samples: {len(desc_dataset)}')
    logging.info(f'  Classes: {len(label_encoder.classes_)}')
    logging.info(f'  Epochs: {epochs}, LR: {lr}, Batch: {batch_size}')

    # Prototype classifier (not trained, just used for cosine matching)
    proto_cls = PrototypeClassifier(prototype_matrix).to(device)

    # Optimizer: only encoder parameters
    optimizer = AdamW(
        model.encoder.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    total_steps = len(desc_loader) * epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    model.encoder.train()
    # Freeze classification head during Stage 0
    for param in model.classifier.parameters():
        param.requires_grad = False

    best_loss = float('inf')

    for epoch in range(epochs):
        epoch_loss = 0.0
        steps = 0

        pbar = tqdm(desc_loader, desc=f'S0 Epoch {epoch+1}/{epochs}')
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # Get [CLS] embedding from encoder
            cls_emb = model.get_embeddings(input_ids, attention_mask)

            # Cosine similarity → logits
            logits = proto_cls(cls_emb)

            # Cross-entropy against true class
            loss = F.cross_entropy(logits, labels)

            if not torch.isnan(loss):
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.encoder.parameters(), max_norm=grad_clip
                )
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                steps += 1
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = epoch_loss / max(steps, 1)
        logging.info(f'  S0 Epoch {epoch+1}/{epochs}: avg_loss={avg_loss:.4f}')

        if avg_loss < best_loss and save_path:
            best_loss = avg_loss
            torch.save(model.encoder.state_dict(), save_path)
            logging.info(f'  Saved best encoder to {save_path}')

    # Unfreeze classifier for subsequent stages
    for param in model.classifier.parameters():
        param.requires_grad = True

    logging.info('Stage 0 complete — encoder warmed up with description semantics')
    logging.info('=' * 60)

    return model
