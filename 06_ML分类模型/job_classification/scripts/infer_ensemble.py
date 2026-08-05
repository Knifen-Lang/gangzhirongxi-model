"""
Heterogeneous Multi-Model Ensemble Inference
=============================================
Loads multiple heterogeneous models, performs MC Dropout TTA per model,
computes dynamic entropy-weighted ensemble, optionally fuses with Bayesian kNN.

Migrated from: 人工智能挑战赛/v3/v3/infer_ensemble_v3.py
Original: PaddlePaddle → Converted to: PyTorch

Key improvements over single-model infer.py:
1. Multi-model loading loop (DeBERTa-v3-large/base, RoBERTa-large, BGE)
2. Per-model MC Dropout TTA (5 samples each)
3. CV-base-weight + per-sample entropy dynamic weighting
4. Bayesian kNN fusion with adaptive lambda

Usage:
    python infer_ensemble.py \
        --model_configs ./ensemble_config.json \
        --input_csv ./test_jobs.csv \
        --output_file ./ensemble_predictions.csv \
        --use_knn --mc_samples 5
"""

import argparse
import json
import os
import sys
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.models.classifier import JobClassifier
from src.data.data_utils import load_tokenizer, encode_job_pair
from src.inference.bayesian_knn import (
    BayesianKNNClassifier, compute_train_embeddings,
    dynamic_entropy_ensemble,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


class EnsembleInference:
    """
    Heterogeneous multi-model ensemble with MC Dropout TTA + Bayesian kNN.
    """

    def __init__(self, model_configs, label_encoder, device='cuda',
                 mc_samples=5, temperature=1.5, use_amp=False):
        """
        Args:
            model_configs: list of dicts, each with:
                - model_path: path to .pt checkpoint
                - model_name: HuggingFace model name
                - cv_weight: CV fold performance weight (float 0-1)
                - proto_init_scale: prototype initialization scale (default 0.05)
            label_encoder: fitted sklearn LabelEncoder
            device: 'cuda' or 'cpu'
            mc_samples: number of MC Dropout forward passes
            temperature: temperature scaling factor
            use_amp: use automatic mixed precision
        """
        self.device = device
        self.mc_samples = mc_samples
        self.temperature = temperature
        self.use_amp = use_amp
        self.label_encoder = label_encoder
        self.num_classes = len(label_encoder.classes_)

        self.models = []
        self.tokenizers = []
        self.cv_weights = []

        for cfg in model_configs:
            logging.info(f"Loading: {cfg['model_name']} from {cfg['model_path']}")
            model = JobClassifier(
                cfg['model_name'], self.num_classes,
                dropout=0.15,  # Keep dropout on for MC
                proto_init_scale=cfg.get('proto_init_scale', 0.05),
            ).to(device)

            state_dict = torch.load(cfg['model_path'], map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
            model.eval()

            # Enable dropout in eval mode for MC TTA
            for m in model.modules():
                if isinstance(m, torch.nn.Dropout):
                    m.train()

            tokenizer = load_tokenizer(cfg['model_name'])

            self.models.append(model)
            self.tokenizers.append(tokenizer)
            self.cv_weights.append(cfg.get('cv_weight', 1.0))

        # Normalize CV weights
        total_w = sum(self.cv_weights)
        self.cv_weights = [w / total_w for w in self.cv_weights]

        self.knn_classifier = None
        self.train_embeddings = None
        self.train_labels = None

        logging.info(f'Loaded {len(self.models)} models for ensemble')
        logging.info(f'CV weights: {[round(w, 3) for w in self.cv_weights]}')

    def fit_knn(self, train_dataset, batch_size=16):
        """Precompute training embeddings for Bayesian kNN."""
        logging.info('Computing training embeddings for kNN...')
        self.train_embeddings, self.train_labels = compute_train_embeddings(
            self.models[0],  # Use first model as embedding extractor
            self.tokenizers[0],
            train_dataset,
            batch_size=batch_size,
            device=self.device,
        )
        self.knn_classifier = BayesianKNNClassifier(
            self.train_embeddings, self.train_labels,
            n_neighbors=20,
        )
        logging.info(f'kNN ready: {len(self.train_labels)} reference samples')

    def predict_single(self, text_a, text_b=None, tta_formats=None):
        """
        Predict for a single input pair with full ensemble.

        Args:
            text_a: job description or resume text
            text_b: optional second text (for pair encoding)
            tta_formats: list of encoding formats for TTA (A/B/C)

        Returns:
            dict with 'probabilities', 'prediction', 'confidence', 'entropy'
        """
        all_model_probs = []

        for model, tokenizer, cv_w in zip(self.models, self.tokenizers, self.cv_weights):
            mc_probs = []

            for _ in range(self.mc_samples):
                with torch.no_grad():
                    if tta_formats:
                        format_probs = []
                        for fmt in tta_formats:
                            encoded = encode_job_pair(
                                tokenizer, text_a, text_b,
                                encoding_format=fmt,
                            )
                            input_ids = encoded['input_ids'].to(self.device)
                            attention_mask = encoded['attention_mask'].to(self.device)

                            logits = model(input_ids, attention_mask)
                            probs = F.softmax(logits / self.temperature, dim=-1)
                            format_probs.append(probs)
                        # Average across TTA formats
                        avg_probs = torch.stack(format_probs).mean(dim=0)
                    else:
                        encoded = encode_job_pair(tokenizer, text_a, text_b)
                        input_ids = encoded['input_ids'].to(self.device)
                        attention_mask = encoded['attention_mask'].to(self.device)

                        logits = model(input_ids, attention_mask)
                        avg_probs = F.softmax(logits / self.temperature, dim=-1)

                    mc_probs.append(avg_probs.cpu().numpy())

            # Average MC samples for this model
            model_avg = np.stack(mc_probs).mean(axis=0)  # (1, num_classes)
            all_model_probs.append(model_avg)

        # Dynamic entropy-weighted ensemble across models
        ensemble_probs = dynamic_entropy_ensemble(
            all_model_probs, self.cv_weights
        )  # (1, num_classes)

        # Bayesian kNN fusion
        if self.knn_classifier is not None:
            # Get embedding from first model
            encoded = encode_job_pair(self.tokenizers[0], text_a, text_b)
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)

            with torch.no_grad():
                cls_emb = self.models[0].get_embeddings(input_ids, attention_mask)
                knn_probs = self.knn_classifier.predict_proba(
                    cls_emb.cpu().numpy()
                )

            # Adaptive lambda: entropy-aware fusion
            entropy = -np.sum(ensemble_probs * np.log(ensemble_probs + 1e-8))
            lambda_knn = min(0.3, entropy / 4.0)  # Higher entropy → more kNN weight
            ensemble_probs = (1 - lambda_knn) * ensemble_probs + lambda_knn * knn_probs

        pred_idx = int(np.argmax(ensemble_probs))
        confidence = float(np.max(ensemble_probs))
        entropy = float(-np.sum(ensemble_probs * np.log(ensemble_probs + 1e-8)))

        return {
            'probabilities': ensemble_probs[0].tolist(),
            'prediction': self.label_encoder.inverse_transform([pred_idx])[0],
            'prediction_idx': pred_idx,
            'confidence': confidence,
            'entropy': entropy,
        }

    def predict_batch(self, texts_a, texts_b=None, batch_size=16, output_file=None):
        """Batch prediction with progress bar."""
        results = []
        for i in range(0, len(texts_a), batch_size):
            batch_a = texts_a[i:i+batch_size]
            batch_b = texts_b[i:i+batch_size] if texts_b else [None] * len(batch_a)

            for text_a, text_b in zip(batch_a, batch_b):
                result = self.predict_single(text_a, text_b)
                results.append(result)

            if (i // batch_size + 1) % 10 == 0:
                logging.info(f'  Progress: {i + len(batch_a)}/{len(texts_a)}')

        if output_file:
            self._save_results(results, output_file)

        return results

    def _save_results(self, results, output_file):
        """Save predictions to CSV."""
        rows = []
        for r in results:
            # Top-5 predictions
            probs = r['probabilities']
            top5_idx = np.argsort(probs)[-5:][::-1]
            top5 = [
                (self.label_encoder.inverse_transform([i])[0], round(probs[i], 4))
                for i in top5_idx
            ]

            rows.append({
                'prediction': r['prediction'],
                'confidence': round(r['confidence'], 4),
                'entropy': round(r['entropy'], 4),
                'top5_predictions': '|'.join(f'{name}({prob})' for name, prob in top5),
            })

        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        logging.info(f'Saved {len(rows)} predictions to {output_file}')


def main():
    parser = argparse.ArgumentParser(description='Heterogeneous Ensemble Inference')
    parser.add_argument('--model_configs', type=str, required=True,
                        help='JSON config file with model list')
    parser.add_argument('--label_encoder_path', type=str, required=True,
                        help='Path to label_encoder.json')
    parser.add_argument('--input_csv', type=str,
                        help='CSV with job_name/skill_requirements columns')
    parser.add_argument('--input_text', type=str,
                        help='Single text input for quick test')
    parser.add_argument('--output_file', type=str, default='ensemble_predictions.csv')
    parser.add_argument('--mc_samples', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=1.5)
    parser.add_argument('--use_knn', action='store_true')
    parser.add_argument('--knn_train_data', type=str,
                        help='Path to CSV with training data for kNN embeddings')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    # Load config
    with open(args.model_configs, 'r') as f:
        model_configs = json.load(f)

    # Load label encoder
    with open(args.label_encoder_path, 'r') as f:
        le_data = json.load(f)
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(le_data['classes'])

    # Initialize ensemble
    ensemble = EnsembleInference(
        model_configs, label_encoder,
        device=args.device,
        mc_samples=args.mc_samples,
        temperature=args.temperature,
    )

    # Fit kNN if requested
    if args.use_knn and args.knn_train_data:
        from src.data.data_utils import JobDataset, load_zhilian_data
        train_df = load_zhilian_data(args.knn_train_data)
        train_dataset = JobDataset(train_df, ensemble.tokenizers[0], label_encoder)
        ensemble.fit_knn(train_dataset, batch_size=args.batch_size)

    # Inference
    if args.input_text:
        result = ensemble.predict_single(args.input_text)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.input_csv:
        df = pd.read_csv(args.input_csv)
        texts_a = df['skill_requirements'].fillna('').tolist()
        texts_b = df['job_name'].fillna('').tolist() if 'job_name' in df.columns else None
        ensemble.predict_batch(texts_a, texts_b, args.batch_size, args.output_file)


if __name__ == '__main__':
    main()
