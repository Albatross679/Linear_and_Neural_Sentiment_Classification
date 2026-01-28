#!/usr/bin/env python3
"""
Run Logistic Regression experiments with different feature extractors and LR schedules.
Outputs metrics for the report.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import random
from collections import Counter
from typing import List, Dict, Tuple

from src.sentiment_data import read_sentiment_examples, SentimentExample
from utilities.utils import Indexer


# ==================== Feature Extractors ====================

class UnigramFeatureExtractor:
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        features = Counter()
        for word in sentence:
            feature_name = "UNIGRAM=" + word
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(feature_name, add=True)
            else:
                idx = self.indexer.index_of(feature_name)
            if idx != -1:
                features[idx] += 1
        return features


class BigramFeatureExtractor:
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        features = Counter()
        # Unigrams
        for word in sentence:
            feature_name = "UNIGRAM=" + word
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(feature_name, add=True)
            else:
                idx = self.indexer.index_of(feature_name)
            if idx != -1:
                features[idx] += 1
        # Bigrams
        for i in range(len(sentence) - 1):
            feature_name = "BIGRAM=" + sentence[i] + "|" + sentence[i + 1]
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(feature_name, add=True)
            else:
                idx = self.indexer.index_of(feature_name)
            if idx != -1:
                features[idx] += 1
        return features


class BetterFeatureExtractor:
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        features = Counter()

        # Unigrams (binary presence)
        for word in sentence:
            feature_name = "UNIGRAM=" + word
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(feature_name, add=True)
            else:
                idx = self.indexer.index_of(feature_name)
            if idx != -1:
                features[idx] = 1

        # Bigrams
        for i in range(len(sentence) - 1):
            feature_name = "BIGRAM=" + sentence[i] + "|" + sentence[i + 1]
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(feature_name, add=True)
            else:
                idx = self.indexer.index_of(feature_name)
            if idx != -1:
                features[idx] = 1

        # Trigrams
        for i in range(len(sentence) - 2):
            feature_name = "TRIGRAM=" + sentence[i] + "|" + sentence[i + 1] + "|" + sentence[i + 2]
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(feature_name, add=True)
            else:
                idx = self.indexer.index_of(feature_name)
            if idx != -1:
                features[idx] = 1

        # Length features
        length = len(sentence)
        if length < 10:
            len_feat = "LEN=short"
        elif length < 25:
            len_feat = "LEN=medium"
        else:
            len_feat = "LEN=long"
        if add_to_indexer:
            idx = self.indexer.add_and_get_index(len_feat, add=True)
        else:
            idx = self.indexer.index_of(len_feat)
        if idx != -1:
            features[idx] = 1

        return features


# ==================== Logistic Regression Classifier ====================

class LogisticRegressionClassifier:
    def __init__(self, weights: np.ndarray, feat_extractor):
        self.weights = weights
        self.feat_extractor = feat_extractor

    def predict(self, ex_words: List[str]) -> int:
        features = self.feat_extractor.extract_features(ex_words, add_to_indexer=False)
        score = 0.0
        for idx, count in features.items():
            if idx < len(self.weights):
                score += self.weights[idx] * count
        prob = 1.0 / (1.0 + np.exp(-np.clip(score, -500, 500)))
        return 1 if prob >= 0.5 else 0

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        return [self.predict(ex_words) for ex_words in all_ex_words]


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor,
                               num_epochs: int = 30, initial_lr: float = 0.5,
                               lr_decay: float = 0.95, seed: int = 42) -> LogisticRegressionClassifier:
    """Train logistic regression with configurable hyperparameters."""

    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # First pass: extract features and grow the indexer
    train_features = []
    for ex in train_exs:
        features = feat_extractor.extract_features(ex.words, add_to_indexer=True)
        train_features.append(features)

    # Initialize weights with small random values
    num_features = len(feat_extractor.get_indexer())
    weights = np.random.uniform(-0.1, 0.1, num_features)

    # SGD training
    learning_rate = initial_lr
    for epoch in range(num_epochs):
        # Shuffle training data
        indices = list(range(len(train_exs)))
        random.shuffle(indices)

        for i in indices:
            ex = train_exs[i]
            features = train_features[i]

            # Compute score
            score = 0.0
            for idx, count in features.items():
                score += weights[idx] * count

            # Sigmoid with clipping
            prob = 1.0 / (1.0 + np.exp(-np.clip(score, -500, 500)))

            # Gradient update: w = w + alpha * (y - prob) * x
            y = ex.label
            gradient_scalar = y - prob

            for idx, count in features.items():
                weights[idx] += learning_rate * gradient_scalar * count

        # Decay learning rate
        learning_rate *= lr_decay

    return LogisticRegressionClassifier(weights, feat_extractor)


# ==================== Evaluation ====================

def compute_metrics(golds: List[int], predictions: List[int]) -> Dict[str, float]:
    """Compute accuracy, precision, recall, F1."""
    num_correct = 0
    num_pos_correct = 0
    num_pred = 0
    num_gold = 0
    num_total = len(golds)

    for gold, pred in zip(golds, predictions):
        if pred == gold:
            num_correct += 1
        if pred == 1:
            num_pred += 1
        if gold == 1:
            num_gold += 1
        if pred == 1 and gold == 1:
            num_pos_correct += 1

    accuracy = num_correct / num_total if num_total > 0 else 0.0
    precision = num_pos_correct / num_pred if num_pred > 0 else 0.0
    recall = num_pos_correct / num_gold if num_gold > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def evaluate_model(model, exs: List[SentimentExample]) -> Dict[str, float]:
    """Evaluate model on examples and return metrics."""
    golds = [ex.label for ex in exs]
    predictions = model.predict_all([ex.words for ex in exs])
    return compute_metrics(golds, predictions)


# ==================== Main Experiment Runner ====================

def run_experiment(train_exs, dev_exs, feat_extractor_class, config_name: str,
                   num_epochs: int = 30, initial_lr: float = 0.5, lr_decay: float = 0.95) -> Dict:
    """Run a single experiment and return results."""

    print(f"\n{'='*60}")
    print(f"Running: {config_name}")
    print(f"  num_epochs={num_epochs}, initial_lr={initial_lr}, lr_decay={lr_decay}")
    print(f"{'='*60}")

    # Create fresh feature extractor
    feat_extractor = feat_extractor_class(Indexer())

    # Train model
    model = train_logistic_regression(
        train_exs, feat_extractor,
        num_epochs=num_epochs, initial_lr=initial_lr, lr_decay=lr_decay
    )

    # Evaluate
    train_metrics = evaluate_model(model, train_exs)
    dev_metrics = evaluate_model(model, dev_exs)

    print(f"  Train Accuracy: {train_metrics['accuracy']*100:.2f}%")
    print(f"  Dev Accuracy:   {dev_metrics['accuracy']*100:.2f}%")
    print(f"  Dev Precision:  {dev_metrics['precision']*100:.2f}%")
    print(f"  Dev Recall:     {dev_metrics['recall']*100:.2f}%")
    print(f"  Dev F1:         {dev_metrics['f1']*100:.2f}%")

    return {
        'config_name': config_name,
        'train_accuracy': train_metrics['accuracy'],
        'dev_accuracy': dev_metrics['accuracy'],
        'dev_precision': dev_metrics['precision'],
        'dev_recall': dev_metrics['recall'],
        'dev_f1': dev_metrics['f1'],
        'num_features': len(feat_extractor.get_indexer())
    }


def main():
    # Load data
    print("Loading data...")
    train_exs = read_sentiment_examples("data/train.txt")
    dev_exs = read_sentiment_examples("data/dev.txt")
    print(f"Loaded {len(train_exs)} train / {len(dev_exs)} dev examples")

    results = []

    # ==================== Feature Extractor Experiments ====================
    print("\n" + "="*70)
    print("PART 1: Feature Extractor Comparison (default LR settings)")
    print("="*70)

    # Unigram (default settings)
    results.append(run_experiment(
        train_exs, dev_exs, UnigramFeatureExtractor, "Unigram",
        num_epochs=30, initial_lr=0.5, lr_decay=0.95
    ))

    # Bigram (default settings)
    results.append(run_experiment(
        train_exs, dev_exs, BigramFeatureExtractor, "Bigram",
        num_epochs=30, initial_lr=0.5, lr_decay=0.95
    ))

    # Better (default settings)
    results.append(run_experiment(
        train_exs, dev_exs, BetterFeatureExtractor, "Better",
        num_epochs=30, initial_lr=0.5, lr_decay=0.95
    ))

    # ==================== LR Schedule Experiments ====================
    print("\n" + "="*70)
    print("PART 2: Learning Rate Schedule Comparison (Unigram features)")
    print("="*70)

    # Config 1: Fixed LR (no decay)
    results.append(run_experiment(
        train_exs, dev_exs, UnigramFeatureExtractor, "Fixed LR (decay=1.0)",
        num_epochs=30, initial_lr=0.5, lr_decay=1.0
    ))

    # Config 2: Default decay (already ran as "Unigram" above, but re-label for clarity)
    # results.append(run_experiment(
    #     train_exs, dev_exs, UnigramFeatureExtractor, "Default Decay (decay=0.95)",
    #     num_epochs=30, initial_lr=0.5, lr_decay=0.95
    # ))

    # Config 3: Aggressive decay
    results.append(run_experiment(
        train_exs, dev_exs, UnigramFeatureExtractor, "Aggressive Decay (decay=0.8)",
        num_epochs=30, initial_lr=0.5, lr_decay=0.8
    ))

    # ==================== Summary Table ====================
    print("\n" + "="*70)
    print("SUMMARY RESULTS")
    print("="*70)

    # Header
    print(f"\n{'Configuration':<35} {'Train Acc':>10} {'Dev Acc':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 95)

    for r in results:
        print(f"{r['config_name']:<35} {r['train_accuracy']*100:>9.2f}% {r['dev_accuracy']*100:>9.2f}% "
              f"{r['dev_precision']*100:>9.2f}% {r['dev_recall']*100:>9.2f}% {r['dev_f1']*100:>9.2f}%")

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)

    # Output for easy copy-paste into report
    print("\n\nFor report (markdown format):")
    print("\n|                     | Unigram | Bigram | Better | Fixed LR | Default Decay | Aggressive Decay |")
    print("|---------------------|---------|--------|--------|----------|---------------|------------------|")

    # Collect by name for easier formatting
    by_name = {r['config_name']: r for r in results}

    # Note: Default Decay is same as Unigram
    print(f"| Train Accuracy      | {by_name['Unigram']['train_accuracy']*100:.1f}%   | "
          f"{by_name['Bigram']['train_accuracy']*100:.1f}%  | "
          f"{by_name['Better']['train_accuracy']*100:.1f}%  | "
          f"{by_name['Fixed LR (decay=1.0)']['train_accuracy']*100:.1f}%    | "
          f"{by_name['Unigram']['train_accuracy']*100:.1f}%         | "
          f"{by_name['Aggressive Decay (decay=0.8)']['train_accuracy']*100:.1f}%            |")

    print(f"| Dev Accuracy        | {by_name['Unigram']['dev_accuracy']*100:.1f}%   | "
          f"{by_name['Bigram']['dev_accuracy']*100:.1f}%  | "
          f"{by_name['Better']['dev_accuracy']*100:.1f}%  | "
          f"{by_name['Fixed LR (decay=1.0)']['dev_accuracy']*100:.1f}%    | "
          f"{by_name['Unigram']['dev_accuracy']*100:.1f}%         | "
          f"{by_name['Aggressive Decay (decay=0.8)']['dev_accuracy']*100:.1f}%            |")

    print(f"| Precision           | {by_name['Unigram']['dev_precision']*100:.1f}%   | "
          f"{by_name['Bigram']['dev_precision']*100:.1f}%  | "
          f"{by_name['Better']['dev_precision']*100:.1f}%  | "
          f"{by_name['Fixed LR (decay=1.0)']['dev_precision']*100:.1f}%    | "
          f"{by_name['Unigram']['dev_precision']*100:.1f}%         | "
          f"{by_name['Aggressive Decay (decay=0.8)']['dev_precision']*100:.1f}%            |")

    print(f"| Recall              | {by_name['Unigram']['dev_recall']*100:.1f}%   | "
          f"{by_name['Bigram']['dev_recall']*100:.1f}%  | "
          f"{by_name['Better']['dev_recall']*100:.1f}%  | "
          f"{by_name['Fixed LR (decay=1.0)']['dev_recall']*100:.1f}%    | "
          f"{by_name['Unigram']['dev_recall']*100:.1f}%         | "
          f"{by_name['Aggressive Decay (decay=0.8)']['dev_recall']*100:.1f}%            |")

    print(f"| F1 Score            | {by_name['Unigram']['dev_f1']*100:.1f}%   | "
          f"{by_name['Bigram']['dev_f1']*100:.1f}%  | "
          f"{by_name['Better']['dev_f1']*100:.1f}%  | "
          f"{by_name['Fixed LR (decay=1.0)']['dev_f1']*100:.1f}%    | "
          f"{by_name['Unigram']['dev_f1']*100:.1f}%         | "
          f"{by_name['Aggressive Decay (decay=0.8)']['dev_f1']*100:.1f}%            |")


if __name__ == "__main__":
    main()
