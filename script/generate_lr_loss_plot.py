#!/usr/bin/env python3
"""
Generate training loss plot for Logistic Regression with SGD.
Trains 5 configurations and saves a combined plot to media/lr_epoch_loss.png

Configurations:
1. Unigram (default decay=0.95)
2. Bigram (default decay=0.95)
3. Better (default decay=0.95)
4. Unigram + Fixed LR (decay=1.0)
5. Unigram + Aggressive Decay (decay=0.8)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import random
import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Tuple

from src.sentiment_data import read_sentiment_examples, SentimentExample
from utilities.utils import Indexer


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

        # Unigrams with binary presence
        for word in sentence:
            feature_name = "UNIGRAM=" + word
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(feature_name, add=True)
            else:
                idx = self.indexer.index_of(feature_name)
            if idx != -1:
                features[idx] = 1  # Binary presence

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


def compute_loss(y: int, prob: float) -> float:
    """Compute binary cross-entropy loss."""
    eps = 1e-15
    prob = np.clip(prob, eps, 1 - eps)
    return -y * np.log(prob) - (1 - y) * np.log(1 - prob)


def train_and_track_loss(train_exs: List[SentimentExample], feat_extractor,
                         lr_decay: float = 0.95, num_epochs: int = 30,
                         initial_lr: float = 0.5) -> Tuple[List[float], np.ndarray]:
    """
    Train logistic regression and track average loss per epoch.

    Args:
        train_exs: Training examples
        feat_extractor: Feature extractor instance
        lr_decay: Learning rate decay factor per epoch (default 0.95)
        num_epochs: Number of training epochs (default 30)
        initial_lr: Initial learning rate (default 0.5)

    Returns:
        epoch_avg_losses: List of average losses per epoch
        weights: Trained weight vector
    """
    # First pass: extract features and grow the indexer
    train_features = []
    for ex in train_exs:
        features = feat_extractor.extract_features(ex.words, add_to_indexer=True)
        train_features.append(features)

    # Initialize weights
    num_features = len(feat_extractor.get_indexer())
    weights = np.zeros(num_features)

    # Track losses
    epoch_avg_losses = []
    learning_rate = initial_lr

    # SGD training
    for epoch in range(num_epochs):
        # Shuffle training data
        indices = list(range(len(train_exs)))
        random.shuffle(indices)

        epoch_loss = 0.0

        for i in indices:
            ex = train_exs[i]
            features = train_features[i]

            # Compute score
            score = 0.0
            for idx, count in features.items():
                score += weights[idx] * count

            # Sigmoid with clipping
            prob = 1.0 / (1.0 + np.exp(-np.clip(score, -500, 500)))

            # Compute loss for tracking
            loss = compute_loss(ex.label, prob)
            epoch_loss += loss

            # Gradient update: w = w + alpha * (y - prob) * x
            y = ex.label
            gradient_scalar = y - prob

            for idx, count in features.items():
                weights[idx] += learning_rate * gradient_scalar * count

        # Track average loss per epoch
        avg_loss = epoch_loss / len(train_exs)
        epoch_avg_losses.append(avg_loss)

        # Decrease learning rate each epoch
        learning_rate *= lr_decay

        print(f"  Epoch {epoch + 1}/{num_epochs}, Avg Loss: {avg_loss:.4f}, LR: {learning_rate:.4f}")

    return epoch_avg_losses, weights


def main():
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Load training data
    print("Loading training data...")
    train_exs = read_sentiment_examples("data/train.txt")
    print(f"Loaded {len(train_exs)} training examples\n")

    # Define configurations to train
    # (name, feature_extractor_class, lr_decay, color, linestyle)
    configs = [
        ("Unigram (decay=0.95)", UnigramFeatureExtractor, 0.95, "#1f77b4", "-"),
        ("Bigram (decay=0.95)", BigramFeatureExtractor, 0.95, "#ff7f0e", "-"),
        ("Better (decay=0.95)", BetterFeatureExtractor, 0.95, "#2ca02c", "-"),
        ("Unigram + Fixed LR (decay=1.0)", UnigramFeatureExtractor, 1.0, "#d62728", "--"),
        ("Unigram + Aggressive (decay=0.8)", UnigramFeatureExtractor, 0.8, "#9467bd", "--"),
    ]

    # Store results for each configuration
    all_results = []

    for name, extractor_class, lr_decay, color, linestyle in configs:
        print(f"Training: {name}")

        # Reset random seeds for fair comparison
        random.seed(42)
        np.random.seed(42)

        # Create fresh feature extractor
        feat_extractor = extractor_class(Indexer())

        # Train and track loss
        epoch_losses, weights = train_and_track_loss(
            train_exs, feat_extractor, lr_decay=lr_decay
        )

        all_results.append({
            "name": name,
            "losses": epoch_losses,
            "color": color,
            "linestyle": linestyle,
            "num_features": len(feat_extractor.get_indexer())
        })

        print(f"  Final loss: {epoch_losses[-1]:.4f}, Features: {len(feat_extractor.get_indexer())}\n")

    # Create output directory if needed
    os.makedirs("media", exist_ok=True)

    # Plot all configurations on the same chart
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = list(range(1, 31))

    for result in all_results:
        ax.plot(epochs, result["losses"],
                color=result["color"],
                linestyle=result["linestyle"],
                linewidth=2,
                marker='o',
                markersize=3,
                label=result["name"])

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Average Binary Cross-Entropy Loss', fontsize=12)
    ax.set_title('Logistic Regression: Training Loss Comparison', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 30)

    plt.tight_layout()
    plt.savefig("media/lr_epoch_loss.png", dpi=150, bbox_inches='tight')
    print("Saved combined plot to media/lr_epoch_loss.png")

    plt.close('all')

    # Print summary
    print("\n" + "=" * 60)
    print("Summary of Results")
    print("=" * 60)
    for result in all_results:
        print(f"{result['name']:40} | Initial: {result['losses'][0]:.4f} | Final: {result['losses'][-1]:.4f} | Features: {result['num_features']}")


if __name__ == "__main__":
    main()
