#!/usr/bin/env python3
"""
Train all 4 neural models, save them, and generate a combined loss plot.

Models:
1. DAN (batch_size=1)
2. DAN + Batching (batch_size=32)
3. LSTM (batch_size=32)
4. CNN (batch_size=32)

Outputs:
- model/dan.pt, model/dan_batched.pt, model/lstm.pt, model/cnn.pt
- media/neural_models_loss.png (combined loss plot)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import random
import torch
import torch.nn as nn
from torch import optim
from typing import List, Dict, Tuple
from argparse import Namespace
import matplotlib.pyplot as plt

from src.sentiment_data import read_sentiment_examples, read_word_embeddings, SentimentExample, WordEmbeddings
from src.models import (
    DeepAveragingNetwork, LSTMSentimentClassifier, CNNSentimentClassifier,
    NeuralSentimentClassifier
)


def set_seeds(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(predictions: List[int], labels: List[int]) -> dict:
    """Compute accuracy, precision, recall, and F1 score."""
    tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
    tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)
    fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def evaluate_model(model: NeuralSentimentClassifier, examples: List[SentimentExample]) -> dict:
    """Evaluate model on a set of examples."""
    predictions = model.predict_all([ex.words for ex in examples])
    labels = [ex.label for ex in examples]
    return compute_metrics(predictions, labels)


def train_with_loss_tracking(
    network: nn.Module,
    train_exs: List[SentimentExample],
    word_embeddings: WordEmbeddings,
    args: Namespace
) -> Tuple[nn.Module, List[float]]:
    """
    Train a neural network and return epoch-level loss history.

    Returns:
        Tuple of (trained network, list of average losses per epoch)
    """
    optimizer = optim.Adam(network.parameters(), lr=args.lr)
    loss_fn = nn.NLLLoss()
    word_indexer = word_embeddings.word_indexer

    network.train()
    epoch_losses = []

    for epoch in range(args.num_epochs):
        indices = list(range(len(train_exs)))
        random.shuffle(indices)
        total_loss = 0.0
        num_batches = 0

        if args.batch_size == 1:
            # Non-batched training
            for i in indices:
                ex = train_exs[i]
                word_indices = [word_indexer.add_and_get_index(word, add=False) for word in ex.words]
                word_indices = [idx if idx != -1 else 1 for idx in word_indices]

                x = torch.tensor(word_indices, dtype=torch.long)
                y = torch.tensor(ex.label, dtype=torch.long)

                optimizer.zero_grad()
                log_probs = network(x)
                loss = loss_fn(log_probs.unsqueeze(0), y.unsqueeze(0))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1
        else:
            # Batched training
            for batch_start in range(0, len(indices), args.batch_size):
                batch_indices = indices[batch_start:batch_start + args.batch_size]

                batch_words = []
                batch_labels = []
                lengths = []

                for i in batch_indices:
                    ex = train_exs[i]
                    word_idx = [word_indexer.add_and_get_index(word, add=False) for word in ex.words]
                    word_idx = [idx if idx != -1 else 1 for idx in word_idx]
                    batch_words.append(word_idx)
                    batch_labels.append(ex.label)
                    lengths.append(len(word_idx))

                max_len = max(lengths)
                padded = [w + [0] * (max_len - len(w)) for w in batch_words]

                x = torch.tensor(padded, dtype=torch.long)
                y = torch.tensor(batch_labels, dtype=torch.long)
                lengths_tensor = torch.tensor(lengths, dtype=torch.long)

                optimizer.zero_grad()
                log_probs = network(x, lengths_tensor)
                loss = loss_fn(log_probs, y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        epoch_losses.append(avg_loss)
        print(f"  Epoch {epoch + 1}/{args.num_epochs}, Loss: {avg_loss:.4f}")

    return network, epoch_losses


def create_network(model_type: str, word_embeddings: WordEmbeddings, hidden_size: int) -> nn.Module:
    """Create a neural network of the specified type."""
    embedding_layer = word_embeddings.get_initialized_embedding_layer(frozen=True)
    embedding_dim = word_embeddings.get_embedding_length()

    if model_type == "dan":
        return DeepAveragingNetwork(embedding_layer, embedding_dim, hidden_size, num_classes=2, dropout=0.3)
    elif model_type == "lstm":
        return LSTMSentimentClassifier(embedding_layer, embedding_dim, hidden_size, num_classes=2, dropout=0.3)
    elif model_type == "cnn":
        return CNNSentimentClassifier(embedding_layer, embedding_dim, hidden_size, num_classes=2, dropout=0.3)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def plot_combined_loss(all_losses: Dict[str, List[float]], output_path: str):
    """Generate a combined loss plot for all models."""
    plt.figure(figsize=(10, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
    markers = ['o', 's', '^', 'D']  # Circle, Square, Triangle, Diamond

    for i, (model_name, losses) in enumerate(all_losses.items()):
        epochs = list(range(1, len(losses) + 1))
        plt.plot(epochs, losses,
                 label=model_name,
                 color=colors[i % len(colors)],
                 marker=markers[i % len(markers)],
                 markersize=6,
                 linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Average Loss (NLLLoss)', fontsize=12)
    plt.title('Training Loss Curves for Neural Sentiment Classifiers', fontsize=14)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, 11))
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nCombined loss plot saved to {output_path}")


def print_results_table(results: List[dict]):
    """Print results in a formatted table."""
    print("\n" + "=" * 100)
    print("EXPERIMENT RESULTS")
    print("=" * 100)

    header = "| {:20} | {:10} | {:10} | {:10} | {:10} | {:10} |".format(
        "Model", "Train Acc", "Dev Acc", "Precision", "Recall", "F1"
    )
    separator = "|" + "-" * 22 + "|" + "-" * 12 + "|" + "-" * 12 + "|" + "-" * 12 + "|" + "-" * 12 + "|" + "-" * 12 + "|"

    print(separator)
    print(header)
    print(separator)

    for r in results:
        row = "| {:20} | {:10.1%} | {:10.1%} | {:10.1%} | {:10.1%} | {:10.1%} |".format(
            r['name'],
            r['train']['accuracy'],
            r['dev']['accuracy'],
            r['dev']['precision'],
            r['dev']['recall'],
            r['dev']['f1']
        )
        print(row)

    print(separator)

    # Markdown format for report
    print("\nMarkdown format for report:")
    print("-" * 60)
    print("| Model | Train Acc | Dev Acc | Precision | Recall | F1 |")
    print("|-------|-----------|---------|-----------|--------|-----|")
    for r in results:
        print("| {} | {:.1%} | {:.1%} | {:.1%} | {:.1%} | {:.1%} |".format(
            r['name'],
            r['train']['accuracy'],
            r['dev']['accuracy'],
            r['dev']['precision'],
            r['dev']['recall'],
            r['dev']['f1']
        ))


def main():
    # Load data
    print("Loading training data...")
    train_exs = read_sentiment_examples("data/train.txt")
    print(f"Loaded {len(train_exs)} training examples")

    print("Loading dev data...")
    dev_exs = read_sentiment_examples("data/dev.txt")
    print(f"Loaded {len(dev_exs)} dev examples")

    print("Loading GloVe embeddings...")
    word_embeddings = read_word_embeddings("data/glove.6B.300d-relativized.txt")
    print(f"Loaded {len(word_embeddings.word_indexer)} word embeddings")

    # Create output directories
    os.makedirs("model", exist_ok=True)
    os.makedirs("media", exist_ok=True)

    # Define experiments
    experiments = [
        {"name": "DAN", "model": "dan", "batch_size": 1, "save_name": "dan.pt"},
        {"name": "DAN + Batching", "model": "dan", "batch_size": 32, "save_name": "dan_batched.pt"},
        {"name": "LSTM", "model": "lstm", "batch_size": 32, "save_name": "lstm.pt"},
        {"name": "CNN", "model": "cnn", "batch_size": 32, "save_name": "cnn.pt"},
    ]

    # Track losses for all models
    all_losses = {}
    results = []

    for exp in experiments:
        print(f"\n{'=' * 60}")
        print(f"Training {exp['name']}...")
        print(f"{'=' * 60}")

        # Reset seeds for reproducibility
        set_seeds(42)

        args = Namespace(
            lr=0.001,
            num_epochs=10,
            hidden_size=100,
            batch_size=exp["batch_size"]
        )

        # Create and train network
        network = create_network(exp["model"], word_embeddings, args.hidden_size)
        network, epoch_losses = train_with_loss_tracking(network, train_exs, word_embeddings, args)

        # Store losses for plotting
        all_losses[exp["name"]] = epoch_losses

        # Save model
        save_path = f"model/{exp['save_name']}"
        torch.save({
            'model_state_dict': network.state_dict(),
            'model_type': exp["model"],
            'batch_size': exp["batch_size"],
            'args': vars(args),
            'epoch_losses': epoch_losses
        }, save_path)
        print(f"Model saved to {save_path}")

        # Evaluate
        model = NeuralSentimentClassifier(network, word_embeddings)
        train_metrics = evaluate_model(model, train_exs)
        dev_metrics = evaluate_model(model, dev_exs)

        results.append({
            'name': exp['name'],
            'train': train_metrics,
            'dev': dev_metrics
        })

    # Generate combined loss plot
    plot_combined_loss(all_losses, "media/neural_models_loss.png")

    # Print results
    print_results_table(results)

    # Save results to file
    os.makedirs("output", exist_ok=True)
    with open("output/neural_experiment_results.txt", "w") as f:
        f.write("Neural Network Experiment Results\n")
        f.write("=" * 60 + "\n\n")
        f.write("| Model | Train Acc | Dev Acc | Precision | Recall | F1 |\n")
        f.write("|-------|-----------|---------|-----------|--------|-----|\n")
        for r in results:
            f.write("| {} | {:.1%} | {:.1%} | {:.1%} | {:.1%} | {:.1%} |\n".format(
                r['name'],
                r['train']['accuracy'],
                r['dev']['accuracy'],
                r['dev']['precision'],
                r['dev']['recall'],
                r['dev']['f1']
            ))
        f.write("\n")
    print("\nResults saved to output/neural_experiment_results.txt")


if __name__ == "__main__":
    main()
