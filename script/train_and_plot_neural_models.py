#!/usr/bin/env python3
"""
Train all 4 neural models, save them, and generate combined plots.

Models:
1. DAN (batch_size=1)
2. DAN + Batching (batch_size=32)
3. LSTM (batch_size=32)
4. CNN (batch_size=32)

Outputs:
- model/dan.pt, model/dan_batched.pt, model/lstm.pt, model/cnn.pt
- media/neural_models_loss.png (combined loss plot)
- media/neural_models_train_acc.png (training accuracy plot)
- media/neural_models_dev_acc.png (dev accuracy plot)
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

# GPU device detection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def predict_single(network: nn.Module, ex: SentimentExample, word_indexer) -> int:
    """Predict label for a single example."""
    word_indices = [word_indexer.add_and_get_index(word, add=False) for word in ex.words]
    word_indices = [idx if idx != -1 else 1 for idx in word_indices]
    x = torch.tensor(word_indices, dtype=torch.long).to(device)

    with torch.no_grad():
        log_probs = network(x)
        return torch.argmax(log_probs).item()


def train_with_metrics_tracking(
    network: nn.Module,
    train_exs: List[SentimentExample],
    dev_exs: List[SentimentExample],
    word_embeddings: WordEmbeddings,
    args: Namespace
) -> Tuple[nn.Module, List[float], List[float], List[float]]:
    """
    Train a neural network and return epoch-level metrics history.

    Returns:
        Tuple of (trained network, epoch_losses, epoch_train_accs, epoch_dev_accs)
    """
    # Move network to device
    network.to(device)

    optimizer = optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.NLLLoss()
    word_indexer = word_embeddings.word_indexer

    network.train()
    epoch_losses = []
    epoch_train_accs = []
    epoch_dev_accs = []

    for epoch in range(args.num_epochs):
        network.train()
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

                x = torch.tensor(word_indices, dtype=torch.long).to(device)
                y = torch.tensor(ex.label, dtype=torch.long).to(device)

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

                x = torch.tensor(padded, dtype=torch.long).to(device)
                y = torch.tensor(batch_labels, dtype=torch.long).to(device)
                lengths_tensor = torch.tensor(lengths, dtype=torch.long).to(device)

                optimizer.zero_grad()
                log_probs = network(x, lengths_tensor)
                loss = loss_fn(log_probs, y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        epoch_losses.append(avg_loss)

        # Evaluate train and dev accuracy at end of epoch
        network.eval()

        # Train accuracy
        train_preds = [predict_single(network, ex, word_indexer) for ex in train_exs]
        train_acc = sum(p == ex.label for p, ex in zip(train_preds, train_exs)) / len(train_exs)
        epoch_train_accs.append(train_acc)

        # Dev accuracy
        dev_preds = [predict_single(network, ex, word_indexer) for ex in dev_exs]
        dev_acc = sum(p == ex.label for p, ex in zip(dev_preds, dev_exs)) / len(dev_exs)
        epoch_dev_accs.append(dev_acc)

        print(f"  Epoch {epoch + 1}/{args.num_epochs}, Loss: {avg_loss:.4f}, Train Acc: {train_acc:.1%}, Dev Acc: {dev_acc:.1%}")

    return network, epoch_losses, epoch_train_accs, epoch_dev_accs


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

    num_epochs = max(len(losses) for losses in all_losses.values())

    for i, (model_name, losses) in enumerate(all_losses.items()):
        epochs = list(range(1, len(losses) + 1))
        plt.plot(epochs, losses,
                 label=model_name,
                 color=colors[i % len(colors)],
                 marker=markers[i % len(markers)],
                 markersize=5,
                 linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Average Loss (NLLLoss)', fontsize=12)
    plt.title('Training Loss Curves for Neural Sentiment Classifiers', fontsize=14)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, num_epochs + 1))
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nCombined loss plot saved to {output_path}")


def plot_train_accuracy(all_train_accs: Dict[str, List[float]], output_path: str):
    """Generate training accuracy plot for all models."""
    plt.figure(figsize=(10, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
    markers = ['o', 's', '^', 'D']  # Circle, Square, Triangle, Diamond

    num_epochs = max(len(accs) for accs in all_train_accs.values())

    for i, (model_name, accs) in enumerate(all_train_accs.items()):
        epochs = list(range(1, len(accs) + 1))
        # Convert to percentage
        accs_pct = [a * 100 for a in accs]
        plt.plot(epochs, accs_pct,
                 label=model_name,
                 color=colors[i % len(colors)],
                 marker=markers[i % len(markers)],
                 markersize=5,
                 linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Training Accuracy (%)', fontsize=12)
    plt.title('Training Accuracy Over Epochs for Neural Sentiment Classifiers', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, num_epochs + 1))
    plt.ylim(50, 105)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Training accuracy plot saved to {output_path}")


def plot_dev_accuracy(all_dev_accs: Dict[str, List[float]], output_path: str):
    """Generate dev accuracy plot for all models."""
    plt.figure(figsize=(10, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
    markers = ['o', 's', '^', 'D']  # Circle, Square, Triangle, Diamond

    num_epochs = max(len(accs) for accs in all_dev_accs.values())

    for i, (model_name, accs) in enumerate(all_dev_accs.items()):
        epochs = list(range(1, len(accs) + 1))
        # Convert to percentage
        accs_pct = [a * 100 for a in accs]
        plt.plot(epochs, accs_pct,
                 label=model_name,
                 color=colors[i % len(colors)],
                 marker=markers[i % len(markers)],
                 markersize=5,
                 linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Dev Accuracy (%)', fontsize=12)
    plt.title('Dev Accuracy Over Epochs for Neural Sentiment Classifiers', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, num_epochs + 1))
    plt.ylim(50, 90)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Dev accuracy plot saved to {output_path}")


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
    # Print device info
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    print("\nLoading training data...")
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

    # Track all metrics for all models
    all_losses = {}
    all_train_accs = {}
    all_dev_accs = {}
    results = []

    for exp in experiments:
        print(f"\n{'=' * 60}")
        print(f"Training {exp['name']}...")
        print(f"{'=' * 60}")

        # Reset seeds for reproducibility
        set_seeds(42)

        args = Namespace(
            lr=0.0005,
            num_epochs=20,
            hidden_size=150,
            batch_size=exp["batch_size"],
            weight_decay=1e-5
        )

        # Create and train network
        network = create_network(exp["model"], word_embeddings, args.hidden_size)
        network, epoch_losses, epoch_train_accs, epoch_dev_accs = train_with_metrics_tracking(
            network, train_exs, dev_exs, word_embeddings, args
        )

        # Store metrics for plotting
        all_losses[exp["name"]] = epoch_losses
        all_train_accs[exp["name"]] = epoch_train_accs
        all_dev_accs[exp["name"]] = epoch_dev_accs

        # Save model
        save_path = f"model/{exp['save_name']}"
        torch.save({
            'model_state_dict': network.state_dict(),
            'model_type': exp["model"],
            'batch_size': exp["batch_size"],
            'args': vars(args),
            'epoch_losses': epoch_losses,
            'epoch_train_accs': epoch_train_accs,
            'epoch_dev_accs': epoch_dev_accs
        }, save_path)
        print(f"Model saved to {save_path}")

        # Evaluate (final metrics)
        model = NeuralSentimentClassifier(network, word_embeddings)
        train_metrics = evaluate_model(model, train_exs)
        dev_metrics = evaluate_model(model, dev_exs)

        results.append({
            'name': exp['name'],
            'train': train_metrics,
            'dev': dev_metrics
        })

    # Generate all 3 plots
    plot_combined_loss(all_losses, "media/neural_models_loss.png")
    plot_train_accuracy(all_train_accs, "media/neural_models_train_acc.png")
    plot_dev_accuracy(all_dev_accs, "media/neural_models_dev_acc.png")

    # Print results
    print_results_table(results)

    # Save results to file
    os.makedirs("output", exist_ok=True)
    with open("output/neural_experiment_results.txt", "w") as f:
        f.write("Neural Network Experiment Results\n")
        f.write("=" * 60 + "\n\n")
        f.write("Configuration: lr=0.0005, epochs=20, hidden_size=150, weight_decay=1e-5\n\n")
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
