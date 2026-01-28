#!/usr/bin/env python3
"""
Run experiments for all 4 neural model configurations and collect metrics.
Outputs a results table with Accuracy, Precision, Recall, F1 for each model.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import random
import torch
import torch.nn as nn
from torch import optim
from typing import List
from argparse import Namespace

from src.sentiment_data import read_sentiment_examples, read_word_embeddings, SentimentExample, WordEmbeddings
from src.models import (
    DeepAveragingNetwork, LSTMSentimentClassifier, CNNSentimentClassifier,
    NeuralSentimentClassifier, train_deep_averaging_network,
    train_lstm_classifier, train_cnn_classifier
)


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


def set_seeds(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_experiment(name: str, model_type: str, batch_size: int, train_exs: List[SentimentExample],
                   dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings,
                   save_path: str = None) -> dict:
    """Run a single experiment and return metrics."""
    print(f"\n{'='*60}")
    print(f"Training {name}...")
    print(f"{'='*60}")

    # Reset seeds before each experiment for reproducibility
    set_seeds(42)

    # Create args namespace
    args = Namespace(
        lr=0.001,
        num_epochs=10,
        hidden_size=100,
        batch_size=batch_size
    )

    # Train the appropriate model
    if model_type == "dan":
        model = train_deep_averaging_network(args, train_exs, dev_exs, word_embeddings)
    elif model_type == "lstm":
        model = train_lstm_classifier(args, train_exs, dev_exs, word_embeddings)
    elif model_type == "cnn":
        model = train_cnn_classifier(args, train_exs, dev_exs, word_embeddings)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Save model if path provided
    if save_path:
        torch.save({
            'model_state_dict': model.network.state_dict(),
            'model_type': model_type,
            'batch_size': batch_size,
            'args': vars(args)
        }, save_path)
        print(f"Model saved to {save_path}")

    # Evaluate on train and dev sets
    train_metrics = evaluate_model(model, train_exs)
    dev_metrics = evaluate_model(model, dev_exs)

    return {
        'name': name,
        'train': train_metrics,
        'dev': dev_metrics,
        'model': model
    }


def print_results_table(results: List[dict]):
    """Print results in a formatted table."""
    print("\n" + "="*100)
    print("EXPERIMENT RESULTS")
    print("="*100)

    # Header
    header = "| {:20} | {:10} | {:10} | {:10} | {:10} | {:10} |".format(
        "Model", "Train Acc", "Dev Acc", "Precision", "Recall", "F1"
    )
    separator = "|" + "-"*22 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*12 + "|"

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
    print()

    # Also print in markdown format for easy copying to report
    print("\nMarkdown format for report:")
    print("-"*60)
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
    # Set seeds for reproducibility
    set_seeds(42)

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

    # Create model directory if needed
    os.makedirs("model", exist_ok=True)

    # Define experiments
    experiments = [
        {"name": "DAN", "model": "dan", "batch_size": 1, "save_name": "dan.pt"},
        {"name": "DAN + Batching", "model": "dan", "batch_size": 32, "save_name": "dan_batched.pt"},
        {"name": "LSTM", "model": "lstm", "batch_size": 32, "save_name": "lstm.pt"},
        {"name": "CNN", "model": "cnn", "batch_size": 32, "save_name": "cnn.pt"},
    ]

    # Run experiments
    results = []
    for exp in experiments:
        result = run_experiment(
            name=exp["name"],
            model_type=exp["model"],
            batch_size=exp["batch_size"],
            train_exs=train_exs,
            dev_exs=dev_exs,
            word_embeddings=word_embeddings,
            save_path=f"model/{exp['save_name']}"
        )
        results.append(result)

    # Print results table
    print_results_table(results)

    # Save results to file
    os.makedirs("output", exist_ok=True)
    with open("output/neural_experiment_results.txt", "w") as f:
        f.write("Neural Network Experiment Results\n")
        f.write("="*60 + "\n\n")
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
    print("Results saved to output/neural_experiment_results.txt")


if __name__ == "__main__":
    main()
