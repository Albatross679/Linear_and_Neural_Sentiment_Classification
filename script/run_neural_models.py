#!/usr/bin/env python3
"""
Run LSTM and CNN models and generate blind test predictions.
These models aren't wired up in sentiment_classifier.py, so we run them here.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import random
import numpy as np
import torch

from src.sentiment_data import read_sentiment_examples, read_word_embeddings, SentimentExample, write_sentiment_examples
from src.models import train_lstm_classifier, train_cnn_classifier


def evaluate(model, exs):
    """Evaluate model on examples and return accuracy, F1, and output string."""
    predictions = model.predict_all([ex.words for ex in exs])

    num_correct = 0
    num_pos_correct = 0
    num_pred = 0
    num_gold = 0

    for i, ex in enumerate(exs):
        if predictions[i] == ex.label:
            num_correct += 1
        if predictions[i] == 1:
            num_pred += 1
        if ex.label == 1:
            num_gold += 1
        if predictions[i] == 1 and ex.label == 1:
            num_pos_correct += 1

    acc = num_correct / len(exs)
    prec = num_pos_correct / num_pred if num_pred > 0 else 0
    rec = num_pos_correct / num_gold if num_gold > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    output_str = f"Accuracy: {num_correct} / {len(exs)} = {acc:.6f}"
    output_str += f";\nPrecision: {num_pos_correct} / {num_pred} = {prec:.6f}"
    output_str += f";\nRecall: {num_pos_correct} / {num_gold} = {rec:.6f}"
    output_str += f";\nF1: {f1:.6f};\n"

    return acc, f1, output_str


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['LSTM', 'CNN'])
    parser.add_argument('--word_vecs_path', type=str, default='data/glove.6B.300d-relativized.txt')
    parser.add_argument('--train_path', type=str, default='data/train.txt')
    parser.add_argument('--dev_path', type=str, default='data/dev.txt')
    parser.add_argument('--blind_test_path', type=str, default='data/test-blind.txt')
    parser.add_argument('--test_output_path', type=str, default='output/test-blind.output.txt')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--hidden_size', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    # Set seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Load data
    print("Loading data...")
    train_exs = read_sentiment_examples(args.train_path)
    dev_exs = read_sentiment_examples(args.dev_path)

    # Load blind test (just words, no labels)
    test_exs = []
    with open(args.blind_test_path, 'r') as f:
        for line in f:
            words = line.strip().split()
            test_exs.append(words)

    print(f"{len(train_exs)} / {len(dev_exs)} / {len(test_exs)} train/dev/test examples")

    # Load embeddings
    print("Loading word embeddings...")
    word_embeddings = read_word_embeddings(args.word_vecs_path)

    # Train model
    print(f"\nTraining {args.model}...")
    start_time = time.time()

    if args.model == 'LSTM':
        model = train_lstm_classifier(args, train_exs, dev_exs, word_embeddings)
    else:  # CNN
        model = train_cnn_classifier(args, train_exs, dev_exs, word_embeddings)

    # Evaluate
    print("\n=====Train Accuracy=====")
    train_acc, train_f1, train_out = evaluate(model, train_exs)
    print(train_out)

    print("=====Dev Accuracy=====")
    dev_acc, dev_f1, dev_out = evaluate(model, dev_exs)
    print(dev_out)

    train_eval_time = time.time() - start_time
    print(f"Time for training and evaluation: {train_eval_time:.2f} seconds")

    # Generate blind test predictions
    print(f"\nGenerating predictions for {len(test_exs)} test examples...")
    test_predictions = model.predict_all(test_exs)

    # Write output
    test_exs_predicted = [SentimentExample(words, pred) for words, pred in zip(test_exs, test_predictions)]
    write_sentiment_examples(test_exs_predicted, args.test_output_path)
    print(f"Saved predictions to {args.test_output_path}")

    print(f"\n=====Results=====")
    print(f"Dev Accuracy: {dev_acc:.4f}")
    print(f"Dev F1: {dev_f1:.4f}")


if __name__ == "__main__":
    main()
