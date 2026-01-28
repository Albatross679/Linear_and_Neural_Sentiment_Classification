#!/usr/bin/env python3
"""
Generate training loss plot for Deep Averaging Network.
Saves plot to media/dan_epoch_loss.png
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import random
import torch
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
from typing import List

from src.sentiment_data import read_sentiment_examples, read_word_embeddings, SentimentExample


class DeepAveragingNetwork(nn.Module):
    """Deep Averaging Network for sentiment classification."""
    def __init__(self, embedding_layer: nn.Embedding, embedding_dim: int, hidden_size: int, num_classes: int = 2, dropout: float = 0.3):
        super(DeepAveragingNetwork, self).__init__()
        self.embedding = embedding_layer
        self.fc1 = nn.Linear(embedding_dim, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        # Initialize weights with Xavier Uniform
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, word_indices: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(word_indices)
        if embeds.dim() == 2:
            avg_embed = embeds.mean(dim=0)
        else:
            avg_embed = embeds.mean(dim=1)

        out = self.fc1(avg_embed)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return self.log_softmax(out)


def train_and_track_loss(train_exs: List[SentimentExample], word_embeddings, num_epochs=10, hidden_size=100, lr=0.001):
    """Train DAN and track loss at each epoch."""

    # Get embedding layer
    embedding_layer = word_embeddings.get_initialized_embedding_layer(frozen=True)
    embedding_dim = word_embeddings.get_embedding_length()
    word_indexer = word_embeddings.word_indexer

    # Create network
    network = DeepAveragingNetwork(embedding_layer, embedding_dim, hidden_size, num_classes=2, dropout=0.3)

    # Optimizer and loss
    optimizer = optim.Adam(network.parameters(), lr=lr)
    loss_fn = nn.NLLLoss()

    # Track losses
    epoch_losses = []

    network.train()

    for epoch in range(num_epochs):
        # Shuffle training data
        indices = list(range(len(train_exs)))
        random.shuffle(indices)

        total_loss = 0.0

        # Non-batched training (batch_size=1)
        for i in indices:
            ex = train_exs[i]

            # Convert words to indices
            word_indices = [word_indexer.add_and_get_index(word, add=False) for word in ex.words]
            word_indices = [idx if idx != -1 else 1 for idx in word_indices]  # UNK = 1

            # Create tensors
            x = torch.tensor(word_indices, dtype=torch.long)
            y = torch.tensor(ex.label, dtype=torch.long)

            # Forward pass
            optimizer.zero_grad()
            log_probs = network(x)
            loss = loss_fn(log_probs.unsqueeze(0), y.unsqueeze(0))

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_exs)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    return epoch_losses


def main():
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Load data
    print("Loading training data...")
    train_exs = read_sentiment_examples("data/train.txt")
    print(f"Loaded {len(train_exs)} training examples")

    print("Loading GloVe embeddings...")
    word_embeddings = read_word_embeddings("data/glove.6B.300d-relativized.txt")
    print(f"Loaded {len(word_embeddings.word_indexer)} word embeddings")

    # Train and track loss
    print("\nTraining Deep Averaging Network...")
    epoch_losses = train_and_track_loss(train_exs, word_embeddings, num_epochs=10, hidden_size=100, lr=0.001)

    # Create output directory if needed
    os.makedirs("media", exist_ok=True)

    # Plot average loss per epoch
    fig, ax = plt.subplots(figsize=(6, 4))
    epochs = list(range(1, len(epoch_losses) + 1))
    ax.plot(epochs, epoch_losses, 'b-o', linewidth=2, markersize=5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average NLLLoss')
    ax.set_title('Deep Averaging Network: Training Loss over Epochs')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("media/dan_epoch_loss.png", dpi=150, bbox_inches='tight')
    print("\nSaved plot to media/dan_epoch_loss.png")

    plt.close('all')

    print("\nTraining statistics:")
    print(f"  Initial loss (epoch 1): {epoch_losses[0]:.4f}")
    print(f"  Final loss (epoch 10): {epoch_losses[-1]:.4f}")


if __name__ == "__main__":
    main()
