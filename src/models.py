# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from typing import List
from src.sentiment_data import *
from utilities.utils import *
from collections import Counter

# GPU device detection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """

    def __init__(self):
        self.idf = {}  # word -> IDF score

    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def compute_idf(self, train_exs):
        """Compute IDF scores from training data."""
        doc_freq = Counter()
        for ex in train_exs:
            unique_words = set(ex.words)
            for word in unique_words:
                doc_freq[word] += 1
        N = len(train_exs)
        self.idf = {word: np.log(N / (df + 1)) for word, df in doc_freq.items()}

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """

    def __init__(self, indexer: Indexer):
        super().__init__()
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        features = Counter()
        word_counts = Counter(sentence)
        for word, tf in word_counts.items():
            # Words are already lowercased by the data reader
            feature_name = "UNIGRAM=" + word
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(feature_name, add=True)
            else:
                idx = self.indexer.index_of(feature_name)
            if idx != -1:
                idf = self.idf.get(word, 1.0)
                features[idx] = tf * idf  # TF-IDF weighting
        return features


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """

    def __init__(self, indexer: Indexer):
        super().__init__()
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        features = Counter()
        # Unigrams with TF-IDF
        word_counts = Counter(sentence)
        for word, tf in word_counts.items():
            feature_name = "UNIGRAM=" + word
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(feature_name, add=True)
            else:
                idx = self.indexer.index_of(feature_name)
            if idx != -1:
                idf = self.idf.get(word, 1.0)
                features[idx] = tf * idf  # TF-IDF weighting
        # Bigrams (count-based)
        for i in range(len(sentence) - 1):
            feature_name = "BIGRAM=" + sentence[i] + "|" + sentence[i + 1]
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(feature_name, add=True)
            else:
                idx = self.indexer.index_of(feature_name)
            if idx != -1:
                features[idx] += 1
        return features


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    Uses: unigrams with TF-IDF, bigrams, trigrams, and length features.
    """

    def __init__(self, indexer: Indexer):
        super().__init__()
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        features = Counter()

        # Unigrams with TF-IDF weighting
        word_counts = Counter(sentence)
        for word, tf in word_counts.items():
            feature_name = "UNIGRAM=" + word
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(feature_name, add=True)
            else:
                idx = self.indexer.index_of(feature_name)
            if idx != -1:
                idf = self.idf.get(word, 1.0)
                features[idx] = tf * idf  # TF-IDF weighting

        # Bigrams (count-based)
        for i in range(len(sentence) - 1):
            feature_name = "BIGRAM=" + sentence[i] + "|" + sentence[i + 1]
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(feature_name, add=True)
            else:
                idx = self.indexer.index_of(feature_name)
            if idx != -1:
                features[idx] += 1

        # Trigrams (count-based)
        for i in range(len(sentence) - 2):
            feature_name = "TRIGRAM=" + sentence[i] + "|" + sentence[i + 1] + "|" + sentence[i + 2]
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(feature_name, add=True)
            else:
                idx = self.indexer.index_of(feature_name)
            if idx != -1:
                features[idx] += 1

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


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weights: np.ndarray, feat_extractor: FeatureExtractor):
        self.weights = weights
        self.feat_extractor = feat_extractor

    def predict(self, ex_words: List[str]) -> int:
        features = self.feat_extractor.extract_features(ex_words, add_to_indexer=False)
        score = 0.0
        for idx, count in features.items():
            if idx < len(self.weights):
                score += self.weights[idx] * count
        prob = 1.0 / (1.0 + np.exp(-score))
        return 1 if prob >= 0.5 else 0


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor,
                              dev_exs: List[SentimentExample] = None,
                              lr_decay: float = 0.95,
                              return_metrics: bool = False) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :param dev_exs: optional dev set for tracking dev accuracy per epoch
    :param lr_decay: learning rate decay factor per epoch (default 0.95)
    :param return_metrics: if True, return (model, metrics_dict) tuple
    :return: trained LogisticRegressionClassifier model (or tuple if return_metrics=True)
    """
    # Compute IDF scores from training data
    feat_extractor.compute_idf(train_exs)

    # First pass: extract features and grow the indexer
    train_features = []
    for ex in train_exs:
        features = feat_extractor.extract_features(ex.words, add_to_indexer=True)
        train_features.append(features)

    # Feature frequency thresholding (min_count = 2)
    feature_counts = Counter()
    for features in train_features:
        for idx in features:
            feature_counts[idx] += 1

    min_count = 2
    valid_features = {idx for idx, cnt in feature_counts.items() if cnt >= min_count}

    # Filter features
    train_features = [
        Counter({idx: cnt for idx, cnt in f.items() if idx in valid_features})
        for f in train_features
    ]

    # Initialize weights with random values for better convergence
    num_features = len(feat_extractor.get_indexer())
    np.random.seed(42)  # Fixed seed for reproducibility
    weights = np.random.uniform(-0.1, 0.1, num_features)

    # Hyperparameters
    num_epochs = 30
    learning_rate = 0.5
    lambda_reg = 0.1  # L2 regularization strength

    # Metrics tracking
    metrics = {"loss": [], "train_acc": [], "dev_acc": []}

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

            # Sigmoid
            prob = 1.0 / (1.0 + np.exp(-np.clip(score, -500, 500)))

            # Compute loss (negative log-likelihood)
            y = ex.label
            if y == 1:
                epoch_loss += -np.log(prob + 1e-10)
            else:
                epoch_loss += -np.log(1 - prob + 1e-10)

            # Gradient update with L2 regularization: w = w + alpha * ((y - prob) * x - lambda * w)
            gradient_scalar = y - prob

            for idx, count in features.items():
                weights[idx] += learning_rate * (gradient_scalar * count - lambda_reg * weights[idx])

        # Track metrics
        avg_loss = epoch_loss / len(train_exs)
        metrics["loss"].append(avg_loss)

        # Compute train accuracy
        train_correct = 0
        for i, ex in enumerate(train_exs):
            features = train_features[i]
            score = sum(weights[idx] * count for idx, count in features.items() if idx < len(weights))
            pred = 1 if 1.0 / (1.0 + np.exp(-np.clip(score, -500, 500))) >= 0.5 else 0
            if pred == ex.label:
                train_correct += 1
        metrics["train_acc"].append(train_correct / len(train_exs))

        # Compute dev accuracy if dev set provided
        if dev_exs is not None:
            dev_correct = 0
            for ex in dev_exs:
                features = feat_extractor.extract_features(ex.words, add_to_indexer=False)
                # Filter to valid features
                features = Counter({idx: cnt for idx, cnt in features.items() if idx in valid_features})
                score = sum(weights[idx] * count for idx, count in features.items() if idx < len(weights))
                pred = 1 if 1.0 / (1.0 + np.exp(-np.clip(score, -500, 500))) >= 0.5 else 0
                if pred == ex.label:
                    dev_correct += 1
            metrics["dev_acc"].append(dev_correct / len(dev_exs))
        else:
            metrics["dev_acc"].append(0.0)

        # Decrease learning rate each epoch
        learning_rate *= lr_decay

    model = LogisticRegressionClassifier(weights, feat_extractor)

    if return_metrics:
        return model, metrics
    return model


def plot_lr_metrics(all_metrics: dict, output_dir: str = "media"):
    """
    Plot loss and accuracy over epochs for multiple LR configurations.
    Generates 2 plots: lr_epoch_loss.png and lr_accuracy.png.

    all_metrics: {
        "Unigram": {"loss": [...], "train_acc": [...], "dev_acc": [...]},
        "Bigram": {...},
        ...
    }
    """
    import matplotlib.pyplot as plt
    import os

    os.makedirs(output_dir, exist_ok=True)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Plot 1: Loss
    plt.figure(figsize=(8, 5))
    for i, (name, metrics) in enumerate(all_metrics.items()):
        plt.plot(range(1, len(metrics["loss"]) + 1), metrics["loss"],
                 color=colors[i % len(colors)], label=name, linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Negative Log-Likelihood)")
    plt.title("Logistic Regression Training Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/lr_epoch_loss.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Combined Train + Dev Accuracy
    plt.figure(figsize=(8, 5))
    for i, (name, metrics) in enumerate(all_metrics.items()):
        epochs = range(1, len(metrics["train_acc"]) + 1)
        plt.plot(epochs, [acc * 100 for acc in metrics["train_acc"]],
                 color=colors[i % len(colors)], linestyle='-', label=f"{name} (train)", linewidth=1.5)
        plt.plot(epochs, [acc * 100 for acc in metrics["dev_acc"]],
                 color=colors[i % len(colors)], linestyle='--', label=f"{name} (dev)", linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Logistic Regression Accuracy")
    plt.legend(ncol=2, fontsize='small')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/lr_accuracy.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"LR plots saved to {output_dir}/")


def train_linear_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your linear model. You may modify this, but do not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model with metrics tracking
    model, metrics = train_logistic_regression(train_exs, feat_extractor, dev_exs=dev_exs,
                                                lr_decay=0.95, return_metrics=True)

    # Run all 5 configurations and plot if this is the BETTER extractor (run once)
    if args.feats == "BETTER":
        print("\nRunning all 5 LR configurations for plotting...")
        all_metrics = {}

        # Config 1: Unigram (lr=0.5, decay=0.95)
        print("  Training Unigram...")
        uni_ext = UnigramFeatureExtractor(Indexer())
        _, uni_metrics = train_logistic_regression(train_exs, uni_ext, dev_exs=dev_exs,
                                                    lr_decay=0.95, return_metrics=True)
        all_metrics["Unigram"] = uni_metrics

        # Config 2: Bigram (lr=0.5, decay=0.95)
        print("  Training Bigram...")
        bi_ext = BigramFeatureExtractor(Indexer())
        _, bi_metrics = train_logistic_regression(train_exs, bi_ext, dev_exs=dev_exs,
                                                   lr_decay=0.95, return_metrics=True)
        all_metrics["Bigram"] = bi_metrics

        # Config 3: Better (lr=0.5, decay=0.95)
        print("  Training Better...")
        all_metrics["Better"] = metrics  # Already trained above

        # Config 4: Unigram Fixed LR (lr=0.5, decay=1.0)
        print("  Training Unigram (Fixed LR)...")
        uni_fixed_ext = UnigramFeatureExtractor(Indexer())
        _, uni_fixed_metrics = train_logistic_regression(train_exs, uni_fixed_ext, dev_exs=dev_exs,
                                                          lr_decay=1.0, return_metrics=True)
        all_metrics["Unigram (Fixed LR)"] = uni_fixed_metrics

        # Config 5: Unigram Aggressive decay (lr=0.5, decay=0.8)
        print("  Training Unigram (Aggressive)...")
        uni_agg_ext = UnigramFeatureExtractor(Indexer())
        _, uni_agg_metrics = train_logistic_regression(train_exs, uni_agg_ext, dev_exs=dev_exs,
                                                        lr_decay=0.8, return_metrics=True)
        all_metrics["Unigram (Aggressive)"] = uni_agg_metrics

        # Generate plots
        plot_lr_metrics(all_metrics, output_dir="media")

    return model


# ==================== Neural Network Models ====================

class DeepAveragingNetwork(nn.Module):
    """
    Deep Averaging Network for sentiment classification.
    Averages word embeddings and passes through feedforward layers.
    """
    def __init__(self, embedding_layer: nn.Embedding, embedding_dim: int, hidden_size: int, num_classes: int = 2, dropout: float = 0.3):
        super(DeepAveragingNetwork, self).__init__()
        self.embedding = embedding_layer
        self.fc1 = nn.Linear(embedding_dim, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, word_indices: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.
        :param word_indices: [batch_size, seq_len] or [seq_len] tensor of word indices
        :param lengths: optional tensor of actual sequence lengths for masking
        :return: log probabilities [batch_size, num_classes] or [num_classes]
        """
        # Get embeddings
        embeds = self.embedding(word_indices)  # [batch, seq_len, embed_dim] or [seq_len, embed_dim]
        
        # Average embeddings (handle both batched and non-batched)
        if embeds.dim() == 2:
            # Single example: [seq_len, embed_dim]
            avg_embed = embeds.mean(dim=0)
        else:
            # Batched: [batch_size, seq_len, embed_dim]
            if lengths is not None:
                # Mask out padding
                mask = torch.arange(embeds.size(1)).unsqueeze(0).to(embeds.device) < lengths.unsqueeze(1)
                mask = mask.unsqueeze(-1).float()
                avg_embed = (embeds * mask).sum(dim=1) / lengths.unsqueeze(1).float()
            else:
                avg_embed = embeds.mean(dim=1)
        
        # Feedforward
        out = self.fc1(avg_embed)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return self.log_softmax(out)


class LSTMSentimentClassifier(nn.Module):
    """
    LSTM-based sentiment classifier (Exploration task).
    """
    def __init__(self, embedding_layer: nn.Embedding, embedding_dim: int, hidden_size: int, num_classes: int = 2, dropout: float = 0.3):
        super(LSTMSentimentClassifier, self).__init__()
        self.embedding = embedding_layer
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional
        self.log_softmax = nn.LogSoftmax(dim=-1)
        
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, word_indices: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        embeds = self.embedding(word_indices)
        
        if embeds.dim() == 2:
            embeds = embeds.unsqueeze(0)  # Add batch dimension
        
        # LSTM
        lstm_out, (hidden, _) = self.lstm(embeds)
        
        # Concatenate final hidden states from both directions
        hidden_cat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        
        out = self.dropout(hidden_cat)
        out = self.fc(out)
        
        if word_indices.dim() == 1:
            return self.log_softmax(out.squeeze(0))
        return self.log_softmax(out)


class CNNSentimentClassifier(nn.Module):
    """
    CNN-based sentiment classifier (Exploration task).
    Uses multiple kernel sizes for n-gram feature extraction.
    """
    def __init__(self, embedding_layer: nn.Embedding, embedding_dim: int, hidden_size: int, num_classes: int = 2, dropout: float = 0.3):
        super(CNNSentimentClassifier, self).__init__()
        self.embedding = embedding_layer
        
        # Multiple kernel sizes for different n-grams
        kernel_sizes = [2, 3, 4, 5]
        num_filters = hidden_size // len(kernel_sizes)
        
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, word_indices: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        embeds = self.embedding(word_indices)
        
        if embeds.dim() == 2:
            embeds = embeds.unsqueeze(0)  # Add batch dimension
        
        # Transpose for Conv1d: [batch, embed_dim, seq_len]
        embeds = embeds.transpose(1, 2)
        
        # Apply convolutions and max-pool
        conv_outs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(embeds))
            pooled = torch.max(conv_out, dim=2)[0]  # Max-over-time pooling
            conv_outs.append(pooled)
        
        # Concatenate all filter outputs
        cat_out = torch.cat(conv_outs, dim=1)
        
        out = self.dropout(cat_out)
        out = self.fc(out)
        
        if word_indices.dim() == 1:
            return self.log_softmax(out.squeeze(0))
        return self.log_softmax(out)


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """
    def __init__(self, network: nn.Module, word_embeddings: WordEmbeddings):
        self.network = network
        self.word_embeddings = word_embeddings
        self.device = next(network.parameters()).device
        self.network.eval()

    def predict(self, ex_words: List[str]) -> int:
        # Convert words to indices
        word_indexer = self.word_embeddings.word_indexer
        indices = [word_indexer.add_and_get_index(word, add=False) for word in ex_words]
        # Replace -1 (unknown) with UNK index (1)
        indices = [idx if idx != -1 else 1 for idx in indices]

        # Create tensor and move to device
        indices_tensor = torch.tensor(indices, dtype=torch.long).to(self.device)

        with torch.no_grad():
            log_probs = self.network(indices_tensor)
            prediction = torch.argmax(log_probs).item()

        return prediction

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        Batched prediction (Exploration task).
        """
        if len(all_ex_words) == 0:
            return []

        word_indexer = self.word_embeddings.word_indexer

        # Convert all examples to indices
        all_indices = []
        lengths = []
        for ex_words in all_ex_words:
            indices = [word_indexer.add_and_get_index(word, add=False) for word in ex_words]
            indices = [idx if idx != -1 else 1 for idx in indices]
            all_indices.append(indices)
            lengths.append(len(indices))

        # Pad sequences
        max_len = max(lengths)
        padded = [indices + [0] * (max_len - len(indices)) for indices in all_indices]

        # Create tensors and move to device
        batch_tensor = torch.tensor(padded, dtype=torch.long).to(self.device)
        lengths_tensor = torch.tensor(lengths, dtype=torch.long).to(self.device)

        with torch.no_grad():
            log_probs = self.network(batch_tensor, lengths_tensor)
            predictions = torch.argmax(log_probs, dim=1).tolist()

        return predictions


def _compute_neural_accuracy(network: nn.Module, examples: List[SentimentExample],
                              word_embeddings: WordEmbeddings, batch_size: int = 256) -> float:
    """Compute accuracy of a neural network on a set of examples."""
    word_indexer = word_embeddings.word_indexer
    network.eval()
    correct = 0
    total = len(examples)

    for batch_start in range(0, total, batch_size):
        batch_exs = examples[batch_start:batch_start + batch_size]
        all_indices = []
        lengths = []
        for ex in batch_exs:
            indices = [word_indexer.add_and_get_index(word, add=False) for word in ex.words]
            indices = [idx if idx != -1 else 1 for idx in indices]
            all_indices.append(indices)
            lengths.append(len(indices))

        max_len = max(lengths)
        padded = [idx + [0] * (max_len - len(idx)) for idx in all_indices]

        x = torch.tensor(padded, dtype=torch.long).to(device)
        lengths_tensor = torch.tensor(lengths, dtype=torch.long).to(device)

        with torch.no_grad():
            log_probs = network(x, lengths_tensor)
            preds = torch.argmax(log_probs, dim=1)
            labels = torch.tensor([ex.label for ex in batch_exs], dtype=torch.long).to(device)
            correct += (preds == labels).sum().item()

    network.train()
    return correct / total


def _train_neural_with_metrics(network: nn.Module, train_exs: List[SentimentExample],
                                dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings,
                                batch_size: int = 1, num_epochs: int = 20, lr: float = 0.0005,
                                weight_decay: float = 1e-5):
    """
    Generic neural training loop. Returns (trained_network, metrics_dict).
    metrics_dict has keys: "loss", "train_acc", "dev_acc" (lists per epoch).
    """
    network.to(device)
    optimizer = optim.Adam(network.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.NLLLoss()
    word_indexer = word_embeddings.word_indexer

    metrics = {"loss": [], "train_acc": [], "dev_acc": []}
    network.train()

    for epoch in range(num_epochs):
        indices = list(range(len(train_exs)))
        random.shuffle(indices)

        total_loss = 0.0
        num_batches = 0

        if batch_size == 1:
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
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]

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
        metrics["loss"].append(avg_loss)

        train_acc = _compute_neural_accuracy(network, train_exs, word_embeddings)
        dev_acc = _compute_neural_accuracy(network, dev_exs, word_embeddings)
        metrics["train_acc"].append(train_acc)
        metrics["dev_acc"].append(dev_acc)

        print(f"  Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, Dev Acc: {dev_acc:.4f}")

    return network, metrics


def plot_neural_metrics(all_metrics: dict, output_dir: str = "media"):
    """
    Plot loss and accuracy over epochs for multiple neural configurations.
    Generates 2 plots: neural_loss.png and neural_accuracy.png.
    """
    import matplotlib.pyplot as plt
    import os

    os.makedirs(output_dir, exist_ok=True)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']

    # Plot 1: Loss
    plt.figure(figsize=(8, 5))
    for i, (name, metrics) in enumerate(all_metrics.items()):
        plt.plot(range(1, len(metrics["loss"]) + 1), metrics["loss"],
                 color=colors[i % len(colors)], marker=markers[i % len(markers)],
                 label=name, linewidth=1.5, markersize=4)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Neural Models Training Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/neural_loss.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Combined Train + Dev Accuracy
    plt.figure(figsize=(8, 5))
    for i, (name, metrics) in enumerate(all_metrics.items()):
        epochs = range(1, len(metrics["train_acc"]) + 1)
        plt.plot(epochs, [acc * 100 for acc in metrics["train_acc"]],
                 color=colors[i % len(colors)], linestyle='-', label=f"{name} (train)", linewidth=1.5)
        plt.plot(epochs, [acc * 100 for acc in metrics["dev_acc"]],
                 color=colors[i % len(colors)], linestyle='--', label=f"{name} (dev)", linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Neural Models Accuracy")
    plt.legend(ncol=2, fontsize='small')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/neural_accuracy.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Neural plots saved to {output_dir}/")


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    Main entry point for your deep averaging network model.
    Trains all 4 neural scenarios (DAN, DAN+B, LSTM, CNN), generates plots,
    and returns the primary DAN (batch_size=1) classifier for evaluation.
    """
    hidden_size = 150
    num_epochs = 20
    lr = 0.0005
    weight_decay = 1e-5
    dropout = 0.3
    embedding_dim = word_embeddings.get_embedding_length()

    all_metrics = {}

    # Define the 4 experiments: (name, network_class, batch_size)
    experiments = [
        ("DAN", DeepAveragingNetwork, 1),
        ("DAN+B", DeepAveragingNetwork, 32),
        ("LSTM", LSTMSentimentClassifier, 32),
        ("CNN", CNNSentimentClassifier, 32),
    ]

    dan_network = None  # Will hold the primary DAN network for return

    for name, NetworkClass, batch_size in experiments:
        print(f"\n--- Training {name} (batch_size={batch_size}) ---")

        # Reset seeds for reproducibility
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        # Fresh embedding layer per model
        embedding_layer = word_embeddings.get_initialized_embedding_layer(frozen=True)

        # Create network
        network = NetworkClass(embedding_layer, embedding_dim, hidden_size, num_classes=2, dropout=dropout)

        # Train with metrics
        trained_network, metrics = _train_neural_with_metrics(
            network, train_exs, dev_exs, word_embeddings,
            batch_size=batch_size, num_epochs=num_epochs, lr=lr, weight_decay=weight_decay
        )

        all_metrics[name] = metrics

        # Keep the primary DAN (batch_size=1) for return
        if name == "DAN":
            dan_network = trained_network

    # Generate plots
    plot_neural_metrics(all_metrics, output_dir="media")

    return NeuralSentimentClassifier(dan_network, word_embeddings)


def train_lstm_classifier(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    Train an LSTM-based sentiment classifier.
    :param args: Command-line args
    :param train_exs: training examples
    :param dev_exs: development set
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """
    # Fixed hyperparameters (matching train_and_plot_neural_models.py for reproducibility)
    # lr=0.0005, num_epochs=20, hidden_size=150, weight_decay=1e-5, dropout=0.3
    hidden_size = 150
    num_epochs = 20
    lr = 0.0005
    weight_decay = 1e-5

    # Get embedding layer
    embedding_layer = word_embeddings.get_initialized_embedding_layer(frozen=True)
    embedding_dim = word_embeddings.get_embedding_length()

    # Create network and move to GPU
    network = LSTMSentimentClassifier(embedding_layer, embedding_dim, hidden_size, num_classes=2, dropout=0.3)
    network.to(device)

    # Optimizer with weight decay
    optimizer = optim.Adam(network.parameters(), lr=lr, weight_decay=weight_decay)

    # Loss function
    loss_fn = nn.NLLLoss()

    # Training
    batch_size = args.batch_size
    word_indexer = word_embeddings.word_indexer

    network.train()

    for epoch in range(num_epochs):
        # Shuffle training data
        indices = list(range(len(train_exs)))
        random.shuffle(indices)

        total_loss = 0.0
        num_batches = 0

        if batch_size == 1:
            # Non-batched training
            for i in indices:
                ex = train_exs[i]

                # Convert words to indices
                word_indices = [word_indexer.add_and_get_index(word, add=False) for word in ex.words]
                word_indices = [idx if idx != -1 else 1 for idx in word_indices]

                # Create tensors and move to device
                x = torch.tensor(word_indices, dtype=torch.long).to(device)
                y = torch.tensor(ex.label, dtype=torch.long).to(device)

                # Forward pass
                optimizer.zero_grad()
                log_probs = network(x)
                loss = loss_fn(log_probs.unsqueeze(0), y.unsqueeze(0))

                # Backward pass
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1
        else:
            # Batched training
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]

                # Prepare batch
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

                # Pad sequences
                max_len = max(lengths)
                padded = [w + [0] * (max_len - len(w)) for w in batch_words]

                # Create tensors and move to device
                x = torch.tensor(padded, dtype=torch.long).to(device)
                y = torch.tensor(batch_labels, dtype=torch.long).to(device)
                lengths_tensor = torch.tensor(lengths, dtype=torch.long).to(device)

                # Forward pass
                optimizer.zero_grad()
                log_probs = network(x, lengths_tensor)
                loss = loss_fn(log_probs, y)

                # Backward pass
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        # Print progress
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    return NeuralSentimentClassifier(network, word_embeddings)


def train_cnn_classifier(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    Train a CNN-based sentiment classifier.
    :param args: Command-line args
    :param train_exs: training examples
    :param dev_exs: development set
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """
    # Fixed hyperparameters (matching train_and_plot_neural_models.py for reproducibility)
    # lr=0.0005, num_epochs=20, hidden_size=150, weight_decay=1e-5, dropout=0.3
    hidden_size = 150
    num_epochs = 20
    lr = 0.0005
    weight_decay = 1e-5

    # Get embedding layer
    embedding_layer = word_embeddings.get_initialized_embedding_layer(frozen=True)
    embedding_dim = word_embeddings.get_embedding_length()

    # Create network and move to GPU
    network = CNNSentimentClassifier(embedding_layer, embedding_dim, hidden_size, num_classes=2, dropout=0.3)
    network.to(device)

    # Optimizer with weight decay
    optimizer = optim.Adam(network.parameters(), lr=lr, weight_decay=weight_decay)

    # Loss function
    loss_fn = nn.NLLLoss()

    # Training
    batch_size = args.batch_size
    word_indexer = word_embeddings.word_indexer

    network.train()

    for epoch in range(num_epochs):
        # Shuffle training data
        indices = list(range(len(train_exs)))
        random.shuffle(indices)

        total_loss = 0.0
        num_batches = 0

        if batch_size == 1:
            # Non-batched training
            for i in indices:
                ex = train_exs[i]

                # Convert words to indices
                word_indices = [word_indexer.add_and_get_index(word, add=False) for word in ex.words]
                word_indices = [idx if idx != -1 else 1 for idx in word_indices]

                # Create tensors and move to device
                x = torch.tensor(word_indices, dtype=torch.long).to(device)
                y = torch.tensor(ex.label, dtype=torch.long).to(device)

                # Forward pass
                optimizer.zero_grad()
                log_probs = network(x)
                loss = loss_fn(log_probs.unsqueeze(0), y.unsqueeze(0))

                # Backward pass
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1
        else:
            # Batched training
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]

                # Prepare batch
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

                # Pad sequences
                max_len = max(lengths)
                padded = [w + [0] * (max_len - len(w)) for w in batch_words]

                # Create tensors and move to device
                x = torch.tensor(padded, dtype=torch.long).to(device)
                y = torch.tensor(batch_labels, dtype=torch.long).to(device)
                lengths_tensor = torch.tensor(lengths, dtype=torch.long).to(device)

                # Forward pass
                optimizer.zero_grad()
                log_probs = network(x, lengths_tensor)
                loss = loss_fn(log_probs, y)

                # Backward pass
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        # Print progress
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    return NeuralSentimentClassifier(network, word_embeddings)
