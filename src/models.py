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

    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

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
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        features = Counter()
        for word in sentence:
            # Words are already lowercased by the data reader
            feature_name = "UNIGRAM=" + word
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(feature_name, add=True)
            else:
                idx = self.indexer.index_of(feature_name)
            if idx != -1:
                features[idx] += 1
        return features


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """

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


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    Uses: unigrams, bigrams, trigrams, and length features with clipped counts.
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        features = Counter()
        
        # Unigrams with clipped counts (presence/absence)
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


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    # First pass: extract features and grow the indexer
    train_features = []
    for ex in train_exs:
        features = feat_extractor.extract_features(ex.words, add_to_indexer=True)
        train_features.append(features)
    
    # Initialize weights with random values for better convergence
    num_features = len(feat_extractor.get_indexer())
    np.random.seed(42)  # Fixed seed for reproducibility
    weights = np.random.uniform(-0.1, 0.1, num_features)
    
    # Hyperparameters
    num_epochs = 30
    learning_rate = 0.5
    
    # SGD training
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
            
            # Sigmoid
            prob = 1.0 / (1.0 + np.exp(-np.clip(score, -500, 500)))
            
            # Gradient update: w = w + alpha * (y - prob) * x
            y = ex.label
            gradient_scalar = y - prob
            
            for idx, count in features.items():
                weights[idx] += learning_rate * gradient_scalar * count
        
        # Optional: decrease learning rate each epoch
        learning_rate *= 0.95
    
    return LogisticRegressionClassifier(weights, feat_extractor)


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

    # Train the model
    model = train_logistic_regression(train_exs, feat_extractor)
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
        self.network.eval()

    def predict(self, ex_words: List[str]) -> int:
        # Convert words to indices
        word_indexer = self.word_embeddings.word_indexer
        indices = [word_indexer.add_and_get_index(word, add=False) for word in ex_words]
        # Replace -1 (unknown) with UNK index (1)
        indices = [idx if idx != -1 else 1 for idx in indices]
        
        # Create tensor
        indices_tensor = torch.tensor(indices, dtype=torch.long)
        
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
        
        # Create tensors
        batch_tensor = torch.tensor(padded, dtype=torch.long)
        lengths_tensor = torch.tensor(lengths, dtype=torch.long)
        
        with torch.no_grad():
            log_probs = self.network(batch_tensor, lengths_tensor)
            predictions = torch.argmax(log_probs, dim=1).tolist()
        
        return predictions


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    Main entry point for your deep averaging network model.
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """
    # Get embedding layer
    embedding_layer = word_embeddings.get_initialized_embedding_layer(frozen=True)
    embedding_dim = word_embeddings.get_embedding_length()
    hidden_size = args.hidden_size
    
    # Create network
    network = DeepAveragingNetwork(embedding_layer, embedding_dim, hidden_size, num_classes=2, dropout=0.3)
    
    # Optimizer
    optimizer = optim.Adam(network.parameters(), lr=args.lr)
    
    # Loss function
    loss_fn = nn.NLLLoss()
    
    # Training
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    word_indexer = word_embeddings.word_indexer
    
    network.train()
    
    for epoch in range(num_epochs):
        # Shuffle training data
        indices = list(range(len(train_exs)))
        random.shuffle(indices)
        
        total_loss = 0.0
        
        if batch_size == 1:
            # Non-batched training
            for i in indices:
                ex = train_exs[i]
                
                # Convert words to indices
                word_indices = [word_indexer.add_and_get_index(word, add=False) for word in ex.words]
                word_indices = [idx if idx != -1 else 1 for idx in word_indices]
                
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
        else:
            # Batched training (Exploration task)
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
                
                # Create tensors
                x = torch.tensor(padded, dtype=torch.long)
                y = torch.tensor(batch_labels, dtype=torch.long)
                lengths_tensor = torch.tensor(lengths, dtype=torch.long)
                
                # Forward pass
                optimizer.zero_grad()
                log_probs = network(x, lengths_tensor)
                loss = loss_fn(log_probs, y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
        
        # Print progress
        avg_loss = total_loss / len(train_exs)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    return NeuralSentimentClassifier(network, word_embeddings)


def train_lstm_classifier(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    Train an LSTM-based sentiment classifier.
    :param args: Command-line args
    :param train_exs: training examples
    :param dev_exs: development set
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """
    # Get embedding layer
    embedding_layer = word_embeddings.get_initialized_embedding_layer(frozen=True)
    embedding_dim = word_embeddings.get_embedding_length()
    hidden_size = args.hidden_size

    # Create network
    network = LSTMSentimentClassifier(embedding_layer, embedding_dim, hidden_size, num_classes=2, dropout=0.3)

    # Optimizer
    optimizer = optim.Adam(network.parameters(), lr=args.lr)

    # Loss function
    loss_fn = nn.NLLLoss()

    # Training
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    word_indexer = word_embeddings.word_indexer

    network.train()

    for epoch in range(num_epochs):
        # Shuffle training data
        indices = list(range(len(train_exs)))
        random.shuffle(indices)

        total_loss = 0.0

        if batch_size == 1:
            # Non-batched training
            for i in indices:
                ex = train_exs[i]

                # Convert words to indices
                word_indices = [word_indexer.add_and_get_index(word, add=False) for word in ex.words]
                word_indices = [idx if idx != -1 else 1 for idx in word_indices]

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

                # Create tensors
                x = torch.tensor(padded, dtype=torch.long)
                y = torch.tensor(batch_labels, dtype=torch.long)
                lengths_tensor = torch.tensor(lengths, dtype=torch.long)

                # Forward pass
                optimizer.zero_grad()
                log_probs = network(x, lengths_tensor)
                loss = loss_fn(log_probs, y)

                # Backward pass
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

        # Print progress
        avg_loss = total_loss / len(train_exs)
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
    # Get embedding layer
    embedding_layer = word_embeddings.get_initialized_embedding_layer(frozen=True)
    embedding_dim = word_embeddings.get_embedding_length()
    hidden_size = args.hidden_size

    # Create network
    network = CNNSentimentClassifier(embedding_layer, embedding_dim, hidden_size, num_classes=2, dropout=0.3)

    # Optimizer
    optimizer = optim.Adam(network.parameters(), lr=args.lr)

    # Loss function
    loss_fn = nn.NLLLoss()

    # Training
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    word_indexer = word_embeddings.word_indexer

    network.train()

    for epoch in range(num_epochs):
        # Shuffle training data
        indices = list(range(len(train_exs)))
        random.shuffle(indices)

        total_loss = 0.0

        if batch_size == 1:
            # Non-batched training
            for i in indices:
                ex = train_exs[i]

                # Convert words to indices
                word_indices = [word_indexer.add_and_get_index(word, add=False) for word in ex.words]
                word_indices = [idx if idx != -1 else 1 for idx in word_indices]

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

                # Create tensors
                x = torch.tensor(padded, dtype=torch.long)
                y = torch.tensor(batch_labels, dtype=torch.long)
                lengths_tensor = torch.tensor(lengths, dtype=torch.long)

                # Forward pass
                optimizer.zero_grad()
                log_probs = network(x, lengths_tensor)
                loss = loss_fn(log_probs, y)

                # Backward pass
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

        # Print progress
        avg_loss = total_loss / len(train_exs)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    return NeuralSentimentClassifier(network, word_embeddings)
