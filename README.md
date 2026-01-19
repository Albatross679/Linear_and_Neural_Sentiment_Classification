# Linear and Neural Sentiment Classification

A sentiment classification project implementing both traditional machine learning (Logistic Regression) and deep learning (Deep Averaging Network) approaches for binary sentiment analysis.

## Overview

This project provides two approaches to sentiment classification:

1. **Logistic Regression** with feature extraction (unigrams, bigrams, n-grams)
2. **Deep Averaging Network (DAN)** using pre-trained GloVe word embeddings

## Project Structure

```
Linear_and_Neural_Sentiment_Classification/
├── models.py                 # Core implementations (LR + DAN classifiers)
├── sentiment_classifier.py   # Main entry point and training pipeline
├── sentiment_data.py         # Data loading and preprocessing
├── utils.py                  # Utility functions and indexer
├── ffnn_example.py           # Feed-forward neural network example
├── data/
│   ├── train.txt             # Training data
│   ├── dev.txt               # Development/validation data
│   ├── test-blind.txt        # Test data (blind evaluation)
│   ├── glove.6B.50d-relativized.txt   # GloVe embeddings (50d)
│   └── glove.6B.300d-relativized.txt  # GloVe embeddings (300d)
├── ISSUES_AND_FIXES.md       # Documentation of bugs and solutions
└── setup.sh                  # Environment setup script
```

## Requirements

- Python 3.13+
- PyTorch 2.9+
- NumPy
- Matplotlib (optional, for visualization)

## Quick Start

### Environment Setup

```bash
# Run the setup script to create virtual environment and install dependencies
./setup.sh
```

Or manually:

```bash
# Create virtual environment with Python 3.13
python3.13 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install torch numpy matplotlib
```

### Training and Evaluation

```bash
# Activate virtual environment
source .venv/bin/activate

# Train Logistic Regression with unigram features
python sentiment_classifier.py --model LR --feats UNIGRAM

# Train Logistic Regression with bigram features
python sentiment_classifier.py --model LR --feats BIGRAM

# Train Logistic Regression with better features (unigrams + bigrams + trigrams)
python sentiment_classifier.py --model LR --feats BETTER

# Train Deep Averaging Network
python sentiment_classifier.py --model DAN
```

## Models

### Logistic Regression Classifier

- **Unigram Features**: Bag-of-words with individual word counts
- **Bigram Features**: Consecutive word pairs as features
- **Better Features**: Combination of unigrams, bigrams, trigrams, and length features

Best achieved accuracy: **77.8%** (Bigram features)

### Deep Averaging Network (DAN)

A neural approach that:
1. Embeds words using pre-trained GloVe vectors
2. Averages word embeddings across the sentence
3. Passes through fully connected layers with dropout

Best achieved accuracy: **79.2%** (batch_size=1)

## Performance Summary

| Model                    | Dev Accuracy | Train Time |
|--------------------------|--------------|------------|
| Trivial (baseline)       | 50.9%        | 0s         |
| LR + Unigram             | 77.5%        | 2.7s       |
| LR + Bigram              | 77.8%        | 4.7s       |
| LR + Better              | 77.2%        | 6.6s       |
| DAN (batch=1)            | 79.2%        | 118s       |
| DAN (batch=32)           | 77.6%        | 9.2s       |

## Implementation Notes

- Numerical stability in sigmoid handled via clipping
- Batched training with padding for variable-length sequences
- Dropout disabled during inference via `model.eval()`

See [ISSUES_AND_FIXES.md](ISSUES_AND_FIXES.md) for detailed bug documentation and solutions.

## License

MIT License
