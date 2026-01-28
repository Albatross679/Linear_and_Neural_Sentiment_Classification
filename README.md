# Linear and Neural Sentiment Classification

A sentiment classification project implementing both traditional machine learning (Logistic Regression) and deep learning (Deep Averaging Network) approaches for binary sentiment analysis.

## Overview

This project provides two approaches to sentiment classification:

1. **Logistic Regression** with feature extraction (unigrams, bigrams, n-grams)
2. **Deep Averaging Network (DAN)** using pre-trained GloVe word embeddings

## Project Structure

```
Linear_and_Neural_Sentiment_Classification/
├── data/
│   ├── train.txt                       # Training data (6920 examples)
│   ├── dev.txt                         # Development/validation data (872 examples)
│   ├── test-blind.txt                  # Blind test data (1821 examples)
│   ├── glove.6B.50d-relativized.txt    # GloVe embeddings (50d)
│   └── glove.6B.300d-relativized.txt   # GloVe embeddings (300d)
├── doc/
│   ├── ISSUES_AND_FIXES.md             # Bug documentation and solutions
│   ├── writeup.tex                     # Main LaTeX writeup (ACL format)
│   └── *.md                            # Additional documentation
├── media/
│   ├── lr_epoch_loss.png               # LR training loss comparison (5 configs)
│   ├── dan_epoch_loss.png              # DAN training loss plot
│   └── neural_models_loss.png          # Neural models comparison plot
├── model/
│   ├── dan.pt                          # Saved DAN weights
│   ├── dan_batched.pt                  # Saved DAN (batched) weights
│   ├── lstm.pt                         # Saved LSTM weights
│   └── cnn.pt                          # Saved CNN weights
├── output/
│   ├── test-blind.LR-UNIGRAM.txt       # LR Unigram predictions
│   ├── test-blind.LR-BIGRAM.txt        # LR Bigram predictions
│   ├── test-blind.LR-BETTER.txt        # LR Better predictions
│   ├── test-blind.DAN.txt              # DAN predictions
│   ├── test-blind.LSTM.txt             # LSTM predictions
│   ├── test-blind.CNN.txt              # CNN predictions
│   └── *_results.txt                   # Experiment logs
├── script/
│   ├── setup.sh                        # Environment setup script
│   ├── ffnn_example.py                 # Feed-forward neural network example
│   ├── generate_lr_loss_plot.py        # Generate LR loss comparison plot
│   ├── generate_dan_loss_plot.py       # Generate DAN loss plot
│   ├── run_lr_experiments.py           # Run all LR experiments
│   ├── run_neural_experiments.py       # Run all neural experiments
│   └── run_neural_models.py            # Train LSTM/CNN models
├── src/
│   ├── models.py                       # All model implementations
│   ├── sentiment_classifier.py         # Main entry point (LR, DAN)
│   └── sentiment_data.py               # Data loading and preprocessing
└── utilities/
    └── utils.py                        # Indexer utility class
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
./script/setup.sh
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
python -m src.sentiment_classifier --model LR --feats UNIGRAM

# Train Logistic Regression with bigram features
python -m src.sentiment_classifier --model LR --feats BIGRAM

# Train Logistic Regression with better features (unigrams + bigrams + trigrams)
python -m src.sentiment_classifier --model LR --feats BETTER

# Train Deep Averaging Network
python -m src.sentiment_classifier --model DAN

# Train LSTM (via script)
python script/run_neural_models.py --model LSTM --test_output_path output/test-blind.LSTM.txt

# Train CNN (via script)
python script/run_neural_models.py --model CNN --test_output_path output/test-blind.CNN.txt
```

## Models

### Logistic Regression Classifier

- **Unigram Features**: Bag-of-words with individual word counts
- **Bigram Features**: Consecutive word pairs as features
- **Better Features**: Combination of unigrams, bigrams, trigrams, and length features

Best achieved accuracy: **78.3%** (Unigram features)

### Deep Averaging Network (DAN)

A neural approach that:
1. Embeds words using pre-trained GloVe vectors
2. Averages word embeddings across the sentence
3. Passes through fully connected layers with dropout

Best achieved accuracy: **78.6%** (batch_size=32)

### LSTM Classifier

Bidirectional LSTM that captures sequential dependencies:
1. Embeds words using pre-trained GloVe vectors
2. Processes sequence with bidirectional LSTM
3. Concatenates final hidden states from both directions

Best achieved accuracy: **82.5%** (batch_size=32)

### CNN Classifier

Convolutional neural network with multi-scale filters:
1. Embeds words using pre-trained GloVe vectors
2. Applies parallel convolutions with kernel sizes 2, 3, 4, 5
3. Max-over-time pooling and concatenation

Best achieved accuracy: **81.2%** (batch_size=32)

## Performance Summary

| Model                    | Dev Accuracy | Dev F1  | Train Time |
|--------------------------|--------------|---------|------------|
| Trivial (baseline)       | 50.9%        | 67.5%   | 0s         |
| LR + Unigram             | 78.3%        | 79.1%   | ~5s        |
| LR + Bigram              | 77.4%        | 78.3%   | ~8s        |
| LR + Better              | 76.8%        | 77.8%   | ~12s       |
| DAN (batch=1)            | 78.0%        | 78.3%   | ~145s      |
| DAN (batch=32)           | 78.6%        | 80.4%   | ~15s       |
| **LSTM (batch=32)**      | **82.5%**    | **83.6%** | ~67s     |
| CNN (batch=32)           | 81.2%        | 82.0%   | ~35s       |

## Implementation Notes

- Numerical stability in sigmoid handled via clipping
- Batched training with padding for variable-length sequences
- Dropout disabled during inference via `model.eval()`

See [doc/ISSUES_AND_FIXES.md](doc/ISSUES_AND_FIXES.md) for detailed bug documentation and solutions.

## License

MIT License
