# Sentiment Classification - Issues and Fixes

This document lists all bugs and issues encountered during implementation and how they were resolved.

## Environment Setup

### Issue 1: Python Version Check
**Problem**: Needed to verify Python 3.13 was available on system.  
**Solution**: Ran `python3.13 --version` which confirmed Python 3.13.11.

### Issue 2: PyTorch Installation
**Problem**: Initial pip install took some time due to large CUDA dependencies.  
**Solution**: Used `pip install torch numpy matplotlib` which installed torch-2.9.1 with CUDA support (~900MB download).

---

## Logistic Regression Implementation

### Issue 3: Numerical Instability in Sigmoid
**Problem**: Very large or small scores could cause overflow in `exp(-score)`.  
**Solution**: Added clipping: `np.exp(-np.clip(score, -500, 500))` to prevent overflow.

### Issue 4: Learning Rate Tuning
**Problem**: Initial learning rate of 0.001 was too small; training was slow.  
**Solution**: Increased learning rate to 0.5 with 0.95x decay per epoch. Achieved 77.5% dev accuracy in 30 epochs.

### Issue 5: Feature Index Out of Bounds
**Problem**: At test time, unseen features could return -1 from indexer.  
**Solution**: Check `if idx != -1` before accessing weights, and also ensure `idx < len(self.weights)` in predict method.

---

## Deep Averaging Network Implementation

### Issue 6: Embedding Layer Initialization
**Problem**: Needed to ensure pretrained embeddings were properly loaded.  
**Solution**: Used `word_embeddings.get_initialized_embedding_layer(frozen=True)` which handles PAD/UNK tokens automatically.

### Issue 7: Slow Training Speed (Unbatched)
**Problem**: Training 15 epochs took ~118 seconds with batch_size=1.  
**Solution**: Implemented batched training with padding. With batch_size=32, training dropped to ~9 seconds (13x speedup).

### Issue 8: Handling Variable Sequence Lengths
**Problem**: Different sentences have different lengths; cannot form regular tensors.  
**Solution**: Implemented padding to max_length with PAD token (index 0), and used length masking when computing averages.

### Issue 9: Dropout During Inference
**Problem**: Dropout should not be applied during evaluation.  
**Solution**: Called `network.eval()` in `NeuralSentimentClassifier.__init__()` to disable dropout for inference.

---

## Feature Extractors

### Issue 10: Feature Naming Conflicts
**Problem**: Need to distinguish unigrams from bigrams when combining features.  
**Solution**: Used prefixes: `"UNIGRAM=word"`, `"BIGRAM=word1|word2"`, `"TRIGRAM=word1|word2|word3"`.

---

## Summary Table

| Model                    | Dev Accuracy | Train Time |
|--------------------------|--------------|------------|
| Trivial (baseline)       | 50.9%        | 0s         |
| LR + Unigram             | 77.5%        | 2.7s       |
| LR + Bigram              | 77.8%        | 4.7s       |
| LR + Better              | 77.2%        | 6.6s       |
| DAN (batch=1)            | 79.2%        | 118s       |
| DAN (batch=32)           | 77.6%        | 9.2s       |

---

## Files Modified

- **models.py** - All implementations (as specified in assignment)

## Files NOT Modified (per assignment instructions)

- sentiment_classifier.py
- sentiment_data.py
- utils.py
