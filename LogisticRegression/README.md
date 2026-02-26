# Logistic Regression

A manual implementation of logistic regression using PyTorch tensors for binary classification.

## Dataset

**Breast Cancer (sklearn)** — 30 features computed from digitized images of breast mass fine needle aspirates. Binary classification: malignant vs benign.

## Implementation

- **Model:** Manual weight matrix `W` (30x1) and bias `b` with sigmoid activation: `sigmoid(X @ W + b)`
- **Loss:** Binary Cross-Entropy (BCELoss)
- **Optimization:** Manual gradient descent with backpropagation
- **Learning rate:** 0.01
- **Epochs:** 1000
- **Decision threshold:** 0.5

## Data Processing

- 80/20 train-test split
- StandardScaler normalization
- Converted to PyTorch tensors

## Results

- Binary classification accuracy on test set
- Training loss curve saved to `loss_curve.png`
