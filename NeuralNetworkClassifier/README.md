# Neural Network — Classification

A 3-layer multi-layer perceptron (MLP) for binary classification, built manually with PyTorch tensors.

## Dataset

**Breast Cancer (sklearn)** — 30 features computed from breast mass images. Binary classification: malignant vs benign.

## Implementation

- **Architecture:**
  - Input layer: 30 features
  - Hidden layer 1: 64 neurons (ReLU)
  - Hidden layer 2: 32 neurons (ReLU)
  - Output layer: 1 neuron (sigmoid)
- **Weight initialization:** He initialization for improved convergence
- **Loss:** Binary Cross-Entropy (BCELoss)
- **Optimizer:** Adam (lr=0.001)
- **Epochs:** 1000
- **Decision threshold:** 0.5

## Data Processing

- 80/20 train-test split
- StandardScaler normalization
- Converted to PyTorch tensors

## Results

- Binary classification accuracy on test set
- Training loss curve saved to `loss_curve.png`
