# Linear Regression

A manual implementation of linear regression using PyTorch tensors, without relying on `torch.nn` layers.

## Dataset

**California Housing** — 8 features describing housing attributes (median income, house age, average rooms, etc.) with median house price as the target.

## Implementation

- **Model:** Manual weight matrix `W` (8x1) and bias `b`, computing `y = X @ W + b`
- **Loss:** Mean Squared Error (MSE)
- **Optimization:** Manual gradient descent with backpropagation
- **Learning rate:** 0.01
- **Epochs:** 250

## Data Processing

- 80/20 train-test split
- StandardScaler normalization (zero mean, unit variance)
- Converted to PyTorch tensors

## Results

- Training loss curve saved to `loss_curve.png`
- Final test MSE reported after training
