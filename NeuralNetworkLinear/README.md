# Neural Network — Regression

A 3-layer multi-layer perceptron (MLP) for regression, built manually with PyTorch tensors.

## Dataset

**California Housing** — 8 features describing housing attributes with median house price as the target.

## Implementation

- **Architecture:**
  - Input layer: 8 features
  - Hidden layer 1: 64 neurons (ReLU)
  - Hidden layer 2: 32 neurons (ReLU)
  - Output layer: 1 neuron (linear)
- **Loss:** Mean Squared Error (MSE)
- **Optimizer:** Adam (lr=0.001)
- **Epochs:** 600
- **Backpropagation:** Manual gradient computation and parameter updates

## Data Processing

- 80/20 train-test split
- StandardScaler normalization
- Converted to PyTorch tensors

## Results

- Improved MSE over the basic LinearRegression implementation
- Training loss curve saved to `loss_curve.png`
