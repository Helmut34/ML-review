# Decision Tree

A decision tree classifier built from scratch using recursive binary splits with Gini impurity.

## Dataset

**Digits (sklearn)** — 8x8 pixel images of handwritten digits (0-9), 10-class classification.

## Implementation

- **Algorithm:** Recursive binary splitting
- **Split criterion:** Gini impurity
- **Max depth:** 5 (to prevent overfitting)
- **Split selection:** Evaluates all features and possible thresholds, selecting the split that maximizes Gini impurity reduction

## Data Processing

- 80/20 train-test split
- StandardScaler normalization

## Results

- Classification accuracy on test set
