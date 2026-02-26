# Random Forest

A random forest classifier using an ensemble of decision trees with bootstrap aggregation.

## Dataset

**Digits (sklearn)** — 8x8 pixel images of handwritten digits (0-9), 10-class classification.

## Implementation

- **Algorithm:** Ensemble of decision trees with majority voting
- **Number of trees:** 100
- **Max depth:** 5 per tree
- **Feature selection:** `sqrt(n_features)` considered at each split
- **Sampling:** Bootstrap sampling for each tree
- **Random state:** 42 for reproducibility

## Data Processing

- 80/20 train-test split
- StandardScaler normalization

## Results

- Classification accuracy on test set
