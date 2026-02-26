# XGBoost

An XGBoost (eXtreme Gradient Boosting) classifier with L1/L2 regularization and feature importance analysis.

## Dataset

**Synthetic (sklearn)** — 5000 samples, 20 features (10 informative), binary classification.

## Implementation

- **Algorithm:** Gradient boosted decision trees (XGBClassifier)
- **Boosting rounds:** 200
- **Max depth:** 6
- **Learning rate:** 0.1
- **Subsample:** 0.8 (fraction of samples per tree)
- **Column subsampling:** 0.8 (fraction of features per tree)
- **Regularization:** L1 (alpha=0.1) and L2 (lambda=1.0)
- **Eval metric:** Log loss with eval set monitoring

## Data Processing

- 80/20 train-test split
- StandardScaler normalization

## Results

- Classification accuracy on test set
- Top 5 feature importances
