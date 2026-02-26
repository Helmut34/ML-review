# LightGBM

A LightGBM classifier using leaf-wise tree growth with L1/L2 regularization.

## Dataset

**Synthetic classification dataset**

## Implementation

- **Algorithm:** Light Gradient Boosting Machine (LGBMClassifier)
- **Boosting rounds:** 200
- **Tree growth:** Leaf-wise (no max depth limit)
- **Number of leaves:** 31
- **Learning rate:** 0.1
- **Subsample:** 0.8
- **Column subsampling:** 0.8
- **Regularization:** L1 (alpha=0.1) and L2 (lambda=1.0)
- **Min child samples:** 20 (prevents overfitting on small leaf nodes)

## Data Processing

- Eval set monitoring on test data

## Results

- Classification accuracy on test set
