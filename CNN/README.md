# Convolutional Neural Network

A CNN for image classification on CIFAR-10, using batch normalization, dropout, and data augmentation.

## Dataset

**CIFAR-10** — 60,000 32x32 color images across 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).

## Implementation

- **Architecture:**
  - Conv2d: 3 → 32 filters (3x3 kernel) + BatchNorm + ReLU + MaxPool(2x2)
  - Conv2d: 32 → 64 filters (3x3 kernel) + BatchNorm + ReLU + MaxPool(2x2)
  - Fully connected: 64x8x8 → 128 → 10 output classes
- **Regularization:**
  - Batch Normalization after each conv layer
  - Dropout (0.5) before the output layer
- **Data Augmentation:**
  - RandomHorizontalFlip
  - RandomCrop with padding
  - Normalization with CIFAR-10 mean/std
- **Loss:** CrossEntropyLoss
- **Optimizer:** Adam (lr=0.001)
- **Epochs:** 250
- **Batch size:** 64
- **Device:** Automatic GPU/CPU detection

## Data Processing

- CIFAR-10 loaded via torchvision
- Separate train/test transforms (augmentation only on training)

## Results

- Tracks train and validation loss/accuracy per epoch
- Loss and accuracy curves plotted after training
