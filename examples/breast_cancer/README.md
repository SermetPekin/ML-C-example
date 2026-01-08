# Breast Cancer Wisconsin - Medical Diagnosis

**Dataset:** 569 samples (455 train, 114 test)
**Features:** 30 diagnostic measurements
**Classes:** 2 (Benign, Malignant)
**Expected Accuracy:** 95-97%
**Training Time:** 1-2 minutes

## Quick Start

```bash
# 1. Prepare data (from the examples/breast_cancer directory)
cd examples/breast_cancer
python prepare.py

# 2. Train model with SGD (from project root)
cd ../..
./build/ml_trainer examples/breast_cancer/config.txt

# 3. Train model with Adam optimizer
./build/ml_trainer examples/breast_cancer/breast_cancer_adam.txt
```

## Data Location

Data files are automatically saved to `examples/breast_cancer/data/` directory:
- `breast_cancer_train_features.mat` - Training features (455 × 30)
- `breast_cancer_train_labels.mat` - Training labels (455 × 1)
- `breast_cancer_test_features.mat` - Test features (114 × 30)
- `breast_cancer_test_labels.mat` - Test labels (114 × 1)

## Expected Output

```
Test Completed. Accuracy: 111 / 114 (97.4%), Average Cost: 0.1567
```

## Real-World Application

This dataset contains actual medical diagnostic data. The model learns to classify tumors as benign or malignant based on cell measurements—a practical classification problem.

## Optimizers

This example includes configurations for two optimizers:

- **SGD** (`config.txt`) - Stochastic Gradient Descent with learning rate 0.01
- **Adam** (`breast_cancer_adam.txt`) - Adaptive Moment Estimation with learning rate 0.001

Compare their performance by running both and observing convergence speed and final accuracy.

## Experiments

**Improve accuracy:**
```ini
[training]
epochs = 50
learning_rate = 0.005
batch_size = 16
```

**Different architecture:**
```ini
[architecture]
layer = dense 30 32 relu
layer = dense 32 16 relu
layer = dense 16 8 relu
layer = dense 8 2 softmax
```

## Notes

Features are standardized (mean=0, std=1) because they have different scales and ranges. This helps the optimizer train faster.
