# Iris - Flower Classification

**Dataset:** 150 samples (120 train, 30 test)
**Features:** 4 (sepal/petal measurements)
**Classes:** 3 (Setosa, Versicolor, Virginica)
**Expected Accuracy:** 95-98%
**Training Time:** < 1 minute

## Quick Start

```bash
# 1. Prepare data (from the examples/iris directory)
cd examples/iris
python prepare.py

# 2. Train model with SGD (from project root)
cd ../..
./build/ml_trainer examples/iris/config.txt

# 3. Train model with Adam optimizer
./build/ml_trainer examples/iris/iris_adam.txt
```

## Data Location

Data files are automatically saved to `examples/iris/data/` directory:
- `iris_train_features.mat` - Training features (120 × 4)
- `iris_train_labels.mat` - Training labels (120 × 1)
- `iris_test_features.mat` - Test features (30 × 4)
- `iris_test_labels.mat` - Test labels (30 × 1)

## Expected Output

```
Test Completed. Accuracy: 29 / 30 (96.7%), Average Cost: 0.1234
```

## Why Iris?

Perfect for:
- Testing the framework (tiny dataset, fast training)
- Hyperparameter experiments
- Understanding model behavior on small data

## Optimizers

This example includes configurations for two optimizers:

- **SGD** (`config.txt`) - Stochastic Gradient Descent with learning rate 0.01
- **Adam** (`iris_adam.txt`) - Adaptive Moment Estimation with learning rate 0.001

Compare their performance by running both and observing convergence speed and final accuracy.

## Experiments

**Try different architectures:**
```ini
[architecture]
layer = dense 4 16 relu
layer = dense 16 8 relu
layer = dense 8 3 softmax
```

**Change learning rate:**
```ini
[training]
learning_rate = 0.001
epochs = 50
```
