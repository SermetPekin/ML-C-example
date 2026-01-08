# MNIST - Handwritten Digit Recognition

**Dataset:** 70,000 images (60,000 train, 10,000 test)
**Features:** 784 (28×28 pixels, grayscale)
**Classes:** 10 (digits 0-9)
**Expected Accuracy:** 93-94%
**Training Time:** 5-10 minutes

## Quick Start

```bash
# 1. Prepare data (from the examples/mnist directory, first time only ~50MB download)
cd examples/mnist
python prepare.py

# 2. Train full model (from project root, 10 epochs)
cd ../..
./build/ml_trainer examples/mnist/config.txt

# 3. Quick test (from project root, 2 epochs ~1 minute)
./build/ml_trainer examples/mnist/config_quick.txt

# 4. Adam opt
./build/ml_trainer examples/mnist/mnist_adam.txt

```

## Data Location

Data files are automatically saved to `examples/mnist/data/` directory:
- `mnist_train_images.mat` - Training images (60,000 × 784)
- `mnist_train_labels.mat` - Training labels (60,000 × 1)
- `mnist_test_images.mat` - Test images (10,000 × 784)
- `mnist_test_labels.mat` - Test labels (10,000 × 1)

## Expected Output

```
Epoch 10 / 10, Batch 1200 / 1200, Average Cost: 0.2120
Test Completed. Accuracy: 9382 / 10000 (93.8%), Average Cost: 0.2154
```

## Experiments

**Faster training:** Edit config, change `epochs = 2`

**Better accuracy:** Edit config, change `epochs = 30`

**Deeper network:** Edit config architecture:
```ini
[architecture]
layer = dense 784 256 relu
layer = dense 256 128 relu
layer = dense 128 64 relu
layer = dense 64 10 softmax
```
