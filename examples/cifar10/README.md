# CIFAR-10 - Image Classification

**Dataset:** 60,000 images (50,000 train, 10,000 test)
**Features:** 3,072 (32×32×3 RGB flattened)
**Classes:** 10 (airplane, car, bird, cat, deer, dog, frog, horse, ship, truck)
**Expected Accuracy:** 60-75%
**Training Time:** 10-15 minutes

## Quick Start

```bash
# 1. Prepare data (from the examples/cifar10 directory, first time only ~160MB download)
cd examples/cifar10
python prepare.py

# 2. Train model (from project root)
cd ../..
./build/ml_trainer examples/cifar10/config.txt
```

## Data Location

Data files are automatically saved to `examples/cifar10/data/` directory:
- `cifar10_train_images.mat` - Training images (50,000 × 3,072)
- `cifar10_train_labels.mat` - Training labels (50,000 × 1)
- `cifar10_test_images.mat` - Test images (10,000 × 3,072)
- `cifar10_test_labels.mat` - Test labels (10,000 × 1)

## Expected Output

```
Test Completed. Accuracy: 6500 / 10000 (65%), Average Cost: 1.2345
```

## Why Lower Accuracy?

CIFAR-10 images have complex spatial structure. Fully-connected networks struggle because:
- Images have local patterns (edges, textures) that **convolutional layers** capture better
- Modern models (CNNs) achieve 95%+ with specialized architecture

This is a baseline, not a weakness of the framework!

## How to Improve

1. **More epochs:** Change `epochs = 30` (takes longer)
2. **Deeper network:** Add more layers (slower but better)
3. **Smaller learning rate:** Try `learning_rate = 0.001` (more stable)
4. **Smaller batches:** Change `batch_size = 16` (noisier updates, better generalization)

Example:
```ini
[training]
epochs = 30
batch_size = 32
learning_rate = 0.005

[architecture]
layer = dense 3072 512 relu
layer = dense 512 256 relu
layer = dense 256 128 relu
layer = dense 128 10 softmax
```

## Note

For production image classification, you'd use convolutional neural networks (CNNs), which aren't yet implemented in this framework.
