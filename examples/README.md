# ML Framework Examples

Collection of ready-to-run datasets for training neural networks.

## Available Examples

| Dataset | Command | Features | Classes | Time |
|---------|---------|----------|---------|------|
| **Iris** | `./build/ml_trainer examples/iris/config.txt` | 4 | 3 | < 1 min |
| **Breast Cancer** | `./build/ml_trainer examples/breast_cancer/config.txt` | 30 | 2 | 1-2 min |
| **MNIST** | `./build/ml_trainer examples/mnist/config.txt` | 784 | 10 | 5-10 min |
| **CIFAR-10** | `./build/ml_trainer examples/cifar10/config.txt` | 3,072 | 10 | 10-15 min |

## Quick Start

### Iris (Fastest)
```bash
python examples/iris/prepare.py
./build/ml_trainer examples/iris/config.txt
```
Expected: 96-98% accuracy in < 1 minute

### MNIST
```bash
python examples/mnist/prepare.py
./build/ml_trainer examples/mnist/config.txt
```
Expected: 93-94% accuracy in 5-10 minutes

Or quick test (2 epochs):
```bash
./build/ml_trainer examples/mnist/config_quick.txt
```

## Per-Dataset Documentation

Each dataset has its own folder with a README and preparation script:

- **[examples/iris](iris/)** - Small, fast, perfect for testing
- **[examples/breast_cancer](breast_cancer/)** - Real medical data
- **[examples/mnist](mnist/)** - Classic handwritten digits
- **[examples/cifar10](cifar10/)** - Complex image classification

## Creating Your Own Dataset

1. Copy the template:
   ```bash
   cp examples/template_config.txt examples/my_dataset_config.txt
   ```

2. Prepare your data in `.mat` format. See preparation scripts (`examples/*/prepare.py`) for examples.

3. Edit config with your dataset dimensions

4. Train:
   ```bash
   ./build/ml_trainer examples/my_dataset_config.txt
   ```

## Data Format

Data files are binary `.mat` format (32-bit floats, row-major). Example from `examples/iris/prepare.py`:

```python
import numpy as np
import struct

def save_mat(filename, data):
    data = np.asarray(data, dtype=np.float32)
    with open(filename, 'wb') as f:
        flat = data.flatten()
        f.write(struct.pack(f'{len(flat)}f', *flat))

# Save features (samples × features)
save_mat('train_features.mat', X_train)
# Save labels (samples × 1)
save_mat('train_labels.mat', y_train.reshape(-1, 1))
```

## Label Formats

The framework supports three label formats in config:

**Integer indices** (single column):
```
0
2
1
...
```
Set: `label_format = integer_indices`

**One-hot encoded** (multiple columns):
```
1 0 0
0 1 0
0 0 1
...
```
Set: `label_format = one_hot`

**Raw floats** (soft labels/regression):
```
0.9 0.1 0.0
0.1 0.8 0.1
...
```
Set: `label_format = raw_floats`

## Configuration Reference

See `template_config.txt` for a full template. Key sections:

```ini
[dataset]
train_images = file.mat
train_labels = file.mat
test_images = file.mat
test_labels = file.mat
train_size = 100        # number of training samples
test_size = 20
input_size = 50         # features per sample
output_size = 3         # number of classes
label_format = integer_indices

[training]
epochs = 10
batch_size = 32
learning_rate = 0.01

[architecture]
layer = dense 50 32 relu
layer = dense 32 3 softmax
```

## Troubleshooting

**Data files not found?**
- Make sure you ran the preparation script: `python examples/dataset/prepare.py`
- Files should be created in the `examples/` directory

**Out of memory?**
- Reduce `batch_size` in config (e.g., 16 instead of 128)

**Training is slow?**
- Use fewer epochs for testing: `epochs = 2`

**Model not learning?**
- Increase learning_rate: try `0.01` or `0.05`
- Reduce batch_size for noisier gradients
- Check data normalization (features should be ~0-1 or ~0-mean)
