[![Test](https://github.com/SermetPekin/ML-C-example/actions/workflows/test.yml/badge.svg?2)](https://github.com/SermetPekin/ML-C-example/actions/workflows/test.yml?2)

# Machine Learning in C

A neural network framework in C. Configure the architecture and data paths in a text config file, compile once, run on any dataset. No dependencies beyond standard C—all matrix operations, training, and memory management built from scratch.

## Setup

```bash
python -m pip install uv
uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
python -m pip install -r requirements.txt
python mnist.py            # Prepare a dataset
```

## Usage

Define your network in a config file (see `examples/mnist_config.txt`):

```
[dataset]
input_size = 784
output_size = 10
num_classes = 10

[training]
epochs = 30
batch_size = 32
learning_rate = 0.01
optimizer = sgd          # or 'adam'

[architecture]
dense 128 relu
dense 10 softmax
```

Then compile and run with that config:

```bash
make CONFIG=examples/mnist_config.txt
make run CONFIG=examples/mnist_config.txt
```

## Features

- **Config-driven architecture**: Change the network structure without modifying code
- **Multiple optimizers**: Support for SGD and Adam optimizers with configurable hyperparameters
- **Multiple label formats**: Auto-detects or accepts integer indices, one-hot, raw floats
- **Generic training**: Same binary works with any dataset—change config, not code
- **Validated datasets**: Tested on Iris, Breast Cancer, MNIST, CIFAR-10
- **Comprehensive metrics**: Per-class precision, recall, F1-score and overall accuracy
- **Confusion matrix**: Visual representation of prediction accuracy per class
- **Results export**: Evaluation results saved to timestamped files in table format

## Optimizers

The framework supports multiple optimization algorithms. Configure your choice in the `[training]` section:

### SGD (Stochastic Gradient Descent)
```
[training]
optimizer = sgd
learning_rate = 0.01
```

### Adam
Uses adaptive learning rates with momentum. Configure with these hyperparameters:
```
[training]
optimizer = adam
learning_rate = 0.001        # Adam typically uses smaller learning rates
adam_beta1 = 0.9             # Decay rate for 1st moment estimate
adam_beta2 = 0.999           # Decay rate for 2nd moment estimate
adam_epsilon = 1e-8          # Numerical stability constant
```

See `examples/mnist/mnist_adam.txt` for a complete Adam configuration example.

## Evaluation & Metrics

After training, the framework automatically evaluates your model on the test set and generates comprehensive metrics:

**Per-Class Metrics (displayed in table format):**
- **Precision**: Ratio of correct positive predictions to all positive predictions
- **Recall**: Ratio of correct positive predictions to all actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **Support**: Number of samples for each class

**Overall Metrics:**
- **Overall Accuracy**: Percentage of correct predictions across all classes
- **Average Loss**: Mean loss value on the test set

**Confusion Matrix:**
Shows actual vs predicted class labels, helping identify which classes are being confused with each other.

**Results Export:**
All evaluation results are automatically saved to a timestamped file (e.g., `results_20260109_010005.txt`) in a formatted table layout for easy review and comparison across different training runs.

## Tests

```bash
make test              # Run all tests
make test_matrix       # Matrix operations
make test_arena        # Memory allocator
make test_prng         # Random number generation
```

## Notes

Based on [Magicalbat](https://github.com/Magicalbat)'s implementation. Major enhancements include:
- Decoupled from MNIST-specific code for generic dataset support
- Config-driven architecture (no code changes needed for different networks)
- Multi-format label support (integer indices, one-hot, raw floats)
- Multiple optimization algorithms (SGD and Adam)
