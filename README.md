[![Test](https://github.com/SermetPekin/ML-C-example/actions/workflows/test.yml/badge.svg?2)](https://github.com/SermetPekin/ML-C-example/actions/workflows/test.yml?2)

# Machine Learning in C

A generic neural network framework in pure C. Configure the architecture and data paths via YAML, compile once, run on any dataset. No dependencies beyond standard C—all matrix operations, training, and memory management built from scratch.

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

- **Config-driven architecture**: Define network topology without changing code
- **Multiple label formats**: Auto-detects or accepts integer indices, one-hot, raw floats
- **Generic training**: Same binary works with any dataset—change config, not code
- **Validated datasets**: Tested on Iris, Breast Cancer, MNIST, CIFAR-10

## Tests

```bash
make test              # Run all tests
make test_matrix       # Matrix operations
make test_arena        # Memory allocator
make test_prng         # Random number generation
```

## Notes

Fork of [Magicalbat](https://github.com/Magicalbat)'s implementation. Decoupled from MNIST-specific code, added config-driven architecture, multi-format label support, and comprehensive testing infrastructure.
