[![Test](https://github.com/SermetPekin/ML-C-example/actions/workflows/test.yml/badge.svg?2)](https://github.com/SermetPekin/ML-C-example/actions/workflows/test.yml?2)

# Machine Learning in C

A machine learning project written in C, featuring neural network implementations and utilities for working with the MNIST dataset.

## Getting Started

### Prerequisites

Make sure you have Python and pip installed on your system.

### Installation

1. **Install the `uv` package manager** (faster Python package management):
   ```bash
   python -m pip install uv
   ```

2. **Create and activate a virtual environment**:
   ```bash
   uv venv
   ```

   Then activate it:
   - **Linux/macOS**:
     ```bash
     source .venv/bin/activate
     ```
   - **Windows**:
     ```bash
     .venv\Scripts\activate
     ```

3. **Install dependencies**:
   ```bash
   python -m ensurepip
   python -m pip install -r requirements.txt
   ```

### Preparing the Dataset

To download and prepare the MNIST dataset:
```bash
python mnist.py
```

This will download the MNIST dataset and convert it to binary format files for use with the C implementation.

### Building and Running

**Build the MNIST trainer**:
```bash
make
```

**Run the trainer**:
```bash
make run
```

This will train the neural network on the MNIST dataset and evaluate it on the test set.

## Testing

The project includes comprehensive unit tests for all core modules.

### Running Tests

Run all tests:
```bash
make test
```

Run individual test suites:
```bash
make test_matrix    # Matrix operations tests
make test_arena     # Memory arena allocator tests
make test_prng      # Pseudo-random number generator tests
```

### Test Coverage

- **Matrix Operations** (`test_matrix`): Tests for creation, arithmetic, activation functions, and basic operations
- **Arena Allocator** (`test_arena`): Tests for memory allocation, deallocation, and temporary arena functionality
- **PRNG** (`test_prng`): Tests for random number generation and value distribution


## Development Notes

This is a fork of a machine learning framework originally created by [Magicalbat](https://github.com/Magicalbat). The original implementation provided solid foundational work on neural networks, matrix operations, and memory management in C. This fork reorganizes and improves the project with better testing infrastructure and development tooling.

### Improvements Made

- **Code reorganization**: Restructured from a flat file layout to a proper `include/` and `src/` directory hierarchy, moving each component (matrix, model, MNIST utilities) into its own module for better maintainability.

- **Added comprehensive tests**: Built test suites for matrix operations, arena memory allocator, and PRNG to catch regressions and verify correctness during development.

- **Development infrastructure**: Added GitHub Actions for CI/CD, requirements.txt for Python dependencies, detailed README for environment setup, and file checks (linting, formatting) to maintain code quality.
