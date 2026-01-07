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

Each test suite reports pass/fail counts and detailed error information for any failures.
