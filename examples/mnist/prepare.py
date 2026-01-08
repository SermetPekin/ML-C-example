#!/usr/bin/env python3
"""
Prepare MNIST dataset for training
Downloads MNIST handwritten digits and converts to .mat binary format

Usage:
    python prepare.py
"""

import numpy as np
import struct
from pathlib import Path
import sys


def save_mat(filepath, data):
    """Save numpy array as .mat binary file (row-major, float32)"""
    try:
        data = np.asarray(data, dtype=np.float32)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            flat = data.flatten()
            f.write(struct.pack(f'{len(flat)}f', *flat))
        print(f"  Saved: {filepath.name} ({data.shape})")
        return True
    except Exception as e:
        print(f"Error saving {filepath.name}: {e}", file=sys.stderr)
        return False


def main():
    try:
        from sklearn.datasets import fetch_openml
        from sklearn.model_selection import train_test_split
    except ImportError:
        print("Error: scikit-learn required")
        print("Install with: pip install scikit-learn")
        return False

    # Determine data directory (same as script location)
    data_dir = Path(__file__).parent / "data"

    print("\n=== Preparing MNIST Dataset ===\n")
    print("Downloading MNIST (70,000 handwritten digit images)...")
    print("This may take a few minutes on first run...\n")
    print(f"Data will be saved to: {data_dir}\n")

    try:
        # Download MNIST
        mnist = fetch_openml('mnist_784', version=1, parser='liac-arff', as_frame=False)
        X, y = mnist.data, mnist.target

        print(f"Downloaded: {X.shape[0]} images, {X.shape[1]} features each")

        # Normalize to 0-1 range
        X = X.astype(np.float32) / 255.0
        y = y.astype(np.int32)

        print(f"Classes: {np.unique(y)}")

        # Split: use 60k for training, 10k for testing
        X_train = X[:60000]
        y_train = y[:60000]
        X_test = X[60000:]
        y_test = y[60000:]

        print(f"\nSplitting into train/test:")
        print(f"  Train: {X_train.shape[0]} samples")
        print(f"  Test: {X_test.shape[0]} samples")

        # Save
        print(f"\nSaving to .mat binary format:\n")
        files_saved = all([
            save_mat(data_dir / 'mnist_train_images.mat', X_train),
            save_mat(data_dir / 'mnist_train_labels.mat', y_train.reshape(-1, 1)),
            save_mat(data_dir / 'mnist_test_images.mat', X_test),
            save_mat(data_dir / 'mnist_test_labels.mat', y_test.reshape(-1, 1)),
        ])

        if not files_saved:
            print("\nError: Failed to save one or more files", file=sys.stderr)
            return False

        print(f"\n✓ MNIST dataset ready!")
        print(f"Data location: {data_dir}")
        print(f"\nTrain model from project root with:")
        print(f"  ./build/ml_trainer examples/mnist/config.txt")

        return True

    except Exception as e:
        print(f"Error downloading dataset: {e}", file=sys.stderr)
        return False


if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)
