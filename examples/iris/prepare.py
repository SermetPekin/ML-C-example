#!/usr/bin/env python3
"""
Prepare Iris Flower dataset for training
Uses scikit-learn's built-in Iris dataset

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
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
    except ImportError:
        print("Error: scikit-learn required")
        print("Install with: pip install scikit-learn")
        return False

    # Determine data directory (same as script location)
    data_dir = Path(__file__).parent / "data"

    print("\n=== Preparing Iris Flower Dataset ===\n")
    print(f"Data will be saved to: {data_dir}\n")

    # Load dataset
    try:
        iris = load_iris()
        X = iris.data.astype(np.float32)
        y = iris.target.astype(np.int32)
    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        return False

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {len(np.unique(y))}")
    print(f"  - Class 0: {iris.target_names[0]}")
    print(f"  - Class 1: {iris.target_names[1]}")
    print(f"  - Class 2: {iris.target_names[2]}")

    # Normalize features to 0-1 range for better training
    print(f"\nNormalizing features to [0, 1] range...")
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X = (X - X_min) / (X_max - X_min)

    # Split into train/test (80/20)
    print(f"\nSplitting into train/test (80/20):")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")

    # Save
    print(f"\nSaving to .mat binary format:\n")
    files_saved = all([
        save_mat(data_dir / 'iris_train_features.mat', X_train),
        save_mat(data_dir / 'iris_train_labels.mat', y_train.reshape(-1, 1)),
        save_mat(data_dir / 'iris_test_features.mat', X_test),
        save_mat(data_dir / 'iris_test_labels.mat', y_test.reshape(-1, 1)),
    ])

    if not files_saved:
        print("\nError: Failed to save one or more files", file=sys.stderr)
        return False

    print(f"\n✓ Iris dataset ready!")
    print(f"Data location: {data_dir}")
    print(f"\nTrain model from project root with:")
    print(f"  ./build/ml_trainer examples/iris/config.txt")

    return True


if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)
