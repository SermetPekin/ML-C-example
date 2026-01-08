#!/usr/bin/env python3
"""
Prepare Breast Cancer Wisconsin (Diagnostic) Dataset for training
Binary classification: Benign (0) or Malignant (1)
Dataset: 569 samples, 30 diagnostic features

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
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("Error: scikit-learn required")
        print("Install with: pip install scikit-learn")
        return False

    # Determine data directory (same as script location)
    data_dir = Path(__file__).parent / "data"

    print("\n=== Preparing Breast Cancer Wisconsin Dataset ===\n")
    print(f"Data will be saved to: {data_dir}\n")

    # Load dataset
    try:
        cancer = load_breast_cancer()
        X = cancer.data.astype(np.float32)
        y = cancer.target.astype(np.int32)
    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        return False

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {len(np.unique(y))}")
    print(f"  - Class 0: {cancer.target_names[0]} (Malignant)")
    print(f"  - Class 1: {cancer.target_names[1]} (Benign)")

    class_dist = np.bincount(y)
    print(f"  - Class distribution: {class_dist[0]} malignant, {class_dist[1]} benign")

    # Standardize features (important for this dataset - large feature value ranges)
    print(f"\nStandardizing features (mean=0, std=1)...")
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

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
        save_mat(data_dir / 'breast_cancer_train_features.mat', X_train),
        save_mat(data_dir / 'breast_cancer_train_labels.mat', y_train.reshape(-1, 1)),
        save_mat(data_dir / 'breast_cancer_test_features.mat', X_test),
        save_mat(data_dir / 'breast_cancer_test_labels.mat', y_test.reshape(-1, 1)),
    ])

    if not files_saved:
        print("\nError: Failed to save one or more files", file=sys.stderr)
        return False

    print(f"\n✓ Breast Cancer dataset ready!")
    print(f"Data location: {data_dir}")
    print(f"\nTrain model from project root with:")
    print(f"  ./build/ml_trainer examples/breast_cancer/config.txt")

    return True


if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)
