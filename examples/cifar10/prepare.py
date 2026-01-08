#!/usr/bin/env python3
"""
Prepare CIFAR-10 dataset for training
Multi-class image classification: 10 object categories
Dataset: 60,000 32x32 RGB images (50,000 train, 10,000 test)

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
        from tensorflow.keras.datasets import cifar10
    except ImportError:
        print("Error: tensorflow required")
        print("Install with: pip install tensorflow")
        return False

    # Determine data directory (same as script location)
    data_dir = Path(__file__).parent / "data"

    print("\n=== Preparing CIFAR-10 Dataset ===\n")

    print("Downloading CIFAR-10 (60,000 32x32 RGB images)...")
    print("This may take a couple minutes on first run...\n")
    print(f"Data will be saved to: {data_dir}\n")

    try:
        # Download CIFAR-10
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()

        print(f"Downloaded: {X_train.shape[0]} train, {X_test.shape[0]} test images")
        print(f"Image shape: {X_train.shape[1:]} (32x32x3 RGB)")

        classes = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        print(f"Classes: {len(classes)}")
        for i, cls in enumerate(classes):
            print(f"  {i}: {cls}")

        # Flatten 32x32x3 images to 3072 features and normalize to 0-1
        print(f"\nFlattening and normalizing images...")
        X_train = X_train.reshape(X_train.shape[0], -1).astype(np.float32) / 255.0
        X_test = X_test.reshape(X_test.shape[0], -1).astype(np.float32) / 255.0
        y_train = y_train.flatten().astype(np.int32)
        y_test = y_test.flatten().astype(np.int32)

        print(f"  Train features shape: {X_train.shape}")
        print(f"  Test features shape: {X_test.shape}")

        # Save
        print(f"\nSaving to .mat binary format:\n")
        files_saved = all([
            save_mat(data_dir / 'cifar10_train_images.mat', X_train),
            save_mat(data_dir / 'cifar10_train_labels.mat', y_train.reshape(-1, 1)),
            save_mat(data_dir / 'cifar10_test_images.mat', X_test),
            save_mat(data_dir / 'cifar10_test_labels.mat', y_test.reshape(-1, 1)),
        ])

        if not files_saved:
            print("\nError: Failed to save one or more files", file=sys.stderr)
            return False

        print(f"\n✓ CIFAR-10 dataset ready!")
        print(f"Data location: {data_dir}")
        print(f"\nNote: CIFAR-10 is challenging for small networks.")
        print(f"Expected accuracy: ~60-75% (deep CNNs achieve 95%+)")
        print(f"\nTrain model from project root with:")
        print(f"  ./build/ml_trainer examples/cifar10/config.txt")

        return True

    except Exception as e:
        print(f"Error downloading dataset: {e}", file=sys.stderr)
        return False


if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)
