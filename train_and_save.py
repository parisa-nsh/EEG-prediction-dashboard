"""
Train EEG classifier and save the model.
Uses 70/15/15 train/validation/test split. Replace synthetic data with real EEG when ready.
"""
import json
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.model_selection import train_test_split
from models.eeg_classifier import EEGClassifier

SAVED_MODELS_DIR = os.path.join(os.path.dirname(__file__), 'saved_models')
DATA_SEED = 42
SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train, val, test


def _generate_and_split_data(model_type, n_samples, seed=DATA_SEED):
    """Generate synthetic data and split into train/val/test."""
    np.random.seed(seed)
    if model_type == 'cnn':
        X = np.random.randn(n_samples, 64, 64, 1).astype(np.float32) * 0.5 + 0.5
    else:
        X = np.random.randn(n_samples, 4096).astype(np.float32)
    y = np.random.randint(0, 4, n_samples)

    # 70% train, 30% rest
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=1 - SPLIT_RATIOS[0], stratify=y, random_state=seed
    )
    # Split rest into 50/50 -> 15% val, 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=seed
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def train_and_save(model_type='random_forest', n_samples=600):
    """
    Train the specified model with train/val/test split and save it.

    Args:
        model_type: One of 'cnn', 'deep_nn', 'shallow_nn', 'random_forest', 'svm'
        n_samples: Total number of samples (split 70/15/15)
    """
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test = _generate_and_split_data(
        model_type, n_samples
    )

    n_train, n_val, n_test = len(X_train), len(X_val), len(X_test)
    print(f"Data split: train={n_train}, val={n_val}, test={n_test}")

    clf = EEGClassifier()
    if model_type not in clf.get_available_models():
        raise ValueError(
            f"Unknown model type: {model_type}. Choose from {clf.get_available_models()}"
        )
    clf.set_current_model(model_type)

    # Train (with validation for neural networks)
    print(f"Training {model_type}...")
    history = clf.train(X_train, y_train, X_val=X_val, y_val=y_val)

    if history is not None:
        final_acc = history.history['accuracy'][-1]
        val_acc = history.history.get('val_accuracy', [0])[-1]
        print(f"  Train accuracy: {final_acc:.4f}  Val accuracy: {val_acc:.4f}")

    # Save model
    if model_type in ['cnn', 'deep_nn', 'shallow_nn']:
        ext = '.keras'
    else:
        ext = '.pkl'

    filepath = os.path.join(SAVED_MODELS_DIR, f'eeg_{model_type}{ext}')
    clf.save_model(filepath)

    if model_type in ['random_forest', 'svm']:
        print(f"Saved: {filepath}")
        print(f"Saved: {filepath.replace('.pkl', '_scaler.pkl')}")
    else:
        print(f"Saved: {filepath}")

    # Save test set for evaluation (flatten for RF/shallow_nn; 64x64x1 -> 4096 for CNN)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    if X_test_flat.shape[1] > 4096:
        X_test_flat = X_test_flat[:, :4096]
    np.savez(
        os.path.join(SAVED_MODELS_DIR, 'test_data.npz'),
        X_test=X_test_flat,
        y_test=y_test,
    )

    # Save split info for dashboard
    split_info = {
        'train': n_train,
        'validation': n_val,
        'test': n_test,
        'total': n_train + n_val + n_test,
        'ratios': {'train': 0.70, 'validation': 0.15, 'test': 0.15},
    }
    with open(os.path.join(SAVED_MODELS_DIR, 'split_info.json'), 'w') as f:
        json.dump(split_info, f, indent=2)

    return filepath


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        default='random_forest',
        choices=['cnn', 'deep_nn', 'shallow_nn', 'random_forest', 'svm'],
        help='Model type to train',
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=600,
        help='Total samples (split 70/15/15), default 600',
    )
    args = parser.parse_args()

    train_and_save(model_type=args.model, n_samples=args.samples)
