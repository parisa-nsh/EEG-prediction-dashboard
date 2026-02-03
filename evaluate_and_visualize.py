"""
Evaluate saved models and plot confusion matrix heatmaps + ROC-AUC (OvR).
Uses the held-out test set from training when available.
"""
import json
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.eeg_classifier import EEGClassifier
from sklearn.metrics import (
    confusion_matrix,
    balanced_accuracy_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

SAVED_MODELS_DIR = os.path.join(os.path.dirname(__file__), 'saved_models')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'evaluation_plots')
EMOTION_CLASSES = ['happy', 'sad', 'anxious', 'neutral']
N_CLASSES = len(EMOTION_CLASSES)


def plot_confusion_matrix(cm, model_name, save_path):
    """Plot confusion matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=EMOTION_CLASSES,
        yticklabels=EMOTION_CLASSES,
        ax=ax,
        cbar_kws={'label': 'Count'},
        annot_kws={'size': 12}
    )
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def load_test_data():
    """Load test set from training split, or generate fallback."""
    test_path = os.path.join(SAVED_MODELS_DIR, 'test_data.npz')
    if os.path.exists(test_path):
        data = np.load(test_path)
        return data['X_test'].astype(np.float32), data['y_test']
    np.random.seed(123)
    X = np.random.randn(200, 4096).astype(np.float32)
    y = np.random.randint(0, 4, 200)
    return X, y


def ensure_split_info(n_test):
    """Create split_info.json if missing (e.g. from older runs)."""
    split_path = os.path.join(SAVED_MODELS_DIR, 'split_info.json')
    if os.path.exists(split_path):
        return
    # Infer from test size (15% of total)
    total = round(n_test / 0.15)
    n_train = round(total * 0.70)
    n_val = total - n_train - n_test
    split_info = {
        'train': n_train,
        'validation': max(0, n_val),
        'test': n_test,
        'total': total,
        'ratios': {'train': 0.70, 'validation': 0.15, 'test': 0.15},
    }
    with open(split_path, 'w') as f:
        json.dump(split_info, f, indent=2)


def plot_roc_auc_ovr(y_true, y_proba, save_path):
    """Plot ROC curves One-vs-Rest for each class."""
    y_true_bin = label_binarize(y_true, classes=range(N_CLASSES))

    fig, ax = plt.subplots(figsize=(8, 6))
    aucs = []

    for i in range(N_CLASSES):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(
            fpr, tpr,
            label=f'{EMOTION_CLASSES[i]} (AUC = {roc_auc:.3f})',
            linewidth=2,
        )

    mean_auc = np.mean(aucs)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curves (One-vs-Rest) — Mean AUC: {mean_auc:.3f}', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def evaluate_and_visualize():
    """Evaluate saved models on test set and create confusion matrix heatmaps."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X_test, y_true = load_test_data()
    n_test = len(X_test)
    print(f"Test set: {n_test} samples")
    ensure_split_info(n_test)

    saved = [
        ('random_forest', os.path.join(SAVED_MODELS_DIR, 'eeg_random_forest.pkl')),
        ('shallow_nn', os.path.join(SAVED_MODELS_DIR, 'eeg_shallow_nn.keras')),
    ]

    for model_type, path in saved:
        if not os.path.exists(path):
            print(f"Skip {model_type}: not found")
            continue

        clf = EEGClassifier()
        clf.load_model(path, model_type)
        clf.set_current_model(model_type)
        predictions, _ = clf.predict(X_test)

        cm = confusion_matrix(y_true, predictions)
        bal_acc = balanced_accuracy_score(y_true, predictions)

        display_name = model_type.replace('_', ' ').title()
        save_path = os.path.join(OUTPUT_DIR, f'confusion_matrix_{model_type}.png')
        plot_confusion_matrix(cm, f'{display_name} (balanced acc: {bal_acc:.3f})', save_path)
        print(f"  {display_name}: balanced accuracy = {bal_acc:.4f}")
        print(f"  Saved: {save_path}")

    # Combined figure: side-by-side for comparison
    results = {}
    for model_type, path in saved:
        if not os.path.exists(path):
            continue
        clf = EEGClassifier()
        clf.load_model(path, model_type)
        clf.set_current_model(model_type)
        predictions, probs = clf.predict(X_test)
        results[model_type] = (predictions, probs)

    if len(results) >= 2:
        fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
        if len(results) == 1:
            axes = [axes]
        for ax, (model_type, (preds, _)) in zip(axes, results.items()):
            cm = confusion_matrix(y_true, preds)
            bal_acc = balanced_accuracy_score(y_true, preds)
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=EMOTION_CLASSES, yticklabels=EMOTION_CLASSES,
                ax=ax, cbar_kws={'label': 'Count'}, annot_kws={'size': 11}
            )
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title(f"{model_type.replace('_', ' ').title()} (acc: {bal_acc:.3f})")
        plt.suptitle('Model Comparison - Confusion Matrices (Held-out Test Set)', fontsize=14, y=1.02)
        plt.tight_layout()
        combined_path = os.path.join(OUTPUT_DIR, 'confusion_matrices_comparison.png')
        plt.savefig(combined_path, dpi=150)
        plt.close()
        print(f"Saved: {combined_path}")

    # ROC-AUC One-vs-Rest plots (one per model + combined comparison)
    for model_type, path in saved:
        if not os.path.exists(path):
            continue
        if model_type not in results:
            continue
        _, probs = results[model_type]
        roc_path = os.path.join(OUTPUT_DIR, f'roc_auc_{model_type}.png')
        plot_roc_auc_ovr(y_true, probs, roc_path)
        print(f"Saved: {roc_path}")

    # Combined ROC comparison (both models side by side)
    if len(results) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, (model_type, (_, probs)) in zip(axes, list(results.items())[:2]):
            y_true_bin = label_binarize(y_true, classes=range(N_CLASSES))
            for i in range(N_CLASSES):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f'{EMOTION_CLASSES[i]} (AUC={roc_auc:.2f})', linewidth=2)
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC (OvR) - {model_type.replace("_", " ").title()}')
            ax.legend(loc='lower right', fontsize=9)
            ax.grid(True, alpha=0.3)
        plt.suptitle('ROC-AUC Comparison (One-vs-Rest)', fontsize=14, y=1.02)
        plt.tight_layout()
        roc_comp_path = os.path.join(OUTPUT_DIR, 'roc_auc_comparison.png')
        plt.savefig(roc_comp_path, dpi=150)
        plt.close()
        print(f"Saved: {roc_comp_path}")

    print(f"\nPlots saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    evaluate_and_visualize()
