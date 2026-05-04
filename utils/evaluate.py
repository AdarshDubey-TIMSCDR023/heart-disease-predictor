"""
utils/evaluate.py — Plotting helpers for model evaluation
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay, roc_curve, roc_auc_score
)


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Plot and optionally save the confusion matrix."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=["No Disease", "Disease"],
        cmap="Blues", ax=ax
    )
    ax.set_title("Confusion Matrix — SVM (RBF)", fontsize=13, pad=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved confusion matrix → {save_path}")
    plt.close()


def plot_roc_curve(y_true, y_prob, save_path=None):
    """Plot and optionally save the ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="#185FA5", lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="#888780", lw=1)
    ax.fill_between(fpr, tpr, alpha=0.08, color="#185FA5")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — SVM", fontsize=13, pad=12)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved ROC curve → {save_path}")
    plt.close()


def compare_kernels(X_train, y_train, X_test, y_test):
    """
    Train SVM with multiple kernels and return a comparison dict.

    Returns:
        dict: {kernel_name: {'accuracy': float, 'roc_auc': float}}
    """
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score

    kernels = {
        "RBF":        SVC(kernel="rbf",    C=1, gamma="scale", probability=True),
        "Linear":     SVC(kernel="linear", C=1,                probability=True),
        "Polynomial": SVC(kernel="poly",   C=1, degree=3,      probability=True),
    }
    results = {}
    for name, model in kernels.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "roc_auc":  round(roc_auc_score(y_test, y_prob), 4),
        }
        print(f"{name:12s}  Accuracy={results[name]['accuracy']:.4f}  "
              f"ROC-AUC={results[name]['roc_auc']:.4f}")
    return results
