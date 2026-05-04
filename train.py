"""
Heart Disease Prediction — SVM
Training script: load → preprocess → train → evaluate → save
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)

from utils.preprocess import load_and_preprocess
from utils.evaluate   import plot_confusion_matrix, plot_roc_curve

# ─── Config ───────────────────────────────────────────────────────────────────
DATA_URL   = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
MODEL_PATH = "models/svm_heart_model.pkl"
SCALER_PATH= "models/scaler.pkl"
RANDOM_STATE = 42

# ─── Load & Preprocess ────────────────────────────────────────────────────────
print("Loading dataset...")
X, y, scaler = load_and_preprocess(DATA_URL)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# ─── Train SVM ────────────────────────────────────────────────────────────────
print("\nTraining SVM (RBF kernel)...")
svm = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=RANDOM_STATE)
svm.fit(X_train, y_train)

# Cross-validation
cv_scores = cross_val_score(svm, X_train, y_train, cv=5, scoring="roc_auc")
print(f"5-Fold CV ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# ─── Evaluate ─────────────────────────────────────────────────────────────────
print("\nEvaluation on test set:")
y_pred = svm.predict(X_test)
y_prob = svm.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")

plot_confusion_matrix(y_test, y_pred, save_path="models/confusion_matrix.png")
plot_roc_curve(y_test, y_prob, save_path="models/roc_curve.png")

# ─── Save ─────────────────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
joblib.dump(svm,    MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
print(f"\nModel saved to {MODEL_PATH}")
print(f"Scaler saved to {SCALER_PATH}")
