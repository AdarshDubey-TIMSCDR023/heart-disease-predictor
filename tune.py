"""
tune.py — GridSearchCV hyperparameter tuning for SVM
Run after train.py to find optimal C, kernel, gamma
"""

import joblib
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split

from utils.preprocess import load_and_preprocess

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

print("Loading data...")
X, y, scaler = load_and_preprocess(DATA_URL)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

param_grid = {
    "C":      [0.01, 0.1, 1, 10, 100],
    "kernel": ["rbf", "linear"],
    "gamma":  ["scale", "auto", 0.001, 0.01, 0.1],
}

print("\nRunning GridSearchCV (5-fold, scoring=roc_auc)...")
grid = GridSearchCV(
    SVC(probability=True, random_state=42),
    param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=1,
)
grid.fit(X_train, y_train)

print(f"\nBest parameters : {grid.best_params_}")
print(f"Best CV ROC-AUC : {grid.best_score_:.4f}")

# Re-evaluate best model on hold-out test set
from sklearn.metrics import roc_auc_score, classification_report
best = grid.best_estimator_
y_pred = best.predict(X_test)
y_prob = best.predict_proba(X_test)[:, 1]
print(f"\nTest ROC-AUC    : {roc_auc_score(y_test, y_prob):.4f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))

# Save best model
joblib.dump(best,   "models/svm_best_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("\nBest model saved → models/svm_best_model.pkl")
