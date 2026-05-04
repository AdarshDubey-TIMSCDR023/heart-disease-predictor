"""
utils/preprocess.py — Data loading and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "target"
]

CATEGORICAL_COLS = ["cp", "restecg", "slope", "thal"]


def load_and_preprocess(source):
    """
    Load the Cleveland UCI heart disease dataset, clean, encode, and scale.

    Args:
        source: file path or URL to the raw .data file

    Returns:
        X_scaled (np.ndarray), y (pd.Series), fitted_scaler (StandardScaler)
    """
    df = pd.read_csv(source, names=COLUMNS, na_values="?")

    # Binarise target: 0 = no disease, 1 = disease
    df["target"] = (df["target"] > 0).astype(int)

    # Fill missing values with column median (pandas 2.x compatible)
    for col in ["ca", "thal"]:
        df[col] = df[col].fillna(df[col].median())

    # One-hot encode categoricals
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)

    X = df.drop("target", axis=1)
    y = df["target"]

    # Scale features (critical for SVM!)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Dataset shape: {X_scaled.shape}  |  "
          f"Class balance: {y.value_counts().to_dict()}")

    return X_scaled, y, scaler


def preprocess_patient(patient_dict, scaler, feature_columns):
    """
    Preprocess a single new patient record for inference.

    Args:
        patient_dict:    dict of raw feature values
        scaler:          fitted StandardScaler from training
        feature_columns: list of column names (from training X)

    Returns:
        X_scaled (np.ndarray, shape [1, n_features])
    """
    df_p = pd.DataFrame([patient_dict])
    df_p = pd.get_dummies(df_p)
    df_p = df_p.reindex(columns=feature_columns, fill_value=0)
    return scaler.transform(df_p)
