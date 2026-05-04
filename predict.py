import joblib
import pandas as pd

# Load model and scaler
model = joblib.load("models/svm_heart_model.pkl")
scaler = joblib.load("models/scaler.pkl")

def predict_patient(data):

    df = pd.DataFrame([data])

    # 🔥 IMPORTANT: match training format
    df["cp"] = df["cp"].astype(float)
    df["restecg"] = df["restecg"].astype(float)
    df["slope"] = df["slope"].astype(float)
    df["thal"] = df["thal"].astype(float)

    # Apply same encoding
    df = pd.get_dummies(df, drop_first=True)

    # ✅ Match training columns (manual fix)
    expected_cols = [
        "age", "sex", "trestbps", "chol", "fbs", "thalach",
        "exang", "oldpeak", "ca",
        "cp_2.0", "cp_3.0", "cp_4.0",
        "restecg_1.0", "restecg_2.0",
        "slope_2.0", "slope_3.0",
        "thal_6.0", "thal_7.0"
    ]

    df = df.reindex(columns=expected_cols, fill_value=0)

    # Scale
    X = scaler.transform(df)

    # Predict
    prob = model.predict_proba(X)[0][1]
    pred = "Disease" if prob >= 0.5 else "No Disease"

    return pred, prob


# 🔹 Test input
patient = {
    "age": 54, "sex": 1, "cp": 0,
    "trestbps": 122, "chol": 286, "fbs": 0,
    "restecg": 0, "thalach": 116,
    "exang": 1, "oldpeak": 3.2,
    "slope": 1, "ca": 2, "thal": 2.0
}

prediction, probability = predict_patient(patient)

print("Prediction:", prediction)
print("Probability:", round(probability * 100, 2), "%")