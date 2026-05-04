from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# ─── Correct path handling (VERY IMPORTANT for Render) ─────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "svm_heart_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

# ─── Load model safely ─────────────────────────────────────────────
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model loaded successfully")
except Exception as e:
    print("ERROR LOADING MODEL:", e)
    model = None
    scaler = None


EXPECTED_COLS = [
    "age", "sex", "trestbps", "chol", "fbs", "thalach",
    "exang", "oldpeak", "ca",
    "cp_2.0", "cp_3.0", "cp_4.0",
    "restecg_1.0", "restecg_2.0",
    "slope_2.0", "slope_3.0",
    "thal_6.0", "thal_7.0"
]


def preprocess(data):
    df = pd.DataFrame([data])

    # Convert categorical to float
    for col in ["cp", "restecg", "slope", "thal"]:
        df[col] = df[col].astype(float)

    # One-hot encoding
    df = pd.get_dummies(df, drop_first=True)

    # Match training columns
    df = df.reindex(columns=EXPECTED_COLS, fill_value=0)

    return scaler.transform(df)


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            # ❗ Check if model loaded
            if model is None or scaler is None:
                return "ERROR: Model or scaler not loaded properly"

            data = {
                "age": int(request.form["age"]),
                "sex": int(request.form["sex"]),
                "cp": int(request.form["cp"]),
                "trestbps": int(request.form["trestbps"]),
                "chol": int(request.form["chol"]),
                "fbs": int(request.form["fbs"]),
                "restecg": int(request.form["restecg"]),
                "thalach": int(request.form["thalach"]),
                "exang": int(request.form["exang"]),
                "oldpeak": float(request.form["oldpeak"]),
                "slope": int(request.form["slope"]),
                "ca": int(request.form["ca"]),
                "thal": float(request.form["thal"]),
            }

            X = preprocess(data)

            prob = model.predict_proba(X)[0][1]
            result = "Disease" if prob >= 0.5 else "No Disease"

            return render_template(
                "index.html",
                prediction=result,
                probability=round(prob * 100, 2)
            )

        except Exception as e:
            # 🔥 SHOW REAL ERROR (for debugging)
            return f"ERROR OCCURRED: {str(e)}"

    return render_template("index.html")


# ─── Render deployment config ──────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
