from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# ─── Safe model loading (prevents crash) ───────────────────────────
try:
    model = joblib.load("models/svm_heart_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
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

    # Fix datatype
    for col in ["cp", "restecg", "slope", "thal"]:
        df[col] = df[col].astype(float)

    # Encoding
    df = pd.get_dummies(df, drop_first=True)

    # Match training columns
    df = df.reindex(columns=EXPECTED_COLS, fill_value=0)

    return scaler.transform(df)


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
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
            return render_template(
                "index.html",
                prediction="Error",
                probability=0,
                error=str(e)
            )

    return render_template("index.html")


# ─── IMPORTANT FOR RENDER ──────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
