from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load("models/svm_heart_model.pkl")
scaler = joblib.load("models/scaler.pkl")

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

        return render_template("index.html",
                               prediction=result,
                               probability=round(prob * 100, 2))

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)