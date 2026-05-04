# Heart Disease Prediction — Support Vector Machine

Binary classification using SVM on the Cleveland UCI Heart Disease dataset.

---

## Project Structure

```
svm_heart_disease/
├── train.py              # Main training pipeline
├── tune.py               # GridSearchCV hyperparameter tuning
├── predict.py            # Inference on new patients
├── requirements.txt
├── utils/
│   ├── preprocess.py     # Data loading, cleaning, encoding, scaling
│   └── evaluate.py       # Confusion matrix, ROC curve, kernel comparison
├── notebooks/
│   └── heart_disease_svm.ipynb   # Full EDA + training walkthrough
├── data/                 # (auto-downloaded from UCI)
└── models/               # Saved model + scaler (generated after training)
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model
python train.py

# 3. Tune hyperparameters (optional, slower)
python tune.py

# 4. Predict on a new patient
python predict.py
```

---

## Dataset

**Source**: [UCI Heart Disease (Cleveland)](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)  
**Kaggle mirror**: https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci

| Feature     | Description                                  |
|-------------|----------------------------------------------|
| age         | Age in years                                 |
| sex         | 1 = male, 0 = female                         |
| cp          | Chest pain type (0–3)                        |
| trestbps    | Resting blood pressure (mm Hg)               |
| chol        | Serum cholesterol (mg/dl)                    |
| fbs         | Fasting blood sugar > 120 mg/dl (1 = true)  |
| restecg     | Resting ECG results (0–2)                   |
| thalach     | Maximum heart rate achieved                  |
| exang       | Exercise-induced angina (1 = yes)            |
| oldpeak     | ST depression induced by exercise            |
| slope       | Slope of peak exercise ST segment           |
| ca          | Number of major vessels colored by fluoroscopy |
| thal        | Thalassemia (3 = normal, 6 = fixed, 7 = reversible) |
| **target**  | **0 = no disease, 1 = disease**              |

---

## Model Results

| Kernel  | Accuracy | ROC-AUC |
|---------|----------|---------|
| RBF     | ~85%     | ~0.917  |
| Linear  | ~84%     | ~0.903  |
| Poly    | ~82%     | ~0.875  |

---

## Why SVM for Medical Classification?

- Works well in high-dimensional feature spaces
- Effective with small-to-medium datasets (303 samples here)
- The kernel trick captures non-linear decision boundaries
- Robust to outliers via the soft-margin formulation

**Important**: Always scale features with `StandardScaler` before training SVM.  
SVM computes distances — unscaled features with large ranges (e.g. cholesterol 100–600) will dominate binary features (0/1).

---

## Key Hyperparameters

| Parameter | Effect |
|-----------|--------|
| `C`       | Regularization. High C = lower bias, higher variance |
| `gamma`   | (RBF/Poly) Kernel spread. High = tight boundary, risk of overfitting |
| `kernel`  | `rbf` recommended for tabular medical data |

---

## License

For educational use. Dataset credit: UCI Machine Learning Repository.
