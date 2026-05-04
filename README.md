# Heart Disease Prediction — Support Vector Machine (SVM + Flask)

A machine learning and web application that predicts the likelihood of heart disease using a Support Vector Machine (SVM) model. The system is deployed as a Flask web app for real-time prediction.

---

## Live Demo

https://heart-disease-predictor-bj9h.onrender.com/

---

## Overview

This project uses the Cleveland UCI Heart Disease dataset to train an SVM model for binary classification (Disease / No Disease). It integrates machine learning with a Flask-based web interface, allowing users to input patient data and receive instant predictions.

---

## Features

- SVM-based heart disease prediction  
- Probability-based risk estimation  
- Flask web interface for real-time prediction  
- Data preprocessing (encoding + scaling)  
- Modular pipeline (train, tune, predict)  
- Deployed on cloud platform  

---

## Project Structure

```bash
svm_heart_disease/
├── app.py                  
├── train.py                
├── tune.py                 
├── predict.py              
├── requirements.txt        
│
├── models/
│   ├── svm_heart_model.pkl
│   └── scaler.pkl
│
├── templates/
│   └── index.html          
│
├── utils/
│   ├── preprocess.py
│   └── evaluate.py
│
├── notebooks/
│   └── heart_disease_svm.ipynb
│
└── data/

## Installation

git clone https://github.com/AdarshDubey-TIMSCDR023/heart-disease-predictor.git  
cd heart-disease-predictor  
python -m pip install -r requirements.txt  

---

## Usage

Train Model  
python train.py  

Hyperparameter Tuning  
python tune.py  

CLI Prediction  
python predict.py  

Run Web App  
python app.py  

Open in browser:  
http://127.0.0.1:5000/

---

## Sample Input

{
    "age": 60,
    "sex": 1,
    "cp": 3,
    "trestbps": 160,
    "chol": 300,
    "fbs": 1,
    "restecg": 2,
    "thalach": 120,
    "exang": 1,
    "oldpeak": 3.5,
    "slope": 0,
    "ca": 3,
    "thal": 7
}

---

## Deployment

Backend hosted on Render  
Version controlled with GitHub  

---

## Author

Adarsh Dubey  

---

## License

For educational use only  
