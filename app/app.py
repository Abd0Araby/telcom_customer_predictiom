from flask import Flask, jsonify, render_template, request
import pandas as pd
import joblib
from pathlib import Path

app = Flask(__name__)


def load_model():
    model_path = Path(__file__).resolve().parent.parent / 'model/log_reg_pipeline.joblib'
    return joblib.load(model_path)

def extract_data(data):
    df = pd.DataFrame({
        "gender": [data.get('gender')],
        "SeniorCitizen": [data.get('SeniorCitizen')],
        "Partner": [data.get('Partner')],
        "Dependents": [data.get('Dependents')],
        "tenure": [data.get('tenure')],
        "PhoneService": [data.get('PhoneService')],
        "MultipleLines": [data.get('MultipleLines')],
        "InternetService": [data.get('InternetService')],
        "OnlineSecurity": [data.get('OnlineSecurity')],
        "OnlineBackup": [data.get('OnlineBackup')],
        "DeviceProtection": [data.get('DeviceProtection')],
        "TechSupport": [data.get('TechSupport')],
        "StreamingTV": [data.get('StreamingTV')],
        "StreamingMovies": [data.get('StreamingMovies')],
        "Contract": [data.get('Contract')],
        "PaperlessBilling": [data.get('PaperlessBilling')],
        "PaymentMethod": [data.get('PaymentMethod')],
        "MonthlyCharges": [data.get('MonthlyCharges')],
        "TotalCharges": [data.get('TotalCharges')]
    })
    return df

@app.post('/predict')
def predict():
    data = request.get_json()
    df = extract_data(data)
    model = load_model()
    pred = model.predict(df)
    return jsonify({'Result': f'the churn status is: {pred}'})



@app.get('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 5000)