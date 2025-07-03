# predict.py

import pandas as pd
import pickle

with open("models/churn_model.pkl", "rb") as f:
    model = pickle.load(f)

input_data = {
    'gender': 1,
    'SeniorCitizen': 0,
    'Partner': 1,
    'Dependents': 0,
    'tenure': 12,
    'PhoneService': 1,
    'MultipleLines': 0,
    'InternetService': 1,
    'OnlineSecurity': 0,
    'OnlineBackup': 1,
    'DeviceProtection': 0,
    'TechSupport': 0,
    'StreamingTV': 1,
    'StreamingMovies': 0,
    'Contract': 1,
    'PaperlessBilling': 1,
    'PaymentMethod': 2,
    'MonthlyCharges': 70.35,
    'TotalCharges': 830.5
}

df = pd.DataFrame([input_data])
prediction = model.predict(df)

print("🚨 Churn Prediction:", "Yes" if prediction[0] == 1 else "No")
