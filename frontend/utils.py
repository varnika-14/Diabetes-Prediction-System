import pandas as pd
import numpy as np
import joblib
import json
import os

def load_model():
    model_path = os.path.join('..', 'backend', 'models', 'diabetes_model.pkl')
    scaler_path = os.path.join('..', 'backend', 'models', 'scaler.pkl')
    imputer_path = os.path.join('..', 'backend', 'models', 'imputer.pkl')
    info_path = os.path.join('..', 'backend', 'models', 'model_info.json')
    
    if not os.path.exists(model_path):
        return None, None, None, None
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    imputer = joblib.load(imputer_path)
    
    with open(info_path, 'r') as f:
        model_info = json.load(f)
    
    return model, scaler, imputer, model_info

def predict_risk(model, scaler, imputer, features):
    feature_names = [
        'BMI', 'Age', 'HighBP', 'HeartDiseaseorAttack', 'Sex',
        'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Education',
        'Income', 'Smoker', 'Stroke', 'PhysActivity', 'Fruits',
        'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost',
        'HbA1c_level', 'blood_glucose_level', 'smoking_history_encoded'
    ]
    
    input_array = np.array([[features[f] for f in feature_names]])
    
    input_imputed = imputer.transform(input_array)
    input_scaled = scaler.transform(input_imputed)
    
    probability = model.predict_proba(input_scaled)[0][1]
    risk_score = round(probability * 100, 1)
    prediction = model.predict(input_scaled)[0]
    
    return risk_score, prediction

def get_classification(risk_score):
    if risk_score < 30:
        return "Low Risk", "green"
    elif risk_score < 60:
        return "Moderate Risk", "orange"
    else:
        return "High Risk", "red"

def get_recommendations(risk_score):
    if risk_score < 30:
        return [
            "Continue maintaining a healthy lifestyle",
            "Exercise regularly (at least 30 minutes daily)",
            "Eat a balanced diet rich in fruits and vegetables",
            "Schedule annual check-ups",
            "Monitor blood glucose levels periodically"
        ]
    elif risk_score < 60:
        return [
            "Consult a healthcare provider for preventive care",
            "Consider lifestyle modifications",
            "Monitor blood glucose levels more frequently",
            "Increase physical activity",
            "Review your diet with a nutritionist"
        ]
    else:
        return [
            "Consult a healthcare provider immediately",
            "Schedule comprehensive diabetes screening",
            "Monitor blood glucose levels daily",
            "Adopt a diabetes-prevention diet",
            "Consider medication consultation",
            "Regular exercise routine is crucial"
        ]