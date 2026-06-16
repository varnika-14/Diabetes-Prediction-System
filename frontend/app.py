import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import sys
from datetime import datetime

st.set_page_config(
    page_title="Diabetes Risk Screening Tool",
    page_icon="hospital",
    layout="wide"
)

st.markdown("""
<style>
    .main .block-container {
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
    }
    .stApp h1 {
        font-size: 40px !important;
        font-weight: 700 !important;
        color: #2C7BE5 !important;
        margin-top: -20px !important;
    }
    .stApp, .stApp p {
        font-size: 18px !important;
    }
    .stApp h2 {
        font-size: 32px !important;
        font-weight: 600 !important;
    }
    .stApp h3 {
        font-size: 26px !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_path = os.path.join('..', 'backend', 'models', 'diabetes_model.pkl')
    scaler_path = os.path.join('..', 'backend', 'models', 'scaler.pkl')
    imputer_path = os.path.join('..', 'backend', 'models', 'imputer.pkl')
    info_path = os.path.join('..', 'backend', 'models', 'model_info.json')
    
    if not os.path.exists(model_path):
        st.error("Model not found. Please run model_training.py first.")
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
    
    input_values = [features[f] for f in feature_names]
    input_array = np.array([input_values])
    
    input_imputed = imputer.transform(input_array)
    input_scaled = scaler.transform(input_imputed)
    
    probability = model.predict_proba(input_scaled)[0][1]
    risk_score = round(probability * 100, 1)
    prediction = model.predict(input_scaled)[0]
    
    feature_importance = {}
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        for i, name in enumerate(feature_names):
            if i < len(importance):
                feature_importance[name] = round(importance[i] * 100, 1)
    else:
        base_prob = probability
        feature_importance = {}
        for i, name in enumerate(feature_names):
            modified_values = input_values.copy()
            modified_values[i] = 0
            modified_array = np.array([modified_values])
            modified_imputed = imputer.transform(modified_array)
            modified_scaled = scaler.transform(modified_imputed)
            modified_prob = model.predict_proba(modified_scaled)[0][1]
            importance = abs(base_prob - modified_prob) * 100
            feature_importance[name] = round(importance, 1)
    
    return risk_score, prediction, feature_importance
def get_classification(risk_score):
    if risk_score < 30:
        return "Low Risk"
    elif risk_score < 60:
        return "Moderate Risk"
    else:
        return "High Risk"

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

model, scaler, imputer, model_info = load_model()

if model is None:
    st.stop()

tab1, tab2 = st.tabs(["Risk Screening", "Chat Assistant"])

with tab1:
    st.title("Diabetes Risk Screening Tool")
    st.markdown("Enter your health information to get an accurate diabetes risk assessment")
    
    with st.expander("Model Information", expanded=False):
        st.write("Model: " + model_info['model_name'])
        st.write("Accuracy: " + str(round(model_info['accuracy'] * 100, 1)) + "%")
        st.write("AUC Score: " + str(round(model_info['auc'], 4)))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal Information")
        
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1, key="bmi")
        age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1, key="age")
        
        sex = st.selectbox("Sex", options=[("Female", 0), ("Male", 1)], format_func=lambda x: x[0], key="sex")[1]
        
        high_bp = st.selectbox("High Blood Pressure", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0], key="high_bp")[1]
        heart_disease = st.selectbox("Heart Disease", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0], key="heart_disease")[1]
        stroke = st.selectbox("Stroke History", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0], key="stroke")[1]
        
        smoker = st.selectbox("Smoker", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0], key="smoker")[1]
        phys_activity = st.selectbox("Physical Activity", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0], key="phys_activity")[1]
        
        fruits = st.selectbox("Fruits Consumption", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0], key="fruits")[1]
        veggies = st.selectbox("Vegetables Consumption", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0], key="veggies")[1]
    
    with col2:
        st.subheader("Health Indicators")
        
        gen_hlth = st.selectbox(
            "General Health",
            options=[("Excellent", 1), ("Very Good", 2), ("Good", 3), ("Fair", 4), ("Poor", 5)],
            format_func=lambda x: x[0],
            key="gen_hlth"
        )[1]
        
        ment_hlth = st.number_input("Mental Health (days/month)", min_value=0, max_value=30, value=0, step=1, key="ment_hlth")
        phys_hlth = st.number_input("Physical Health (days/month)", min_value=0, max_value=30, value=0, step=1, key="phys_hlth")
        
        diff_walk = st.selectbox("Difficulty Walking", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0], key="diff_walk")[1]
        
        education = st.selectbox(
            "Education Level",
            options=[("Select", 0), ("Less than High School", 1), ("High School", 2), ("Some College", 3), 
                     ("College Graduate", 4), ("Post Graduate", 5), ("Professional Degree", 6)],
            format_func=lambda x: x[0],
            key="education"
        )[1]
        
        income = st.selectbox(
            "Income Level",
            options=[("Select", 0), ("Less than $10,000", 1), ("$10,000 - $15,000", 2), ("$15,000 - $20,000", 3),
                     ("$20,000 - $25,000", 4), ("$25,000 - $35,000", 5), ("$35,000 - $50,000", 6),
                     ("$50,000 - $75,000", 7), ("$75,000+", 8)],
            format_func=lambda x: x[0],
            key="income"
        )[1]
        
        any_healthcare = st.selectbox("Any Healthcare", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0], key="any_healthcare")[1]
        no_doc_cost = st.selectbox("No Doctor Cost", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0], key="no_doc_cost")[1]
        heavy_alcohol = st.selectbox("Heavy Alcohol Consumption", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0], key="heavy_alcohol")[1]
        
        hba1c = st.number_input("HbA1c Level", min_value=0.0, max_value=15.0, value=5.0, step=0.1, key="hba1c")
        blood_glucose = st.number_input("Blood Glucose Level (mg/dL)", min_value=0, max_value=400, value=100, step=5, key="blood_glucose")
        
        smoking_history = st.selectbox(
            "Smoking History",
            options=[("Never", 0), ("No Info", 1), ("Current", 2), ("Former", 3), ("Ever", 4), ("Not Current", 5)],
            format_func=lambda x: x[0],
            key="smoking_history"
        )[1]
    
    if st.button("Predict Diabetes Risk", type="primary", use_container_width=True, key="predict"):
        features = {
            'BMI': bmi,
            'Age': float(age),
            'HighBP': float(high_bp),
            'HeartDiseaseorAttack': float(heart_disease),
            'Sex': float(sex),
            'GenHlth': float(gen_hlth),
            'MentHlth': float(ment_hlth),
            'PhysHlth': float(phys_hlth),
            'DiffWalk': float(diff_walk),
            'Education': float(education),
            'Income': float(income),
            'Smoker': float(smoker),
            'Stroke': float(stroke),
            'PhysActivity': float(phys_activity),
            'Fruits': float(fruits),
            'Veggies': float(veggies),
            'HvyAlcoholConsump': float(heavy_alcohol),
            'AnyHealthcare': float(any_healthcare),
            'NoDocbcCost': float(no_doc_cost),
            'HbA1c_level': float(hba1c),
            'blood_glucose_level': float(blood_glucose),
            'smoking_history_encoded': float(smoking_history)
        }
        
        with st.spinner("Analyzing your health data..."):
            risk_score, prediction, feature_importance = predict_risk(model, scaler, imputer, features)
            classification = get_classification(risk_score)
            recommendations = get_recommendations(risk_score)
            
            st.markdown("---")
            st.subheader("Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Risk Score", str(risk_score) + "%")
            
            with col2:
                st.metric("Classification", classification)
            
            with col3:
                st.metric("Prediction", "Diabetic" if prediction == 1 else "Not Diabetic")
            
            if risk_score < 30:
                st.success("Low Risk - Continue maintaining a healthy lifestyle")
            elif risk_score < 60:
                st.warning("Moderate Risk - Consider consulting a healthcare provider")
            else:
                st.error("High Risk - Please consult a healthcare provider immediately")
            
            if feature_importance:
                st.markdown("#### Key Factors Affecting Your Risk")
                
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.write("**Factor**")
                with col2:
                    st.write("**Impact**")
                
                for feature, importance in sorted_features[:5]:
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.write(feature)
                    with col2:
                        st.progress(importance / 100)
                        st.write(str(importance) + "%")
            
            st.markdown("#### Recommendations")
            for rec in recommendations:
                st.write("- " + rec)

with tab2:
    from chatbot import show_chatbot
    show_chatbot()