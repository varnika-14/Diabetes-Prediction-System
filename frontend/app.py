import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

st.set_page_config(
    page_title="Diabetes Risk Screening Tool",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem !important;
    }
    .stApp h1 {
        font-size: 38px !important;
        font-weight: 700 !important;
        color: #2C7BE5 !important;
    }
    .risk-low {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
    }
    .risk-moderate {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
    }
    .risk-high {
        background-color: #f8d7da;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    base_path = os.path.join('..','backend','models')
    
    model1 = joblib.load(os.path.join(base_path, 'model_nonclinical.pkl'))
    scaler1 = joblib.load(os.path.join(base_path, 'scaler_nonclinical.pkl'))
    with open(os.path.join(base_path, 'features_nonclinical.json'), 'r') as f:
        features1 = json.load(f)
    
    model2 = joblib.load(os.path.join(base_path, 'model_clinical.pkl'))
    scaler2 = joblib.load(os.path.join(base_path, 'scaler_clinical.pkl'))
    imputer2 = joblib.load(os.path.join(base_path, 'imputer_clinical.pkl'))
    with open(os.path.join(base_path, 'features_clinical.json'), 'r') as f:
        features2 = json.load(f)
    
    return {
        'nonclinical': {'model': model1, 'scaler': scaler1, 'features': features1},
        'clinical': {'model': model2, 'scaler': scaler2, 'imputer': imputer2, 'features': features2}
    }

def predict_risk(model_data, features_dict, is_clinical=False):
    model = model_data['model']
    scaler = model_data['scaler']
    
    feature_names = model_data['features']
    input_values = [features_dict[f] for f in feature_names]
    input_array = np.array([input_values])
    
    if is_clinical:
        imputer = model_data['imputer']
        input_array = imputer.transform(input_array)
    
    input_scaled = scaler.transform(input_array)
    
    probability = model.predict_proba(input_scaled)[0][1]
    risk_score = round(probability * 100, 1)
    prediction = model.predict(input_scaled)[0]
    
    return risk_score, prediction

def get_classification(risk_score):
    if risk_score < 30:
        return "Low Risk", "risk-low"
    elif risk_score < 60:
        return "Moderate Risk", "risk-moderate"
    else:
        return "High Risk", "risk-high"

def get_recommendations(risk_score, is_clinical=False):
    if risk_score < 30:
        return [
            "Continue maintaining a healthy lifestyle",
            "Exercise regularly (at least 30 minutes daily)",
            "Eat a balanced diet rich in fruits and vegetables",
            "Schedule annual check-ups"
        ]
    elif risk_score < 60:
        return [
            "Consult a healthcare provider for preventive care",
            "Consider lifestyle modifications",
            "Monitor blood glucose levels more frequently",
            "Increase physical activity"
        ]
    else:
        if is_clinical:
            return [
                "Consult a healthcare provider IMMEDIATELY",
                "Schedule comprehensive diabetes screening",
                "Consider medication consultation",
                "Monitor blood glucose levels daily",
                "Adopt a diabetes-prevention diet"
            ]
        else:
            return [
                "You have elevated risk based on lifestyle factors",
                "Schedule a comprehensive screening with blood tests",
                "Start moderate physical activity",
                "Review your diet with a nutritionist"
            ]

def main():
    st.title("Diabetes Risk Screening Tool")
    st.markdown("""
    ### Two Ways to Assess Your Risk
    
    Choose the option that works best for you:
    
    | Option | When to Use | Features | Accuracy |
    |--------|------------|----------|----------|
    | **Quick Screening** | Don't have recent blood test results | Lifestyle & demographic factors only | ~75% |
    | **Detailed Assessment** | Have HbA1c and blood glucose values | All factors including clinical tests | ~97% |
    """)
    
    models = load_models()
    
    tab1, tab2, tab3 = st.tabs(["Quick Screening (No Blood Tests)", "Detailed Assessment (With Blood Tests)", "Chat Assistant"])    
    with tab1:
        st.markdown("### Quick Screening - No Blood Tests Required")
        st.info("This assessment uses only your lifestyle and demographic information. No blood tests needed. You can complete this in 2 minutes.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Personal Information")
            bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1, key="bmi1")
            age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1, key="age1")
            sex = st.selectbox("Sex", [("Female", 0), ("Male", 1)], format_func=lambda x: x[0], key="sex1")[1]
            
            high_bp = st.selectbox("High Blood Pressure", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0], key="bp1")[1]
            heart_disease = st.selectbox("Heart Disease", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0], key="heart1")[1]
            stroke = st.selectbox("Stroke History", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0], key="stroke1")[1]
            
            smoker = st.selectbox("Smoker", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0], key="smoker1")[1]
            phys_activity = st.selectbox("Physical Activity", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0], key="phys1")[1]
            fruits = st.selectbox("Fruits Consumption", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0], key="fruits1")[1]
            veggies = st.selectbox("Vegetables Consumption", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0], key="veggies1")[1]
            
            gen_hlth = st.selectbox("General Health", 
                [("Excellent", 1), ("Very Good", 2), ("Good", 3), ("Fair", 4), ("Poor", 5)],
                format_func=lambda x: x[0], key="gen1")[1]
            
            ment_hlth = st.number_input("Mental Health (days/month)", min_value=0, max_value=30, value=0, step=1, key="ment1")
            phys_hlth = st.number_input("Physical Health (days/month)", min_value=0, max_value=30, value=0, step=1, key="physhlth1")
        
        with col2:
            st.subheader("Lifestyle Factors")
            
            diff_walk = st.selectbox("Difficulty Walking", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0], key="walk1")[1]
            
            education = st.selectbox("Education Level",
                [("Less than High School", 1), ("High School", 2), ("Some College", 3), 
                 ("College Graduate", 4), ("Post Graduate", 5), ("Professional Degree", 6)],
                format_func=lambda x: x[0], key="edu1")[1]
            
            income = st.selectbox("Income Level",
                [("Less than $25,000", 1), ("$25,000 - $35,000", 2), ("$35,000 - $50,000", 3),
                 ("$50,000 - $75,000", 4), ("$75,000+", 5)],
                format_func=lambda x: x[0], key="inc1")[1]
            
            any_healthcare = st.selectbox("Have Healthcare", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0], key="hc1")[1]
            no_doc_cost = st.selectbox("No Doctor Due to Cost", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0], key="doc1")[1]
            heavy_alcohol = st.selectbox("Heavy Alcohol Consumption", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0], key="alc1")[1]
            
            smoking_history = st.selectbox("Smoking History",
                [("Never", 0), ("Former", 1), ("Current", 2)],
                format_func=lambda x: x[0], key="smokehist1")[1]
        
        if st.button("Get Quick Risk Assessment", type="primary", use_container_width=True, key="predict1"):
            features = {
                'BMI': bmi, 'Age': float(age), 'HighBP': float(high_bp),
                'HeartDiseaseorAttack': float(heart_disease), 'Sex': float(sex),
                'GenHlth': float(gen_hlth), 'MentHlth': float(ment_hlth),
                'PhysHlth': float(phys_hlth), 'DiffWalk': float(diff_walk),
                'Education': float(education), 'Income': float(income),
                'Smoker': float(smoker), 'Stroke': float(stroke),
                'PhysActivity': float(phys_activity), 'Fruits': float(fruits),
                'Veggies': float(veggies), 'HvyAlcoholConsump': float(heavy_alcohol),
                'AnyHealthcare': float(any_healthcare), 'NoDocbcCost': float(no_doc_cost),
                'smoking_history_encoded': float(smoking_history)
            }
            
            with st.spinner("Analyzing your health data..."):
                risk_score, prediction = predict_risk(models['nonclinical'], features, is_clinical=False)
                classification, css_class = get_classification(risk_score)
                recommendations = get_recommendations(risk_score, is_clinical=False)
                
                st.markdown("---")
                st.subheader("Your Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Risk Score", f"{risk_score}%")
                with col2:
                    st.metric("Classification", classification)
                with col3:
                    st.metric("Prediction", "Diabetic" if prediction == 1 else "Not Diabetic")
                
                st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
                st.write(f"**{classification}** - Based on your lifestyle and demographic factors.")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown("### Recommendations")
                for rec in recommendations:
                    st.write(rec)
                
                if risk_score >= 30:
                    st.warning("Consider a detailed assessment with blood tests for more accurate results")
    
    with tab2:
        st.markdown("### Detailed Assessment - With Blood Test Results")
        st.info("This assessment uses clinical test results for highest accuracy. You'll need your recent HbA1c and blood glucose values.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Personal Information")
            bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1, key="bmi2")
            age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1, key="age2")
            sex = st.selectbox("Sex", [("Female", 0), ("Male", 1)], format_func=lambda x: x[0], key="sex2")[1]
            
            high_bp = st.selectbox("High Blood Pressure", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0], key="bp2")[1]
            heart_disease = st.selectbox("Heart Disease", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0], key="heart2")[1]
            stroke = st.selectbox("Stroke History", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0], key="stroke2")[1]
            
            smoker = st.selectbox("Smoker", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0], key="smoker2")[1]
            phys_activity = st.selectbox("Physical Activity", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0], key="phys2")[1]
            
            fruits = st.selectbox("Fruits Consumption", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0], key="fruits2")[1]
            veggies = st.selectbox("Vegetables Consumption", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0], key="veggies2")[1]
            
            gen_hlth = st.selectbox("General Health", 
                [("Excellent", 1), ("Very Good", 2), ("Good", 3), ("Fair", 4), ("Poor", 5)],
                format_func=lambda x: x[0], key="gen2")[1]
            
            ment_hlth = st.number_input("Mental Health (days/month)", min_value=0, max_value=30, value=0, step=1, key="ment2")
            phys_hlth = st.number_input("Physical Health (days/month)", min_value=0, max_value=30, value=0, step=1, key="physhlth2")
        
        with col2:
            st.subheader("Clinical Tests Required")
            
            st.markdown("**Blood Test Values**")
            st.markdown("These are the most important predictors of diabetes.")
            
            hba1c = st.number_input(
                "HbA1c Level (%)", 
                min_value=0.0, max_value=15.0, value=5.0, step=0.1, 
                key="hba1c",
                help="Normal: < 5.7% | Prediabetes: 5.7-6.4% | Diabetes: ≥ 6.5%"
            )
            
            blood_glucose = st.number_input(
                "Blood Glucose Level (mg/dL)", 
                min_value=0, max_value=400, value=100, step=5,
                key="glucose",
                help="Normal: < 100 mg/dL | Prediabetes: 100-125 mg/dL | Diabetes: ≥ 126 mg/dL"
            )
            
            st.markdown("---")
            st.subheader("Lifestyle Factors")
            
            diff_walk = st.selectbox("Difficulty Walking", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0], key="walk2")[1]
            
            education = st.selectbox("Education Level",
                [("Less than High School", 1), ("High School", 2), ("Some College", 3), 
                 ("College Graduate", 4), ("Post Graduate", 5), ("Professional Degree", 6)],
                format_func=lambda x: x[0], key="edu2")[1]
            
            income = st.selectbox("Income Level",
                [("Less than $25,000", 1), ("$25,000 - $35,000", 2), ("$35,000 - $50,000", 3),
                 ("$50,000 - $75,000", 4), ("$75,000+", 5)],
                format_func=lambda x: x[0], key="inc2")[1]
            
            any_healthcare = st.selectbox("Have Healthcare", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0], key="hc2")[1]
            no_doc_cost = st.selectbox("No Doctor Due to Cost", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0], key="doc2")[1]
            heavy_alcohol = st.selectbox("Heavy Alcohol Consumption", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0], key="alc2")[1]
            
            smoking_history = st.selectbox("Smoking History",
                [("Never", 0), ("Former", 1), ("Current", 2)],
                format_func=lambda x: x[0], key="smokehist2")[1]
        
        if st.button("Get Detailed Assessment", type="primary", use_container_width=True, key="predict2"):
            features = {
                'BMI': bmi, 'Age': float(age), 'HighBP': float(high_bp),
                'HeartDiseaseorAttack': float(heart_disease), 'Sex': float(sex),
                'GenHlth': float(gen_hlth), 'MentHlth': float(ment_hlth),
                'PhysHlth': float(phys_hlth), 'DiffWalk': float(diff_walk),
                'Education': float(education), 'Income': float(income),
                'Smoker': float(smoker), 'Stroke': float(stroke),
                'PhysActivity': float(phys_activity), 'Fruits': float(fruits),
                'Veggies': float(veggies), 'HvyAlcoholConsump': float(heavy_alcohol),
                'AnyHealthcare': float(any_healthcare), 'NoDocbcCost': float(no_doc_cost),
                'HbA1c_level': float(hba1c),
                'blood_glucose_level': float(blood_glucose),
                'smoking_history_encoded': float(smoking_history)
            }
            
            with st.spinner("Analyzing your health data..."):
                risk_score, prediction = predict_risk(models['clinical'], features, is_clinical=True)
                classification, css_class = get_classification(risk_score)
                recommendations = get_recommendations(risk_score, is_clinical=True)
                
                st.markdown("---")
                st.subheader("Your Detailed Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Risk Score", f"{risk_score}%")
                with col2:
                    st.metric("Classification", classification)
                with col3:
                    st.metric("Prediction", "Diabetic" if prediction == 1 else "Not Diabetic")
                
                st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
                if risk_score < 30:
                    st.write("**Low Risk** - Your clinical values and lifestyle indicate low diabetes risk.")
                elif risk_score < 60:
                    st.write("**Moderate Risk** - Your clinical values suggest elevated risk. Consider lifestyle changes.")
                else:
                    st.write("**High Risk** - Your clinical values indicate high diabetes risk. Please consult a doctor immediately.")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown("### Recommendations")
                for rec in recommendations:
                    st.write(rec)
                
                if hba1c > 0 or blood_glucose > 0:
                    st.markdown("### Key Clinical Indicators")
                    col1, col2 = st.columns(2)
                    with col1:
                        hba1c_status = "Normal" if hba1c < 5.7 else "Prediabetes" if hba1c < 6.5 else "Diabetes"
                        st.metric("HbA1c Level", f"{hba1c}%", delta=hba1c_status)
                    with col2:
                        glucose_status = "Normal" if blood_glucose < 100 else "Prediabetes" if blood_glucose < 126 else "Diabetes"
                        st.metric("Blood Glucose", f"{blood_glucose} mg/dL", delta=glucose_status)
    with tab3:
        from chatbot import show_chatbot
        show_chatbot()

if __name__ == "__main__":
    main()