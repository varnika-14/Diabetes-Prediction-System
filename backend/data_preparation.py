import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import joblib

print("STEP 1: DATA PREPARATION - VERSION 2")
print("")

CLINICAL_FEATURES = [
    'BMI', 'Age', 'HighBP', 'HeartDiseaseorAttack', 'Sex',
    'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Education',
    'Income', 'Smoker', 'Stroke', 'PhysActivity', 'Fruits',
    'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost',
    'HbA1c_level', 'blood_glucose_level', 'smoking_history_encoded'
]

NON_CLINICAL_FEATURES = [
    'BMI', 'Age', 'HighBP', 'HeartDiseaseorAttack', 'Sex',
    'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Education',
    'Income', 'Smoker', 'Stroke', 'PhysActivity', 'Fruits',
    'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost',
    'smoking_history_encoded'
]

print("CLINICAL FEATURES (with blood tests):", len(CLINICAL_FEATURES))
print("NON-CLINICAL FEATURES (no blood tests):", len(NON_CLINICAL_FEATURES))
print("")

REAL_FILE_PATH = "data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
SYNTHETIC_FILE_PATH = "data/diabetes_prediction_dataset.csv"

df_real = pd.read_csv(REAL_FILE_PATH)
df_synthetic = pd.read_csv(SYNTHETIC_FILE_PATH)

print("Loading datasets...")
print("Real dataset:", df_real.shape)
print("Synthetic dataset:", df_synthetic.shape)
print("")

print("="*60)
print("PREPARING MODEL 1: Non-Clinical Model")
print("="*60)

df_real_nonclinical = df_real.copy()
df_real_nonclinical = df_real_nonclinical.rename(columns={'Diabetes_binary': 'target'})

nonclinical_available = []
for feature in NON_CLINICAL_FEATURES:
    if feature in df_real_nonclinical.columns:
        nonclinical_available.append(feature)

X1 = df_real_nonclinical[nonclinical_available]
y1 = df_real_nonclinical['target']

print("Features for Model 1 (Non-Clinical):", len(nonclinical_available))
print("Target distribution:")
print(y1.value_counts())

X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1, y1, test_size=0.2, stratify=y1, random_state=42
)

scaler1 = MinMaxScaler()
X1_train_scaled = scaler1.fit_transform(X1_train)
X1_test_scaled = scaler1.transform(X1_test)

print("Training Model 1 (Non-Clinical)...")
model1 = HistGradientBoostingClassifier(
    max_iter=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
model1.fit(X1_train_scaled, y1_train)

y1_pred = model1.predict(X1_test_scaled)
y1_prob = model1.predict_proba(X1_test_scaled)[:, 1]

acc1 = accuracy_score(y1_test, y1_pred)
auc1 = roc_auc_score(y1_test, y1_prob)

print("Model 1 (Non-Clinical) Results:")
print(f"  Accuracy: {acc1:.4f}")
print(f"  AUC: {auc1:.4f}")
print("  Confusion Matrix:")
print(confusion_matrix(y1_test, y1_pred))
print("")

os.makedirs('models', exist_ok=True)
joblib.dump(model1, 'models/model_nonclinical.pkl')
joblib.dump(scaler1, 'models/scaler_nonclinical.pkl')
with open('models/features_nonclinical.json', 'w') as f:
    json.dump(nonclinical_available, f)
print("Model 1 saved: models/model_nonclinical.pkl")

print("")
print("="*60)
print("PREPARING MODEL 2: Clinical Model")
print("="*60)

df_real_aligned = df_real.copy()
df_real_aligned = df_real_aligned.rename(columns={'Diabetes_binary': 'target'})

real_mapping = {
    'BMI': 'BMI', 'Age': 'Age', 'HighBP': 'HighBP',
    'HeartDiseaseorAttack': 'HeartDiseaseorAttack',
    'Sex': 'Sex', 'GenHlth': 'GenHlth', 'MentHlth': 'MentHlth',
    'PhysHlth': 'PhysHlth', 'DiffWalk': 'DiffWalk',
    'Education': 'Education', 'Income': 'Income',
    'Smoker': 'Smoker', 'Stroke': 'Stroke',
    'PhysActivity': 'PhysActivity', 'Fruits': 'Fruits',
    'Veggies': 'Veggies', 'HvyAlcoholConsump': 'HvyAlcoholConsump',
    'AnyHealthcare': 'AnyHealthcare', 'NoDocbcCost': 'NoDocbcCost'
}

real_aligned = {}
for real_col, target_col in real_mapping.items():
    if real_col in df_real_aligned.columns:
        real_aligned[target_col] = df_real_aligned[real_col]

real_aligned['HbA1c_level'] = np.nan
real_aligned['blood_glucose_level'] = np.nan
real_aligned['smoking_history_encoded'] = np.nan
real_aligned['target'] = df_real_aligned['target']
df_real_aligned = pd.DataFrame(real_aligned)

df_synthetic_aligned = df_synthetic.copy()
df_synthetic_aligned = df_synthetic_aligned.rename(columns={'diabetes': 'target'})

synth_aligned = {}
synth_aligned['BMI'] = df_synthetic_aligned['bmi']
synth_aligned['Age'] = df_synthetic_aligned['age']
synth_aligned['HighBP'] = df_synthetic_aligned['hypertension']
synth_aligned['HeartDiseaseorAttack'] = df_synthetic_aligned['heart_disease']
synth_aligned['Sex'] = df_synthetic_aligned['gender'].map({'Female': 0, 'Male': 1}).fillna(0)
synth_aligned['GenHlth'] = np.nan
synth_aligned['MentHlth'] = np.nan
synth_aligned['PhysHlth'] = np.nan
synth_aligned['DiffWalk'] = np.nan
synth_aligned['Education'] = np.nan
synth_aligned['Income'] = np.nan
synth_aligned['Smoker'] = np.nan
synth_aligned['Stroke'] = np.nan
synth_aligned['PhysActivity'] = np.nan
synth_aligned['Fruits'] = np.nan
synth_aligned['Veggies'] = np.nan
synth_aligned['HvyAlcoholConsump'] = np.nan
synth_aligned['AnyHealthcare'] = np.nan
synth_aligned['NoDocbcCost'] = np.nan
synth_aligned['HbA1c_level'] = df_synthetic_aligned['HbA1c_level']
synth_aligned['blood_glucose_level'] = df_synthetic_aligned['blood_glucose_level']

smoking_map = {'never': 0, 'No Info': 1, 'current': 2, 'former': 3, 'ever': 4, 'not current': 5}
synth_aligned['smoking_history_encoded'] = df_synthetic_aligned['smoking_history'].map(smoking_map).fillna(0)
synth_aligned['target'] = df_synthetic_aligned['target']
df_synthetic_aligned = pd.DataFrame(synth_aligned)

df_combined = pd.concat([df_real_aligned, df_synthetic_aligned], ignore_index=True)

imputer = SimpleImputer(strategy='median')
X2 = df_combined[CLINICAL_FEATURES]
X2_imputed = imputer.fit_transform(X2)
X2 = pd.DataFrame(X2_imputed, columns=CLINICAL_FEATURES)
y2 = df_combined['target']

print("Features for Model 2 (Clinical):", len(CLINICAL_FEATURES))

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, stratify=y2, random_state=42
)

scaler2 = MinMaxScaler()
X2_train_scaled = scaler2.fit_transform(X2_train)
X2_test_scaled = scaler2.transform(X2_test)

print("Training Model 2 (Clinical)...")
model2 = HistGradientBoostingClassifier(
    max_iter=100,
    learning_rate=0.1,
    max_depth=5,
    class_weight='balanced',
    random_state=42
)
model2.fit(X2_train_scaled, y2_train)

y2_pred = model2.predict(X2_test_scaled)
y2_prob = model2.predict_proba(X2_test_scaled)[:, 1]

acc2 = accuracy_score(y2_test, y2_pred)
auc2 = roc_auc_score(y2_test, y2_prob)

print("Model 2 (Clinical) Results:")
print(f"  Accuracy: {acc2:.4f}")
print(f"  AUC: {auc2:.4f}")
print("  Confusion Matrix:")
print(confusion_matrix(y2_test, y2_pred))
print("")

joblib.dump(model2, 'models/model_clinical.pkl')
joblib.dump(scaler2, 'models/scaler_clinical.pkl')
joblib.dump(imputer, 'models/imputer_clinical.pkl')
with open('models/features_clinical.json', 'w') as f:
    json.dump(CLINICAL_FEATURES, f)
print("Model 2 saved: models/model_clinical.pkl")

print("")
print("="*60)
print("SUMMARY")
print("="*60)
print(f"Model 1 (Non-Clinical): Accuracy = {acc1:.4f}, AUC = {auc1:.4f}")
print(f"Model 2 (Clinical): Accuracy = {acc2:.4f}, AUC = {auc2:.4f}")
print("")
print("Generated Files:")
print("  models/model_nonclinical.pkl - Model for users without blood tests")
print("  models/model_clinical.pkl - Model for users with blood tests")
print("  models/scaler_nonclinical.pkl")
print("  models/scaler_clinical.pkl")
print("  models/imputer_clinical.pkl")
print("  models/features_nonclinical.json")
print("  models/features_clinical.json")
print("")
print("STEP 1 COMPLETED")