import pandas as pd
import numpy as np
import os
import json

print("STEP 1: DATA PREPARATION & MERGING")
print("")

REAL_FILE_PATH = "data\diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
SYNTHETIC_FILE_PATH = "data\diabetes_prediction_dataset.csv"

print("Loading datasets...")
df_real = pd.read_csv(REAL_FILE_PATH)
df_synthetic = pd.read_csv(SYNTHETIC_FILE_PATH)

print("Real dataset: " + str(df_real.shape[0]) + " rows, " + str(df_real.shape[1]) + " columns")
print("Synthetic dataset: " + str(df_synthetic.shape[0]) + " rows, " + str(df_synthetic.shape[1]) + " columns")
print("")

print("Preparing real dataset...")
df_real_clean = df_real.copy()
df_real_clean = df_real_clean.rename(columns={'Diabetes_binary': 'target'})

your_features = [
    'BMI', 'Age', 'HighBP', 'HeartDiseaseorAttack', 'Sex',
    'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Education',
    'Income', 'Smoker', 'Stroke', 'PhysActivity', 'Fruits',
    'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost',
    'HbA1c_level', 'blood_glucose_level', 'smoking_history_encoded'
]

real_feature_map = {
    'BMI': 'BMI',
    'Age': 'Age',
    'HighBP': 'HighBP',
    'HeartDiseaseorAttack': 'HeartDiseaseorAttack',
    'Sex': 'Sex',
    'GenHlth': 'GenHlth',
    'MentHlth': 'MentHlth',
    'PhysHlth': 'PhysHlth',
    'DiffWalk': 'DiffWalk',
    'Education': 'Education',
    'Income': 'Income',
    'Smoker': 'Smoker',
    'Stroke': 'Stroke',
    'PhysActivity': 'PhysActivity',
    'Fruits': 'Fruits',
    'Veggies': 'Veggies',
    'HvyAlcoholConsump': 'HvyAlcoholConsump',
    'AnyHealthcare': 'AnyHealthcare',
    'NoDocbcCost': 'NoDocbcCost'
}

real_data = {}
for real_col, target_col in real_feature_map.items():
    if real_col in df_real_clean.columns:
        real_data[target_col] = df_real_clean[real_col]
    else:
        real_data[target_col] = np.nan

real_data['HbA1c_level'] = np.nan
real_data['blood_glucose_level'] = np.nan
real_data['smoking_history_encoded'] = np.nan
real_data['target'] = df_real_clean['target']

df_real_aligned = pd.DataFrame(real_data)
print("Real dataset prepared: " + str(df_real_aligned.shape))

print("Preparing synthetic dataset...")
df_synthetic_clean = df_synthetic.copy()
df_synthetic_clean = df_synthetic_clean.rename(columns={'diabetes': 'target'})

synth_data = {}

synth_data['BMI'] = df_synthetic_clean['bmi'] if 'bmi' in df_synthetic_clean.columns else np.nan
synth_data['Age'] = df_synthetic_clean['age'] if 'age' in df_synthetic_clean.columns else np.nan
synth_data['HighBP'] = df_synthetic_clean['hypertension'] if 'hypertension' in df_synthetic_clean.columns else np.nan
synth_data['HeartDiseaseorAttack'] = df_synthetic_clean['heart_disease'] if 'heart_disease' in df_synthetic_clean.columns else np.nan

if 'gender' in df_synthetic_clean.columns:
    synth_data['Sex'] = df_synthetic_clean['gender'].map({'Female': 0, 'Male': 1, 'F': 0, 'M': 1}).fillna(0)
else:
    synth_data['Sex'] = np.nan

synth_data['GenHlth'] = np.nan
synth_data['MentHlth'] = np.nan
synth_data['PhysHlth'] = np.nan
synth_data['DiffWalk'] = np.nan
synth_data['Education'] = np.nan
synth_data['Income'] = np.nan
synth_data['Smoker'] = np.nan
synth_data['Stroke'] = np.nan
synth_data['PhysActivity'] = np.nan
synth_data['Fruits'] = np.nan
synth_data['Veggies'] = np.nan
synth_data['HvyAlcoholConsump'] = np.nan
synth_data['AnyHealthcare'] = np.nan
synth_data['NoDocbcCost'] = np.nan

synth_data['HbA1c_level'] = df_synthetic_clean['HbA1c_level'] if 'HbA1c_level' in df_synthetic_clean.columns else np.nan
synth_data['blood_glucose_level'] = df_synthetic_clean['blood_glucose_level'] if 'blood_glucose_level' in df_synthetic_clean.columns else np.nan

if 'smoking_history' in df_synthetic_clean.columns:
    smoking_map = {
        'never': 0,
        'No Info': 1,
        'current': 2,
        'former': 3,
        'ever': 4,
        'not current': 5
    }
    synth_data['smoking_history_encoded'] = df_synthetic_clean['smoking_history'].map(smoking_map).fillna(0)
else:
    synth_data['smoking_history_encoded'] = np.nan

synth_data['target'] = df_synthetic_clean['target']

df_synthetic_aligned = pd.DataFrame(synth_data)
print("Synthetic dataset prepared: " + str(df_synthetic_aligned.shape))

print("Merging datasets...")
for feature in your_features:
    if feature not in df_real_aligned.columns:
        df_real_aligned[feature] = np.nan
    if feature not in df_synthetic_aligned.columns:
        df_synthetic_aligned[feature] = np.nan

df_combined = pd.concat([df_real_aligned, df_synthetic_aligned], ignore_index=True)

print("Combined dataset created")
print("Total records: " + str(len(df_combined)))
print("Real records: " + str(len(df_real_aligned)))
print("Synthetic records: " + str(len(df_synthetic_aligned)))
print("Features: " + str(len(your_features)))

print("Handling missing values...")
numeric_cols = df_combined.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df_combined[col].isnull().any():
        median_val = df_combined[col].median()
        df_combined[col].fillna(median_val, inplace=True)

print("Missing values handled")

print("Class distribution:")
target_dist = df_combined['target'].value_counts()
target_pct = df_combined['target'].value_counts(normalize=True) * 100

for label in [0, 1]:
    count = target_dist.get(label, 0)
    pct = target_pct.get(label, 0)
    status = "Non-Diabetic" if label == 0 else "Diabetic"
    print(status + " (" + str(label) + "): " + str(count) + " records (" + str(round(pct, 1)) + "%)")

print("Saving prepared data...")
os.makedirs('data', exist_ok=True)

df_combined.to_csv('data/combined_diabetes_dataset.csv', index=False)
df_real_aligned.to_csv('data/real_dataset_aligned.csv', index=False)
df_synthetic_aligned.to_csv('data/synthetic_dataset_aligned.csv', index=False)

with open('data/features_list.json', 'w') as f:
    json.dump(your_features, f, indent=2)

info = {
    'total_records': int(len(df_combined)),
    'real_records': int(len(df_real_aligned)),
    'synthetic_records': int(len(df_synthetic_aligned)),
    'features': your_features,
    'feature_count': len(your_features),
    'target_distribution': {int(k): int(v) for k, v in target_dist.to_dict().items()},
    'target_percentage': {int(k): float(v) for k, v in target_pct.to_dict().items()}
}

with open('data/dataset_info.json', 'w') as f:
    json.dump(info, f, indent=2)

print("Combined dataset saved to: data/combined_diabetes_dataset.csv")
print("Features list saved to: data/features_list.json")
print("")

print("Total samples: " + str(len(df_combined)))
print("Features: " + str(len(your_features)))
print("")
print("Features:")
for i, feature in enumerate(your_features, 1):
    print(str(i) + ". " + feature)
