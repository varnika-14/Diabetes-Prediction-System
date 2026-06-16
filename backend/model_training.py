import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight
import joblib
import os
import json
import warnings
warnings.filterwarnings("ignore")

print("STEP 2: MODEL TRAINING")
print("")

print("Loading combined dataset...")
df = pd.read_csv('data/combined_diabetes_dataset.csv')
print("Dataset loaded: " + str(df.shape[0]) + " rows, " + str(df.shape[1]) + " columns")

your_features = [
    'BMI', 'Age', 'HighBP', 'HeartDiseaseorAttack', 'Sex',
    'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Education',
    'Income', 'Smoker', 'Stroke', 'PhysActivity', 'Fruits',
    'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost',
    'HbA1c_level', 'blood_glucose_level', 'smoking_history_encoded'
]

print("Features used for training:")
for i, f in enumerate(your_features, 1):
    print(str(i) + ". " + f)

X = df[your_features]
y = df['target']

print("")
print("Features shape: " + str(X.shape))
print("Target shape: " + str(y.shape))
print("Class distribution:")
print(y.value_counts())

print("")
print("Checking for missing values...")
nan_counts = X.isnull().sum()
if nan_counts.sum() > 0:
    print("NaN counts per feature:")
    print(nan_counts[nan_counts > 0])
else:
    print("No missing values found")

print("")
print("Imputing missing values...")
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=your_features)
print("Missing values imputed with median")

print("")
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training set: " + str(X_train.shape[0]) + " samples")
print("Testing set: " + str(X_test.shape[0]) + " samples")

print("")
print("Scaling features...")
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Feature scaling completed")

os.makedirs('models', exist_ok=True)
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(imputer, 'models/imputer.pkl')
print("Scaler and imputer saved")

print("")
print("Calculating class weights for imbalance...")
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
weight_dict = dict(zip(np.unique(y_train), class_weights))
print("Class weights: " + str(weight_dict))

print("")
print("Training Random Forest Model...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight=weight_dict,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_scaled, y_train)
print("Random Forest training completed")

y_pred_rf = rf_model.predict(X_test_scaled)
y_prob_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
auc_rf = roc_auc_score(y_test, y_prob_rf)

print("Random Forest Results:")
print("Accuracy: " + str(round(accuracy_rf, 4)))
print("Precision: " + str(round(precision_rf, 4)))
print("Recall: " + str(round(recall_rf, 4)))
print("F1 Score: " + str(round(f1_rf, 4)))
print("AUC Score: " + str(round(auc_rf, 4)))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

print("")
print("Training Hist Gradient Boosting Model...")
gb_model = HistGradientBoostingClassifier(
    max_iter=100,
    learning_rate=0.1,
    max_depth=5,
    class_weight='balanced',
    random_state=42
)

gb_model.fit(X_train_scaled, y_train)
print("Hist Gradient Boosting training completed")

y_pred_gb = gb_model.predict(X_test_scaled)
y_prob_gb = gb_model.predict_proba(X_test_scaled)[:, 1]

accuracy_gb = accuracy_score(y_test, y_pred_gb)
precision_gb = precision_score(y_test, y_pred_gb)
recall_gb = recall_score(y_test, y_pred_gb)
f1_gb = f1_score(y_test, y_pred_gb)
auc_gb = roc_auc_score(y_test, y_prob_gb)

print("Hist Gradient Boosting Results:")
print("Accuracy: " + str(round(accuracy_gb, 4)))
print("Precision: " + str(round(precision_gb, 4)))
print("Recall: " + str(round(recall_gb, 4)))
print("F1 Score: " + str(round(f1_gb, 4)))
print("AUC Score: " + str(round(auc_gb, 4)))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_gb))

print("")
print("Training Logistic Regression Model...")
lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight='balanced',
    C=1.0
)

lr_model.fit(X_train_scaled, y_train)
print("Logistic Regression training completed")

y_pred_lr = lr_model.predict(X_test_scaled)
y_prob_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

accuracy_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)
auc_lr = roc_auc_score(y_test, y_prob_lr)

print("Logistic Regression Results:")
print("Accuracy: " + str(round(accuracy_lr, 4)))
print("Precision: " + str(round(precision_lr, 4)))
print("Recall: " + str(round(recall_lr, 4)))
print("F1 Score: " + str(round(f1_lr, 4)))
print("AUC Score: " + str(round(auc_lr, 4)))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))

print("")
print("Model Comparison:")
results = pd.DataFrame({
    "Model": ["Random Forest", "Hist Gradient Boosting", "Logistic Regression"],
    "Accuracy": [accuracy_rf, accuracy_gb, accuracy_lr],
    "Precision": [precision_rf, precision_gb, precision_lr],
    "Recall": [recall_rf, recall_gb, recall_lr],
    "F1 Score": [f1_rf, f1_gb, f1_lr],
    "AUC": [auc_rf, auc_gb, auc_lr]
})
print(results.to_string(index=False))

best_model_name = results.loc[results["AUC"].idxmax(), "Model"]
print("")
print("Best model based on AUC: " + best_model_name)

print("")
print("Feature Importance Analysis:")
if best_model_name == "Random Forest":
    best_model = rf_model
elif best_model_name == "Hist Gradient Boosting":
    best_model = gb_model
else:
    best_model = lr_model

if hasattr(best_model, "feature_importances_"):
    feature_importance = pd.DataFrame({
        'Feature': your_features,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("Top 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))

print("")
print("Saving Best Model...")
if best_model_name == "Random Forest":
    final_model = rf_model
elif best_model_name == "Hist Gradient Boosting":
    final_model = gb_model
else:
    final_model = lr_model

joblib.dump(final_model, 'models/diabetes_model.pkl')
print("Best model (" + best_model_name + ") saved to: models/diabetes_model.pkl")

model_info = {
    'model_name': best_model_name,
    'accuracy': float(results[results["Model"] == best_model_name]["Accuracy"].values[0]),
    'precision': float(results[results["Model"] == best_model_name]["Precision"].values[0]),
    'recall': float(results[results["Model"] == best_model_name]["Recall"].values[0]),
    'f1_score': float(results[results["Model"] == best_model_name]["F1 Score"].values[0]),
    'auc': float(results[results["Model"] == best_model_name]["AUC"].values[0]),
    'features': your_features,
    'feature_count': len(your_features)
}

with open('models/model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print("Model info saved to: models/model_info.json")
print("")
print("STEP 2 COMPLETED")
print("")
print("Final Model Summary:")
print("Best Model: " + best_model_name)
print("Accuracy: " + str(round(model_info['accuracy'], 4)) + " (" + str(round(model_info['accuracy']*100, 1)) + "%)")
print("AUC Score: " + str(round(model_info['auc'], 4)))
print("Recall: " + str(round(model_info['recall'], 4)) + " (How well it finds diabetics)")
print("Features: " + str(len(your_features)))
print("")
print("Generated Files:")
print("models/diabetes_model.pkl - Trained model")
print("models/scaler.pkl - Feature scaler")
print("models/imputer.pkl - NaN imputer")
print("models/model_info.json - Model metrics")