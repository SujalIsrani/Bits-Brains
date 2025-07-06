import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import shap
import joblib

# Load the feature dataset
df = pd.read_csv("keystroke_features.csv")

# Encode categorical columns
df["diagnosis_label"] = df["diagnosis"].map({"Healthy": 0, "Parkinson's": 1})
df["medication_status"] = df["medication_status"].map({"NA": 0, "Off": 1, "On": 2})

# Select features and targets
features = [
    "mean_hold_time", "std_hold_time",
    "mean_flight_time", "std_flight_time",
    "hold_time_variability", "backspace_rate",
    "pause_frequency", "session_duration",
    "total_keystrokes", "typing_speed", "medication_status"
]

X = df[features]

# Apply Standard Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Targets
y_updrs = df["updrs_score"]
y_tremor = df["tremor_severity"]
y_diagnosis = df["diagnosis_label"]

# Train/Test split (on scaled data)
X_train, X_test, y_updrs_train, y_updrs_test = train_test_split(X_scaled, y_updrs, test_size=0.2, random_state=42)
_, _, y_tremor_train, y_tremor_test = train_test_split(X_scaled, y_tremor, test_size=0.2, random_state=42)
_, _, y_diag_train, y_diag_test = train_test_split(X_scaled, y_diagnosis, test_size=0.2, random_state=42)

# Train Random Forest Models
updrs_model = RandomForestRegressor(n_estimators=100, random_state=42)
updrs_model.fit(X_train, y_updrs_train)

tremor_model = RandomForestRegressor(n_estimators=100, random_state=42)
tremor_model.fit(X_train, y_tremor_train)

diagnosis_model = RandomForestClassifier(n_estimators=100, random_state=42)
diagnosis_model.fit(X_train, y_diag_train)

# Predictions
updrs_pred = updrs_model.predict(X_test)
tremor_pred = tremor_model.predict(X_test)
diag_pred = diagnosis_model.predict(X_test)

# Evaluation
print("\n=== UPDRS Score Regression ===")
print(f"MSE: {mean_squared_error(y_updrs_test, updrs_pred):.2f}")

print("\n=== Tremor Severity Regression ===")
print(f"MSE: {mean_squared_error(y_tremor_test, tremor_pred):.2f}")

print("\n=== Diagnosis Classification ===")
print(f"Accuracy: {accuracy_score(y_diag_test, diag_pred):.2f}")
print(classification_report(y_diag_test, diag_pred, target_names=["Healthy", "Parkinson's"]))

# SHAP Analysis (TreeExplainer - CPU friendly)
explainer = shap.TreeExplainer(updrs_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, show=True)

# Save models
joblib.dump(updrs_model, "updrs_model.pkl")
joblib.dump(tremor_model, "tremor_model.pkl")
joblib.dump(diagnosis_model, "diagnosis_model.pkl")
joblib.dump(scaler, "scaler.pkl")  # <-- Save the scaler

print("\nModels saved as:")
print(" - updrs_model.pkl")
print(" - tremor_model.pkl")
print(" - diagnosis_model.pkl")
print(" - scaler.pkl ✅ (for consistent input scaling)")
