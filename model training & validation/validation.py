import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import joblib

# Load your synthetic dataset
df = pd.read_csv("keystroke_features.csv")  # Use your filename

# Separate features and targets
features = [
    'mean_hold_time', 'std_hold_time', 'mean_flight_time', 'std_flight_time',
    'hold_time_variability', 'backspace_rate', 'pause_frequency',
    'session_duration', 'total_keystrokes', 'typing_speed'
]

X = df[features]
y_updrs = df['updrs_score']
y_tremor = df['tremor_severity']
y_diag = df['diagnosis'].map({'Healthy': 0, "Parkinson's": 1})

# Split into train, validation, and test
X_temp, X_test, y_updrs_temp, y_updrs_test, y_tremor_temp, y_tremor_test, y_diag_temp, y_diag_test = train_test_split(
    X, y_updrs, y_tremor, y_diag, test_size=0.2, random_state=42
)
X_train, X_val, y_updrs_train, y_updrs_val, y_tremor_train, y_tremor_val, y_diag_train, y_diag_val = train_test_split(
    X_temp, y_updrs_temp, y_tremor_temp, y_diag_temp, test_size=0.25, random_state=42
)  # 0.25 x 0.8 = 20% validation

# Train or load models
updrs_model = RandomForestRegressor(n_estimators=100, random_state=42)
tremor_model = RandomForestRegressor(n_estimators=100, random_state=42)
diagnosis_model = RandomForestClassifier(n_estimators=100, random_state=42)

updrs_model.fit(X_train, y_updrs_train)
tremor_model.fit(X_train, y_tremor_train)
diagnosis_model.fit(X_train, y_diag_train)

# --- Validation Scores ---
print("\n=== VALIDATION RESULTS ===")
updrs_val_pred = updrs_model.predict(X_val)
tremor_val_pred = tremor_model.predict(X_val)
diag_val_pred = diagnosis_model.predict(X_val)

print("\n[UPDRS Regression]")
print("MSE:", mean_squared_error(y_updrs_val, updrs_val_pred))

print("\n[Tremor Regression]")
print("MSE:", mean_squared_error(y_tremor_val, tremor_val_pred))

print("\n[Diagnosis Classification]")
print("Accuracy:", accuracy_score(y_diag_val, diag_val_pred))
print(classification_report(y_diag_val, diag_val_pred))

# --- Test Scores ---
print("\n=== TEST RESULTS ===")
updrs_test_pred = updrs_model.predict(X_test)
tremor_test_pred = tremor_model.predict(X_test)
diag_test_pred = diagnosis_model.predict(X_test)

print("\n[UPDRS Regression]")
print("MSE:", mean_squared_error(y_updrs_test, updrs_test_pred))

print("\n[Tremor Regression]")
print("MSE:", mean_squared_error(y_tremor_test, tremor_test_pred))

print("\n[Diagnosis Classification]")
print("Accuracy:", accuracy_score(y_diag_test, diag_test_pred))
print(classification_report(y_diag_test, diag_test_pred))

# Save models
joblib.dump(updrs_model, "updrs_model.pkl")
joblib.dump(tremor_model, "tremor_model.pkl")
joblib.dump(diagnosis_model, "diagnosis_model.pkl")
