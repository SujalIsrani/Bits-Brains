import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Load dataset
df = pd.read_csv("keystroke_features_noisy_stronger.csv")  # Update with your actual filename

# Features and labels
features = [
    'mean_hold_time', 'std_hold_time', 'mean_flight_time', 'std_flight_time',
    'hold_time_variability', 'backspace_rate', 'pause_frequency',
    'session_duration', 'total_keystrokes', 'typing_speed'
]

X = df[features]
y_class = df['diagnosis'].map({"Healthy": 0, "Parkinson's": 1})
y_updrs = df['updrs_score']
y_tremor = df['tremor_severity']

# Train-test split
X_train, X_test, y_class_train, y_class_test = train_test_split(X, y_class, test_size=0.2, random_state=42)
_, _, y_updrs_train, y_updrs_test = train_test_split(X, y_updrs, test_size=0.2, random_state=42)
_, _, y_tremor_train, y_tremor_test = train_test_split(X, y_tremor, test_size=0.2, random_state=42)

# Classifiers to try
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

# Regressors to try
regressors = {
    "Linear Regression": LinearRegression(),
    "SVR": SVR(),
    "Decision Tree Regressor": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(),
    "Gradient Boosting Regressor": GradientBoostingRegressor()
}

print("\n=== DIAGNOSIS CLASSIFICATION RESULTS ===")
for name, clf in classifiers.items():
    clf.fit(X_train, y_class_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_class_test, y_pred)
    print(f"\n{name}")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_class_test, y_pred, target_names=["Healthy", "Parkinson's"]))

print("\n=== UPDRS SCORE REGRESSION RESULTS ===")
for name, reg in regressors.items():
    reg.fit(X_train, y_updrs_train)
    y_pred = reg.predict(X_test)
    mse = mean_squared_error(y_updrs_test, y_pred)
    print(f"{name}: MSE = {mse:.2f}")

print("\n=== TREMOR SEVERITY REGRESSION RESULTS ===")
for name, reg in regressors.items():
    reg.fit(X_train, y_tremor_train)
    y_pred = reg.predict(X_test)
    mse = mean_squared_error(y_tremor_test, y_pred)
    print(f"{name}: MSE = {mse:.4f}")
