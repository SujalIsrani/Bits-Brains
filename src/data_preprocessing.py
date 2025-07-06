import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path):
    """
    Loads the merged NeuroQWERTY dataset and preprocesses it for ML modeling.

    Args:
        file_path (str): Path to the merged CSV file.

    Returns:
        X_train, X_test, y_train, y_test: Preprocessed data splits.
        scaler: Fitted scaler (optional use in app).
        feature_names: List of feature names used.
    """

    # Load the merged CSV
    df = pd.read_csv(file_path)

    # Preview
    print("Data Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # Drop rows with missing critical data (optional: fillna instead)
    df = df.dropna(subset=['gt', 'nqScore', 'Typing speed', 'afTap', 'sTap'])

    # Define Features (X) and Target (y)
    features = ['nqScore', 'Typing speed', 'afTap', 'sTap']
    target = 'gt'  # PD = 1, Healthy = 0

    X = df[features]
    y = df[target]

    # Scale Features (important for XGBoost SHAP consistency & other models)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, scaler, features

