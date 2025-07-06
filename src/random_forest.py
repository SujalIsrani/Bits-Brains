import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def load_data(file_path):
    """
    Load preprocessed data from CSV and split into train-test.
    """
    from sklearn.model_selection import train_test_split  # Moved import inside

    df = pd.read_csv(file_path)
    df = df.dropna(subset=['gt', 'nqScore', 'Typing speed', 'afTap', 'sTap'])

    X = df[['nqScore', 'Typing speed', 'afTap', 'sTap']]
    y = df['gt']

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y), X.columns.tolist()


def train_and_save_rf(file_path, model_path, scaler_path):
    """
    Train Random Forest Classifier on the data, save model and scaler.
    """

    # Load data
    (X_train, X_test, y_train, y_test), feature_names = load_data(file_path)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Predict & Evaluate
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Healthy', 'PD'])
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n✅ Random Forest Accuracy: {acc * 100:.2f}%")
    print(f"\n✅ Classification Report:\n{report}")

    # Plot Confusion Matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy', 'PD'], yticklabels=['Healthy', 'PD'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Random Forest Confusion Matrix')
    plt.tight_layout()
    plt.show()

    # Save model and scaler
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"\n✅ Model saved to: {model_path}")
    print(f"✅ Scaler saved to: {scaler_path}")
