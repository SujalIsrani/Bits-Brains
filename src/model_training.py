import pickle
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from data_preprocessing import load_and_preprocess_data
import matplotlib.pyplot as plt
import seaborn as sns
import os

def train_and_evaluate(file_path, model_save_path, scaler_save_path):
    """
    Train an XGBoost classifier on the Parkinson's keystroke dataset and save both model and scaler.

    Args:
        file_path (str): Path to the processed merged CSV file.
        model_save_path (str): Path to save the trained model (.pkl).
        scaler_save_path (str): Path to save the fitted scaler (.pkl).
    """
    
    # Load data
    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess_data(file_path)

    # Define model
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Healthy', 'PD'])

    print("\n✅ Model Accuracy: {:.2f}%".format(acc * 100))
    print("\n✅ Classification Report:\n", report)

    # Plot confusion matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy', 'PD'], yticklabels=['Healthy', 'PD'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

    # Save model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    with open(model_save_path, 'wb') as f:
        pickle.dump(model, f)

    # Save scaler
    os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
    with open(scaler_save_path, 'wb') as f:
        pickle.dump(scaler, f)

    print(f'\n✅ Model saved to: {model_save_path}')
    print(f'✅ Scaler saved to: {scaler_save_path}')
