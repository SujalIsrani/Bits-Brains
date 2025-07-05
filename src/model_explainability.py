import shap
import pickle
import matplotlib.pyplot as plt
import pandas as pd

def load_model(model_path):
    """
    Load the trained XGBoost model from a pickle file.
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def explain_model(model, X_train, feature_names, save_summary_path=None):
    """
    Generate SHAP summary plot to explain global feature importance.

    Args:
        model: Trained XGBoost model.
        X_train: Scaled training feature set (numpy array).
        feature_names: List of feature names.
        save_summary_path: Optional path to save the summary plot as image.
    """

    # Create SHAP explainer
    explainer = shap.Explainer(model)

    # Compute SHAP values for training data
    shap_values = explainer(X_train)

    # Summary plot (global feature importance)
    shap.summary_plot(shap_values, features=X_train, feature_names=feature_names, show=False)

    # Save plot if path provided
    if save_summary_path:
        plt.savefig(save_summary_path, bbox_inches='tight')
        print(f"✅ SHAP summary plot saved to: {save_summary_path}")
    
    plt.show()


def explain_single_prediction(model, scaler, feature_names, input_features):
    """
    Explain a single manual prediction using SHAP force plot.

    Args:
        model: Trained XGBoost model.
        scaler: StandardScaler fitted on training data.
        feature_names: List of feature names.
        input_features: Dict of manual input features {feature_name: value}

    Returns:
        SHAP values, prediction probability
    """
    # Prepare input as DataFrame
    df_input = pd.DataFrame([input_features])

    # Scale using same scaler
    X_input_scaled = scaler.transform(df_input)

    # Create SHAP explainer
    explainer = shap.Explainer(model)

    # Compute SHAP values
    shap_values = explainer(X_input_scaled)

    # Predict probability
    pred_prob = model.predict_proba(X_input_scaled)[0][1]  # Probability of PD class

    # Display force plot
    shap.plots.waterfall(shap_values[0], show=True, feature_names=feature_names)

    return shap_values, pred_prob
