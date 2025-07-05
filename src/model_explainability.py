import shap
import pickle
import matplotlib.pyplot as plt
import pandas as pd

def load_model(model_path):
    """
    Load the trained model from a pickle file (any scikit-learn compatible model).
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def explain_model(model, X_train, feature_names, save_summary_path=None):
    """
    Generate SHAP summary plot to explain global feature importance.

    Args:
        model: Trained model (Random Forest, XGBoost, etc.).
        X_train: Scaled training feature set (numpy array).
        feature_names: List of feature names.
        save_summary_path: Optional path to save the summary plot as image.
    """
    # Create SHAP explainer
    explainer = shap.Explainer(model, X_train)

    # Compute SHAP values
    shap_values = explainer(X_train)

    # SHAP summary plot
    shap.summary_plot(shap_values, features=X_train, feature_names=feature_names, show=False)

    if save_summary_path:
        plt.savefig(save_summary_path, bbox_inches='tight')
        print(f"✅ SHAP summary plot saved to: {save_summary_path}")

    plt.show()


def explain_single_prediction(model, scaler, feature_names, input_features):
    """
    Explain a single manual prediction using SHAP waterfall plot.

    Args:
        model: Trained model.
        scaler: Fitted scaler.
        feature_names: List of feature names.
        input_features: Dict of manual inputs {feature_name: value}

    Returns:
        SHAP values, prediction probability
    """
    # Prepare input as DataFrame
    df_input = pd.DataFrame([input_features])

    # Scale input
    X_input_scaled = scaler.transform(df_input)

    # Create explainer
    explainer = shap.Explainer(model, X_input_scaled)

    # Compute SHAP values
    shap_values = explainer(X_input_scaled)

    # Predict probability
    pred_prob = model.predict_proba(X_input_scaled)[0][1]

    # Plot waterfall
    shap.plots.waterfall(shap_values[0], show=True, feature_names=feature_names)

    return shap_values, pred_prob
