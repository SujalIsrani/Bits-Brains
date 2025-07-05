import streamlit as st
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load Model and Scaler
@st.cache_resource
def load_model_and_scaler():
    with open('models/xgb_pd_classifier.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

# App Title
st.set_page_config(page_title="NeuroKey PD Detector", layout="centered")
st.title("🧠 NeuroKey Parkinson's Detection")
st.write("This app predicts the likelihood of Parkinson’s Disease based on keystroke dynamics features.")

# Load trained model and scaler
model, scaler = load_model_and_scaler()

# Manual Input Form
st.header("👉 Enter Keystroke Metrics")

col1, col2 = st.columns(2)

with col1:
    nqScore = st.number_input("NeuroQWERTY Score", min_value=0.0, max_value=10.0, value=2.5, step=0.1)
    afTap = st.number_input("Alternating Finger Tapping Score", min_value=0.0, max_value=5.0, value=1.5, step=0.1)
    
with col2:
    typing_speed = st.number_input("Typing Speed (chars/sec)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
    sTap = st.number_input("Single Key Tapping Score", min_value=0.0, max_value=5.0, value=1.2, step=0.1)

# Prediction Button
if st.button("🔍 Predict Parkinson's Probability"):
    
    # Input feature vector as DataFrame
    input_df = pd.DataFrame([{
        'nqScore': nqScore,
        'Typing speed': typing_speed,
        'afTap': afTap,
        'sTap': sTap
    }])

    # Scale inputs
    scaled_input = scaler.transform(input_df)

    # Predict probabilities
    pred_proba = model.predict_proba(scaled_input)[0][1]  # Probability of PD class
    pred_class = 1 if pred_proba >= 0.3 else 0  # Threshold set to 30% for higher sensitivity

    # Display Prediction
    st.subheader("🩺 Prediction Result:")
    st.write(f"**Probability of Parkinson's:** `{pred_proba*100:.2f}%`")
    if pred_class == 1:
        st.success("✅ Likely PD Detected")
    else:
        st.error("❌ No PD Detected")

    # SHAP Explanation
    st.subheader("📊 Feature Impact (Explainability)")

    # SHAP explainer (TreeExplainer for XGBoost)
    explainer = shap.Explainer(model)
    shap_values = explainer(scaled_input)

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)
