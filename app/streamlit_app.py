import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load Model and Scaler
@st.cache_resource
def load_model_and_scaler():
    with open('models/rf_pd_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

# App Title
st.set_page_config(page_title="NeuroKey PD Detector", layout="centered")
st.title("🧠 NeuroKey Parkinson's Detection")
st.write("This app predicts the likelihood of Parkinson’s Disease based on keystroke dynamics features.")

# Load model and scaler
model, scaler = load_model_and_scaler()

# Feature names and statistics (from your scaler)
feature_names = ['nqScore', 'Typing speed', 'afTap', 'sTap']
scaler_means = [0.0943, 106.44, 112.54, 168.38]
scaler_scales = [0.0849, 53.55, 28.28, 19.75]

# Display reference statistics
with st.expander("ℹ️ Feature Statistics (Training Data)"):
    stats_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean': scaler_means,
        'Std Dev': scaler_scales
    })
    st.dataframe(stats_df)
    st.markdown("➡️ Try to input values close to these ranges for accurate predictions.")

# Input Form
st.header("👉 Enter Keystroke Metrics")

col1, col2 = st.columns(2)

with col1:
    nqScore = st.number_input("NeuroQWERTY Score (e.g., 0.0 – 1.0)", value=0.1, step=0.05)
    afTap = st.number_input("Alternating Finger Tapping Score (e.g., 80 – 140)", value=112.5, step=1.0)

with col2:
    typing_speed = st.number_input("Typing Speed (chars/sec) (e.g., 50 – 150)", value=106.4, step=1.0)
    sTap = st.number_input("Single Key Tapping Score (e.g., 140 – 200)", value=168.4, step=1.0)

# Predict Button
if st.button("🔍 Predict Parkinson's Probability"):
    
    # Prepare raw input
    input_df = pd.DataFrame([{
        'nqScore': nqScore,
        'Typing speed': typing_speed,
        'afTap': afTap,
        'sTap': sTap
    }])

    # Scale input
    scaled_input = scaler.transform(input_df)

    # Predict
    pred_prob = model.predict_proba(scaled_input)[0][1]
    pred_class = model.predict(scaled_input)[0]

    st.subheader("🩺 Prediction Result:")
    st.write(f"**Probability of Parkinson's:** `{pred_prob * 100:.2f}%`")

    if pred_class == 1:
        st.success("🧩 Likely PD Detected")
    else:
        st.info("✅ No PD Detected")
