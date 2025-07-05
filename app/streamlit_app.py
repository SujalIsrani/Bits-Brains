import streamlit as st
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

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

# Input Form
st.header("👉 Enter Keystroke Metrics")

col1, col2 = st.columns(2)

with col1:
    nqScore = st.number_input("NeuroQWERTY Score", value=2.5, step=0.1)
    afTap = st.number_input("Alternating Finger Tapping Score", value=1.5, step=0.1)

with col2:
    typing_speed = st.number_input("Typing Speed (chars/sec)", value=3.0, step=0.1)
    sTap = st.number_input("Single Key Tapping Score", value=1.2, step=0.1)

# Predict Button
if st.button("🔍 Predict Parkinson's Probability"):
    
    # Prepare input
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
        st.success("Likely PD Detected")
    else:
        st.info("No PD Detected")

    