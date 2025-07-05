import streamlit as st
import pickle
import numpy as np
import pandas as pd
import json
import streamlit.components.v1 as components

# Load Model and Scaler
@st.cache_resource
def load_model_and_scaler():
    with open('models/rf_pd_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

# Streamlit Config
st.set_page_config(page_title="NeuroKey PD Detector", layout="centered")
st.title("🧠 NeuroKey Parkinson's Detection")
st.write("This app predicts the likelihood of Parkinson’s Disease based on keystroke dynamics features.")

model, scaler = load_model_and_scaler()

# Reference Stats
feature_names = ['nqScore', 'Typing speed', 'afTap', 'sTap']
scaler_means = [0.0943, 106.44, 112.54, 168.38]
scaler_scales = [0.0849, 53.55, 28.28, 19.75]

with st.expander("ℹ️ Feature Statistics (Training Data)"):
    stats_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean': scaler_means,
        'Std Dev': scaler_scales
    })
    st.dataframe(stats_df)
    st.markdown("➡️ Try to input values close to these ranges for accurate predictions.")

# Typing Test Section
st.header("⌨️ Typing Test")

with open("app/keystroke_logic.js", "r") as f:
    js_code = f.read()

st.markdown("""
    <style>
        #keystroke-input {
            color: white;
            background-color: #1e1e1e;
            border: 1px solid #ccc;
            padding: 10px;
            border-radius: 5px;
            font-size: 16px;
            width: 100%;
            min-height: 150px;
        }
    </style>
""", unsafe_allow_html=True)


components.html(f"""
<html>
<body>
    <h4 style="color:white;">Type the following paragraph below:</h4>
    <p style="border:1px solid #ccc;padding:10px;color:white;background-color:#1e1e1e;">
        The quick brown fox jumps over the lazy dog. Type this text repeatedly for around 5 minutes or as long as you can.
    </p>
    <textarea id="keystroke-input" style="width:100%;height:150px;padding:10px;font-size:16px;border-radius:5px;border:1px solid #ccc;color:white;background-color:#1e1e1e;"></textarea>
    <br><br>
    <button onclick="sendResults()">End Test & Send Data</button>

    <script>
        {js_code}
        function sendResults() {{
            const results = stopTestAndReturnResults();
            const json = JSON.stringify(results);
            const textarea = document.createElement('textarea');
            textarea.value = json;
            document.body.appendChild(textarea);
            textarea.select();
            document.execCommand('copy');
            alert('✅ Keystroke data copied to clipboard! Please paste it into the box in Streamlit.');
        }}
    </script>
</body>
</html>
""", height=500)

# Paste Section
st.header("📥 Paste Keystroke JSON Data Here")
user_json = st.text_area("Paste the JSON you copied after the typing test")

if user_json:
    try:
        user_data = json.loads(user_json)
        st.success("✅ Data successfully loaded!")

        # Display extracted values
        st.write("Extracted Metrics:", user_data)

        nqScore = user_data.get('nqScore', 0.1)
        typing_speed = user_data.get('typing_speed', 100)
        afTap = user_data.get('afTap', 100)
        sTap = user_data.get('sTap', 150)

        input_df = pd.DataFrame([{
            'nqScore': nqScore,
            'Typing speed': typing_speed,
            'afTap': afTap,
            'sTap': sTap
        }])

        scaled_input = scaler.transform(input_df)

        pred_prob = model.predict_proba(scaled_input)[0][1]
        pred_class = model.predict(scaled_input)[0]

        st.subheader("🩺 Prediction Result:")
        st.write(f"**Probability of Parkinson's:** `{pred_prob * 100:.2f}%`")

        if pred_class == 1:
            st.success("🧩 Likely PD Detected")
        else:
            st.info("✅ No PD Detected")

    except Exception as e:
        st.error(f"Error parsing input: {e}")
