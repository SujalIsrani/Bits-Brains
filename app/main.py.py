import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from math import pi
import shap


# Load models (update paths as needed)
updrs_model = joblib.load("updrs_model.pkl")
tremor_model = joblib.load("tremor_model.pkl")
diagnosis_model = joblib.load("diagnosis_model.pkl")

st.title("Parkinson's Keystroke Analysis Dashboard")

# Sidebar inputs for keystroke features
st.sidebar.header("Input Keystroke Features")

def user_input_features():
    features = {}
    features['mean_hold_time'] = float(st.sidebar.text_input('Mean Hold Time (s)', "0.2"))
    features['std_hold_time'] = float(st.sidebar.text_input('STD Hold Time (s)', "0.05"))
    features['mean_flight_time'] = float(st.sidebar.text_input('Mean Flight Time (s)', "0.3"))
    features['std_flight_time'] = float(st.sidebar.text_input('STD Flight Time (s)', "0.05"))
    features['hold_time_variability'] = float(st.sidebar.text_input('Hold Time Variability', "0.25"))
    features['backspace_rate'] = float(st.sidebar.text_input('Backspace Rate', "0.1"))
    features['pause_frequency'] = float(st.sidebar.text_input('Pause Frequency', "0.1"))
    features['session_duration'] = float(st.sidebar.text_input('Session Duration (s)', "50"))
    features['total_keystrokes'] = int(st.sidebar.text_input('Total Keystrokes', "300"))
    features['typing_speed'] = float(st.sidebar.text_input('Typing Speed (keys/min)', "180"))

    med_status_map = {"NA": 0, "Off": 1, "On": 2}
    med_status = st.sidebar.selectbox("Medication Status", options=["NA", "Off", "On"])
    features['medication_status'] = med_status_map[med_status]

    return pd.DataFrame([features])

input_df = user_input_features()


# Helper function to compute confidence intervals from an ensemble model
def get_prediction_with_ci(model, X, ci=0.95):
    try:
        if hasattr(model, 'estimators_'):  # works for RandomForest
            preds = np.array([est.predict(X)[0] for est in model.estimators_])
            mean_pred = np.mean(preds)
            lower = np.percentile(preds, (1 - ci) / 2 * 100)
            upper = np.percentile(preds, (1 + ci) / 2 * 100)
            return mean_pred, (lower, upper)
        else:
            # fallback if not ensemble
            pred = model.predict(X)[0]
            return pred, (pred, pred)
    except Exception as e:
        return model.predict(X)[0], (np.nan, np.nan)

# Predict with CI
updrs_pred, updrs_ci = get_prediction_with_ci(updrs_model, input_df)
tremor_pred, tremor_ci = get_prediction_with_ci(tremor_model, input_df)
diagnosis_pred = diagnosis_model.predict(input_df)[0]  # classification, no CI


# Calculate derived values
progression_score = np.clip((updrs_pred / 13.2), 0, 10)  # Scale 0-132 to 0-10
bradykinesia_index = input_df['std_hold_time'].values[0] / input_df['mean_hold_time'].values[0] * 100

# Create random key pressure heatmap data (simulate)
key_pressure = np.random.randint(1, 100, (5, 5))

# Weekly trend sparkline (simulate)
weekly_trend = np.random.uniform(3, 9, 7)

# Motor features for radar chart
motor_features = {
    'Hold Time Variability': input_df['hold_time_variability'].values[0],
    'Flight Time Variability': input_df['std_flight_time'].values[0],
    'Backspace Rate': input_df['backspace_rate'].values[0],
    'Pause Frequency': input_df['pause_frequency'].values[0],
    'Typing Speed': (input_df['typing_speed'].values[0]/350)  # Normalize for radar
}

# Medication response simulated data
med_on = np.random.uniform(0.1, 0.3, 5)
med_off = np.random.uniform(0.2, 0.4, 5)
med_metrics = ['Metric 1', 'Metric 2', 'Metric 3', 'Metric 4', 'Metric 5']

# Longitudinal timeline simulated data
timeline_x = np.arange(0, 12, 1)
timeline_y = np.random.normal(50, 5, len(timeline_x))

# Tabs
tab1, tab2, tab3 = st.tabs(["Patient", "Clinic", "SHAP Explanation"])


with tab1:
    st.header("Patient Dashboard")
    st.markdown(f"### Progression Score: {progression_score:.2f} / 10")
    st.caption(f"UPDRS CI: {updrs_ci[0]:.2f} - {updrs_ci[1]:.2f}")
    
    st.markdown("### Tremor Severity")
    st.progress(min(max(tremor_pred/2, 0), 1))  # Gauge as progress bar scaled 0-2
    st.caption(f"Tremor CI: {tremor_ci[0]:.2f} - {tremor_ci[1]:.2f}")

    st.markdown(f"### Bradykinesia Index: {bradykinesia_index:.1f}%")
    
    st.markdown("### Key Pressure Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(key_pressure, cmap="Reds", cbar=False, square=True, ax=ax)
    ax.axis('off')
    st.pyplot(fig)
    
    st.markdown("### Weekly Trend")
    fig2, ax2 = plt.subplots(figsize=(6,1))
    ax2.plot(weekly_trend, marker='o', color='blue')
    ax2.axis('off')
    st.pyplot(fig2)

    with st.expander("📂 Upload and Compare Multiple Sessions", expanded=False):
        uploaded_sessions = st.file_uploader("Upload CSV file with multiple session records", type=["csv"])

        if uploaded_sessions:
            try:
                multi_df = pd.read_csv(uploaded_sessions)

                required_columns = [
                "mean_hold_time", "std_hold_time", "mean_flight_time", "std_flight_time",
                "hold_time_variability", "backspace_rate", "pause_frequency", "session_duration",
                "total_keystrokes", "typing_speed", "medication_status"
                ]

                if not all(col in multi_df.columns for col in required_columns):
                        st.error("Uploaded file is missing required feature columns.")
                else:
                # Run predictions on each row
                    session_results = []
                    for i, row in multi_df.iterrows():
                        input_row = row[required_columns].to_frame().T
                        updrs_pred, updrs_ci = get_prediction_with_ci(updrs_model, input_row)
                        tremor_pred, tremor_ci = get_prediction_with_ci(tremor_model, input_row)
                        diagnosis_pred = diagnosis_model.predict(input_row)[0]

                        result = {
                        "Session #": i + 1,
                        "UPDRS": round(updrs_pred, 2),
                        "UPDRS CI": f"{updrs_ci[0]:.1f} - {updrs_ci[1]:.1f}",
                        "Tremor": round(tremor_pred, 2),
                        "Diagnosis": "Parkinson's" if diagnosis_pred == 1 else "Healthy",
                        "Typing Speed": round(row["typing_speed"], 2),
                        "Keystrokes": int(row["total_keystrokes"])
                        }
                        session_results.append(result)
 
                    result_df = pd.DataFrame(session_results)

                    st.success("✅ Prediction completed for all sessions.")
                    st.dataframe(result_df, use_container_width=True)

                # Progression chart
                    st.markdown("#### 📈 Progression Over Sessions")
                    st.line_chart(result_df.set_index("Session #")[["UPDRS", "Tremor"]])

                # Optional matplotlib chart
                    fig, ax = plt.subplots()
                    ax.plot(result_df["Session #"], result_df["UPDRS"], label="UPDRS", marker='o')
                    ax.plot(result_df["Session #"], result_df["Tremor"], label="Tremor", marker='x')
                    ax.set_title("Motor Feature Progression")
                    ax.set_xlabel("Session #")
                    ax.set_ylabel("Score")
                    ax.set_ylim(min(result_df["UPDRS"].min(), result_df["Tremor"].min()) - 1,
                            max(result_df["UPDRS"].max(), result_df["Tremor"].max()) + 1)
                    ax.legend()
                    st.pyplot(fig)

            except Exception as e:
                st.error(f"❌ Error processing file: {e}")



with tab2:
    st.header("Clinic Dashboard")
    
    st.markdown(f"### UPDRS-Aligned Progression: {updrs_pred:.1f} / 132")
    st.caption(f"UPDRS Confidence Interval: {updrs_ci[0]:.2f} - {updrs_ci[1]:.2f}")

    # Color coding based on progression severity
    if updrs_pred < 40:
        st.success("Low progression")
    elif updrs_pred < 80:
        st.warning("Moderate progression")
    else:
        st.error("High progression")
    
    st.markdown("### Motor Feature Breakdown (Radar Chart)")
    labels = list(motor_features.keys())
    stats = list(motor_features.values())
    
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    stats += stats[:1]
    angles += angles[:1]
    
    fig3, ax3 = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))
    ax3.fill(angles, stats, color='skyblue', alpha=0.4)
    ax3.plot(angles, stats, color='blue', linewidth=2)
    ax3.set_yticklabels([])
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(labels)
    st.pyplot(fig3)
    
    st.markdown("### Medication Response")
    fig4, ax4 = plt.subplots()
    x = np.arange(len(med_metrics))
    ax4.bar(x - 0.2, med_on, width=0.4, label='On')
    ax4.bar(x + 0.2, med_off, width=0.4, label='Off')
    ax4.set_xticks(x)
    ax4.set_xticklabels(med_metrics)
    ax4.legend()
    st.pyplot(fig4)

    st.markdown("### Risk Stratification")
    ax.set_ylim(33.3, 33.7)
    st.pyplot(fig)


with tab3:
    st.header("🧠 SHAP-Based Explanation (Why this prediction?)")

    shap_input = input_df.copy()
    explainer = shap.TreeExplainer(updrs_model)
    shap_values = explainer.shap_values(shap_input)
    expected_value = explainer.expected_value
    base_val = expected_value[0] if isinstance(expected_value, np.ndarray) else expected_value
    predicted_value = updrs_pred

    st.markdown(f"**Predicted UPDRS Score:** {predicted_value:.2f}")
    st.markdown(f"**SHAP Base Value (Expected):** {base_val:.2f}")

    shap_df = pd.DataFrame({
        'Feature': shap_input.columns,
        'SHAP Value': shap_values[0]
    }).sort_values(by='SHAP Value', key=abs, ascending=False)
    st.markdown("### Feature Impact (Top Contributors)")
    st.dataframe(shap_df.style.bar(subset=["SHAP Value"], color='#ff6961'))

    # Interpretation Summary Logic
    def interpret_feature(feat, val):
        if feat == 'hold_time_variability':
            if val > 0:
                return "High variability in hold time increases predicted UPDRS, indicating worse motor function."
            else:
                return "Low variability in hold time decreases predicted UPDRS, indicating better motor function."
        if feat == 'typing_speed':
            if val > 0:
                return "Higher typing speed increases predicted UPDRS, indicating worse motor function."
            else:
                return "Higher typing speed decreases predicted UPDRS, indicating better motor function."
        if feat == 'pause_frequency':
            if val > 0:
                return "Frequent pauses increase predicted UPDRS, indicating impaired motor control."
            else:
                return "Low pause frequency decreases predicted UPDRS, indicating better motor function."
        if feat == 'backspace_rate':
            if val > 0:
                return "High backspace rate increases predicted UPDRS, indicating more typing errors."
            else:
                return "Low backspace rate decreases predicted UPDRS, indicating better motor control."
        # Add more features if you want
        return None

    st.markdown("### Summary Interpretation")
    summaries = []
    for _, row in shap_df.head(3).iterrows():
        sentence = interpret_feature(row['Feature'], row['SHAP Value'])
        if sentence:
            summaries.append(sentence)
    if summaries:
        for s in summaries:
            st.markdown(f"- {s}")
    else:
        st.markdown("No clear feature impact detected.")

    # Optional: SHAP bar plot visualization here (same as before)
    fig, ax = plt.subplots()
    shap.plots.bar(shap.Explanation(values=shap_values[0], base_values=base_val,
                                    data=shap_input.values[0], feature_names=shap_input.columns.tolist()), max_display=10)
    st.pyplot(fig)




# Dynamic risk logic (you can tune thresholds as needed)
if updrs_pred >= 80 or bradykinesia_index > 50 or tremor_pred > 1.8:
    st.error("High decline risk next 30 days")
elif updrs_pred >= 40 or bradykinesia_index > 30:
    st.warning("Moderate risk of decline in next 30 days")
else:
    st.success("Low risk – stable condition")  # Example text
    
    st.markdown("### Longitudinal Analysis (Timeline)")
    fig5, ax5 = plt.subplots(figsize=(10, 2))
    ax5.plot(timeline_x, timeline_y, marker='o', linestyle='-')
    for i in range(0, len(timeline_x), 3):
        ax5.annotate('Note', (timeline_x[i], timeline_y[i]), textcoords="offset points", xytext=(0,10), ha='center')
    ax5.set_yticks([])
    st.pyplot(fig5)
