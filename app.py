from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import json
from datetime import datetime
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load ML Model and Scaler at startup
with open('models/rf_pd_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Ensure logs folder exists
os.makedirs("logs", exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        if username:
            return redirect(url_for("test", username=username))
    return render_template("login.html")

@app.route("/test/<username>")
def test(username):
    return render_template("test.html", username=username)

@app.route("/submit", methods=["POST"])
def submit():
    data = request.get_json()
    username = data.get("username")

    if not username:
        return jsonify({"status": "error", "message": "Username missing"}), 400

    # Extract the 4 features
    try:
        nqScore = float(data.get("nqScore", 0))
        typing_speed = float(data.get("typing_speed", 0))
        afTap = float(data.get("afTap", 0))
        sTap = float(data.get("sTap", 0))
    except ValueError:
        return jsonify({"status": "error", "message": "Invalid feature values"}), 400

    # Prepare DataFrame for model
    input_df = pd.DataFrame([{
        'nqScore': nqScore,
        'Typing speed': typing_speed,
        'afTap': afTap,
        'sTap': sTap
    }])

    # Scale input
    scaled_input = scaler.transform(input_df)

    # Predict
    pred_prob = model.predict_proba(scaled_input)[0][1]  # PD probability
    pred_class = model.predict(scaled_input)[0]

    # Save raw data
    data["timestamp"] = datetime.now().isoformat()
    filepath = os.path.join("logs", f"{username}.json")
    with open(filepath, "a") as f:
        f.write(json.dumps(data) + "\n")

    # Send prediction result back
    return jsonify({
        "status": "success",
        "probability": round(pred_prob * 100, 2),
        "prediction": int(pred_class)
    })

if __name__ == "__main__":
    app.run(debug=True)
