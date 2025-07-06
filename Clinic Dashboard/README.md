
# Parkinson's Keystroke Analysis Dashboard

This is a Streamlit web application that analyzes keystroke dynamics to assist in the diagnosis and monitoring of Parkinson's Disease using machine learning models.

---

## 🧠 Features

- Predict UPDRS score
- Tremor assessment
- Diagnosis classification
- Visualizations with Seaborn and Matplotlib
- SHAP explanations (optional)

---

## 📁 Folder Structure

```
Clinic/
│
├── data/
│   ├── keystroke_features.csv
│   └── ...
│
├── model files/
│   ├── updrs_model.pkl
│   ├── tremor_model.pkl
│   ├── diagnosis_model.pkl
│   └── scaler.pkl
│
├── model training & validation/
│   ├── model_trainer.py
│   ├── train_ks_models.py
│   └── validation.py
│
└── main.py  ← Streamlit App
```

---

## 🚀 How to Run

### 1. Clone the repository or copy the folder
```bash
git clone https://github.com/your-repo/keystroke-parkinson-dashboard.git
cd Clinic
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

### 3. Install the required packages
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app
```bash
streamlit run main.py
```

---

## ✅ Notes
- Ensure all the `.pkl` model files are in the correct path as used in `main.py`.
- If any model or data file is missing, update the paths accordingly.

---

## 📧 Contact

If you encounter issues, contact the author or maintainer of this repo.
