import streamlit as st
import numpy as np
import pickle
from pathlib import Path
from PIL import Image

st.set_page_config(page_title="Diabetes Prediction", layout="centered")

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "diabetes_model.sav"
SCALER_PATH = BASE_DIR / "scaler.sav"
FLOWCHART_PATH = BASE_DIR / "assets" / "A_flowchart_diagram_illustrates_a_Diabetes_Predict.png"

@st.cache_resource
def load_model_and_scaler():
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        raise FileNotFoundError("Model or scaler not found. Run 'train_and_save_model.py' to generate them.")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model_and_scaler()

st.title("🩺 Diabetes Prediction")
st.write("Enter patient parameters to predict diabetes likelihood.")

if FLOWCHART_PATH.exists():
    st.image(str(FLOWCHART_PATH), caption="Workflow: Dataset → Model → App", use_column_width=True)

Pregnancies = st.number_input('Pregnancies', min_value=0, value=0)
Glucose = st.number_input('Glucose', min_value=0, value=120)
BloodPressure = st.number_input('Blood Pressure', min_value=0, value=70)
SkinThickness = st.number_input('Skin Thickness', min_value=0, value=20)
Insulin = st.number_input('Insulin', min_value=0, value=79)
BMI = st.number_input('BMI', min_value=0.0, value=32.0)
DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', min_value=0.0, value=0.5)
Age = st.number_input('Age', min_value=1, value=30)

if st.button('Predict'):
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                            Insulin, BMI, DiabetesPedigreeFunction, Age]])
    input_scaled = scaler.transform(input_data)
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]
    if pred == 1:
        st.error(f"⚠️ Likely to have diabetes (probability: {prob:.2f})")
    else:
        st.success(f"✅ Not likely to have diabetes (probability: {prob:.2f})")
