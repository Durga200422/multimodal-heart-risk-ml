import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import sys
from PIL import Image
import requests

# ---------- Paths ----------

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_A = BASE_DIR / "results" / "scenario_a"
RESULTS_B = BASE_DIR / "results" / "scenario_b"

# Ensure models directory exists (important for Streamlit Cloud)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Add src to path
sys.path.append(str(BASE_DIR / "src"))
from src.data_prep import encode_scenario_a_raw


# ---------- Google Drive Download ----------

def download_file_from_drive(file_id: str, dest_path: Path):
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={"id": file_id}, stream=True)
    response.raise_for_status()

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(8192):
            if chunk:
                f.write(chunk)


# ---------- Page config ----------

st.set_page_config(
    page_title="Multimodal Heart Risk Demo",
    page_icon="❤️",
    layout="wide",
)

st.title("Multimodal Heart Attack Risk Demo")

st.markdown(
    "This app predicts heart disease risk using:\n\n"
    "• **Scenario A:** Lifestyle survey (Linear SVM)\n"
    "• **Scenario B:** Clinical vitals (Stacking ensemble)\n\n"
    "It also shows global explanations (XAI)."
)

# ---------- Sidebar ----------

st.sidebar.markdown("### About")
st.sidebar.info(
    "Scenario A: Lifestyle model\n"
    "Scenario B: Clinical stacking model\n"
    "XAI tab shows global feature importance."
)

preset = st.sidebar.selectbox(
    "Load example profile",
    ["Custom", "Low risk", "Moderate risk", "High risk"],
)

tab_a, tab_b, tab_xai = st.tabs(
    ["Scenario A – Lifestyle", "Scenario B – Clinical", "Global explanations (XAI)"]
)

# ---------- Load Scenario A Model ----------

model_a = joblib.load(MODELS_DIR / "best_scenario_a_linear_svm.pkl")
scaler_a = joblib.load(MODELS_DIR / "scaler_scenario_a.pkl")

feat_a = [
    "BMI", "AgeCategory", "Sex", "SmokerStatus",
    "PhysicalActivities", "SleepHours",
    "HadDiabetes", "HighRiskLastYear"
]

# ---------- Load Scenario B Model (Auto Download) ----------

MODEL_B_PATH = MODELS_DIR / "best_scenario_b_stacking.pkl"
FILE_ID = "1ibg40pfwmeStcoLm1T7okPj7K5SWcaIk"

if not MODEL_B_PATH.exists():
    with st.spinner("Downloading Scenario B model from Google Drive..."):
        download_file_from_drive(FILE_ID, MODEL_B_PATH)

model_b = joblib.load(MODEL_B_PATH)
scaler_b = joblib.load(MODELS_DIR / "scaler_scenario_b.pkl")

feat_b = [
    "age", "gender", "height", "weight",
    "ap_hi", "ap_lo", "cholesterol", "gluc",
    "smoke", "alco", "active"
]

# ---------- Preprocessing ----------

def preprocess_single_a(inputs_dict):
    df_raw = pd.DataFrame([inputs_dict])
    sub, _ = encode_scenario_a_raw(df_raw)
    sub = sub[feat_a]
    X = sub.values.astype(float)
    return scaler_a.transform(X)

def preprocess_single_b(inputs_dict):
    df = pd.DataFrame([inputs_dict])
    X = df[feat_b].astype(float).values
    return scaler_b.transform(X)

# ---------- Scenario A ----------

with tab_a:
    st.subheader("Scenario A – Lifestyle screening")

    BMI = st.number_input("BMI", 10.0, 60.0, 25.0)
    age_cat = st.selectbox("Age category", [
        "Age 18 to 24","Age 25 to 29","Age 30 to 34","Age 35 to 39",
        "Age 40 to 44","Age 45 to 49","Age 50 to 54","Age 55 to 59",
        "Age 60 to 64","Age 65 to 69","Age 70 to 74",
        "Age 75 to 79","Age 80 or older"
    ])
    sex = st.selectbox("Sex", ["Female", "Male"])
    smoker_status = st.selectbox(
        "SmokerStatus",
        ["Never smoked", "Former smoker", "Current smoker"]
    )
    phys_act = st.selectbox("PhysicalActivities", ["No", "Yes"])
    sleep = st.number_input("SleepHours", 0.0, 14.0, 7.0)
    had_diab = st.selectbox("HadDiabetes", ["No", "Yes"])
    high_risk = st.selectbox("HighRiskLastYear", ["No", "Yes"])

    if st.button("Predict risk (Scenario A)"):
        inputs = {
            "BMI": BMI,
            "AgeCategory": age_cat,
            "Sex": sex,
            "SmokerStatus": smoker_status,
            "PhysicalActivities": phys_act,
            "SleepHours": sleep,
            "HadDiabetes": had_diab,
            "HighRiskLastYear": high_risk,
        }

        X = preprocess_single_a(inputs)
        score = float(model_a.decision_function(X)[0])
        proba = 1.0 / (1.0 + np.exp(-score))

        st.success(f"Estimated lifestyle-based risk: {proba*100:.1f}%")
        st.progress(min(max(proba, 0.01), 0.99))
        
                # ---------- Local Explanation Table (Scenario A) ----------

        ref = {
            "BMI": 22.0,
            "AgeCategory": "Age 30 to 34",
            "Sex": "Male",
            "SmokerStatus": "Never smoked",
            "PhysicalActivities": "Yes",
            "SleepHours": 7.0,
            "HadDiabetes": "No",
            "HighRiskLastYear": "No",
        }

        rows = []

        for k, v in inputs.items():
            rv = ref.get(k, None)
            if rv is None:
                continue

            if k == "BMI":
                diff = abs(float(v) - float(rv))
            elif k == "SleepHours":
                diff = abs(float(v) - float(rv))
            else:
                diff = 0.0 if str(v) == str(rv) else 1.0

            if k == "BMI":
                direction = "Higher" if v > rv else "Lower"
            elif k == "SleepHours":
                direction = "Higher" if v > rv else "Lower"
            elif k == "SmokerStatus":
                direction = "Higher" if v == "Current smoker" else "Lower"
            elif k == "PhysicalActivities":
                direction = "Lower" if v == "Yes" else "Higher"
            elif k == "HadDiabetes":
                direction = "Higher" if v == "Yes" else "Lower"
            elif k == "HighRiskLastYear":
                direction = "Higher" if v == "Yes" else "Lower"
            else:
                direction = "Neutral"

            if direction == "Higher" and k not in ["PhysicalActivities", "SleepHours"]:
                effect = "Likely increases risk"
            elif direction == "Lower" and k not in ["PhysicalActivities", "SleepHours"]:
                effect = "Likely decreases risk"
            elif k == "PhysicalActivities" and direction == "Lower":
                effect = "Regular activity lowers risk"
            elif k == "PhysicalActivities" and direction == "Higher":
                effect = "Low activity may raise risk"
            elif k == "SleepHours" and direction == "Higher":
                effect = "Long sleep, effect unclear"
            elif k == "SleepHours" and direction == "Lower":
                effect = "Short sleep may increase risk"
            else:
                effect = "Near reference"

            rows.append([k, v, rv, diff, effect])

        explanation_df_a = pd.DataFrame(
            rows,
            columns=["Feature", "Your value", "Reference", "Distance from healthy", "Effect"]
        ).sort_values(by="Distance from healthy", ascending=False)

        explanation_df_a = explanation_df_a.astype(str)
        st.markdown("### Individual Feature Explanation (Scenario A)")
        st.dataframe(explanation_df_a, width="stretch")

# ---------- Scenario B ----------

with tab_b:
    st.subheader("Scenario B – Clinical assessment")

    age_years = st.number_input("Age (years)", 18.0, 90.0, 50.0)
    height = st.number_input("Height (cm)", 120.0, 220.0, 170.0)
    weight = st.number_input("Weight (kg)", 35.0, 180.0, 75.0)
    gender = st.selectbox("Gender", ["Female (1)", "Male (2)"])
    ap_hi = st.number_input("Systolic BP", 80, 260, 130)
    ap_lo = st.number_input("Diastolic BP", 40, 150, 80)
    chol = st.selectbox("Cholesterol (1-3)", [1, 2, 3])
    gluc = st.selectbox("Glucose (1-3)", [1, 2, 3])
    smoke = st.selectbox("Smokes? (0/1)", [0, 1])
    alco = st.selectbox("Alcohol? (0/1)", [0, 1])
    active = st.selectbox("Physically active? (0/1)", [1, 0])

    if st.button("Predict risk (Scenario B)"):
        inputs = {
            "age": age_years * 365.25,
            "gender": 1 if "Female" in gender else 2,
            "height": height,
            "weight": weight,
            "ap_hi": int(ap_hi),
            "ap_lo": int(ap_lo),
            "cholesterol": int(chol),
            "gluc": int(gluc),
            "smoke": int(smoke),
            "alco": int(alco),
            "active": int(active),
        }

        X = preprocess_single_b(inputs)
        proba = float(model_b.predict_proba(X)[0, 1])

        st.success(f"Estimated clinical-based risk: {proba*100:.1f}%")
        st.progress(min(max(proba, 0.01), 0.99))
        
                # ---------- Local Explanation Table (Scenario B) ----------

        ref = {
            "age": 45 * 365.25,
            "height": 170,
            "weight": 70,
            "ap_hi": 120,
            "ap_lo": 80,
            "cholesterol": 1,
            "gluc": 1,
            "smoke": 0,
            "alco": 0,
            "active": 1,
        }

        rows = []

        for k, v in inputs.items():
            if k not in ref:
                continue

            rv = ref[k]

            if k == "age":
                diff = abs(v - rv) / 365.25
            elif k in ["weight", "ap_hi", "ap_lo", "cholesterol", "gluc"]:
                diff = abs(v - rv)
            elif k in ["smoke", "alco", "active"]:
                diff = 1.0 if v != rv else 0.0
            else:
                diff = 0.0

            if k == "age":
                direction = "Higher" if v > rv else "Lower"
            elif k in ["weight", "ap_hi", "ap_lo", "cholesterol", "gluc"]:
                direction = "Higher" if v > rv else "Lower"
            elif k in ["smoke", "alco"]:
                direction = "Higher" if v == 1 else "Lower"
            elif k == "active":
                direction = "Lower" if v == 1 else "Higher"
            else:
                direction = "Neutral"

            if direction == "Higher" and k != "active":
                effect = "Likely increases risk"
            elif direction == "Lower" and k != "active":
                effect = "Likely decreases risk"
            elif k == "active" and direction == "Lower":
                effect = "Regular activity lowers risk"
            elif k == "active" and direction == "Higher":
                effect = "Low activity may raise risk"
            else:
                effect = "Near reference"

            rows.append([k, v, rv, diff, effect])

        explanation_df_b = pd.DataFrame(
            rows,
            columns=["Feature", "Your value", "Reference", "Distance from healthy", "Effect"]
        ).sort_values(by="Distance from healthy", ascending=False)

        explanation_df_b = explanation_df_b.astype(str)

        st.markdown("### Individual Feature Explanation (Scenario B)")
        st.dataframe(explanation_df_b, width="stretch")

# ---------- XAI ----------

def load_image_safe(path: Path):
    if path.is_file():
        return Image.open(path)
    return None

with tab_xai:
    st.subheader("Global explanations")

    col1, col2 = st.columns(2)

    with col1:
        img = load_image_safe(RESULTS_A / "scenario_a_perm_importance.png")
        if img:
            st.image(img, caption="Scenario A – Permutation Importance")

    with col2:
        img1 = load_image_safe(RESULTS_B / "scenario_b_shap_bar.png")
        img2 = load_image_safe(RESULTS_B / "scenario_b_shap_beeswarm.png")

        if img1:
            st.image(img1, caption="Scenario B – SHAP Bar")
        if img2:
            st.image(img2, caption="Scenario B – SHAP Beeswarm")