import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import sys
from PIL import Image

# ---------- Paths & imports ----------

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "src"))

from data_prep_inference import (
    prepare_scenario_a_scaler,
    prepare_scenario_b_scaler,
)
from data_prep import encode_scenario_a_raw  # shared encoder

MODELS_DIR = BASE_DIR / "models"
RESULTS_A = BASE_DIR / "results" / "scenario_a"
RESULTS_B = BASE_DIR / "results" / "scenario_b"


def load_image_safe(path: Path):
    if path.is_file():
        return Image.open(path)
    return None


# ---------- Page config ----------

st.set_page_config(
    page_title="Multimodal Heart Risk Demo",
    page_icon="❤️",
    layout="wide",
)

st.title("Multimodal Heart Attack Risk Demo")

st.markdown(
    "This app uses trained machine learning models on **lifestyle survey data** "
    "(Scenario A) and **clinical vitals** (Scenario B) to estimate heart disease risk, "
    "and also provides **global explanations (XAI)** for both models."
)

# ---------- Sidebar ----------

st.sidebar.markdown("### About")
st.sidebar.info(
    "Scenario A uses lifestyle survey data with a Linear SVM model.\n\n"
    "Scenario B uses clinical measurements with a stacking ensemble model.\n\n"
    "The XAI tab shows which features are most important globally."
)

preset = st.sidebar.selectbox(
    "Load example profile",
    ["Custom", "Low risk", "Moderate risk", "High risk"],
)

# Tabs: A prediction, B prediction, global XAI
tab_a, tab_b, tab_xai = st.tabs(
    ["Scenario A – Lifestyle", "Scenario B – Clinical", "Global explanations (XAI)"]
)

# ---------- Load models & scalers ----------

model_a = joblib.load(MODELS_DIR / "best_scenario_a_linear_svm.pkl")

try:
    # On Streamlit Cloud this file may be missing (too large for GitHub)
    model_b = joblib.load(MODELS_DIR / "best_scenario_b_stacking.pkl")
    MODEL_B_AVAILABLE = True
except FileNotFoundError:
    model_b = None
    MODEL_B_AVAILABLE = False

prep_a = prepare_scenario_a_scaler()
prep_b = prepare_scenario_b_scaler()
feat_a = prep_a.feature_names
feat_b = prep_b.feature_names
scaler_a = prep_a.scaler
scaler_b = prep_b.scaler


# ---------- Preprocessing helpers ----------

def preprocess_single_a(inputs_dict: dict) -> np.ndarray:
    df_raw = pd.DataFrame([inputs_dict])
    sub, _ = encode_scenario_a_raw(df_raw)
    sub = sub[feat_a]
    X = sub.values.astype(float)
    X_scaled = scaler_a.transform(X)
    return X_scaled


def preprocess_single_b(inputs_dict: dict) -> np.ndarray:
    df = pd.DataFrame([inputs_dict])
    X = df[feat_b].astype(float).values
    X_scaled = scaler_b.transform(X)
    return X_scaled


# ---------- Local explanation helpers ----------

def format_feature_effects_scenario_a(inputs: dict) -> pd.DataFrame:
    """
    Simple explanation for lifestyle inputs: compare to a healthy reference
    and sort by how far each feature is from that reference.
    """
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

        rows.append(
            {
                "Feature": str(k),
                "Your value": str(v),
                "Reference": str(rv),
                "Effect": str(effect),
                "Strength": float(diff),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Strength", ascending=False).drop(columns=["Strength"])
        df = df.astype(str)  # force everything to string for Arrow safety
    return df


def format_feature_effects_scenario_b(inputs: dict) -> pd.DataFrame:
    """
    Explanation for clinical inputs: numeric distance from a healthy
    reference, sorted by strength.
    """
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
        elif k in ["weight", "ap_hi", "ap_lo"]:
            diff = abs(v - rv)
        elif k in ["cholesterol", "gluc"]:
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

        rows.append(
            {
                "Feature": k,
                "Your value": round(float(v), 2),
                "Reference": round(float(rv), 2),
                "Effect": effect,
                "Strength": diff,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Strength", ascending=False).drop(columns=["Strength"])
    return df


# ---------- Scenario A – Lifestyle tab ----------

with tab_a:
    st.subheader("Scenario A – Lifestyle screening")

    age_options = [
        "Age 18 to 24",
        "Age 25 to 29",
        "Age 30 to 34",
        "Age 35 to 39",
        "Age 40 to 44",
        "Age 45 to 49",
        "Age 50 to 54",
        "Age 55 to 59",
        "Age 60 to 64",
        "Age 65 to 69",
        "Age 70 to 74",
        "Age 75 to 79",
        "Age 80 or older",
    ]

    if preset == "Low risk":
        default_bmi = 22.0
        default_age = "Age 30 to 34"
        default_sex = "Male"
        default_smoke = "Never smoked"
        default_phys = "Yes"
        default_sleep = 7.5
        default_diab = "No"
        default_highrisk = "No"
    elif preset == "High risk":
        default_bmi = 32.0
        default_age = "Age 65 to 69"
        default_sex = "Male"
        default_smoke = "Current smoker"
        default_phys = "No"
        default_sleep = 5.0
        default_diab = "Yes"
        default_highrisk = "Yes"
    else:  # Custom / Moderate
        default_bmi = 27.0
        default_age = "Age 50 to 54"
        default_sex = "Male"
        default_smoke = "Former smoker"
        default_phys = "Yes"
        default_sleep = 7.0
        default_diab = "No"
        default_highrisk = "No"

    col1, col2 = st.columns(2)
    with col1:
        BMI = st.number_input("BMI", 10.0, 60.0, default_bmi, step=0.1)
        age_cat = st.selectbox("Age category", age_options, index=age_options.index(default_age))
        sex = st.selectbox("Sex", ["Female", "Male"], index=1 if default_sex == "Male" else 0)
        smoker_status = st.selectbox(
            "SmokerStatus",
            ["Never smoked", "Former smoker", "Current smoker"],
            index=["Never smoked", "Former smoker", "Current smoker"].index(default_smoke),
        )
    with col2:
        phys_act = st.selectbox(
            "PhysicalActivities", ["No", "Yes"], index=1 if default_phys == "Yes" else 0
        )
        sleep = st.number_input("SleepHours", 0.0, 14.0, default_sleep, step=0.5)
        had_diab = st.selectbox("HadDiabetes", ["No", "Yes"], index=1 if default_diab == "Yes" else 0)
        high_risk = st.selectbox(
            "HighRiskLastYear", ["No", "Yes"], index=1 if default_highrisk == "Yes" else 0
        )

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

        st.success(f"Estimated lifestyle-based risk: {proba * 100:.1f}%")
        st.progress(min(max(proba, 0.01), 0.99))

        with st.expander("See how your lifestyle values affect risk"):
            df_explain_a = format_feature_effects_scenario_a(inputs)
            st.table(df_explain_a)


# ---------- Scenario B – Clinical tab ----------

with tab_b:
    st.subheader("Scenario B – Clinical assessment")

    if preset == "Low risk":
        default_age = 35.0
        default_height = 170.0
        default_weight = 65.0
        default_ap_hi = 115
        default_ap_lo = 75
        default_chol = 1
        default_gluc = 1
        default_smoke = 0
        default_alco = 0
        default_active = 1
    elif preset == "High risk":
        default_age = 70.0
        default_height = 165.0
        default_weight = 90.0
        default_ap_hi = 150
        default_ap_lo = 95
        default_chol = 3
        default_gluc = 3
        default_smoke = 1
        default_alco = 1
        default_active = 0
    else:  # Custom / Moderate
        default_age = 50.0
        default_height = 170.0
        default_weight = 75.0
        default_ap_hi = 130
        default_ap_lo = 80
        default_chol = 2
        default_gluc = 2
        default_smoke = 0
        default_alco = 0
        default_active = 1

    col1, col2 = st.columns(2)
    with col1:
        age_years = st.number_input("Age (years)", 18.0, 90.0, default_age)
        height = st.number_input("Height (cm)", 120.0, 220.0, default_height)
        weight = st.number_input("Weight (kg)", 35.0, 180.0, default_weight)
        gender = st.selectbox("Gender", ["Female (1)", "Male (2)"])
    with col2:
        ap_hi = st.number_input("Systolic BP (ap_hi)", 80, 260, default_ap_hi)
        ap_lo = st.number_input("Diastolic BP (ap_lo)", 40, 150, default_ap_lo)
        chol = st.selectbox("Cholesterol (1-3)", [1, 2, 3], index=default_chol - 1)
        gluc = st.selectbox("Glucose (1-3)", [1, 2, 3], index=default_gluc - 1)
        smoke = st.selectbox("Smokes? (0/1)", [0, 1], index=default_smoke)
        alco = st.selectbox("Alcohol? (0/1)", [0, 1], index=default_alco)
        active = st.selectbox(
            "Physically active? (0/1)",
            [1, 0],
            index=0 if default_active == 1 else 1,
        )

    if st.button("Predict risk (Scenario B)"):
        if not MODEL_B_AVAILABLE:
            st.error(
                "The clinical stacking model file is not available on this online demo "
                "(too large for GitHub). Run the project locally with the full "
                "`models` folder to enable Scenario B predictions."
            )
        else:
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

            st.success(f"Estimated clinical-based risk: {proba * 100:.1f}%")
            st.progress(min(max(proba, 0.01), 0.99))

            with st.expander("See how your clinical values affect risk"):
                df_explain_b = format_feature_effects_scenario_b(inputs)
                st.table(df_explain_b)


# ---------- Global explanations tab ----------

with tab_xai:
    st.subheader("Global explanations for both models")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Scenario A – Lifestyle (LinearSVM)")
        perm_path = RESULTS_A / "scenario_a_perm_importance.png"
        perm_img = load_image_safe(perm_path)
        if perm_img:
            st.image(perm_img, caption="LinearSVM permutation importance (Scenario A)")
        else:
            st.warning(
                "Permutation-importance image not found for Scenario A.\n\n"
                "Run `python -m src.xai_utils` to generate it."
            )

    with col_b:
        st.markdown("#### Scenario B – Clinical (surrogate LightGBM SHAP)")
        bar_path_b = RESULTS_B / "scenario_b_shap_bar.png"
        swarm_path_b = RESULTS_B / "scenario_b_shap_beeswarm.png"
        bar_img_b = load_image_safe(bar_path_b)
        swarm_img_b = load_image_safe(swarm_path_b)

        if bar_img_b or swarm_img_b:
            if bar_img_b:
                st.image(
                    bar_img_b,
                    caption="SHAP feature importance (bar, surrogate model)",
                )
            if swarm_img_b:
                st.image(
                    swarm_img_b,
                    caption="SHAP summary (beeswarm, surrogate model)",
                )
        else:
            st.warning(
                "SHAP images not found for Scenario B.\n\n"
                "Run `python -m src.xai_utils` to generate them."
            )
