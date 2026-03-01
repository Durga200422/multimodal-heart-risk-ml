# Multimodal Heart Attack Risk Prediction (Lifestyle + Clinical)

This repository contains an end‑to‑end machine learning project that estimates heart disease risk from two complementary data sources:
- **Scenario A – Lifestyle model** using survey‑style risk factors (BRFSS‑like data)
- **Scenario B – Clinical model** using structured clinical variables (blood pressure, cholesterol, etc.)

The project includes data preprocessing, model training, evaluation, global and local explainability, and a Streamlit web app for interactive inference.

## 🎯 Overview
Cardiovascular disease is a leading global cause of death. Early risk detection can be approached from:
- **Lifestyle information** (self‑reported survey data)
- **Clinical measurements** (vitals, labs, habits captured in the clinic)

This project implements a multimodal risk prediction demo with:

- Scenario A: **Lifestyle screening** (can be used in community or telehealth settings)
- Scenario B: **Clinical assessment** (for patients with vitals and basic labs available)

Both scenarios output:

- A **risk score** (probability of heart attack / cardiovascular disease)
- **Global explanations** (feature importance via permutation importance or SHAP)
- **Local explanations** (per‑patient feature explanation tables in the app)

**⚠️ Disclaimer:**
**This project is for research and educational purposes only. It is not a medical device and must not be used for clinical decision‑making.**

## 📂 Repository Structure
```
multimodal-heart-risk-ml/
├── data/
    ├── cardio_train.csv              # Clinical dataset (Kaggle "Cardiovascular Disease" data)
├── run.bat               # One-click Studio Launcher
├── src/
│   └── mixmaster/
│        ├── api/          # Endpoints & Pydantic Schemas
│        ├── core/         # The "Brain" (Audio & Video Processing)
│        ├── ui/           # Streamlit Dashboard Code
│        └── utils/        # Cleanup, Logging, & Global Settings
├── data/                 # Local Storage (Uploads/Exports)
└── tests/                # Pytest Suite for Logic Validation

```


