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
│   ├── cardio_train.csv              # Clinical dataset (Kaggle "Cardiovascular Disease" data)
│   └── heart_2022_with_nans.csv*     # BRFSS-style lifestyle dataset (large; may not be tracked)
│
├── models/
│   ├── best_scenario_a_linear_svm.pkl   # Deployed Scenario A model
│   ├── best_scenario_b_stacking.pkl*    # Deployed Scenario B model (downloaded at runtime)
│   ├── lgbm_scenario_a.pkl              # Auxiliary models used in experiments
│   ├── lgbm_scenario_b.pkl
│   ├── scaler_scenario_a.pkl            # Feature scaler for Scenario A
│   └── scaler_scenario_b.pkl            # Feature scaler for Scenario B
│
├── results/
│   ├── scenario_a/
│   │   ├── metrics_scenario_a.csv
│   │   ├── roc_curves_scenario_a.png
│   │   ├── metrics_bar_scenario_a.png
│   │   └── scenario_a_perm_importance.png
│   └── scenario_b/
│       ├── metrics_scenario_b.csv
│       ├── roc_curves_scenario_b.png
│       ├── metrics_bar_scenario_b.png
│       ├── scenario_b_shap_bar.png
│       └── scenario_b_shap_beeswarm.png
│
├── src/
│   ├── __init__.py
│   ├── data_prep.py              # Core preprocessing & train/test split for both scenarios
│   ├── data_prep_inference.py    # Preprocessing helpers for inference (scalers, features)
│   ├── eval_utils.py             # Metrics, ROC curves, confusion matrices, bar plots
│   ├── models_scenario_a.py      # Training/evaluation for Scenario A models
│   ├── models_scenario_b.py      # Training/evaluation for Scenario B models
│   ├── save_best_models.py       # Train & persist best models for deployment
│   └── xai_utils.py              # XAI scripts (permutation importance, SHAP)
│
├── app_streamlit.py              # Streamlit web app (multimodal demo)
├── save_scalers.py               # Script to fit & save scalers for inference
├── requirements.txt
└── README.md
```

Files marked with * may be too large for GitHub and are either:
- Kept locally (heart_2022_with_nans.csv)
- Downloaded at runtime (best_scenario_b_stacking.pkl)

## Data sources
Scenario A – Lifestyle dataset
- BRFSS‑style heart attack indicators with lifestyle variables:

    - BMI
    - Age category
    - Sex
    - Smoking status
    - Physical activity
    - Sleep hours
    - Diabetes status
    - High risk last year
    - Target: HadHeartAttack (Yes/No → 1/0)

This dataset is saved locally as data/heart_2022_with_nans.csv and is not committed to GitHub due to size. You can replace it with your own BRFSS‑style dataset if needed.

Scenario B – Clinical dataset
- Source: Kaggle “Cardiovascular Disease Dataset”.
​- File: data/cardio_train.csv (70k rows, 11 features + target)
- Key features:
    - Age (in days, converted to years)
    - Gender
    - Height, weight
    - Systolic and diastolic blood pressure (ap_hi, ap_lo)
    - Cholesterol (1–3)
    - Glucose (1–3)
    - Smoking, alcohol, physical activity
- Target: cardio (0/1)

## Environment setup
```
# Clone the repository
git clone https://github.com/<your-username>/multimodal-heart-risk-ml.git
cd multimodal-heart-risk-ml

# (Optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
Note: xgboost, lightgbm, and shap can be heavy—install may take a few minutes.

## Training and evaluation
All training scripts run from the project root.

**1. Scenario A – Lifestyle models**
```
python -m src.models_scenario_a

```
This script:

- Loads and preprocesses lifestyle data (load_scenario_a).
- Trains a range of models:
    - Logistic Regression
    - Decision Tree, Random Forest
    - Linear SVM
    - KNN
    - XGBoost, LightGBM
    - Stacking ensemble (RF + XGB + LGB, meta logistic regression)
    - MLP
- Evaluates each model with:
    - Accuracy, precision, recall, F1, ROC‑AUC
- Saves:
    - results/scenario_a/metrics_scenario_a.csv
    - results/scenario_a/roc_curves_scenario_a.png
    - results/scenario_a/metrics_bar_scenario_a.png
    - Confusion matrix PNG for the best ROC‑AUC model


**2. Scenario B – Clinical models**
```
python -m src.models_scenario_b

```
This script mirrors the Scenario A workflow for the clinical dataset and saves **metrics, ROC curves, bar plots, and a confusion matrix** to results/scenario_b/.

**3.Saving best models for deployment**
To retrain and persist the deployed models:
```
python -m src.save_best_models

```
This will:

- Train **LinearSVC** on Scenario A and save it as:
    - models/best_scenario_a_linear_svm.pkl

- Train a **stacking ensemble** (RandomForest + XGBoost → LogisticRegression meta) on Scenario B and save it as:
    - models/best_scenario_b_stacking.pkl (used in the app)

The Scenario B model file can be large. In the deployed app, it is hosted on Google Drive and downloaded on first run.

**4. Fitting and saving scalers**
For deployment, feature scalers are saved as separate objects:
```
python save_scalers.py

```
This produces:

- models/scaler_scenario_b.pkl (clinical features)

scaler_scenario_a.pkl can be generated similarly or directly persisted from the training pipeline.

## Explainability (XAI)
Global explanations are computed offline and rendered in the Streamlit app.

**1. Scenario A – Permutation importance (Linear SVM)**
```
python -m src.xai_utils run_xai_scenario_a

```
- Loads the deployed LinearSVM model for Scenario A.
- Computes **permutation importance** on a holdout subset using ROC‑AUC as the scoring metric.
- Saves:
    - results/scenario_a/scenario_a_perm_importance.png

**2. Scenario B – SHAP (surrogate LightGBM)**
```
python -m src.xai_utils run_xai_scenario_b

```
- Trains a **surrogate LightGBM** on Scenario B (separate from the deployed stacking model).
- Uses **SHAP (TreeExplainer)** for global explanations.
- ​Saves:
    - results/scenario_b/scenario_b_shap_bar.png
    - results/scenario_b/scenario_b_shap_beeswarm.png
These plots show which features most strongly drive the model’s predictions (e.g., systolic BP, age, cholesterol, etc.), and they align well with established cardiovascular risk factors.

## Streamlit web application
**A. Running locally**
After models and scalers are prepared:
```
streamlit run app_streamlit.py
```
The app provides three tabs:

**1. Scenario A – Lifestyle**
- Inputs: BMI, age category, sex, smoker status, physical activities, sleep hours, diabetes status, high risk last year.
- Uses the deployed Linear SVM and lifestyle scaler.
- Outputs:
    - Estimated lifestyle‑based risk (%)
    - Progress bar visualization
    - A **local explanation table** comparing user values against a “healthy reference” profile, with interpretations like “Likely increases risk” or “Regular activity lowers risk”.

**2. Scenario B – Clinical**
- Inputs: age, height, weight, gender, systolic/diastolic BP, cholesterol, glucose, smoking, alcohol, physical activity.
- Uses the stacking ensemble and clinical scaler.
- Outputs:
    - Estimated clinical‑based risk (%)
    - Progress bar
    - Local explanation table quantifying deviations from healthy reference levels.
 **3. Global explanations (XAI)**
- Displays:
    - Scenario A permutation importance plot
    - Scenario B SHAP bar and beeswarm plots
- These figures give a global view of which features matter most on average.

**B. Online deployment (Render / Streamlit Cloud)**
The app is designed to run on platforms like **Render** or **Streamlit Community Cloud:**
- Only small model artifacts are stored in the repo.
- A large Scenario B model (best_scenario_b_stacking.pkl) is downloaded on demand from Google Drive using requests.
- Build and start commands (Render):
```
# Build
pip install -r requirements.txt

# Start
streamlit run app_streamlit.py --server.port 10000 --server.address 0.0.0.0
```
When deploying, ensure that the data/ directory and required model/scaler files are available or that the app correctly downloads them.

## Methodology summary
**Preprocessing**
- **Scenario A**
      - Centralized encoding via encode_scenario_a_raw:
          - String cleaning
          - Ordinal mapping for age categories
          - Binary mapping for sex and Yes/No style features
          - Median/mode imputation
      - Train/test split with stratification
      - SMOTE on training data only (to handle class imbalance)
      - Standardization with StandardScaler

- **Scenario B**
      - Age converted from days to years
      - Median imputation for numeric features
      - Stratified train/test split
      - SMOTE on training data
      - Standardization with StandardScaler

Modeling & evaluation
- **Models:**
      - Logistic Regression, Decision Tree, Random Forest
      - Linear and RBF SVM
      - KNN
      - XGBoost, LightGBM
      - Stacking ensembles
      - MLP
- **Metrics:**
      - Accuracy, Precision, Recall, F1, ROC‑AUC
- **Visuals:**
      - ROC curves
      - Confusion matrices
      - Metric bar plots

Scenario A achieves ROC‑AUC around 0.77 with high recall but low precision (suitable for screening).
Scenario B achieves ROC‑AUC around 0.80 with balanced precision/recall, consistent with reported performance on this dataset.

## Limitations and future work
- Single train/test split; no external validation dataset.

- Scalers and preprocessing logic should ideally be wrapped into a single sklearn Pipeline and persisted to avoid any train–inference drift.

- Threshold selection and calibration (e.g., Platt scaling, isotonic regression) are not yet optimized for clinical workflows.

- Fairness / subgroup analysis (e.g., by age, sex) is not yet explored.

- Best Scenario B model is downloaded from Google Drive; a more robust hosting solution (e.g., object storage or model hub) is recommended for production‑grade deployments.

Planned improvements:

- Cross‑validated evaluation and hyperparameter tuning

- Explicit risk categories (low / moderate / high) based on ROC analysis

- Joint multimodal model combining lifestyle + clinical features

- Additional datasets and external validation

## How to cite / reference
If you reference this project in a portfolio, blog post, or report, you can describe it as:

“A multimodal machine learning demo for heart disease risk prediction, combining BRFSS‑style lifestyle survey data and the Kaggle cardiovascular disease dataset, with full training pipeline, explainability (permutation importance, SHAP), and an interactive Streamlit app.”





