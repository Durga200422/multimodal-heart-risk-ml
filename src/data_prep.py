from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@dataclass
class ScenarioData:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]


# ---------- Shared helper for Scenario A ----------

def encode_scenario_a_raw(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Take a dataframe with the raw BRFSS columns and return the numeric
    feature dataframe plus the feature_names list.
    This is the ONLY place where we do mappings/encodings for Scenario A.
    """
    feature_cols = [
        "BMI",
        "AgeCategory",
        "Sex",
        "SmokerStatus",
        "PhysicalActivities",
        "SleepHours",
        "HadDiabetes",
        "HighRiskLastYear",
    ]
    sub = df_raw[feature_cols].copy()

    # strip strings
    for col in sub.select_dtypes(include="object").columns:
        sub[col] = sub[col].astype(str).str.strip()

    # AgeCategory ordinal mapping
    if "AgeCategory" in sub.columns:
        age_order = {
            name: idx
            for idx, name in enumerate(sorted(sub["AgeCategory"].unique()))
        }
        sub["AgeCategory"] = sub["AgeCategory"].map(age_order)

    # Sex mapping
    if "Sex" in sub.columns:
        sub["Sex"] = sub["Sex"].map({"Male": 1, "Female": 0})

    # Yes/No style fields
    yes_no_cols = [
        "SmokerStatus",
        "PhysicalActivities",
        "HadDiabetes",
        "HighRiskLastYear",
    ]
    for col in yes_no_cols:
        if col in sub.columns:
            sub[col] = sub[col].map(
                {
                    "Yes": 1,
                    "No": 0,
                    "Former smoker": 1,   # treat as risk
                    "Never smoked": 0,
                    "Current smoker": 1,
                    "Never sm": 0,        # truncated category safeguard
                }
            )

    # Fill NaNs
    for col in sub.columns:
        if sub[col].dtype.kind in "if":
            sub[col] = sub[col].fillna(sub[col].median())
        else:
            sub[col] = sub[col].fillna(sub[col].mode().iloc[0])

    feature_names = list(sub.columns)
    return sub, feature_names


# ---------- Scenario A: BRFSS lifestyle ----------

def load_scenario_a(
    csv_name: str = "heart_2022_with_nans.csv",
    test_size: float = 0.2,
    random_state: int = 42,
) -> ScenarioData:
    """
    Loads and preprocesses the BRFSS-based heart indicators dataset.

    Target: HadHeartAttack (Yes/No -> 1/0)
    """
    path = DATA_DIR / csv_name
    df = pd.read_csv(path)

    # Target
    y = (df["HadHeartAttack"].astype(str).str.strip() == "Yes").astype(int)

    # Use shared encoding helper
    sub, feature_names = encode_scenario_a_raw(df)
    X = sub.values.astype(float)

    # Train/test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y.values, test_size=test_size, random_state=random_state, stratify=y
    )

    # Address class imbalance via SMOTE on training set only
    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Standardize numeric features
    scaler = StandardScaler()
    X_train_res = scaler.fit_transform(X_train_res)
    X_test = scaler.transform(X_test)

    return ScenarioData(
        X_train=X_train_res,
        X_test=X_test,
        y_train=y_train_res,
        y_test=y_test,
        feature_names=feature_names,
    )


# ---------- Scenario B: Cardio clinical ----------

def load_scenario_b(
    csv_name: str = "cardio_train.csv",
    test_size: float = 0.2,
    random_state: int = 42,
) -> ScenarioData:
    """
    Loads and preprocesses the Sulianova cardiovascular dataset.

    Target: cardio (0/1)
    """
    path = DATA_DIR / csv_name
    df = pd.read_csv(path, sep=";")

    # Target
    y = df["cardio"].astype(int)

    # Features
    feature_cols = [
        "age",
        "gender",
        "height",
        "weight",
        "ap_hi",
        "ap_lo",
        "cholesterol",
        "gluc",
        "smoke",
        "alco",
        "active",
    ]
    sub = df[feature_cols].copy()

    # age in years
    sub["age"] = (sub["age"] / 365.25).astype(float)

    # Fill NaNs
    for col in sub.columns:
        if sub[col].isna().any():
            sub[col] = sub[col].fillna(sub[col].median())

    X = sub.values.astype(float)
    feature_names = list(sub.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y.values, test_size=test_size, random_state=random_state, stratify=y
    )

    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_res = scaler.fit_transform(X_train_res)
    X_test = scaler.transform(X_test)

    return ScenarioData(
        X_train=X_train_res,
        X_test=X_test,
        y_train=y_train_res,
        y_test=y_test,
        feature_names=feature_names,
    )
