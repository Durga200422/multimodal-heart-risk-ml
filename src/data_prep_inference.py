from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data_prep import encode_scenario_a_raw  # <-- new import

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@dataclass
class InferencePrep:
    scaler: StandardScaler
    feature_names: list[str]


def prepare_scenario_a_scaler(
    csv_name: str = "heart_2022_with_nans.csv",
) -> InferencePrep:
    path = DATA_DIR / csv_name
    df = pd.read_csv(path)

    # use the SAME encoding as training
    sub, feature_names = encode_scenario_a_raw(df)
    X = sub.values.astype(float)

    scaler = StandardScaler()
    scaler.fit(X)  # only fit, no SMOTE

    return InferencePrep(scaler=scaler, feature_names=feature_names)


def prepare_scenario_b_scaler(
    csv_name: str = "cardio_train.csv",
) -> InferencePrep:
    path = DATA_DIR / csv_name
    df = pd.read_csv(path, sep=";")

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
    sub["age"] = (sub["age"] / 365.25).astype(float)

    for col in sub.columns:
        if sub[col].isna().any():
            sub[col] = sub[col].fillna(sub[col].median())

    X = sub.values.astype(float)

    scaler = StandardScaler()
    scaler.fit(X)

    return InferencePrep(scaler=scaler, feature_names=feature_cols)
