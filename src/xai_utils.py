#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.inspection import permutation_importance
import joblib

from src.data_prep import load_scenario_a, load_scenario_b
from lightgbm import LGBMClassifier

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"

RESULTS_DIR_A = BASE_DIR / "results" / "scenario_a"
RESULTS_DIR_B = BASE_DIR / "results" / "scenario_b"


# ---------- Scenario A: LinearSVM – permutation importance ----------

def run_xai_scenario_a():
    print("🔹 XAI for Scenario A (Lifestyle / LinearSVM)...")

    data = load_scenario_a()
    X_train, X_test = data.X_train, data.X_test
    y_train, y_test = data.y_train, data.y_test
    feature_names = data.feature_names

    # Load deployed best model
    model_path = MODELS_DIR / "best_scenario_a_linear_svm.pkl"
    model = joblib.load(model_path)
    print(f"Loaded model for XAI from: {model_path}")

    # Use a subset for speed
    rng = np.random.RandomState(42)
    idx = rng.choice(X_test.shape[0], size=min(5000, X_test.shape[0]), replace=False)
    X_sample = X_test[idx]
    y_sample = y_test[idx]

    print("Computing permutation importance...")
    r = permutation_importance(
        model,
        X_sample,
        y_sample,
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
        scoring="roc_auc",
    )

    importances = r["importances_mean"]
    stds = r["importances_std"]

    # Sort by importance
    order = np.argsort(importances)[::-1]
    importances = importances[order]
    stds = stds[order]
    names = np.array(feature_names)[order]

    RESULTS_DIR_A.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR_A / "scenario_a_perm_importance.png"

    plt.figure(figsize=(7, 5))
    y_pos = np.arange(len(names))
    plt.barh(y_pos, importances, xerr=stds, align="center")
    plt.yticks(y_pos, names)
    plt.xlabel("Mean decrease in ROC-AUC")
    plt.title("Scenario A – LinearSVM permutation importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out_path}")


# ---------- Scenario B: surrogate LightGBM SHAP ----------

def _fit_lgbm_for_xai(X_train, y_train):
    model = LGBMClassifier(
        n_estimators=400,
        max_depth=-1,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def _plot_shap_global(model, X, feature_names, out_prefix: Path, title_prefix: str):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):  # binary classification
        shap_values = shap_values[1]

    # Bar plot
    plt.figure()
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        max_display=len(feature_names),
    )
    plt.title(f"{title_prefix} – SHAP importance (bar, surrogate model)")
    bar_path = Path(str(out_prefix) + "_shap_bar.png")
    plt.tight_layout()
    plt.savefig(bar_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {bar_path}")

    # Beeswarm
    plt.figure()
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        show=False,
        max_display=len(feature_names),
    )
    plt.title(f"{title_prefix} – SHAP summary (beeswarm, surrogate model)")
    swarm_path = Path(str(out_prefix) + "_shap_beeswarm.png")
    plt.tight_layout()
    plt.savefig(swarm_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {swarm_path}")


def run_xai_scenario_b():
    print("🔹 XAI for Scenario B (Clinical / Stacking – surrogate LightGBM)...")

    data = load_scenario_b()
    X_train, X_test = data.X_train, data.X_test
    y_train = data.y_train
    feature_names = data.feature_names

    rng = np.random.RandomState(42)
    idx = rng.choice(X_test.shape[0], size=min(5000, X_test.shape[0]), replace=False)
    X_sample = X_test[idx]

    # Fit a separate LightGBM as a surrogate explainer
    model = _fit_lgbm_for_xai(X_train, y_train)

    RESULTS_DIR_B.mkdir(parents=True, exist_ok=True)
    out_prefix = RESULTS_DIR_B / "scenario_b"
    _plot_shap_global(
        model,
        X_sample,
        feature_names,
        out_prefix=out_prefix,
        title_prefix="Scenario B",
    )


# ---------- Main ----------

if __name__ == "__main__":
    RESULTS_DIR_A.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR_B.mkdir(parents=True, exist_ok=True)

    run_xai_scenario_a()
    run_xai_scenario_b()
