#!/usr/bin/env python3
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.neural_network import MLPClassifier

from data_prep import load_scenario_b
from eval_utils import (
    compute_classification_metrics,
    save_metrics_table,
    plot_roc_curves,
    plot_confusion,
    plot_metric_bar,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "scenario_b"


def train_and_evaluate_scenario_b() -> None:
    print("🔹 Loading Scenario B (Clinical / Cardio) data...")
    data = load_scenario_b()

    X_train, X_test = data.X_train, data.X_test
    y_train, y_test = data.y_train, data.y_test

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    models: Dict[str, object] = {}

    # ------- Classical models -------
    models["LogisticRegression"] = LogisticRegression(
        max_iter=500,
        n_jobs=-1,
        class_weight="balanced",
    )

    models["DecisionTree"] = DecisionTreeClassifier(
        max_depth=None,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced",
    )

    models["RandomForest"] = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=10,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
    )

    # RBF SVM is feasible here (dataset ~60k)
    models["SVM_RBF"] = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True,
        class_weight="balanced",
        random_state=42,
    )

    models["KNN"] = KNeighborsClassifier(
        n_neighbors=15,
        n_jobs=-1,
    )

    # ------- Gradient Boosting models -------
    models["XGBoost"] = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
    )

    models["LightGBM"] = LGBMClassifier(
        n_estimators=400,
        max_depth=-1,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )

    # ------- Stacking ensemble (RF + XGB + LGB, meta LR) -------
    base_estimators = [
        ("rf", models["RandomForest"]),
        ("xgb", models["XGBoost"]),
        ("lgb", models["LightGBM"]),
    ]

    meta_clf = LogisticRegression(
        max_iter=500,
        n_jobs=-1,
        class_weight="balanced",
    )

    models["StackingEnsemble"] = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_clf,
        n_jobs=-1,
        passthrough=False,
    )
    
    models["MLP"] = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=128,
        learning_rate_init=1e-3,
        max_iter=50,
        random_state=42,
        verbose=False,
    )


    # ------- Fit and evaluate -------
    metrics_all: Dict[str, Dict[str, float]] = {}
    roc_curves = []

    for name, model in models.items():
        print(f"\n🚀 Training {name}...")
        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        metrics = compute_classification_metrics(y_test, y_pred, y_proba)

        metrics_all[name] = metrics
        roc_curves.append((name, y_proba, None))
        print(f"{name} metrics:", metrics)

    # ------- Save metrics table -------
    out_csv = RESULTS_DIR / "metrics_scenario_b.csv"
    df_metrics = save_metrics_table(metrics_all, str(out_csv))
    print(f"\n📄 Metrics saved to: {out_csv}")
    print(df_metrics.sort_values("roc_auc", ascending=False))

    # ------- ROC curves for all models -------
    colors = [f"C{i}" for i in range(len(roc_curves))]
    selected_curves = []
    for (name, y_score, _), color in zip(roc_curves, colors):
        selected_curves.append((name, y_score, color))

    roc_path = RESULTS_DIR / "roc_curves_scenario_b.png"
    plot_roc_curves(
        curves=selected_curves,
        y_true=y_test,
        out_path=str(roc_path),
        title="Scenario B – ROC curves",
    )
    print(f"📈 ROC curves saved to: {roc_path}")

    # ------- Metric bar plot (ROC-AUC + F1) -------
    bar_path = RESULTS_DIR / "metrics_bar_scenario_b.png"
    plot_metric_bar(
        df_metrics,
        out_path=str(bar_path),
        metric="roc_auc",
        secondary="f1",
        title="Scenario B – Model performance (ROC-AUC & F1)",
    )
    print(f"📊 Metric bar plot saved to: {bar_path}")

    # ------- Confusion matrix for best model -------
    best_model_name = max(metrics_all, key=lambda m: metrics_all[m].get("roc_auc", 0.0))
    print(f"\n🏆 Best model by ROC-AUC: {best_model_name}")
    best_model = models[best_model_name]

    y_scores_best = best_model.predict_proba(X_test)[:, 1]
    y_pred_best = (y_scores_best >= 0.5).astype(int)

    cm_path = RESULTS_DIR / f"confusion_{best_model_name}.png"
    title = f"Scenario B – Confusion ({best_model_name})"
    plot_confusion(
        y_true=y_test,
        y_pred=y_pred_best,
        out_path=str(cm_path),
        title=title,
    )
    print(f"🧩 Confusion matrix saved to: {cm_path}")


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train_and_evaluate_scenario_b()
