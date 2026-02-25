#!/usr/bin/env python3
#!/usr/bin/env python3
from pathlib import Path
import joblib

from src.data_prep import load_scenario_a, load_scenario_b
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


def train_and_save_scenario_a():
    data = load_scenario_a()
    X_train, y_train = data.X_train, data.y_train

    model = LinearSVC(
        max_iter=5000,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    out_path = MODELS_DIR / "best_scenario_a_linear_svm.pkl"
    joblib.dump(model, out_path)
    print("Saved:", out_path)


def train_and_save_scenario_b():
    data = load_scenario_b()
    X_train, y_train = data.X_train, data.y_train

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=10,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
    )

    xgb = XGBClassifier(
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

    base_estimators = [
        ("rf", rf),
        ("xgb", xgb),
    ]

    meta_clf = LogisticRegression(
        max_iter=500,
        n_jobs=-1,
        class_weight="balanced",
    )

    model = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_clf,
        n_jobs=-1,
        passthrough=False,
    )

    model.fit(X_train, y_train)
    out_path = MODELS_DIR / "best_scenario_b_stacking.pkl"
    joblib.dump(model, out_path)
    print("Saved:", out_path)


if __name__ == "__main__":
    train_and_save_scenario_a()
    train_and_save_scenario_b()
