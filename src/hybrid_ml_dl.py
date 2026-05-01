#!/usr/bin/env python3
"""
Hybrid ML+DL Model: Two-stage architecture combining traditional ML with Deep Learning

Stage 1: Train multiple ML models (RF, XGB, LGB, SVM/LinearSVC) to generate probability predictions
Stage 2: Feed ML predictions + original features to a Deep Neural Network for final classification

This approach leverages the strengths of both paradigms:
- ML models capture different patterns and provide diverse predictions
- DL meta-learner learns complex non-linear combinations of ML predictions and features
"""

from pathlib import Path
from typing import Dict, Tuple
import warnings

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.data_prep import load_scenario_a, load_scenario_b
from src.eval_utils import (
    compute_classification_metrics,
    save_metrics_table,
    plot_roc_curves,
    plot_confusion,
)

warnings.filterwarnings('ignore')

RESULTS_DIR_A = Path(__file__).resolve().parents[1] / "results" / "scenario_a"
RESULTS_DIR_B = Path(__file__).resolve().parents[1] / "results" / "scenario_b"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


class HybridMLDL:
    """
    Two-stage hybrid model combining ML and DL.
    
    Architecture:
    1. Base ML models produce probability predictions
    2. DNN meta-learner takes [original_features + ML_predictions] as input
    """
    
    def __init__(self, scenario: str = "a", verbose: int = 0):
        self.scenario = scenario
        self.verbose = verbose
        self.base_models = {}
        self.meta_model = None
        self.feature_names = []
        
    def _build_base_models(self) -> Dict:
        """Build base ML models tailored to each scenario."""
        
        if self.scenario == "a":
            # Scenario A: Lifestyle data - simpler models
            models = {
                "rf": RandomForestClassifier(
                    n_estimators=200,
                    max_depth=None,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    n_jobs=-1,
                    random_state=42,
                    class_weight="balanced",
                ),
                "xgb": XGBClassifier(
                    n_estimators=300,
                    max_depth=5,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    n_jobs=-1,
                    random_state=42,
                ),
                "lgb": LGBMClassifier(
                    n_estimators=300,
                    max_depth=-1,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=42,
                    verbose=-1,
                ),
                "svm": LinearSVC(
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=42,
                ),
            }
        else:  # scenario == "b"
            # Scenario B: Clinical data - more complex models
            models = {
                "rf": RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    n_jobs=-1,
                    random_state=42,
                    class_weight="balanced",
                ),
                "xgb": XGBClassifier(
                    n_estimators=400,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    n_jobs=-1,
                    random_state=42,
                ),
                "lgb": LGBMClassifier(
                    n_estimators=400,
                    max_depth=-1,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=42,
                    verbose=-1,
                ),
                "svm": SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    probability=True,
                    class_weight="balanced",
                    random_state=42,
                ),
            }
        
        return models
    
    def _build_dnn_meta_learner(self, input_dim: int) -> keras.Model:
        """
        Build Deep Neural Network meta-learner.
        
        Architecture:
        - Input: [original_features + base_model_predictions]
        - 3 hidden layers with batch normalization and dropout
        - Output: Binary classification
        """
        
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(input_dim,)),
            
            # First hidden layer
            layers.Dense(
                128,
                activation='relu',
                kernel_regularizer=regularizers.l2(0.001)
            ),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Second hidden layer
            layers.Dense(
                64,
                activation='relu',
                kernel_regularizer=regularizers.l2(0.001)
            ),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Third hidden layer
            layers.Dense(
                32,
                activation='relu',
                kernel_regularizer=regularizers.l2(0.001)
            ),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile with Adam optimizer
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        return model
    
    def _get_base_predictions(
        self,
        X: np.ndarray,
        fit: bool = False,
        y: np.ndarray = None
    ) -> np.ndarray:
        """
        Get probability predictions from all base ML models.
        
        Args:
            X: Input features
            fit: Whether to fit the models (training phase)
            y: Target labels (required if fit=True)
            
        Returns:
            Array of shape (n_samples, n_base_models) with probability predictions
        """
        predictions = []
        
        for name, model in self.base_models.items():
            if fit:
                if self.verbose > 0:
                    print(f"  Training base model: {name}")
                model.fit(X, y)
            
            # Get probability predictions
            if name == "svm" and isinstance(model, LinearSVC):
                # LinearSVC doesn't have predict_proba, use decision_function
                scores = model.decision_function(X)
                # Normalize to [0, 1] range
                proba = 1 / (1 + np.exp(-scores))
            else:
                proba = model.predict_proba(X)[:, 1]
            
            predictions.append(proba)
        
        return np.column_stack(predictions)
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = 100,
        batch_size: int = 128,
    ):
        """
        Train the hybrid ML+DL model.
        
        Stage 1: Train base ML models
        Stage 2: Train DNN meta-learner on [features + ML predictions]
        """
        
        if self.verbose > 0:
            print(f"\n🔷 Training Hybrid ML+DL for Scenario {self.scenario.upper()}")
            print(f"Stage 1: Training {len(self.base_models)} base ML models...")
        
        # Stage 1: Train base ML models and get predictions
        self.base_models = self._build_base_models()
        base_preds_train = self._get_base_predictions(X_train, fit=True, y=y_train)
        
        # Combine original features with base model predictions
        X_meta_train = np.hstack([X_train, base_preds_train])
        
        if self.verbose > 0:
            print(f"Stage 2: Training DNN meta-learner...")
            print(f"  Meta-learner input shape: {X_meta_train.shape}")
        
        # Stage 2: Build and train DNN meta-learner
        input_dim = X_meta_train.shape[1]
        self.meta_model = self._build_dnn_meta_learner(input_dim)
        
        # Prepare validation data if provided
        validation_data = None
        if X_val is not None and y_val is not None:
            base_preds_val = self._get_base_predictions(X_val, fit=False)
            X_meta_val = np.hstack([X_val, base_preds_val])
            validation_data = (X_meta_val, y_val)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=15,
                restore_best_weights=True,
                verbose=self.verbose
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=self.verbose
            )
        ]
        
        # Train DNN meta-learner
        history = self.meta_model.fit(
            X_meta_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=self.verbose
        )
        
        if self.verbose > 0:
            print("✅ Hybrid ML+DL training complete!")
        
        return history
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions from the hybrid model."""
        # Get base model predictions
        base_preds = self._get_base_predictions(X, fit=False)
        
        # Combine with original features
        X_meta = np.hstack([X, base_preds])
        
        # Get DNN predictions
        proba = self.meta_model.predict(X_meta, verbose=0)
        
        return proba.flatten()
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Get binary predictions from the hybrid model."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def save(self, path: Path):
        """Save the hybrid model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save base models
        base_models_path = path.parent / f"{path.stem}_base_models.pkl"
        joblib.dump(self.base_models, base_models_path)
        
        # Save DNN meta-learner
        meta_model_path = path.parent / f"{path.stem}_meta_dnn.h5"
        self.meta_model.save(meta_model_path)
        
        # Save metadata
        metadata = {
            'scenario': self.scenario,
            'base_models_path': str(base_models_path),
            'meta_model_path': str(meta_model_path),
        }
        joblib.dump(metadata, path)
        
        if self.verbose > 0:
            print(f"💾 Hybrid model saved to: {path}")
    
    @classmethod
    def load(cls, path: Path, verbose: int = 0):
        """Load a saved hybrid model."""
        metadata = joblib.load(path)
        
        model = cls(scenario=metadata['scenario'], verbose=verbose)
        model.base_models = joblib.load(metadata['base_models_path'])
        model.meta_model = keras.models.load_model(metadata['meta_model_path'])
        
        if verbose > 0:
            print(f"📂 Hybrid model loaded from: {path}")
        
        return model


def train_and_evaluate_hybrid_scenario_a():
    """Train and evaluate Hybrid ML+DL for Scenario A."""
    print("\n" + "="*70)
    print("HYBRID ML+DL - SCENARIO A (Lifestyle)")
    print("="*70)
    
    # Load data
    data = load_scenario_a()
    X_train, X_test = data.X_train, data.X_test
    y_train, y_test = data.y_train, data.y_test
    
    # Create validation split from training data
    val_size = int(0.15 * len(X_train))
    X_val = X_train[:val_size]
    y_val = y_train[:val_size]
    X_train_sub = X_train[val_size:]
    y_train_sub = y_train[val_size:]
    
    # Initialize and train hybrid model
    hybrid = HybridMLDL(scenario="a", verbose=1)
    hybrid.fit(
        X_train_sub,
        y_train_sub,
        X_val=X_val,
        y_val=y_val,
        epochs=100,
        batch_size=256,
    )
    
    # Evaluate
    y_proba = hybrid.predict_proba(X_test)
    y_pred = (y_proba >= 0.5).astype(int)
    
    metrics = compute_classification_metrics(y_test, y_pred, y_proba)
    
    print("\n📊 Hybrid ML+DL Metrics (Scenario A):")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save model
    RESULTS_DIR_A.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "hybrid_ml_dl_scenario_a.pkl"
    hybrid.save(model_path)
    
    # Save metrics
    metrics_dict = {"HybridMLDL": metrics}
    metrics_csv = RESULTS_DIR_A / "hybrid_ml_dl_metrics.csv"
    save_metrics_table(metrics_dict, str(metrics_csv))
    
    # Plot confusion matrix
    cm_path = RESULTS_DIR_A / "confusion_HybridMLDL.png"
    plot_confusion(
        y_true=y_test,
        y_pred=y_pred,
        out_path=str(cm_path),
        title="Scenario A - Hybrid ML+DL Confusion Matrix",
    )
    
    return hybrid, metrics


def train_and_evaluate_hybrid_scenario_b():
    """Train and evaluate Hybrid ML+DL for Scenario B."""
    print("\n" + "="*70)
    print("HYBRID ML+DL - SCENARIO B (Clinical)")
    print("="*70)
    
    # Load data
    data = load_scenario_b()
    X_train, X_test = data.X_train, data.X_test
    y_train, y_test = data.y_train, data.y_test
    
    # Create validation split from training data
    val_size = int(0.15 * len(X_train))
    X_val = X_train[:val_size]
    y_val = y_train[:val_size]
    X_train_sub = X_train[val_size:]
    y_train_sub = y_train[val_size:]
    
    # Initialize and train hybrid model
    hybrid = HybridMLDL(scenario="b", verbose=1)
    hybrid.fit(
        X_train_sub,
        y_train_sub,
        X_val=X_val,
        y_val=y_val,
        epochs=100,
        batch_size=128,
    )
    
    # Evaluate
    y_proba = hybrid.predict_proba(X_test)
    y_pred = (y_proba >= 0.5).astype(int)
    
    metrics = compute_classification_metrics(y_test, y_pred, y_proba)
    
    print("\n📊 Hybrid ML+DL Metrics (Scenario B):")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save model
    RESULTS_DIR_B.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "hybrid_ml_dl_scenario_b.pkl"
    hybrid.save(model_path)
    
    # Save metrics
    metrics_dict = {"HybridMLDL": metrics}
    metrics_csv = RESULTS_DIR_B / "hybrid_ml_dl_metrics.csv"
    save_metrics_table(metrics_dict, str(metrics_csv))
    
    # Plot confusion matrix
    cm_path = RESULTS_DIR_B / "confusion_HybridMLDL.png"
    plot_confusion(
        y_true=y_test,
        y_pred=y_pred,
        out_path=str(cm_path),
        title="Scenario B - Hybrid ML+DL Confusion Matrix",
    )
    
    return hybrid, metrics


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Ensure directories exist
    RESULTS_DIR_A.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR_B.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Train both scenarios
    hybrid_a, metrics_a = train_and_evaluate_hybrid_scenario_a()
    hybrid_b, metrics_b = train_and_evaluate_hybrid_scenario_b()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Scenario A ROC-AUC: {metrics_a['roc_auc']:.4f}")
    print(f"Scenario B ROC-AUC: {metrics_b['roc_auc']:.4f}")