import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    RocCurveDisplay,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> Dict[str, float]:
    """Compute standard binary classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    return metrics


def save_metrics_table(
    all_metrics: Dict[str, Dict[str, float]],
    out_csv: str,
) -> pd.DataFrame:
    """
    all_metrics: {model_name: {metric_name: value}}
    """
    df = pd.DataFrame(all_metrics).T  # models as rows
    df.index.name = "model"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=True)
    return df


def plot_roc_curves(
    curves: List[Tuple[str, np.ndarray, str | None]],
    y_true: np.ndarray,
    out_path: str,
    title: str = "ROC curves",
) -> None:
    """
    curves: list of (label, y_score, color) where y_score is proba/score for class 1.
    """
    plt.figure(figsize=(7, 6))
    for label, y_score, color in curves:
        RocCurveDisplay.from_predictions(
            y_true,
            y_score,
            name=label,
            plot_chance_level=False,
            **({"color": color} if color is not None else {}),
        )
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(alpha=0.3)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()


def plot_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: str,
    title: str = "Confusion matrix",
    labels: Tuple[str, str] = ("No disease", "Disease"),
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()


def plot_metric_bar(
    df_metrics: pd.DataFrame,
    out_path: str,
    metric: str = "roc_auc",
    secondary: str | None = "f1",
    title: str = "Model comparison",
) -> None:
    """
    Bar chart of one (and optional second) metric per model.
    """
    plt.figure(figsize=(8, 5))
    models = df_metrics.index.tolist()
    x = np.arange(len(models))

    values = df_metrics[metric].values
    plt.bar(x, values, label=metric.upper(), alpha=0.8)

    if secondary is not None and secondary in df_metrics.columns:
        sec_vals = df_metrics[secondary].values
        plt.bar(
            x,
            sec_vals,
            width=0.5,
            alpha=0.6,
            label=secondary.upper(),
        )

    plt.xticks(x, models, rotation=45, ha="right")
    plt.ylabel("Score")
    plt.title(title)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()
