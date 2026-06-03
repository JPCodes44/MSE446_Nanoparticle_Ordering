"""Evaluation and plotting helpers for baseline classifiers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def classification_metrics(
    model_name: str,
    y_true,
    y_pred,
) -> dict[str, float | str]:
    """Compute binary classification metrics for ordered/disordered labels."""
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, pos_label="ordered", zero_division=0
        ),
        "recall": recall_score(y_true, y_pred, pos_label="ordered", zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label="ordered", zero_division=0),
    }


def save_confusion_matrix(
    model_name: str,
    y_true,
    y_pred,
    output_dir: str | Path,
) -> Path:
    """Save a confusion matrix plot for one model."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    figure_path = output_path / f"confusion_matrix_{model_name}.png"

    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=["disordered", "ordered"],
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title(model_name.replace("_", " "))
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    return figure_path


def save_scores(scores: list[dict[str, float | str]], output_csv: str | Path) -> pd.DataFrame:
    """Save model score dictionaries as a sorted CSV."""
    score_table = pd.DataFrame(scores).sort_values("balanced_accuracy", ascending=False)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    score_table.to_csv(output_csv, index=False)
    return score_table
