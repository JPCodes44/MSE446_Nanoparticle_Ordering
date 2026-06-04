"""Train a majority-class DummyClassifier baseline."""

from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "mse446_matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METADATA_CSV = DATA_DIR / "dataset_metadata.csv"
SCORES_CSV = RESULTS_DIR / "model_scores_dummy.csv"
CONFUSION_MATRIX_FIGURE = FIGURES_DIR / "confusion_matrix_dummy.png"
RANDOM_STATE = 42
TEST_SIZE = 0.20
CLASS_LABELS = ["disordered", "ordered"]


def load_metadata(path: str | Path = METADATA_CSV) -> pd.DataFrame:
    """Load parsed metadata for dummy baseline training."""
    metadata_path = Path(path)
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}. Run python src/parse_metadata.py first."
        )
    metadata = pd.read_csv(metadata_path)
    required_columns = {"label", "area_group"}
    missing = required_columns.difference(metadata.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")
    if metadata.empty:
        raise ValueError("Metadata table is empty.")
    return metadata


def make_placeholder_features(n_samples: int) -> np.ndarray:
    """Create placeholder features because DummyClassifier ignores X."""
    return np.zeros((n_samples, 1), dtype=float)


def split_group_aware(
    metadata: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Split rows while keeping sample-area groups out of both train and test."""
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    X_placeholder = make_placeholder_features(len(metadata))
    y = metadata["label"]
    groups = metadata["area_group"]
    train_idx, test_idx = next(splitter.split(X_placeholder, y, groups))

    train_groups = set(groups.iloc[train_idx])
    test_groups = set(groups.iloc[test_idx])
    if not train_groups.isdisjoint(test_groups):
        raise RuntimeError("Group-aware split failed: train/test area groups overlap.")
    return train_idx, test_idx


def evaluate_dummy_baseline(
    y_train: pd.Series,
    y_test: pd.Series,
    y_train_pred: np.ndarray,
    y_test_pred: np.ndarray,
) -> tuple[pd.DataFrame, str, np.ndarray]:
    """Compute aggregate, per-class, and confusion-matrix metrics."""
    aggregate_scores = {
        "model": "dummy_most_frequent",
        "strategy": "most_frequent",
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "test_balanced_accuracy": balanced_accuracy_score(y_test, y_test_pred),
        "macro_precision": precision_score(
            y_test, y_test_pred, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(y_test, y_test_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_test, y_test_pred, average="macro", zero_division=0),
        "train_size": len(y_train),
        "test_size": len(y_test),
    }
    scores = pd.DataFrame([aggregate_scores])
    report = classification_report(
        y_test,
        y_test_pred,
        labels=CLASS_LABELS,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, y_test_pred, labels=CLASS_LABELS)
    return scores, report, matrix


def save_confusion_matrix_figure(
    y_test: pd.Series,
    y_test_pred: np.ndarray,
    output_path: str | Path = CONFUSION_MATRIX_FIGURE,
) -> Path:
    """Save a confusion matrix figure for the dummy baseline."""
    figure_path = Path(output_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_test_pred,
        labels=CLASS_LABELS,
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title("Dummy majority baseline")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    return figure_path


def train_dummy_baseline(metadata: pd.DataFrame) -> tuple[pd.DataFrame, str, np.ndarray, Path]:
    """Train and evaluate the majority-class dummy baseline."""
    train_idx, test_idx = split_group_aware(metadata)
    X = make_placeholder_features(len(metadata))
    y = metadata["label"]

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    model = DummyClassifier(strategy="most_frequent")
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    scores, report, matrix = evaluate_dummy_baseline(
        y_train,
        y_test,
        y_train_pred,
        y_test_pred,
    )
    figure_path = save_confusion_matrix_figure(y_test, y_test_pred)
    return scores, report, matrix, figure_path


def main() -> int:
    """Run the dummy baseline script."""
    try:
        metadata = load_metadata(METADATA_CSV)
        scores, report, matrix, figure_path = train_dummy_baseline(metadata)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    SCORES_CSV.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(SCORES_CSV, index=False)

    print("WARNING: Raw accuracy is misleading because the dataset is imbalanced.")
    print("\nDummyClassifier(strategy='most_frequent') metrics:")
    print(scores.to_string(index=False))
    print("\nPer-class precision/recall/F1:")
    print(report)
    print("Confusion matrix rows=true, columns=predicted:")
    print(pd.DataFrame(matrix, index=CLASS_LABELS, columns=CLASS_LABELS).to_string())
    print(f"\nSaved scores to {SCORES_CSV}")
    print(f"Saved confusion matrix figure to {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
