"""Train Logistic Regression on basic image-derived features."""

from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "mse446_matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extract_basic_features import BASIC_FEATURE_COLUMNS

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FEATURES_CSV = DATA_DIR / "features_basic.csv"
DUMMY_SCORES_CSV = RESULTS_DIR / "model_scores_dummy.csv"
SCORES_CSV = RESULTS_DIR / "model_scores_logistic_basic.csv"
CONFUSION_MATRIX_FIGURE = FIGURES_DIR / "confusion_matrix_logistic_basic.png"
CLASS_LABELS = ["disordered", "ordered"]
TEST_SIZE = 0.20
SEED_RANGE = range(1000)


def load_features(path: str | Path = FEATURES_CSV) -> pd.DataFrame:
    """Load basic feature table for logistic regression."""
    features_path = Path(path)
    if not features_path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {features_path}. "
            "Run python src/extract_basic_features.py first."
        )
    features = pd.read_csv(features_path)
    required_columns = set(BASIC_FEATURE_COLUMNS).union({"label", "area_group"})
    missing = required_columns.difference(features.columns)
    if missing:
        raise ValueError(f"Feature table is missing required columns: {sorted(missing)}")
    if features.empty:
        raise ValueError("Feature table is empty.")
    return features


def label_proportions(y: pd.Series) -> pd.Series:
    """Return label proportions indexed by known labels."""
    return y.value_counts(normalize=True).reindex(CLASS_LABELS, fill_value=0.0)


def split_distribution_distance(full_y: pd.Series, test_y: pd.Series) -> float:
    """Measure how close the test label distribution is to the full dataset."""
    return float(np.abs(label_proportions(full_y) - label_proportions(test_y)).sum())


def choose_group_split(
    features: pd.DataFrame,
    test_size: float = TEST_SIZE,
    seeds=SEED_RANGE,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Choose a group-aware split with test labels closest to full labels."""
    y = features["label"]
    groups = features["area_group"]
    X_placeholder = np.zeros((len(features), 1), dtype=float)
    best: tuple[np.ndarray, np.ndarray, int, float] | None = None

    for seed in seeds:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=seed,
        )
        train_idx, test_idx = next(splitter.split(X_placeholder, y, groups))
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            continue

        train_groups = set(groups.iloc[train_idx])
        test_groups = set(groups.iloc[test_idx])
        if not train_groups.isdisjoint(test_groups):
            continue

        distance = split_distribution_distance(y, y_test)
        if best is None or distance < best[3]:
            best = (train_idx, test_idx, seed, distance)

    if best is None:
        raise RuntimeError("Could not find a valid group-aware split containing both labels.")
    return best


def make_model():
    """Create the scaled Logistic Regression pipeline."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000),
    )


def evaluate_predictions(
    y_train: pd.Series,
    y_test: pd.Series,
    y_train_pred: np.ndarray,
    y_test_pred: np.ndarray,
    selected_seed: int,
    split_distance: float,
) -> tuple[pd.DataFrame, str, np.ndarray]:
    """Compute aggregate, per-class, and confusion-matrix metrics."""
    disordered_recall = recall_score(
        y_test,
        y_test_pred,
        labels=CLASS_LABELS,
        average=None,
        zero_division=0,
    )[0]
    aggregate_scores = {
        "model": "logistic_regression_basic",
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "test_balanced_accuracy": balanced_accuracy_score(y_test, y_test_pred),
        "macro_precision": precision_score(
            y_test, y_test_pred, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(y_test, y_test_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_test, y_test_pred, average="macro", zero_division=0),
        "disordered_recall": disordered_recall,
        "selected_split_seed": selected_seed,
        "split_distribution_distance": split_distance,
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
    """Save a confusion matrix figure for logistic regression."""
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
    ax.set_title("Logistic regression basic")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    return figure_path


def train_logistic_regression(
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, str, np.ndarray, Path, pd.Series, pd.Series]:
    """Train and evaluate Logistic Regression on basic image features."""
    train_idx, test_idx, selected_seed, split_distance = choose_group_split(features)

    X = features[BASIC_FEATURE_COLUMNS]
    y = features["label"]
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    model = make_model()
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    scores, report, matrix = evaluate_predictions(
        y_train,
        y_test,
        y_train_pred,
        y_test_pred,
        selected_seed,
        split_distance,
    )
    figure_path = save_confusion_matrix_figure(y_test, y_test_pred)
    return scores, report, matrix, figure_path, y_train, y_test


def load_dummy_scores(path: str | Path = DUMMY_SCORES_CSV) -> pd.DataFrame | None:
    """Load dummy baseline scores when available."""
    scores_path = Path(path)
    if not scores_path.exists():
        return None
    return pd.read_csv(scores_path)


def print_dummy_comparison(logistic_scores: pd.DataFrame) -> None:
    """Print dummy-vs-logistic headline metrics."""
    dummy_scores = load_dummy_scores(DUMMY_SCORES_CSV)
    logistic = logistic_scores.iloc[0]

    if dummy_scores is None:
        print("\nDummy baseline scores not found. Run python src/train_dummy_baseline.py first.")
        print("\nLogistic Regression headline metrics:")
        print(
            f"accuracy={logistic['test_accuracy']:.4f}, "
            f"balanced_accuracy={logistic['test_balanced_accuracy']:.4f}, "
            f"macro_f1={logistic['macro_f1']:.4f}, "
            f"disordered_recall={logistic['disordered_recall']:.4f}"
        )
        return

    dummy = dummy_scores.iloc[0]
    print("\nDummy vs Logistic Regression:")
    print(f"dummy accuracy: {dummy['test_accuracy']:.4f}")
    print(f"dummy balanced accuracy: {dummy['test_balanced_accuracy']:.4f}")
    print(f"dummy macro F1: {dummy['macro_f1']:.4f}")
    print(f"logistic accuracy: {logistic['test_accuracy']:.4f}")
    print(f"logistic balanced accuracy: {logistic['test_balanced_accuracy']:.4f}")
    print(f"logistic macro F1: {logistic['macro_f1']:.4f}")
    print(f"logistic disordered recall: {logistic['disordered_recall']:.4f}")


def main() -> int:
    """Run Logistic Regression training on basic image features."""
    try:
        features = load_features(FEATURES_CSV)
        scores, report, matrix, figure_path, y_train, y_test = train_logistic_regression(
            features
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    SCORES_CSV.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(SCORES_CSV, index=False)

    print("LogisticRegression(class_weight='balanced', max_iter=2000) metrics:")
    print(scores.to_string(index=False))
    print("\nSelected split label counts:")
    print("Train:")
    print(y_train.value_counts().sort_index().to_string())
    print("Test:")
    print(y_test.value_counts().sort_index().to_string())
    print("\nPer-class precision/recall/F1:")
    print(report)
    print("Confusion matrix rows=true, columns=predicted:")
    print(pd.DataFrame(matrix, index=CLASS_LABELS, columns=CLASS_LABELS).to_string())
    print_dummy_comparison(scores)
    print(f"\nSaved scores to {SCORES_CSV}")
    print(f"Saved confusion matrix figure to {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
