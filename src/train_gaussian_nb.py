"""Train Gaussian Naive Bayes on basic image-derived features."""

from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "mse446_matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd
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
from sklearn.naive_bayes import GaussianNB


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extract_basic_features import BASIC_FEATURE_COLUMNS
from src.train_logistic_regression import CLASS_LABELS, FEATURES_CSV, choose_group_split, load_features


RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
SCORES_CSV = RESULTS_DIR / "model_scores_gaussian_nb_basic.csv"
CONFUSION_MATRIX_FIGURE = FIGURES_DIR / "confusion_matrix_gaussian_nb_basic.png"
COMPARISON_SCORE_FILES = [
    ("DummyClassifier", RESULTS_DIR / "model_scores_dummy.csv"),
    ("LogisticRegression", RESULTS_DIR / "model_scores_logistic_basic.csv"),
    ("DecisionTree tuned", RESULTS_DIR / "model_scores_decision_tree_tuned.csv"),
    ("RandomForest basic", RESULTS_DIR / "model_scores_random_forest_basic.csv"),
    ("RandomForest tuned", RESULTS_DIR / "model_scores_random_forest_tuned.csv"),
]


def make_model() -> GaussianNB:
    """Create the Gaussian Naive Bayes baseline model."""
    return GaussianNB()


def score_set(y_true: pd.Series, y_pred) -> dict[str, float]:
    """Compute core metrics for one split."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def evaluate_predictions(
    y_train: pd.Series,
    y_test: pd.Series,
    y_train_pred,
    y_test_pred,
    selected_seed: int,
    split_distance: float,
) -> tuple[pd.DataFrame, str, pd.DataFrame]:
    """Compute train/test metrics, per-class report, and confusion matrix."""
    train_scores = score_set(y_train, y_train_pred)
    test_scores = score_set(y_test, y_test_pred)
    disordered_recall = recall_score(
        y_test,
        y_test_pred,
        labels=CLASS_LABELS,
        average=None,
        zero_division=0,
    )[0]

    scores = pd.DataFrame(
        [
            {
                "model": "gaussian_nb_basic",
                "train_accuracy": train_scores["accuracy"],
                "test_accuracy": test_scores["accuracy"],
                "train_balanced_accuracy": train_scores["balanced_accuracy"],
                "test_balanced_accuracy": test_scores["balanced_accuracy"],
                "train_macro_precision": train_scores["macro_precision"],
                "macro_precision": test_scores["macro_precision"],
                "train_macro_recall": train_scores["macro_recall"],
                "macro_recall": test_scores["macro_recall"],
                "train_macro_f1": train_scores["macro_f1"],
                "macro_f1": test_scores["macro_f1"],
                "disordered_recall": disordered_recall,
                "selected_split_seed": selected_seed,
                "split_distribution_distance": split_distance,
                "train_size": len(y_train),
                "test_size": len(y_test),
            }
        ]
    )
    report = classification_report(
        y_test,
        y_test_pred,
        labels=CLASS_LABELS,
        zero_division=0,
    )
    matrix = pd.DataFrame(
        confusion_matrix(y_test, y_test_pred, labels=CLASS_LABELS),
        index=CLASS_LABELS,
        columns=CLASS_LABELS,
    )
    return scores, report, matrix


def save_confusion_matrix_figure(
    y_test: pd.Series,
    y_test_pred,
    output_path: str | Path = CONFUSION_MATRIX_FIGURE,
) -> Path:
    """Save a confusion matrix figure for Gaussian Naive Bayes."""
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
    ax.set_title("Gaussian Naive Bayes basic")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    return figure_path


def train_gaussian_nb(
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, str, pd.DataFrame, Path, pd.Series, pd.Series]:
    """Train and evaluate GaussianNB on basic image features."""
    train_idx, test_idx, selected_seed, split_distance = choose_group_split(features)

    X = features[BASIC_FEATURE_COLUMNS]
    y = features["label"]
    groups = features["area_group"]
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    train_groups = set(groups.iloc[train_idx])
    test_groups = set(groups.iloc[test_idx])
    if not train_groups.isdisjoint(test_groups):
        raise RuntimeError("Grouped split leakage detected: train and test share area groups.")

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


def comparison_row(label: str, path: Path) -> dict[str, object] | None:
    """Load one comparison row from a score CSV if available."""
    if not path.exists():
        print(f"Missing comparison file for {label}: {path}")
        return None

    row = pd.read_csv(path).iloc[0]
    balanced_accuracy = row.get("test_balanced_accuracy")
    if balanced_accuracy is None:
        balanced_accuracy = row.get("balanced_accuracy")
    disordered_recall = row.get("disordered_recall", 0.0 if label == "DummyClassifier" else None)
    return {
        "model": label,
        "test_accuracy": row["test_accuracy"],
        "test_balanced_accuracy": balanced_accuracy,
        "macro_f1": row["macro_f1"],
        "disordered_recall": disordered_recall,
    }


def build_comparison_table(gaussian_scores: pd.DataFrame) -> pd.DataFrame:
    """Build a comparison table against prior model score CSVs."""
    rows = [
        row
        for label, path in COMPARISON_SCORE_FILES
        if (row := comparison_row(label, path)) is not None
    ]
    gaussian = gaussian_scores.iloc[0]
    rows.append(
        {
            "model": "GaussianNB",
            "test_accuracy": gaussian["test_accuracy"],
            "test_balanced_accuracy": gaussian["test_balanced_accuracy"],
            "macro_f1": gaussian["macro_f1"],
            "disordered_recall": gaussian["disordered_recall"],
        }
    )
    return pd.DataFrame(rows)


def main() -> int:
    """Run Gaussian Naive Bayes training on basic image features."""
    try:
        features = load_features(FEATURES_CSV)
        scores, report, matrix, figure_path, y_train, y_test = train_gaussian_nb(features)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    SCORES_CSV.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(SCORES_CSV, index=False)

    print("GaussianNB basic metrics:")
    print(scores.to_string(index=False))
    print("\nSelected split label counts:")
    print("Train:")
    print(y_train.value_counts().sort_index().to_string())
    print("Test:")
    print(y_test.value_counts().sort_index().to_string())
    print("\nPer-class precision/recall/F1:")
    print(report)
    print("Confusion matrix rows=true, columns=predicted:")
    print(matrix.to_string())
    print("\nModel comparison:")
    print(build_comparison_table(scores).to_string(index=False))
    print(
        "\nNote: GaussianNB is a probability-based baseline. Its independence and "
        "normal-feature assumptions are probably weak for image features, so compare "
        "it mainly by balanced accuracy, macro F1, and disordered recall."
    )
    print(f"\nSaved scores to {SCORES_CSV}")
    print(f"Saved confusion matrix figure to {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
