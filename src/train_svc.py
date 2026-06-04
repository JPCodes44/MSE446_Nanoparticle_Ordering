"""Train basic and tuned Support Vector Classifier models on basic image features."""

from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "mse446_matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
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
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extract_basic_features import BASIC_FEATURE_COLUMNS
from src.train_logistic_regression import CLASS_LABELS, FEATURES_CSV, choose_group_split, load_features


RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
BASIC_SCORES_CSV = RESULTS_DIR / "model_scores_svc_basic.csv"
TUNED_SCORES_CSV = RESULTS_DIR / "model_scores_svc_tuned.csv"
BASIC_CONFUSION_MATRIX_FIGURE = FIGURES_DIR / "confusion_matrix_svc_basic.png"
TUNED_CONFUSION_MATRIX_FIGURE = FIGURES_DIR / "confusion_matrix_svc_tuned.png"
COMPARISON_SCORE_FILES = [
    ("DummyClassifier", RESULTS_DIR / "model_scores_dummy.csv"),
    ("LogisticRegression", RESULTS_DIR / "model_scores_logistic_basic.csv"),
    ("DecisionTree tuned", RESULTS_DIR / "model_scores_decision_tree_tuned.csv"),
    ("RandomForest basic", RESULTS_DIR / "model_scores_random_forest_basic.csv"),
    ("RandomForest tuned", RESULTS_DIR / "model_scores_random_forest_tuned.csv"),
    ("GaussianNB", RESULTS_DIR / "model_scores_gaussian_nb_basic.csv"),
    ("KNN basic", RESULTS_DIR / "model_scores_knn_basic.csv"),
    ("KNN tuned", RESULTS_DIR / "model_scores_knn_tuned.csv"),
]
RANDOM_STATE = 42
CV_FOLDS = 5
PARAM_GRID = {
    "svc__C": [0.1, 1, 10, 100],
    "svc__kernel": ["linear", "rbf"],
    "svc__gamma": ["scale", "auto", 0.01, 0.1, 1],
    "svc__class_weight": ["balanced"],
}


def make_basic_model():
    """Create the scaled basic SVC pipeline."""
    return make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    )


def score_set(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
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


def svc_params(model) -> dict[str, object]:
    """Return SVC hyperparameters from a fitted pipeline."""
    svc = model.named_steps["svc"]
    return {
        "C": svc.C,
        "kernel": svc.kernel,
        "gamma": svc.gamma,
        "class_weight": svc.class_weight,
    }


def evaluate_predictions(
    model_name: str,
    y_train: pd.Series,
    y_test: pd.Series,
    y_train_pred: np.ndarray,
    y_test_pred: np.ndarray,
    selected_seed: int,
    split_distance: float,
    model,
    best_cv_macro_f1: float | None = None,
    best_params: dict[str, object] | None = None,
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
    params = svc_params(model)

    scores = pd.DataFrame(
        [
            {
                "model": model_name,
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
                "best_cv_macro_f1": best_cv_macro_f1,
                "best_params": "" if best_params is None else repr(best_params),
                "C": params["C"],
                "kernel": params["kernel"],
                "gamma": params["gamma"],
                "class_weight": str(params["class_weight"]),
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
    y_test_pred: np.ndarray,
    output_path: str | Path,
    title: str,
) -> Path:
    """Save a confusion matrix figure for one SVC experiment."""
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
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    return figure_path


def cross_validation_folds(groups: pd.Series, requested_folds: int = CV_FOLDS) -> int:
    """Return a valid number of group-aware CV folds for the training groups."""
    unique_groups = groups.nunique()
    if unique_groups < 2:
        raise ValueError("Need at least two training area groups for GroupKFold tuning.")
    return min(requested_folds, unique_groups)


def tune_svc(X_train: pd.DataFrame, y_train: pd.Series, groups: pd.Series) -> GridSearchCV:
    """Tune a scaled SVC pipeline with GroupKFold on the training split only."""
    search = GridSearchCV(
        estimator=make_basic_model(),
        param_grid=PARAM_GRID,
        scoring="f1_macro",
        cv=GroupKFold(n_splits=cross_validation_folds(groups)),
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train, groups=groups)
    return search


def evaluate_model(
    model,
    model_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    selected_seed: int,
    split_distance: float,
    figure_path: str | Path,
    figure_title: str,
    best_cv_macro_f1: float | None = None,
    best_params: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, str, pd.DataFrame, Path]:
    """Fit an SVC pipeline and evaluate it on train and held-out test rows."""
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    scores, report, matrix = evaluate_predictions(
        model_name,
        y_train,
        y_test,
        y_train_pred,
        y_test_pred,
        selected_seed,
        split_distance,
        model,
        best_cv_macro_f1=best_cv_macro_f1,
        best_params=best_params,
    )
    saved_figure_path = save_confusion_matrix_figure(
        y_test, y_test_pred, figure_path, figure_title
    )
    return scores, report, matrix, saved_figure_path


def train_svc(
    features: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    str,
    pd.DataFrame,
    Path,
    pd.DataFrame,
    str,
    pd.DataFrame,
    Path,
    pd.Series,
    pd.Series,
    GridSearchCV,
]:
    """Train and evaluate basic and tuned SVC experiments."""
    train_idx, test_idx, selected_seed, split_distance = choose_group_split(features)

    X = features[BASIC_FEATURE_COLUMNS]
    y = features["label"]
    groups = features["area_group"]
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    train_groups = groups.iloc[train_idx]

    train_group_set = set(train_groups)
    test_group_set = set(groups.iloc[test_idx])
    if not train_group_set.isdisjoint(test_group_set):
        raise RuntimeError("Grouped split leakage detected: train and test share area groups.")

    basic_scores, basic_report, basic_matrix, basic_figure_path = evaluate_model(
        make_basic_model(),
        "svc_basic",
        X_train,
        X_test,
        y_train,
        y_test,
        selected_seed,
        split_distance,
        BASIC_CONFUSION_MATRIX_FIGURE,
        "SVC basic",
    )

    search = tune_svc(X_train, y_train, train_groups)
    tuned_scores, tuned_report, tuned_matrix, tuned_figure_path = evaluate_model(
        search.best_estimator_,
        "svc_tuned",
        X_train,
        X_test,
        y_train,
        y_test,
        selected_seed,
        split_distance,
        TUNED_CONFUSION_MATRIX_FIGURE,
        "SVC tuned",
        best_cv_macro_f1=float(search.best_score_),
        best_params=search.best_params_,
    )
    return (
        basic_scores,
        basic_report,
        basic_matrix,
        basic_figure_path,
        tuned_scores,
        tuned_report,
        tuned_matrix,
        tuned_figure_path,
        y_train,
        y_test,
        search,
    )


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


def build_comparison_table(
    basic_scores: pd.DataFrame,
    tuned_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Build a comparison table against prior model score CSVs."""
    rows = [
        row
        for label, path in COMPARISON_SCORE_FILES
        if (row := comparison_row(label, path)) is not None
    ]
    for label, scores in [("SVC basic", basic_scores), ("SVC tuned", tuned_scores)]:
        row = scores.iloc[0]
        rows.append(
            {
                "model": label,
                "test_accuracy": row["test_accuracy"],
                "test_balanced_accuracy": row["test_balanced_accuracy"],
                "macro_f1": row["macro_f1"],
                "disordered_recall": row["disordered_recall"],
            }
        )
    return pd.DataFrame(rows)


def print_overfitting_summary(scores: pd.DataFrame, label: str) -> None:
    """Print train-vs-test metrics for quick overfitting inspection."""
    row = scores.iloc[0]
    print(f"\n{label} train vs test:")
    print(f"  train accuracy:           {row['train_accuracy']:.4f}")
    print(f"  test accuracy:            {row['test_accuracy']:.4f}")
    print(f"  train balanced accuracy:  {row['train_balanced_accuracy']:.4f}")
    print(f"  test balanced accuracy:   {row['test_balanced_accuracy']:.4f}")
    print(f"  train macro F1:           {row['train_macro_f1']:.4f}")
    print(f"  test macro F1:            {row['macro_f1']:.4f}")


def print_model_outputs(
    scores: pd.DataFrame,
    report: str,
    matrix: pd.DataFrame,
    label: str,
) -> None:
    """Print metrics, per-class report, and confusion matrix."""
    print(f"\n{label} metrics:")
    print(scores.to_string(index=False))
    print("\nPer-class precision/recall/F1:")
    print(report)
    print("Confusion matrix rows=true, columns=predicted:")
    print(matrix.to_string())


def main() -> int:
    """Run basic and tuned SVC training on basic image features."""
    try:
        features = load_features(FEATURES_CSV)
        (
            basic_scores,
            basic_report,
            basic_matrix,
            basic_figure_path,
            tuned_scores,
            tuned_report,
            tuned_matrix,
            tuned_figure_path,
            y_train,
            y_test,
            search,
        ) = train_svc(features)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    BASIC_SCORES_CSV.parent.mkdir(parents=True, exist_ok=True)
    basic_scores.to_csv(BASIC_SCORES_CSV, index=False)
    tuned_scores.to_csv(TUNED_SCORES_CSV, index=False)

    print("Support Vector Classifier basic and tuned experiments")
    print("\nSelected split label counts:")
    print("Train:")
    print(y_train.value_counts().sort_index().to_string())
    print("Test:")
    print(y_test.value_counts().sort_index().to_string())

    print_model_outputs(basic_scores, basic_report, basic_matrix, "Basic SVC")
    print_model_outputs(tuned_scores, tuned_report, tuned_matrix, "Tuned SVC")
    print_overfitting_summary(basic_scores, "Basic SVC")
    print_overfitting_summary(tuned_scores, "Tuned SVC")

    tuned_row = tuned_scores.iloc[0]
    print("\nTuning summary:")
    print(f"Best parameters: {search.best_params_}")
    print(f"Best CV macro F1: {search.best_score_:.4f}")
    print(f"Final test macro F1: {tuned_row['macro_f1']:.4f}")
    print(f"Final test balanced accuracy: {tuned_row['test_balanced_accuracy']:.4f}")
    print(f"Disordered recall: {tuned_row['disordered_recall']:.4f}")

    print("\nModel comparison:")
    print(build_comparison_table(basic_scores, tuned_scores).to_string(index=False))
    print(
        "\nNote: SVC finds a maximum-margin decision boundary. Scaling is required "
        "because margins and RBF distances depend on feature magnitude."
    )
    print(f"\nSaved basic scores to {BASIC_SCORES_CSV}")
    print(f"Saved tuned scores to {TUNED_SCORES_CSV}")
    print(f"Saved basic confusion matrix figure to {basic_figure_path}")
    print(f"Saved tuned confusion matrix figure to {tuned_figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
