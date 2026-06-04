"""Train a DecisionTreeClassifier baseline on basic image-derived features."""

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
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.tree import DecisionTreeClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extract_basic_features import BASIC_FEATURE_COLUMNS
from src.train_logistic_regression import (
    CLASS_LABELS,
    FEATURES_CSV,
    choose_group_split,
    load_features,
)


RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
SCORES_CSV = RESULTS_DIR / "model_scores_decision_tree_basic.csv"
CONFUSION_MATRIX_FIGURE = FIGURES_DIR / "confusion_matrix_decision_tree_basic.png"
TUNED_SCORES_CSV = RESULTS_DIR / "model_scores_decision_tree_tuned.csv"
TUNED_CONFUSION_MATRIX_FIGURE = FIGURES_DIR / "confusion_matrix_decision_tree_tuned.png"
RANDOM_STATE = 42
PARAM_GRID = {
    "max_depth": [2, 3, 4, 5, 8, 10, None],
    "min_samples_leaf": [1, 2, 5, 10, 20],
    "min_samples_split": [2, 5, 10, 20],
    "criterion": ["gini", "entropy"],
}


def make_model() -> DecisionTreeClassifier:
    """Create the decision tree baseline model."""
    return DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced")


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
    model_name: str,
    y_train: pd.Series,
    y_test: pd.Series,
    y_train_pred,
    y_test_pred,
    selected_seed: int,
    split_distance: float,
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
    title: str = "Decision tree basic",
) -> Path:
    """Save a confusion matrix figure for the decision tree baseline."""
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


def tune_decision_tree(X_train: pd.DataFrame, y_train: pd.Series, groups: pd.Series) -> GridSearchCV:
    """Tune tree hyperparameters with group-aware cross-validation."""
    search = GridSearchCV(
        estimator=make_model(),
        param_grid=PARAM_GRID,
        scoring="f1_macro",
        cv=GroupKFold(n_splits=5),
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train, groups=groups)
    return search


def evaluate_model(
    model: DecisionTreeClassifier,
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
    """Fit a tree model and evaluate it on train and held-out test rows."""
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
        best_cv_macro_f1,
        best_params,
    )
    saved_figure_path = save_confusion_matrix_figure(
        y_test, y_test_pred, figure_path, figure_title
    )
    return scores, report, matrix, saved_figure_path


def train_decision_tree(
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
    """Train untuned and tuned DecisionTreeClassifier baselines."""
    train_idx, test_idx, selected_seed, split_distance = choose_group_split(features)

    X = features[BASIC_FEATURE_COLUMNS]
    y = features["label"]
    groups = features["area_group"]
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    train_groups = groups.iloc[train_idx]

    basic_scores, basic_report, basic_matrix, basic_figure_path = evaluate_model(
        make_model(),
        "decision_tree_basic",
        X_train,
        X_test,
        y_train,
        y_test,
        selected_seed,
        split_distance,
        CONFUSION_MATRIX_FIGURE,
        "Decision tree basic",
    )

    search = tune_decision_tree(X_train, y_train, train_groups)
    tuned_scores, tuned_report, tuned_matrix, tuned_figure_path = evaluate_model(
        search.best_estimator_,
        "decision_tree_tuned",
        X_train,
        X_test,
        y_train,
        y_test,
        selected_seed,
        split_distance,
        TUNED_CONFUSION_MATRIX_FIGURE,
        "Decision tree tuned",
        float(search.best_score_),
        search.best_params_,
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


def print_overfitting_summary(scores: pd.DataFrame, label: str) -> None:
    """Print train-vs-test metrics for quick overfitting inspection."""
    row = scores.iloc[0]
    print(f"\n{label} train vs test overfitting check:")
    print(f"train accuracy: {row['train_accuracy']:.4f}")
    print(f"test accuracy: {row['test_accuracy']:.4f}")
    print(f"train balanced accuracy: {row['train_balanced_accuracy']:.4f}")
    print(f"test balanced accuracy: {row['test_balanced_accuracy']:.4f}")
    print(f"train macro F1: {row['train_macro_f1']:.4f}")
    print(f"test macro F1: {row['macro_f1']:.4f}")


def main() -> int:
    """Run DecisionTreeClassifier training on basic image features."""
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
        ) = train_decision_tree(features)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    SCORES_CSV.parent.mkdir(parents=True, exist_ok=True)
    basic_scores.to_csv(SCORES_CSV, index=False)
    tuned_scores.to_csv(TUNED_SCORES_CSV, index=False)

    print("Untuned DecisionTreeClassifier(random_state=42, class_weight='balanced') metrics:")
    print(basic_scores.to_string(index=False))
    print_overfitting_summary(basic_scores, "Untuned decision tree")
    print("\nTuned DecisionTreeClassifier metrics:")
    print(tuned_scores.to_string(index=False))
    print_overfitting_summary(tuned_scores, "Tuned decision tree")
    print("\nTuning details:")
    print(f"best CV macro F1: {search.best_score_:.4f}")
    print(f"best params: {search.best_params_}")
    print("\nSelected split label counts:")
    print("Train:")
    print(y_train.value_counts().sort_index().to_string())
    print("Test:")
    print(y_test.value_counts().sort_index().to_string())
    print("\nUntuned per-class precision/recall/F1:")
    print(basic_report)
    print("Untuned confusion matrix rows=true, columns=predicted:")
    print(basic_matrix.to_string())
    print("\nTuned per-class precision/recall/F1:")
    print(tuned_report)
    print("Tuned confusion matrix rows=true, columns=predicted:")
    print(tuned_matrix.to_string())
    print(f"\nSaved scores to {SCORES_CSV}")
    print(f"Saved confusion matrix figure to {basic_figure_path}")
    print(f"Saved tuned scores to {TUNED_SCORES_CSV}")
    print(f"Saved tuned confusion matrix figure to {tuned_figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
