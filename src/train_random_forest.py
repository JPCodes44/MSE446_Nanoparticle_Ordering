"""Train basic and tuned RandomForestClassifier models on basic image features."""

from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "mse446_matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extract_basic_features import BASIC_FEATURE_COLUMNS
from src.train_logistic_regression import CLASS_LABELS, FEATURES_CSV, choose_group_split, load_features


RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
BASIC_SCORES_CSV = RESULTS_DIR / "model_scores_random_forest_basic.csv"
BASIC_CONFUSION_MATRIX_FIGURE = FIGURES_DIR / "confusion_matrix_random_forest_basic.png"
BASIC_FEATURE_IMPORTANCE_CSV = RESULTS_DIR / "feature_importance_random_forest_basic.csv"
BASIC_FEATURE_IMPORTANCE_FIGURE = FIGURES_DIR / "feature_importance_random_forest_basic.png"
TUNED_SCORES_CSV = RESULTS_DIR / "model_scores_random_forest_tuned.csv"
TUNED_CONFUSION_MATRIX_FIGURE = FIGURES_DIR / "confusion_matrix_random_forest_tuned.png"
TUNED_FEATURE_IMPORTANCE_CSV = RESULTS_DIR / "feature_importance_random_forest_tuned.csv"
TUNED_FEATURE_IMPORTANCE_FIGURE = FIGURES_DIR / "feature_importance_random_forest_tuned.png"
COMPARISON_SCORE_FILES = [
    ("DummyClassifier", RESULTS_DIR / "model_scores_dummy.csv"),
    ("LogisticRegression", RESULTS_DIR / "model_scores_logistic_basic.csv"),
    ("DecisionTree untuned", RESULTS_DIR / "model_scores_decision_tree_basic.csv"),
    ("DecisionTree tuned", RESULTS_DIR / "model_scores_decision_tree_tuned.csv"),
]
RANDOM_STATE = 42
CV_FOLDS = 5
PARAM_GRID = {
    "n_estimators": [100, 200, 500],
    "max_depth": [4, 8, 12, None],
    "min_samples_leaf": [1, 2, 5, 10],
    "min_samples_split": [2, 5, 10],
    "max_features": ["sqrt", "log2", None],
    "class_weight": ["balanced", "balanced_subsample"],
}


def make_basic_model() -> RandomForestClassifier:
    """Create the untuned random forest baseline model."""
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=1,
        min_samples_split=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


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
    model: RandomForestClassifier,
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
                "best_params": str(best_params) if best_params else None,
                "n_estimators": model.n_estimators,
                "max_depth": model.max_depth,
                "min_samples_leaf": model.min_samples_leaf,
                "min_samples_split": model.min_samples_split,
                "max_features": model.max_features,
                "class_weight": str(model.class_weight),
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
    output_path: str | Path,
    title: str,
) -> Path:
    """Save a confusion matrix figure for one random forest model."""
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


def build_feature_importance_table(model: RandomForestClassifier) -> pd.DataFrame:
    """Return impurity-based feature importances sorted descending."""
    return (
        pd.DataFrame(
            {
                "feature": BASIC_FEATURE_COLUMNS,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def save_feature_importance_outputs(
    importance: pd.DataFrame,
    output_csv: str | Path,
    output_figure: str | Path,
    title: str,
) -> tuple[Path, Path]:
    """Save feature importance table and plot."""
    csv_path = Path(output_csv)
    figure_path = Path(output_figure)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    importance.to_csv(csv_path, index=False)

    plot_table = importance.sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(plot_table["feature"], plot_table["importance"], color="#4c78a8")
    ax.set_xlabel("Impurity-based importance")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    return csv_path, figure_path


def cross_validation_folds(groups: pd.Series, requested_folds: int = CV_FOLDS) -> int:
    """Return a valid number of group-aware CV folds for the training groups."""
    unique_groups = groups.nunique()
    if unique_groups < 2:
        raise ValueError("Need at least two training area groups for GroupKFold tuning.")
    return min(requested_folds, unique_groups)


def tune_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups: pd.Series,
) -> GridSearchCV:
    """Tune RandomForestClassifier with GroupKFold on the training split only."""
    n_splits = cross_validation_folds(groups)
    search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        param_grid=PARAM_GRID,
        scoring="f1_macro",
        cv=GroupKFold(n_splits=n_splits),
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train, groups=groups)
    return search


def evaluate_model(
    model: RandomForestClassifier,
    model_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    selected_seed: int,
    split_distance: float,
    matrix_figure: str | Path,
    importance_csv: str | Path,
    importance_figure: str | Path,
    figure_title: str,
    importance_title: str,
    best_cv_macro_f1: float | None = None,
    best_params: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, str, pd.DataFrame, Path, pd.DataFrame, Path, Path]:
    """Fit and evaluate one RandomForestClassifier experiment."""
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
    matrix_path = save_confusion_matrix_figure(y_test, y_test_pred, matrix_figure, figure_title)
    importance = build_feature_importance_table(model)
    importance_csv_path, importance_figure_path = save_feature_importance_outputs(
        importance,
        importance_csv,
        importance_figure,
        importance_title,
    )
    return scores, report, matrix, matrix_path, importance, importance_csv_path, importance_figure_path


def train_random_forest(
    features: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    str,
    pd.DataFrame,
    Path,
    pd.DataFrame,
    Path,
    Path,
    pd.DataFrame,
    str,
    pd.DataFrame,
    Path,
    pd.DataFrame,
    Path,
    Path,
    pd.Series,
    pd.Series,
    GridSearchCV,
]:
    """Train and evaluate basic and tuned RandomForestClassifier experiments."""
    train_idx, test_idx, selected_seed, split_distance = choose_group_split(features)

    X = features[BASIC_FEATURE_COLUMNS]
    y = features["label"]
    groups = features["area_group"]
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    train_groups = groups.iloc[train_idx]

    basic_results = evaluate_model(
        make_basic_model(),
        "random_forest_basic",
        X_train,
        X_test,
        y_train,
        y_test,
        selected_seed,
        split_distance,
        BASIC_CONFUSION_MATRIX_FIGURE,
        BASIC_FEATURE_IMPORTANCE_CSV,
        BASIC_FEATURE_IMPORTANCE_FIGURE,
        "Random forest basic",
        "Random forest basic feature importance",
    )

    search = tune_random_forest(X_train, y_train, train_groups)
    tuned_results = evaluate_model(
        search.best_estimator_,
        "random_forest_tuned",
        X_train,
        X_test,
        y_train,
        y_test,
        selected_seed,
        split_distance,
        TUNED_CONFUSION_MATRIX_FIGURE,
        TUNED_FEATURE_IMPORTANCE_CSV,
        TUNED_FEATURE_IMPORTANCE_FIGURE,
        "Random forest tuned",
        "Random forest tuned feature importance",
        best_cv_macro_f1=float(search.best_score_),
        best_params=search.best_params_,
    )

    return (*basic_results, *tuned_results, y_train, y_test, search)


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
    """Build comparison table against prior model score CSVs."""
    rows = [
        row
        for label, path in COMPARISON_SCORE_FILES
        if (row := comparison_row(label, path)) is not None
    ]
    for label, scores in [
        ("RandomForest basic", basic_scores),
        ("RandomForest tuned", tuned_scores),
    ]:
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
    importance: pd.DataFrame,
    label: str,
) -> None:
    """Print metrics, per-class report, confusion matrix, and importances."""
    print(f"\n{label} metrics:")
    print(scores.to_string(index=False))
    print("\nPer-class precision/recall/F1:")
    print(report)
    print("Confusion matrix rows=true, columns=predicted:")
    print(matrix.to_string())
    print("\nTop feature importances:")
    print(importance.head(12).to_string(index=False))


def main() -> int:
    """Run basic and tuned RandomForestClassifier training on basic image features."""
    try:
        features = load_features(FEATURES_CSV)
        (
            basic_scores,
            basic_report,
            basic_matrix,
            basic_matrix_path,
            basic_importance,
            basic_importance_csv,
            basic_importance_figure,
            tuned_scores,
            tuned_report,
            tuned_matrix,
            tuned_matrix_path,
            tuned_importance,
            tuned_importance_csv,
            tuned_importance_figure,
            y_train,
            y_test,
            search,
        ) = train_random_forest(features)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    BASIC_SCORES_CSV.parent.mkdir(parents=True, exist_ok=True)
    basic_scores.to_csv(BASIC_SCORES_CSV, index=False)
    tuned_scores.to_csv(TUNED_SCORES_CSV, index=False)

    print("RandomForestClassifier basic and tuned experiments")
    print("\nSelected split label counts:")
    print("Train:")
    print(y_train.value_counts().sort_index().to_string())
    print("Test:")
    print(y_test.value_counts().sort_index().to_string())

    print_model_outputs(
        basic_scores,
        basic_report,
        basic_matrix,
        basic_importance,
        "Basic Random Forest",
    )
    print_model_outputs(
        tuned_scores,
        tuned_report,
        tuned_matrix,
        tuned_importance,
        "Tuned Random Forest",
    )

    print_overfitting_summary(basic_scores, "Basic Random Forest")
    print_overfitting_summary(tuned_scores, "Tuned Random Forest")

    tuned_row = tuned_scores.iloc[0]
    print("\nTuning summary:")
    print(f"Best parameters: {search.best_params_}")
    print(f"Best CV macro F1: {search.best_score_:.4f}")
    print(f"Final test macro F1: {tuned_row['macro_f1']:.4f}")
    print(f"Final test balanced accuracy: {tuned_row['test_balanced_accuracy']:.4f}")
    print(f"Disordered recall: {tuned_row['disordered_recall']:.4f}")

    print("\nModel comparison:")
    print(build_comparison_table(basic_scores, tuned_scores).to_string(index=False))
    print(f"\nSaved basic scores to {BASIC_SCORES_CSV}")
    print(f"Saved basic confusion matrix figure to {basic_matrix_path}")
    print(f"Saved basic feature importance table to {basic_importance_csv}")
    print(f"Saved basic feature importance figure to {basic_importance_figure}")
    print(f"Saved tuned scores to {TUNED_SCORES_CSV}")
    print(f"Saved tuned confusion matrix figure to {tuned_matrix_path}")
    print(f"Saved tuned feature importance table to {tuned_importance_csv}")
    print(f"Saved tuned feature importance figure to {tuned_importance_figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
