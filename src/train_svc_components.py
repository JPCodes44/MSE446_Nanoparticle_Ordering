"""Train tuned SVC on combined basic + connected-component image features."""

from __future__ import annotations

import json
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

from src.extract_basic_features import BASIC_FEATURE_COLUMNS, TRACKING_COLUMNS
from src.extract_component_features import (
    COMBINED_FEATURES_CSV,
    component_feature_columns,
    zero_component_count,
)
from src.train_logistic_regression import CLASS_LABELS, choose_group_split


RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
SCORES_CSV = RESULTS_DIR / "model_scores_svc_basic_components.csv"
CONFUSION_MATRIX_FIGURE = FIGURES_DIR / "confusion_matrix_svc_basic_components.png"
BEST_PARAMS_JSON = RESULTS_DIR / "svc_basic_components_best_params.json"
COMPARISON_SCORE_FILES = [
    ("DummyClassifier", RESULTS_DIR / "model_scores_dummy.csv"),
    ("LogisticRegression", RESULTS_DIR / "model_scores_logistic_basic.csv"),
    ("DecisionTree tuned", RESULTS_DIR / "model_scores_decision_tree_tuned.csv"),
    ("RandomForest basic", RESULTS_DIR / "model_scores_random_forest_basic.csv"),
    ("RandomForest tuned", RESULTS_DIR / "model_scores_random_forest_tuned.csv"),
    ("GaussianNB", RESULTS_DIR / "model_scores_gaussian_nb_basic.csv"),
    ("KNN basic", RESULTS_DIR / "model_scores_knn_basic.csv"),
    ("KNN tuned", RESULTS_DIR / "model_scores_knn_tuned.csv"),
    ("SVC basic", RESULTS_DIR / "model_scores_svc_basic.csv"),
    ("SVC tuned", RESULTS_DIR / "model_scores_svc_tuned.csv"),
    ("SVC basic+HOG tuned", RESULTS_DIR / "model_scores_svc_basic_hog.csv"),
]
CV_FOLDS = 5
PARAM_GRID = {
    "svc__C": [0.1, 1, 10, 100],
    "svc__kernel": ["linear", "rbf"],
    "svc__gamma": ["scale", "auto", 0.001, 0.01, 0.1],
    "svc__class_weight": ["balanced"],
}
SMALL_PARAM_GRID = {
    "svc__C": [1, 10, 100],
    "svc__kernel": ["rbf"],
    "svc__gamma": ["scale", 0.001, 0.01],
    "svc__class_weight": ["balanced"],
}
USE_SMALL_GRID = False


def load_combined_features(path: str | Path = COMBINED_FEATURES_CSV) -> pd.DataFrame:
    """Load combined basic + component features for SVC training."""
    features_path = Path(path)
    if not features_path.exists():
        raise FileNotFoundError(
            f"Combined feature file not found: {features_path}. "
            "Run python src/extract_component_features.py first."
        )
    features = pd.read_csv(features_path)
    required_columns = set(TRACKING_COLUMNS).union(BASIC_FEATURE_COLUMNS)
    missing = required_columns.difference(features.columns)
    if missing:
        raise ValueError(f"Combined feature table is missing columns: {sorted(missing)}")
    if features.empty:
        raise ValueError("Combined feature table is empty.")
    component_feature_columns(features)
    return features


def feature_columns(features: pd.DataFrame) -> list[str]:
    """Return basic image feature columns plus connected-component columns."""
    return BASIC_FEATURE_COLUMNS + component_feature_columns(features)


def make_model():
    """Create the scaled SVC pipeline for basic + component features."""
    return make_pipeline(StandardScaler(), SVC(class_weight="balanced"))


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
    y_train: pd.Series,
    y_test: pd.Series,
    y_train_pred: np.ndarray,
    y_test_pred: np.ndarray,
    selected_seed: int,
    split_distance: float,
    model,
    best_cv_macro_f1: float,
    best_params: dict[str, object],
    n_features: int,
    n_component_features: int,
    zero_component_images: int,
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
                "model": "svc_basic_components_tuned",
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
                "best_params": repr(best_params),
                "C": params["C"],
                "kernel": params["kernel"],
                "gamma": params["gamma"],
                "class_weight": str(params["class_weight"]),
                "n_features": n_features,
                "n_component_features": n_component_features,
                "zero_component_images": zero_component_images,
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
    output_path: str | Path = CONFUSION_MATRIX_FIGURE,
) -> Path:
    """Save confusion matrix figure for tuned SVC with basic + component features."""
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
    ax.set_title("SVC basic + components")
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


def tune_svc_components(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups: pd.Series,
    use_small_grid: bool = USE_SMALL_GRID,
) -> GridSearchCV:
    """Tune a scaled SVC on the training split only."""
    grid = SMALL_PARAM_GRID if use_small_grid else PARAM_GRID
    search = GridSearchCV(
        estimator=make_model(),
        param_grid=grid,
        scoring="f1_macro",
        cv=GroupKFold(n_splits=cross_validation_folds(groups)),
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train, groups=groups)
    return search


def save_best_params(search: GridSearchCV, output_path: str | Path = BEST_PARAMS_JSON) -> Path:
    """Save best parameters and CV score as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "best_params": search.best_params_,
        "best_cv_macro_f1": float(search.best_score_),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def train_svc_components(
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, str, pd.DataFrame, Path, Path, pd.Series, pd.Series, GridSearchCV]:
    """Train and evaluate tuned SVC on combined basic + component features."""
    train_idx, test_idx, selected_seed, split_distance = choose_group_split(features)

    columns = feature_columns(features)
    component_columns = component_feature_columns(features)
    X = features[columns]
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

    search = tune_svc_components(X_train, y_train, train_groups)
    model = search.best_estimator_
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    scores, report, matrix = evaluate_predictions(
        y_train,
        y_test,
        y_train_pred,
        y_test_pred,
        selected_seed,
        split_distance,
        model,
        best_cv_macro_f1=float(search.best_score_),
        best_params=search.best_params_,
        n_features=len(columns),
        n_component_features=len(component_columns),
        zero_component_images=zero_component_count(features),
    )
    figure_path = save_confusion_matrix_figure(y_test, y_test_pred)
    best_params_path = save_best_params(search)
    return scores, report, matrix, figure_path, best_params_path, y_train, y_test, search


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


def build_comparison_table(component_scores: pd.DataFrame) -> pd.DataFrame:
    """Build a comparison table against prior model score CSVs."""
    rows = [
        row
        for label, path in COMPARISON_SCORE_FILES
        if (row := comparison_row(label, path)) is not None
    ]
    row = component_scores.iloc[0]
    rows.append(
        {
            "model": "SVC basic+components tuned",
            "test_accuracy": row["test_accuracy"],
            "test_balanced_accuracy": row["test_balanced_accuracy"],
            "macro_f1": row["macro_f1"],
            "disordered_recall": row["disordered_recall"],
        }
    )
    return pd.DataFrame(rows)


def print_previous_svc_comparison(scores: pd.DataFrame) -> None:
    """Print headline comparison against the prior tuned SVC if available."""
    previous_path = RESULTS_DIR / "model_scores_svc_tuned.csv"
    current = scores.iloc[0]
    if not previous_path.exists():
        print("\nPrevious tuned SVC scores not found.")
        return

    previous = pd.read_csv(previous_path).iloc[0]
    print("\nTuned SVC basic-only vs basic+components:")
    print(f"basic-only macro F1:              {previous['macro_f1']:.4f}")
    print(f"basic+components macro F1:        {current['macro_f1']:.4f}")
    print(f"basic-only balanced accuracy:     {previous['test_balanced_accuracy']:.4f}")
    print(f"basic+components balanced acc:    {current['test_balanced_accuracy']:.4f}")
    print(f"basic-only disordered recall:     {previous['disordered_recall']:.4f}")
    print(f"basic+components disordered rec:  {current['disordered_recall']:.4f}")


def main() -> int:
    """Run tuned SVC training on combined basic + component features."""
    try:
        features = load_combined_features(COMBINED_FEATURES_CSV)
        scores, report, matrix, figure_path, params_path, y_train, y_test, search = (
            train_svc_components(features)
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    SCORES_CSV.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(SCORES_CSV, index=False)

    print("Tuned SVC with basic + connected-component features")
    print(f"Combined feature table shape: {features.shape}")
    print(f"Feature columns used: {len(feature_columns(features))}")
    print(f"Component feature columns used: {len(component_feature_columns(features))}")
    print(f"Images with zero detected components: {zero_component_count(features)}")
    print("\nSelected split label counts:")
    print("Train:")
    print(y_train.value_counts().sort_index().to_string())
    print("Test:")
    print(y_test.value_counts().sort_index().to_string())
    print("\nMetrics:")
    print(scores.to_string(index=False))
    print("\nPer-class precision/recall/F1:")
    print(report)
    print("Confusion matrix rows=true, columns=predicted:")
    print(matrix.to_string())

    row = scores.iloc[0]
    print("\nTuning summary:")
    print(f"Best parameters: {search.best_params_}")
    print(f"Best CV macro F1: {search.best_score_:.4f}")
    print(f"Final test macro F1: {row['macro_f1']:.4f}")
    print(f"Final test balanced accuracy: {row['test_balanced_accuracy']:.4f}")
    print(f"Disordered recall: {row['disordered_recall']:.4f}")
    print_previous_svc_comparison(scores)
    print("\nModel comparison:")
    print(build_comparison_table(scores).to_string(index=False))
    print(
        "\nNote: connected-component features approximate bright particle/blob "
        "and aggregate structure. Compare basic-only SVC vs basic+components SVC "
        "using macro F1, balanced accuracy, and disordered recall."
    )
    print(f"\nSaved scores to {SCORES_CSV}")
    print(f"Saved confusion matrix figure to {figure_path}")
    print(f"Saved best parameters to {params_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
