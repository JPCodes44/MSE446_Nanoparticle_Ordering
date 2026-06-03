"""Classical ML baselines for engineered SEM image features."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from src.evaluate import classification_metrics, save_confusion_matrix, save_scores


NON_FEATURE_COLUMNS = {
    "filename",
    "path",
    "label",
    "sample",
    "area",
    "group",
    "kv",
    "mm",
    "mag",
}


def feature_columns(features: pd.DataFrame) -> list[str]:
    """Return numeric image-derived feature columns."""
    candidate_columns = [column for column in features.columns if column not in NON_FEATURE_COLUMNS]
    return features[candidate_columns].select_dtypes("number").columns.tolist()


def make_group_labels(features: pd.DataFrame) -> pd.Series:
    """Build sample+area group labels for leakage-aware splitting."""
    if "group" in features.columns:
        return features["group"].astype(str)
    return features["sample"].astype(str) + "__" + features["area"].astype(str)


def split_train_test(
    features: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 446,
) -> tuple[pd.Index, pd.Index]:
    """Create a train/test split that keeps repeated sample-area groups together."""
    y = features["label"]
    groups = make_group_labels(features)

    try:
        splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=random_state)
        train_idx, test_idx = next(splitter.split(features, y, groups))
    except ValueError:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_state,
        )
        train_idx, test_idx = next(splitter.split(features, y, groups))

    return features.index[train_idx], features.index[test_idx]


def make_baseline_models(random_state: int = 446) -> dict[str, object]:
    """Create baseline classifiers with comments captured in notebook markdown."""
    return {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=random_state),
        ),
        "decision_tree": DecisionTreeClassifier(random_state=random_state),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
        ),
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "svc": make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=random_state)),
    }


def train_and_evaluate_models(
    features: pd.DataFrame,
    figures_dir: str | Path,
    scores_csv: str | Path,
    random_state: int = 446,
) -> pd.DataFrame:
    """Train all baseline models and save scores plus confusion matrices."""
    columns = feature_columns(features)
    train_idx, test_idx = split_train_test(features, random_state=random_state)

    X_train = features.loc[train_idx, columns]
    X_test = features.loc[test_idx, columns]
    y_train = features.loc[train_idx, "label"]
    y_test = features.loc[test_idx, "label"]

    scores: list[dict[str, float | str]] = []
    for model_name, model in make_baseline_models(random_state=random_state).items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores.append(classification_metrics(model_name, y_test, y_pred))
        save_confusion_matrix(model_name, y_test, y_pred, figures_dir)

    return save_scores(scores, scores_csv)
