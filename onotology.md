# Function Ontology

This file is the agent-facing index of reusable code in this repository. The pre-hook prints this file at the start of an agent run so existing functions can be reused before new code is written.

## How To Use

- Check this file before creating new helpers, feature extractors, parsers, model utilities, or plotting code.
- Reuse listed functions when they match the current task.
- Run `python scripts/hooks/post_agent.py` after code changes to refresh the generated function index.

## Function Index

### `src/config.py`

- `get_image_dir` at line 26
  - Signature: `get_image_dir() -> Path`
  - Purpose: Return the required local image directory.
- `ensure_output_dirs` at line 31
  - Signature: `ensure_output_dirs() -> None`
  - Purpose: Create directories used for generated tables and figures.
### `src/evaluate.py`

- `classification_metrics` at line 19
  - Signature: `classification_metrics(model_name: str, y_true, y_pred) -> dict[str, float | str]`
  - Purpose: Compute binary classification metrics for ordered/disordered labels.
- `save_confusion_matrix` at line 37
  - Signature: `save_confusion_matrix(model_name: str, y_true, y_pred, output_dir: str | Path) -> Path`
  - Purpose: Save a confusion matrix plot for one model.
- `save_scores` at line 64
  - Signature: `save_scores(scores: list[dict[str, float | str]], output_csv: str | Path) -> pd.DataFrame`
  - Purpose: Save model score dictionaries as a sorted CSV.
### `src/extract_basic_features.py`

- `load_metadata` at line 48
  - Signature: `load_metadata(path: str | Path = METADATA_CSV) -> pd.DataFrame`
  - Purpose: Load metadata needed for basic feature extraction.
- `resolve_image_path` at line 65
  - Signature: `resolve_image_path(row: pd.Series, image_dir: str | Path = IMAGE_DIR) -> Path`
  - Purpose: Find the image path from metadata path, falling back to image_dir/filename.
- `load_resized_grayscale` at line 80
  - Signature: `load_resized_grayscale(path: str | Path, size: tuple[int, int] = RESIZE_SHAPE) -> np.ndarray`
  - Purpose: Load an image as normalized grayscale after resizing to a fixed shape.
- `extract_basic_image_features` at line 91
  - Signature: `extract_basic_image_features(image: np.ndarray) -> dict[str, float]`
  - Purpose: Compute basic intensity, entropy, bright-pixel, and edge features.
- `extract_features_for_metadata` at line 111
  - Signature: `extract_features_for_metadata(metadata: pd.DataFrame) -> pd.DataFrame`
  - Purpose: Extract basic image features and retain tracking metadata columns.
- `save_features` at line 126
  - Signature: `save_features(features: pd.DataFrame, output_csv: str | Path = FEATURES_CSV) -> Path`
  - Purpose: Save basic image features to CSV.
- `main` at line 134
  - Signature: `main() -> int`
  - Purpose: Extract and save basic image-derived features.
### `src/extract_features.py`

- `load_grayscale_image` at line 19
  - Signature: `load_grayscale_image(path: str | Path) -> np.ndarray`
  - Purpose: Load an image as a normalized float grayscale array.
- `basic_intensity_features` at line 29
  - Signature: `basic_intensity_features(image: np.ndarray) -> dict[str, float]`
  - Purpose: Compute summary statistics from normalized image intensity.
- `edge_density_feature` at line 44
  - Signature: `edge_density_feature(image: np.ndarray) -> dict[str, float]`
  - Purpose: Compute the fraction of pixels identified as Canny edges.
- `hog_features` at line 50
  - Signature: `hog_features(image: np.ndarray, resize_shape: tuple[int, int] = (128, 128), orientations: int = 9, pixels_per_cell: tuple[int, int] = (16, 16), cells_per_block: tuple[int, int] = (2, 2)) -> dict[str, float]`
  - Purpose: Compute HOG descriptor values after resizing to a stable shape.
- `extract_image_features` at line 75
  - Signature: `extract_image_features(path: str | Path) -> dict[str, float]`
  - Purpose: Extract all engineered features from one image.
- `extract_features_for_metadata` at line 85
  - Signature: `extract_features_for_metadata(metadata: pd.DataFrame, sample_n: int | None = None) -> pd.DataFrame`
  - Purpose: Extract image-derived features and keep metadata columns for splitting.
- `load_or_build_features` at line 110
  - Signature: `load_or_build_features(metadata: pd.DataFrame, output_csv: str | Path, force_rebuild: bool = False, sample_n: int | None = None) -> pd.DataFrame`
  - Purpose: Load cached features or compute and save them.
### `src/parse_metadata.py`

- `parse_numeric_token` at line 18
  - Signature: `parse_numeric_token(value: str, suffix: str) -> float | None`
  - Purpose: Parse tokens such as '10p0kV' or '6p4mm' into floats.
- `parse_filename` at line 25
  - Signature: `parse_filename(path: str | Path) -> dict[str, object]`
  - Purpose: Parse a metadata-rich crop filename into structured fields.
- `list_image_files` at line 67
  - Signature: `list_image_files(image_dir: str | Path) -> list[Path]`
  - Purpose: List TIFF image files in a directory without recursing.
- `build_metadata_table` at line 82
  - Signature: `build_metadata_table(image_dir: str | Path) -> pd.DataFrame`
  - Purpose: Parse all image filenames in image_dir into a metadata table.
- `summarize_dataset` at line 112
  - Signature: `summarize_dataset(metadata: pd.DataFrame) -> pd.DataFrame`
  - Purpose: Return compact dataset-level counts for audit output.
- `counts_by_magnification_and_label` at line 120
  - Signature: `counts_by_magnification_and_label(metadata: pd.DataFrame) -> pd.DataFrame`
  - Purpose: Count images by magnification and label.
- `counts_by_parameter_group_and_label` at line 130
  - Signature: `counts_by_parameter_group_and_label(metadata: pd.DataFrame) -> pd.DataFrame`
  - Purpose: Count images by kv/mm/magnification parameter group and label.
- `save_metadata` at line 140
  - Signature: `save_metadata(metadata: pd.DataFrame, output_csv: str | Path = METADATA_CSV) -> Path`
  - Purpose: Save parsed metadata to CSV.
- `print_dataset_audit` at line 148
  - Signature: `print_dataset_audit(metadata: pd.DataFrame) -> None`
  - Purpose: Print the required dataset audit summary.
- `main` at line 158
  - Signature: `main() -> int`
  - Purpose: Parse local TIFF filenames, save metadata, and print audit counts.
### `src/train_decision_tree.py`

- `make_model` at line 55
  - Signature: `make_model() -> DecisionTreeClassifier`
  - Purpose: Create the decision tree baseline model.
- `score_set` at line 60
  - Signature: `score_set(y_true: pd.Series, y_pred) -> dict[str, float]`
  - Purpose: Compute core metrics for one split.
- `evaluate_predictions` at line 73
  - Signature: `evaluate_predictions(model_name: str, y_train: pd.Series, y_test: pd.Series, y_train_pred, y_test_pred, selected_seed: int, split_distance: float, best_cv_macro_f1: float | None = None, best_params: dict[str, object] | None = None) -> tuple[pd.DataFrame, str, pd.DataFrame]`
  - Purpose: Compute train/test metrics, per-class report, and confusion matrix.
- `save_confusion_matrix_figure` at line 133
  - Signature: `save_confusion_matrix_figure(y_test: pd.Series, y_test_pred, output_path: str | Path = CONFUSION_MATRIX_FIGURE, title: str = 'Decision tree basic') -> Path`
  - Purpose: Save a confusion matrix figure for the decision tree baseline.
- `tune_decision_tree` at line 159
  - Signature: `tune_decision_tree(X_train: pd.DataFrame, y_train: pd.Series, groups: pd.Series) -> GridSearchCV`
  - Purpose: Tune tree hyperparameters with group-aware cross-validation.
- `evaluate_model` at line 173
  - Signature: `evaluate_model(model: DecisionTreeClassifier, model_name: str, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, selected_seed: int, split_distance: float, figure_path: str | Path, figure_title: str, best_cv_macro_f1: float | None = None, best_params: dict[str, object] | None = None) -> tuple[pd.DataFrame, str, pd.DataFrame, Path]`
  - Purpose: Fit a tree model and evaluate it on train and held-out test rows.
- `train_decision_tree` at line 209
  - Signature: `train_decision_tree(features: pd.DataFrame) -> tuple[pd.DataFrame, str, pd.DataFrame, Path, pd.DataFrame, str, pd.DataFrame, Path, pd.Series, pd.Series, GridSearchCV]`
  - Purpose: Train untuned and tuned DecisionTreeClassifier baselines.
- `print_overfitting_summary` at line 280
  - Signature: `print_overfitting_summary(scores: pd.DataFrame, label: str) -> None`
  - Purpose: Print train-vs-test metrics for quick overfitting inspection.
- `main` at line 292
  - Signature: `main() -> int`
  - Purpose: Run DecisionTreeClassifier training on basic image features.
### `src/train_dummy_baseline.py`

- `load_metadata` at line 40
  - Signature: `load_metadata(path: str | Path = METADATA_CSV) -> pd.DataFrame`
  - Purpose: Load parsed metadata for dummy baseline training.
- `make_placeholder_features` at line 57
  - Signature: `make_placeholder_features(n_samples: int) -> np.ndarray`
  - Purpose: Create placeholder features because DummyClassifier ignores X.
- `split_group_aware` at line 62
  - Signature: `split_group_aware(metadata: pd.DataFrame, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE) -> tuple[np.ndarray, np.ndarray]`
  - Purpose: Split rows while keeping sample-area groups out of both train and test.
- `evaluate_dummy_baseline` at line 85
  - Signature: `evaluate_dummy_baseline(y_train: pd.Series, y_test: pd.Series, y_train_pred: np.ndarray, y_test_pred: np.ndarray) -> tuple[pd.DataFrame, str, np.ndarray]`
  - Purpose: Compute aggregate, per-class, and confusion-matrix metrics.
- `save_confusion_matrix_figure` at line 117
  - Signature: `save_confusion_matrix_figure(y_test: pd.Series, y_test_pred: np.ndarray, output_path: str | Path = CONFUSION_MATRIX_FIGURE) -> Path`
  - Purpose: Save a confusion matrix figure for the dummy baseline.
- `train_dummy_baseline` at line 142
  - Signature: `train_dummy_baseline(metadata: pd.DataFrame) -> tuple[pd.DataFrame, str, np.ndarray, Path]`
  - Purpose: Train and evaluate the majority-class dummy baseline.
- `main` at line 168
  - Signature: `main() -> int`
  - Purpose: Run the dummy baseline script.
### `src/train_logistic_regression.py`

- `load_features` at line 48
  - Signature: `load_features(path: str | Path = FEATURES_CSV) -> pd.DataFrame`
  - Purpose: Load basic feature table for logistic regression.
- `label_proportions` at line 66
  - Signature: `label_proportions(y: pd.Series) -> pd.Series`
  - Purpose: Return label proportions indexed by known labels.
- `split_distribution_distance` at line 71
  - Signature: `split_distribution_distance(full_y: pd.Series, test_y: pd.Series) -> float`
  - Purpose: Measure how close the test label distribution is to the full dataset.
- `choose_group_split` at line 76
  - Signature: `choose_group_split(features: pd.DataFrame, test_size: float = TEST_SIZE, seeds = SEED_RANGE) -> tuple[np.ndarray, np.ndarray, int, float]`
  - Purpose: Choose a group-aware split with test labels closest to full labels.
- `make_model` at line 113
  - Signature: `make_model()`
  - Purpose: Create the scaled Logistic Regression pipeline.
- `evaluate_predictions` at line 121
  - Signature: `evaluate_predictions(y_train: pd.Series, y_test: pd.Series, y_train_pred: np.ndarray, y_test_pred: np.ndarray, selected_seed: int, split_distance: float) -> tuple[pd.DataFrame, str, np.ndarray]`
  - Purpose: Compute aggregate, per-class, and confusion-matrix metrics.
- `save_confusion_matrix_figure` at line 164
  - Signature: `save_confusion_matrix_figure(y_test: pd.Series, y_test_pred: np.ndarray, output_path: str | Path = CONFUSION_MATRIX_FIGURE) -> Path`
  - Purpose: Save a confusion matrix figure for logistic regression.
- `train_logistic_regression` at line 189
  - Signature: `train_logistic_regression(features: pd.DataFrame) -> tuple[pd.DataFrame, str, np.ndarray, Path, pd.Series, pd.Series]`
  - Purpose: Train and evaluate Logistic Regression on basic image features.
- `load_dummy_scores` at line 219
  - Signature: `load_dummy_scores(path: str | Path = DUMMY_SCORES_CSV) -> pd.DataFrame | None`
  - Purpose: Load dummy baseline scores when available.
- `print_dummy_comparison` at line 227
  - Signature: `print_dummy_comparison(logistic_scores: pd.DataFrame) -> None`
  - Purpose: Print dummy-vs-logistic headline metrics.
- `main` at line 254
  - Signature: `main() -> int`
  - Purpose: Run Logistic Regression training on basic image features.
### `src/train_models.py`

- `feature_columns` at line 34
  - Signature: `feature_columns(features: pd.DataFrame) -> list[str]`
  - Purpose: Return numeric image-derived feature columns.
- `make_group_labels` at line 40
  - Signature: `make_group_labels(features: pd.DataFrame) -> pd.Series`
  - Purpose: Build sample+area group labels for leakage-aware splitting.
- `split_train_test` at line 47
  - Signature: `split_train_test(features: pd.DataFrame, test_size: float = 0.25, random_state: int = 446) -> tuple[pd.Index, pd.Index]`
  - Purpose: Create a train/test split that keeps repeated sample-area groups together.
- `make_baseline_models` at line 70
  - Signature: `make_baseline_models(random_state: int = 446) -> dict[str, object]`
  - Purpose: Create baseline classifiers with comments captured in notebook markdown.
- `train_and_evaluate_models` at line 89
  - Signature: `train_and_evaluate_models(features: pd.DataFrame, figures_dir: str | Path, scores_csv: str | Path, random_state: int = 446) -> pd.DataFrame`
  - Purpose: Train all baseline models and save scores plus confusion matrices.
### `src/train_random_forest.py`

- `make_model` at line 49
  - Signature: `make_model() -> RandomForestClassifier`
  - Purpose: Create the random forest baseline model.
- `score_set` at line 61
  - Signature: `score_set(y_true: pd.Series, y_pred) -> dict[str, float]`
  - Purpose: Compute core metrics for one split.
- `evaluate_predictions` at line 74
  - Signature: `evaluate_predictions(y_train: pd.Series, y_test: pd.Series, y_train_pred, y_test_pred, selected_seed: int, split_distance: float) -> tuple[pd.DataFrame, str, pd.DataFrame]`
  - Purpose: Compute train/test metrics, per-class report, and confusion matrix.
- `save_confusion_matrix_figure` at line 132
  - Signature: `save_confusion_matrix_figure(y_test: pd.Series, y_test_pred, output_path: str | Path = CONFUSION_MATRIX_FIGURE) -> Path`
  - Purpose: Save a confusion matrix figure for the random forest baseline.
- `build_feature_importance_table` at line 157
  - Signature: `build_feature_importance_table(model: RandomForestClassifier) -> pd.DataFrame`
  - Purpose: Return impurity-based feature importances sorted descending.
- `save_feature_importance_outputs` at line 171
  - Signature: `save_feature_importance_outputs(importance: pd.DataFrame, output_csv: str | Path = FEATURE_IMPORTANCE_CSV, output_figure: str | Path = FEATURE_IMPORTANCE_FIGURE) -> tuple[Path, Path]`
  - Purpose: Save feature importance table and plot.
- `train_random_forest` at line 195
  - Signature: `train_random_forest(features: pd.DataFrame) -> tuple[pd.DataFrame, str, pd.DataFrame, Path, pd.DataFrame, Path, Path, pd.Series, pd.Series]`
  - Purpose: Train and evaluate RandomForestClassifier on basic image features.
- `comparison_row` at line 227
  - Signature: `comparison_row(label: str, path: Path) -> dict[str, object] | None`
  - Purpose: Load one comparison row from a score CSV if available.
- `build_comparison_table` at line 247
  - Signature: `build_comparison_table(random_forest_scores: pd.DataFrame) -> pd.DataFrame`
  - Purpose: Build comparison table against prior model score CSVs.
- `main` at line 267
  - Signature: `main() -> int`
  - Purpose: Run RandomForestClassifier training on basic image features.
