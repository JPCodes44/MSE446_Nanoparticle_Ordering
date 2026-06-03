# Function Ontology

This file is the agent-facing index of reusable code in this repository. The pre-hook prints this file at the start of an agent run so existing functions can be reused before new code is written.

## How To Use

- Check this file before creating new helpers, feature extractors, parsers, model utilities, or plotting code.
- Reuse listed functions when they match the current task.
- Run `python scripts/hooks/post_agent.py` after code changes to refresh the generated function index.

## Function Index

### `src/config.py`

- `get_image_dir` at line 27
  - Signature: `get_image_dir() -> Path`
  - Purpose: Return the image directory without moving or copying local TIFF files.
- `ensure_output_dirs` at line 36
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

- `parse_numeric_token` at line 13
  - Signature: `parse_numeric_token(value: str, suffix: str) -> float | None`
  - Purpose: Parse tokens such as '10p0kV' or '6p4mm' into floats.
- `parse_filename` at line 20
  - Signature: `parse_filename(path: str | Path) -> dict[str, object]`
  - Purpose: Parse a metadata-rich crop filename into structured fields.
- `list_image_files` at line 50
  - Signature: `list_image_files(image_dir: str | Path) -> list[Path]`
  - Purpose: List TIFF image files in a directory without recursing.
- `build_metadata_table` at line 60
  - Signature: `build_metadata_table(image_dir: str | Path) -> pd.DataFrame`
  - Purpose: Parse all image filenames in image_dir into a metadata table.
- `summarize_dataset` at line 83
  - Signature: `summarize_dataset(metadata: pd.DataFrame) -> pd.DataFrame`
  - Purpose: Return compact dataset-level counts for audit output.
- `counts_by_magnification_and_label` at line 91
  - Signature: `counts_by_magnification_and_label(metadata: pd.DataFrame) -> pd.DataFrame`
  - Purpose: Count images by magnification and label.
- `counts_by_parameter_group_and_label` at line 101
  - Signature: `counts_by_parameter_group_and_label(metadata: pd.DataFrame) -> pd.DataFrame`
  - Purpose: Count images by kv/mm/magnification parameter group and label.
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
