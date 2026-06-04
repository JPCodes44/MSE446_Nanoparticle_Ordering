"""Extract HOG features and combine them with existing basic image features."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from skimage.feature import hog


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
METADATA_CSV = DATA_DIR / "dataset_metadata.csv"
BASIC_FEATURES_CSV = DATA_DIR / "features_basic.csv"
HOG_FEATURES_CSV = DATA_DIR / "features_hog.csv"
COMBINED_FEATURES_CSV = DATA_DIR / "features_basic_hog.csv"
HOG_RESIZE_SHAPE = (128, 128)
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_BLOCK_NORM = "L2-Hys"
HOG_TRANSFORM_SQRT = True
FORCE_REBUILD = False
SAMPLE_N: int | None = None


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extract_basic_features import (  # noqa: E402
    BASIC_FEATURE_COLUMNS,
    TRACKING_COLUMNS,
    load_metadata,
    load_resized_grayscale,
    resolve_image_path,
)


def hog_feature_names(n_features: int) -> list[str]:
    """Return stable zero-padded HOG feature column names."""
    return [f"hog_{index:04d}" for index in range(n_features)]


def extract_hog_from_image(image: np.ndarray) -> np.ndarray:
    """Compute one HOG descriptor vector from a normalized grayscale image."""
    return hog(
        image,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm=HOG_BLOCK_NORM,
        transform_sqrt=HOG_TRANSFORM_SQRT,
        feature_vector=True,
    )


def extract_hog_features_for_metadata(
    metadata: pd.DataFrame,
    sample_n: int | None = SAMPLE_N,
) -> pd.DataFrame:
    """Extract HOG features for metadata rows while retaining tracking columns."""
    rows = metadata.head(sample_n) if sample_n is not None else metadata
    records: list[dict[str, object]] = []
    feature_columns: list[str] | None = None

    for row_number, (_, row) in enumerate(rows.iterrows(), start=1):
        image_path = resolve_image_path(row)
        image = load_resized_grayscale(image_path, size=HOG_RESIZE_SHAPE)
        descriptor = extract_hog_from_image(image)

        if feature_columns is None:
            feature_columns = hog_feature_names(len(descriptor))

        record = {column: row[column] for column in TRACKING_COLUMNS}
        record.update(
            {column: float(value) for column, value in zip(feature_columns, descriptor)}
        )
        records.append(record)

        if row_number % 100 == 0:
            print(f"Extracted HOG features for {row_number} images...")

    if not records or feature_columns is None:
        raise ValueError("No HOG feature rows were created.")
    return pd.DataFrame(records, columns=TRACKING_COLUMNS + feature_columns)


def load_or_build_hog_features(
    metadata: pd.DataFrame,
    output_csv: str | Path = HOG_FEATURES_CSV,
    force_rebuild: bool = FORCE_REBUILD,
    sample_n: int | None = SAMPLE_N,
) -> pd.DataFrame:
    """Load cached HOG features or extract and save them."""
    output_path = Path(output_csv)
    if output_path.exists() and not force_rebuild:
        features = pd.read_csv(output_path)
        if features.empty:
            raise ValueError(f"Cached HOG feature file is empty: {output_path}")
        return features

    features = extract_hog_features_for_metadata(metadata, sample_n=sample_n)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    return features


def hog_feature_columns(features: pd.DataFrame) -> list[str]:
    """Return HOG feature columns from a feature table."""
    columns = [column for column in features.columns if column.startswith("hog_")]
    if not columns:
        raise ValueError("No HOG feature columns found.")
    return columns


def load_basic_features(path: str | Path = BASIC_FEATURES_CSV) -> pd.DataFrame:
    """Load the existing basic feature table."""
    features_path = Path(path)
    if not features_path.exists():
        raise FileNotFoundError(
            f"Basic feature file not found: {features_path}. "
            "Run python src/extract_basic_features.py first."
        )
    features = pd.read_csv(features_path)
    required_columns = set(TRACKING_COLUMNS).union(BASIC_FEATURE_COLUMNS)
    missing = required_columns.difference(features.columns)
    if missing:
        raise ValueError(f"Basic feature table is missing columns: {sorted(missing)}")
    return features


def validate_feature_alignment(basic: pd.DataFrame, hog_features: pd.DataFrame) -> None:
    """Validate that basic and HOG tables describe the same labeled images."""
    if basic["filename"].duplicated().any():
        raise ValueError("Basic feature table contains duplicate filenames.")
    if hog_features["filename"].duplicated().any():
        raise ValueError("HOG feature table contains duplicate filenames.")

    basic_keys = set(basic["filename"])
    hog_keys = set(hog_features["filename"])
    if basic_keys != hog_keys:
        missing_hog = sorted(basic_keys.difference(hog_keys))[:5]
        missing_basic = sorted(hog_keys.difference(basic_keys))[:5]
        raise ValueError(
            "Basic and HOG feature tables have different filenames. "
            f"Missing HOG examples: {missing_hog}; missing basic examples: {missing_basic}"
        )

    basic_indexed = basic.set_index("filename").sort_index()
    hog_indexed = hog_features.set_index("filename").sort_index()
    for column in TRACKING_COLUMNS:
        if column == "filename":
            continue
        if not basic_indexed[column].equals(hog_indexed[column]):
            raise ValueError(f"Basic and HOG feature tables disagree on column: {column}")


def combine_basic_and_hog_features(
    basic: pd.DataFrame,
    hog_features: pd.DataFrame,
) -> pd.DataFrame:
    """Combine basic image features with HOG descriptors without duplicating metadata."""
    validate_feature_alignment(basic, hog_features)
    hog_columns = hog_feature_columns(hog_features)
    combined = basic[TRACKING_COLUMNS + BASIC_FEATURE_COLUMNS].merge(
        hog_features[["filename"] + hog_columns],
        on="filename",
        how="inner",
        validate="one_to_one",
    )
    return combined[TRACKING_COLUMNS + BASIC_FEATURE_COLUMNS + hog_columns]


def save_combined_features(
    features: pd.DataFrame,
    output_csv: str | Path = COMBINED_FEATURES_CSV,
) -> Path:
    """Save combined basic + HOG features to CSV."""
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    return output_path


def main() -> int:
    """Extract HOG features, combine with basic features, and save both tables."""
    try:
        metadata = load_metadata(METADATA_CSV)
        hog_features = load_or_build_hog_features(
            metadata,
            HOG_FEATURES_CSV,
            force_rebuild=FORCE_REBUILD,
            sample_n=SAMPLE_N,
        )
        basic_features = load_basic_features(BASIC_FEATURES_CSV)
        combined = combine_basic_and_hog_features(basic_features, hog_features)
        combined_path = save_combined_features(combined, COMBINED_FEATURES_CSV)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"HOG feature table shape: {hog_features.shape}")
    print(f"Combined basic + HOG feature table shape: {combined.shape}")
    print(f"HOG feature columns: {len(hog_feature_columns(hog_features))}")
    print(f"Saved HOG features to {HOG_FEATURES_CSV}")
    print(f"Saved combined features to {combined_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
