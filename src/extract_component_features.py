"""Extract connected-component features and combine them with basic features."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
METADATA_CSV = DATA_DIR / "dataset_metadata.csv"
BASIC_FEATURES_CSV = DATA_DIR / "features_basic.csv"
COMPONENT_FEATURES_CSV = DATA_DIR / "features_components.csv"
COMBINED_FEATURES_CSV = DATA_DIR / "features_basic_components.csv"
RESIZE_SHAPE = (256, 256)
MIN_OBJECT_SIZE = 8
FALLBACK_PERCENTILE = 85
FORCE_REBUILD = False
SAMPLE_N: int | None = None
COMPONENT_FEATURE_COLUMNS = [
    "component_count",
    "component_density",
    "bright_pixel_ratio",
    "mean_component_area",
    "median_component_area",
    "std_component_area",
    "max_component_area",
    "min_component_area",
    "total_component_area",
    "largest_component_area_ratio",
    "mean_eccentricity",
    "std_eccentricity",
    "mean_solidity",
    "std_solidity",
    "mean_perimeter",
    "std_perimeter",
    "mean_major_axis_length",
    "mean_minor_axis_length",
]
COMBINED_COMPONENT_FEATURE_COLUMNS = [
    "component_bright_pixel_ratio" if column == "bright_pixel_ratio" else column
    for column in COMPONENT_FEATURE_COLUMNS
]


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extract_basic_features import (  # noqa: E402
    BASIC_FEATURE_COLUMNS,
    TRACKING_COLUMNS,
    load_metadata,
    load_resized_grayscale,
    resolve_image_path,
)
from src.extract_hog_features import load_basic_features, validate_feature_alignment  # noqa: E402


def bright_particle_mask(image: np.ndarray) -> np.ndarray:
    """Threshold bright particle-like regions and remove tiny objects."""
    try:
        threshold = threshold_otsu(image)
    except ValueError:
        threshold = float(np.percentile(image, FALLBACK_PERCENTILE))

    # Otsu can be degenerate on nearly constant images; fall back to a high percentile.
    if not np.isfinite(threshold) or threshold <= float(np.min(image)):
        threshold = float(np.percentile(image, FALLBACK_PERCENTILE))

    mask = image > threshold
    return remove_small_objects(mask, max_size=MIN_OBJECT_SIZE - 1)


def zero_component_features(image_area: int, bright_pixel_ratio: float) -> dict[str, float]:
    """Return finite component features for images with no labeled components."""
    return {
        "component_count": 0.0,
        "component_density": 0.0,
        "bright_pixel_ratio": bright_pixel_ratio,
        "mean_component_area": 0.0,
        "median_component_area": 0.0,
        "std_component_area": 0.0,
        "max_component_area": 0.0,
        "min_component_area": 0.0,
        "total_component_area": 0.0,
        "largest_component_area_ratio": 0.0,
        "mean_eccentricity": 0.0,
        "std_eccentricity": 0.0,
        "mean_solidity": 0.0,
        "std_solidity": 0.0,
        "mean_perimeter": 0.0,
        "std_perimeter": 0.0,
        "mean_major_axis_length": 0.0,
        "mean_minor_axis_length": 0.0,
    }


def extract_component_image_features(image: np.ndarray) -> dict[str, float]:
    """Compute connected-component area and shape features for one image."""
    image_area = int(image.shape[0] * image.shape[1])
    mask = bright_particle_mask(image)
    bright_pixel_ratio = float(np.mean(mask))
    labeled = label(mask, connectivity=2)
    props = regionprops(labeled)

    if not props:
        return zero_component_features(image_area, bright_pixel_ratio)

    areas = np.asarray([prop.area for prop in props], dtype=float)
    eccentricities = np.asarray([prop.eccentricity for prop in props], dtype=float)
    solidities = np.asarray([prop.solidity for prop in props], dtype=float)
    perimeters = np.asarray([prop.perimeter for prop in props], dtype=float)
    major_axes = np.asarray([prop.axis_major_length for prop in props], dtype=float)
    minor_axes = np.asarray([prop.axis_minor_length for prop in props], dtype=float)
    count = float(len(props))
    max_area = float(np.max(areas))

    return {
        "component_count": count,
        "component_density": count / float(image_area),
        "bright_pixel_ratio": bright_pixel_ratio,
        "mean_component_area": float(np.mean(areas)),
        "median_component_area": float(np.median(areas)),
        "std_component_area": float(np.std(areas)),
        "max_component_area": max_area,
        "min_component_area": float(np.min(areas)),
        "total_component_area": float(np.sum(areas)),
        "largest_component_area_ratio": max_area / float(image_area),
        "mean_eccentricity": float(np.mean(eccentricities)),
        "std_eccentricity": float(np.std(eccentricities)),
        "mean_solidity": float(np.mean(solidities)),
        "std_solidity": float(np.std(solidities)),
        "mean_perimeter": float(np.mean(perimeters)),
        "std_perimeter": float(np.std(perimeters)),
        "mean_major_axis_length": float(np.mean(major_axes)),
        "mean_minor_axis_length": float(np.mean(minor_axes)),
    }


def extract_component_features_for_metadata(
    metadata: pd.DataFrame,
    sample_n: int | None = SAMPLE_N,
) -> pd.DataFrame:
    """Extract component features for metadata rows while retaining tracking columns."""
    rows = metadata.head(sample_n) if sample_n is not None else metadata
    records: list[dict[str, object]] = []

    for row_number, (_, row) in enumerate(rows.iterrows(), start=1):
        image_path = resolve_image_path(row)
        image = load_resized_grayscale(image_path, size=RESIZE_SHAPE)
        record = {column: row[column] for column in TRACKING_COLUMNS}
        record.update(extract_component_image_features(image))
        records.append(record)

        if row_number % 100 == 0:
            print(f"Extracted component features for {row_number} images...")

    if not records:
        raise ValueError("No component feature rows were created.")

    features = pd.DataFrame(records, columns=TRACKING_COLUMNS + COMPONENT_FEATURE_COLUMNS)
    features[COMPONENT_FEATURE_COLUMNS] = features[COMPONENT_FEATURE_COLUMNS].fillna(0.0)
    return features


def load_or_build_component_features(
    metadata: pd.DataFrame,
    output_csv: str | Path = COMPONENT_FEATURES_CSV,
    force_rebuild: bool = FORCE_REBUILD,
    sample_n: int | None = SAMPLE_N,
) -> pd.DataFrame:
    """Load cached component features or extract and save them."""
    output_path = Path(output_csv)
    if output_path.exists() and not force_rebuild:
        features = pd.read_csv(output_path)
        if features.empty:
            raise ValueError(f"Cached component feature file is empty: {output_path}")
        return features

    features = extract_component_features_for_metadata(metadata, sample_n=sample_n)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    return features


def component_feature_columns(features: pd.DataFrame) -> list[str]:
    """Return component feature columns from a feature table."""
    columns = (
        COMBINED_COMPONENT_FEATURE_COLUMNS
        if "component_bright_pixel_ratio" in features.columns
        else COMPONENT_FEATURE_COLUMNS
    )
    missing = [column for column in columns if column not in features.columns]
    if missing:
        raise ValueError(f"Component feature table is missing columns: {missing}")
    return columns.copy()


def zero_component_count(features: pd.DataFrame) -> int:
    """Return how many images have zero detected components."""
    component_feature_columns(features)
    return int((features["component_count"] == 0).sum())


def combine_basic_and_component_features(
    basic: pd.DataFrame,
    component_features: pd.DataFrame,
) -> pd.DataFrame:
    """Combine basic image features with component features without duplicating metadata."""
    validate_feature_alignment(basic, component_features)
    component_columns = COMPONENT_FEATURE_COLUMNS
    renamed_component_features = component_features.rename(
        columns={"bright_pixel_ratio": "component_bright_pixel_ratio"}
    )
    combined = basic[TRACKING_COLUMNS + BASIC_FEATURE_COLUMNS].merge(
        renamed_component_features[["filename"] + COMBINED_COMPONENT_FEATURE_COLUMNS],
        on="filename",
        how="inner",
        validate="one_to_one",
    )
    return combined[TRACKING_COLUMNS + BASIC_FEATURE_COLUMNS + COMBINED_COMPONENT_FEATURE_COLUMNS]


def save_combined_features(
    features: pd.DataFrame,
    output_csv: str | Path = COMBINED_FEATURES_CSV,
) -> Path:
    """Save combined basic + component features to CSV."""
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    return output_path


def main() -> int:
    """Extract component features, combine with basic features, and save both tables."""
    try:
        metadata = load_metadata(METADATA_CSV)
        component_features = load_or_build_component_features(
            metadata,
            COMPONENT_FEATURES_CSV,
            force_rebuild=FORCE_REBUILD,
            sample_n=SAMPLE_N,
        )
        basic_features = load_basic_features(BASIC_FEATURES_CSV)
        combined = combine_basic_and_component_features(basic_features, component_features)
        combined_path = save_combined_features(combined, COMBINED_FEATURES_CSV)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Component feature table shape: {component_features.shape}")
    print(f"Combined basic + component feature table shape: {combined.shape}")
    print(f"Component feature columns: {len(component_feature_columns(component_features))}")
    print(f"Images with zero detected components: {zero_component_count(component_features)}")
    print(f"Saved component features to {COMPONENT_FEATURES_CSV}")
    print(f"Saved combined features to {combined_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
