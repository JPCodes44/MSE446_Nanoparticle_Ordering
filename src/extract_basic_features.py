"""Extract basic image-derived features for the first real classifier."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import canny
from skimage.measure import shannon_entropy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
IMAGE_DIR = DATA_DIR / "flat_with_kv_mm_filenames_cropped"
METADATA_CSV = DATA_DIR / "dataset_metadata.csv"
FEATURES_CSV = DATA_DIR / "features_basic.csv"
RESIZE_SHAPE = (256, 256)
TRACKING_COLUMNS = [
    "filename",
    "label",
    "sample",
    "area",
    "area_group",
    "kv",
    "mm",
    "mag",
    "param_group",
]
BASIC_FEATURE_COLUMNS = [
    "mean_intensity",
    "std_intensity",
    "min_intensity",
    "max_intensity",
    "p10_intensity",
    "p25_intensity",
    "p50_intensity",
    "p75_intensity",
    "p90_intensity",
    "entropy",
    "bright_pixel_ratio",
    "edge_density",
]


def load_metadata(path: str | Path = METADATA_CSV) -> pd.DataFrame:
    """Load metadata needed for basic feature extraction."""
    metadata_path = Path(path)
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}. Run python src/parse_metadata.py first."
        )
    metadata = pd.read_csv(metadata_path)
    required_columns = set(TRACKING_COLUMNS).union({"path"})
    missing = required_columns.difference(metadata.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")
    if metadata.empty:
        raise ValueError("Metadata table is empty.")
    return metadata


def resolve_image_path(row: pd.Series, image_dir: str | Path = IMAGE_DIR) -> Path:
    """Find the image path from metadata path, falling back to image_dir/filename."""
    metadata_path = Path(str(row["path"]))
    if metadata_path.exists():
        return metadata_path

    fallback_path = Path(image_dir) / str(row["filename"])
    if fallback_path.exists():
        return fallback_path

    raise FileNotFoundError(
        f"Image not found for {row['filename']}. Checked {metadata_path} and {fallback_path}."
    )


def load_resized_grayscale(path: str | Path, size: tuple[int, int] = RESIZE_SHAPE) -> np.ndarray:
    """Load an image as normalized grayscale after resizing to a fixed shape."""
    try:
        with Image.open(path) as image:
            grayscale = image.convert("L").resize(size, Image.Resampling.BILINEAR)
            array = np.asarray(grayscale, dtype=np.float32) / 255.0
    except OSError as exc:
        raise OSError(f"Could not read image {path}: {exc}") from exc
    return array


def extract_basic_image_features(image: np.ndarray) -> dict[str, float]:
    """Compute basic intensity, entropy, bright-pixel, and edge features."""
    p10, p25, p50, p75, p90 = np.percentile(image, [10, 25, 50, 75, 90])
    edges = canny(image, sigma=1.0)
    return {
        "mean_intensity": float(np.mean(image)),
        "std_intensity": float(np.std(image)),
        "min_intensity": float(np.min(image)),
        "max_intensity": float(np.max(image)),
        "p10_intensity": float(p10),
        "p25_intensity": float(p25),
        "p50_intensity": float(p50),
        "p75_intensity": float(p75),
        "p90_intensity": float(p90),
        "entropy": float(shannon_entropy(image)),
        "bright_pixel_ratio": float(np.mean(image > p90)),
        "edge_density": float(np.mean(edges)),
    }


def extract_features_for_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    """Extract basic image features and retain tracking metadata columns."""
    records: list[dict[str, object]] = []
    for _, row in metadata.iterrows():
        image_path = resolve_image_path(row)
        image = load_resized_grayscale(image_path)
        record = {column: row[column] for column in TRACKING_COLUMNS}
        record.update(extract_basic_image_features(image))
        records.append(record)

    if not records:
        raise ValueError("No feature rows were created.")
    return pd.DataFrame(records, columns=TRACKING_COLUMNS + BASIC_FEATURE_COLUMNS)


def save_features(features: pd.DataFrame, output_csv: str | Path = FEATURES_CSV) -> Path:
    """Save basic image features to CSV."""
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    return output_path


def main() -> int:
    """Extract and save basic image-derived features."""
    try:
        metadata = load_metadata(METADATA_CSV)
        features = extract_features_for_metadata(metadata)
        output_path = save_features(features, FEATURES_CSV)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Extracted basic features for {len(features)} images.")
    print(f"Feature columns: {', '.join(BASIC_FEATURE_COLUMNS)}")
    print(f"Saved features to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
