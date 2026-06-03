"""Engineered image features for cropped SEM micrographs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage import exposure
from skimage.feature import canny, hog
from skimage.measure import shannon_entropy
from skimage.transform import resize


PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)


def load_grayscale_image(path: str | Path) -> np.ndarray:
    """Load an image as a normalized float grayscale array."""
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"), dtype=np.float32)
    max_value = float(array.max())
    if max_value > 0:
        array = array / max_value
    return array


def basic_intensity_features(image: np.ndarray) -> dict[str, float]:
    """Compute summary statistics from normalized image intensity."""
    features = {
        "mean_intensity": float(np.mean(image)),
        "std_intensity": float(np.std(image)),
        "min_intensity": float(np.min(image)),
        "max_intensity": float(np.max(image)),
        "entropy": float(shannon_entropy(image)),
        "bright_pixel_ratio": float(np.mean(image > np.percentile(image, 90))),
    }
    for percentile in PERCENTILES:
        features[f"percentile_{percentile}"] = float(np.percentile(image, percentile))
    return features


def edge_density_feature(image: np.ndarray) -> dict[str, float]:
    """Compute the fraction of pixels identified as Canny edges."""
    edge_mask = canny(image, sigma=1.0)
    return {"edge_density": float(np.mean(edge_mask))}


def hog_features(
    image: np.ndarray,
    resize_shape: tuple[int, int] = (128, 128),
    orientations: int = 9,
    pixels_per_cell: tuple[int, int] = (16, 16),
    cells_per_block: tuple[int, int] = (2, 2),
) -> dict[str, float]:
    """Compute HOG descriptor values after resizing to a stable shape."""
    resized = resize(
        image,
        resize_shape,
        anti_aliasing=True,
        preserve_range=True,
    )
    equalized = exposure.rescale_intensity(resized, out_range=(0.0, 1.0))
    descriptor = hog(
        equalized,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        feature_vector=True,
    )
    return {f"hog_{index:04d}": float(value) for index, value in enumerate(descriptor)}


def extract_image_features(path: str | Path) -> dict[str, float]:
    """Extract all engineered features from one image."""
    image = load_grayscale_image(path)
    features: dict[str, float] = {}
    features.update(basic_intensity_features(image))
    features.update(edge_density_feature(image))
    features.update(hog_features(image))
    return features


def extract_features_for_metadata(
    metadata: pd.DataFrame,
    sample_n: int | None = None,
) -> pd.DataFrame:
    """Extract image-derived features and keep metadata columns for splitting."""
    working = metadata.head(sample_n).copy() if sample_n is not None else metadata.copy()
    records: list[dict[str, object]] = []
    for row in working.itertuples(index=False):
        feature_values = extract_image_features(row.path)
        record = {
            "filename": row.filename,
            "path": row.path,
            "label": row.label,
            "sample": row.sample,
            "area": row.area,
            "group": row.group,
            "kv": row.kv,
            "mm": row.mm,
            "mag": row.mag,
        }
        record.update(feature_values)
        records.append(record)
    return pd.DataFrame(records)


def load_or_build_features(
    metadata: pd.DataFrame,
    output_csv: str | Path,
    force_rebuild: bool = False,
    sample_n: int | None = None,
) -> pd.DataFrame:
    """Load cached features or compute and save them."""
    output_path = Path(output_csv)
    if output_path.exists() and not force_rebuild:
        return pd.read_csv(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features = extract_features_for_metadata(metadata, sample_n=sample_n)
    features.to_csv(output_path, index=False)
    return features
