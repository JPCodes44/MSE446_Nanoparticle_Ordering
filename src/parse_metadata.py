"""Parse SEM crop filenames and build dataset audit tables."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


IMAGE_EXTENSIONS = {".tif", ".tiff"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
IMAGE_DIR = DATA_DIR / "flat_with_kv_mm_filenames_cropped"
METADATA_CSV = DATA_DIR / "dataset_metadata.csv"


def parse_numeric_token(value: str, suffix: str) -> float | None:
    """Parse tokens such as '10p0kV' or '6p4mm' into floats."""
    if not value.endswith(suffix):
        return None
    return float(value[: -len(suffix)].replace("p", "."))


def parse_filename(path: str | Path) -> dict[str, object]:
    """Parse a metadata-rich crop filename into structured fields."""
    image_path = Path(path)
    stem = image_path.stem
    fields: dict[str, object] = {
        "filename": image_path.name,
        "path": str(image_path),
    }

    for part in stem.split("__"):
        if "-" not in part:
            continue
        key, value = part.split("-", 1)
        fields[key] = value

    required = ["kv", "mm", "label", "sample", "area", "mag", "id"]
    missing = [key for key in required if key not in fields]
    if missing:
        raise ValueError(f"{image_path.name} is missing metadata fields: {missing}")

    try:
        image_id = int(str(fields["id"]))
    except ValueError as exc:
        raise ValueError(f"{image_path.name} has a non-integer id field") from exc

    metadata = {
        "filename": fields["filename"],
        "path": fields["path"],
        "kv": fields["kv"],
        "mm": fields["mm"],
        "label": fields["label"],
        "sample": fields["sample"],
        "area": fields["area"],
        "mag": fields["mag"],
        "image_id": image_id,
        "original_filename": fields.get("orig", ""),
    }
    metadata["area_group"] = f"{metadata['sample']}__{metadata['area']}"
    metadata["param_group"] = f"{metadata['kv']}__{metadata['mm']}__{metadata['mag']}"
    return metadata


def list_image_files(image_dir: str | Path) -> list[Path]:
    """List TIFF image files in a directory without recursing."""
    path = Path(image_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Image directory does not exist: {path}. "
            "Place cropped TIFFs under data/flat_with_kv_mm_filenames_cropped/."
        )
    return sorted(
        file_path
        for file_path in path.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_metadata_table(image_dir: str | Path) -> pd.DataFrame:
    """Parse all image filenames in image_dir into a metadata table."""
    image_paths = list_image_files(image_dir)
    if not image_paths:
        raise FileNotFoundError(f"No .tif or .tiff files found in {Path(image_dir)}")

    records = []
    for path in image_paths:
        try:
            records.append(parse_filename(path))
        except ValueError as exc:
            raise ValueError(f"Could not parse metadata from {path.name}: {exc}") from exc

    columns = [
        "filename",
        "path",
        "kv",
        "mm",
        "label",
        "sample",
        "area",
        "mag",
        "image_id",
        "original_filename",
        "area_group",
        "param_group",
    ]
    return pd.DataFrame(records).reindex(columns=columns)


def summarize_dataset(metadata: pd.DataFrame) -> pd.DataFrame:
    """Return compact dataset-level counts for audit output."""
    rows = [{"metric": "total_images", "value": int(len(metadata))}]
    for label, count in metadata["label"].value_counts().sort_index().items():
        rows.append({"metric": f"label_{label}", "value": int(count)})
    return pd.DataFrame(rows)


def counts_by_magnification_and_label(metadata: pd.DataFrame) -> pd.DataFrame:
    """Count images by magnification and label."""
    return (
        metadata.groupby(["mag", "label"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["mag", "label"])
    )


def counts_by_parameter_group_and_label(metadata: pd.DataFrame) -> pd.DataFrame:
    """Count images by kv/mm/magnification parameter group and label."""
    return (
        metadata.groupby(["kv", "mm", "mag", "label"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["kv", "mm", "mag", "label"])
    )


def save_metadata(metadata: pd.DataFrame, output_csv: str | Path = METADATA_CSV) -> Path:
    """Save parsed metadata to CSV."""
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(output_path, index=False)
    return output_path


def print_dataset_audit(metadata: pd.DataFrame) -> None:
    """Print the required dataset audit summary."""
    print(f"Total image count: {len(metadata)}")
    print("\nLabel counts:")
    print(metadata["label"].value_counts().sort_index().to_string())
    print("\nCounts by magnification and label:")
    print(counts_by_magnification_and_label(metadata).to_string(index=False))
    print(f"\nUnique area groups: {metadata['area_group'].nunique()}")


def main() -> int:
    """Parse local TIFF filenames, save metadata, and print audit counts."""
    try:
        metadata = build_metadata_table(IMAGE_DIR)
        output_path = save_metadata(metadata, METADATA_CSV)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print_dataset_audit(metadata)
    print(f"\nSaved metadata to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
