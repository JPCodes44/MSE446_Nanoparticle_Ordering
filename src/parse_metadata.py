"""Parse SEM crop filenames and build dataset audit tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


IMAGE_EXTENSIONS = {".tif", ".tiff"}


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

    fields["kv_value"] = parse_numeric_token(str(fields["kv"]), "kV")
    fields["mm_value"] = parse_numeric_token(str(fields["mm"]), "mm")
    fields["id"] = int(str(fields["id"]))
    fields["group"] = f"{fields['sample']}__{fields['area']}"
    fields["parameter_group"] = (
        f"kv-{fields['kv']}__mm-{fields['mm']}__mag-{fields['mag']}"
    )
    return fields


def list_image_files(image_dir: str | Path) -> list[Path]:
    """List TIFF image files in a directory without recursing."""
    path = Path(image_dir)
    return sorted(
        file_path
        for file_path in path.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_metadata_table(image_dir: str | Path) -> pd.DataFrame:
    """Parse all image filenames in image_dir into a metadata table."""
    image_paths = list_image_files(image_dir)
    records = [parse_filename(path) for path in image_paths]
    columns = [
        "filename",
        "path",
        "kv",
        "kv_value",
        "mm",
        "mm_value",
        "label",
        "sample",
        "area",
        "mag",
        "id",
        "orig",
        "group",
        "parameter_group",
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
