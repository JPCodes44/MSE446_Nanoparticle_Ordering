"""Project paths and default workflow settings."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
IMAGE_DIR = DATA_DIR / "flat_with_kv_mm_filenames_cropped"

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

METADATA_CSV = DATA_DIR / "dataset_metadata.csv"
FEATURES_CSV = DATA_DIR / "features.csv"
DATASET_SUMMARY_CSV = RESULTS_DIR / "dataset_summary.csv"
PARAMETER_GROUP_COUNTS_CSV = RESULTS_DIR / "parameter_group_counts.csv"
MODEL_SCORES_CSV = RESULTS_DIR / "model_scores.csv"

RANDOM_STATE = 446
TEST_SIZE = 0.25
SAMPLE_N = None
FORCE_REBUILD = False


def get_image_dir() -> Path:
    """Return the required local image directory."""
    return IMAGE_DIR


def ensure_output_dirs() -> None:
    """Create directories used for generated tables and figures."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
