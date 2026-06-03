# MSE446 Palladium Nanoparticle Ordering

Classical machine learning workflow for classifying cropped SEM micrographs of palladium nanoparticles on carbon as `ordered` or `disordered`.

The first pass uses engineered image features only. Filename metadata is used for parsing labels, auditing the dataset, and group-aware train/test splitting, but not as model input.

## Project Structure

```text
.
├── README.md
├── requirements.txt
├── src/
│   ├── config.py
│   ├── parse_metadata.py
│   ├── extract_features.py
│   ├── train_models.py
│   └── evaluate.py
├── notebooks/
│   ├── 00_template.ipynb
│   ├── 01_dataset_audit.ipynb
│   ├── 02_feature_extraction.ipynb
│   └── 03_modeling_baselines.ipynb
├── data/
│   └── .gitkeep
└── results/
    ├── figures/
    └── .gitkeep
```

## Data Layout

Place the cropped TIFF images locally at:

```text
data/flat_with_kv_mm_filenames_cropped/
```

The code also falls back to the legacy local folder:

```text
flat_with_kv_mm_filenames_cropped/
```

Do not commit raw or cropped `.tif` / `.tiff` files. They are ignored by Git.

Expected filename format:

```text
kv-10p0kV__mm-6p4mm__label-disordered__sample-S5__area-A56__mag-100k__id-33__orig-33-S5-A56-100k-disordered.tif
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run The Workflow

Start Jupyter:

```bash
jupyter notebook
```

Run notebooks in order:

1. `notebooks/01_dataset_audit.ipynb`
2. `notebooks/02_feature_extraction.ipynb`
3. `notebooks/03_modeling_baselines.ipynb`

For a quick feature smoke test, keep this setting in `02_feature_extraction.ipynb`:

```python
SAMPLE_N = 50
FORCE_REBUILD = False
```

When ready to build the full feature table:

```python
SAMPLE_N = None
FORCE_REBUILD = True
```

## Outputs

Generated files are intentionally ignored:

- `data/dataset_metadata.csv`
- `data/features.csv`
- `results/dataset_summary.csv`
- `results/parameter_group_counts.csv`
- `results/model_scores.csv`
- `results/figures/*.png`

## Modeling Notes

The baseline notebook trains:

- `DummyClassifier`
- `LogisticRegression`
- `DecisionTreeClassifier`
- `RandomForestClassifier`
- `KNeighborsClassifier`
- `SVC`

The split uses `sample + area` groups so repeated crops or magnifications from the same area do not leak across train and test sets. Metadata columns such as `kv`, `mm`, `mag`, `sample`, and `area` are excluded from model features.

## Lightweight Checks

```bash
python -m compileall src
python - <<'PY'
from src.config import get_image_dir
from src.parse_metadata import build_metadata_table

metadata = build_metadata_table(get_image_dir())
print(len(metadata))
print(metadata["label"].value_counts().sort_index())
PY
```
