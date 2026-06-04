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
│   ├── train_dummy_baseline.py
│   └── evaluate.py
├── notebooks/
│   ├── 00_template.ipynb
│   ├── 01_dataset_audit.ipynb
│   ├── 02_feature_extraction.ipynb
│   └── 03_dummy_baseline.ipynb
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

## Dummy baseline

Run:

```bash
python src/parse_metadata.py
python src/train_dummy_baseline.py
```

This trains only the majority-class `DummyClassifier` baseline. Raw accuracy is expected to be misleading because the dataset is imbalanced, so use balanced accuracy and macro F1 when deciding whether future image-derived models improve on this baseline.

The dummy classifier ignores `X`, so the script uses a one-column placeholder feature matrix. Labels come from parsed filenames, and the train/test split uses `area_group = sample + "__" + area` to avoid repeated-area leakage.

## Run The Workflow

Start Jupyter:

```bash
jupyter notebook
```

Run notebooks in order:

1. `notebooks/01_dataset_audit.ipynb`
2. `notebooks/03_dummy_baseline.ipynb`

## Outputs

Generated files are intentionally ignored:

- `data/dataset_metadata.csv`
- `data/features.csv`
- `results/dataset_summary.csv`
- `results/parameter_group_counts.csv`
- `results/model_scores_dummy.csv`
- `results/figures/*.png`

## Modeling Notes

This milestone trains only `DummyClassifier(strategy="most_frequent")`. Logistic regression, random forests, SVC, KNN, naive Bayes, CNNs, and hyperparameter tuning are intentionally out of scope until later project steps.

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
