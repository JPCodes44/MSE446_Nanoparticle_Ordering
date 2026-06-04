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
│   ├── extract_basic_features.py
│   ├── extract_component_features.py
│   ├── extract_hog_features.py
│   ├── extract_features.py
│   ├── train_dummy_baseline.py
│   ├── train_logistic_regression.py
│   ├── train_decision_tree.py
│   ├── train_random_forest.py
│   ├── train_gaussian_nb.py
│   ├── train_knn.py
│   ├── train_svc.py
│   ├── train_svc_components.py
│   ├── train_svc_hog.py
│   └── evaluate.py
├── notebooks/
│   ├── 00_template.ipynb
│   ├── 01_dataset_audit.ipynb
│   ├── 02_feature_extraction.ipynb
│   ├── 03_dummy_baseline.ipynb
│   ├── 04_logistic_regression_basic_features.ipynb
│   ├── 05_decision_tree_basic_features.ipynb
│   ├── 06_random_forest_basic_features.ipynb
│   ├── 07_gaussian_nb_basic_features.ipynb
│   ├── 08_knn_basic_features.ipynb
│   ├── 09_svc_basic_features.ipynb
│   ├── 10_svc_hog_features.ipynb
│   └── 11_svc_component_features.ipynb
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

## Logistic Regression with basic image features

Run:

```bash
python src/parse_metadata.py
python src/train_dummy_baseline.py
python src/extract_basic_features.py
python src/train_logistic_regression.py
python src/train_decision_tree.py
python src/train_random_forest.py
python src/train_gaussian_nb.py
python src/train_knn.py
python src/train_svc.py
python src/extract_hog_features.py
python src/train_svc_hog.py
python src/extract_component_features.py
python src/train_svc_components.py
```

This is the first real supervised model. It uses only basic image-derived features from resized grayscale crops, then trains `LogisticRegression(class_weight="balanced", max_iter=2000)` with `StandardScaler`. Metadata is retained for labels, grouping, and auditing, but metadata columns are not used as predictors.

`train_decision_tree.py` trains an untuned `DecisionTreeClassifier(random_state=42, class_weight="balanced")`, then tunes the same model family with grouped cross-validation on the training split. It prints train-vs-test metrics for both trees to check whether tuning reduces overfitting.

`train_random_forest.py` trains `RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)` on the same basic feature table and grouped split logic. It then tunes only the Random Forest model family with `GridSearchCV` and `GroupKFold` on the training split, using macro F1 as the score. It also saves impurity-based feature importance outputs for both the basic and tuned forests.

`train_gaussian_nb.py` trains `GaussianNB` on the same basic image feature columns and grouped split. Gaussian Naive Bayes is a probability-based baseline that assumes numeric features are conditionally independent and approximately normally distributed within each class, which is probably weak for image features. It is included as a simple course-aligned comparison model.

`train_knn.py` trains a scaled `KNeighborsClassifier` baseline and then tunes only KNN with grouped cross-validation on the training split. KNN is a similarity-based model that predicts from nearby examples in scaled feature space, so `StandardScaler` is required before distance calculations.

`train_svc.py` trains a scaled `SVC(kernel="rbf", class_weight="balanced")` baseline and then tunes only SVC with grouped cross-validation on the training split. SVC finds a maximum-margin decision boundary; the linear kernel tests a linear boundary, while the RBF kernel can model nonlinear boundaries in the scaled feature space.

`extract_hog_features.py` extracts HOG descriptors from resized grayscale crops and combines them with the existing basic image features. `train_svc_hog.py` then tunes SVC on the combined basic + HOG feature table. HOG is an engineered image feature that captures local edge direction and shape structure.

`extract_component_features.py` extracts connected-component features from bright particle-like regions and combines them with the existing basic image features. `train_svc_components.py` then tunes SVC on the combined basic + component feature table. Connected-component features approximate particle/blob and aggregate structure.

## Run The Workflow

Start Jupyter:

```bash
jupyter notebook
```

Run notebooks in order:

1. `notebooks/01_dataset_audit.ipynb`
2. `notebooks/03_dummy_baseline.ipynb`
3. `notebooks/04_logistic_regression_basic_features.ipynb`
4. `notebooks/05_decision_tree_basic_features.ipynb`
5. `notebooks/06_random_forest_basic_features.ipynb`
6. `notebooks/07_gaussian_nb_basic_features.ipynb`
7. `notebooks/08_knn_basic_features.ipynb`
8. `notebooks/09_svc_basic_features.ipynb`
9. `notebooks/10_svc_hog_features.ipynb`
10. `notebooks/11_svc_component_features.ipynb`

## Outputs

Generated files are intentionally ignored:

- `data/dataset_metadata.csv`
- `data/features.csv`
- `data/features_basic.csv`
- `data/features_hog.csv`
- `data/features_basic_hog.csv`
- `data/features_components.csv`
- `data/features_basic_components.csv`
- `results/dataset_summary.csv`
- `results/parameter_group_counts.csv`
- `results/model_scores_dummy.csv`
- `results/model_scores_logistic_basic.csv`
- `results/model_scores_decision_tree_basic.csv`
- `results/model_scores_decision_tree_tuned.csv`
- `results/model_scores_random_forest_basic.csv`
- `results/model_scores_random_forest_tuned.csv`
- `results/model_scores_gaussian_nb_basic.csv`
- `results/model_scores_knn_basic.csv`
- `results/model_scores_knn_tuned.csv`
- `results/model_scores_svc_basic.csv`
- `results/model_scores_svc_tuned.csv`
- `results/model_scores_svc_basic_hog.csv`
- `results/svc_basic_hog_best_params.json`
- `results/model_scores_svc_basic_components.csv`
- `results/svc_basic_components_best_params.json`
- `results/feature_importance_random_forest_basic.csv`
- `results/feature_importance_random_forest_tuned.csv`
- `results/figures/*.png`

## Modeling Notes

The current modeling milestones are `DummyClassifier(strategy="most_frequent")`, Logistic Regression on basic image features, Decision Tree on the same basic features, Random Forest on the same basic features, Gaussian Naive Bayes on the same basic features, KNN on the same basic features, SVC on the same basic features, tuned SVC on combined basic + HOG features, and tuned SVC on combined basic + connected-component features. The Decision Tree, Random Forest, KNN, and SVC experiments include simple grouped hyperparameter tuning. CNNs, graph features, and augmentation are intentionally out of scope until later project steps.

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
