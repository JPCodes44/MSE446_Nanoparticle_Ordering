# Repository Guidelines

## Project Structure & Module Organization

This repository supports an MSE446 machine learning project for classifying ordered vs. disordered palladium nanoparticle SEM crops. The current tracked project documentation is `README.md`; the local image dataset is in `flat_with_kv_mm_filenames_cropped/` and contains 1,000 `.tif` crops. Filenames encode metadata such as voltage, working distance, label, sample, area, magnification, and source id:

`kv-10p0kV__mm-11p3mm__label-ordered__sample-S1__area-no_area__mag-100k__id-100__orig-...tif`

As code is added, keep reusable Python modules in `src/`, exploratory notebooks in `notebooks/`, tests in `tests/`, and generated outputs under ignored paths such as `data/features/` or model artifact files.

## Build, Test, and Development Commands

No package manager, Makefile, or test runner is currently committed. Useful repository checks today:

- `git status --short`: confirm only intended files are changed.
- `find flat_with_kv_mm_filenames_cropped -maxdepth 1 -type f | wc -l`: verify the local crop count.
- `python -m pytest`: use this once tests are added under `tests/`.

When adding Python workflow code, include a `requirements.txt` or `pyproject.toml` and document exact setup commands in `README.md`.

## Coding Style & Naming Conventions

Use Python 3 with 4-space indentation, clear snake_case names for modules, functions, and variables, and PascalCase only for classes. Prefer small, deterministic functions for feature extraction, train/test splitting, and model evaluation. Keep dataset filename parsing structured around the `__`-separated metadata fields rather than ad hoc substring checks.

## Testing Guidelines

Place tests in `tests/` with filenames like `test_feature_extraction.py` or `test_filename_parsing.py`. Use small fixtures or synthetic arrays instead of committing additional image data. Cover filename metadata parsing, label extraction, feature calculations, and any train/test split logic that could leak samples across evaluation sets.

## Commit & Pull Request Guidelines

The current history uses short, imperative commit summaries such as `Initial commit` and `added a gitignroe file`. Continue with concise, action-oriented subjects, for example `Add filename metadata parser`. Pull requests should describe the experiment or workflow change, list validation commands run, note any data assumptions, and include key metrics or plots when model behavior changes.

## Data & Configuration Tips

Large data and generated artifacts are intentionally ignored by `.gitignore`: `.tif`, `.tiff`, `data/features/`, `.npy`, `.pkl`, and `.joblib`. Do not commit raw SEM images, trained models, local `.env` files, virtual environments, or notebook checkpoints. If a result depends on local data paths, make the path configurable and document the expected directory layout.

## Agent Hook Instructions

Before code work, run `scripts/hooks/pre_agent.sh` and review `ontology.md` for reusable functions. After adding or changing Python functions, run `python scripts/hooks/post_agent.py` so the function index is refreshed for the next agent run.
