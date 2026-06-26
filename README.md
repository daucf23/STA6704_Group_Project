# STA6704 Group Project — SkyView Aerial Landscape Dataset

Exploratory data analysis and preprocessing for the [SkyView dataset](https://www.kaggle.com/datasets/ankit1743/skyview-an-aerial-landscape-dataset) (15 classes, ~12,000 aerial images → **11,997** unique images after MD5 dedup).

## What’s in git vs local

Raw images (~12,000 JPEGs, hundreds of MB to ~1 GB+) are **too large for GitHub** — each teammate downloads `data/` locally. **Preprocessing outputs** (`outputs/`) are **committed** so everyone can review the same manifest, features, and EDA figures without re-running the full pipeline first.

| Path | Tracked in git? |
|------|-----------------|
| `notebooks/`, `scripts/`, `docs/`, `requirements.txt`, `README.md` | Yes |
| `outputs/` (manifest, features, figures) | **Yes** — shared EDA & preprocessing artifacts |
| `data/` (class image folders) | **No** — download locally |
| `~/Downloads/skyview-*.zip` | **No** — local download cache |

After changing `notebooks/step1.py`, re-run the pipeline and commit updated `outputs/` so the team stays in sync.

## Setup

**Python 3.14** — use the same minor version across the team.

1. **Clone the repo**
   ```bash
   git clone <repo-url>
   cd STA6704_Group_Project
   ```

2. **Create a virtual environment and install dependencies**
   ```bash
   python3.14 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Download the dataset

Everyone uses the same script so images land in the shared project layout: `data/<ClassName>/*.jpg`.

```bash
bash scripts/download_dataset.sh
```

What it does:
1. Downloads `~/Downloads/skyview-an-aerial-landscape-dataset.zip` via the Kaggle API (`curl`)
2. Unzips the archive
3. Moves the 15 class folders into `data/` at the project root

Re-run is a no-op if `data/` already looks complete. To force a fresh download:

```bash
bash scripts/download_dataset.sh --force
```

## Run the EDA notebook

Open and run **`notebooks/01_eda_preprocessing.ipynb`** top to bottom, or run the equivalent script:

```bash
python notebooks/step1.py
```

Both paths expect ~20–40 minutes on a laptop because feature extraction loops over all images in the manifest (~12k after dedup). **Skip this on first clone** if you only need to review committed artifacts in `outputs/`; run it when you change the pipeline or need to refresh features locally.

`step1.py` is the source of truth; sync the notebook from it when needed. Project paths resolve via `resolve_project_root()` (works from the script or Jupyter).

**Design rationale** for EDA plots, feature definitions, and preprocessing choices: [`docs/eda_feature_decisions.md`](docs/eda_feature_decisions.md).

| Section | What it does |
|---------|----------------|
| 1 | Imports, `resolve_project_root()`, class label map |
| 2 | Locate dataset in `data/` (all 15 class folders required) |
| 3 | Build manifest, validate images, drop MD5 duplicates, fail on corrupt files |
| 4 | Class distribution bar chart and sample gallery |
| 5 | Color, brightness, and contrast profiles by class |
| 6 | GLCM texture sample and inter-class similarity heatmap |
| 7 | PCA 2D scatter on color/luminance features |
| 8 | Augmentation preview (brightness, contrast, rotation, noise) |
| 9 | Stratified 80/10/10 train/val/test split (`random_state=42`) |
| 10 | Preprocessing helpers + CNN batch loader stub |
| 11 | Full feature extraction (43 predictors per image) |
| 12 | Save CSVs, `scaler.joblib`, correlation heatmap |
| 13 | Summary of generated artifact paths |

### Generated artifacts (`outputs/` — committed)

These ship with the repo for shared review. Re-run Section 12 / `step1.py` after pipeline changes and commit updates.

| File | Description |
|------|-------------|
| `outputs/manifest.csv` | `relative_path`, `class_name`, `label_idx`, `split` (11,997 rows after dedup) |
| `outputs/duplicates.csv` | Audit log of removed MD5 duplicate copies (`removed_path`, `kept_path`, `md5_hash`, `class_name`) |
| `outputs/features/image_features.csv` | 43 predictors + metadata + ontology regression targets |
| `outputs/features/regression_targets.csv` | Ontology targets only (`is_human_made`, `is_water_related`, `is_vegetation_related`, `urban_density_proxy`) |
| `outputs/features/feature_columns.txt` | Predictor column names for downstream models |
| `outputs/features/scaler.joblib` | `StandardScaler` fit on train split (for tabular models) |
| `outputs/figures/*.png` | EDA plots and feature correlation heatmap |

Paths in CSVs are **relative** to `data/` (e.g. `Forest/001.jpg`). Resolve at runtime:

```python
from pathlib import Path
DATA_DIR = Path("data")  # or PROJECT_ROOT / "data"
full_path = DATA_DIR / row["relative_path"]
```

For modeling, filter by `split` (`train` / `val` / `test`). Load the fitted scaler with `joblib.load("outputs/features/scaler.joblib")`. Exact MD5 duplicates are already removed from the manifest; see `outputs/duplicates.csv` for the 3 dropped Airport copies.

## Class labels (15 categories)

`Agriculture`, `Airport`, `Beach`, `City`, `Desert`, `Forest`, `Grassland`, `Highway`, `Lake`, `Mountain`, `Parking`, `Port`, `Railway`, `Residential`, `River`
