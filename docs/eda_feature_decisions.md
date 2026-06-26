# EDA & Feature Extraction — Design Decisions

This document records the reasoning behind exploratory analysis, feature engineering, and preprocessing choices in `notebooks/01_eda_preprocessing.ipynb`. It is meant for teammates and for the final project write-up.

**Source of truth:** `notebooks/01_eda_preprocessing.ipynb`  
**Outputs:** `outputs/manifest.csv`, `outputs/duplicates.csv`, `outputs/features/`, `outputs/figures/`

---

## Pipeline overview (Sections 1–13)

| Section | Action | Key output |
|---------|--------|------------|
| 1 | Imports, `resolve_project_root()`, class map | — |
| 2 | Verify all 15 class folders under `data/` | — |
| 3 | Build manifest, validate images, **drop MD5 duplicates**, fail on corrupt files | `manifest.csv`, `duplicates.csv` |
| 4 | Class distribution bar chart, random sample gallery | `class_distribution.png`, `class_sample_gallery.png` |
| 5 | Per-image color/luminance stats over full manifest | `color_df` (in memory), profile plots |
| 6a | GLCM texture on 50 images/class sample | `texture_glcm_by_class.png` |
| 6b | Inter-class cosine similarity (image-level z-score → class mean) | `interclass_similarity_heatmap.png` |
| 7 | PCA 2D scatter on color/luminance features | `pca_color_features_scatter.png` |
| 8 | Augmentation preview (not applied to tabular features) | `augmentation_preview.png` |
| 9 | Stratified 80/10/10 split by `class_name` | `split` column in manifest |
| 10 | `load_image`, normalization, augmentation, CNN batch stub | — |
| 11 | Full tabular feature extraction (no augmentation) | — |
| 12 | Save feature CSVs, scaler, correlation heatmap | `image_features.csv`, `scaler.joblib`, etc. |
| 13 | Summary of artifact paths | — |

**Runtime:** ~20–40 minutes on a laptop (dominated by Section 5 color stats and Section 11 GLCM on ~12k images).

---

## 1. Dataset & manifest

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Class order | Fixed 15-class list (`CLASS_NAMES`) | Consistent `label_idx` (0–14) across manifest, plots, and models. |
| Paths in CSV | Relative to `data/` (e.g. `Forest/001.jpg`) | Manifest is shareable; absolute paths differ per machine. Resolve at runtime via `DATA_DIR / relative_path`. |
| Project root | `resolve_project_root()` | Works when run as `python notebooks/01_eda_preprocessing.ipynb` (`__file__`) or from Jupyter (`Path.cwd()`). |
| Expected inventory | 12,000 raw images, 800/class, 256×256 JPEG | Matches published SkyView layout; validation flags deviations before dedup. |
| Image discovery | `class_dir.iterdir()` (flat folders only) | Avoids pulling nested/stray files via `rglob`. |
| Dataset readiness | All 15 `CLASS_NAMES` folders must exist | Stricter than checking two endpoints; catches partial downloads early. |
| Duplicate handling | MD5 hash groups; keep lexicographically first path; **drop extra copies from manifest** | 3 exact duplicate copies in Airport (`457.jpg`, `575.jpg`, `555.jpg` are copies of `050.jpg`, `064.jpg`, `391.jpg`). Final manifest: **11,997** images. Negligible class imbalance (Airport 797 vs 800). Removed paths logged in `outputs/duplicates.csv`. |
| Corrupt / wrong-size / bad extension | Collected in validation report; **pipeline stops** | `RuntimeError` after summary if any integrity issue is found — prevents obscure failures in later sections. |
| Exact duplicates in validation | Do not fail `all_ok` | Inventory mismatch only; duplicates are handled by removal, not rejection. |

### `outputs/duplicates.csv` (audit log)

Written only when duplicates exist. One row per **removed** path:

| Column | Meaning |
|--------|---------|
| `removed_path` | Extra copy dropped from manifest |
| `kept_path` | Canonical path retained (lexicographically first in MD5 group) |
| `md5_hash` | Shared MD5 hex digest |
| `class_name` | Class folder |

---

## 2. EDA visualizations

### 2.1 Class distribution & sample gallery (Section 4)

- **Bar chart:** Confirms ~800/class before dedup (797 for Airport after).
- **Random gallery (3/class, `random_state=42`):** Visual sanity check for labels, resolution, outliers. Skips classes with fewer than 3 images instead of crashing.

### 2.2 Color, brightness & contrast profiles (Section 5)

**Per-image statistics** (`compute_color_luminance_stats`):

| Feature | Definition | Scale |
|---------|------------|-------|
| `mean_r`, `mean_g`, `mean_b` | Channel means | 0–1 (pixels ÷ 255) |
| `std_r`, `std_g`, `std_b` | Channel standard deviations | 0–1 |
| `mean_h`, `mean_s`, `mean_v` | HSV channel means (skimage `rgb2hsv`) | hue 0–1, sat/val 0–1 |
| `brightness` | Alias of `mean_v` (HSV value) | 0–1 |
| `contrast` | Std. dev. of grayscale luminance | 0–1 |

**Why these features?** Aerial scenes differ in overall color (desert vs forest vs water) and local variation (urban texture vs smooth water). Simple moments are fast over all images and interpretable in plots.

**Plots produced:**
- Boxplots per class (`class_color_profiles.png`) — uses `brightness` label for readability.
- Class-mean RGB heatmap (`class_mean_rgb_heatmap.png`).

**Note on `brightness`:** Kept in `color_df` for EDA plots and PCA-adjacent exploration, but **excluded from tabular predictors** (`REDUNDANT_PREDICTOR_COLS`) because it equals `mean_v`.

### 2.3 GLCM texture sample (Section 6a)

#### What is GLCM?

A **Gray-Level Co-occurrence Matrix (GLCM)** counts how often pairs of pixel intensities occur at a given spatial offset. For each gray level *i* and neighbor gray level *j*, the matrix entry GLCM(*i*, *j*) records how many times a pixel with intensity *i* has a neighbor with intensity *j* at distance *d* along direction *θ*.

Intuitively, GLCM captures **local texture** — whether an image patch is smooth, repetitive, or sharply varying — independent of absolute color. Two scenes can share similar mean RGB (Section 5) but differ in spatial structure (e.g. smooth Lake vs textured Forest).


#### Pipeline in `01_eda_preprocessing.ipynb`

```python
gray = (skcolor.rgb2gray(img) * 255).astype(np.uint8)  # 0–255 uint8
glcm = graycomatrix(gray, distances=[1], angles=[0, π/4, π/2, 3π/4],
                    levels=256, symmetric=True, normed=True)
# Each property averaged over the 4 angles → one scalar per image
```

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `distances` | `[1]` | Neighbor is one pixel away (fine-scale texture at 256×256). |
| `angles` | 0°, 45°, 90°, 135° | Four orientations; props are **averaged** so texture is not tied to one direction. |
| `levels` | `256` | Full 8-bit gray range; matches native image resolution. |
| `symmetric` | `True` | Count (*i*→*j*) and (*j*→*i*); orientation-invariant co-occurrence. |
| `normed` | `True` | Matrix entries sum to 1; comparable across images with different content area. |

#### Haralick properties extracted

| Feature | Column | High values suggest | Low values suggest |
|---------|--------|---------------------|-------------------|
| **Contrast** | `glcm_contrast` | Sharp intensity jumps between neighbors; edges, urban grid, rough terrain | Uniform regions (open water, flat sand) |
| **Homogeneity** | `glcm_homogeneity` | Similar neighbor intensities; smooth surfaces | High local variation; fine or chaotic texture |
| **Energy** (ASM) | `glcm_energy` | Repetitive, ordered patterns | More distributed gray-level pairs; diverse texture |
| **Correlation** | `glcm_correlation` | Predictable intensity relationship along offset | Weak linear relationship between paired grays |

These four are standard in remote-sensing and medical texture work. They complement Section 5 color moments: Desert and Beach may overlap in brightness but can separate in contrast/homogeneity; Highway vs Parking may differ in energy (lane/grid repetition).

#### EDA sample vs full extraction

| Stage | Scope | Purpose |
|-------|-------|---------|
| **Section 6a (EDA)** | 50 images/class (`random_state=42`), ~750 total | Violin plots (`texture_glcm_by_class.png`); fast enough for interactive review (~minutes, not 30+). |
| **Section 11 (modeling)** | All ~11,997 manifest images | Same `compute_glcm_features()`; four GLCM columns in `image_features.csv`. |

The EDA sample is **stratified by class** so plots reflect per-class texture distributions, not a single global subsample.

#### Implementation notes

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Grayscale source | `rgb2gray` on uint8-scaled image | Single channel for co-occurrence; consistent with `edge_density` (Sobel on gray). |
| NaN handling | `np.nanmean` on `graycoprops` | **Correlation** can be NaN on near-uniform patches (e.g. calm water); avoids breaking feature CSV. |
| Visualization | Violin plots per class | Shows full distribution shape, not only class means. |

#### How to read `texture_glcm_by_class.png`

- **Water classes** (Lake, River, Beach): often lower contrast, higher homogeneity on smooth areas.
- **Vegetation** (Forest, Agriculture, Grassland): moderate–high contrast; forest can show higher energy where canopy texture is repetitive.
- **Built environment** (City, Highway, Parking, Airport): contrast and energy often elevated from edges, lanes, and structures.
- **Desert / Mountain**: class-dependent; desert smooth areas vs mountain relief can pull contrast in opposite directions within-class (wide violins are expected).

GLCM here is **global per image** (one matrix over the full 256×256 tile), not per-patch. It summarizes overall texture of the aerial tile; it does not localize objects within the frame.

#### Limitations

1. **O(pixels × levels²)** — `levels=256` on 12k images dominates Section 11 runtime.
2. **Rotation** — Averaging four angles helps but is not fully rotation-invariant for strongly directional scenes (e.g. runways).
3. **Color blind** — GLCM sees luminance texture only; chromatic texture (e.g. crop patterns) is partly captured by RGB/histogram features in Section 11.
4. **EDA sample ≠ full data** — Class rankings in violin plots should be confirmed on full features if used in modeling decisions.

Full feature extraction (Section 11) runs the same GLCM code on **every** image; Section 6a is visualization-only.

### 2.4 Inter-class similarity heatmap (Section 6b)

**Problem (v1 — raw centroids):** Cosine similarity on all-positive `[mean_r, mean_g, mean_b, brightness, contrast]` class means produced off-diagonal scores **0.966–1.000** (heatmap uniformly red).

**Problem (v2 — z-score 15 centroids only):** Z-scoring only the 15 class centroids then cosine similarity inflated scores to **±0.98** and produced counterintuitive pairs (e.g. Forest ↔ River ≈ 0.96) because `n_classes ≈ p_features` and the metric captured shared *deviation patterns* across classes, not visual similarity.

**Current approach (v3):**

```python
SIMILARITY_FEATURES = [
    "mean_r", "mean_g", "mean_b",
    "std_r", "std_g", "std_b",
    "mean_s", "contrast",
]
# 1. StandardScaler fit on all images in color_df
# 2. Mean per class in z-scored space
# 3. cosine_similarity between class centroid vectors
```

| Change | Rationale |
|--------|-----------|
| Image-level z-score, then class mean | Stable scaling using ~12k samples; centroids are average standardized profiles. |
| Drop `mean_h` | Hue is circular; per-image mean hue is unstable. |
| Drop `brightness` | Redundant with RGB / `mean_v`. |
| Add RGB stds | Within-class color variation helps separate uniform (water) vs textured (forest) scenes. |

**Interpretation:** Exploratory view of overlap in hand-crafted color/texture space — not a substitute for CNN embeddings. Moderate off-diagonal scores are expected; extreme ±0.98 under v2 was a methodology artifact.

### 2.5 PCA scatter (Section 7)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Features | RGB/HSV means & stds + `contrast` (no `brightness`) | Extended color/luminance space; avoids perfect multicollinearity with `mean_v`. |
| Preprocessing | `StandardScaler` before PCA | Prevents high-variance channels from dominating components. |
| Components | 2 (`random_state=42`) | Visualizable class overlap; informs expectations for linear models. |

### 2.6 Augmentation preview (Section 8)

Demonstrates **intended training-time transforms** (not applied during tabular feature extraction):

| Transform | Parameters | Rationale |
|-----------|------------|-----------|
| Brightness | ×0.8 – ×1.2 | Lighting variation in aerial imagery. |
| Contrast | ×0.8 – ×1.3 | Sensor / atmospheric differences. |
| Rotation | 90°, 180°, ±15° | Orientation prior for overhead scenes. |
| Gaussian noise | σ=15 on uint8 | Sensor noise robustness. |

Preview uses fixed examples; `apply_augmentation()` picks one transform at random per call for CNN training stubs. Skips classes with no images in manifest.

---

## 3. Train / validation / test split (Section 9)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Ratios | 80% / 10% / 10% | ~640 / 80 / 80 per class (797 → ~638/80/79 for Airport). |
| Stratification | By `class_name` | Preserves class balance in each split. |
| `random_state` | 42 | Reproducible across teammates. |
| Input manifest | Post-dedup only (11,997 rows) | No duplicate rows; no special split rules needed. |

Splits are written back to `manifest.csv` before feature extraction so every row carries its `split` label.

---

## 4. Preprocessing helpers (Section 10)

| Function | Behavior | Notes |
|----------|----------|-------|
| `load_image` | RGB uint8, optional resize to 256×256, bilinear | Matches dataset native resolution. |
| `normalize_pixels` | `minmax` → [0,1] or per-image `standard` | CNN stub uses minmax. |
| `to_grayscale` | skimage `rgb2gray` on normalized image | Consistent with feature extraction. |
| `iter_cnn_batches` | Batch loader with optional `augment`, `shuffle`, `split` filter | For downstream CNN notebooks. |

**Tabular features use original (non-augmented) images** so each row maps to one file on disk.

---

## 5. Tabular feature extraction (Section 11)

### 5.1 Feature groups (43 predictors)

| Group | Columns | Count |
|-------|---------|-------|
| Color / luminance | `mean_*`, `std_*`, `mean_h/s/v`, `contrast` | 10 |
| Channel ratios | `ratio_r_g`, `ratio_r_b`, `ratio_g_b` | 3 |
| Dynamic range | `gray.max() - gray.min()` | 1 |
| Histograms | 8 bins × 3 channels (`hist_r_0` … `hist_b_7`), density=True | 24 |
| GLCM | contrast, homogeneity, energy, correlation | 4 |
| Edges | `edge_density` = mean Sobel magnitude on grayscale | 1 |

`brightness` is computed internally but **excluded** from `feature_columns.txt` via `REDUNDANT_PREDICTOR_COLS` (alias of `mean_v`).

### 5.2 Constants

```python
HIST_BINS = 8          # 24 histogram features
REDUNDANT_PREDICTOR_COLS = {"brightness"}
```

### 5.3 Ontology regression targets (not predictors)

Class-derived metadata for future multi-task / regression work:

| Column | Rule |
|--------|------|
| `is_human_made` | Airport, City, Highway, Parking, Port, Railway, Residential |
| `is_water_related` | Beach, Lake, River, Port |
| `is_vegetation_related` | Agriculture, Forest, Grassland |
| `urban_density_proxy` | Ordinal 0–3 (City=3, Residential=2, …; 0 if not urban) |

Saved separately in `regression_targets.csv`. Excluded from `feature_columns.txt` and predictor correlation heatmap.

### 5.4 Modeling hygiene (Section 12)

| Step | Choice | Rationale |
|------|--------|-----------|
| `StandardScaler` | Fit on **train** split only; saved to `scaler.joblib` | Prevents val/test leakage; downstream notebooks can load the same scaler. |
| Correlation heatmap | Subset of 11 interpretable columns | Full 43×43 matrix is hard to read. |
| Missing values | Raise if any predictors are NaN | Broken extraction surfaces immediately. |

**Manifest / feature metadata columns:** `relative_path`, `class_name`, `label_idx`, `split` (no `is_duplicate` after dedup).

---

## 6. Reproducibility checklist

- `random_state=42` for splits, texture sample, PCA, gallery sampling.
- Pin Python **3.14** and `requirements.txt` versions across the team.
- Regenerate outputs after code changes: `python notebooks/01_eda_preprocessing.ipynb`, then commit `outputs/`.
- Do not commit `data/` (gitignored). `outputs/` **is** committed for team review.

---

## 7. Known limitations & future work

1. **Hand-crafted features** miss spatial layout; CNNs expected to outperform linear models on this task.
2. **GLCM at 256 levels** is slow; acceptable for ~12k×256² images but not for larger data without downsampling.
3. **Inter-class similarity** is exploratory color/texture space only; use CNN embeddings for model-driven similarity.
4. **Augmentation** is previewed only; tabular CSV reflects raw images. CNN pipelines should enable `apply_augmentation` during training only.
5. **Dedup** removes only exact MD5 duplicates; near-duplicates (re-crops, JPEG re-encodes) are not detected.

---

## 8. Changelog

| Date | Change |
|------|--------|
| 2025-06 | Initial EDA pipeline (`01_eda_preprocessing.ipynb`). |
| 2025-06 | Inter-class similarity v1 fix: z-score class centroids, drop `brightness`, add hue/saturation. |
| 2025-06 | Code review fixes: fail-fast validation, `resolve_project_root()`, gallery guards, GLCM NaN guard, `scaler.joblib`, 43 predictors (`brightness` excluded). |
| 2025-06 | Inter-class similarity v3: image-level z-score → class mean; drop `mean_h`; add RGB stds. |
| 2025-06 | Duplicate policy: drop 3 extra MD5 copies from manifest (11,997 images); `duplicates.csv` audit log; remove `is_duplicate` column. |
