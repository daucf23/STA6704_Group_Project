#!/usr/bin/env python3
"""Apply cleanup edits to notebooks/01_eda_preprocessing.ipynb."""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "01_eda_preprocessing.ipynb"


def set_cell_source(cells: list, idx: int, source: str, *, cell_type: str | None = None) -> None:
    cells[idx]["source"] = source.splitlines(keepends=True)
    if (cell_type or cells[idx]["cell_type"]) == "code":
        cells[idx]["outputs"] = []
        cells[idx]["execution_count"] = None
    else:
        cells[idx].pop("execution_count", None)
        cells[idx].pop("outputs", None)


def main() -> None:
    with NOTEBOOK.open() as f:
        nb = json.load(f)
    cells = nb["cells"]

    # Cell 2 — ontology column constant
    src = "".join(cells[2]["source"])
    src = src.replace(
        'VALID_EXTENSIONS = {".jpg", ".jpeg"}\n\nprint(f"Project root:',
        'VALID_EXTENSIONS = {".jpg", ".jpeg"}\n\n'
        "# Class-derived columns used as regression targets (not model inputs)\n"
        "ONTOLOGY_COLS = [\n"
        '    "is_human_made",\n'
        '    "is_water_related",\n'
        '    "is_vegetation_related",\n'
        '    "urban_density_proxy",\n'
        "]\n\n"
        'print(f"Project root:',
    )
    set_cell_source(cells, 2, src)

    # Cell 8 — duplicate summary
    src = "".join(cells[8]["source"])
    src = src.replace(
        'print(f"Duplicate groups: {len(validation[\'duplicate_hashes\'])}")\n'
        'print(f"Overall:          {\'PASS\' if validation[\'all_ok\'] else \'ISSUES FOUND\'}")\n'
        "print()",
        "duplicate_paths = {\n"
        "    p\n"
        "    for group in validation[\"duplicate_hashes\"]\n"
        "    for p in sorted(group[\"paths\"])[1:]  # keep first copy per hash\n"
        "}\n"
        'print(f"Duplicate groups: {len(validation[\'duplicate_hashes\'])}")\n'
        'print(f"Duplicate images: {len(duplicate_paths)} (flagged in manifest as is_duplicate=1)")\n'
        'print(f"Overall:          {\'PASS\' if validation[\'all_ok\'] else \'ISSUES FOUND\'}")\n'
        "if duplicate_paths:\n"
        '    print("Note: exact duplicates do not fail validation; exclude is_duplicate=1 for modeling.")\n'
        "print()",
    )
    set_cell_source(cells, 8, src)

    # Cell 9 — is_duplicate column before save
    src = "".join(cells[9]["source"])
    src = src.replace(
        "# Save manifest (relative paths only — no absolute DATA_DIR paths)\n"
        "manifest.to_csv(MANIFEST_PATH, index=False)",
        "# Flag duplicate images (later copy in each MD5 group)\n"
        'manifest["is_duplicate"] = manifest["relative_path"].isin(duplicate_paths).astype(int)\n\n'
        "# Save manifest (relative paths only — no absolute DATA_DIR paths)\n"
        "manifest.to_csv(MANIFEST_PATH, index=False)",
    )
    set_cell_source(cells, 9, src)

    # Cell 28 — CNN stub markdown
    set_cell_source(
        cells,
        28,
        "## Section 10 — Preprocessing helpers\n\n"
        "Reusable image-loading and normalization functions (later moved to `src/preprocessing.py`).\n\n"
        "### 10b — CNN data loader stub (follow-up notebook)\n\n"
        "The CNN notebook will read `manifest.csv` splits, load images with `load_image`, "
        "apply `apply_augmentation` on the train split only, and feed `normalize_pixels` tensors "
        "into a PyTorch or Keras pipeline. Tabular feature extraction intentionally skips "
        "augmentation so splits stay comparable.\n",
    )

    # Cell 29 — CNN batch iterator stub
    src = "".join(cells[29]["source"])
    cnn_stub = '''

def iter_cnn_batches(
    df: pd.DataFrame,
    batch_size: int = 32,
    augment: bool = False,
    seed: int | None = None,
):
    """Yield (batch_images, batch_labels) for CNN training loops (stub)."""
    rng = np.random.default_rng(seed)
    rows = df.to_dict("records")
    if augment:
        rng.shuffle(rows)
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        images, labels = [], []
        for row in batch_rows:
            img = load_image(row["relative_path"])
            if augment:
                img = apply_augmentation(img, seed=int(rng.integers(0, 2**31 - 1)))
            images.append(normalize_pixels(img))
            labels.append(row["label_idx"])
        yield np.stack(images), np.array(labels)


'''
    if "iter_cnn_batches" not in src:
        src = src.replace("\n\n_sample = load_image", cnn_stub + "\n_sample = load_image")
    src = src.replace(
        'print(f"to_grayscale shape: {to_grayscale(_sample).shape}")\n',
        'print(f"to_grayscale shape: {to_grayscale(_sample).shape}")\n'
        "_batch_x, _batch_y = next(iter_cnn_batches(manifest.head(4), batch_size=2, augment=False))\n"
        'print(f"CNN batch stub: X={_batch_x.shape}, y={_batch_y.shape}")\n',
    )
    set_cell_source(cells, 29, src)

    # Cell 31 — propagate is_duplicate
    src = "".join(cells[31]["source"])
    if '"is_duplicate"' not in src:
        src = src.replace(
            '    feats["split"] = row["split"]\n'
            "    feature_records.append(feats)",
            '    feats["split"] = row["split"]\n'
            '    feats["is_duplicate"] = row["is_duplicate"]\n'
            "    feature_records.append(feats)",
        )
    set_cell_source(cells, 31, src)

    # Cell 33 — separate predictors from regression targets
    set_cell_source(
        cells,
        33,
        "FEATURES_DIR = OUTPUTS_DIR / \"features\"\n"
        "FEATURES_DIR.mkdir(parents=True, exist_ok=True)\n"
        "FEATURES_PATH = FEATURES_DIR / \"image_features.csv\"\n"
        "REGRESSION_TARGETS_PATH = FEATURES_DIR / \"regression_targets.csv\"\n"
        "FEATURE_COLUMNS_PATH = FEATURES_DIR / \"feature_columns.txt\"\n"
        "\n"
        "META_COLS = {\"relative_path\", \"class_name\", \"label_idx\", \"split\", \"is_duplicate\"}\n"
        "feature_cols = [\n"
        "    c for c in features_df.columns if c not in META_COLS and c not in ONTOLOGY_COLS\n"
        "]\n"
        "\n"
        "features_df.to_csv(FEATURES_PATH, index=False)\n"
        "print(f\"Features saved to: {FEATURES_PATH}\")\n"
        'print(f"Rows: {len(features_df):,}  |  Columns: {features_df.shape[1]}")\n'
        'print(f"Predictor columns: {len(feature_cols)}  |  Ontology targets: {len(ONTOLOGY_COLS)}")\n'
        "\n"
        "regression_cols = [\"relative_path\", \"class_name\", \"label_idx\", \"split\", \"is_duplicate\"] + ONTOLOGY_COLS\n"
        "features_df[regression_cols].to_csv(REGRESSION_TARGETS_PATH, index=False)\n"
        'print(f"Regression targets saved to: {REGRESSION_TARGETS_PATH}")\n'
        "\n"
        'FEATURE_COLUMNS_PATH.write_text("\\n".join(feature_cols) + "\\n")\n'
        'print(f"Feature column list saved to: {FEATURE_COLUMNS_PATH}")\n'
        "\n"
        "missing = features_df[feature_cols].isna().sum()\n"
        "missing = missing[missing > 0]\n"
        'print("\\nMissing values (predictors):")\n'
        "if missing.empty:\n"
        '    print("  none")\n'
        "else:\n"
        "    print(missing.to_string())\n"
        "\n"
        'print("\\nOntology target summary (per class):")\n'
        "ontology_summary = (\n"
        "    features_df.groupby(\"class_name\", observed=True)[ONTOLOGY_COLS]\n"
        "    .first()\n"
        "    .sort_index()\n"
        ")\n"
        "print(ontology_summary.to_string())\n"
        "\n"
        "# Correlation heatmap (color + texture subset to keep figure readable)\n"
        "corr_subset = [\n"
        '    "mean_r", "mean_g", "mean_b", "brightness", "contrast", "dynamic_range",\n'
        '    "glcm_contrast", "glcm_homogeneity", "glcm_energy", "glcm_correlation", "edge_density",\n'
        "]\n"
        "corr_subset = [c for c in corr_subset if c in features_df.columns]\n"
        "corr = features_df[corr_subset].corr()\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(10, 8))\n"
        'sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True, ax=ax)\n'
        'ax.set_title("Feature correlation (selected columns)")\n'
        "plt.tight_layout()\n"
        'fig.savefig(FIGURES_DIR / "feature_correlation_heatmap.png", dpi=150, bbox_inches="tight")\n'
        "plt.show()\n"
        'print(f"Saved: {FIGURES_DIR / \'feature_correlation_heatmap.png\'}")\n'
        "\n"
        'train_mask = (features_df["split"] == "train") & (features_df["is_duplicate"] == 0)\n'
        "scaler = StandardScaler()\n"
        "X_train = features_df.loc[train_mask, feature_cols].to_numpy()\n"
        "X_train_scaled = scaler.fit_transform(X_train)\n"
        'print(f"\\nStandardScaler fit on train split (non-duplicate): {X_train.shape[0]:,} rows x {X_train.shape[1]} features")\n'
        'print(f"Scaled train mean (first 5 cols): {X_train_scaled[:, :5].mean(axis=0).round(4)}")\n'
        'print(f"Scaled train std  (first 5 cols): {X_train_scaled[:, :5].std(axis=0).round(4)}")\n',
    )

    # Cell 34 — updated artifact list
    set_cell_source(
        cells,
        34,
        "## Section 13 — Next steps\n\n"
        "Foundation artifacts produced by this notebook:\n\n"
        "- `outputs/manifest.csv` — relative paths, labels, stratified splits, `is_duplicate` flag\n"
        "- `outputs/features/image_features.csv` — 44 predictors + metadata + ontology targets\n"
        "- `outputs/features/regression_targets.csv` — ontology targets only (for regression notebooks)\n"
        "- `outputs/features/feature_columns.txt` — predictor column names for PCA+SVM / GLM / trees\n"
        "- `outputs/figures/` — EDA plots + `feature_correlation_heatmap.png`\n\n"
        "**Follow-up notebooks (not in scope here):**\n\n"
        "1. **PCA + SVM** — read `feature_columns.txt`, fit PCA on train, evaluate on val/test\n"
        "2. **GLM / tree models** — same feature matrix; compare interpretability vs SVM\n"
        "3. **CNN baseline** — use `manifest.csv` splits with `load_image` / `iter_cnn_batches`\n"
        "4. **Robustness sweeps** — systematic brightness/contrast/rotation/noise experiments\n"
        "5. **Regression** — predict ontology columns in `regression_targets.csv` from image features\n",
    )

    # Cell 35 — summary print
    set_cell_source(
        cells,
        35,
        'print("EDA & preprocessing complete.")\n'
        'print(f"  Manifest:           {MANIFEST_PATH}")\n'
        'print(f"  Features:           {FEATURES_PATH}")\n'
        'print(f"  Regression targets: {REGRESSION_TARGETS_PATH}")\n'
        'print(f"  Feature columns:    {FEATURE_COLUMNS_PATH}")\n'
        'print(f"  Figures:            {FIGURES_DIR}")\n',
    )

    # Clear all other code cell outputs for a clean re-run
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "code" and i not in {2, 8, 9, 29, 31, 33, 35}:
            cell["outputs"] = []
            cell["execution_count"] = None

    with NOTEBOOK.open("w") as f:
        json.dump(nb, f, indent=1)
        f.write("\n")

    print(f"Patched {NOTEBOOK}")


if __name__ == "__main__":
    main()
