# %% [markdown]
#  # 01 — EDA & Preprocessing: SkyView Dataset
# 
# 
# 
#  This notebook builds a master manifest and runs validation checks on the [SkyView aerial landscape dataset](https://www.kaggle.com/datasets/ankit1743/skyview-an-aerial-landscape-dataset).
# 
# 
# 
#  **Workflow:** clone repo → install deps → `bash scripts/download_dataset.sh` → run this notebook.
# 
# 
# 
#  Outputs are written locally to `outputs/` (gitignored). Manifest paths are **relative** to the dataset root (`data/`); full paths are resolved at runtime via `DATA_DIR`.

# %% [markdown]
#  ## Section 1 — Setup & imports

# %%
import hashlib
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageEnhance
from skimage import color as skcolor
from skimage.feature import graycomatrix, graycoprops
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

import matplotlib

matplotlib.use("Agg")

def resolve_project_root() -> Path:
    """Project root when run as step1.py or from a Jupyter notebook."""
    try:
        here = Path(__file__).resolve().parent
    except NameError:
        here = Path.cwd()
    if here.name == "notebooks":
        return here.parent
    if (here / "notebooks").is_dir():
        return here
    return here


PROJECT_ROOT = resolve_project_root()
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MANIFEST_PATH = OUTPUTS_DIR / "manifest.csv"
DUPLICATES_PATH = OUTPUTS_DIR / "duplicates.csv"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Duplicate policy: MD5-identical files form a group; keep the lexicographically first
# path and drop extra copies from the manifest (~3 in Airport). Removed paths are
# logged to outputs/duplicates.csv for audit. No class imbalance of concern.

# Canonical class order (index 0–14)
CLASS_NAMES = [
    "Agriculture",
    "Airport",
    "Beach",
    "City",
    "Desert",
    "Forest",
    "Grassland",
    "Highway",
    "Lake",
    "Mountain",
    "Parking",
    "Port",
    "Railway",
    "Residential",
    "River",
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
EXPECTED_IMAGES_PER_CLASS = 800
EXPECTED_TOTAL_IMAGES = len(CLASS_NAMES) * EXPECTED_IMAGES_PER_CLASS
EXPECTED_IMAGE_SIZE = (256, 256)
VALID_EXTENSIONS = {".jpg", ".jpeg"}

# Class-derived columns used as regression targets (not model inputs)
ONTOLOGY_COLS = [
    "is_human_made",
    "is_water_related",
    "is_vegetation_related",
    "urban_density_proxy",
]

# Alias of mean_v — kept in color_df for EDA plots, excluded from model predictors
REDUNDANT_PREDICTOR_COLS = {"brightness"}

print(f"Project root: {PROJECT_ROOT}")
print(f"Outputs dir:  {OUTPUTS_DIR}")
print(f"Classes:      {len(CLASS_NAMES)}")


# %% [markdown]
#  ## Section 2 — Locate dataset
# 
# 
# 
#  Images live in `data/<ClassName>/` after everyone runs the shared download script from the project root:
# 
# 
# 
#  ```bash
# 
#  bash scripts/download_dataset.sh
# 
#  ```
# 
# 
# 
#  That script downloads the Kaggle zip to `~/Downloads/`, unzips it, and moves class folders into `data/`.

# %%
DATA_DIR = PROJECT_ROOT / "data"

def dataset_ready(data_dir: Path) -> bool:
    """True when all expected class folders exist under data/."""
    return all((data_dir / class_name).is_dir() for class_name in CLASS_NAMES)


if not dataset_ready(DATA_DIR):
    raise FileNotFoundError(
        f"Dataset not found in {DATA_DIR}.\n"
        "From the project root, run:\n"
        "  bash scripts/download_dataset.sh"
    )

print(f"Dataset root: {DATA_DIR}")
print(f"Exists:       {DATA_DIR.exists()}")

top_level_dirs = sorted([p.name for p in DATA_DIR.iterdir() if p.is_dir()])
print(f"Class folders ({len(top_level_dirs)}): {top_level_dirs}")


# %% [markdown]
#  ## Section 3 — Build manifest & validation checks
# 
# 
# 
#  Scans the dataset, builds a manifest with **relative paths** (safe to share), and validates:
# 
#  - Total count = 12,000 (800 per class)
# 
#  - All images readable; flag corrupt files
# 
#  - Uniform 256×256 resolution
# 
#  - File extension consistency
# 
#  - Exact duplicates (MD5 hash); extra copies dropped from manifest (see Section 3)

# %%
def discover_images(data_dir: Path) -> list[dict]:
    """Walk class folders and collect image metadata with relative paths."""
    records = []
    data_dir = Path(data_dir)

    for class_name in CLASS_NAMES:
        class_dir = data_dir / class_name
        if not class_dir.is_dir():
            print(f"WARNING: missing class folder: {class_name}")
            continue

        for img_path in sorted(class_dir.iterdir()):
            if not img_path.is_file():
                continue
            if img_path.suffix.lower() not in VALID_EXTENSIONS:
                continue

            relative_path = img_path.relative_to(data_dir).as_posix()
            records.append(
                {
                    "relative_path": relative_path,
                    "class_name": class_name,
                    "label_idx": CLASS_TO_IDX[class_name],
                    "split": None,  # assigned later in stratified split section
                }
            )

    return records


records = discover_images(DATA_DIR)
manifest = pd.DataFrame(records)

print(f"Images discovered: {len(manifest):,}")
print(manifest.head())


# %%
def resolve_filepath(relative_path: str) -> Path:
    """Resolve a manifest relative path to an absolute path at runtime."""
    return DATA_DIR / relative_path


def validate_manifest(manifest: pd.DataFrame, data_dir: Path) -> dict:
    """Run inventory checks and return a summary report."""
    report = {
        "total_images": len(manifest),
        "expected_total": EXPECTED_TOTAL_IMAGES,
        "count_ok": len(manifest) == EXPECTED_TOTAL_IMAGES,
        "per_class_counts": manifest["class_name"].value_counts().sort_index().to_dict(),
        "per_class_ok": True,
        "missing_classes": [],
        "unexpected_classes": [],
        "corrupt_files": [],
        "wrong_size": [],
        "invalid_extensions": [],
        "duplicate_hashes": [],
    }

    # Per-class count check
    for class_name in CLASS_NAMES:
        count = report["per_class_counts"].get(class_name, 0)
        if count != EXPECTED_IMAGES_PER_CLASS:
            report["per_class_ok"] = False

    found_classes = set(manifest["class_name"].unique())
    report["missing_classes"] = sorted(set(CLASS_NAMES) - found_classes)
    report["unexpected_classes"] = sorted(found_classes - set(CLASS_NAMES))

    # Per-image checks
    hash_to_paths: dict[str, list[str]] = {}

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Validating images"):
        rel_path = row["relative_path"]
        abs_path = data_dir / rel_path
        ext = Path(rel_path).suffix.lower()

        if ext not in VALID_EXTENSIONS:
            report["invalid_extensions"].append(rel_path)
            continue

        try:
            with Image.open(abs_path) as img:
                img.verify()
            with Image.open(abs_path) as img:
                size = img.size
                if size != EXPECTED_IMAGE_SIZE:
                    report["wrong_size"].append({"path": rel_path, "size": size})
        except Exception as exc:
            report["corrupt_files"].append({"path": rel_path, "error": str(exc)})
            continue

        # MD5 for duplicate detection
        with open(abs_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        hash_to_paths.setdefault(file_hash, []).append(rel_path)

    report["duplicate_hashes"] = [
        {"hash": h, "paths": paths}
        for h, paths in hash_to_paths.items()
        if len(paths) > 1
    ]

    report["all_ok"] = (
        report["count_ok"]
        and report["per_class_ok"]
        and not report["missing_classes"]
        and not report["unexpected_classes"]
        and not report["corrupt_files"]
        and not report["wrong_size"]
        and not report["invalid_extensions"]
    )

    return report


validation = validate_manifest(manifest, DATA_DIR)


# %%
# Print validation summary
print("=" * 60)
print("VALIDATION SUMMARY")
print("=" * 60)
print(f"Total images:     {validation['total_images']:,} / {validation['expected_total']:,}  "
      f"{'OK' if validation['count_ok'] else 'MISMATCH'}")
print(f"Per-class counts: {'OK' if validation['per_class_ok'] else 'MISMATCH'}")
print(f"Missing classes:  {validation['missing_classes'] or 'none'}")
print(f"Unexpected class: {validation['unexpected_classes'] or 'none'}")
print(f"Corrupt files:    {len(validation['corrupt_files'])}")
print(f"Wrong resolution: {len(validation['wrong_size'])}")
print(f"Bad extensions:   {len(validation['invalid_extensions'])}")
duplicate_paths = {
    p
    for group in validation["duplicate_hashes"]
    for p in sorted(group["paths"])[1:]  # keep first copy per hash
}

n_dup_groups = len(validation["duplicate_hashes"])
n_dup_copies = len(duplicate_paths)
print(f"Duplicate groups:  {n_dup_groups}")
print(f"Extra copies:      {n_dup_copies} (removed from manifest; see {DUPLICATES_PATH.name})")
print(f"Overall:           {'PASS' if validation['all_ok'] else 'ISSUES FOUND'}")
if duplicate_paths:
    print("Policy: keep lexicographically first path per MD5 group; drop extra copies.")
print()

if not validation["per_class_ok"]:
    print("Per-class counts:")
    for class_name, count in sorted(validation["per_class_counts"].items()):
        status = "OK" if count == EXPECTED_IMAGES_PER_CLASS else f"expected {EXPECTED_IMAGES_PER_CLASS}"
        print(f"  {class_name:15s} {count:4d}  ({status})")

if validation["corrupt_files"]:
    print("\nCorrupt files (first 5):")
    for item in validation["corrupt_files"][:5]:
        print(f"  {item['path']}: {item['error']}")

if validation["duplicate_hashes"]:
    print(f"\nDuplicate inventory ({n_dup_groups} group(s), {n_dup_copies} extra copy/copies):")
    for group in validation["duplicate_hashes"]:
        paths = sorted(group["paths"])
        canonical = paths[0]
        print(f"  hash {group['hash'][:8]}...  canonical: {canonical}")
        for dup_path in paths[1:]:
            print(f"    extra copy: {dup_path}")

fatal_issues = []
if validation["corrupt_files"]:
    fatal_issues.append(f"{len(validation['corrupt_files'])} corrupt file(s)")
if validation["wrong_size"]:
    fatal_issues.append(f"{len(validation['wrong_size'])} wrong-resolution file(s)")
if validation["invalid_extensions"]:
    fatal_issues.append(f"{len(validation['invalid_extensions'])} invalid extension(s)")
if fatal_issues:
    raise RuntimeError(
        "Validation found image integrity issues; fix the dataset before continuing.\n"
        + "\n".join(f"  - {issue}" for issue in fatal_issues)
    )


# %%
# Drop extra MD5 copies from manifest (keep lexicographically first path per group)
if duplicate_paths:
    removed_records = []
    for group in validation["duplicate_hashes"]:
        paths = sorted(group["paths"])
        kept_path = paths[0]
        for removed_path in paths[1:]:
            removed_records.append(
                {
                    "removed_path": removed_path,
                    "kept_path": kept_path,
                    "md5_hash": group["hash"],
                    "class_name": manifest.loc[
                        manifest["relative_path"] == removed_path, "class_name"
                    ].iloc[0],
                }
            )
    pd.DataFrame(removed_records).to_csv(DUPLICATES_PATH, index=False)
    manifest = manifest[~manifest["relative_path"].isin(duplicate_paths)].reset_index(drop=True)
    print(f"Removed {len(duplicate_paths)} duplicate copy/copies → {len(manifest):,} images in manifest")
    print(f"Removed paths logged to: {DUPLICATES_PATH}")
else:
    print("No exact duplicates found; duplicates.csv not written.")

# Save manifest (relative paths only — no absolute DATA_DIR paths)
manifest.to_csv(MANIFEST_PATH, index=False)
print(f"Manifest saved to: {MANIFEST_PATH}")
print(f"Rows: {len(manifest):,}")

# Example: resolve full path at runtime (not stored in CSV)
sample = manifest.iloc[0]
sample_abs = resolve_filepath(sample["relative_path"])
print(f"\nExample resolution:")
print(f"  relative_path: {sample['relative_path']}")
print(f"  full path:     {sample_abs}")
print(f"  exists:        {sample_abs.exists()}")


# %% [markdown]
#  ## Section 4 — EDA: class distribution & sample gallery
# 
# 
# 
#  Bar chart of per-class counts (expect 800 each) and a random sample grid for visual sanity checks.

# %%
FIGURES_DIR = OUTPUTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)
sns.set_theme(style="whitegrid", context="notebook")

def load_image_array(relative_path: str) -> np.ndarray:
    """Load image as RGB uint8 array (H, W, 3)."""
    with Image.open(resolve_filepath(relative_path)) as img:
        return np.asarray(img.convert("RGB"))


# %%
# 4a — Class distribution
class_counts = manifest["class_name"].value_counts().reindex(CLASS_NAMES)

fig, ax = plt.subplots(figsize=(12, 5))
sns.barplot(x=class_counts.index, y=class_counts.values, hue=class_counts.index, ax=ax, palette="viridis", legend=False)
ax.set_title("Images per class (expected: 800 each)")
ax.set_xlabel("Class")
ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "class_distribution.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {FIGURES_DIR / 'class_distribution.png'}")


# %%
# 4b — Sample gallery (3 random images per class)
SAMPLES_PER_CLASS = 3

fig, axes = plt.subplots(len(CLASS_NAMES), SAMPLES_PER_CLASS, figsize=(3 * SAMPLES_PER_CLASS, 2.2 * len(CLASS_NAMES)))
fig.suptitle("Random sample gallery (3 per class)", y=1.01, fontsize=14)

for row_idx, class_name in enumerate(CLASS_NAMES):
    class_paths = manifest.loc[manifest["class_name"] == class_name, "relative_path"].tolist()
    if len(class_paths) < SAMPLES_PER_CLASS:
        print(f"WARNING: {class_name} has only {len(class_paths)} image(s); skipping gallery row")
        for col_idx in range(SAMPLES_PER_CLASS):
            axes[row_idx, col_idx].axis("off")
        continue
    sample_paths = RNG.choice(class_paths, size=SAMPLES_PER_CLASS, replace=False)

    for col_idx, rel_path in enumerate(sample_paths):
        ax = axes[row_idx, col_idx]
        ax.imshow(load_image_array(rel_path))
        ax.axis("off")
        if col_idx == 0:
            ax.set_ylabel(class_name, rotation=0, labelpad=50, va="center", fontsize=9)

plt.tight_layout()
fig.savefig(FIGURES_DIR / "class_sample_gallery.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {FIGURES_DIR / 'class_sample_gallery.png'}")


# %% [markdown]
#  ## Section 5 — EDA: color, brightness & contrast by class
# 
# 
# 
#  Per-image RGB/HSV statistics, global brightness, and grayscale contrast aggregated per class.

# %%
def compute_color_luminance_stats(img: np.ndarray) -> dict:
    """Per-image color and luminance statistics (values in 0–1 scale)."""
    img_f = img.astype(np.float64) / 255.0
    r, g, b = img_f[..., 0], img_f[..., 1], img_f[..., 2]
    hsv = skcolor.rgb2hsv(img_f)
    gray = skcolor.rgb2gray(img_f)

    return {
        "mean_r": r.mean(),
        "mean_g": g.mean(),
        "mean_b": b.mean(),
        "std_r": r.std(),
        "std_g": g.std(),
        "std_b": b.std(),
        "mean_h": hsv[..., 0].mean(),
        "mean_s": hsv[..., 1].mean(),
        "mean_v": hsv[..., 2].mean(),
        "brightness": hsv[..., 2].mean(),
        "contrast": gray.std(),
    }


color_records = []
for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Color/luminance stats"):
    stats = compute_color_luminance_stats(load_image_array(row["relative_path"]))
    stats["relative_path"] = row["relative_path"]
    stats["class_name"] = row["class_name"]
    color_records.append(stats)

color_df = pd.DataFrame(color_records)
print(f"Computed stats for {len(color_df):,} images")
color_df.head()


# %%
# 5b — Boxplots for brightness and contrast by class
profile_metrics = ["mean_r", "mean_g", "mean_b", "brightness", "contrast"]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.ravel()

for ax, metric in zip(axes, profile_metrics):
    sns.boxplot(data=color_df, x="class_name", y=metric, order=CLASS_NAMES, ax=ax, linewidth=0.8)
    ax.set_title(metric)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=90, labelsize=7)

axes[-1].axis("off")
fig.suptitle("Color, brightness, and contrast distributions by class", y=1.02)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "class_color_profiles.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {FIGURES_DIR / 'class_color_profiles.png'}")


# %%
# 5c — Heatmap of class-mean RGB vectors
class_rgb_means = color_df.groupby("class_name", observed=True)[["mean_r", "mean_g", "mean_b"]].mean()
class_rgb_means = class_rgb_means.reindex(CLASS_NAMES)

fig, ax = plt.subplots(figsize=(6, 8))
sns.heatmap(
    class_rgb_means,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn",
    vmin=0,
    vmax=1,
    ax=ax,
    cbar_kws={"label": "Channel mean (0–1)"},
)
ax.set_title("Class-mean RGB vectors")
ax.set_ylabel("Class")
plt.tight_layout()
fig.savefig(FIGURES_DIR / "class_mean_rgb_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {FIGURES_DIR / 'class_mean_rgb_heatmap.png'}")


# %% [markdown]
#  ## Section 6 — EDA: texture sample & inter-class similarity
# 
# 
# 
#  GLCM texture features on a stratified sample (~50 images/class), plus a cosine-similarity heatmap of class centroids in a color/luminance feature space (image-level z-score before averaging).
# 
#  Raw cosine similarity on all-positive RGB/brightness vectors collapses near 1.0 (see `docs/eda_feature_decisions.md`).

# %%
TEXTURE_SAMPLES_PER_CLASS = 50

def compute_glcm_features(img: np.ndarray) -> dict:
    """GLCM texture props averaged over distances/angles."""
    gray = (skcolor.rgb2gray(img) * 255).astype(np.uint8)
    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=256,
        symmetric=True,
        normed=True,
    )

    def _prop_mean(prop: str) -> float:
        return float(np.nanmean(graycoprops(glcm, prop)))

    return {
        "glcm_contrast": _prop_mean("contrast"),
        "glcm_homogeneity": _prop_mean("homogeneity"),
        "glcm_energy": _prop_mean("energy"),
        "glcm_correlation": _prop_mean("correlation"),
    }


texture_sample = pd.concat(
    [
        group.sample(n=min(TEXTURE_SAMPLES_PER_CLASS, len(group)), random_state=42)
        for _, group in manifest.groupby("class_name", sort=False)
    ],
    ignore_index=True,
)

texture_records = []
for _, row in tqdm(texture_sample.iterrows(), total=len(texture_sample), desc="GLCM texture"):
    feats = compute_glcm_features(load_image_array(row["relative_path"]))
    feats["class_name"] = row["class_name"]
    texture_records.append(feats)

texture_df = pd.DataFrame(texture_records)
print(f"Texture sample size: {len(texture_df):,} images ({TEXTURE_SAMPLES_PER_CLASS}/class)")
texture_df.head()


# %%
# 6a — GLCM feature distributions by class
glcm_metrics = ["glcm_contrast", "glcm_homogeneity", "glcm_energy", "glcm_correlation"]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for ax, metric in zip(axes, glcm_metrics):
    sns.violinplot(data=texture_df, x="class_name", y=metric, order=CLASS_NAMES, ax=ax, inner="quartile", linewidth=0.8)
    ax.set_title(metric)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=90, labelsize=7)

fig.suptitle(f"GLCM texture distributions (n={TEXTURE_SAMPLES_PER_CLASS}/class)", y=1.02)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "texture_glcm_by_class.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {FIGURES_DIR / 'texture_glcm_by_class.png'}")


# %%
# 6b — Inter-class similarity from standardized color/luminance centroids
# Z-score each image first, then average within class. Z-scoring only the 15 class
# centroids over-amplifies cosine scores (±0.98) because n_classes ≈ p_features.
# mean_h omitted: hue is circular and unstable when averaged per image.
SIMILARITY_FEATURES = [
    "mean_r", "mean_g", "mean_b",
    "std_r", "std_g", "std_b",
    "mean_s", "contrast",
]

sim_scaler = StandardScaler()
X_sim = sim_scaler.fit_transform(color_df[SIMILARITY_FEATURES])

class_centroids = (
    pd.DataFrame(X_sim, columns=SIMILARITY_FEATURES)
    .assign(class_name=color_df["class_name"].to_numpy())
    .groupby("class_name", observed=True)[SIMILARITY_FEATURES]
    .mean()
    .reindex(CLASS_NAMES)
)

centroid_matrix = class_centroids.to_numpy()
similarity = cosine_similarity(centroid_matrix)

similarity_df = pd.DataFrame(similarity, index=CLASS_NAMES, columns=CLASS_NAMES)

off_diag_mask = ~np.eye(len(CLASS_NAMES), dtype=bool)
off_diag = similarity[off_diag_mask]
print(
    f"Off-diagonal cosine similarity range: "
    f"{off_diag.min():.3f} – {off_diag.max():.3f} "
    f"(image-level z-score → class mean; measures shared color/texture profile)"
)
print(
    f"Features: {len(SIMILARITY_FEATURES)} dims — RGB means & stds, saturation, contrast"
)

fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(
    similarity_df,
    annot=False,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    center=0,
    square=True,
    ax=ax,
    cbar_kws={"label": "Cosine similarity (image-z-scored centroids)"},
)
ax.set_title("Inter-class similarity (per-image z-score, then class mean)")
plt.tight_layout()
fig.savefig(FIGURES_DIR / "interclass_similarity_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {FIGURES_DIR / 'interclass_similarity_heatmap.png'}")

# Most / least similar class pairs (excluding self)
pairs = []
for i, c1 in enumerate(CLASS_NAMES):
    for j, c2 in enumerate(CLASS_NAMES):
        if i < j:
            pairs.append((c1, c2, similarity_df.loc[c1, c2]))
top_pairs = sorted(pairs, key=lambda x: x[2], reverse=True)[:5]
bottom_pairs = sorted(pairs, key=lambda x: x[2])[:5]
print("\nTop 5 most similar class pairs:")
for c1, c2, score in top_pairs:
    print(f"  {c1} ↔ {c2}: {score:.3f}")
print("\nTop 5 least similar class pairs:")
for c1, c2, score in bottom_pairs:
    print(f"  {c1} ↔ {c2}: {score:.3f}")


# %% [markdown]
#  ## Section 7 — EDA: PCA 2D scatter
# 
# 
# 
#  PCA on per-image color/luminance statistics (fast proxy for pixel space). Points colored by class reveal overlap clusters that may be hard for linear models.

# %%
PCA_FEATURES = [
    "mean_r", "mean_g", "mean_b",
    "std_r", "std_g", "std_b",
    "mean_h", "mean_s", "mean_v",
    "contrast",
]

X = color_df[PCA_FEATURES].to_numpy()
X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_scaled)

pca_df = color_df[["class_name"]].copy()
pca_df["pc1"] = coords[:, 0]
pca_df["pc2"] = coords[:, 1]

fig, ax = plt.subplots(figsize=(12, 9))
sns.scatterplot(
    data=pca_df,
    x="pc1",
    y="pc2",
    hue="class_name",
    hue_order=CLASS_NAMES,
    palette="tab20",
    alpha=0.35,
    s=12,
    linewidth=0,
    ax=ax,
    legend=False,
)
ax.set_title(
    f"PCA of color/luminance features "
    f"(explained variance: {pca.explained_variance_ratio_.sum():.1%})"
)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
plt.tight_layout()
fig.savefig(FIGURES_DIR / "pca_color_features_scatter.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {FIGURES_DIR / 'pca_color_features_scatter.png'}")


# %% [markdown]
#  ## Section 8 — EDA: augmentation preview
# 
# 
# 
#  Side-by-side preview of intended training transforms on one example image per class (brightness, contrast, rotation, Gaussian noise).

# %%
def apply_augmentation_preview(img: np.ndarray, seed: int | None = None) -> dict[str, np.ndarray]:
    """Return original plus single-example augmentations for visualization."""
    rng = np.random.default_rng(seed)
    pil = Image.fromarray(img)

    bright_up = ImageEnhance.Brightness(pil).enhance(1.2)
    bright_down = ImageEnhance.Brightness(pil).enhance(0.8)
    contrast_up = ImageEnhance.Contrast(pil).enhance(1.3)
    rotated = pil.rotate(15, expand=False, fillcolor=(0, 0, 0))

    noisy = img.astype(np.float32)
    noise = rng.normal(0, 15, noisy.shape)
    noisy = np.clip(noisy + noise, 0, 255).astype(np.uint8)

    return {
        "original": img,
        "brightness +20%": np.asarray(bright_up),
        "brightness -20%": np.asarray(bright_down),
        "contrast +30%": np.asarray(contrast_up),
        "rotate 15°": np.asarray(rotated),
        "gaussian noise": noisy,
    }


AUGMENT_CLASSES = ["Agriculture", "City", "Lake", "Forest", "Airport"]
aug_titles = ["original", "brightness +20%", "brightness -20%", "contrast +30%", "rotate 15°", "gaussian noise"]

fig, axes = plt.subplots(len(AUGMENT_CLASSES), len(aug_titles), figsize=(3 * len(aug_titles), 2.8 * len(AUGMENT_CLASSES)))
fig.suptitle("Augmentation preview (one example per selected class)", y=1.01, fontsize=14)

for row_idx, class_name in enumerate(AUGMENT_CLASSES):
    class_rows = manifest.loc[manifest["class_name"] == class_name, "relative_path"]
    if class_rows.empty:
        print(f"WARNING: no images for {class_name}; skipping augmentation row")
        for col_idx in range(len(aug_titles)):
            axes[row_idx, col_idx].axis("off")
        continue
    rel_path = class_rows.iloc[0]
    img = load_image_array(rel_path)
    augmented = apply_augmentation_preview(img, seed=42 + row_idx)

    for col_idx, title in enumerate(aug_titles):
        ax = axes[row_idx, col_idx]
        ax.imshow(augmented[title])
        ax.axis("off")
        if row_idx == 0:
            ax.set_title(title, fontsize=10)
        if col_idx == 0:
            ax.set_ylabel(class_name, rotation=0, labelpad=45, va="center", fontsize=9)

plt.tight_layout()
fig.savefig(FIGURES_DIR / "augmentation_preview.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {FIGURES_DIR / 'augmentation_preview.png'}")


# %% [markdown]
#  ## Section 9 — Stratified train/val/test split
# 
# 
# 
#  80 / 10 / 10 split stratified by `class_name` (~640 / 80 / 80 per class; Airport has 3 fewer after dedup).

# %%
from sklearn.model_selection import train_test_split

SPLIT_RANDOM_STATE = 42
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.8, 0.1, 0.1

train_idx, holdout_idx = train_test_split(
    manifest.index,
    test_size=VAL_FRAC + TEST_FRAC,
    stratify=manifest["class_name"],
    random_state=SPLIT_RANDOM_STATE,
)

val_idx, test_idx = train_test_split(
    holdout_idx,
    test_size=TEST_FRAC / (VAL_FRAC + TEST_FRAC),
    stratify=manifest.loc[holdout_idx, "class_name"],
    random_state=SPLIT_RANDOM_STATE,
)

manifest["split"] = pd.NA
manifest.loc[train_idx, "split"] = "train"
manifest.loc[val_idx, "split"] = "val"
manifest.loc[test_idx, "split"] = "test"

split_counts = manifest.groupby(["class_name", "split"], observed=True).size().unstack(fill_value=0)
split_counts = split_counts.reindex(CLASS_NAMES)
print("Split counts per class (expected ~640 / 80 / 80; Airport −3 after dedup):")
print(split_counts.to_string())
print()
print("Overall split sizes:")
print(manifest["split"].value_counts().sort_index().to_string())

manifest.to_csv(MANIFEST_PATH, index=False)
print(f"\nManifest with splits saved to: {MANIFEST_PATH}")


# %% [markdown]
#  ## Section 10 — Preprocessing helpers
# 
# 
# 
#  Reusable image-loading and normalization functions (later moved to `src/preprocessing.py`).
# 
#  CNN follow-up notebooks use `iter_cnn_batches` below.

# %%
def load_image(path: str | Path, size: tuple[int, int] = (256, 256)) -> np.ndarray:
    """Load an image as RGB uint8 array, optionally resizing."""
    path = Path(path)
    if not path.is_absolute():
        path = resolve_filepath(path.as_posix())
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        if rgb.size != size:
            rgb = rgb.resize(size, Image.Resampling.BILINEAR)
        return np.asarray(rgb)


def normalize_pixels(img: np.ndarray, method: str = "minmax") -> np.ndarray:
    """Scale pixel values to [0, 1]."""
    img_f = img.astype(np.float64)
    if method == "minmax":
        return img_f / 255.0
    if method == "standard":
        return (img_f - img_f.mean()) / (img_f.std() + 1e-8)
    raise ValueError(f"Unknown normalization method: {method}")


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert RGB uint8 image to grayscale float in [0, 1]."""
    return skcolor.rgb2gray(normalize_pixels(img))


def apply_augmentation(img: np.ndarray, seed: int | None = None) -> np.ndarray:
    """Apply a random training augmentation (brightness, contrast, rotation, or noise)."""
    rng = np.random.default_rng(seed)
    pil = Image.fromarray(img)
    choice = rng.integers(0, 4)

    if choice == 0:
        factor = rng.uniform(0.8, 1.2)
        out = ImageEnhance.Brightness(pil).enhance(factor)
    elif choice == 1:
        factor = rng.uniform(0.8, 1.3)
        out = ImageEnhance.Contrast(pil).enhance(factor)
    elif choice == 2:
        angle = float(rng.choice([90, 180, -15, 15]))
        out = pil.rotate(angle, expand=False, fillcolor=(0, 0, 0))
    else:
        noisy = img.astype(np.float32)
        noisy += rng.normal(0, 15, noisy.shape)
        return np.clip(noisy, 0, 255).astype(np.uint8)

    return np.asarray(out)


def iter_cnn_batches(
    df: pd.DataFrame,
    batch_size: int = 32,
    augment: bool = False,
    shuffle: bool = False,
    split: str | None = None,
    seed: int | None = None,
):
    """Yield (batch_images, batch_labels) for CNN training loops."""
    work = df
    if split is not None:
        work = work[work["split"] == split]
    rng = np.random.default_rng(seed)
    rows = work.to_dict("records")
    if shuffle or augment:
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


_sample = load_image(manifest.iloc[0]["relative_path"])
print(f"load_image shape: {_sample.shape}, dtype: {_sample.dtype}")
print(f"normalize_pixels range: [{normalize_pixels(_sample).min():.3f}, {normalize_pixels(_sample).max():.3f}]")
print(f"to_grayscale shape: {to_grayscale(_sample).shape}")
_batch_x, _batch_y = next(iter_cnn_batches(manifest.head(4), batch_size=2, augment=False))
print(f"CNN batch stub: X={_batch_x.shape}, y={_batch_y.shape}")


# %% [markdown]
#  ## Section 11 — Feature extraction
# 
# 
# 
#  Batch tabular feature extraction over the full manifest (original images, no augmentation).

# %%
from skimage.filters import sobel

HIST_BINS = 8

HUMAN_MADE_CLASSES = {"Airport", "City", "Highway", "Parking", "Port", "Railway", "Residential"}
WATER_RELATED_CLASSES = {"Beach", "Lake", "River", "Port"}
VEGETATION_RELATED_CLASSES = {"Agriculture", "Forest", "Grassland"}
URBAN_DENSITY_PROXY = {
    "City": 3,
    "Residential": 2,
    "Highway": 1,
    "Parking": 2,
    "Airport": 2,
    "Port": 2,
    "Railway": 2,
}


def extract_image_features(img: np.ndarray, hist_bins: int = HIST_BINS) -> dict:
    """Extract per-image tabular features for traditional ML models."""
    feats = compute_color_luminance_stats(img)

    img_f = img.astype(np.float64) / 255.0
    r, g, b = img_f[..., 0], img_f[..., 1], img_f[..., 2]
    gray = skcolor.rgb2gray(img_f)

    feats["ratio_r_g"] = r.mean() / (g.mean() + 1e-8)
    feats["ratio_r_b"] = r.mean() / (b.mean() + 1e-8)
    feats["ratio_g_b"] = g.mean() / (b.mean() + 1e-8)
    feats["dynamic_range"] = float(gray.max() - gray.min())

    for ch_idx, ch_name in enumerate(["r", "g", "b"]):
        hist, _ = np.histogram(img[..., ch_idx], bins=hist_bins, range=(0, 255), density=True)
        for bin_idx, val in enumerate(hist):
            feats[f"hist_{ch_name}_{bin_idx}"] = float(val)

    feats.update(compute_glcm_features(img))
    feats["edge_density"] = float(sobel(gray).mean())
    return feats


def add_ontology_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add class-derived metadata columns for future regression targets."""
    out = df.copy()
    out["is_human_made"] = out["class_name"].isin(HUMAN_MADE_CLASSES).astype(int)
    out["is_water_related"] = out["class_name"].isin(WATER_RELATED_CLASSES).astype(int)
    out["is_vegetation_related"] = out["class_name"].isin(VEGETATION_RELATED_CLASSES).astype(int)
    out["urban_density_proxy"] = out["class_name"].map(URBAN_DENSITY_PROXY).fillna(0).astype(int)
    return out


feature_records = []
for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Extracting features"):
    feats = extract_image_features(load_image(row["relative_path"]))
    feats["relative_path"] = row["relative_path"]
    feats["class_name"] = row["class_name"]
    feats["label_idx"] = row["label_idx"]
    feats["split"] = row["split"]
    feature_records.append(feats)

features_df = pd.DataFrame(feature_records)
features_df = add_ontology_columns(features_df)
print(f"Feature matrix shape: {features_df.shape}")
print(features_df.head())


# %% [markdown]
#  ## Section 12 — Save features CSV & sanity checks

# %%
FEATURES_DIR = OUTPUTS_DIR / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_PATH = FEATURES_DIR / "image_features.csv"
REGRESSION_TARGETS_PATH = FEATURES_DIR / "regression_targets.csv"
FEATURE_COLUMNS_PATH = FEATURES_DIR / "feature_columns.txt"
SCALER_PATH = FEATURES_DIR / "scaler.joblib"

META_COLS = {"relative_path", "class_name", "label_idx", "split"}
feature_cols = [
    c
    for c in features_df.columns
    if c not in META_COLS and c not in ONTOLOGY_COLS and c not in REDUNDANT_PREDICTOR_COLS
]

features_df.to_csv(FEATURES_PATH, index=False)
print(f"Features saved to: {FEATURES_PATH}")
print(f"Rows: {len(features_df):,}  |  Columns: {features_df.shape[1]}")
print(f"Predictor columns: {len(feature_cols)}  |  Ontology targets: {len(ONTOLOGY_COLS)}")

regression_cols = ["relative_path", "class_name", "label_idx", "split"] + ONTOLOGY_COLS
features_df[regression_cols].to_csv(REGRESSION_TARGETS_PATH, index=False)
print(f"Regression targets saved to: {REGRESSION_TARGETS_PATH}")

FEATURE_COLUMNS_PATH.write_text("\n".join(feature_cols) + "\n")
print(f"Feature column list saved to: {FEATURE_COLUMNS_PATH}")

missing = features_df[feature_cols].isna().sum()
missing = missing[missing > 0]
print("\nMissing values (predictors):")
if missing.empty:
    print("  none")
else:
    print(missing.to_string())
    raise RuntimeError("Feature extraction produced missing predictor values (see above).")

print("\nOntology target summary (per class):")
ontology_summary = (
    features_df.groupby("class_name", observed=True)[ONTOLOGY_COLS]
    .first()
    .sort_index()
)
print(ontology_summary.to_string())

corr_subset = [
    "mean_r", "mean_g", "mean_b", "mean_v", "contrast", "dynamic_range",
    "glcm_contrast", "glcm_homogeneity", "glcm_energy", "glcm_correlation", "edge_density",
]
corr_subset = [c for c in corr_subset if c in features_df.columns]
corr = features_df[corr_subset].corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True, ax=ax)
ax.set_title("Feature correlation (selected columns)")
plt.tight_layout()
fig.savefig(FIGURES_DIR / "feature_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {FIGURES_DIR / 'feature_correlation_heatmap.png'}")

train_mask = features_df["split"] == "train"
scaler = StandardScaler()
X_train = features_df.loc[train_mask, feature_cols].to_numpy()
X_train_scaled = scaler.fit_transform(X_train)
joblib.dump(scaler, SCALER_PATH)
print(f"\nStandardScaler fit on train split: {X_train.shape[0]:,} rows x {X_train.shape[1]} features")
print(f"StandardScaler saved to: {SCALER_PATH}")
print(f"Scaled train mean (first 5 cols): {X_train_scaled[:, :5].mean(axis=0).round(4)}")
print(f"Scaled train std  (first 5 cols): {X_train_scaled[:, :5].std(axis=0).round(4)}")


# %% [markdown]
#  ## Section 13 — Done

# %%
print("EDA & preprocessing complete.")
print(f"  Manifest:           {MANIFEST_PATH}")
if DUPLICATES_PATH.exists():
    print(f"  Duplicates:         {DUPLICATES_PATH}")
print(f"  Features:           {FEATURES_PATH}")
print(f"  Regression targets: {REGRESSION_TARGETS_PATH}")
print(f"  Feature columns:    {FEATURE_COLUMNS_PATH}")
print(f"  Scaler:             {SCALER_PATH}")
print(f"  Figures:            {FIGURES_DIR}")




