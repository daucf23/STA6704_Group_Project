"""Shared train/val/test splits from EDA ``outputs/manifest.csv``.

All modeling notebooks should load image membership from this module rather
than re-splitting (e.g. ``random_split``). The manifest already encodes the
stratified 80/10/10 split (``random_state=42``) after MD5 dedup.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

REQUIRED_COLUMNS: Sequence[str] = (
    "relative_path",
    "class_name",
    "label_idx",
    "split",
)

VALID_SPLITS = frozenset({"train", "val", "test"})

# Canonical class order from EDA (label_idx 0–14). Do not reorder.
CLASS_NAMES: Sequence[str] = (
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
)

CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def load_manifest(project_root: Path | str) -> pd.DataFrame:
    """Read and validate ``outputs/manifest.csv`` under ``project_root``."""
    root = Path(project_root)
    path = root / "outputs" / "manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")

    manifest = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in manifest.columns]
    if missing:
        raise ValueError(
            f"Manifest missing required columns {missing}; found {list(manifest.columns)}"
        )

    unknown = set(manifest["split"].dropna().unique()) - VALID_SPLITS
    if unknown:
        raise ValueError(f"Unexpected split values: {sorted(unknown)}")

    return manifest


def resolve_image_path(project_root: Path | str, relative_path: str) -> Path:
    """Map a manifest ``relative_path`` to ``project_root/data/<relative_path>``."""
    return Path(project_root) / "data" / relative_path


def split_frames(
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return ``(train_df, val_df, test_df)`` filtered by the ``split`` column."""
    if "split" not in manifest.columns:
        raise ValueError("manifest must include a 'split' column")

    train_df = manifest.loc[manifest["split"] == "train"].reset_index(drop=True)
    val_df = manifest.loc[manifest["split"] == "val"].reset_index(drop=True)
    test_df = manifest.loc[manifest["split"] == "test"].reset_index(drop=True)
    return train_df, val_df, test_df


def paths_and_labels(
    split_df: pd.DataFrame,
    project_root: Path | str,
) -> tuple[list[Path], list[int]]:
    """Absolute image paths and ``label_idx`` values for CNN/tree indexing."""
    for col in ("relative_path", "label_idx"):
        if col not in split_df.columns:
            raise ValueError(f"split_df must include '{col}'")

    root = Path(project_root)
    paths = [resolve_image_path(root, rel) for rel in split_df["relative_path"]]
    labels = [int(y) for y in split_df["label_idx"]]
    return paths, labels
