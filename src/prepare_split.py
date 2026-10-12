"""
prepare_split.py
────────────────
Group-aware stratified train / val / test split for the HAM10000 dataset.

Key design decisions
────────────────────
* Splitting unit = lesion_id  (not image_id)
  Multiple images from the same lesion MUST land in the same split to prevent
  data leakage (the model would otherwise see the same lesion during training
  AND evaluation).

* Stratification is by dx (diagnosis label) applied at the lesion level.
  Each lesion is assigned its majority dx label (all images of a lesion share
  the same dx in HAM10000, but this handles any edge cases defensively).

* Split ratios:  70 % train  |  15 % val  |  15 % test
  Applied to the set of unique lesion_ids per class.

Output
──────
results/splits.csv   – columns: image_id, lesion_id, dx, split, filepath
"""

import os
import sys
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent
DATASET_ROOT = ROOT / "dataset" / "ham10000"
IMG_PART1    = DATASET_ROOT / "HAM10000_images_part_1"
IMG_PART2    = DATASET_ROOT / "HAM10000_images_part_2"
META_CSV     = DATASET_ROOT / "HAM10000_metadata.csv"
RESULTS_DIR  = ROOT / "results"
OUTPUT_CSV   = RESULTS_DIR / "splits.csv"

# ── Split ratios ──────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9, "Ratios must sum to 1"


def build_image_lookup(part1: Path, part2: Path) -> dict:
    """Return {image_id: filepath_str} from both image folders."""
    lookup = {}
    for folder in [part1, part2]:
        if not folder.exists():
            print(f"  [WARN]  Folder not found: {folder}", file=sys.stderr)
            continue
        for fp in folder.glob("*.jpg"):
            lookup[fp.stem] = str(fp)
    return lookup


def assign_lesion_label(group: pd.DataFrame) -> str:
    """Return the majority dx label for a lesion (defensive: all images
    of a HAM10000 lesion share the same dx, but majority-vote handles edge cases)."""
    return group["dx"].mode()[0]


def stratified_group_split(
    lesion_df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    random_state: int,
) -> pd.DataFrame:
    """
    Perform a two-stage stratified split at the lesion level.

    Stage 1: split off test  → (1 - test_ratio) train+val  vs  test
    Stage 2: split train+val → train  vs  val

    Returns lesion_df with a new 'split' column.
    """
    test_ratio    = 1.0 - train_ratio - val_ratio
    # relative val ratio within the train+val portion
    val_of_trainval = val_ratio / (train_ratio + val_ratio)

    trainval_idx, test_idx = train_test_split(
        lesion_df.index,
        test_size=test_ratio,
        stratify=lesion_df["dx_label"],
        random_state=random_state,
    )

    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=val_of_trainval,
        stratify=lesion_df.loc[trainval_idx, "dx_label"],
        random_state=random_state,
    )

    lesion_df = lesion_df.copy()
    lesion_df["split"] = "train"
    lesion_df.loc[val_idx,  "split"] = "val"
    lesion_df.loc[test_idx, "split"] = "test"
    return lesion_df


def verify_no_leakage(splits_df: pd.DataFrame) -> None:
    """Assert zero lesion_id overlap between train / val / test."""
    train_lesions = set(splits_df[splits_df["split"] == "train"]["lesion_id"])
    val_lesions   = set(splits_df[splits_df["split"] == "val"]["lesion_id"])
    test_lesions  = set(splits_df[splits_df["split"] == "test"]["lesion_id"])

    tv  = train_lesions & val_lesions
    tt  = train_lesions & test_lesions
    vt  = val_lesions   & test_lesions

    if tv or tt or vt:
        raise RuntimeError(
            f"Data leakage detected!\n"
            f"  train & val  : {len(tv)} lesions\n"
            f"  train & test : {len(tt)} lesions\n"
            f"  val & test   : {len(vt)} lesions"
        )
    print("[OK]  Zero lesion_id overlap between train / val / test  (no data leakage)")


def print_split_counts(splits_df: pd.DataFrame) -> None:
    """Print per-class, per-split image counts."""
    print("\n-- Per-class, per-split image counts -------------------------")
    pivot = (
        splits_df.groupby(["dx", "split"])
        .size()
        .unstack(fill_value=0)
        [["train", "val", "test"]]
    )
    pivot["total"] = pivot.sum(axis=1)
    print(pivot.to_string())
    print()
    totals = splits_df["split"].value_counts()[["train", "val", "test"]]
    print("-- Total images per split ------------------------------------")
    for split, cnt in totals.items():
        print(f"  {split:5s}  {cnt:,}")
    print(f"  {'total':5s}  {totals.sum():,}")


def main():
    # ── 1. Validate paths ────────────────────────────────────────────────────
    for p in [META_CSV, IMG_PART1, IMG_PART2]:
        if not p.exists():
            sys.exit(f"ERROR: Required path not found: {p}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 2. Load metadata ─────────────────────────────────────────────────────
    meta = pd.read_csv(META_CSV)
    print(f"Metadata loaded  : {len(meta):,} rows, {meta['lesion_id'].nunique():,} unique lesions")

    # ── 3. Build image lookup & attach filepath ───────────────────────────────
    print("Building image lookup from disk...")
    img_lookup = build_image_lookup(IMG_PART1, IMG_PART2)
    print(f"  Images found on disk : {len(img_lookup):,}")

    meta["filepath"] = meta["image_id"].map(img_lookup)
    missing = meta["filepath"].isna().sum()
    if missing:
        print(f"  [WARN]  {missing} image_ids in CSV have no matching file - they will be excluded.")
    meta = meta[meta["filepath"].notna()].copy()

    # ── 4. Build one row per lesion (for stratified group split) ──────────────
    lesion_df = (
        meta.groupby("lesion_id")
        .apply(assign_lesion_label)
        .reset_index()
    )
    lesion_df.columns = ["lesion_id", "dx_label"]
    print(f"\nUnique lesions   : {len(lesion_df):,}")
    print("Lesion class distribution:")
    print(lesion_df["dx_label"].value_counts().to_string())

    # ── 5. Stratified group split ─────────────────────────────────────────────
    print("\nPerforming group-aware stratified split (70 / 15 / 15)...")
    lesion_split = stratified_group_split(lesion_df, TRAIN_RATIO, VAL_RATIO, RANDOM_STATE)

    # ── 6. Propagate split label back to image level ──────────────────────────
    split_map = lesion_split.set_index("lesion_id")["split"].to_dict()
    meta["split"] = meta["lesion_id"].map(split_map)

    # ── 7. Assemble output ────────────────────────────────────────────────────
    output = meta[["image_id", "lesion_id", "dx", "split", "filepath"]].copy()
    output.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSplits saved to  : {OUTPUT_CSV}")

    # ── 8. Verify & report ────────────────────────────────────────────────────
    verify_no_leakage(output)
    print_split_counts(output)


if __name__ == "__main__":
    main()
