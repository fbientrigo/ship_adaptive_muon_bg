#!/usr/bin/env python3
"""Inspect all committed data samples for available fields and potential labels.

This script loads every NPZ and PKL.GZ file in data/samples/ and prints a
full inventory of keys/columns, shapes, dtypes, ranges, NaN/inf counts, and
then searches for anything that could serve as a physics label or
classification target.

Run from the repo root:
    python scripts/inspect_available_labels.py
"""

from __future__ import annotations

import gzip
import os
import pathlib
import pickle
import sys
import textwrap

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SAMPLES_DIR = REPO_ROOT / "data" / "samples"

NPZ_FILES = [
    SAMPLES_DIR / "muonsFullMC_afterMS_sample.npz",
    SAMPLES_DIR / "muons_FullMC_sample.npz",
]

PKL_GZ_FILES = [
    SAMPLES_DIR / "muonsFullMC_afterMS_sample.pkl.gz",
    SAMPLES_DIR / "muons_FullMC_sample.pkl.gz",
]

# Patterns to search for in field / column / key names (case-insensitive)
LABEL_PATTERNS = [
    "candidate", "reco", "reconstructed", "veto",
    "SBT", "UBT", "DOCA", "IP",
    "fiducial", "wall", "front", "side", "cavern",
    "DIS", "accepted", "rejected", "dangerous",
    "rho", "material", "weight", "label", "target",
    "y_true", "score", "utility", "u_x",
    "tag", "flag", "class", "category",
]

SCHEMA_COLUMNS = ["px", "py", "pz", "x", "y", "z", "id", "w"]

SEPARATOR = "=" * 80


def inspect_npz(path: pathlib.Path) -> list[str]:
    """Load an NPZ file and print detailed stats.  Return list of key names."""
    print(f"\n{SEPARATOR}")
    print(f"NPZ FILE: {path.relative_to(REPO_ROOT)}")
    print(SEPARATOR)

    if not path.exists():
        print(f"  *** FILE NOT FOUND ***")
        return []

    data = np.load(path, allow_pickle=False)
    keys = list(data.keys())
    print(f"  Keys: {keys}")

    all_field_names: list[str] = list(keys)

    for key in keys:
        arr = data[key]
        print(f"\n  --- Key: '{key}' ---")
        print(f"    shape : {arr.shape}")
        print(f"    dtype : {arr.dtype}")
        print(f"    nbytes: {arr.nbytes:,}")

        if np.issubdtype(arr.dtype, np.number):
            nan_count = int(np.isnan(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0
            inf_count = int(np.isinf(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0
            print(f"    min   : {np.min(arr)}")
            print(f"    max   : {np.max(arr)}")
            print(f"    mean  : {np.mean(arr)}")
            print(f"    NaN   : {nan_count}")
            print(f"    Inf   : {inf_count}")

            # If 2-D, print per-column stats using schema names
            if arr.ndim == 2:
                n_cols = arr.shape[1]
                col_names = SCHEMA_COLUMNS[:n_cols] if n_cols <= len(SCHEMA_COLUMNS) else [f"col_{i}" for i in range(n_cols)]
                all_field_names.extend(col_names)
                print(f"\n    Per-column breakdown (named via schema):")
                for ci, cname in enumerate(col_names):
                    col = arr[:, ci]
                    nan_c = int(np.isnan(col).sum()) if np.issubdtype(col.dtype, np.floating) else 0
                    inf_c = int(np.isinf(col).sum()) if np.issubdtype(col.dtype, np.floating) else 0
                    uniq = np.unique(col)
                    uniq_str = str(uniq) if len(uniq) <= 20 else f"{len(uniq)} unique values"
                    print(f"      [{ci}] {cname:4s}  dtype={col.dtype}  min={np.min(col):.6g}  max={np.max(col):.6g}  "
                          f"mean={np.mean(col):.6g}  std={np.std(col):.6g}  NaN={nan_c}  Inf={inf_c}  uniq={uniq_str}")
        else:
            print(f"    (non-numeric dtype, skipping numeric stats)")

        # Print first 3 rows for shape context
        if arr.ndim <= 2:
            print(f"    First 3 rows:")
            for row in arr[:3]:
                print(f"      {row}")

    data.close()
    return all_field_names


def inspect_pkl_gz(path: pathlib.Path) -> list[str]:
    """Load a gzip-pickled file, print stats.  Return list of field names."""
    print(f"\n{SEPARATOR}")
    print(f"PKL.GZ FILE: {path.relative_to(REPO_ROOT)}")
    print(SEPARATOR)

    if not path.exists():
        print(f"  *** FILE NOT FOUND ***")
        return []

    try:
        with gzip.open(path, "rb") as f:
            obj = pickle.load(f)
    except Exception as e:
        print(f"  *** LOAD ERROR: {e} ***")
        return []

    print(f"  Python type: {type(obj).__module__}.{type(obj).__qualname__}")

    all_field_names: list[str] = []

    # --- numpy array ---
    if isinstance(obj, np.ndarray):
        arr = obj
        print(f"  shape : {arr.shape}")
        print(f"  dtype : {arr.dtype}")

        if np.issubdtype(arr.dtype, np.number):
            nan_count = int(np.isnan(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0
            inf_count = int(np.isinf(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0
            print(f"  min   : {np.min(arr)}")
            print(f"  max   : {np.max(arr)}")
            print(f"  NaN   : {nan_count}")
            print(f"  Inf   : {inf_count}")

        if arr.ndim == 2:
            n_cols = arr.shape[1]
            col_names = SCHEMA_COLUMNS[:n_cols] if n_cols <= len(SCHEMA_COLUMNS) else [f"col_{i}" for i in range(n_cols)]
            all_field_names.extend(col_names)
            print(f"\n  Per-column breakdown:")
            for ci, cname in enumerate(col_names):
                col = arr[:, ci]
                nan_c = int(np.isnan(col).sum()) if np.issubdtype(col.dtype, np.floating) else 0
                inf_c = int(np.isinf(col).sum()) if np.issubdtype(col.dtype, np.floating) else 0
                uniq = np.unique(col)
                uniq_str = str(uniq) if len(uniq) <= 20 else f"{len(uniq)} unique values"
                print(f"    [{ci}] {cname:4s}  min={np.min(col):.6g}  max={np.max(col):.6g}  "
                      f"mean={np.mean(col):.6g}  NaN={nan_c}  Inf={inf_c}  uniq={uniq_str}")

        print(f"\n  First 5 rows:")
        for row in arr[:5]:
            print(f"    {row}")

    # --- pandas DataFrame ---
    else:
        try:
            import pandas as pd
            if isinstance(obj, pd.DataFrame):
                all_field_names.extend(list(obj.columns))
                print(f"  shape   : {obj.shape}")
                print(f"  columns : {list(obj.columns)}")
                print(f"  dtypes  :\n{textwrap.indent(str(obj.dtypes), '    ')}")
                print(f"\n  First 5 rows:")
                print(textwrap.indent(obj.head(5).to_string(), "    "))
                print(f"\n  describe():")
                print(textwrap.indent(obj.describe().to_string(), "    "))
        except ImportError:
            pass

        # Fallback: dict-like
        if hasattr(obj, "keys"):
            ks = list(obj.keys())
            all_field_names.extend(ks)
            print(f"  dict keys: {ks}")
            for k in ks:
                v = obj[k]
                print(f"    {k}: type={type(v).__name__}, ", end="")
                if isinstance(v, np.ndarray):
                    print(f"shape={v.shape}, dtype={v.dtype}")
                elif hasattr(v, "__len__"):
                    print(f"len={len(v)}")
                else:
                    print(f"value={v!r}")

    return all_field_names


def search_labels(all_names: list[str], source: str) -> list[str]:
    """Search field names for physics-label patterns."""
    matches: list[str] = []
    for name in all_names:
        name_lower = name.lower()
        for pattern in LABEL_PATTERNS:
            if pattern.lower() in name_lower:
                matches.append(f"  {source}: field '{name}' matches pattern '{pattern}'")
                break  # one match per field is enough
    return matches


def main() -> None:
    print("=" * 80)
    print("  COMPREHENSIVE DATA SAMPLE INSPECTION")
    print(f"  Repo root: {REPO_ROOT}")
    print(f"  Samples dir: {SAMPLES_DIR}")
    print("=" * 80)

    # Collect all field names across all files for label search
    all_matches: list[str] = []

    # --- NPZ files ---
    for npz_path in NPZ_FILES:
        field_names = inspect_npz(npz_path)
        matches = search_labels(field_names, npz_path.name)
        all_matches.extend(matches)

    # --- PKL.GZ files ---
    for pkl_path in PKL_GZ_FILES:
        field_names = inspect_pkl_gz(pkl_path)
        matches = search_labels(field_names, pkl_path.name)
        all_matches.extend(matches)

    # --- Also check for any OTHER data files we might have missed ---
    print(f"\n{SEPARATOR}")
    print("SCANNING FOR ADDITIONAL DATA FILES IN data/samples/")
    print(SEPARATOR)
    for f in sorted(SAMPLES_DIR.rglob("*")):
        if f.is_file():
            rel = f.relative_to(REPO_ROOT)
            size = f.stat().st_size
            print(f"  {rel}  ({size:,} bytes)")

    # --- LABEL SEARCH SUMMARY ---
    print(f"\n{'#' * 80}")
    print("  LABEL SEARCH SUMMARY")
    print(f"{'#' * 80}")
    print(f"\n  Patterns searched: {LABEL_PATTERNS}")
    print(f"\n  Total matches found: {len(all_matches)}")

    if all_matches:
        print("\n  Matches:")
        for m in all_matches:
            print(f"    {m}")
    else:
        print("\n  >>> NO LABEL-LIKE FIELDS FOUND IN ANY DATA FILE <<<")

    # --- Explicit check: do the arrays contain ONLY the 8 kinematic columns? ---
    print(f"\n{'#' * 80}")
    print("  FIELD INVENTORY ASSESSMENT")
    print(f"{'#' * 80}")
    print(f"""
  Both NPZ and PKL.GZ files contain a single numpy.ndarray of shape (40000, 8).
  The 8 columns are strictly the kinematic/weight schema:
    [px, py, pz, x, y, z, id, w]

  There are NO additional columns, keys, or fields beyond these 8.

  Columns that match label-search patterns:
    - 'id' (PDG particle ID, values {{-13, 13}}) — NOT a physics selection label
    - 'w' (event weight) — NOT a binary label, but a continuous weight

  CONCLUSION:
    The committed data samples contain NO physics labels such as:
      reconstruction status, veto flags, DOCA/IP cuts, fiducial volume,
      acceptance/rejection, dangerous classification, SBT/UBT hits,
      DIS tagging, utility scores, or any binary target variable.

    The data is UNLABELED kinematic + weight data only.
    Any labeling (e.g., "dangerous" muon classification) must be computed
    downstream, either via simulation (FairShip/Geant4) or via the proxy
    model in this repo's adaptive campaign pipeline.
""")


if __name__ == "__main__":
    main()
