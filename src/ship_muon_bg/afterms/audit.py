"""Phase A: one-time descriptive audit of the after-MS muon PKL.

Pure description, no physics claims. Reuses ``data_contracts`` for loading,
schema, and content hashing; this module adds only the fields the after-MS
nightly mission additionally requires (file hash, sign counts, duplicate
sampling, quantile grid, correlation matrix, weighted summaries).
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np

from ship_muon_bg.data_contracts import dataset_hash, load_muon_pkl, run_checks, schema

AUDIT_SCHEMA_VERSION = "0"

QUANTILE_LEVELS = (0.0, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 1.0)

# Bound the duplicate-check and scatter-overview sample so a multi-million-row
# file stays cheap to audit; the report labels these results as sampled.
DEFAULT_SAMPLE_SIZE = 200_000
DEFAULT_SAMPLE_SEED = 20260720

# Rounding used to call two rows "near-duplicate" in physical (px,py,pz,x,y,z)
# space. Not a physics claim -- a coarse, documented proxy only.
NEAR_DUP_DECIMALS = 3


def file_sha256(path):
    """SHA-256 of the raw file bytes on disk (distinct from the content hash,
    which hashes canonicalized array bytes and ignores file-level framing)."""
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _sign_counts(col):
    return {
        "positive": int(np.count_nonzero(col > 0)),
        "zero": int(np.count_nonzero(col == 0)),
        "negative": int(np.count_nonzero(col < 0)),
    }


def _quantiles(col):
    return {str(q): float(np.quantile(col, q)) for q in QUANTILE_LEVELS}


def _column_summary(array, weights=None):
    summary = {}
    for name, idx in schema.COLUMN_INDEX.items():
        col = array[:, idx]
        entry = {
            "min": float(np.min(col)),
            "max": float(np.max(col)),
            "mean": float(np.mean(col)),
            "std": float(np.std(col)),
            "quantiles": _quantiles(col),
        }
        if weights is not None:
            wsum = float(np.sum(weights))
            entry["weighted_mean"] = (
                float(np.sum(col * weights) / wsum) if wsum > 0 else None
            )
        summary[name] = entry
    return summary


def _duplicate_stats(sample, *, near_dup_decimals=NEAR_DUP_DECIMALS):
    """Exact and near-duplicate counts on an in-memory sample (not the full set)."""
    n = sample.shape[0]
    _, exact_counts = np.unique(sample, axis=0, return_counts=True)
    exact_dup_rows = int(np.sum(exact_counts) - exact_counts.size)

    rounded = np.round(sample, near_dup_decimals)
    _, near_counts = np.unique(rounded, axis=0, return_counts=True)
    near_dup_rows = int(np.sum(near_counts) - near_counts.size)

    return {
        "sample_size": int(n),
        "exact_duplicate_row_count": exact_dup_rows,
        "exact_duplicate_rate": exact_dup_rows / n if n else None,
        "near_duplicate_decimals": near_dup_decimals,
        "near_duplicate_row_count": near_dup_rows,
        "near_duplicate_rate": near_dup_rows / n if n else None,
        "note": (
            "Computed on a bounded deterministic sample, not the full dataset. "
            "Row-level (near-)duplicate checks do not resolve possible "
            "source-muon lineage leakage: no source-muon identifier is "
            "available in this contract."
        ),
    }


def build_afterms_audit(
    path,
    *,
    sample_size=DEFAULT_SAMPLE_SIZE,
    sample_seed=DEFAULT_SAMPLE_SEED,
):
    """Load ``path`` exactly once and build the full Phase A audit dict.

    Returns a JSON-serializable mapping. Never infers DIS labels, acceptance,
    or source-level independence -- see the ``non_claims`` field.
    """
    array = load_muon_pkl(path)
    n_rows, n_cols = array.shape

    pdg_col = array[:, schema.COLUMN_INDEX[schema.ID_COLUMN]]
    weight_col = array[:, schema.COLUMN_INDEX[schema.WEIGHT_COLUMN]]
    pz_col = array[:, schema.COLUMN_INDEX["pz"]]

    values, counts = np.unique(np.rint(pdg_col).astype(np.int64), return_counts=True)
    pdg_counts = {str(int(v)): int(c) for v, c in zip(values, counts)}

    rng = np.random.default_rng(sample_seed)
    sample_n = min(sample_size, n_rows)
    sample_idx = np.sort(rng.choice(n_rows, size=sample_n, replace=False))
    sample = array[sample_idx]

    corr = np.corrcoef(array, rowvar=False)

    scatter_idx = sample_idx[: min(5000, sample_idx.size)]

    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "source_path": os.path.abspath(path),
        "file_sha256": file_sha256(path),
        "content_dataset_hash": dataset_hash(array),
        "n_rows": int(n_rows),
        "n_columns": int(n_cols),
        "columns": list(schema.COLUMNS),
        "dtype": str(array.dtype),
        "finite_value_counts": {
            "finite": int(np.count_nonzero(np.isfinite(array))),
            "non_finite": int(np.count_nonzero(~np.isfinite(array))),
        },
        "pdg_counts": pdg_counts,
        "weight_sign_counts": _sign_counts(weight_col),
        "pz_sign_counts": _sign_counts(pz_col),
        "duplicate_stats_sampled": _duplicate_stats(sample),
        "column_summary_unweighted": _column_summary(array),
        "column_summary_weighted": _column_summary(array, weights=weight_col),
        "correlation_matrix": {
            "columns": list(schema.COLUMNS),
            "matrix": corr.tolist(),
        },
        "scatter_overview_sample_row_indices": scatter_idx.tolist(),
        "sample_seed": int(sample_seed),
        "contract_validation": run_checks(array, allow_zero_weight=True),
        "non_claims": [
            "Does not infer DIS labels.",
            "Does not infer detector acceptance.",
            "Does not infer physical validity beyond units-sanity bounds.",
            "Does not infer source-level (per-parent-muon) independence.",
            "Does not infer background-dangerous regions.",
            "Row-level duplicate checks do not resolve possible source-muon "
            "lineage leakage; no source identifier exists in this contract.",
        ],
    }


def write_afterms_audit(audit, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "afterms_audit.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return out_path
