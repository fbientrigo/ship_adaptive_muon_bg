"""Orchestration: turn a PKL path into the three v0 artifacts.

This module holds the business logic so that ``scripts/build_dataset_report.py``
stays a thin CLI wrapper. It produces ``dataset_report``, ``split_manifest`` and
``normalization`` dictionaries (JSON-serializable), and a helper to write them.
"""

from __future__ import annotations

import json
import os
import subprocess

import numpy as np

from . import normalization as normalization_mod
from . import schema, validation
from .hashing import dataset_hash as compute_dataset_hash
from .loader import load_muon_pkl
from .splitting import make_split

DATASET_REPORT_SCHEMA_VERSION = "0"


def _git_commit():
    """Best-effort current git commit hash; ``None`` if unavailable.

    No hardcoded paths; runs ``git`` in the current working directory.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _column_stats(array):
    """Per-column min/max/mean/quantiles for the support audit."""
    quantile_levels = [0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0]
    stats = {}
    for name, idx in schema.COLUMN_INDEX.items():
        col = array[:, idx]
        stats[name] = {
            "min": float(np.min(col)),
            "max": float(np.max(col)),
            "mean": float(np.mean(col)),
            "std": float(np.std(col)),
            "quantiles": {
                str(q): float(np.quantile(col, q)) for q in quantile_levels
            },
        }
    return stats


def _id_histogram(array):
    """Histogram of integer-valued PDG ids, and any unexpected ids."""
    ids = array[:, schema.COLUMN_INDEX[schema.ID_COLUMN]]
    rounded = np.rint(ids).astype(int)
    values, counts = np.unique(rounded, return_counts=True)
    histogram = {str(int(v)): int(c) for v, c in zip(values, counts)}
    unexpected = sorted(
        int(v) for v in values if int(v) not in schema.EXPECTED_MUON_IDS
    )
    return histogram, unexpected


def build_dataset_report(array, *, source_path, bounds=None, allow_zero_weight=False):
    """Build the ``dataset_report`` dictionary (does not raise on bad data).

    Validation outcomes are recorded as data via :func:`validation.run_checks`
    so a report can be produced even when the dataset is invalid.
    """
    ds_hash = compute_dataset_hash(array)
    id_hist, unexpected_ids = _id_histogram(array)
    return {
        "schema_version": DATASET_REPORT_SCHEMA_VERSION,
        "contract_version": schema.CONTRACT_VERSION,
        "source_path": str(source_path),
        "git_commit": _git_commit(),
        "dataset_hash": ds_hash,
        "columns": list(schema.COLUMNS),
        "units": dict(schema.UNITS),
        "n_rows": int(array.shape[0]),
        "n_columns": int(array.shape[1]),
        "post_shield_muon_states": True,
        "validation": validation.run_checks(
            array, bounds=bounds, allow_zero_weight=allow_zero_weight
        ),
        "column_stats": _column_stats(array),
        "id_histogram": id_hist,
        "unexpected_ids": unexpected_ids,
    }


def process_pkl(
    path,
    *,
    seed,
    val_fraction=0.2,
    bounds=None,
    allow_zero_weight=False,
    validate=True,
):
    """Load, (optionally) validate, and build all three v0 artifacts.

    Returns a dict with keys ``dataset_report``, ``split_manifest`` and
    ``normalization``. When ``validate`` is true the array must pass the full
    contract (raising a typed error otherwise) before splitting/normalization.
    """
    array = load_muon_pkl(path)
    if validate:
        validation.validate_muon_array(
            array, bounds=bounds, allow_zero_weight=allow_zero_weight
        )

    ds_hash = compute_dataset_hash(array)
    dataset_report = build_dataset_report(
        array, source_path=path, bounds=bounds, allow_zero_weight=allow_zero_weight
    )
    split_manifest = make_split(
        array.shape[0], seed=seed, val_fraction=val_fraction, dataset_hash=ds_hash
    )
    normalization = normalization_mod.fit_normalization(
        array, split_manifest["train_indices"], dataset_hash=ds_hash
    )
    return {
        "dataset_report": dataset_report,
        "split_manifest": split_manifest,
        "normalization": normalization,
    }


def write_artifacts(artifacts, output_dir):
    """Write the three artifacts as JSON into ``output_dir``; return their paths."""
    os.makedirs(output_dir, exist_ok=True)
    filenames = {
        "dataset_report": "dataset_report.json",
        "split_manifest": "split_manifest.json",
        "normalization": "normalization.json",
    }
    written = {}
    for key, filename in filenames.items():
        out_path = os.path.join(output_dir, filename)
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(artifacts[key], handle, indent=2, sort_keys=True)
            handle.write("\n")
        written[key] = out_path
    return written
