"""Orchestration: turn a PKL/NPZ path into the v0 report/split/normalization
artifacts, plus a bounded, train-free dataset validation report.

This module holds the business logic so that ``scripts/build_dataset_report.py``
stays a thin CLI wrapper. It produces ``dataset_report``, ``split_manifest`` and
``normalization`` dictionaries (JSON-serializable), and a helper to write them.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any, Dict, Mapping, Optional

import numpy as np

from . import normalization as normalization_mod
from . import schema, validation
from .hashing import dataset_hash as compute_dataset_hash
from .loader import load_muon_pkl
from .pdg import pdg_counts as compute_pdg_counts
from .splitting import make_split, make_three_way_split
from .subsampling import load_muon_npz, representative_subset

DATASET_REPORT_SCHEMA_VERSION = "0"
DATASET_VALIDATION_REPORT_SCHEMA_VERSION = "0"

# Bytes per float64 element, used only for the memory-footprint estimate.
_BYTES_PER_FLOAT64 = 8
# Above this row count, the exact-duplicate-row check (np.unique over all
# columns) is skipped rather than silently run at a memory cost nobody
# asked for; a full local afterMS file (~13.8M rows) is well past this.
DEFAULT_DUPLICATE_CHECK_ROW_LIMIT = 2_000_000


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


def process_array(
    array,
    *,
    source_path,
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
    ``source_path`` is recorded for provenance only; ``array`` is used as
    given (already loaded, and already capped if the caller wants a bounded
    run), so this is the single business-logic entry point for both
    :func:`process_pkl` and any NPZ/full-local-dataset caller.
    """
    if validate:
        validation.validate_muon_array(
            array, bounds=bounds, allow_zero_weight=allow_zero_weight
        )

    ds_hash = compute_dataset_hash(array)
    dataset_report = build_dataset_report(
        array, source_path=source_path, bounds=bounds, allow_zero_weight=allow_zero_weight
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


def process_pkl(
    path,
    *,
    seed,
    val_fraction=0.2,
    bounds=None,
    allow_zero_weight=False,
    validate=True,
):
    """``process_array`` for a gzip-PKL path specifically (unchanged v0 API)."""

    array = load_muon_pkl(path)
    return process_array(
        array,
        source_path=path,
        seed=seed,
        val_fraction=val_fraction,
        bounds=bounds,
        allow_zero_weight=allow_zero_weight,
        validate=validate,
    )


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


# --- bounded, train-free dataset validation (repository fixture and full local data) ---


def load_muon_array(path):
    """Load a muon array, dispatching on file extension.

    ``.npz`` uses :func:`subsampling.load_muon_npz` (no code execution on
    load, preferred); anything else is treated as a gzip-PKL and uses
    :func:`loader.load_muon_pkl`. This is the single dispatch point so the
    validation CLI and any other caller accept either committed-sample format
    without duplicating format-detection logic.
    """

    if str(path).endswith(".npz"):
        return load_muon_npz(path)
    return load_muon_pkl(path)


def cap_rows(array, max_rows, *, seed):
    """Deterministically cap ``array`` to at most ``max_rows`` rows.

    Delegates to :func:`subsampling.representative_subset` (uniform core plus
    per-column range anchors) rather than a plain head/tail truncation or an
    unweighted random draw, so a bounded validation run still reports the
    dataset's true observed envelope. ``max_rows=None`` returns ``array``
    unchanged (no cap).
    """

    if max_rows is None:
        return array, array.shape[0]
    available = int(array.shape[0])
    subset, _selected = representative_subset(array, int(max_rows), seed=seed)
    return subset, available


def _weight_column_status(array, *, allow_zero_weight):
    w = array[:, schema.COLUMN_INDEX[schema.WEIGHT_COLUMN]]
    finite = np.isfinite(w)
    n_zero = int(np.count_nonzero(w[finite] == 0.0)) if finite.any() else 0
    n_negative = int(np.count_nonzero(w[finite] < 0.0)) if finite.any() else 0
    return {
        # The weight column is a fixed part of the (N, 8) schema, so it is
        # always structurally present; this field exists so a reader never
        # has to assume that from the schema alone.
        "present": True,
        "n_non_finite": int(np.count_nonzero(~finite)),
        "n_zero": n_zero,
        "n_negative": n_negative,
        "min": float(np.min(w)) if w.size else None,
        "max": float(np.max(w)) if w.size else None,
        "mean": float(np.mean(w[finite])) if finite.any() else None,
        "all_positive": bool(finite.all() and n_zero == 0 and n_negative == 0),
        "allow_zero_weight_policy": bool(allow_zero_weight),
        # This is the *physical/event* weight recorded in the source data. It
        # is orthogonal to any sampling-correction weight the rare-aware
        # minibatch estimator contract computes (docs/contracts/
        # rare_aware_minibatch_estimators_v0.md, assumption A6); row-empirical
        # D7 training does not apply it to the loss by default -- see
        # density_lab.empirical.
        "semantics": "physical_event_weight_from_source_data_not_a_sampling_correction",
    }


def _duplicate_row_status(array, *, row_limit):
    n_rows = int(array.shape[0])
    if n_rows > row_limit:
        return {
            "checked": False,
            "duplicate_row_count": None,
            "reason": (
                "row count {} exceeds the bounded duplicate-check limit {} "
                "(exact-duplicate detection sorts all columns and is memory-"
                "intensive at full-dataset scale); rerun with a smaller "
                "--max-rows to check duplicates, or pass a higher limit "
                "explicitly".format(n_rows, row_limit)
            ),
        }
    unique_rows = np.unique(array, axis=0)
    return {
        "checked": True,
        "reason": None,
        "n_rows": n_rows,
        "n_unique_rows": int(unique_rows.shape[0]),
        "duplicate_row_count": int(n_rows - unique_rows.shape[0]),
    }


def _source_identifier_status():
    return {
        "status": "not_applicable_no_source_event_identifier_column",
        "detail": (
            "the (N, 8) contract [px, py, pz, x, y, z, id, w] has no unique "
            "per-event source identifier column; duplicated rows (identical "
            "8-tuples, see duplicate_rows) are detectable, but a duplicated "
            "*source event* that happened to produce two distinct rows is not"
        ),
    }


def _memory_footprint_estimate(n_rows, *, n_columns=schema.N_COLUMNS):
    raw_bytes = int(n_rows) * int(n_columns) * _BYTES_PER_FLOAT64
    return {
        "raw_array_bytes": raw_bytes,
        "raw_array_mib": raw_bytes / (1024.0 ** 2),
        "note": (
            "raw float64 (N, {}) array only; does not include feature-view or "
            "normalized copies, torch tensors, or multiple PDG-filtered splits "
            "held simultaneously in memory during training".format(n_columns)
        ),
    }


def _split_feasibility(n_rows, *, seed, val_fraction, test_fraction):
    if seed is None:
        return {
            "checked": False,
            "feasible": None,
            "reason": "no seed given; split feasibility not evaluated",
        }
    try:
        split = make_three_way_split(
            int(n_rows), seed=seed, val_fraction=val_fraction, test_fraction=test_fraction
        )
    except (TypeError, ValueError) as exc:
        return {"checked": True, "feasible": False, "reason": str(exc)}
    return {
        "checked": True,
        "feasible": True,
        "reason": None,
        "n_train": split["n_train"],
        "n_val": split["n_val"],
        "n_test": split["n_test"],
    }


def _campaign_config_compatibility(
    id_counts: Mapping[int, int], *, campaign_config: Optional[Mapping[str, Any]]
) -> Dict[str, Any]:
    """Cross-check available rows per requested PDG id against a campaign config.

    ``campaign_config`` is treated as an untyped, duck-typed mapping (no
    import of ``density_lab`` from this layer, which must stay import-light
    and never depend on the modelling stack): only the keys
    ``dataset.n_train`` / ``dataset.n_validation`` / ``dataset.n_test`` and
    ``pdg_ids`` are read, if present.
    """

    if campaign_config is None:
        return {"checked": False, "reason": "no campaign config given"}
    dataset_cfg = dict(campaign_config.get("dataset") or {})
    requested_pdg_ids = campaign_config.get("pdg_ids") or list(schema.EXPECTED_MUON_IDS)
    required_total = sum(
        int(dataset_cfg[key])
        for key in ("n_train", "n_validation", "n_test")
        if isinstance(dataset_cfg.get(key), (int, float))
    )
    per_pdg = {}
    all_sufficient = True
    for pdg_id in requested_pdg_ids:
        available = int(id_counts.get(int(pdg_id), 0))
        sufficient = available >= required_total if required_total else True
        all_sufficient = all_sufficient and sufficient
        per_pdg[str(int(pdg_id))] = {
            "available_rows": available,
            "required_rows": required_total,
            "sufficient": sufficient,
        }
    return {
        "checked": True,
        "requested_pdg_ids": [int(p) for p in requested_pdg_ids],
        "required_rows_per_pdg": required_total,
        "per_pdg": per_pdg,
        "all_requested_pdg_tracks_sufficient": all_sufficient,
    }


def build_validation_report(
    array,
    *,
    source_path,
    bounds=None,
    allow_zero_weight=False,
    requested_max_rows=None,
    available_rows_before_cap=None,
    seed=None,
    val_fraction=0.2,
    test_fraction=0.2,
    campaign_config=None,
    duplicate_check_row_limit=DEFAULT_DUPLICATE_CHECK_ROW_LIMIT,
):
    """Bounded, machine-readable dataset validation report. Never trains.

    A superset of :func:`build_dataset_report`'s fields (schema, validation
    checks, column stats, id histogram are all still present) plus the
    fields the ``--validate-only`` CLI mode needs: PDG counts restricted to
    the expected muon ids, an explicit weight-column status, a bounded exact-
    duplicate-row check, an explicit note on why duplicated *source events*
    cannot be detected from this schema, requested-vs-available row counts,
    a memory-footprint estimate, split feasibility, and compatibility with an
    optional campaign configuration.
    """

    base = build_dataset_report(
        array, source_path=source_path, bounds=bounds, allow_zero_weight=allow_zero_weight
    )
    id_counts = compute_pdg_counts(array)
    expected_pdg_counts = {
        str(pdg_id): id_counts.get(pdg_id, 0) for pdg_id in schema.EXPECTED_MUON_IDS
    }
    base.update({
        "report_schema_version": DATASET_VALIDATION_REPORT_SCHEMA_VERSION,
        "validate_only": True,
        "expected_pdg_counts": expected_pdg_counts,
        "weight_column_status": _weight_column_status(array, allow_zero_weight=allow_zero_weight),
        "duplicate_rows": _duplicate_row_status(array, row_limit=duplicate_check_row_limit),
        "duplicate_source_identifiers": _source_identifier_status(),
        "row_budget": {
            "requested_max_rows": requested_max_rows,
            "available_rows_before_cap": available_rows_before_cap,
            "rows_used": int(array.shape[0]),
        },
        "memory_footprint_estimate": _memory_footprint_estimate(array.shape[0]),
        "split_feasibility": _split_feasibility(
            array.shape[0], seed=seed, val_fraction=val_fraction, test_fraction=test_fraction
        ),
        "campaign_config_compatibility": _campaign_config_compatibility(
            id_counts, campaign_config=campaign_config
        ),
    })
    return base
