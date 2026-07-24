"""D9-5 cross-seed aggregation (Sec 12).

Mirrors ``ship_muon_bg.afterms.d9.aggregate``'s pattern -- report observed
spread, never gate on it, never select from one favorable seed, never hide a
failed/interrupted seed -- generalized to ``(track_id, model_config_id)`` keys
shared by all four families. Deterministic single fits (GAUSS_DIAG/GAUSS_FULL)
report one value, never an invented multi-seed spread.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def _percentile_iqr(values: List[float]):
    if not values:
        return None
    q1, q3 = np.percentile(values, [25, 75])
    return float(q3 - q1)


def aggregate_seeds(
    track_id: str,
    model_config_id: str,
    per_seed_records: List[Dict[str, Any]],
    *,
    metric_key: str = "physical_nll",
) -> Dict[str, Any]:
    """``per_seed_records``: one dict per attempted seed, each with at least
    ``seed``, ``status`` (``"ok"`` or otherwise), and (if ``status == "ok"``)
    ``metric_key``."""

    completed = [r for r in per_seed_records if r.get("status") == "ok" and r.get(metric_key) is not None]
    failed = [r for r in per_seed_records if r.get("status") != "ok"]
    values = [float(r[metric_key]) for r in completed]

    return {
        "track_id": track_id,
        "model_config_id": model_config_id,
        "metric_key": metric_key,
        "fitting_policy": "multi_seed_stochastic",
        "completed_seed_count": len(completed),
        "failed_seed_count": len(failed),
        "total_seed_count": len(per_seed_records),
        "per_seed": [
            {"seed": r.get("seed"), "status": r.get("status"), metric_key: r.get(metric_key)}
            for r in per_seed_records
        ],
        "mean": float(np.mean(values)) if values else None,
        "median": float(np.median(values)) if values else None,
        "std": float(np.std(values)) if len(values) > 1 else (0.0 if values else None),
        "iqr": _percentile_iqr(values),
    }


def deterministic_fit_record(
    track_id: str, model_config_id: str, record: Dict[str, Any], *, metric_key: str = "physical_nll",
) -> Dict[str, Any]:
    return {
        "track_id": track_id,
        "model_config_id": model_config_id,
        "metric_key": metric_key,
        "fitting_policy": "deterministic_single_fit",
        "completed_seed_count": 1 if record.get(metric_key) is not None else 0,
        "failed_seed_count": 0 if record.get(metric_key) is not None else 1,
        "total_seed_count": 1,
        metric_key: record.get(metric_key),
        "note": "deterministic single fit; no seed spread is reported",
    }
