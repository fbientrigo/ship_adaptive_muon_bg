"""D9 multi-seed aggregation (§13).

Pure functions over a candidate's per-seed ``status.json`` records -- no
ranking decision is ever made from a single favorable seed, no seed's
failure/instability is hidden, and no universal variance threshold is
invented (the spec is explicit about this): this module only *reports*
observed spread, it does not gate on it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from . import runner as d9runner


def _percentile_iqr(values: List[float]):
    if not values:
        return None
    q1, q3 = np.percentile(values, [25, 75])
    return float(q3 - q1)


def load_seed_statuses(artifact_root: Path, candidate_id: str, seeds: List[int]) -> List[Dict[str, Any]]:
    statuses = []
    for seed in seeds:
        run_dir = d9runner.run_directory(artifact_root, candidate_id, seed)
        status_path = run_dir / "status.json"
        if status_path.exists():
            statuses.append(json.loads(status_path.read_text(encoding="utf-8")))
        else:
            statuses.append({"status": "planned", "seed": seed})
    return statuses


def aggregate_candidate(candidate_id: str, seed_statuses: List[Dict[str, Any]]) -> Dict[str, Any]:
    completed = [s for s in seed_statuses if s.get("status") == d9runner.STATUS_COMPLETED]
    failed = [s for s in seed_statuses if s.get("status") == d9runner.STATUS_FAILED_TECHNICAL]
    interrupted = [s for s in seed_statuses if s.get("status") == d9runner.STATUS_INTERRUPTED]

    best_vals = [s["best_validation_metric"] for s in completed if s.get("best_validation_metric") is not None]
    best_epochs = [s["best_validation_epoch"] for s in completed if s.get("best_validation_epoch") is not None]

    result = {
        "candidate_id": candidate_id,
        "completed_seed_count": len(completed),
        "failed_seed_count": len(failed),
        "interrupted_seed_count": len(interrupted),
        "total_seed_count": len(seed_statuses),
        "best_validation_nll_per_seed": [
            {"seed": s.get("seed"), "best_validation_metric": s.get("best_validation_metric")} for s in completed
        ],
        "median_best_validation_nll": float(np.median(best_vals)) if best_vals else None,
        "mean_best_validation_nll": float(np.mean(best_vals)) if best_vals else None,
        "std_best_validation_nll": float(np.std(best_vals)) if len(best_vals) > 1 else (0.0 if best_vals else None),
        "iqr_best_validation_nll": _percentile_iqr(best_vals),
        "best_epoch_distribution": best_epochs,
    }
    return result


def aggregate_campaign(training_config: Dict[str, Any], artifact_root: Path) -> Dict[str, Any]:
    per_candidate = []
    for candidate in training_config["candidates"]:
        statuses = load_seed_statuses(artifact_root, candidate["candidate_id"], candidate["seed_set"])
        per_candidate.append(aggregate_candidate(candidate["candidate_id"], statuses))
    return {"campaign_id": training_config["campaign_id"], "candidates": per_candidate}
