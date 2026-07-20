"""Pure deterministic blocked maximin Latin-hypercube DOE generation."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from .config import canonical_hash, canonical_json

DEFAULT_FACTORS = {
    "number_of_blocks": {"low": 2, "high": 10, "kind": "integer"},
    "hidden_width": {
        "levels": [32, 48, 64, 96, 128, 192], "kind": "categorical_snap"
    },
    "hidden_depth": {"low": 1, "high": 3, "kind": "integer"},
    "log10_learning_rate": {"low": -4.0, "high": -2.3, "kind": "continuous"},
    "max_log_scale": {"low": 1.0, "high": 5.0, "kind": "continuous"},
    "batch_size": {"levels": [128, 256, 512], "kind": "categorical_snap"},
}

BLOCKS = {
    "A": {"activation": "relu", "mixing_mode": "alternating_only"},
    "B": {"activation": "relu", "mixing_mode": "fixed_random_permutation"},
    "C": {"activation": "silu", "mixing_mode": "fixed_random_permutation"},
}


def _snap(value: float, spec: Mapping[str, Any]):
    if spec["kind"] == "continuous":
        return float(value)
    if spec["kind"] == "integer":
        return int(round(value))
    levels = np.asarray(spec["levels"], dtype=np.float64)
    return int(levels[int(np.argmin(np.abs(levels - value)))])


def _normalized(value: float, spec: Mapping[str, Any]) -> float:
    if spec["kind"] == "categorical_snap":
        levels = spec["levels"]
        return (float(value) - levels[0]) / (levels[-1] - levels[0])
    return (float(value) - spec["low"]) / (spec["high"] - spec["low"])


def _minimum_distance(points: np.ndarray) -> float:
    distance = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    np.fill_diagonal(distance, np.inf)
    return float(distance.min())


def _candidate(rng: np.random.Generator, n: int, dimensions: int) -> np.ndarray:
    points = np.empty((n, dimensions), dtype=np.float64)
    for column in range(dimensions):
        points[:, column] = (rng.permutation(n) + rng.random(n)) / n
    return points


def generate_blocked_maximin_lhs(
    *, doe_seed: int, points_per_block: int = 8,
    factors: Mapping[str, Mapping[str, Any]] = DEFAULT_FACTORS,
    candidate_count: int = 256,
) -> Dict[str, Any]:
    """Return a deterministic, snapped, duplicate-free blocked maximin LHS."""

    names = list(factors)
    rng = np.random.default_rng(int(doe_seed))
    rows: List[Dict[str, Any]] = []
    block_distances = {}
    for block_id, fixed in BLOCKS.items():
        best_rows = None
        best_distance = -1.0
        for _ in range(candidate_count):
            raw = _candidate(rng, points_per_block, len(names))
            snapped = []
            normalized = []
            for point in raw:
                values = {}
                norm = []
                for index, name in enumerate(names):
                    spec = factors[name]
                    low = spec["low"] if "low" in spec else spec["levels"][0]
                    high = spec["high"] if "high" in spec else spec["levels"][-1]
                    value = low + point[index] * (high - low)
                    value = _snap(value, spec)
                    values[name] = value
                    norm.append(_normalized(value, spec))
                snapped.append(values)
                normalized.append(norm)
            keys = {tuple(row[name] for name in names) for row in snapped}
            if len(keys) != points_per_block:
                continue
            distance = _minimum_distance(np.asarray(normalized))
            if distance > best_distance:
                best_rows, best_distance = snapped, distance
        if best_rows is None:
            raise RuntimeError("unable to generate duplicate-free snapped block {}".format(block_id))
        block_distances[block_id] = best_distance
        for values in best_rows:
            log10_learning_rate = values["log10_learning_rate"]
            learning_rate = 10.0 ** log10_learning_rate
            row = dict(values, learning_rate=learning_rate, **fixed)
            row["doe_id"] = (
                "{}_b{:02d}_w{:03d}_d{}_lr{:0.6g}_mls{:0.6g}_bs{:04d}".format(
                    block_id, row["number_of_blocks"], row["hidden_width"],
                    row["hidden_depth"], learning_rate, row["max_log_scale"],
                    row["batch_size"],
                )
            )
            row["block"] = block_id
            rows.append(row)
    payload = {
        "schema_version": "0",
        "doe_seed": int(doe_seed),
        "points_per_block": int(points_per_block),
        "sampling_estimator_contract": {
            "regime": "stratified_self_normalized_provisional",
            "estimator_family": "self_normalized_importance_weighted_minibatch",
            "unbiasedness_status": "not_established",
            "diagnostic_only": False,
            "scientific_scope": "provisional_target_estimator",
        },
        "factors": factors,
        "blocks": BLOCKS,
        "configs": rows,
        "minimum_normalized_pairwise_distance_by_block": block_distances,
    }
    payload["canonical_hash"] = canonical_hash(payload)
    return payload


def write_doe(payload: Mapping[str, Any], output_dir: Path) -> List[Path]:
    """Write deterministic JSON, CSV, and manifest representations."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "d5_memorization_doe_v0.json"
    csv_path = output / "d5_memorization_doe_v0.csv"
    manifest_path = output / "d5_memorization_doe_v0.manifest.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    fields = list(payload["configs"][0])
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(payload["configs"])
    csv_path.write_text(stream.getvalue())
    manifest = {
        "schema_version": "0", "doe_seed": payload["doe_seed"],
        "config_count": len(payload["configs"]),
        "campaign_matrices": {
            "d5_memorization_matrix_v0.json": {
                "target_count": 2,
                "sampling_regime_count": 3,
                "model_count": len(payload["configs"]),
                "planned_run_count": 144,
            },
            "d3_memorization_control_v0.json": {
                "target_count": 1,
                "sampling_regime_count": 1,
                "model_count": len(payload["configs"]),
                "planned_run_count": 24,
            },
            "total_valid_planned_run_count": 168,
        },
        "sampling_estimator_contract": payload["sampling_estimator_contract"],
        "canonical_hash": payload["canonical_hash"],
        "minimum_normalized_pairwise_distance_by_block": payload[
            "minimum_normalized_pairwise_distance_by_block"
        ],
        "canonical_json_sha256": canonical_hash(json.loads(canonical_json(payload))),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return [csv_path, json_path, manifest_path]
