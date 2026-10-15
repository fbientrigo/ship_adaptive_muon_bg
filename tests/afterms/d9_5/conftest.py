"""Shared tiny synthetic fixtures for the D9-5 test suite.

Never touches the real ``data/shards/afterms_nightly_v1`` scope -- every test
builds its own tiny, self-contained shard directory with the same (N, 8)
schema (px, py, pz, x, y, z, id, w).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ship_muon_bg.data_contracts import dataset_hash as compute_dataset_hash


def _make_rows(n: int, *, seed: int, pdg_fraction_positive: float = 0.5, negative_pz_fraction: float = 0.02) -> np.ndarray:
    rng = np.random.default_rng(seed)
    px = rng.normal(0.0, 1.0, n)
    py = rng.normal(0.0, 1.0, n)
    pz = rng.normal(5.0, 1.0, n)
    n_negative = int(round(n * negative_pz_fraction))
    if n_negative:
        idx = rng.choice(n, size=n_negative, replace=False)
        pz[idx] = -np.abs(pz[idx])
    x = rng.normal(0.0, 0.5, n)
    y = rng.normal(0.0, 0.5, n)
    z = np.full(n, 100.0)
    n_pos = int(round(n * pdg_fraction_positive))
    ids = np.concatenate([np.full(n_pos, 13.0), np.full(n - n_pos, -13.0)])
    rng.shuffle(ids)
    w = rng.uniform(0.5, 1.5, n)
    return np.column_stack([px, py, pz, x, y, z, ids, w]).astype(np.float64)


def _write_shard(shard_dir: Path, name: str, rows: np.ndarray) -> dict:
    npy_file = f"{name}.npy"
    np.save(shard_dir / npy_file, rows)
    return {
        "npy_file": npy_file,
        "manifest_file": f"{name}.json",
        "indices_file": f"{name}.indices.npy",
        "row_count": int(rows.shape[0]),
        "shard_hash": compute_dataset_hash(rows),
        "shard_number": 0,
    }


@pytest.fixture
def tiny_shard_dir(tmp_path: Path) -> Path:
    """Two train shards, one validation shard, one test shard; small enough
    for every D9-5 fit path (including a real, tiny NF_AC training run) to
    run in well under a second."""

    shard_dir = tmp_path / "tiny_shards"
    shard_dir.mkdir()

    shards = []
    for i, name in enumerate(["train_shard_000", "train_shard_001"]):
        rows = _make_rows(600, seed=1000 + i)
        entry = _write_shard(shard_dir, name, rows)
        entry["split"] = "train"
        shards.append(entry)
    for name, seed in [("validation_shard_000", 2000), ("test_shard_000", 3000)]:
        rows = _make_rows(300, seed=seed)
        entry = _write_shard(shard_dir, name, rows)
        entry["split"] = "validation" if "validation" in name else "test"
        shards.append(entry)

    manifest = {"dataset_hash": "tiny_synthetic_v0", "seed": 20260720, "shards": shards, "source_file": "synthetic"}
    (shard_dir / "shard_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return shard_dir


@pytest.fixture
def tiny_data_scope_config(tiny_shard_dir: Path, tmp_path: Path) -> dict:
    return {
        "schema_version": "d9_5_data_scope_v0_test",
        "shard_dir": str(tiny_shard_dir),
        "modeled_features": ["px", "py", "pz", "x", "y"],
        "feature_order": ["px", "py", "pz", "x", "y"],
        "modeled_dimension": 5,
        "preprocessing_name": "identity_standardized_v0",
        "tracks": [
            {"track_id": "TRK_PDG13_UW_ID", "pdg_value": 13, "pdg_policy": "pdg13"},
            {"track_id": "TRK_PDGM13_UW_ID", "pdg_value": -13, "pdg_policy": "pdg_minus13"},
        ],
    }


@pytest.fixture
def tiny_scout_report(tmp_path: Path) -> Path:
    """A minimal AFFINE_COUPLING_CAPACITY_SCOUT_V0-shaped report with one
    completed variant per track (plus a worse alternative, to exercise
    argmin selection) and one non-completed row that must be excluded."""

    records = [
        {"track_id": "TRK_PDG13_UW_ID", "status": "completed", "model_config_id": "NF_AC_b02_w008_d01",
         "best_validation_metric": 3.0, "internal_candidate_id": "toy_a__cap_1.0x"},
        {"track_id": "TRK_PDG13_UW_ID", "status": "completed", "model_config_id": "NF_AC_b04_w016_d01",
         "best_validation_metric": 1.5, "internal_candidate_id": "toy_a__cap_2.0x"},
        {"track_id": "TRK_PDG13_UW_ID", "status": "failed_technical", "model_config_id": "NF_AC_b01_w004_d01",
         "best_validation_metric": 0.1, "internal_candidate_id": "toy_a__cap_0.5x"},
        {"track_id": "TRK_PDGM13_UW_ID", "status": "completed", "model_config_id": "NF_AC_b02_w008_d01",
         "best_validation_metric": 2.2, "internal_candidate_id": "toy_b__cap_1.0x"},
    ]
    path = tmp_path / "tiny_run_inventory_named.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


TINY_NF_EXECUTION_POLICY = {"minimum_epochs": 1, "maximum_epochs": 2, "early_stopping_patience": 1}
TINY_NF_OPTIMIZER_SETTINGS = {
    "optimizer": "adam", "learning_rate": 0.01, "batch_size": 64, "weight_decay": 0.0,
    "gradient_clipping": 5.0, "dtype": "float32", "device_policy": "cpu_only_test",
}
TINY_NF_ARCHITECTURE = {"number_of_blocks": 2, "hidden_width": 8, "hidden_depth": 1}
TINY_EVALUATION_POLICY = {
    "quick_validation_budget": {
        "n_generated_samples": 64, "c2st_subsample": 32, "one_d_test_subsample": 64,
        "two_d_test_subsample": 32, "one_d_n_boot": 20, "two_d_n_permutations": 20,
        "c2st_n_boot": 20, "c2st_n_permutations": 20,
    },
    "final_candidate_budget": {
        "n_generated_samples": 128, "c2st_subsample": 64, "one_d_test_subsample": 128,
        "two_d_test_subsample": 64, "one_d_n_boot": 20, "two_d_n_permutations": 20,
        "c2st_n_boot": 20, "c2st_n_permutations": 20,
    },
}


@pytest.fixture
def tiny_nf_execution_policy() -> dict:
    return dict(TINY_NF_EXECUTION_POLICY)


@pytest.fixture
def tiny_nf_optimizer_settings() -> dict:
    return dict(TINY_NF_OPTIMIZER_SETTINGS)


@pytest.fixture
def tiny_nf_architecture() -> dict:
    return dict(TINY_NF_ARCHITECTURE)


@pytest.fixture
def tiny_evaluation_policy() -> dict:
    return json.loads(json.dumps(TINY_EVALUATION_POLICY))
