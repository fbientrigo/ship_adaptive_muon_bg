"""Tiny synthetic D8 fixtures.

Builds a minimal `afterms_nightly_v1`-shaped artifact tree + shard tree under
`tmp_path`, small enough to exercise the adapter/registry/arenas logic without
ever touching the real (13.7M-row) frozen campaign or raw PKL.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ship_muon_bg.afterms.d8 import legacy_d7_adapter as adapter  # noqa: E402

N_TRAIN, N_VAL, N_TEST = 400, 100, 100


def _make_shard_array(n, seed, pdg_mix=True):
    rng = np.random.default_rng(seed)
    px = rng.normal(size=n)
    py = rng.normal(size=n)
    pz = rng.uniform(0.1, 5.0, size=n)
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    z = rng.normal(size=n)
    ids = rng.choice([13.0, -13.0], size=n) if pdg_mix else np.full(n, 13.0)
    w = rng.uniform(0.1, 1.0, size=n)
    return np.column_stack((px, py, pz, x, y, z, ids, w)).astype(np.float64)


def _write_shard_sidecar(shard_dir: Path, split: str, row_count: int):
    payload = {
        "construction_seed": adapter.TRAINING_SEED,
        "dataset_hash": "content" + "0" * 58,
        "row_count": row_count,
        "shard_hash": f"shardhash_{split}",
        "indices_hash": f"idxhash_{split}",
        "shard_number": 0,
        "source_file": "fixture.pkl",
        "split": split,
    }
    (shard_dir / f"{split}_shard_000.json").write_text(json.dumps(payload))


@pytest.fixture
def raw_pkl(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "data" / "raw" / "nflow_releases"
    raw_dir.mkdir(parents=True)
    path = raw_dir / "muonsFullMC_afterMS.pkl"
    path.write_bytes(b"tiny fixture bytes, not a real pkl")
    return path


@pytest.fixture
def shard_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data" / "shards" / "afterms_nightly_v1"
    d.mkdir(parents=True)
    np.save(d / "train_shard_000.npy", _make_shard_array(N_TRAIN, seed=1))
    np.save(d / "validation_shard_000.npy", _make_shard_array(N_VAL, seed=2))
    np.save(d / "test_shard_000.npy", _make_shard_array(N_TEST, seed=3))
    _write_shard_sidecar(d, "train", N_TRAIN)
    _write_shard_sidecar(d, "validation", N_VAL)
    _write_shard_sidecar(d, "test", N_TEST)
    (d / "shard_manifest.json").write_text(json.dumps({
        "dataset_hash": "c" * 64,
        "seed": adapter.TRAINING_SEED,
        "source_file": "fixture.pkl",
        "shards": [],
        "splits": {},
    }))
    return d


def _epoch_history(n_epochs, val_losses):
    return [
        {"epoch": i + 1, "train_loss": val_losses[i] + 0.1, "validation_loss": val_losses[i]}
        for i in range(n_epochs)
    ]


@pytest.fixture
def input_artifact_dir(tmp_path: Path, raw_pkl: Path) -> Path:
    root = tmp_path / "artifacts" / "afterms_nightly_v1"
    jobs = root / "jobs"
    jobs.mkdir(parents=True)

    file_sha256 = adapter.checkpoint_file_sha256(raw_pkl)
    (root / "afterms_audit.json").write_text(json.dumps({
        "file_sha256": file_sha256,
        "content_dataset_hash": "c" * 64,
        "source_path": str(raw_pkl),
        "schema_version": 0,
        "n_rows": N_TRAIN + N_VAL + N_TEST,
    }))

    report_dir = root / "report"
    report_dir.mkdir()
    (report_dir / "nightly_summary.json").write_text(json.dumps({
        "git_commit": adapter.PRODUCER_GIT_COMMIT,
        "dataset_hash": file_sha256[:16],
        "job_statuses": {name: "completed" for name in adapter.JOB_NAMES},
        "status_code": "NIGHTLY_SMOKES_COMPLETE",
    }))

    (root / "queue_state.json").write_text(json.dumps({
        "completed_jobs": list(adapter.JOB_NAMES[:-1]),
        "failed_jobs": [],
        "pending_jobs": [],
        "active_job": None,
        "pid": None,
    }))

    for name in adapter.JOB_NAMES:
        (jobs / name).mkdir(parents=True)

    # Job 04: legacy single-run.
    (jobs / "04_legacy_available_code_realnvp_quantile" / "metrics.json").write_text(json.dumps({
        "history": _epoch_history(5, [3.0, 2.5, 2.2, 2.0, 1.9]),
        "metrics": {
            "test_feature_space_nll": 1.85,
            "physical_space_nll": None,
            "parameter_count": 123,
            "wall_time_seconds": 1.0,
            "generation_seed": adapter.GENERATION_SEED,
            "checkpoint_hash": "deadbeef",
        },
        "reproduction_scope": "available_code_semantics",
    }))
    (jobs / "04_legacy_available_code_realnvp_quantile" / "status.json").write_text(
        json.dumps({"status": "completed", "job_name": "04_legacy_available_code_realnvp_quantile", "git_commit": adapter.PRODUCER_GIT_COMMIT})
    )

    # Job 13 status (authoritative completion signal).
    (jobs / "13_build_nightly_report" / "status.json").write_text(
        json.dumps({"status": "completed", "job_name": "13_build_nightly_report", "git_commit": adapter.PRODUCER_GIT_COMMIT})
    )

    # Job 05: affine_small, identity_standardized_v0, pdg13, unweighted --
    # a real (untrained, tiny) checkpoint so hash/load tests are genuine.
    import torch

    job05_dir = jobs / "05_affine_preprocessing_ab_pdg13"
    label = adapter.run_label("identity_standardized_v0", "affine_small", weighted=False)
    estimator = adapter.build_frozen_affine_estimator("affine_small", device="cpu")
    ckpt_path = job05_dir / f"{label}_model.pt"
    torch.save(estimator._module.state_dict(), ckpt_path)
    ckpt_hash = adapter.checkpoint_file_sha256(ckpt_path)
    (job05_dir / "metrics.json").write_text(json.dumps({
        label: {
            "history": _epoch_history(5, [2.0, 1.8, 1.6, 1.5, 1.45]),
            "metrics": {
                "test_feature_space_nll": 1.4,
                "physical_space_nll": 3.0,
                "parameter_count": estimator.parameter_count(),
                "wall_time_seconds": 2.0,
                "generation_seed": adapter.GENERATION_SEED,
                "checkpoint_hash": ckpt_hash,
            },
        }
    }))

    # Job 10: Gaussian/GMM baselines, pdg13, unweighted -- no checkpoint.
    job10_dir = jobs / "10_gaussian_controls_pdg13"
    baseline_metrics = {}
    for model_name, base_val in (("diagonal_gaussian", 5.0), ("full_gaussian", 4.0), ("gaussian_mixture", 3.0)):
        label10 = adapter.run_label("identity_standardized_v0", model_name, weighted=False)
        baseline_metrics[label10] = {
            "history": [{"epoch": 1, "train_loss": base_val, "validation_loss": base_val}],
            "metrics": {
                "test_feature_space_nll": base_val - 0.05,
                "physical_space_nll": base_val + 1.5,
                "parameter_count": 20,
                "wall_time_seconds": 0.5,
                "generation_seed": adapter.GENERATION_SEED,
                "checkpoint_hash": None,
            },
        }
    (job10_dir / "metrics.json").write_text(json.dumps(baseline_metrics))

    return root
