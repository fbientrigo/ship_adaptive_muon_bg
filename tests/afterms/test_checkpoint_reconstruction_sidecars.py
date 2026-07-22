"""Focused tests proving the checkpoint/preprocessing sidecars written by
run_neural_training_subprocess() are actually reconstructible -- not just
that the code compiles. Guards three of the session's fixes:

- affine-family checkpoints now go through AffineCouplingFlow's own tested
  save()/load() protocol instead of a bare torch.save(state_dict()).
- Gaussian-family baselines (previously saved nothing at all) now persist
  via their existing DiagonalGaussian/FullGaussian/GaussianMixtureEstimator
  .save()/.load().
- PreprocessingPipeline's fitted state (mean/std or QuantileTransformer),
  previously never serialized despite to_dict()/from_dict() existing, is
  now written alongside every checkpoint.

Uses a tiny synthetic PKL fixture and 5-epoch/closed-form training only --
never the real muonsFullMC_afterMS.pkl.
"""

from __future__ import annotations

import gzip
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import scripts.run_afterms_nightly_queue as queue_mod


def _write_tiny_afterms_pkl(path, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    px = rng.normal(size=n)
    py = rng.normal(size=n)
    pz = rng.uniform(0.1, 5.0, size=n)
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    z = rng.normal(size=n)
    ids = rng.choice([13.0, -13.0], size=n)
    w = rng.uniform(0.1, 1.0, size=n)
    arr = np.column_stack((px, py, pz, x, y, z, ids, w)).astype(np.float64)
    with gzip.open(path, "wb") as f:
        pickle.dump(arr, f)
    return arr


def _build_tiny_shards(tmp_path):
    raw_file = tmp_path / "tiny_afterms.pkl.gz"
    _write_tiny_afterms_pkl(raw_file, n=4000)
    shard_dir = tmp_path / "shards"
    artifact_dir = tmp_path / "artifacts"
    import subprocess

    proc = subprocess.run(
        [
            sys.executable, "-u",
            str(REPO_ROOT / "scripts" / "build_afterms_shards.py"),
            "--raw-file", str(raw_file),
            "--shard-dir", str(shard_dir),
            "--artifact-dir", str(artifact_dir),
            "--target-rows", "1500",
            "--job-name", "01_build_afterms_shards",
        ],
        capture_output=True, text=True, timeout=120, cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    # shard_manifest.json's content hash must use the renamed field.
    manifest = json.loads((shard_dir / "shard_manifest.json").read_text())
    assert "content_dataset_hash" in manifest
    assert "dataset_hash" not in manifest
    return shard_dir


def test_affine_checkpoint_and_preprocessing_sidecar_are_reconstructible(tmp_path):
    shard_dir = _build_tiny_shards(tmp_path)
    job_dir = tmp_path / "jobs" / "12_memory_release_repeat_smoke"

    queue_mod.run_neural_training_subprocess(
        "12_memory_release_repeat_smoke", "cpu", str(shard_dir), str(job_dir)
    )

    run_dir = job_dir / "identity_standardized_v0_affine_tiny_unweighted"
    checkpoint_dir = run_dir / "checkpoint"
    assert (checkpoint_dir / "state_dict.pt").exists()
    assert (checkpoint_dir / "model_config.json").exists()
    assert (checkpoint_dir / "checkpoint_hash.txt").exists()
    preprocessing_path = run_dir / "preprocessing.json"
    assert preprocessing_path.exists()

    from Nflow.torch_models.affine_coupling import AffineCouplingFlow
    from ship_muon_bg.afterms.preprocessing import PreprocessingPipeline

    reloaded_estimator = AffineCouplingFlow.load(run_dir, device="cpu")
    reloaded_pipeline = PreprocessingPipeline.from_dict(
        json.loads(preprocessing_path.read_text())
    )
    assert reloaded_pipeline.variant_id == "identity_standardized_v0"
    assert reloaded_pipeline.mean is not None and reloaded_pipeline.std is not None

    probe = np.zeros((3, reloaded_estimator.dimension))
    log_prob = reloaded_estimator.log_prob(probe)
    assert np.all(np.isfinite(log_prob))

    metrics_path = job_dir / "metrics.json"
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text())
    run_metrics = metrics["identity_standardized_v0_affine_tiny_unweighted"]["metrics"]
    assert run_metrics["checkpoint_hash"] == reloaded_estimator.checkpoint_hash()


def test_gaussian_baseline_checkpoints_are_reconstructible(tmp_path):
    shard_dir = _build_tiny_shards(tmp_path)
    job_dir = tmp_path / "jobs" / "10_gaussian_controls_pdg13"

    queue_mod.run_neural_training_subprocess(
        "10_gaussian_controls_pdg13", "cpu", str(shard_dir), str(job_dir)
    )

    from Nflow.baselines.gaussian import DiagonalGaussian, FullGaussian
    from Nflow.baselines.gmm import GaussianMixtureEstimator

    for run_label, loader_cls in (
        ("identity_standardized_v0_diagonal_gaussian_unweighted", DiagonalGaussian),
        ("identity_standardized_v0_full_gaussian_unweighted", FullGaussian),
        ("identity_standardized_v0_gaussian_mixture_unweighted", GaussianMixtureEstimator),
    ):
        run_dir = job_dir / run_label
        assert (run_dir / "model_config.json").exists()
        assert (run_dir / "model_parameters.npz").exists()
        assert (run_dir / "preprocessing.json").exists()

        reloaded = loader_cls.load(run_dir)
        probe = np.zeros((3, reloaded.dimension))
        log_prob = reloaded.log_prob(probe)
        assert np.all(np.isfinite(log_prob))

    metrics = json.loads((job_dir / "metrics.json").read_text())
    for run_label in (
        "identity_standardized_v0_diagonal_gaussian_unweighted",
        "identity_standardized_v0_full_gaussian_unweighted",
        "identity_standardized_v0_gaussian_mixture_unweighted",
    ):
        assert metrics[run_label]["metrics"]["checkpoint_hash"] is not None


def test_legacy_job04_sidecars_present_and_loadable(tmp_path):
    shard_dir = _build_tiny_shards(tmp_path)
    job_dir = tmp_path / "jobs" / "04_legacy_available_code_realnvp_quantile"

    queue_mod.run_neural_training_subprocess(
        "04_legacy_available_code_realnvp_quantile", "cpu", str(shard_dir), str(job_dir)
    )

    assert (job_dir / "legacy_model.pt").exists()
    config = json.loads((job_dir / "legacy_model_config.json").read_text())
    assert config["family"] == "legacy_normalizing_flow"
    assert config["input_dim"] == 4
    assert config["hidden_dim"] == 160
    assert config["n_layers"] == 10
    assert config["feature_order"] == ["px", "py", "pz", "E"]
    assert len(config["checkpoint_hash"]) == 64

    preprocessing = json.loads((job_dir / "legacy_preprocessing.json").read_text())
    assert preprocessing["variant_id"] == "quantile_transformer_legacy"
    # Trusted: this pickle was written moments ago by this same test process
    # from its own tiny fixture (matching PreprocessingPipeline.to_dict()'s
    # existing, documented use of pickle for QuantileTransformer state),
    # never an untrusted/external source.
    qt = pickle.loads(bytes.fromhex(preprocessing["qt_pkl"]))
    sample = qt.transform(np.zeros((2, 4)))
    assert sample.shape == (2, 4)

    import torch
    from Nflow.legacy.utils.flow_models import NormalizingFlow

    model = NormalizingFlow(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        n_layers=config["n_layers"],
    )
    # Trusted: checkpoint written by this same test moments ago (matching
    # AffineCouplingFlow.load()'s existing, identical torch.load() pattern
    # for its own self-generated checkpoints), never an untrusted source.
    state = torch.load(job_dir / "legacy_model.pt", map_location="cpu")
    model.load_state_dict(state)
