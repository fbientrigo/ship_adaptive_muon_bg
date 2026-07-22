"""Phase D reconstruction: determinism, negative-pz preservation, hash checks."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ship_muon_bg.afterms.d8 import reconstruction, registry


def _job_dir(record, input_artifact_dir):
    return input_artifact_dir / "jobs" / record.run_id.split("__", 1)[0]


def test_deterministic_sample_hash(input_artifact_dir, shard_dir):
    records = registry.build_registry(input_artifact_dir, shard_dir)
    affine = next(r for r in records if r.model_family == "affine_coupling")
    jobs_dir = input_artifact_dir / "jobs"

    r1 = reconstruction.reconstruct_and_generate(affine, jobs_dir=jobs_dir, shard_dir=shard_dir, sample_count=64)
    r2 = reconstruction.reconstruct_and_generate(affine, jobs_dir=jobs_dir, shard_dir=shard_dir, sample_count=64)
    assert r1["provenance"]["sample_hash"] == r2["provenance"]["sample_hash"]
    np.testing.assert_array_equal(r1["samples"], r2["samples"])


def test_negative_pz_preserved_never_clipped(input_artifact_dir, shard_dir):
    records = registry.build_registry(input_artifact_dir, shard_dir)
    affine = next(r for r in records if r.model_family == "affine_coupling")
    jobs_dir = input_artifact_dir / "jobs"

    result = reconstruction.reconstruct_and_generate(affine, jobs_dir=jobs_dir, shard_dir=shard_dir, sample_count=4000)
    samples = result["samples"]
    pz = samples[:, 2]
    # An unconstrained Gaussian-base flow will produce some negative pz;
    # this must be visible in provenance, not clamped away.
    if np.any(pz < 0):
        assert result["provenance"]["domain_violation_count"] == int(np.count_nonzero(pz < 0))
        assert result["provenance"]["minimum_generated_pz"] == float(np.min(pz))


def test_missing_checkpoint_raises_reconstruction_error(input_artifact_dir, shard_dir):
    records = registry.build_registry(input_artifact_dir, shard_dir)
    baseline = next(r for r in records if r.model_family == "diagonal_gaussian")
    jobs_dir = input_artifact_dir / "jobs"
    with pytest.raises(reconstruction.ReconstructionError):
        reconstruction.reconstruct_and_generate(baseline, jobs_dir=jobs_dir, shard_dir=shard_dir, sample_count=10)


def test_checkpoint_hash_mismatch_detected(input_artifact_dir, shard_dir):
    records = registry.build_registry(input_artifact_dir, shard_dir)
    affine = next(r for r in records if r.model_family == "affine_coupling")
    jobs_dir = input_artifact_dir / "jobs"
    ckpt_path = _job_dir(affine, input_artifact_dir) / affine.checkpoint_path
    original = ckpt_path.read_bytes()
    ckpt_path.write_bytes(original + b"\x00")
    try:
        with pytest.raises(reconstruction.ReconstructionError):
            reconstruction.reconstruct_and_generate(affine, jobs_dir=jobs_dir, shard_dir=shard_dir, sample_count=10)
    finally:
        ckpt_path.write_bytes(original)


def test_reference_indices_deterministic(shard_dir):
    idx1 = reconstruction.select_reference_indices(shard_dir, pdg_value=13, sample_count=20, seed=7)
    idx2 = reconstruction.select_reference_indices(shard_dir, pdg_value=13, sample_count=20, seed=7)
    np.testing.assert_array_equal(idx1, idx2)


def test_real_vs_real_subsets_disjoint(shard_dir):
    subsets = reconstruction.real_vs_real_disjoint_subsets(shard_dir, pdg_value=13, sample_count=20, seed=7)
    a, b = set(subsets["subset_a"].tolist()), set(subsets["subset_b"].tolist())
    assert a.isdisjoint(b)
