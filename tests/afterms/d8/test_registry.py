"""Phase B run registry: null handling, loss semantics, exact-lookup coverage."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ship_muon_bg.afterms.d8 import registry


def test_registry_builds_expected_run_count(input_artifact_dir, shard_dir):
    records = registry.build_registry(input_artifact_dir, shard_dir)
    run_ids = {r.run_id for r in records}
    assert "04_legacy_available_code_realnvp_quantile__default" in run_ids
    assert "05_affine_preprocessing_ab_pdg13__identity_standardized_v0_affine_small_unweighted" in run_ids
    assert len(records) == 1 + 1 + 3  # job04 default, job05 one run, job10 three baselines


def test_missing_physical_nll_is_null_not_zero_or_nan(input_artifact_dir, shard_dir):
    records = registry.build_registry(input_artifact_dir, shard_dir)
    job04 = next(r for r in records if r.job_id == "04")
    assert job04.physical_space_nll is None  # quantile-family legacy run: genuinely undefined


def test_missing_checkpoint_is_null_and_status_is_missing(input_artifact_dir, shard_dir):
    records = registry.build_registry(input_artifact_dir, shard_dir)
    baseline = next(r for r in records if r.model_family == "diagonal_gaussian")
    assert baseline.checkpoint_path is None
    assert baseline.checkpoint_hash is None
    assert baseline.reconstruction_status == "MISSING_HISTORICAL_CHECKPOINT"


def test_feature_and_physical_space_nll_are_distinct_fields(input_artifact_dir, shard_dir):
    records = registry.build_registry(input_artifact_dir, shard_dir)
    affine = next(r for r in records if r.model_family == "affine_coupling")
    assert affine.feature_space_nll != affine.physical_space_nll
    assert affine.physical_space_nll is not None


def test_job04_legacy_target_measure_is_legacy_row_empirical(input_artifact_dir, shard_dir):
    records = registry.build_registry(input_artifact_dir, shard_dir)
    job04 = next(r for r in records if r.job_id == "04")
    assert job04.target_measure == "legacy_row_empirical"
    assert job04.weighting_policy is False
    assert job04.weighting_reduction is None


def test_pdg_and_dimension_recorded_distinctly(input_artifact_dir, shard_dir):
    records = registry.build_registry(input_artifact_dir, shard_dir)
    job04 = next(r for r in records if r.job_id == "04")
    job05 = next(r for r in records if r.job_id == "05")
    assert job04.modeled_dimension == 4
    assert job05.modeled_dimension == 5
    assert job04.pdg_policy == "combined_unfiltered"
    assert job05.pdg_policy == "pdg13"


def test_best_validation_epoch_from_history_not_test_metric(input_artifact_dir, shard_dir):
    records = registry.build_registry(input_artifact_dir, shard_dir)
    job05 = next(r for r in records if r.job_id == "05")
    # fixture history validation losses: [2.0, 1.8, 1.6, 1.5, 1.45] -> best is epoch 5
    assert job05.best_validation_epoch == 5
    assert job05.validation_nll == 1.45
    # test_nll comes from metrics.test_feature_space_nll (1.4), independent of validation.
    assert job05.test_nll == 1.4


def test_registry_json_round_trips(input_artifact_dir, shard_dir):
    records = registry.build_registry(input_artifact_dir, shard_dir)
    payload = [r.to_json_dict() for r in records]
    assert all("run_id" in row and "reconstruction_status" in row for row in payload)
