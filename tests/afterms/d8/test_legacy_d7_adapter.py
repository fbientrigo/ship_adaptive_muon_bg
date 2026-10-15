"""Legacy D7 adapter: hash normalization, checkpoint contract, Phase A audit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ship_muon_bg.afterms.d8 import legacy_d7_adapter as adapter


def test_raw_file_hash_relation_holds_on_matching_fixture(input_artifact_dir, shard_dir, raw_pkl, tmp_path):
    nightly_summary = json.loads((input_artifact_dir / "report" / "nightly_summary.json").read_text())
    prefix16 = adapter.legacy_raw_file_sha256_prefix16(nightly_summary)
    recomputed = adapter.recompute_raw_file_sha256(tmp_path)
    assert adapter.verify_raw_file_hash_relation(prefix16, recomputed)


def test_raw_file_hash_relation_rejects_mismatch():
    assert not adapter.verify_raw_file_hash_relation("deadbeefdeadbeef", "00000000000000000000000000000000000000000000000000000000000000")


def test_legacy_prefix16_rejects_malformed_field():
    with pytest.raises(adapter.LegacyContractError):
        adapter.legacy_raw_file_sha256_prefix16({"dataset_hash": "not-16-hex-chars-but-close"})


def test_never_compares_content_hash_to_raw_hash_directly(input_artifact_dir, shard_dir):
    afterms_audit = json.loads((input_artifact_dir / "afterms_audit.json").read_text())
    shard_manifest = json.loads((shard_dir / "shard_manifest.json").read_text())
    raw_hash = adapter.audit_raw_file_sha256(afterms_audit)
    content_hash = adapter.audit_content_dataset_hash(afterms_audit)
    shard_hash = adapter.shard_manifest_content_hash(shard_manifest)
    assert raw_hash != content_hash  # distinct quantities by construction of the fixture
    assert content_hash == shard_hash  # same quantity, two producers


def test_phase_a_audit_valid_on_consistent_fixture(tmp_path, input_artifact_dir, shard_dir):
    result = adapter.run_phase_a_audit(tmp_path, input_artifact_dir, shard_dir)
    assert result["classification"] == adapter.INPUT_VALID


def test_phase_a_audit_inconsistent_when_raw_file_mutated(tmp_path, input_artifact_dir, shard_dir, raw_pkl):
    raw_pkl.write_bytes(b"mutated bytes that change the sha256")
    result = adapter.run_phase_a_audit(tmp_path, input_artifact_dir, shard_dir)
    assert result["classification"] == adapter.INPUT_INCONSISTENT


def test_phase_a_audit_does_not_mutate_inputs(tmp_path, input_artifact_dir, shard_dir, raw_pkl):
    before = raw_pkl.read_bytes()
    before_audit = (input_artifact_dir / "afterms_audit.json").read_bytes()
    adapter.run_phase_a_audit(tmp_path, input_artifact_dir, shard_dir)
    assert raw_pkl.read_bytes() == before
    assert (input_artifact_dir / "afterms_audit.json").read_bytes() == before_audit


def test_stale_queue_state_job13_failed_does_not_block(tmp_path, input_artifact_dir, shard_dir):
    queue_state_path = input_artifact_dir / "queue_state.json"
    state = json.loads(queue_state_path.read_text())
    state["failed_jobs"] = ["13_build_nightly_report"]
    queue_state_path.write_text(json.dumps(state))
    result = adapter.run_phase_a_audit(tmp_path, input_artifact_dir, shard_dir)
    assert result["classification"] == adapter.INPUT_VALID
    assert any("queue_state.json" in f for f in result["findings"])


def test_run_label_naming():
    assert adapter.run_label("identity_standardized_v0", "affine_small", False) == "identity_standardized_v0_affine_small_unweighted"
    assert adapter.run_label("identity_standardized_v0", "affine_small", True) == "identity_standardized_v0_affine_small_weighted"


def test_job_pdg_dispatch_matches_producer_substring_rule():
    assert adapter.JOB_PDG["05_affine_preprocessing_ab_pdg13"] == (13, "pdg13")
    assert adapter.JOB_PDG["06_affine_preprocessing_ab_pdg_minus13"] == (-13, "pdg_minus13")
    assert adapter.JOB_PDG["12_memory_release_repeat_smoke"] == (None, "combined_unfiltered")
    assert adapter.JOB_PDG["04_legacy_available_code_realnvp_quantile"] == (None, "combined_unfiltered")


def test_gaussian_gmm_jobs_have_no_checkpoint():
    info = adapter.resolve_checkpoint_info("10_gaussian_controls_pdg13", "identity_standardized_v0_diagonal_gaussian_unweighted")
    assert info.checkpoint_path is None
    assert info.reconstruction_status == "MISSING_HISTORICAL_CHECKPOINT"


def test_job12_run1_checkpoint_overwritten_run2_reconstructible():
    run1 = adapter.job12_run_checkpoint_info(1)
    run2 = adapter.job12_run_checkpoint_info(2)
    assert run1.checkpoint_path is None
    assert run1.reconstruction_status == "MISSING_HISTORICAL_CHECKPOINT"
    assert run2.checkpoint_path is not None
    assert run2.reconstruction_status == "RECONSTRUCTIBLE"


def test_checkpoint_hash_is_raw_file_bytes_not_functional_fingerprint(input_artifact_dir):
    ckpt_path = (
        input_artifact_dir / "jobs" / "05_affine_preprocessing_ab_pdg13"
        / "identity_standardized_v0_affine_small_unweighted_model.pt"
    )
    recorded = json.loads((input_artifact_dir / "jobs" / "05_affine_preprocessing_ab_pdg13" / "metrics.json").read_text())
    recorded_hash = recorded["identity_standardized_v0_affine_small_unweighted"]["metrics"]["checkpoint_hash"]
    assert adapter.checkpoint_file_sha256(ckpt_path) == recorded_hash

    loaded = adapter.load_frozen_affine_checkpoint("affine_small", ckpt_path, device="cpu")
    # A different hashing method (the functional fingerprint) must NOT match
    # the frozen recorded hash -- this is the trap the adapter must avoid.
    assert loaded.checkpoint_hash() != recorded_hash


def test_load_frozen_affine_checkpoint_reproduces_finite_log_prob(input_artifact_dir):
    ckpt_path = (
        input_artifact_dir / "jobs" / "05_affine_preprocessing_ab_pdg13"
        / "identity_standardized_v0_affine_small_unweighted_model.pt"
    )
    estimator = adapter.load_frozen_affine_checkpoint("affine_small", ckpt_path, device="cpu")
    x = np.random.default_rng(0).normal(size=(16, 5))
    lp = estimator.log_prob(x)
    assert np.isfinite(lp).all()


def test_reconstruct_preprocessing_pipeline_deterministic(shard_dir):
    p1 = adapter.reconstruct_preprocessing_pipeline("identity_standardized_v0", shard_dir, pdg_value=13)
    p2 = adapter.reconstruct_preprocessing_pipeline("identity_standardized_v0", shard_dir, pdg_value=13)
    np.testing.assert_array_equal(p1.mean, p2.mean)
    np.testing.assert_array_equal(p1.std, p2.std)


def test_ambiguous_reconstruction_config_error_raised_for_unknown_model():
    with pytest.raises(KeyError):
        adapter.build_frozen_affine_estimator("affine_unknown_capacity_tier")
