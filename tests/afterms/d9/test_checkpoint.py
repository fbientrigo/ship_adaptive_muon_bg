from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from ship_muon_bg.afterms.d9 import checkpoint as ckpt


def _sample_bundle(
    scope,
    epoch=1,
    semantic_training_hash="abc123",
    execution_policy_hash="exec123",
    evaluation_policy_hash="eval123",
    max_epochs=10,
    execution_policy_revision=0,
):
    return ckpt.build_bundle(
        campaign_id="afterms_d9_training_v0",
        run_id="A1_capacity_medium_identity_pdg13_unweighted__seed20260720",
        candidate_id="A1_capacity_medium_identity_pdg13_unweighted",
        seed=20260720,
        epoch=epoch,
        checkpoint_scope=scope,
        model_family="affine_coupling",
        architecture_config={"number_of_blocks": 8, "hidden_width": 128, "hidden_depth": 2},
        feature_order=["px", "py", "pz", "x", "y"],
        pdg_policy="pdg13",
        preprocessing_reference="preprocessing.json",
        preprocessing_hash="deadbeef",
        target_measure="physical_space_nll",
        weighting_policy="row_empirical_unweighted",
        weighting_estimator_version="not_applicable",
        model_state_dict={"layer.weight": torch.zeros(3, 3)},
        optimizer_state_dict={"state": {}, "param_groups": []},
        scheduler_state_dict=None,
        training_history=[{"epoch": 1, "validation_feature_nll": 1.23}],
        best_validation_metric=1.23,
        best_validation_epoch=1,
        rng_states={"torch_manual_seed": 20260720},
        dataset_hash="dataset123",
        split_hashes={"train": "t1", "validation": "v1", "test": "te1"},
        shard_manifest_hash="shardhash",
        training_config_hash="cfg123",
        semantic_training_hash=semantic_training_hash,
        execution_policy_hash=execution_policy_hash,
        evaluation_policy_hash=evaluation_policy_hash,
        max_epochs=max_epochs,
        execution_policy_revision=execution_policy_revision,
        training_code_fingerprint={"runner.py": "fp1"},
        producer_git_commit="deadbeefcafefeed",
    )


def test_round_trip_preserves_all_fields(tmp_path):
    """Required test 17: self-describing checkpoint round trip."""

    bundle = _sample_bundle(ckpt.SCOPE_BEST)
    path = ckpt.save_bundle(tmp_path, ckpt.SCOPE_BEST, bundle)
    assert path.name == "best_checkpoint.pt"
    reloaded = ckpt.load_bundle(path)
    assert reloaded["schema_version"] == ckpt.SCHEMA_VERSION
    assert reloaded["candidate_id"] == bundle["candidate_id"]
    assert reloaded["semantic_training_hash"] == "abc123"
    torch.testing.assert_close(reloaded["model_state_dict"]["layer.weight"], bundle["model_state_dict"]["layer.weight"])


def test_atomic_write_leaves_no_tmp_file(tmp_path):
    """Required test 28: atomic checkpoint write."""

    bundle = _sample_bundle(ckpt.SCOPE_FINAL)
    ckpt.save_bundle(tmp_path, ckpt.SCOPE_FINAL, bundle)
    leftovers = list(tmp_path.glob(".tmp_*"))
    assert leftovers == []
    assert (tmp_path / "final_checkpoint.pt").exists()


def test_best_final_last_resumable_are_distinct_files(tmp_path):
    """Required test 16: best/final/resumable checkpoint distinction."""

    best = _sample_bundle(ckpt.SCOPE_BEST, epoch=3)
    final = _sample_bundle(ckpt.SCOPE_FINAL, epoch=10)
    resumable = _sample_bundle(ckpt.SCOPE_LAST_RESUMABLE, epoch=10)
    ckpt.save_bundle(tmp_path, ckpt.SCOPE_BEST, best)
    ckpt.save_bundle(tmp_path, ckpt.SCOPE_FINAL, final)
    ckpt.save_bundle(tmp_path, ckpt.SCOPE_LAST_RESUMABLE, resumable)

    reloaded_best = ckpt.load_bundle(tmp_path / "best_checkpoint.pt")
    reloaded_final = ckpt.load_bundle(tmp_path / "final_checkpoint.pt")
    assert reloaded_best["checkpoint_scope"] == "best"
    assert reloaded_final["checkpoint_scope"] == "final"
    assert reloaded_best["epoch"] != reloaded_final["epoch"]


def test_save_bundle_refuses_mislabeled_scope(tmp_path):
    """A final checkpoint may not be labeled as the best checkpoint."""

    bundle = _sample_bundle(ckpt.SCOPE_FINAL)
    with pytest.raises(ValueError):
        ckpt.save_bundle(tmp_path, ckpt.SCOPE_BEST, bundle)


def test_verify_compatibility_detects_mismatch():
    bundle = _sample_bundle(ckpt.SCOPE_BEST, semantic_training_hash="hash_v1")
    violations = ckpt.verify_compatibility(bundle, expected={"semantic_training_hash": "hash_v1"})
    assert violations == []
    violations = ckpt.verify_compatibility(bundle, expected={"semantic_training_hash": "hash_v2"})
    assert len(violations) == 1
    assert "semantic_training_hash" in violations[0]


def test_load_and_verify_raises_on_mismatch(tmp_path):
    """Required test 19: training-contract mismatch refuses resume."""

    bundle = _sample_bundle(ckpt.SCOPE_LAST_RESUMABLE, semantic_training_hash="hash_v1")
    path = ckpt.save_bundle(tmp_path, ckpt.SCOPE_LAST_RESUMABLE, bundle)
    with pytest.raises(ckpt.CheckpointCompatibilityError):
        ckpt.load_and_verify(path, expected={"semantic_training_hash": "hash_v2"})
    # Same hash resumes cleanly.
    reloaded = ckpt.load_and_verify(path, expected={"semantic_training_hash": "hash_v1"})
    assert reloaded["candidate_id"] == bundle["candidate_id"]
