"""D9B resume/execution-policy identity contract tests (mission section 13 items 1-6).

Covers the three-hash split (semantic/execution/evaluation), the explicit
max-epochs extension contract, and the explicit execution-policy revision
contract. Uses the same tiny synthetic CPU fixture pattern as
``tests/afterms/d9/test_runner.py`` -- duplicated locally rather than
cross-imported, since ``tests/afterms`` is not an import package (no
``__init__.py`` above ``tests/afterms/d9/``).
"""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ship_muon_bg.afterms.d9 import checkpoint as ckpt
from ship_muon_bg.afterms.d9 import contract as d9contract
from ship_muon_bg.afterms.d9 import runner

REPO_ROOT = Path(__file__).resolve().parents[3]


def _synthetic_raw(n, seed, pdg=13):
    rng = np.random.default_rng(seed)
    px = rng.normal(size=n)
    py = rng.normal(size=n)
    pz = rng.uniform(1.0, 20.0, size=n)
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    z = np.zeros(n)
    ids = np.full(n, pdg, dtype=np.float64)
    w = rng.uniform(0.5, 2.0, size=n)
    return np.column_stack([px, py, pz, x, y, z, ids, w])


def _tiny_candidate_config(tmp_path, weighted=False):
    return {
        "campaign_id": "d9b_test_campaign",
        "candidate_id": "T1_tiny_test_candidate",
        "model_family": "affine_coupling",
        "architecture": {"number_of_blocks": 2, "hidden_width": 8, "hidden_depth": 1, "capacity_label": "tiny_test"},
        "modeled_features": ["px", "py", "pz", "x", "y"],
        "feature_order": ["px", "py", "pz", "x", "y"],
        "pdg_policy": "pdg13",
        "pdg_value": 13,
        "preprocessing_name": "identity_standardized_v0",
        "target_measure": "physical_space_nll",
        "weighting_policy": "production_weighted" if weighted else "row_empirical_unweighted",
        "weighting_estimator_version": "fixed_global_weight_normalization_v1" if weighted else "not_applicable",
        "shard_dir": "synthetic_test_only",
        "train_shards": ["synthetic_train"],
        "validation_shards": ["synthetic_validation"],
        "test_shards": ["synthetic_test"],
        "seed_set": [1, 2, 3],
        "max_epochs": 1,
        "minimum_epochs": 1,
        "early_stopping_patience": 5,
        "optimizer": "adam",
        "learning_rate": 0.01,
        "batch_size": 32,
        "weight_decay": 0.0,
        "gradient_clipping": 5.0,
        "dtype": "float32",
        "device_policy": "cpu_only_for_tests",
        "checkpoint_policy": {"save_best": True, "save_final": True, "save_last_resumable": True, "checkpoint_interval_epochs": 1},
        "evaluation_policy": {
            "quick_validation_budget": {"n_generated_samples": 50},
            "final_candidate_budget": {"n_generated_samples": 100},
        },
    }


def _run_one_epoch(tmp_path, config, artifact_root, train_seed=100):
    train_raw = _synthetic_raw(150, seed=train_seed)
    val_raw = _synthetic_raw(40, seed=train_seed + 1)
    result = runner.train_candidate_seed(
        config, seed=1, train_raw=train_raw, validation_raw=val_raw,
        artifact_root=artifact_root, repo_root=REPO_ROOT, device="cpu",
    )
    assert result["final_epoch"] == 1
    return train_raw, val_raw


def test_semantic_execution_evaluation_hashes_are_independent():
    """The three hashes are keyed on disjoint inputs -- changing one axis
    (semantic config, execution policy, evaluation policy) must never move
    the other two."""

    base_candidate_config = {"model_family": "affine_coupling", "learning_rate": 0.01}
    base_kwargs = dict(
        candidate_config=base_candidate_config,
        preprocessing_contract={"preprocessing_name": "identity_standardized_v0", "serialization_hash": "abc"},
        dataset_identity={"train_split_hash": "t1", "validation_split_hash": "v1"},
        modeled_feature_order=["px", "py", "pz", "x", "y"],
        target_measure="physical_space_nll",
        weighting_estimator_version="not_applicable",
        optimizer_settings={"optimizer": "adam", "learning_rate": 0.01},
        module_fingerprints={"runner.py": "fp1"},
    )
    exec_kwargs = dict(
        minimum_epochs=1, early_stopping_patience=5,
        checkpoint_policy={"save_best": True, "save_final": True, "save_last_resumable": True, "checkpoint_interval_epochs": 1},
    )
    eval_kwargs = dict(evaluation_policy={"quick_validation_budget": {"n_generated_samples": 50}})

    semantic_a = d9contract.semantic_training_hash(**base_kwargs)
    execution_a = d9contract.execution_policy_hash(**exec_kwargs)
    evaluation_a = d9contract.evaluation_policy_hash(**eval_kwargs)

    changed_semantic_kwargs = dict(base_kwargs, candidate_config=dict(base_candidate_config, learning_rate=0.5))
    semantic_b = d9contract.semantic_training_hash(**changed_semantic_kwargs)
    assert semantic_b != semantic_a
    assert d9contract.execution_policy_hash(**exec_kwargs) == execution_a
    assert d9contract.evaluation_policy_hash(**eval_kwargs) == evaluation_a

    changed_exec_kwargs = dict(exec_kwargs, early_stopping_patience=99)
    execution_b = d9contract.execution_policy_hash(**changed_exec_kwargs)
    assert execution_b != execution_a
    assert d9contract.semantic_training_hash(**base_kwargs) == semantic_a
    assert d9contract.evaluation_policy_hash(**eval_kwargs) == evaluation_a

    changed_eval_kwargs = dict(evaluation_policy={"quick_validation_budget": {"n_generated_samples": 999}})
    evaluation_b = d9contract.evaluation_policy_hash(**changed_eval_kwargs)
    assert evaluation_b != evaluation_a
    assert d9contract.semantic_training_hash(**base_kwargs) == semantic_a
    assert d9contract.execution_policy_hash(**exec_kwargs) == execution_a


def test_patience_change_prevents_silent_resume(tmp_path):
    config = _tiny_candidate_config(tmp_path)
    artifact_root = tmp_path / "patience_test"
    _run_one_epoch(tmp_path, config, artifact_root, train_seed=110)

    mismatched_config = dict(config, early_stopping_patience=config["early_stopping_patience"] + 1)
    train_raw = _synthetic_raw(150, seed=110)
    val_raw = _synthetic_raw(40, seed=111)
    with pytest.raises(ckpt.CheckpointCompatibilityError):
        runner.train_candidate_seed(
            mismatched_config, seed=1, train_raw=train_raw, validation_raw=val_raw,
            artifact_root=artifact_root, repo_root=REPO_ROOT, device="cpu", resume=True,
        )


def test_checkpoint_policy_change_prevents_silent_resume(tmp_path):
    config = _tiny_candidate_config(tmp_path)
    artifact_root = tmp_path / "checkpoint_policy_test"
    _run_one_epoch(tmp_path, config, artifact_root, train_seed=120)

    mismatched_config = dict(
        config,
        checkpoint_policy=dict(config["checkpoint_policy"], checkpoint_interval_epochs=2),
    )
    train_raw = _synthetic_raw(150, seed=120)
    val_raw = _synthetic_raw(40, seed=121)
    with pytest.raises(ckpt.CheckpointCompatibilityError):
        runner.train_candidate_seed(
            mismatched_config, seed=1, train_raw=train_raw, validation_raw=val_raw,
            artifact_root=artifact_root, repo_root=REPO_ROOT, device="cpu", resume=True,
        )


def test_explicit_execution_policy_revision_allows_resume_and_is_logged(tmp_path):
    config = _tiny_candidate_config(tmp_path)
    artifact_root = tmp_path / "revision_test"
    _run_one_epoch(tmp_path, config, artifact_root, train_seed=130)

    revised_config = dict(config, early_stopping_patience=config["early_stopping_patience"] + 1)
    train_raw = _synthetic_raw(150, seed=130)
    val_raw = _synthetic_raw(40, seed=131)
    result = runner.train_candidate_seed(
        revised_config, seed=1, train_raw=train_raw, validation_raw=val_raw,
        artifact_root=artifact_root, repo_root=REPO_ROOT, device="cpu", resume=True,
        execution_policy_revision_reason="testing revision",
    )
    assert result["status"] == "completed"

    run_dir = runner.run_directory(artifact_root, config["candidate_id"], 1)
    log_path = run_dir / "execution_policy_log.json"
    assert log_path.exists()
    events = json.loads(log_path.read_text(encoding="utf-8"))
    revised_events = [e for e in events if e["event"] == "execution_policy_revised"]
    assert len(revised_events) == 1
    assert revised_events[0]["reason"] == "testing revision"


def test_max_epochs_change_without_explicit_extension_is_refused(tmp_path):
    config = _tiny_candidate_config(tmp_path)
    artifact_root = tmp_path / "silent_max_epochs_test"
    _run_one_epoch(tmp_path, config, artifact_root, train_seed=140)

    mismatched_config = dict(config, max_epochs=2)
    train_raw = _synthetic_raw(150, seed=140)
    val_raw = _synthetic_raw(40, seed=141)
    with pytest.raises(runner.MaxEpochsExtensionError):
        runner.train_candidate_seed(
            mismatched_config, seed=1, train_raw=train_raw, validation_raw=val_raw,
            artifact_root=artifact_root, repo_root=REPO_ROOT, device="cpu", resume=True,
        )


def test_max_epochs_extension_only_increases(tmp_path):
    config = _tiny_candidate_config(tmp_path)
    artifact_root = tmp_path / "extension_test"
    train_raw, val_raw = _run_one_epoch(tmp_path, config, artifact_root, train_seed=150)

    extended = runner.train_candidate_seed(
        config, seed=1, train_raw=train_raw, validation_raw=val_raw,
        artifact_root=artifact_root, repo_root=REPO_ROOT, device="cpu", resume=True,
        extend_max_epochs=2,
    )
    assert extended["status"] == "completed"
    assert extended["final_epoch"] == 2

    run_dir = runner.run_directory(artifact_root, config["candidate_id"], 1)
    events = json.loads((run_dir / "execution_policy_log.json").read_text(encoding="utf-8"))
    extension_events = [e for e in events if e["event"] == "max_epochs_extended"]
    assert len(extension_events) == 1
    assert extension_events[0]["previous_max_epochs"] == 1
    assert extension_events[0]["new_max_epochs"] == 2

    with pytest.raises(runner.MaxEpochsExtensionError):
        runner.train_candidate_seed(
            config, seed=1, train_raw=train_raw, validation_raw=val_raw,
            artifact_root=artifact_root, repo_root=REPO_ROOT, device="cpu", resume=True,
            extend_max_epochs=2,
        )
    with pytest.raises(runner.MaxEpochsExtensionError):
        runner.train_candidate_seed(
            config, seed=1, train_raw=train_raw, validation_raw=val_raw,
            artifact_root=artifact_root, repo_root=REPO_ROOT, device="cpu", resume=True,
            extend_max_epochs=1,
        )


def test_extension_preserves_history_and_checkpoints(tmp_path):
    config = _tiny_candidate_config(tmp_path)
    artifact_root = tmp_path / "extension_preserve_test"
    train_raw, val_raw = _run_one_epoch(tmp_path, config, artifact_root, train_seed=160)

    runner.train_candidate_seed(
        config, seed=1, train_raw=train_raw, validation_raw=val_raw,
        artifact_root=artifact_root, repo_root=REPO_ROOT, device="cpu", resume=True,
        extend_max_epochs=2,
    )

    run_dir = runner.run_directory(artifact_root, config["candidate_id"], 1)
    history = json.loads((run_dir / "histories" / "training_history.json").read_text(encoding="utf-8"))
    epochs_present = {record["epoch"] for record in history}
    assert 1 in epochs_present
    assert 2 in epochs_present

    # last_resumable/final are unconditionally rewritten every epoch, so both
    # always carry the post-extension max_epochs. best_checkpoint.pt is only
    # rewritten when that epoch's validation loss actually improves -- after
    # an extension, epoch 2 may or may not improve on epoch 1, so its
    # max_epochs can legitimately still read the pre-extension value.
    for filename in ("last_resumable_checkpoint.pt", "final_checkpoint.pt"):
        bundle_path = run_dir / "checkpoints" / filename
        assert bundle_path.exists()
        bundle = ckpt.load_bundle(bundle_path)
        assert bundle["max_epochs"] == 2

    best_bundle_path = run_dir / "checkpoints" / "best_checkpoint.pt"
    assert best_bundle_path.exists()
    best_bundle = ckpt.load_bundle(best_bundle_path)
    assert best_bundle["epoch"] in (1, 2)
    # best_checkpoint.pt's max_epochs reflects whatever effective_max_epochs
    # was in force the last time this scope was actually rewritten -- 2 only
    # if epoch 2 was the improving epoch, else it still reads the
    # pre-extension value from epoch 1.
    expected_best_max_epochs = 2 if best_bundle["epoch"] == 2 else 1
    assert best_bundle["max_epochs"] == expected_best_max_epochs
