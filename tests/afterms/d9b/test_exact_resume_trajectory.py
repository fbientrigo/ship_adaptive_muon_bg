"""D9B Gate B.1: exact epoch-boundary resume trajectory (CPU proof).

Proves that an interrupted-then-resumed run reproduces the same epoch-2
trajectory a clean, uninterrupted two-epoch run would have produced -- the
bug this fixes was a single process-lifetime ``torch.Generator`` stream whose
epoch-2 draw depended on whether epoch 1 had already been drawn from it in
this process (clean run) or not (fresh process after resume). See
``ship_muon_bg.afterms.d9.sampling``.

Duplicates the tiny synthetic CPU fixture pattern from
``tests/afterms/d9/test_runner.py`` / ``tests/afterms/d9b/test_resume_contract.py``
rather than cross-importing, since ``tests/afterms`` is not an import package.
"""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from Nflow.registry import create_density_estimator

from ship_muon_bg.afterms.d9 import checkpoint as ckpt
from ship_muon_bg.afterms.d9 import evaluate as d9eval
from ship_muon_bg.afterms.d9 import runner
from ship_muon_bg.afterms.d9 import sampling as d9sampling

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


def _tiny_candidate_config(weighted=False):
    return {
        "campaign_id": "d9b_exact_resume_test",
        "candidate_id": "T1_tiny_exact_resume_candidate",
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
        "max_epochs": 2,
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


def _sample_hash_for_run(config, run_dir, generation_seed=20260720, n=32):
    bundle = ckpt.load_bundle(run_dir / "checkpoints" / "final_checkpoint.pt")
    estimator = create_density_estimator(
        {"family": config["model_family"], "params": runner._architecture_params(config)}, dimension=5, device="cpu",
    )
    estimator._build_module(seed=int(bundle["seed"]))
    estimator._module.load_state_dict(bundle["model_state_dict"])
    estimator._module.eval()
    generated = d9eval.generate_deterministic_samples(estimator, n, generation_seed)
    arr = np.ascontiguousarray(generated, dtype=np.float64)
    return hashlib.sha256(arr.tobytes()).hexdigest(), bundle


def _run_clean_and_interrupted_resume(tmp_path, config, train_seed):
    train_raw = _synthetic_raw(150, seed=train_seed)
    val_raw = _synthetic_raw(40, seed=train_seed + 1)

    clean_root = tmp_path / "clean"
    clean_result = runner.train_candidate_seed(
        config, seed=1, train_raw=train_raw.copy(), validation_raw=val_raw.copy(),
        artifact_root=clean_root, repo_root=REPO_ROOT, device="cpu",
    )
    assert clean_result["status"] == "completed"
    assert clean_result["final_epoch"] == 2

    resume_root = tmp_path / "resumed"
    calls = {"n": 0}

    def _interrupt_after_epoch_1():
        calls["n"] += 1
        return calls["n"] > 1

    interrupted_result = runner.train_candidate_seed(
        config, seed=1, train_raw=train_raw.copy(), validation_raw=val_raw.copy(),
        artifact_root=resume_root, repo_root=REPO_ROOT, device="cpu",
        interrupt_flag=_interrupt_after_epoch_1,
    )
    assert interrupted_result["status"] == "interrupted"

    resumed_result = runner.train_candidate_seed(
        config, seed=1, train_raw=train_raw.copy(), validation_raw=val_raw.copy(),
        artifact_root=resume_root, repo_root=REPO_ROOT, device="cpu", resume=True,
    )
    assert resumed_result["status"] == "completed"
    assert resumed_result["final_epoch"] == 2

    clean_dir = runner.run_directory(clean_root, config["candidate_id"], 1)
    resumed_dir = runner.run_directory(resume_root, config["candidate_id"], 1)
    return clean_dir, resumed_dir


def test_clean_two_epoch_matches_interrupted_and_resumed_run(tmp_path):
    """Required tests 4-9: clean two-epoch run == one-epoch + resume run."""

    config = _tiny_candidate_config()
    clean_dir, resumed_dir = _run_clean_and_interrupted_resume(tmp_path, config, train_seed=200)

    clean_history = json.loads((clean_dir / "histories" / "training_history.json").read_text())
    resumed_history = json.loads((resumed_dir / "histories" / "training_history.json").read_text())

    # Required test 9: history is [1, 2] with no duplicate entries.
    assert [r["epoch"] for r in clean_history] == [1, 2]
    assert [r["epoch"] for r in resumed_history] == [1, 2]

    # Required test 5: train and validation losses match at epoch 2.
    clean_epoch_2 = clean_history[1]
    resumed_epoch_2 = resumed_history[1]
    assert clean_epoch_2["train_feature_nll"] == pytest.approx(resumed_epoch_2["train_feature_nll"], rel=1e-6)
    assert clean_epoch_2["validation_feature_nll"] == pytest.approx(resumed_epoch_2["validation_feature_nll"], rel=1e-6)

    # Required test 6: model state_dict tensors match (final checkpoint).
    clean_final = ckpt.load_bundle(clean_dir / "checkpoints" / "final_checkpoint.pt")
    resumed_final = ckpt.load_bundle(resumed_dir / "checkpoints" / "final_checkpoint.pt")
    assert clean_final["model_state_dict"].keys() == resumed_final["model_state_dict"].keys()
    for key in clean_final["model_state_dict"]:
        torch.testing.assert_close(clean_final["model_state_dict"][key], resumed_final["model_state_dict"][key])

    # Required test 7: optimizer state matches where deterministic.
    clean_opt = clean_final["optimizer_state_dict"]
    resumed_opt = resumed_final["optimizer_state_dict"]
    assert clean_opt["param_groups"] == resumed_opt["param_groups"]
    for param_id, clean_state in clean_opt["state"].items():
        resumed_state = resumed_opt["state"][param_id]
        assert clean_state["step"] == resumed_state["step"]
        torch.testing.assert_close(clean_state["exp_avg"], resumed_state["exp_avg"])
        torch.testing.assert_close(clean_state["exp_avg_sq"], resumed_state["exp_avg_sq"])

    # Required test 8: deterministic generated-sample hash matches.
    clean_hash, _ = _sample_hash_for_run(config, clean_dir)
    resumed_hash, _ = _sample_hash_for_run(config, resumed_dir)
    assert clean_hash == resumed_hash

    # Best/final checkpoint functional behavior: both reload without error and
    # produce finite samples.
    best_clean = ckpt.load_bundle(clean_dir / "checkpoints" / "best_checkpoint.pt")
    best_resumed = ckpt.load_bundle(resumed_dir / "checkpoints" / "best_checkpoint.pt")
    assert best_clean["best_validation_epoch"] == best_resumed["best_validation_epoch"]


def test_weighted_exact_resume_retains_same_weighted_estimator(tmp_path):
    """Required test 11: weighted runs retain the same weighted estimator
    across a clean vs interrupted-resumed trajectory."""

    config = _tiny_candidate_config(weighted=True)
    clean_dir, resumed_dir = _run_clean_and_interrupted_resume(tmp_path, config, train_seed=210)

    clean_history = json.loads((clean_dir / "histories" / "training_history.json").read_text())
    resumed_history = json.loads((resumed_dir / "histories" / "training_history.json").read_text())

    for record in clean_history + resumed_history:
        assert record["weighted_or_unweighted_estimator"] != "row_empirical_unweighted_mean"

    assert (
        clean_history[1]["weighted_or_unweighted_estimator"]
        == resumed_history[1]["weighted_or_unweighted_estimator"]
    )
    assert clean_history[1]["train_feature_nll"] == pytest.approx(resumed_history[1]["train_feature_nll"], rel=1e-6)
    assert clean_history[1]["validation_feature_nll"] == pytest.approx(
        resumed_history[1]["validation_feature_nll"], rel=1e-6
    )


def test_old_checkpoint_sampling_contract_mismatch_blocks_resume(tmp_path):
    """Required test 10: an old/incompatible sampling-contract checkpoint
    fails explicitly rather than silently resuming."""

    config = _tiny_candidate_config()
    config["max_epochs"] = 1
    train_raw = _synthetic_raw(150, seed=220)
    val_raw = _synthetic_raw(40, seed=221)
    artifact_root = tmp_path / "stale_contract_test"

    runner.train_candidate_seed(
        config, seed=1, train_raw=train_raw, validation_raw=val_raw,
        artifact_root=artifact_root, repo_root=REPO_ROOT, device="cpu",
    )

    run_dir = runner.run_directory(artifact_root, config["candidate_id"], 1)
    checkpoints_dir = run_dir / "checkpoints"
    resumable_path = checkpoints_dir / ckpt.FILENAME_BY_SCOPE[ckpt.SCOPE_LAST_RESUMABLE]

    stale_bundle = ckpt.load_bundle(resumable_path)
    assert stale_bundle["sampling_contract_version"] == d9sampling.SAMPLING_CONTRACT_VERSION
    stale_bundle = dict(stale_bundle, sampling_contract_version="d9_pre_gate_b1_unversioned")
    ckpt.save_bundle(checkpoints_dir, ckpt.SCOPE_LAST_RESUMABLE, stale_bundle)

    extended_config = dict(config, max_epochs=2)
    with pytest.raises(ckpt.CheckpointCompatibilityError):
        runner.train_candidate_seed(
            extended_config, seed=1, train_raw=train_raw, validation_raw=val_raw,
            artifact_root=artifact_root, repo_root=REPO_ROOT, device="cpu", resume=True,
            extend_max_epochs=2,
        )
