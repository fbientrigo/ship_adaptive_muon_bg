import inspect
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ship_muon_bg.afterms.d9 import runner
from ship_muon_bg.afterms.d9 import checkpoint as ckpt

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
        "campaign_id": "d9_test_campaign",
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


def test_train_candidate_seed_has_no_test_data_parameter():
    """Required tests 22/23: structural test-set isolation."""

    params = inspect.signature(runner.train_candidate_seed).parameters
    assert "test_raw" not in params
    assert "test_data" not in params
    assert "test_shards_data" not in params


def test_tiny_two_epoch_end_to_end_run(tmp_path):
    """Required test 33: tiny two-epoch end-to-end run."""

    config = _tiny_candidate_config(tmp_path)
    train_raw = _synthetic_raw(200, seed=10)
    val_raw = _synthetic_raw(50, seed=11)
    artifact_root = tmp_path / "d9_synthetic_artifacts"

    result = runner.train_candidate_seed(
        config, seed=1, train_raw=train_raw, validation_raw=val_raw,
        artifact_root=artifact_root, repo_root=REPO_ROOT, device="cpu",
    )
    assert result["status"] == "completed"
    run_dir = runner.run_directory(artifact_root, config["candidate_id"], 1)
    status = json.loads((run_dir / "status.json").read_text())
    assert status["status"] == "completed"
    assert (run_dir / "checkpoints" / "final_checkpoint.pt").exists()
    assert (run_dir / "checkpoints" / "best_checkpoint.pt").exists()
    assert (run_dir / "checkpoints" / "last_resumable_checkpoint.pt").exists()
    assert (run_dir / "preprocessing" / "preprocessing.json").exists()


def test_full_checkpoint_reload_and_deterministic_sample_generation(tmp_path):
    """Required test 34: full checkpoint reload + deterministic sample generation."""

    from Nflow.registry import create_density_estimator

    config = _tiny_candidate_config(tmp_path)
    train_raw = _synthetic_raw(200, seed=20)
    val_raw = _synthetic_raw(50, seed=21)
    artifact_root = tmp_path / "d9_synthetic_artifacts"

    runner.train_candidate_seed(
        config, seed=1, train_raw=train_raw, validation_raw=val_raw,
        artifact_root=artifact_root, repo_root=REPO_ROOT, device="cpu",
    )
    run_dir = runner.run_directory(artifact_root, config["candidate_id"], 1)
    bundle = ckpt.load_bundle(run_dir / "checkpoints" / "final_checkpoint.pt")

    estimator = create_density_estimator(
        {"family": "affine_coupling", "params": runner._architecture_params(config)}, dimension=5, device="cpu",
    )
    estimator._build_module(seed=1)
    estimator._module.load_state_dict(bundle["model_state_dict"])
    estimator._module.eval()

    def _sample():
        gen = torch.Generator(device="cpu")
        gen.manual_seed(42)
        with torch.no_grad():
            z = torch.randn(16, 5, generator=gen)
            return estimator._module(z).numpy()

    s1 = _sample()
    s2 = _sample()
    np.testing.assert_array_equal(s1, s2)


def test_deterministic_seed_behavior_same_seed_same_result(tmp_path):
    """Required test 24."""

    config = _tiny_candidate_config(tmp_path)
    train_raw = _synthetic_raw(150, seed=30)
    val_raw = _synthetic_raw(40, seed=31)

    root_a = tmp_path / "run_a"
    root_b = tmp_path / "run_b"
    result_a = runner.train_candidate_seed(
        config, seed=7, train_raw=train_raw.copy(), validation_raw=val_raw.copy(),
        artifact_root=root_a, repo_root=REPO_ROOT, device="cpu",
    )
    result_b = runner.train_candidate_seed(
        config, seed=7, train_raw=train_raw.copy(), validation_raw=val_raw.copy(),
        artifact_root=root_b, repo_root=REPO_ROOT, device="cpu",
    )
    assert result_a["best_validation_metric"] == pytest.approx(result_b["best_validation_metric"], rel=1e-6)


def test_different_seeds_produce_distinct_runs(tmp_path):
    """Required test 25."""

    config = _tiny_candidate_config(tmp_path)
    train_raw = _synthetic_raw(150, seed=40)
    val_raw = _synthetic_raw(40, seed=41)

    result_a = runner.train_candidate_seed(
        config, seed=1, train_raw=train_raw.copy(), validation_raw=val_raw.copy(),
        artifact_root=tmp_path / "seed1", repo_root=REPO_ROOT, device="cpu",
    )
    result_b = runner.train_candidate_seed(
        config, seed=2, train_raw=train_raw.copy(), validation_raw=val_raw.copy(),
        artifact_root=tmp_path / "seed2", repo_root=REPO_ROOT, device="cpu",
    )
    assert result_a["best_validation_metric"] != pytest.approx(result_b["best_validation_metric"], rel=1e-9)


def test_interrupted_run_becomes_interrupted_status(tmp_path):
    """Required test 26: interrupted run becomes `interrupted`, not a failure."""

    config = _tiny_candidate_config(tmp_path)
    config["max_epochs"] = 10
    train_raw = _synthetic_raw(150, seed=50)
    val_raw = _synthetic_raw(40, seed=51)
    artifact_root = tmp_path / "interrupt_test"

    calls = {"n": 0}

    def interrupt_after_first_epoch():
        calls["n"] += 1
        return calls["n"] > 1

    result = runner.train_candidate_seed(
        config, seed=1, train_raw=train_raw, validation_raw=val_raw,
        artifact_root=artifact_root, repo_root=REPO_ROOT, device="cpu",
        interrupt_flag=interrupt_after_first_epoch,
    )
    assert result["status"] == "interrupted"
    run_dir = runner.run_directory(artifact_root, config["candidate_id"], 1)
    status = json.loads((run_dir / "status.json").read_text())
    assert status["status"] == "interrupted"
    assert (run_dir / "checkpoints" / "last_resumable_checkpoint.pt").exists()
    assert not (run_dir / "checkpoints" / "final_checkpoint.pt").exists()


def test_resume_continues_with_matching_contract_and_more_epochs(tmp_path):
    """Required tests 18/20: optimizer/RNG resume; unchanged relevant fingerprint doesn't force a rerun.

    Under the D9B resume contract, increasing max_epochs across a resume must
    go through the explicit ``extend_max_epochs`` operation -- passing a
    bumped ``max_epochs`` in candidate_config alone is a silent change and is
    refused (see ``test_max_epochs_change_without_explicit_extension_is_refused``
    in tests/afterms/d9b/test_resume_contract.py).
    """

    config = _tiny_candidate_config(tmp_path)
    config["max_epochs"] = 1
    train_raw = _synthetic_raw(150, seed=60)
    val_raw = _synthetic_raw(40, seed=61)
    artifact_root = tmp_path / "resume_test"

    first = runner.train_candidate_seed(
        config, seed=1, train_raw=train_raw, validation_raw=val_raw,
        artifact_root=artifact_root, repo_root=REPO_ROOT, device="cpu",
    )
    assert first["final_epoch"] == 1

    second = runner.train_candidate_seed(
        config, seed=1, train_raw=train_raw, validation_raw=val_raw,
        artifact_root=artifact_root, repo_root=REPO_ROOT, device="cpu", resume=True,
        extend_max_epochs=2,
    )
    assert second["status"] == "completed"
    assert second["final_epoch"] == 2


def test_resume_refuses_on_training_contract_mismatch(tmp_path):
    """Required test 19: training-contract mismatch refuses resume."""

    config = _tiny_candidate_config(tmp_path)
    config["max_epochs"] = 1
    train_raw = _synthetic_raw(150, seed=70)
    val_raw = _synthetic_raw(40, seed=71)
    artifact_root = tmp_path / "resume_mismatch_test"

    runner.train_candidate_seed(
        config, seed=1, train_raw=train_raw, validation_raw=val_raw,
        artifact_root=artifact_root, repo_root=REPO_ROOT, device="cpu",
    )

    mismatched_config = dict(config, learning_rate=0.5, max_epochs=2)
    with pytest.raises(ckpt.CheckpointCompatibilityError):
        runner.train_candidate_seed(
            mismatched_config, seed=1, train_raw=train_raw, validation_raw=val_raw,
            artifact_root=artifact_root, repo_root=REPO_ROOT, device="cpu", resume=True,
        )


def test_train_only_preprocessing_fit(tmp_path):
    """Required test 13: preprocessing is fit on the training split only."""

    config = _tiny_candidate_config(tmp_path)
    train_raw = _synthetic_raw(150, seed=80)
    # Validation drawn from a very different distribution; if the pipeline
    # were (incorrectly) fit on train+validation, its recorded mean would
    # shift measurably toward this offset population.
    val_raw = _synthetic_raw(150, seed=81)
    val_raw[:, :5] += 1000.0

    artifact_root = tmp_path / "train_only_fit_test"
    runner.train_candidate_seed(
        config, seed=1, train_raw=train_raw, validation_raw=val_raw,
        artifact_root=artifact_root, repo_root=REPO_ROOT, device="cpu",
    )
    run_dir = runner.run_directory(artifact_root, config["candidate_id"], 1)
    preprocessing = json.loads((run_dir / "preprocessing" / "preprocessing.json").read_text())
    fitted_mean = np.array(preprocessing["fitted_state"]["mean"])
    train_only_mean = np.mean(train_raw[:, :5], axis=0)
    np.testing.assert_allclose(fitted_mean, train_only_mean, atol=1e-6)
