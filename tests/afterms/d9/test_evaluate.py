import inspect
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ship_muon_bg.afterms.d9 import runner, evaluate as d9eval
from ship_muon_bg.afterms.d9 import aggregate as d9agg

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


def _tiny_config():
    return {
        "campaign_id": "d9_test_campaign",
        "candidate_id": "T2_tiny_eval_candidate",
        "model_family": "affine_coupling",
        "architecture": {"number_of_blocks": 2, "hidden_width": 8, "hidden_depth": 1, "capacity_label": "tiny_test"},
        "modeled_features": ["px", "py", "pz", "x", "y"],
        "feature_order": ["px", "py", "pz", "x", "y"],
        "pdg_policy": "pdg13",
        "pdg_value": 13,
        "preprocessing_name": "identity_standardized_v0",
        "target_measure": "physical_space_nll",
        "weighting_policy": "row_empirical_unweighted",
        "weighting_estimator_version": "not_applicable",
        "shard_dir": "synthetic_test_only",
        "train_shards": ["synthetic_train"],
        "validation_shards": ["synthetic_validation"],
        "test_shards": ["synthetic_test"],
        "seed_set": [1, 2],
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
            "quick_validation_budget": {
                "n_generated_samples": 50, "c2st_subsample": 30, "one_d_n_boot": 20,
                "two_d_test_subsample": 30, "two_d_n_permutations": 20, "c2st_n_boot": 20, "c2st_n_permutations": 20,
            },
            "final_candidate_budget": {"n_generated_samples": 100, "c2st_subsample": 60},
        },
    }


def test_evaluate_never_imports_optimizer_machinery():
    """Required test 32: evaluate never calls optimizer code."""

    source = inspect.getsource(d9eval)
    assert "torch.optim" not in source
    assert "Optimizer(" not in source
    params = inspect.signature(d9eval.evaluate_candidate_seed).parameters
    assert "optimizer" not in params


def test_evaluate_candidate_seed_end_to_end(tmp_path):
    config = _tiny_config()
    train_raw = _synthetic_raw(150, seed=100)
    val_raw = _synthetic_raw(40, seed=101)
    test_raw = _synthetic_raw(40, seed=102)

    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    np.save(shard_dir / "synthetic_test.npy", test_raw)

    artifact_root = tmp_path / "artifacts"
    runner.train_candidate_seed(
        config, seed=1, train_raw=train_raw, validation_raw=val_raw,
        artifact_root=artifact_root, repo_root=REPO_ROOT, device="cpu",
    )
    run_dir = runner.run_directory(artifact_root, config["candidate_id"], 1)

    result = d9eval.evaluate_candidate_seed(
        config, run_dir=run_dir, shard_dir=shard_dir, budget_name="quick_validation_budget", device="cpu",
    )
    assert np.isfinite(result["test_feature_nll"])
    assert result["test_physical_nll"] is not None
    assert result["generated_sample_count"] == 50
    assert (run_dir / "evaluation" / "quick_validation_budget.json").exists()


def test_load_test_split_is_the_only_test_data_entry_point():
    """Required test 22 confirmation: test data is reachable only through
    evaluate.load_test_split, never through the training runner."""

    assert "load_test_split" in dir(d9eval)
    train_params = inspect.signature(
        __import__("ship_muon_bg.afterms.d9.runner", fromlist=["train_candidate_seed"]).train_candidate_seed
    ).parameters
    assert not any("test" in name for name in train_params)


def test_aggregate_reports_without_hiding_failed_or_interrupted(tmp_path):
    """Required test 13 (aggregation section): no seed's instability is hidden."""

    from ship_muon_bg.afterms.d9 import runner as d9runner

    candidate_id = "T2_tiny_eval_candidate"
    run_dir_ok = d9runner.run_directory(tmp_path, candidate_id, 1)
    run_dir_ok.mkdir(parents=True)
    (run_dir_ok / "status.json").write_text(json.dumps({
        "status": "completed", "seed": 1, "best_validation_metric": 1.5, "best_validation_epoch": 3,
    }))
    run_dir_failed = d9runner.run_directory(tmp_path, candidate_id, 2)
    run_dir_failed.mkdir(parents=True)
    (run_dir_failed / "status.json").write_text(json.dumps({"status": "failed_technical", "seed": 2}))

    statuses = d9agg.load_seed_statuses(tmp_path, candidate_id, [1, 2])
    result = d9agg.aggregate_candidate(candidate_id, statuses)
    assert result["completed_seed_count"] == 1
    assert result["failed_seed_count"] == 1
    assert result["median_best_validation_nll"] == pytest.approx(1.5)
