"""GMM control adapter tests: fixed configuration, persistence, seed identity,
non-convergence visibility.

Covers required tests 17, 18, 19, 20.
"""

from __future__ import annotations

import numpy as np

from ship_muon_bg.afterms.d9_5 import config as d9_5config
from ship_muon_bg.afterms.d9_5 import data_scope, gmm_adapter


def test_gmm_configuration_is_exactly_k4_full_covariance():
    arena_config = d9_5config.load_model_family_arena_config()
    gmm_cfg = arena_config["gmm"]
    assert gmm_cfg["n_components"] == 4
    assert gmm_cfg["covariance_type"] == "full"
    assert gmm_cfg["model_config_id"] == "GMM_k04_covFULL_d05"


def test_gmm_rejects_non_full_covariance_type():
    import pytest

    with pytest.raises(ValueError):
        gmm_adapter.GmmAdapter(covariance_type="diag")


def test_gmm_persistence_round_trip(tiny_shard_dir, tmp_path):
    manifest = data_scope.load_shard_manifest(tiny_shard_dir)
    train_raw = data_scope.load_filtered_split(tiny_shard_dir, manifest, "train", 13)
    validation_raw = data_scope.load_filtered_split(tiny_shard_dir, manifest, "validation", 13)

    adapter = gmm_adapter.GmmAdapter(max_iter=10)
    result = adapter.fit(train_raw, validation_raw, seed=20260720)
    assert result["status"] == "ok"
    adapter.save_bundle(tmp_path / "gmm_bundle")

    reloaded = gmm_adapter.GmmAdapter.load_bundle(tmp_path / "gmm_bundle")
    np.testing.assert_allclose(adapter.test_log_prob(validation_raw), reloaded.test_log_prob(validation_raw))
    np.testing.assert_allclose(adapter.sample(30, seed=1), reloaded.sample(30, seed=1))
    assert reloaded._fitting_log["seed"] == 20260720


def test_gmm_distinct_seeds_produce_distinct_run_identities(tiny_shard_dir):
    manifest = data_scope.load_shard_manifest(tiny_shard_dir)
    train_raw = data_scope.load_filtered_split(tiny_shard_dir, manifest, "train", 13)
    validation_raw = data_scope.load_filtered_split(tiny_shard_dir, manifest, "validation", 13)

    adapter_a = gmm_adapter.GmmAdapter(max_iter=10)
    result_a = adapter_a.fit(train_raw, validation_raw, seed=20260720)
    adapter_b = gmm_adapter.GmmAdapter(max_iter=10)
    result_b = adapter_b.fit(train_raw, validation_raw, seed=20260721)

    assert result_a["seed"] != result_b["seed"]
    assert result_a["seed"] == 20260720
    assert result_b["seed"] == 20260721


def test_gmm_non_convergence_remains_visible(tiny_shard_dir):
    manifest = data_scope.load_shard_manifest(tiny_shard_dir)
    train_raw = data_scope.load_filtered_split(tiny_shard_dir, manifest, "train", 13)
    validation_raw = data_scope.load_filtered_split(tiny_shard_dir, manifest, "validation", 13)

    # max_iter=1 on a 4-component full-covariance model is very unlikely to
    # satisfy sklearn's convergence tolerance in a single EM step.
    adapter = gmm_adapter.GmmAdapter(max_iter=1)
    result = adapter.fit(train_raw, validation_raw, seed=20260720)

    assert result["status"] == "ok"
    assert result["converged"] is False
    assert any("did not converge" in w for w in result["warnings"])
    assert result["n_em_iterations_completed"] == 1
    assert len(result["lower_bound_curve"]) == 1
