"""Gaussian control adapter tests: streaming statistics, persistence, physical NLL.

Covers required tests 13, 14, 15, 16.
"""

from __future__ import annotations

import numpy as np

from Nflow.baselines.gaussian import DiagonalGaussian
from ship_muon_bg.afterms.d9_5 import data_scope, gaussian_adapter


def test_streaming_diagonal_statistics_equal_direct_computation():
    rng = np.random.default_rng(42)
    features = rng.normal(loc=3.0, scale=2.0, size=(2500, 5))
    mean_stream, var_stream, n = gaussian_adapter.streaming_diagonal_statistics(features, chunk_rows=333)
    mean_direct = features.mean(axis=0)
    var_direct = np.mean((features - mean_direct) ** 2, axis=0)
    assert n == 2500
    np.testing.assert_allclose(mean_stream, mean_direct, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(var_stream, var_direct, rtol=1e-10, atol=1e-12)


def test_streaming_full_covariance_equals_direct_computation():
    rng = np.random.default_rng(43)
    features = rng.normal(size=(2500, 5)) @ rng.normal(size=(5, 5))  # correlated features
    mean_stream, cov_stream, n = gaussian_adapter.streaming_full_statistics(features, chunk_rows=333)
    mean_direct = features.mean(axis=0)
    centered = features - mean_direct
    cov_direct = centered.T @ centered / features.shape[0]
    assert n == 2500
    np.testing.assert_allclose(mean_stream, mean_direct, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(cov_stream, cov_direct, rtol=1e-10, atol=1e-12)


def test_gaussian_diag_persistence_round_trip(tiny_shard_dir, tmp_path):
    manifest = data_scope.load_shard_manifest(tiny_shard_dir)
    train_raw = data_scope.load_filtered_split(tiny_shard_dir, manifest, "train", 13)
    validation_raw = data_scope.load_filtered_split(tiny_shard_dir, manifest, "validation", 13)

    adapter = gaussian_adapter.DiagonalGaussianAdapter()
    adapter.fit(train_raw, validation_raw, seed=0)
    adapter.save_bundle(tmp_path / "gauss_diag_bundle")

    reloaded = gaussian_adapter.DiagonalGaussianAdapter.load_bundle(tmp_path / "gauss_diag_bundle")
    np.testing.assert_allclose(adapter.test_log_prob(validation_raw), reloaded.test_log_prob(validation_raw))
    np.testing.assert_allclose(adapter.sample(50, seed=1), reloaded.sample(50, seed=1))


def test_gaussian_full_persistence_round_trip(tiny_shard_dir, tmp_path):
    manifest = data_scope.load_shard_manifest(tiny_shard_dir)
    train_raw = data_scope.load_filtered_split(tiny_shard_dir, manifest, "train", -13)
    validation_raw = data_scope.load_filtered_split(tiny_shard_dir, manifest, "validation", -13)

    adapter = gaussian_adapter.FullGaussianAdapter()
    adapter.fit(train_raw, validation_raw, seed=0)
    adapter.save_bundle(tmp_path / "gauss_full_bundle")

    reloaded = gaussian_adapter.FullGaussianAdapter.load_bundle(tmp_path / "gauss_full_bundle")
    np.testing.assert_allclose(adapter.test_log_prob(validation_raw), reloaded.test_log_prob(validation_raw))
    np.testing.assert_allclose(adapter.sample(50, seed=1), reloaded.sample(50, seed=1))


def test_gaussian_physical_nll_matches_independent_raw_space_calculation(tiny_shard_dir):
    """Sec 7: physical-space NLL from (standardized-space model + Jacobian)
    must equal an independently computed raw-space Gaussian NLL -- an exact
    algebraic identity for an affine standardization, not an approximation."""

    manifest = data_scope.load_shard_manifest(tiny_shard_dir)
    train_raw = data_scope.load_filtered_split(tiny_shard_dir, manifest, "train", 13)
    validation_raw = data_scope.load_filtered_split(tiny_shard_dir, manifest, "validation", 13)

    adapter = gaussian_adapter.DiagonalGaussianAdapter()
    adapter.fit(train_raw, validation_raw, seed=0)
    feature_log_prob = adapter.test_log_prob(validation_raw)
    log_jacobian = adapter.pipeline.forward_log_abs_det_jacobian(validation_raw)
    physical_nll = float(-np.mean(feature_log_prob + log_jacobian))

    direct = DiagonalGaussian(dimension=5)
    direct.fit(train_raw[:, :5])
    direct_nll = float(-np.mean(direct.log_prob(validation_raw[:, :5])))

    assert np.isclose(physical_nll, direct_nll, rtol=1e-6, atol=1e-8)


def test_gaussian_capability_metadata_has_required_keys(tiny_shard_dir):
    manifest = data_scope.load_shard_manifest(tiny_shard_dir)
    train_raw = data_scope.load_filtered_split(tiny_shard_dir, manifest, "train", 13)
    validation_raw = data_scope.load_filtered_split(tiny_shard_dir, manifest, "validation", 13)
    adapter = gaussian_adapter.DiagonalGaussianAdapter()
    adapter.fit(train_raw, validation_raw, seed=0)
    capabilities = adapter.capability_metadata()
    assert capabilities["stochastic_fit"] is False
    assert capabilities["supports_physical_log_prob"] is True
