"""Tests for the train-only per-view feature pipeline.

Detects wrong Jacobian sign, validation leakage during fit, normalization
reuse across views, fitted-array mutation/aliasing, and round-trip failures.
Pure NumPy.
"""

from __future__ import annotations

import numpy as np
import pytest

from ship_muon_bg.benchmarks import embed_physical_to_raw, make_controlled_target
from ship_muon_bg.data_contracts import (
    CARTESIAN_LOGPZ_VIEW_ID,
    IDENTITY_CARTESIAN_VIEW_ID,
    SLOPE_LOGPZ_VIEW_ID,
    FeatureView,
)
from ship_muon_bg.density_lab import FeaturePipelineError, FittedFeaturePipeline

VIEW_IDS = (IDENTITY_CARTESIAN_VIEW_ID, CARTESIAN_LOGPZ_VIEW_ID, SLOPE_LOGPZ_VIEW_ID)


def _raw(target_id="D2", pdg_id=13, n=2000, seed=11, plane_z=28.905):
    target = make_controlled_target(target_id)
    batch = target.sample(n, pdg_id=pdg_id, seed=seed)
    return batch.to_raw(plane_z=plane_z), batch.physical, target


@pytest.mark.parametrize("view_id", VIEW_IDS)
def test_standardized_features_have_zero_mean_unit_std_on_train(view_id):
    raw, _, _ = _raw()
    pipe = FittedFeaturePipeline.fit(raw, FeatureView(view_id))
    normalized = pipe.transform_raw(raw)
    np.testing.assert_allclose(normalized.mean(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(normalized.std(axis=0), 1.0, atol=1e-10)


@pytest.mark.parametrize("view_id", VIEW_IDS)
def test_roundtrip_inverse_to_physical(view_id):
    raw, physical, _ = _raw()
    pipe = FittedFeaturePipeline.fit(raw, FeatureView(view_id))
    normalized = pipe.transform_raw(raw)
    recovered = pipe.inverse_to_physical(normalized)
    np.testing.assert_allclose(recovered, physical, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("view_id", VIEW_IDS)
def test_physical_log_prob_accounting_matches_target(view_id):
    # If the pipeline density accounting is correct, a model whose normalized
    # log-density equals the *true* normalized target density must reproduce
    # the exact physical target density.
    raw, physical, target = _raw(target_id="D2", pdg_id=13)
    view = FeatureView(view_id)
    pipe = FittedFeaturePipeline.fit(raw, view)
    physical_lp = target.log_prob(physical, pdg_id=13)
    # normalized-space target log density = physical_lp - (view_jac + norm_jac)
    view_jac = view.forward_log_abs_det_jacobian(raw)
    norm_jac = pipe.manifest()["normalization_forward_log_jacobian"]
    normalized_lp = physical_lp - view_jac - norm_jac
    recovered = pipe.normalized_to_physical_log_prob(normalized_lp, raw)
    np.testing.assert_allclose(recovered, physical_lp, rtol=1e-9, atol=1e-9)


def test_physical_log_prob_accepts_physical_rows():
    raw, physical, target = _raw(target_id="D1", pdg_id=13)
    view = FeatureView(CARTESIAN_LOGPZ_VIEW_ID)
    pipe = FittedFeaturePipeline.fit(raw, view)
    normalized_lp = np.zeros(physical.shape[0])
    from_raw = pipe.normalized_to_physical_log_prob(normalized_lp, raw)
    from_physical = pipe.normalized_to_physical_log_prob(normalized_lp, physical)
    np.testing.assert_allclose(from_raw, from_physical, rtol=1e-12, atol=1e-12)


def test_wrong_jacobian_sign_is_detectable():
    # A sign flip in the normalization Jacobian would shift log-prob by
    # 2*sum(log std) != 0, so the accounting test above would fail. Confirm the
    # magnitude here so the guard is meaningful.
    raw, _, _ = _raw()
    pipe = FittedFeaturePipeline.fit(raw, FeatureView(CARTESIAN_LOGPZ_VIEW_ID))
    forward = pipe.manifest()["normalization_forward_log_jacobian"]
    assert abs(forward) > 1e-3  # non-trivial, so a sign error is observable


def test_fit_uses_only_train_rows_no_leakage():
    # Fitting on a train subset must not see validation rows: the recorded
    # mean/std must equal those computed from the train subset alone.
    raw, _, _ = _raw(n=4000)
    train = raw[:3000]
    view = FeatureView(IDENTITY_CARTESIAN_VIEW_ID)
    pipe = FittedFeaturePipeline.fit(train, view)
    train_features = view.forward(train)
    np.testing.assert_allclose(
        pipe.manifest()["standardization"]["mean"], train_features.mean(axis=0)
    )
    np.testing.assert_allclose(
        pipe.manifest()["standardization"]["std"], train_features.std(axis=0)
    )


def test_normalization_differs_across_views():
    raw, _, _ = _raw()
    hashes = {
        vid: FittedFeaturePipeline.fit(raw, FeatureView(vid)).config_hash()
        for vid in VIEW_IDS
    }
    assert len(set(hashes.values())) == len(VIEW_IDS)


def test_fitted_arrays_are_immutable_and_not_aliased():
    raw, _, _ = _raw()
    pipe = FittedFeaturePipeline.fit(raw, FeatureView(IDENTITY_CARTESIAN_VIEW_ID))
    assert pipe._mean.flags["WRITEABLE"] is False
    assert pipe._std.flags["WRITEABLE"] is False
    with pytest.raises(ValueError):
        pipe._mean[0] = 5.0


def test_zero_variance_policy_error():
    # Build raw rows with an identically-constant feature (all py equal).
    physical = np.tile(np.array([1.0, 2.0, 50.0, 0.3, -0.2]), (100, 1))
    physical[:, 0] = np.linspace(-1, 1, 100)  # px varies, py constant
    raw = embed_physical_to_raw(physical, pdg_id=13, plane_z=0.0)
    with pytest.raises(FeaturePipelineError):
        FittedFeaturePipeline.fit(raw, FeatureView(IDENTITY_CARTESIAN_VIEW_ID))


def test_zero_variance_policy_unit_fallback():
    physical = np.tile(np.array([1.0, 2.0, 50.0, 0.3, -0.2]), (100, 1))
    physical[:, 0] = np.linspace(-1, 1, 100)
    raw = embed_physical_to_raw(physical, pdg_id=13, plane_z=0.0)
    pipe = FittedFeaturePipeline.fit(
        raw, FeatureView(IDENTITY_CARTESIAN_VIEW_ID), zero_variance_policy="unit"
    )
    std = np.array(pipe.manifest()["standardization"]["std"])
    assert std[1] == 1.0  # constant py fell back to unit std


def test_config_hash_deterministic():
    raw, _, _ = _raw()
    a = FittedFeaturePipeline.fit(raw, FeatureView(IDENTITY_CARTESIAN_VIEW_ID))
    b = FittedFeaturePipeline.fit(raw, FeatureView(IDENTITY_CARTESIAN_VIEW_ID))
    assert a.config_hash() == b.config_hash()
