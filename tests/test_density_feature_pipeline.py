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


# --- fit_macro_weighted: exact charge-macro-balanced physical-weight measure ----


def _shifted_raw(px, weight, *, pdg_id):
    """One synthetic (n, 8) raw block; every feature is an affine shift of px.

    Shift-invariance of variance (``Var(px + c) == Var(px)``) means every
    column's expected weighted std equals column 0's, without redoing the
    hand computation per column.
    """

    px = np.asarray(px, dtype=np.float64)
    n = px.shape[0]
    raw = np.zeros((n, 8), dtype=np.float64)
    raw[:, 0] = px
    raw[:, 1] = px + 10.0
    raw[:, 2] = px + 100.0
    raw[:, 3] = px + 1000.0
    raw[:, 4] = px + 10000.0
    raw[:, 6] = float(pdg_id)
    raw[:, 7] = np.asarray(weight, dtype=np.float64)
    return raw


def test_fit_macro_weighted_matches_exact_manual_formula_small_array():
    raw_13 = _shifted_raw([1.0, 3.0], [1.0, 3.0], pdg_id=13)
    raw_m13 = _shifted_raw([2.0, 4.0, 6.0], [1.0, 1.0, 2.0], pdg_id=-13)

    pipeline = FittedFeaturePipeline.fit_macro_weighted(
        raw_by_charge={13: raw_13, -13: raw_m13},
        weights_by_charge={13: raw_13[:, 7], -13: raw_m13[:, 7]},
        feature_view=FeatureView(IDENTITY_CARTESIAN_VIEW_ID),
    )

    pi_13 = np.array([1.0, 3.0]) / 4.0
    pi_m13 = np.array([1.0, 1.0, 2.0]) / 4.0
    mean_13 = float(np.sum(pi_13 * raw_13[:, 0]))
    mean_m13 = float(np.sum(pi_m13 * raw_m13[:, 0]))
    expected_mean_px = 0.5 * (mean_13 + mean_m13)
    var_13 = float(np.sum(pi_13 * (raw_13[:, 0] - expected_mean_px) ** 2))
    var_m13 = float(np.sum(pi_m13 * (raw_m13[:, 0] - expected_mean_px) ** 2))
    expected_std_px = float(np.sqrt(0.5 * (var_13 + var_m13)))

    manifest = pipeline.manifest()
    mean = manifest["standardization"]["mean"]
    std = manifest["standardization"]["std"]
    assert mean[0] == pytest.approx(expected_mean_px)
    assert std[0] == pytest.approx(expected_std_px)
    # Shift-invariance: every other column shares the same std, mean shifted.
    for i, shift in enumerate((10.0, 100.0, 1000.0, 10000.0), start=1):
        assert mean[i] == pytest.approx(expected_mean_px + shift)
        assert std[i] == pytest.approx(expected_std_px)
    assert manifest["standardization"]["weighting"] == "macro_balanced_physical_weight_train_only"
    assert pipeline.n_train_rows == raw_13.shape[0] + raw_m13.shape[0]


def test_fit_macro_weighted_gives_each_charge_equal_share_regardless_of_row_count():
    raw_a = _shifted_raw([9.0, 11.0], [1.0, 1.0], pdg_id=13)  # mean 10, 2 rows
    raw_b = _shifted_raw([15.0, 17.0, 19.0, 21.0, 23.0], [1.0] * 5, pdg_id=-13)  # mean 19, 5 rows

    pipeline = FittedFeaturePipeline.fit_macro_weighted(
        raw_by_charge={13: raw_a, -13: raw_b},
        weights_by_charge={13: raw_a[:, 7], -13: raw_b[:, 7]},
        feature_view=FeatureView(IDENTITY_CARTESIAN_VIEW_ID),
    )
    macro_mean = pipeline.manifest()["standardization"]["mean"][0]
    pooled_row_weighted_mean = float(np.concatenate((raw_a[:, 0], raw_b[:, 0])).mean())
    assert macro_mean == pytest.approx(14.5)  # (1/2)*10 + (1/2)*19
    assert macro_mean != pytest.approx(pooled_row_weighted_mean)


def test_fit_macro_weighted_requires_matching_charge_keys():
    raw_13 = _shifted_raw([1.0, 2.0], [1.0, 1.0], pdg_id=13)
    with pytest.raises(FeaturePipelineError):
        FittedFeaturePipeline.fit_macro_weighted(
            raw_by_charge={13: raw_13},
            weights_by_charge={-13: raw_13[:, 7]},
            feature_view=FeatureView(IDENTITY_CARTESIAN_VIEW_ID),
        )


def test_fit_macro_weighted_zero_variance_error_and_unit_fallback():
    raw_13 = _shifted_raw([5.0, 5.0], [1.0, 1.0], pdg_id=13)  # constant px
    raw_m13 = _shifted_raw([5.0, 5.0, 5.0], [1.0, 1.0, 1.0], pdg_id=-13)  # constant px
    charges = {13: raw_13, -13: raw_m13}
    weights = {13: raw_13[:, 7], -13: raw_m13[:, 7]}
    with pytest.raises(FeaturePipelineError):
        FittedFeaturePipeline.fit_macro_weighted(
            raw_by_charge=charges, weights_by_charge=weights,
            feature_view=FeatureView(IDENTITY_CARTESIAN_VIEW_ID),
        )
    pipeline = FittedFeaturePipeline.fit_macro_weighted(
        raw_by_charge=charges, weights_by_charge=weights,
        feature_view=FeatureView(IDENTITY_CARTESIAN_VIEW_ID),
        zero_variance_policy="unit",
    )
    assert pipeline.manifest()["standardization"]["std"][0] == 1.0


def test_fit_macro_weighted_never_uses_a_seed_and_is_reproducible():
    raw_13 = _shifted_raw([1.0, 3.0, 5.0], [2.0, 1.0, 1.0], pdg_id=13)
    raw_m13 = _shifted_raw([2.0, 4.0], [1.0, 3.0], pdg_id=-13)
    charges = {13: raw_13, -13: raw_m13}
    weights = {13: raw_13[:, 7], -13: raw_m13[:, 7]}
    a = FittedFeaturePipeline.fit_macro_weighted(
        raw_by_charge=charges, weights_by_charge=weights,
        feature_view=FeatureView(IDENTITY_CARTESIAN_VIEW_ID),
    )
    b = FittedFeaturePipeline.fit_macro_weighted(
        raw_by_charge=charges, weights_by_charge=weights,
        feature_view=FeatureView(IDENTITY_CARTESIAN_VIEW_ID),
    )
    assert a.config_hash() == b.config_hash()
