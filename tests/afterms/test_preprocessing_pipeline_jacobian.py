"""Focused tests for PreprocessingPipeline.forward_log_abs_det_jacobian.

This method had zero test coverage before this file -- the only Jacobian
tested elsewhere in the repo belongs to an unrelated class (`FeatureView` in
data_contracts/feature_views.py). These tests verify the sign convention
(physical_lp = normalized_lp + forward_log_abs_det_jacobian, i.e. the
log-abs-det-Jacobian of the forward x->z map) against numerical
differentiation for all three variants -- identity_standardized_v0 and
cartesian_log1p_pz_v0 were previously untested too -- plus the new
quantile_normal_v0 implementation's exact-domain rejection behavior.
"""

from __future__ import annotations

import numpy as np
import pytest

from ship_muon_bg.afterms.preprocessing import PreprocessingPipeline


def _make_raw_data(n, seed):
    rng = np.random.default_rng(seed)
    px = rng.normal(size=n)
    py = rng.normal(size=n)
    pz = rng.uniform(0.5, 5.0, size=n)
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    z = rng.normal(size=n)
    ids = rng.choice([13.0, -13.0], size=n)
    w = rng.uniform(0.1, 1.0, size=n)
    return np.column_stack([px, py, pz, x, y, z, ids, w]).astype(np.float64)


def _numerical_forward_log_abs_det_jacobian(pipeline, raw_data, h=1e-5):
    """Finite-difference log|det(dz/dx)| for a feature-wise (diagonal
    Jacobian) transform: perturb each of the first 5 raw columns
    independently and read off the corresponding output column's slope."""
    n = raw_data.shape[0]
    total = np.zeros(n)
    for j in range(5):
        plus = raw_data.copy()
        minus = raw_data.copy()
        plus[:, j] += h
        minus[:, j] -= h
        z_plus = pipeline.transform(plus)[:, j]
        z_minus = pipeline.transform(minus)[:, j]
        slope = (z_plus - z_minus) / (2 * h)
        total += np.log(np.abs(slope))
    return total


def test_identity_standardized_jacobian_matches_numerical_derivative():
    raw = _make_raw_data(500, seed=1)
    pipeline = PreprocessingPipeline("identity_standardized_v0", seed=20260720).fit(raw)
    analytic = pipeline.forward_log_abs_det_jacobian(raw)
    numerical = _numerical_forward_log_abs_det_jacobian(pipeline, raw)
    np.testing.assert_allclose(analytic, numerical, atol=1e-4)


def test_cartesian_log1p_pz_jacobian_matches_numerical_derivative():
    raw = _make_raw_data(500, seed=2)
    pipeline = PreprocessingPipeline("cartesian_log1p_pz_v0", seed=20260720).fit(raw)
    analytic = pipeline.forward_log_abs_det_jacobian(raw)
    numerical = _numerical_forward_log_abs_det_jacobian(pipeline, raw)
    np.testing.assert_allclose(analytic, numerical, atol=1e-4)


@pytest.mark.lab
def test_quantile_normal_jacobian_matches_numerical_derivative():
    train = _make_raw_data(5000, seed=3)
    pipeline = PreprocessingPipeline("quantile_normal_v0", seed=20260720).fit(train)

    eval_rows = _make_raw_data(200, seed=4)
    # Keep only rows comfortably inside the training range for all 5
    # features, so neither the analytic Jacobian's domain check nor the
    # finite-difference probe itself steps outside the fitted knots.
    features = eval_rows[:, :5]
    train_features = train[:, :5]
    margin = 0.05 * (train_features.max(axis=0) - train_features.min(axis=0))
    lo = train_features.min(axis=0) + margin
    hi = train_features.max(axis=0) - margin
    mask = np.all((features > lo) & (features < hi), axis=1)
    eval_rows = eval_rows[mask]
    assert eval_rows.shape[0] > 50

    # h must be small relative to the local quantile-knot spacing (which can
    # be ~5e-4 in dense regions for this fixture) -- otherwise the secant
    # straddles a knot boundary and the finite difference itself, not the
    # analytic formula, is wrong. h=1e-6 was verified safe here.
    analytic = pipeline.forward_log_abs_det_jacobian(eval_rows)
    numerical = _numerical_forward_log_abs_det_jacobian(pipeline, eval_rows, h=1e-6)
    np.testing.assert_allclose(analytic, numerical, atol=1e-4, rtol=1e-4)


@pytest.mark.lab
def test_quantile_normal_jacobian_raises_outside_fitted_range():
    train = _make_raw_data(2000, seed=5)
    pipeline = PreprocessingPipeline("quantile_normal_v0", seed=20260720).fit(train)
    out_of_range = train[:5].copy()
    out_of_range[:, 0] = train[:, 0].max() + 100.0  # far outside px's fitted range
    with pytest.raises(ValueError, match="outside the fitted quantile range"):
        pipeline.forward_log_abs_det_jacobian(out_of_range)


@pytest.mark.lab
def test_quantile_normal_jacobian_raises_on_degenerate_constant_feature():
    rng = np.random.default_rng(6)
    n = 2000
    px = np.full(n, 3.0)  # perfectly constant feature -> zero-width quantile knots
    py = rng.normal(size=n)
    pz = rng.uniform(0.5, 5.0, size=n)
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    z = rng.normal(size=n)
    ids = rng.choice([13.0, -13.0], size=n)
    w = rng.uniform(0.1, 1.0, size=n)
    raw = np.column_stack([px, py, pz, x, y, z, ids, w]).astype(np.float64)

    pipeline = PreprocessingPipeline("quantile_normal_v0", seed=20260720).fit(raw)
    with pytest.raises(ValueError):
        pipeline.forward_log_abs_det_jacobian(raw[:50])


@pytest.mark.lab
def test_quantile_normal_jacobian_raises_near_outer_knots():
    train = _make_raw_data(2000, seed=7)
    pipeline = PreprocessingPipeline("quantile_normal_v0", seed=20260720).fit(train)
    # Rows exactly at the training min are at the outermost knot, where
    # sklearn's internal probability clipping (before norm.ppf) applies --
    # the closed-form Jacobian must refuse these, not silently misreport.
    boundary_rows = train[:3].copy()
    boundary_rows[:, 1] = train[:, 1].min()
    with pytest.raises(ValueError):
        pipeline.forward_log_abs_det_jacobian(boundary_rows)
