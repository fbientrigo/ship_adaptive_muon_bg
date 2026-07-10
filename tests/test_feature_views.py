"""Tests for deterministic density feature views.

These tests validate the scientific contract before any flow backend exists:
shape/domain failures are explicit, both views are invertible, and analytic
Jacobian terms agree with finite-difference determinants. No FairShip, ROOT,
PyTorch or GPU is required.
"""

from __future__ import annotations

import numpy as np
import pytest

from ship_muon_bg.data_contracts import (
    CARTESIAN_LOG_VIEW_ID,
    SLOPE_LOG_VIEW_ID,
    FeatureView,
    FeatureViewConfigError,
    FeatureViewDomainError,
    FeatureViewShapeError,
)


VIEW_IDS = (CARTESIAN_LOG_VIEW_ID, SLOPE_LOG_VIEW_ID)


def _raw_rows() -> np.ndarray:
    """Return valid raw rows with non-trivial scales and both charges."""

    return np.asarray(
        [
            [1.2, -0.3, 12.0, -2.5, 0.7, 28.905, 13.0, 7.6875],
            [-4.5, 0.08, 55.0, 3.1, -1.2, 28.901, -13.0, 768.75],
            [0.02, 1.7, 180.0, 0.0, 2.8, 28.905, 13.0, 7.6875],
        ],
        dtype=np.float64,
    )


def _raw_from_physical(physical: np.ndarray) -> np.ndarray:
    """Embed ``[px, py, pz, x, y]`` into the fixed raw schema."""

    physical = np.asarray(physical, dtype=np.float64)
    raw = np.empty((physical.shape[0], 8), dtype=np.float64)
    raw[:, 0:3] = physical[:, 0:3]
    raw[:, 3:5] = physical[:, 3:5]
    raw[:, 5] = 28.905
    raw[:, 6] = 13.0
    raw[:, 7] = 1.0
    return raw


def _numerical_forward_logdet(view: FeatureView, physical: np.ndarray) -> float:
    """Central-difference estimate of one forward-transform log determinant."""

    point = np.asarray(physical, dtype=np.float64)
    jacobian = np.empty((5, 5), dtype=np.float64)
    for column in range(5):
        step = 1.0e-6 * max(1.0, abs(float(point[column])))
        plus = point.copy()
        minus = point.copy()
        plus[column] += step
        minus[column] -= step
        assert minus[2] > 0.0
        out_plus = view.forward(_raw_from_physical(plus[None, :]))[0]
        out_minus = view.forward(_raw_from_physical(minus[None, :]))[0]
        jacobian[:, column] = (out_plus - out_minus) / (2.0 * step)

    sign, log_abs_det = np.linalg.slogdet(jacobian)
    assert sign > 0.0
    return float(log_abs_det)


@pytest.mark.parametrize("view_id", VIEW_IDS)
def test_forward_inverse_round_trip(view_id: str) -> None:
    raw = _raw_rows()
    expected = raw[:, [0, 1, 2, 3, 4]]
    view = FeatureView(view_id)

    features = view.forward(raw)
    recovered = view.inverse(features)

    assert features.shape == (raw.shape[0], 5)
    assert features.dtype == np.float64
    assert features.flags.c_contiguous
    np.testing.assert_allclose(recovered, expected, rtol=1.0e-13, atol=1.0e-13)


@pytest.mark.parametrize("view_id", VIEW_IDS)
def test_forward_does_not_mutate_raw_input(view_id: str) -> None:
    raw = _raw_rows()
    original = raw.copy()
    FeatureView(view_id).forward(raw)
    np.testing.assert_array_equal(raw, original)


@pytest.mark.parametrize("view_id", VIEW_IDS)
def test_z_id_and_weight_are_not_features(view_id: str) -> None:
    raw_a = _raw_rows()
    raw_b = raw_a.copy()
    raw_b[:, 5] += np.asarray([100.0, -50.0, 0.2])
    raw_b[:, 6] *= -1.0
    raw_b[:, 7] *= np.asarray([3.0, 0.5, 10.0])

    view = FeatureView(view_id)
    np.testing.assert_array_equal(view.forward(raw_a), view.forward(raw_b))


@pytest.mark.parametrize("view_id", VIEW_IDS)
def test_non_default_pz_unit_is_invertible(view_id: str) -> None:
    raw = _raw_rows()
    view = FeatureView(view_id, pz_unit_gev=10.0)
    features = view.forward(raw)
    recovered = view.inverse(features)
    np.testing.assert_allclose(
        recovered,
        raw[:, [0, 1, 2, 3, 4]],
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_cartesian_forward_values() -> None:
    raw = _raw_rows()[:1]
    features = FeatureView(CARTESIAN_LOG_VIEW_ID).forward(raw)
    expected = np.asarray(
        [[raw[0, 0], raw[0, 1], np.log(raw[0, 2]), raw[0, 3], raw[0, 4]]]
    )
    np.testing.assert_allclose(features, expected)


def test_slope_forward_values() -> None:
    raw = _raw_rows()[:1]
    features = FeatureView(SLOPE_LOG_VIEW_ID).forward(raw)
    expected = np.asarray(
        [[
            raw[0, 0] / raw[0, 2],
            raw[0, 1] / raw[0, 2],
            np.log(raw[0, 2]),
            raw[0, 3],
            raw[0, 4],
        ]]
    )
    np.testing.assert_allclose(features, expected)


@pytest.mark.parametrize(
    ("view_id", "power"),
    ((CARTESIAN_LOG_VIEW_ID, 1.0), (SLOPE_LOG_VIEW_ID, 3.0)),
)
def test_analytic_jacobian_formula(view_id: str, power: float) -> None:
    raw = _raw_rows()
    expected = -power * np.log(raw[:, 2])
    actual = FeatureView(view_id).forward_log_abs_det_jacobian(raw)
    np.testing.assert_allclose(actual, expected, rtol=1.0e-14, atol=1.0e-14)


@pytest.mark.parametrize("view_id", VIEW_IDS)
def test_forward_and_inverse_jacobians_cancel(view_id: str) -> None:
    raw = _raw_rows()
    view = FeatureView(view_id)
    features = view.forward(raw)
    total = (
        view.forward_log_abs_det_jacobian(raw)
        + view.inverse_log_abs_det_jacobian(features)
    )
    np.testing.assert_allclose(total, np.zeros(raw.shape[0]), atol=1.0e-13)


@pytest.mark.parametrize("view_id", VIEW_IDS)
def test_analytic_jacobian_matches_finite_difference(view_id: str) -> None:
    view = FeatureView(view_id)
    physical = np.asarray([1.7, -0.4, 42.0, 2.3, -0.8], dtype=np.float64)
    raw = _raw_from_physical(physical[None, :])

    analytic = float(view.forward_log_abs_det_jacobian(raw)[0])
    numerical = _numerical_forward_logdet(view, physical)

    assert analytic == pytest.approx(numerical, rel=2.0e-7, abs=2.0e-7)


@pytest.mark.parametrize("view_id", VIEW_IDS)
def test_physical_log_prob_adds_forward_jacobian(view_id: str) -> None:
    raw = _raw_rows()
    feature_log_prob = np.asarray([-2.0, -3.5, -8.0], dtype=np.float64)
    view = FeatureView(view_id)
    expected = feature_log_prob + view.forward_log_abs_det_jacobian(raw)
    actual = view.physical_log_prob_from_feature(feature_log_prob, raw)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("view_id", VIEW_IDS)
def test_manifest_is_explicit_and_hash_is_deterministic(view_id: str) -> None:
    view = FeatureView(view_id)
    manifest = view.manifest()

    assert manifest["feature_view_id"] == view_id
    assert manifest["physical_input_columns"] == ["px", "py", "pz", "x", "y"]
    assert manifest["excluded_raw_columns"] == ["z", "id", "w"]
    assert manifest["domain"] == {"pz": "strictly_positive"}
    assert manifest["includes_train_fitted_standardization"] is False
    assert view.config_hash() == FeatureView(view_id).config_hash()


def test_config_hash_changes_with_view_or_unit() -> None:
    cartesian = FeatureView(CARTESIAN_LOG_VIEW_ID)
    slope = FeatureView(SLOPE_LOG_VIEW_ID)
    rescaled = FeatureView(CARTESIAN_LOG_VIEW_ID, pz_unit_gev=10.0)
    assert len({cartesian.config_hash(), slope.config_hash(), rescaled.config_hash()}) == 3


@pytest.mark.parametrize("bad_view", ["", "cartesian", "unknown_v0"])
def test_unsupported_view_fails(bad_view: str) -> None:
    with pytest.raises(FeatureViewConfigError):
        FeatureView(bad_view)


@pytest.mark.parametrize("bad_unit", [0.0, -1.0, np.nan, np.inf, "not-a-number"])
def test_invalid_pz_unit_fails(bad_unit: object) -> None:
    with pytest.raises(FeatureViewConfigError):
        FeatureView(CARTESIAN_LOG_VIEW_ID, pz_unit_gev=bad_unit)  # type: ignore[arg-type]


@pytest.mark.parametrize("view_id", VIEW_IDS)
@pytest.mark.parametrize("bad_pz", [0.0, -0.1])
def test_non_positive_pz_fails(view_id: str, bad_pz: float) -> None:
    raw = _raw_rows()
    raw[1, 2] = bad_pz
    view = FeatureView(view_id)
    with pytest.raises(FeatureViewDomainError):
        view.forward(raw)
    with pytest.raises(FeatureViewDomainError):
        view.forward_log_abs_det_jacobian(raw)


@pytest.mark.parametrize("view_id", VIEW_IDS)
def test_non_finite_modelled_coordinate_fails(view_id: str) -> None:
    raw = _raw_rows()
    raw[0, 0] = np.nan
    with pytest.raises(FeatureViewDomainError):
        FeatureView(view_id).forward(raw)


@pytest.mark.parametrize("view_id", VIEW_IDS)
def test_invalid_raw_shape_fails(view_id: str) -> None:
    with pytest.raises(FeatureViewShapeError):
        FeatureView(view_id).forward(np.ones((4, 7), dtype=np.float64))


@pytest.mark.parametrize("view_id", VIEW_IDS)
def test_invalid_feature_shape_fails(view_id: str) -> None:
    with pytest.raises(FeatureViewShapeError):
        FeatureView(view_id).inverse(np.ones((4, 4), dtype=np.float64))


@pytest.mark.parametrize("view_id", VIEW_IDS)
def test_inverse_overflow_fails_explicitly(view_id: str) -> None:
    features = np.zeros((1, 5), dtype=np.float64)
    features[0, 2] = 1.0e4
    with pytest.raises(FeatureViewDomainError):
        FeatureView(view_id).inverse(features)


def test_log_prob_shape_is_checked() -> None:
    raw = _raw_rows()
    view = FeatureView(CARTESIAN_LOG_VIEW_ID)
    with pytest.raises(FeatureViewShapeError):
        view.physical_log_prob_from_feature(np.zeros((raw.shape[0], 1)), raw)
