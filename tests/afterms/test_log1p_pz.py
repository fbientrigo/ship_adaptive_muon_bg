import numpy as np
import pytest

from ship_muon_bg.afterms import log1p_pz


def test_roundtrip():
    rng = np.random.default_rng(0)
    rows = np.column_stack(
        [
            rng.normal(size=1000),
            rng.normal(size=1000),
            rng.uniform(0, 50, size=1000),
            rng.normal(size=1000),
            rng.normal(size=1000),
        ]
    )
    features = log1p_pz.transform_rows(rows)
    back = log1p_pz.inverse_transform_rows(features)
    np.testing.assert_allclose(back, rows, atol=1e-8)


def test_zero_pz_is_finite():
    u = log1p_pz.forward_log1p_pz(np.array([0.0]))
    assert np.isfinite(u).all()
    assert u[0] == 0.0
    pz_back = log1p_pz.inverse_log1p_pz(u)
    assert pz_back[0] == 0.0


def test_negative_pz_raises_by_default():
    with pytest.raises(log1p_pz.NegativePzError):
        log1p_pz.forward_log1p_pz(np.array([1.0, -0.5]))


def test_negative_pz_counted_when_not_raising():
    count, mask = log1p_pz.validate_pz_domain(
        np.array([1.0, -0.5, -2.0]), raise_on_negative=False
    )
    assert count == 2
    np.testing.assert_array_equal(mask, [False, True, True])


def test_jacobian_matches_numeric_derivative():
    s_pz = 1.0
    pz = np.array([1.0, 5.0, 50.0])
    analytic = log1p_pz.forward_log_abs_det_jacobian(pz, s_pz=s_pz)

    eps = 1e-6
    numeric = np.log(
        np.abs(
            (log1p_pz.forward_log1p_pz(pz + eps, s_pz=s_pz) - log1p_pz.forward_log1p_pz(pz - eps, s_pz=s_pz))
            / (2 * eps)
        )
    )
    np.testing.assert_allclose(analytic, numeric, atol=1e-4)


def test_config_hash_deterministic():
    assert log1p_pz.config_hash() == log1p_pz.config_hash()
    assert log1p_pz.config_hash(s_pz=1.0) != log1p_pz.config_hash(s_pz=2.0)
