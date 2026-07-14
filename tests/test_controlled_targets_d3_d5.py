"""Tests for the exact D3-D5 controlled targets and their transforms (v0).

D3 (curved banana), D4 (heteroscedastic + sinh-arcsinh skew) and D5 (rare tail
mode) are exact: analytic inverse, analytic log-Jacobian, pz preserved, exact
component labels. Pure NumPy; no torch/ROOT/FairShip.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys

import numpy as np
import pytest

from ship_muon_bg.benchmarks import (
    ComposedTransform,
    ControlledTargetConfigError,
    SinhArcsinhSkewTransform,
    TransformedControlledTarget,
    TriangularBananaTransform,
    calibrate_d5_rare_region,
    make_controlled_target,
    numerical_forward_log_abs_det_jacobian,
)
from ship_muon_bg.benchmarks.controlled_targets import (
    D5_CALIBRATION_N,
    D5_CALIBRATION_SEED,
)
from ship_muon_bg.data_contracts import (
    CARTESIAN_LOGPZ_VIEW_ID,
    IDENTITY_CARTESIAN_VIEW_ID,
    SLOPE_LOGPZ_VIEW_ID,
    FeatureView,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO_ROOT, "src")
PZ_INDEX = 2
FEATURE_VIEW_IDS = (
    IDENTITY_CARTESIAN_VIEW_ID,
    CARTESIAN_LOGPZ_VIEW_ID,
    SLOPE_LOGPZ_VIEW_ID,
)


def _sample_base(target, pdg_id=13, n=64, seed=3):
    """Draw base-coordinate points for Jacobian/roundtrip checks."""

    batch = target.sample(n, pdg_id=pdg_id, seed=seed)
    return target.base_of(batch.physical, pdg_id=pdg_id)


# --- 1. Factory / variants --------------------------------------------------


@pytest.mark.parametrize("target_id", ["D3", "D4"])
def test_d3_d4_factory(target_id):
    target = make_controlled_target(target_id)
    assert target.target_id == target_id
    assert isinstance(target, TransformedControlledTarget)


@pytest.mark.parametrize("variant", ["rare_1e-2", "rare_1e-3"])
def test_d5_variants(variant):
    target = make_controlled_target("D5", variant=variant)
    assert target.target_id == "D5"
    assert target.target_variant == variant


def test_d5_default_variant():
    target = make_controlled_target("D5")
    assert target.target_variant == "rare_1e-2"


def test_d5_rejects_unknown_variant():
    with pytest.raises(ControlledTargetConfigError):
        make_controlled_target("D5", variant="rare_1e-5")


def test_non_d5_rejects_variant():
    with pytest.raises(ControlledTargetConfigError):
        make_controlled_target("D3", variant="rare_1e-2")


# --- 2. Transforms: roundtrip and analytic vs numerical Jacobian ------------


def _transforms():
    banana = TriangularBananaTransform(
        scale_px=3.0, scale_py=3.0, scale_x=0.5, curvature=0.8, shear=0.5
    )
    skew = SinhArcsinhSkewTransform(
        scales=(4.0, 4.0, 5.0, 0.6, 0.6), skews=(0.5, -0.3, 0.0, 0.2, -0.2)
    )
    composed = ComposedTransform([skew, banana])
    return [("banana", banana), ("skew", skew), ("composed", composed)]


@pytest.mark.parametrize("name,transform", _transforms())
def test_transform_forward_inverse_roundtrip(name, transform):
    rng = np.random.default_rng(0)
    base = rng.standard_normal((100, 5))
    base[:, PZ_INDEX] = 50.0 + 4.0 * rng.standard_normal(100)
    recovered = transform.inverse(transform.forward(base))
    np.testing.assert_allclose(recovered, base, rtol=0, atol=1e-9)


@pytest.mark.parametrize("name,transform", _transforms())
def test_transform_pz_unchanged(name, transform):
    rng = np.random.default_rng(1)
    base = rng.standard_normal((50, 5))
    base[:, PZ_INDEX] = 50.0 + 4.0 * rng.standard_normal(50)
    forward = transform.forward(base)
    np.testing.assert_array_equal(forward[:, PZ_INDEX], base[:, PZ_INDEX])


@pytest.mark.parametrize("name,transform", _transforms())
def test_transform_analytic_jacobian_matches_finite_difference(name, transform):
    rng = np.random.default_rng(2)
    base = rng.standard_normal((30, 5))
    base[:, PZ_INDEX] = 50.0 + 3.0 * rng.standard_normal(30)
    analytic = transform.forward_log_abs_det_jacobian(base)
    numerical = numerical_forward_log_abs_det_jacobian(transform, base)
    np.testing.assert_allclose(analytic, numerical, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("name,transform", _transforms())
def test_transform_forward_inverse_jacobian_cancel(name, transform):
    rng = np.random.default_rng(3)
    base = rng.standard_normal((40, 5))
    base[:, PZ_INDEX] = 50.0 + 4.0 * rng.standard_normal(40)
    forward = transform.forward(base)
    fwd_jac = transform.forward_log_abs_det_jacobian(base)
    inv_jac = transform.inverse_log_abs_det_jacobian(forward)
    np.testing.assert_allclose(fwd_jac + inv_jac, 0.0, atol=1e-9)


def test_transform_does_not_mutate_input():
    transform = TriangularBananaTransform(
        scale_px=3.0, scale_py=3.0, scale_x=0.5, curvature=0.8, shear=0.5
    )
    base = np.ones((5, 5)) * 2.0
    base[:, PZ_INDEX] = 50.0
    original = base.copy()
    transform.forward(base)
    transform.inverse(base)
    transform.forward_log_abs_det_jacobian(base)
    np.testing.assert_array_equal(base, original)


def test_banana_transform_has_unit_determinant():
    transform = TriangularBananaTransform(
        scale_px=3.0, scale_py=3.0, scale_x=0.5, curvature=0.8, shear=0.5
    )
    base = np.random.default_rng(4).standard_normal((20, 5))
    base[:, PZ_INDEX] = 50.0
    np.testing.assert_array_equal(
        transform.forward_log_abs_det_jacobian(base), np.zeros(20)
    )


def test_skew_rejects_nonzero_pz_skew():
    with pytest.raises(ValueError):
        SinhArcsinhSkewTransform(
            scales=(4.0, 4.0, 5.0, 0.6, 0.6), skews=(0.5, 0.0, 0.3, 0.0, 0.0)
        )


# --- 3. Target exactness: change-of-variables density consistency -----------


@pytest.mark.parametrize("target_id", ["D3", "D4"])
@pytest.mark.parametrize("pdg_id", [13, -13])
def test_target_density_change_of_variables(target_id, pdg_id):
    target = make_controlled_target(target_id)
    batch = target.sample(500, pdg_id=pdg_id, seed=7)
    base = target.base_of(batch.physical, pdg_id=pdg_id)
    # p_T(t) = p_base(inverse(t)) * |det d(inverse)/dt|
    expected = target._base.log_prob(
        base, pdg_id=pdg_id
    ) + target._transform.inverse_log_abs_det_jacobian(batch.physical)
    actual = target.log_prob(batch.physical, pdg_id=pdg_id)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert np.isfinite(actual).all()


@pytest.mark.parametrize("target_id", ["D3", "D4", "D5"])
def test_deterministic_sampling(target_id):
    target = make_controlled_target(target_id)
    a = target.sample(256, pdg_id=13, seed=11)
    b = target.sample(256, pdg_id=13, seed=11)
    np.testing.assert_array_equal(a.physical, b.physical)
    np.testing.assert_array_equal(a.component_id, b.component_id)


@pytest.mark.parametrize("target_id", ["D3", "D4", "D5"])
def test_pz_positive_on_samples(target_id):
    target = make_controlled_target(target_id)
    for pdg_id in (13, -13):
        batch = target.sample(8192, pdg_id=pdg_id, seed=5)
        assert np.all(batch.physical[:, PZ_INDEX] > 0.0)


# --- 4. Component posterior normalization -----------------------------------


@pytest.mark.parametrize("target_id", ["D3", "D4", "D5"])
def test_component_posterior_normalized(target_id):
    target = make_controlled_target(target_id)
    batch = target.sample(300, pdg_id=13, seed=2)
    posterior = target.component_posterior(batch.physical, pdg_id=13)
    row_sums = posterior.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, rtol=1e-10, atol=1e-10)
    assert np.all(posterior >= -1e-12)


def test_component_log_prob_logsumexp_equals_log_prob():
    target = make_controlled_target("D4")
    batch = target.sample(200, pdg_id=-13, seed=6)
    clp = target.component_log_prob(batch.physical, pdg_id=-13)
    lse = np.logaddexp.reduce(clp, axis=1)
    np.testing.assert_allclose(
        lse, target.log_prob(batch.physical, pdg_id=-13), rtol=1e-10, atol=1e-10
    )


# --- 5. Feature-view physical log-prob consistency (A/B1/B2) -----------------


@pytest.mark.parametrize("target_id", ["D3", "D4", "D5"])
@pytest.mark.parametrize("view_id", FEATURE_VIEW_IDS)
def test_feature_view_recovers_physical_log_prob(target_id, view_id):
    target = make_controlled_target(target_id)
    batch = target.sample(200, pdg_id=13, seed=11)
    raw = batch.to_raw(plane_z=28.905)
    physical_log_prob = target.log_prob(batch.physical, pdg_id=13)
    view = FeatureView(view_id)
    feature_log_prob = physical_log_prob - view.forward_log_abs_det_jacobian(raw)
    recovered = view.physical_log_prob_from_feature(feature_log_prob, raw)
    np.testing.assert_allclose(recovered, physical_log_prob, rtol=1e-10, atol=1e-10)


# --- 6. D3 falsifies a full Gaussian (nonlinear correlation) ----------------


def test_d3_is_non_gaussian_curved():
    # A banana transform makes py depend quadratically on px, so the
    # conditional mean of py given px is not linear: fit a linear model and
    # check a quadratic term explains additional variance.
    target = make_controlled_target("D3")
    batch = target.sample(20000, pdg_id=13, seed=1)
    px = batch.physical[:, 0]
    py = batch.physical[:, 1]
    # residual of py after removing best linear fit on px
    A = np.vstack([np.ones_like(px), px]).T
    lin_coef, *_ = np.linalg.lstsq(A, py, rcond=None)
    lin_resid = py - A @ lin_coef
    Aq = np.vstack([np.ones_like(px), px, px**2]).T
    quad_coef, *_ = np.linalg.lstsq(Aq, py, rcond=None)
    quad_resid = py - Aq @ quad_coef
    # quadratic fit must explain materially more variance than linear
    assert quad_resid.var() < 0.9 * lin_resid.var()


# --- 7. D5 rare-region: frequency, calibration, low contamination -----------


@pytest.mark.parametrize("variant,expected", [("rare_1e-2", 1e-2), ("rare_1e-3", 1e-3)])
def test_d5_rare_component_frequency(variant, expected):
    target = make_controlled_target("D5", variant=variant)
    n = 200000
    batch = target.sample(n, pdg_id=13, seed=123)
    rare_id = target.rare_component_id(pdg_id=13)
    frac = np.mean(batch.component_id == rare_id)
    # binomial std ~ sqrt(p/n); allow 6 sigma
    sigma = math.sqrt(expected * (1 - expected) / n)
    assert abs(frac - expected) < 6 * sigma


def test_d5_declared_regions():
    target = make_controlled_target("D5")
    assert target.declared_regions() == ("rare_tail",)


def test_d5_rare_region_mask_in_base_coordinates():
    target = make_controlled_target("D5")
    batch = target.sample(50000, pdg_id=13, seed=9)
    mask = target.region_mask(batch.physical, pdg_id=13, region_id="rare_tail")
    rare_id = target.rare_component_id(pdg_id=13)
    # Every in-region row should overwhelmingly be the rare component.
    in_region_labels = batch.component_id[mask]
    if in_region_labels.size:
        contamination = np.mean(in_region_labels != rare_id)
        assert contamination < 0.05


def test_d5_calibration_matches_versioned_manifest():
    # Recompute the calibration deterministically and check it agrees with the
    # versioned manifest constants (which training must not recompute).
    for variant in ("rare_1e-2", "rare_1e-3"):
        target = make_controlled_target("D5", variant=variant)
        manifest = target.manifest()
        assert manifest["rare_mass"] == (1e-2 if variant == "rare_1e-2" else 1e-3)
        stored = manifest["rare_region_calibration"]["per_pdg_id"]
        for pdg_id in (13, -13):
            fresh = calibrate_d5_rare_region(
                target,
                pdg_id=pdg_id,
                seed=D5_CALIBRATION_SEED,
                n_samples=D5_CALIBRATION_N,
            )
            s = stored[str(pdg_id)]
            assert fresh["target_probability_in_rare_region"] == pytest.approx(
                s["target_probability_in_rare_region"], rel=1e-9, abs=1e-9
            )
            assert fresh["main_component_contamination_in_rare_region"] == pytest.approx(
                s["main_component_contamination_in_rare_region"], abs=1e-9
            )


def test_d5_rare_region_calibration_probability_close_to_mass():
    for variant, mass in (("rare_1e-2", 1e-2), ("rare_1e-3", 1e-3)):
        target = make_controlled_target("D5", variant=variant)
        cal = target.manifest()["rare_region_calibration"]["per_pdg_id"]["13"]
        prob = cal["target_probability_in_rare_region"]
        # Region should capture most of the rare mass and little else.
        assert 0.5 * mass < prob < 1.5 * mass


# --- 8. Manifest / hash determinism -----------------------------------------


@pytest.mark.parametrize("target_id", ["D3", "D4"])
def test_manifest_hash_deterministic(target_id):
    a = make_controlled_target(target_id)
    b = make_controlled_target(target_id)
    assert a.manifest() == b.manifest()
    assert a.config_hash() == b.config_hash()


def test_d5_variants_have_distinct_hashes():
    a = make_controlled_target("D5", variant="rare_1e-2")
    b = make_controlled_target("D5", variant="rare_1e-3")
    assert a.config_hash() != b.config_hash()


def test_transformed_manifest_marks_no_physics_claim():
    for target_id in ("D3", "D4", "D5"):
        manifest = make_controlled_target(target_id).manifest()
        assert manifest["physics_claim"] is False
        assert manifest["event_level_conservation_applied"] is False
        assert manifest["exact_sample"] is True
        assert manifest["exact_log_prob"] is True


# --- 9. Import hygiene ------------------------------------------------------


def test_transform_module_imports_no_heavy_dependencies():
    code = (
        "import sys\n"
        "import ship_muon_bg.benchmarks\n"
        "banned = [m for m in ('torch', 'ROOT', 'h5py', 'scipy', 'pandas', 'sklearn')"
        " if m in sys.modules]\n"
        "assert not banned, f'heavy modules imported: {banned}'\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [REPO_ROOT, SRC, env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
