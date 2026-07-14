"""Tests for the exact D0-D2 controlled density targets (v0).

These are numerical benchmark distributions in canonical physical
coordinates ``[px, py, pz, x, y]``. No FairShip/ROOT/torch, no energy
variable, no event-level conservation, no proxy/utility logic.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys

import numpy as np
import pytest

from ship_muon_bg.benchmarks import (
    ControlledTarget,
    ControlledTargetConfigError,
    ControlledTargetDomainError,
    ControlledTargetShapeError,
    GaussianComponent,
    SUPPORTED_TARGET_IDS,
    embed_physical_to_raw,
    make_controlled_target,
)
from ship_muon_bg.data_contracts import (
    CARTESIAN_LOGPZ_VIEW_ID,
    IDENTITY_CARTESIAN_VIEW_ID,
    SLOPE_LOGPZ_VIEW_ID,
    FeatureView,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO_ROOT, "src")

FEATURE_VIEW_IDS = (
    IDENTITY_CARTESIAN_VIEW_ID,
    CARTESIAN_LOGPZ_VIEW_ID,
    SLOPE_LOGPZ_VIEW_ID,
)

# Predefined smoke configurations: (target_id, charge, n, seed). Frozen so
# that pz > 0 for every drawn row is a checked invariant, not a hope.
SMOKE_CONFIGS = (
    ("D0", 13, 4096, 11),
    ("D0", -13, 4096, 11),
    ("D1", 13, 4096, 7),
    ("D1", -13, 4096, 7),
    ("D2", 13, 4096, 3),
    ("D2", -13, 4096, 4),
)


# --- 1. Factory -------------------------------------------------------------


def test_supported_target_ids_are_exactly_d0_d1_d2():
    assert SUPPORTED_TARGET_IDS == ("D0", "D1", "D2")


@pytest.mark.parametrize("target_id", SUPPORTED_TARGET_IDS)
def test_factory_accepts_known_targets(target_id):
    target = make_controlled_target(target_id)
    assert target.target_id == target_id


@pytest.mark.parametrize("bad_id", ["", "d0", "D3", "unknown_v0"])
def test_factory_rejects_unknown_targets(bad_id):
    with pytest.raises(ControlledTargetConfigError):
        make_controlled_target(bad_id)


# --- 2. Invalid inputs fail explicitly --------------------------------------


@pytest.mark.parametrize("bad_n", [0, -1, 1.5, "4096"])
def test_sample_rejects_invalid_n(bad_n):
    target = make_controlled_target("D0")
    with pytest.raises(ControlledTargetConfigError):
        target.sample(n=bad_n, charge=13, seed=1)


@pytest.mark.parametrize("bad_charge", [0, 11, -11, 13.0, "13"])
def test_sample_rejects_invalid_charge(bad_charge):
    target = make_controlled_target("D0")
    with pytest.raises(ControlledTargetConfigError):
        target.sample(n=8, charge=bad_charge, seed=1)


@pytest.mark.parametrize("bad_seed", [-1, 1.5, "1"])
def test_sample_rejects_invalid_seed(bad_seed):
    target = make_controlled_target("D0")
    with pytest.raises(ControlledTargetConfigError):
        target.sample(n=8, charge=13, seed=bad_seed)


def test_log_prob_rejects_bad_shape():
    target = make_controlled_target("D0")
    with pytest.raises(ControlledTargetShapeError):
        target.log_prob(np.zeros((4, 4)), charge=13)


def test_log_prob_rejects_non_finite_input():
    target = make_controlled_target("D0")
    physical = np.zeros((4, 5))
    physical[0, 0] = np.nan
    with pytest.raises(ControlledTargetDomainError):
        target.log_prob(physical, charge=13)


def test_log_prob_rejects_invalid_charge():
    target = make_controlled_target("D0")
    with pytest.raises(ControlledTargetConfigError):
        target.log_prob(np.zeros((4, 5)), charge=7)


def test_gaussian_component_rejects_bad_mean_shape():
    with pytest.raises(ControlledTargetShapeError):
        GaussianComponent(mean=np.zeros(4), covariance=np.eye(5))


def test_gaussian_component_rejects_bad_covariance_shape():
    with pytest.raises(ControlledTargetShapeError):
        GaussianComponent(mean=np.zeros(5), covariance=np.eye(4))


def test_gaussian_component_rejects_asymmetric_covariance():
    bad_cov = np.eye(5)
    bad_cov[0, 1] = 5.0
    with pytest.raises(ControlledTargetConfigError):
        GaussianComponent(mean=np.zeros(5), covariance=bad_cov)


def test_gaussian_component_rejects_non_positive_definite_covariance():
    bad_cov = np.diag([1.0, 1.0, 1.0, 1.0, -1.0])
    with pytest.raises(ControlledTargetConfigError):
        GaussianComponent(mean=np.zeros(5), covariance=bad_cov)


def test_gaussian_component_rejects_non_finite_parameters():
    bad_mean = np.array([0.0, 0.0, np.nan, 0.0, 0.0])
    with pytest.raises(ControlledTargetConfigError):
        GaussianComponent(mean=bad_mean, covariance=np.eye(5))


def test_gaussian_component_rejects_non_positive_weight():
    with pytest.raises(ControlledTargetConfigError):
        GaussianComponent(mean=np.zeros(5), covariance=np.eye(5), weight=0.0)


# --- 2b. GaussianComponent configuration is genuinely immutable ------------


def test_gaussian_component_copies_mean_and_covariance_at_construction():
    mean_source = np.array([0.0, 0.0, 50.0, 0.0, 0.0])
    covariance_source = np.eye(5) * 4.0
    component = GaussianComponent(mean=mean_source, covariance=covariance_source)

    original_mean = component.mean.copy()
    original_covariance = component.covariance.copy()

    # Mutating the caller's source arrays after construction must not be
    # visible through the component: construction must copy, not alias.
    mean_source[0] = 999.0
    covariance_source[0, 0] = 999.0

    np.testing.assert_array_equal(component.mean, original_mean)
    np.testing.assert_array_equal(component.covariance, original_covariance)


def test_gaussian_component_mean_and_covariance_are_read_only():
    component = GaussianComponent(mean=np.zeros(5), covariance=np.eye(5))

    assert component.mean.flags["WRITEABLE"] is False
    assert component.covariance.flags["WRITEABLE"] is False

    with pytest.raises(ValueError):
        component.mean[0] = 1.0
    with pytest.raises(ValueError):
        component.covariance[0, 0] = 1.0


def test_gaussian_component_rebinding_mean_attribute_fails():
    component = GaussianComponent(mean=np.zeros(5), covariance=np.eye(5))
    with pytest.raises(Exception):
        component.mean = np.ones(5)


def test_gaussian_component_immutability_preserves_log_prob_sample_manifest_and_hash():
    mean = np.array([0.0, 0.0, 50.0, 0.0, 0.0])
    covariance = np.eye(5) * 4.0
    component = GaussianComponent(mean=mean.copy(), covariance=covariance.copy())

    target = ControlledTarget(
        target_id="IMMUTABILITY_TEST",
        description="t",
        pdg_id_parameterization="shared_across_pdg_ids",
        components_by_pdg_id={13: (component,), -13: (component,)},
    )
    manifest_before = target.manifest()
    hash_before = target.config_hash()
    points = np.array([[1.0, 2.0, 50.0, 0.3, -0.2]])
    log_prob_before = target.log_prob(points, pdg_id=13)
    sample_before = target.sample(n=8, pdg_id=13, seed=5).physical

    # Mutate the original source arrays (already copied at construction) --
    # none of the target's derived state may change as a result.
    mean[0] = 12345.0
    covariance[0, 0] = 999.0

    assert target.manifest() == manifest_before
    assert target.config_hash() == hash_before
    np.testing.assert_array_equal(target.log_prob(points, pdg_id=13), log_prob_before)
    np.testing.assert_array_equal(
        target.sample(n=8, pdg_id=13, seed=5).physical, sample_before
    )


def test_controlled_target_rejects_mixture_weights_not_summing_to_one():
    good = GaussianComponent(
        mean=np.array([0.0, 0.0, 50.0, 0.0, 0.0]), covariance=np.eye(5), weight=0.5
    )
    other = GaussianComponent(
        mean=np.array([1.0, 0.0, 50.0, 0.0, 0.0]), covariance=np.eye(5), weight=0.6
    )
    with pytest.raises(ControlledTargetConfigError):
        ControlledTarget(
            target_id="BAD_WEIGHTS",
            description="t",
            charge_parameterization="shared_across_charges",
            components_by_charge={13: (good, other), -13: (good, other)},
        )


def test_controlled_target_rejects_insufficient_pz_margin():
    low_margin = GaussianComponent(
        mean=np.array([0.0, 0.0, 1.0, 0.0, 0.0]), covariance=np.eye(5)
    )
    with pytest.raises(ControlledTargetConfigError):
        ControlledTarget(
            target_id="BAD_MARGIN",
            description="t",
            charge_parameterization="shared_across_charges",
            components_by_charge={13: (low_margin,), -13: (low_margin,)},
        )


# --- 3, 4. Determinism -------------------------------------------------------


def test_fixed_seed_is_bitwise_deterministic():
    target = make_controlled_target("D0")
    batch1 = target.sample(n=256, charge=13, seed=11)
    batch2 = target.sample(n=256, charge=13, seed=11)
    np.testing.assert_array_equal(batch1.physical, batch2.physical)
    np.testing.assert_array_equal(batch1.component_id, batch2.component_id)


def test_different_seeds_produce_different_samples():
    target = make_controlled_target("D0")
    batch1 = target.sample(n=256, charge=13, seed=11)
    batch2 = target.sample(n=256, charge=13, seed=12)
    assert not np.array_equal(batch1.physical, batch2.physical)


# --- 5. Smoke shape/dtype/finiteness/contiguity/pz domain -------------------


@pytest.mark.parametrize("target_id,charge,n,seed", SMOKE_CONFIGS)
def test_smoke_sample_shape_dtype_and_domain(target_id, charge, n, seed):
    target = make_controlled_target(target_id)
    batch = target.sample(n=n, charge=charge, seed=seed)

    assert batch.physical.shape == (n, 5)
    assert batch.physical.dtype == np.float64
    assert batch.physical.flags["C_CONTIGUOUS"]
    assert np.isfinite(batch.physical).all()
    assert batch.component_id.shape == (n,)
    assert np.issubdtype(batch.component_id.dtype, np.integer)
    assert batch.charge == charge
    assert batch.target_id == target_id
    assert batch.seed == seed

    if np.any(batch.physical[:, 2] <= 0.0):
        raise AssertionError(
            "frozen smoke config {} produced pz <= 0 rows".format(
                (target_id, charge, n, seed)
            )
        )


# --- 6, 7. Exact log density against independent formulas -------------------


def test_d0_log_density_matches_manual_diagonal_formula():
    target = make_controlled_target("D0")
    mean = np.array([0.0, 0.0, 50.0, 0.0, 0.0])
    std = np.array([3.0, 3.0, 4.0, 0.5, 0.5])
    rng = np.random.default_rng(99)
    physical = mean + rng.standard_normal((32, 5)) * std

    expected = np.sum(
        -0.5 * np.log(2 * np.pi * std**2) - 0.5 * ((physical - mean) / std) ** 2,
        axis=1,
    )
    actual = target.log_prob(physical, charge=13)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_d1_log_density_matches_manual_multivariate_formula():
    target = make_controlled_target("D1")
    manifest = target.manifest()
    mean = np.array(manifest["means"]["-13"][0])
    cov = np.array(manifest["covariance_matrices"]["-13"][0])
    inv_cov = np.linalg.inv(cov)
    sign, logdet = np.linalg.slogdet(cov)
    assert sign > 0.0

    points = np.array(
        [
            [0.0, 0.0, 50.0, 0.0, 0.0],
            [3.0, -2.0, 55.0, 0.4, -0.1],
            [-4.0, 1.5, 45.0, -0.3, 0.2],
        ]
    )
    diff = points - mean
    expected = -0.5 * (
        5 * np.log(2 * np.pi) + logdet + np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
    )
    actual = target.log_prob(points, charge=-13)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


# --- 8. D1 covariance is SPD and genuinely correlated -----------------------


def test_d1_covariance_is_symmetric_positive_definite_and_correlated():
    target = make_controlled_target("D1")
    cov = np.array(target.manifest()["covariance_matrices"]["13"][0])
    np.testing.assert_allclose(cov, cov.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    assert np.all(eigenvalues > 0.0)
    off_diagonal = cov - np.diag(np.diag(cov))
    assert np.any(np.abs(off_diagonal) > 1e-8)


# --- 9, 10. D2 component labels and empirical mixture fractions -------------


def test_d2_component_labels_are_valid():
    target = make_controlled_target("D2")
    batch = target.sample(n=512, charge=13, seed=3)
    assert set(np.unique(batch.component_id)).issubset({0, 1})


@pytest.mark.parametrize("charge", [13, -13])
def test_d2_component_fractions_match_declared_weights(charge):
    target = make_controlled_target("D2")
    n = 20000
    batch = target.sample(n=n, charge=charge, seed=3)
    weights = np.array(target.manifest()["mixture_weights"][str(charge)])
    fractions = np.array(
        [np.mean(batch.component_id == k) for k in range(len(weights))]
    )
    # Binomial std at n=20000, worst-case p=0.5 is ~0.0035; 0.02 is a >5-sigma
    # margin above that, chosen ahead of time (not tuned post hoc).
    np.testing.assert_allclose(fractions, weights, atol=0.02)


# --- 11. D2 log density matches manual log-sum-exp --------------------------


def test_d2_log_density_matches_manual_logsumexp():
    target = make_controlled_target("D2")
    manifest = target.manifest()
    means = [np.array(m) for m in manifest["means"]["13"]]
    covs = [np.array(c) for c in manifest["covariance_matrices"]["13"]]
    weights = np.array(manifest["mixture_weights"]["13"])

    points = np.array(
        [
            [2.0, 1.0, 45.0, 0.3, 0.2],
            [-2.0, -1.0, 60.0, -0.3, -0.2],
            [0.0, 0.0, 50.0, 0.0, 0.0],
        ]
    )

    log_terms = np.empty((points.shape[0], len(means)))
    for k, (mean, cov) in enumerate(zip(means, covs)):
        inv_cov = np.linalg.inv(cov)
        sign, logdet = np.linalg.slogdet(cov)
        assert sign > 0.0
        diff = points - mean
        log_terms[:, k] = np.log(weights[k]) - 0.5 * (
            5 * np.log(2 * np.pi)
            + logdet
            + np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
        )
    max_term = np.max(log_terms, axis=1, keepdims=True)
    expected = (
        max_term + np.log(np.sum(np.exp(log_terms - max_term), axis=1, keepdims=True))
    ).squeeze(1)

    actual = target.log_prob(points, charge=13)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


# --- 12. Charges are observably distinct ------------------------------------


def test_d2_charge_distributions_are_observably_distinct():
    target = make_controlled_target("D2")
    batch_pos = target.sample(n=2048, charge=13, seed=3)
    batch_neg = target.sample(n=2048, charge=-13, seed=3)
    assert not np.allclose(
        batch_pos.physical.mean(axis=0), batch_neg.physical.mean(axis=0), atol=0.2
    )


# --- 13, 14. Manifest and config hash determinism ---------------------------


@pytest.mark.parametrize("target_id", SUPPORTED_TARGET_IDS)
def test_manifest_and_hash_are_deterministic(target_id):
    target_a = make_controlled_target(target_id)
    target_b = make_controlled_target(target_id)
    assert target_a.manifest() == target_b.manifest()
    assert target_a.config_hash() == target_b.config_hash()

    manifest = target_a.manifest()
    required_keys = {
        "target_schema_version",
        "target_id",
        "target_description",
        "density_coordinate",
        "physical_columns",
        "supported_charges",
        "charge_parameterization",
        "component_count_by_charge",
        "mixture_weights",
        "means",
        "covariance_matrices",
        "probability_pz_nonpositive",
        "exact_sample",
        "exact_log_prob",
        "physics_claim",
        "event_level_conservation_applied",
    }
    assert required_keys.issubset(manifest.keys())
    assert manifest["density_coordinate"] == "physical_px_py_pz_x_y"
    assert manifest["physical_columns"] == ["px", "py", "pz", "x", "y"]
    assert manifest["exact_sample"] is True
    assert manifest["exact_log_prob"] is True
    assert manifest["physics_claim"] is False
    assert manifest["event_level_conservation_applied"] is False


@pytest.mark.parametrize("target_id", SUPPORTED_TARGET_IDS)
def test_probability_pz_nonpositive_is_positive_finite_and_negligible(target_id):
    # Regression guard: the previous ``0.5 * (1 + erf(-z))`` formula
    # catastrophically cancelled at these margins and silently reported
    # exactly 0.0, which a merely "< 1e-15" assertion could not distinguish
    # from a correct tiny nonzero tail mass. Require strictly positive and
    # finite in addition to "negligible" so a regression to 0.0 fails loudly.
    manifest = make_controlled_target(target_id).manifest()
    for info in manifest["probability_pz_nonpositive"].values():
        assert math.isfinite(info["total"])
        assert info["total"] > 0.0
        assert info["total"] < 1e-20
        for component_probability in info["components"]:
            assert math.isfinite(component_probability)
            assert component_probability > 0.0
            assert component_probability < 1e-20


def _independent_erfc(z: float) -> float:
    """Independent reference for ``erfc(z)`` at large ``z`` (z >= ~9).

    Uses the standard asymptotic expansion
    ``erfc(z) ~ exp(-z^2)/(z*sqrt(pi)) * sum_k (-1)^k (2k-1)!! / (2z^2)^k``,
    a different algorithm from ``math.erfc`` (no libm call), truncated after
    5 terms. At the z values exercised below (>= 10/sqrt(2)) this series
    agrees with ``math.erfc`` to better than 1e-8 relative accuracy, which is
    enough to catch a catastrophic-cancellation regression (which would be
    wrong by 100%, i.e. exactly 0.0) while tolerating the series' own
    truncation error.
    """

    series_sum = 1.0
    term = 1.0
    for k in range(1, 6):
        term *= -(2 * k - 1) / (2.0 * z * z)
        series_sum += term
    return math.exp(-z * z) / (z * math.sqrt(math.pi)) * series_sum


@pytest.mark.parametrize("ratio", [10.0, 11.25, 12.0, 12.5])
def test_gaussian_tail_probability_matches_independent_erfc_reference(ratio):
    std = 1.0
    mean = np.array([ratio * std, 0.0, 50.0, 0.0, 0.0])
    component = GaussianComponent(mean=mean, covariance=np.eye(5))

    probability = component.probability_variable_nonpositive(0)
    assert math.isfinite(probability)
    assert probability > 0.0

    z = ratio / math.sqrt(2.0)
    independent_reference = 0.5 * _independent_erfc(z)
    assert probability == pytest.approx(independent_reference, rel=1e-6)

    # Also cross-check directly against the standard-library erfc, which is
    # the corrected formula this module now uses internally.
    stdlib_reference = 0.5 * math.erfc(z)
    assert probability == pytest.approx(stdlib_reference, rel=1e-12)


def test_config_hash_changes_when_parameters_differ():
    base = GaussianComponent(
        mean=np.array([0.0, 0.0, 50.0, 0.0, 0.0]), covariance=np.eye(5)
    )
    changed = GaussianComponent(
        mean=np.array([0.0, 0.0, 50.0, 0.1, 0.0]), covariance=np.eye(5)
    )

    target_a = ControlledTarget(
        target_id="HASH_TEST",
        description="t",
        charge_parameterization="shared_across_charges",
        components_by_charge={13: (base,), -13: (base,)},
    )
    target_b = ControlledTarget(
        target_id="HASH_TEST",
        description="t",
        charge_parameterization="shared_across_charges",
        components_by_charge={13: (changed,), -13: (changed,)},
    )
    assert target_a.config_hash() != target_b.config_hash()


# --- 15. Raw embedding -------------------------------------------------------


def test_embed_physical_to_raw_preserves_values_and_sets_metadata():
    physical = np.array(
        [[1.0, 2.0, 50.0, 0.5, -0.5], [2.0, -1.0, 60.0, 0.1, 0.2]]
    )
    original = physical.copy()
    raw = embed_physical_to_raw(physical, charge=13, plane_z=28.905)

    np.testing.assert_array_equal(raw[:, 0:5], physical)
    np.testing.assert_array_equal(raw[:, 5], np.full(2, 28.905))
    np.testing.assert_array_equal(raw[:, 6], np.full(2, 13.0))
    np.testing.assert_array_equal(raw[:, 7], np.full(2, 1.0))
    np.testing.assert_array_equal(physical, original)


def test_embed_physical_to_raw_rejects_invalid_charge():
    physical = np.zeros((2, 5))
    with pytest.raises(ControlledTargetConfigError):
        embed_physical_to_raw(physical, charge=7, plane_z=0.0)


def test_embed_physical_to_raw_rejects_bad_shape():
    with pytest.raises(ControlledTargetShapeError):
        embed_physical_to_raw(np.zeros((3, 4)), charge=13, plane_z=0.0)


def test_embed_physical_to_raw_rejects_non_finite_input():
    physical = np.zeros((2, 5))
    physical[0, 0] = np.inf
    with pytest.raises(ControlledTargetDomainError):
        embed_physical_to_raw(physical, charge=13, plane_z=0.0)


def test_sample_batch_to_raw_matches_helper():
    target = make_controlled_target("D0")
    batch = target.sample(n=16, charge=13, seed=11)
    np.testing.assert_array_equal(
        batch.to_raw(plane_z=1.5),
        embed_physical_to_raw(batch.physical, charge=13, plane_z=1.5),
    )


# --- 16, 17. Feature-view compatibility and Jacobian accounting -------------


@pytest.mark.parametrize("view_id", FEATURE_VIEW_IDS)
def test_feature_view_arms_recover_physical_log_prob(view_id):
    target = make_controlled_target("D0")
    batch = target.sample(n=256, charge=13, seed=11)
    raw_rows = batch.to_raw(plane_z=28.905)
    physical_log_prob = target.log_prob(batch.physical, charge=13)

    view = FeatureView(view_id)
    features = view.forward(raw_rows)
    assert features.shape == (256, 5)

    feature_log_prob = physical_log_prob - view.forward_log_abs_det_jacobian(raw_rows)
    recovered = view.physical_log_prob_from_feature(feature_log_prob, raw_rows)

    np.testing.assert_allclose(recovered, physical_log_prob, rtol=1e-10, atol=1e-10)


# --- 18. Import hygiene ------------------------------------------------------


def test_benchmarks_module_imports_no_heavy_dependencies():
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
