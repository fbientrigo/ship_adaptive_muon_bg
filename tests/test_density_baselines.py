"""Tests for the Gaussian baselines and the model registry.

Covers exact log_prob against reference formulas, deterministic sampling,
save/load parameter reconstruction (no pickled estimators), parameter counts,
explicit sample-weight handling, and the registry factory. The GMM test is
marked ``lab`` (needs scikit-learn).
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from Nflow.baselines import DiagonalGaussian, FullGaussian, GaussianMixtureEstimator
from Nflow.interfaces import FIT_STATUS_OK, DensityEstimator, FitResult
from Nflow.registry import (
    SUPPORTED_MODEL_FAMILIES,
    ModelRegistryError,
    create_density_estimator,
)

D = 5


def _train(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    mean = np.array([0.5, -1.0, 50.0, 0.2, -0.3])
    cov = np.diag([2.0, 3.0, 4.0, 0.5, 0.6])
    cov[0, 1] = cov[1, 0] = 0.8
    return rng.multivariate_normal(mean, cov, size=n)


def test_diagonal_gaussian_log_prob_matches_formula():
    x = _train()
    model = DiagonalGaussian(dimension=D)
    result = model.fit(x, x_validation=x[:100], seed=1)
    assert isinstance(result, FitResult)
    assert result.status == FIT_STATUS_OK
    mean = x.mean(axis=0)
    var = np.maximum(x.var(axis=0), 1e-6)
    point = x[:10]
    expected = np.sum(
        -0.5 * np.log(2 * np.pi * var) - 0.5 * (point - mean) ** 2 / var, axis=1
    )
    np.testing.assert_allclose(model.log_prob(point), expected, rtol=1e-10)


def test_full_gaussian_log_prob_matches_formula():
    x = _train()
    model = FullGaussian(dimension=D, covariance_regularization=0.0)
    model.fit(x, x_validation=None, seed=1)
    mean = x.mean(axis=0)
    cov = np.cov(x.T, bias=True)
    inv = np.linalg.inv(cov)
    sign, logdet = np.linalg.slogdet(cov)
    point = x[:10]
    diff = point - mean
    expected = -0.5 * (
        D * np.log(2 * np.pi) + logdet + np.einsum("ij,jk,ik->i", diff, inv, diff)
    )
    np.testing.assert_allclose(model.log_prob(point), expected, rtol=1e-8)


@pytest.mark.parametrize("cls", [DiagonalGaussian, FullGaussian])
def test_deterministic_sampling(cls):
    x = _train()
    model = cls(dimension=D)
    model.fit(x, x_validation=None, seed=1)
    a = model.sample(200, seed=7)
    b = model.sample(200, seed=7)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (200, D)


@pytest.mark.parametrize("cls", [DiagonalGaussian, FullGaussian])
def test_parameter_count(cls):
    model = cls(dimension=D)
    if isinstance(model, DiagonalGaussian):
        assert model.parameter_count() == 2 * D
    else:
        assert model.parameter_count() == D + D * (D + 1) // 2


@pytest.mark.parametrize("cls", [DiagonalGaussian, FullGaussian])
def test_save_load_reconstructs_from_explicit_parameters(cls, tmp_path):
    x = _train()
    model = cls(dimension=D)
    model.fit(x, x_validation=None, seed=1)
    model.save(tmp_path)
    # canonical artifact is explicit parameters, not a pickle
    assert (tmp_path / "model_parameters.npz").exists()
    assert not any(p.suffix == ".pkl" for p in tmp_path.iterdir())
    reloaded = cls.load(tmp_path)
    point = x[:50]
    np.testing.assert_allclose(reloaded.log_prob(point), model.log_prob(point), rtol=1e-12)


def test_diagonal_gaussian_variance_floor_recorded():
    x = _train()
    model = DiagonalGaussian(dimension=D, variance_floor=1e-3)
    model.fit(x, x_validation=None, seed=1)
    assert model.manifest()["variance_floor"] == 1e-3


@pytest.mark.parametrize("family", ["diagonal_gaussian", "full_gaussian"])
def test_weighted_gaussian_fit_remains_supported(family):
    x = _train()
    weights = np.where(x[:, 0] > 0, 2.0, 0.5)
    model = create_density_estimator({"family": family}, dimension=D)
    model.fit(x, x_validation=None, seed=1, sample_weight=weights)
    unweighted = create_density_estimator({"family": family}, dimension=D)
    unweighted.fit(x, x_validation=None, seed=1)
    assert not np.allclose(model._mean, unweighted._mean)


@pytest.mark.lab
def test_gmm_rejects_sample_weight():
    x = _train()
    model = GaussianMixtureEstimator(dimension=D, n_components=2)
    with pytest.raises(NotImplementedError):
        model.fit(x, x_validation=None, seed=1, sample_weight=np.ones(x.shape[0]))


@pytest.mark.lab
def test_gmm_fit_log_prob_and_save_load(tmp_path):
    x = _train()
    model = GaussianMixtureEstimator(dimension=D, n_components=2, n_init=2)
    result = model.fit(x, x_validation=x[:200], seed=3)
    assert result.status == FIT_STATUS_OK
    lp = model.log_prob(x[:100])
    assert lp.shape == (100,) and np.isfinite(lp).all()
    model.save(tmp_path)
    reloaded = GaussianMixtureEstimator.load(tmp_path)
    np.testing.assert_allclose(reloaded.log_prob(x[:100]), lp, rtol=1e-10)


# --- registry ---------------------------------------------------------------


@pytest.mark.parametrize(
    "family", ["diagonal_gaussian", "full_gaussian"]
)
def test_registry_creates_numpy_baselines(family):
    model = create_density_estimator({"family": family}, dimension=D)
    assert isinstance(model, DensityEstimator)


def test_registry_rejects_unknown_family():
    with pytest.raises(ModelRegistryError):
        create_density_estimator({"family": "nope"}, dimension=D)


def test_registry_passes_params():
    model = create_density_estimator(
        {"family": "gaussian_mixture", "params": {"n_components": 3}}, dimension=D
    )
    assert model.n_components == 3


CANONICAL_FIT_KEYWORDS = {
    "sample_weight",
    "validation_sample_weight",
    "component_id",
    "validation_component_id",
    "rare_component_id",
}


def _registry_spec(family):
    params = {}
    if family == "gaussian_mixture":
        params = {"n_components": 2, "n_init": 1, "max_iter": 20}
    elif family == "affine_coupling":
        params = {
            "number_of_blocks": 2,
            "hidden_width": 8,
            "hidden_depth": 1,
            "max_epochs": 1,
            "batch_size": 16,
        }
    return {"family": family, "params": params}


@pytest.mark.parametrize("family", SUPPORTED_MODEL_FAMILIES)
def test_registry_family_exposes_canonical_fit_keywords(family):
    model = create_density_estimator(_registry_spec(family), dimension=D)
    assert CANONICAL_FIT_KEYWORDS <= set(inspect.signature(model.fit).parameters)


@pytest.mark.parametrize("family", SUPPORTED_MODEL_FAMILIES)
def test_neutral_fit_keywords_preserve_results_configs_and_hashes(family):
    x = _train(n=64)
    validation = x[:16]
    legacy = create_density_estimator(_registry_spec(family), dimension=D)
    neutral = create_density_estimator(_registry_spec(family), dimension=D)

    legacy_result = legacy.fit(x, x_validation=validation, seed=7)
    neutral_result = neutral.fit(
        x,
        x_validation=validation,
        seed=7,
        **{name: None for name in CANONICAL_FIT_KEYWORDS},
    )

    assert neutral_result.status == legacy_result.status == FIT_STATUS_OK
    np.testing.assert_allclose(neutral.log_prob(validation), legacy.log_prob(validation))
    assert neutral.manifest() == legacy.manifest()
    if family == "affine_coupling":
        assert neutral.config() == legacy.config()
        assert neutral.checkpoint_hash() == legacy.checkpoint_hash()


UNSUPPORTED_FIT_ARGUMENTS = [
    (family, argument)
    for family, arguments in {
        "diagonal_gaussian": CANONICAL_FIT_KEYWORDS - {"sample_weight"},
        "full_gaussian": CANONICAL_FIT_KEYWORDS - {"sample_weight"},
        "gaussian_mixture": CANONICAL_FIT_KEYWORDS,
    }.items()
    for argument in sorted(arguments)
]


@pytest.mark.parametrize("family,argument", UNSUPPORTED_FIT_ARGUMENTS)
def test_unsupported_non_none_fit_arguments_raise_deliberately(family, argument):
    model = create_density_estimator(_registry_spec(family), dimension=D)
    value = 1 if argument == "rare_component_id" else np.ones(32)
    with pytest.raises(NotImplementedError, match="{}.*{}".format(family, argument)):
        model.fit(
            _train(n=32),
            x_validation=None,
            seed=7,
            **{argument: value},
        )


def test_affine_canonical_fit_arguments_remain_supported():
    x = _train(n=32)
    validation = x[:8]
    model = create_density_estimator(_registry_spec("affine_coupling"), dimension=D)
    result = model.fit(
        x,
        x_validation=validation,
        seed=7,
        sample_weight=np.linspace(1.0, 2.0, len(x)),
        validation_sample_weight=np.linspace(1.0, 2.0, len(validation)),
        component_id=np.arange(len(x)) % 2,
        validation_component_id=np.arange(len(validation)) % 2,
        rare_component_id=1,
    )
    assert result.status == FIT_STATUS_OK
    assert result.train_history[-1]["weight_normalization"] == "sum_weights"
