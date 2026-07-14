"""Explicit density-estimator registry / factory.

``create_density_estimator(model_spec, *, dimension, device)`` maps a model
family name to a concrete :class:`~Nflow.interfaces.DensityEstimator`. The
registry is a plain dict of factory callables -- no decorators, no plugin
discovery, no import scanning, no entry points. Torch is imported lazily and
only when the ``affine_coupling`` family is requested, so importing
``Nflow`` / ``Nflow.registry`` stays NumPy-only.

Adding a future family (e.g. a spline flow) is: one implementation, one entry
in ``_FACTORIES`` below, tests, and a config. The evaluator and campaign
runner do not change.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

SUPPORTED_MODEL_FAMILIES = (
    "diagonal_gaussian",
    "full_gaussian",
    "gaussian_mixture",
    "affine_coupling",
)


class ModelRegistryError(ValueError):
    """Unknown or misconfigured model family."""


def _normalize_spec(model_spec: Any) -> Dict[str, Any]:
    """Accept a mapping or an object exposing ``family``/``params``."""

    if isinstance(model_spec, Mapping):
        family = model_spec.get("family")
        params = model_spec.get("params", {})
    else:
        family = getattr(model_spec, "family", None)
        params = getattr(model_spec, "params", {})
    if not isinstance(family, str):
        raise ModelRegistryError(
            "model_spec must provide a string 'family', got {!r}".format(family)
        )
    return {"family": family, "params": dict(params or {})}


def _make_diagonal_gaussian(*, dimension, device, params):
    from Nflow.baselines.gaussian import DiagonalGaussian

    return DiagonalGaussian(dimension=dimension, **params)


def _make_full_gaussian(*, dimension, device, params):
    from Nflow.baselines.gaussian import FullGaussian

    return FullGaussian(dimension=dimension, **params)


def _make_gaussian_mixture(*, dimension, device, params):
    from Nflow.baselines.gmm import GaussianMixtureEstimator

    return GaussianMixtureEstimator(dimension=dimension, **params)


def _make_affine_coupling(*, dimension, device, params):
    # Lazy: torch is only imported here, when the flow is actually requested.
    from Nflow.torch_models.affine_coupling import AffineCouplingFlow

    return AffineCouplingFlow(dimension=dimension, device=device, **params)


_FACTORIES = {
    "diagonal_gaussian": _make_diagonal_gaussian,
    "full_gaussian": _make_full_gaussian,
    "gaussian_mixture": _make_gaussian_mixture,
    "affine_coupling": _make_affine_coupling,
}


def create_density_estimator(model_spec: Any, *, dimension: int, device: str = "cpu"):
    """Construct the estimator named by ``model_spec['family']``.

    ``model_spec`` is a mapping (or object) with a string ``family`` and an
    optional ``params`` mapping. ``dimension`` is the modelled feature
    dimension; ``device`` is passed through to torch models and ignored by the
    NumPy baselines.
    """

    spec = _normalize_spec(model_spec)
    family = spec["family"]
    try:
        factory = _FACTORIES[family]
    except KeyError as exc:
        raise ModelRegistryError(
            "unknown model family {!r}; expected one of {}".format(
                family, SUPPORTED_MODEL_FAMILIES
            )
        ) from exc
    return factory(dimension=dimension, device=device, params=spec["params"])
