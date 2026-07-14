"""Exact D0-D2 controlled density targets for the nominal density track (v0).

These are numerical benchmark distributions defined directly in canonical
physical coordinates ``[px, py, pz, x, y]``. They make no claim to reproduce
SHiP physics: there is no energy variable, no event-level momentum or energy
conservation, no DIS product modelling, and no dependency on FairShip, ROOT,
GEANT4, proxy labels, or utility tilting. Event-level conservation belongs to
a downstream ``simulation_backend``, not to this benchmark.

Each target exposes exact ``sample`` and ``log_prob`` in physical coordinates,
plus a versioned, hashable manifest. See
``docs/contracts/controlled_targets_v0.md`` for the normative definitions and
``docs/contracts/density_problem_contract_v0.md`` section 7 for the D0-D2
curriculum this module implements.

Targets are conditioned on ``pdg_id``, a PDG particle id -- not an electric
charge value: ``pdg_id = 13`` is mu- and ``pdg_id = -13`` is mu+.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from ..data_contracts import schema
from ..data_contracts.feature_views import N_DENSITY_FEATURES, PHYSICAL_STATE_COLUMNS

TARGET_SCHEMA_VERSION = "0"
SUPPORTED_TARGET_IDS: Tuple[str, ...] = ("D0", "D1", "D2")

# PDG IDs, not electric charges: 13 = mu-, -13 = mu+.
SUPPORTED_PDG_IDS: Tuple[int, ...] = (13, -13)

N_PHYSICAL_DIMS = N_DENSITY_FEATURES
PHYSICAL_COLUMNS = PHYSICAL_STATE_COLUMNS
_PZ_INDEX = PHYSICAL_COLUMNS.index("pz")

# Every Gaussian component must satisfy mean_pz / marginal_std_pz >= this
# ratio, so that pz <= 0 has negligible probability under the exact,
# untruncated density (no clipping/rejection/epsilon is ever applied).
_MIN_PZ_SIGMA_RATIO = 10.0

_LOG_TWO_PI = math.log(2.0 * math.pi)


class ControlledTargetError(ValueError):
    """Base class for controlled-target failures."""


class ControlledTargetConfigError(ControlledTargetError):
    """A target/component identifier or configuration value is unsupported."""


class ControlledTargetShapeError(ControlledTargetError):
    """An array does not have the required controlled-target shape."""


class ControlledTargetDomainError(ControlledTargetError):
    """An array lies outside the required finite/positive domain."""


def _canonical_json_hash(payload: Dict[str, Any]) -> str:
    """Return a stable SHA-256 hash for a JSON-serializable mapping."""

    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _logsumexp(log_terms: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable ``log(sum(exp(log_terms)))`` along ``axis``."""

    max_term = np.max(log_terms, axis=axis, keepdims=True)
    shifted = log_terms - max_term
    summed = np.sum(np.exp(shifted), axis=axis, keepdims=True)
    result = max_term + np.log(summed)
    return np.squeeze(result, axis=axis)


def _gaussian_cdf_at_zero(mean: float, std: float) -> float:
    """Return ``P(X <= 0)`` for ``X ~ Normal(mean, std**2)``.

    Uses ``erfc`` rather than ``1 + erf(-z)``: for the large positive ``z``
    values this module operates at (``mean / std`` in the 10-12.5 range),
    ``erf(-z)`` is a value extremely close to ``-1``, so ``1.0 + erf(-z)``
    catastrophically cancels and rounds to exactly ``0.0`` in float64.
    ``erfc(z) = 1 - erf(z)`` is evaluated directly by the C library without
    that cancellation and returns the correct, tiny, nonzero tail mass.
    """

    return 0.5 * math.erfc(mean / (std * math.sqrt(2.0)))


def _assert_pz_margin(mean_pz: float, std_pz: float, label: str) -> None:
    ratio = mean_pz / std_pz
    if not (ratio >= _MIN_PZ_SIGMA_RATIO):
        raise ControlledTargetConfigError(
            "{}: mean_pz / marginal_std_pz = {:.3f} violates the required "
            "minimum margin of {}".format(label, ratio, _MIN_PZ_SIGMA_RATIO)
        )


def _validate_n(n: Any) -> int:
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        raise ControlledTargetConfigError(
            "n must be a positive Python/NumPy integer, got {!r}".format(n)
        )
    if n < 1:
        raise ControlledTargetConfigError("n must be >= 1, got {}".format(n))
    return int(n)


def _validate_pdg_id(pdg_id: Any) -> int:
    if isinstance(pdg_id, bool) or not isinstance(pdg_id, (int, np.integer)):
        raise ControlledTargetConfigError(
            "pdg_id must be an integer PDG id, got {!r}".format(pdg_id)
        )
    pdg_id = int(pdg_id)
    if pdg_id not in SUPPORTED_PDG_IDS:
        raise ControlledTargetConfigError(
            "pdg_id must be one of {}, got {}".format(SUPPORTED_PDG_IDS, pdg_id)
        )
    return pdg_id


def _validate_seed(seed: Any) -> int:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ControlledTargetConfigError(
            "seed must be an explicit non-negative integer, got {!r}".format(seed)
        )
    if seed < 0:
        raise ControlledTargetConfigError("seed must be >= 0, got {}".format(seed))
    return int(seed)


def _validate_physical_array(physical: Any) -> np.ndarray:
    try:
        array = np.asarray(physical, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ControlledTargetShapeError(
            "physical is not coercible to float64: {}".format(exc)
        ) from exc
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] != N_PHYSICAL_DIMS:
        raise ControlledTargetShapeError(
            "expected physical shape (N, {}) with N >= 1, got {}".format(
                N_PHYSICAL_DIMS, array.shape
            )
        )
    if not np.isfinite(array).all():
        raise ControlledTargetDomainError("physical coordinates contain NaN or inf")
    return array


@dataclass(frozen=True)
class GaussianComponent:
    """One exact Gaussian component in canonical physical coordinates.

    ``covariance`` is validated (and its Cholesky factor cached) at
    construction time so every downstream ``log_prob``/``sample`` call uses
    stable linear algebra rather than an explicitly formed inverse.

    ``@dataclass(frozen=True)`` only prevents rebinding the ``mean`` and
    ``covariance`` attributes to a new object; it does not freeze the NumPy
    buffers those attributes point at, and ``np.ascontiguousarray`` returns
    its input unchanged (aliased, not copied) whenever that input is already
    C-contiguous. So construction here makes an explicit copy of the
    caller-supplied arrays before validating and storing them, and marks the
    stored ``mean``, ``covariance`` and cached Cholesky arrays read-only via
    ``flags.writeable = False``. This guarantees a component's configuration
    cannot change after construction, whether by later mutating the array the
    caller originally passed in or by writing directly into
    ``component.mean``/``component.covariance``.
    """

    mean: np.ndarray
    covariance: np.ndarray
    weight: float = 1.0

    def __post_init__(self) -> None:
        mean = np.array(self.mean, dtype=np.float64, copy=True, order="C")
        covariance = np.array(self.covariance, dtype=np.float64, copy=True, order="C")

        if mean.shape != (N_PHYSICAL_DIMS,):
            raise ControlledTargetShapeError(
                "component mean must have shape ({},), got {}".format(
                    N_PHYSICAL_DIMS, mean.shape
                )
            )
        if covariance.shape != (N_PHYSICAL_DIMS, N_PHYSICAL_DIMS):
            raise ControlledTargetShapeError(
                "component covariance must have shape ({0}, {0}), got {1}".format(
                    N_PHYSICAL_DIMS, covariance.shape
                )
            )
        if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
            raise ControlledTargetConfigError(
                "component mean and covariance must be finite"
            )
        if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-10):
            raise ControlledTargetConfigError("component covariance must be symmetric")
        try:
            cholesky = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            raise ControlledTargetConfigError(
                "component covariance must be positive definite"
            ) from exc

        weight = float(self.weight)
        if not math.isfinite(weight) or weight <= 0.0:
            raise ControlledTargetConfigError(
                "component weight must be finite and positive, got {}".format(weight)
            )

        mean.flags.writeable = False
        covariance.flags.writeable = False
        cholesky = np.ascontiguousarray(cholesky)
        cholesky.flags.writeable = False

        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "_cholesky", cholesky)
        log_det_covariance = 2.0 * float(np.sum(np.log(np.diag(cholesky))))
        object.__setattr__(self, "_log_det_covariance", log_det_covariance)

    def log_prob(self, physical: np.ndarray) -> np.ndarray:
        """Return the exact log density at each row of ``physical``."""

        diff = physical - self.mean
        whitened = np.linalg.solve(self._cholesky, diff.T)
        mahalanobis = np.sum(whitened * whitened, axis=0)
        return (
            -0.5 * N_PHYSICAL_DIMS * _LOG_TWO_PI
            - 0.5 * self._log_det_covariance
            - 0.5 * mahalanobis
        )

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` exact samples using ``rng``."""

        eps = rng.standard_normal((n, N_PHYSICAL_DIMS))
        return self.mean + eps @ self._cholesky.T

    def marginal_std(self, index: int) -> float:
        return float(math.sqrt(self.covariance[index, index]))

    def probability_variable_nonpositive(self, index: int) -> float:
        return _gaussian_cdf_at_zero(float(self.mean[index]), self.marginal_std(index))


@dataclass(frozen=True)
class SampleBatch:
    """One deterministic draw from a :class:`ControlledTarget`."""

    physical: np.ndarray
    component_id: np.ndarray
    pdg_id: int
    target_id: str
    seed: int

    def to_raw(self, *, plane_z: float = 0.0) -> np.ndarray:
        """Embed ``physical`` into the existing raw 8-column schema."""

        return embed_physical_to_raw(self.physical, pdg_id=self.pdg_id, plane_z=plane_z)


class ControlledTarget:
    """A PDG-id-conditioned exact Gaussian/Gaussian-mixture density target."""

    def __init__(
        self,
        *,
        target_id: str,
        description: str,
        pdg_id_parameterization: str,
        components_by_pdg_id: Mapping[int, Sequence[GaussianComponent]],
    ) -> None:
        if set(components_by_pdg_id.keys()) != set(SUPPORTED_PDG_IDS):
            raise ControlledTargetConfigError(
                "components_by_pdg_id must declare exactly the supported PDG "
                "ids {}, got {}".format(
                    sorted(SUPPORTED_PDG_IDS), sorted(components_by_pdg_id.keys())
                )
            )

        normalized: Dict[int, Tuple[GaussianComponent, ...]] = {}
        for pdg_id, components in components_by_pdg_id.items():
            components = tuple(components)
            if not components:
                raise ControlledTargetConfigError(
                    "pdg_id {} must declare at least one component".format(pdg_id)
                )
            total_weight = sum(component.weight for component in components)
            if abs(total_weight - 1.0) > 1e-9:
                raise ControlledTargetConfigError(
                    "mixture weights for pdg_id {} must sum to 1.0, got {}".format(
                        pdg_id, total_weight
                    )
                )
            for component in components:
                _assert_pz_margin(
                    float(component.mean[_PZ_INDEX]),
                    component.marginal_std(_PZ_INDEX),
                    "{} pdg_id {}".format(target_id, pdg_id),
                )
            normalized[int(pdg_id)] = components

        self.target_id = target_id
        self.description = description
        self.pdg_id_parameterization = pdg_id_parameterization
        self._components_by_pdg_id = normalized

    def _components_for(self, pdg_id: int) -> Tuple[GaussianComponent, ...]:
        try:
            return self._components_by_pdg_id[pdg_id]
        except KeyError as exc:
            raise ControlledTargetConfigError(
                "target {} has no configuration for pdg_id {}".format(
                    self.target_id, pdg_id
                )
            ) from exc

    def sample(self, n: int, *, pdg_id: int, seed: int) -> SampleBatch:
        n = _validate_n(n)
        pdg_id = _validate_pdg_id(pdg_id)
        seed = _validate_seed(seed)
        components = self._components_for(pdg_id)

        rng = np.random.default_rng(seed)
        n_components = len(components)
        if n_components > 1:
            weights = np.array([component.weight for component in components])
            labels = rng.choice(n_components, size=n, p=weights)
        else:
            labels = np.zeros(n, dtype=np.int64)

        eps = rng.standard_normal((n, N_PHYSICAL_DIMS))
        physical = np.empty((n, N_PHYSICAL_DIMS), dtype=np.float64)
        for index, component in enumerate(components):
            mask = labels == index
            if np.any(mask):
                physical[mask] = component.mean + eps[mask] @ component._cholesky.T

        physical = np.ascontiguousarray(physical, dtype=np.float64)
        component_id = np.ascontiguousarray(labels.astype(np.int64))
        return SampleBatch(
            physical=physical,
            component_id=component_id,
            pdg_id=pdg_id,
            target_id=self.target_id,
            seed=seed,
        )

    def log_prob(self, physical: np.ndarray, *, pdg_id: int) -> np.ndarray:
        pdg_id = _validate_pdg_id(pdg_id)
        physical = _validate_physical_array(physical)
        components = self._components_for(pdg_id)

        if len(components) == 1:
            return components[0].log_prob(physical)

        log_terms = np.stack(
            [
                math.log(component.weight) + component.log_prob(physical)
                for component in components
            ],
            axis=-1,
        )
        return _logsumexp(log_terms, axis=-1)

    def manifest(self) -> Dict[str, Any]:
        means: Dict[str, Any] = {}
        covariances: Dict[str, Any] = {}
        weights: Dict[str, Any] = {}
        component_counts: Dict[str, Any] = {}
        pz_probabilities: Dict[str, Any] = {}

        for pdg_id in SUPPORTED_PDG_IDS:
            components = self._components_by_pdg_id[pdg_id]
            key = str(pdg_id)
            means[key] = [component.mean.tolist() for component in components]
            covariances[key] = [component.covariance.tolist() for component in components]
            component_weights = [float(component.weight) for component in components]
            weights[key] = component_weights
            component_counts[key] = len(components)

            per_component = [
                component.probability_variable_nonpositive(_PZ_INDEX)
                for component in components
            ]
            total = sum(w * p for w, p in zip(component_weights, per_component))
            pz_probabilities[key] = {"components": per_component, "total": total}

        return {
            "target_schema_version": TARGET_SCHEMA_VERSION,
            "target_id": self.target_id,
            "target_description": self.description,
            "density_coordinate": "physical_px_py_pz_x_y",
            "physical_columns": list(PHYSICAL_COLUMNS),
            "supported_pdg_ids": list(SUPPORTED_PDG_IDS),
            "pdg_id_parameterization": self.pdg_id_parameterization,
            "component_count_by_pdg_id": component_counts,
            "mixture_weights": weights,
            "means": means,
            "covariance_matrices": covariances,
            "probability_pz_nonpositive": pz_probabilities,
            "exact_sample": True,
            "exact_log_prob": True,
            "physics_claim": False,
            "event_level_conservation_applied": False,
        }

    def config_hash(self) -> str:
        return _canonical_json_hash(self.manifest())


def embed_physical_to_raw(
    physical: np.ndarray, *, pdg_id: int, plane_z: float = 0.0
) -> np.ndarray:
    """Embed ``[px, py, pz, x, y]`` rows into the raw ``(N, 8)`` schema.

    ``z`` is set to the requested synthetic plane metadata, ``id`` to the
    requested PDG id (``13`` = mu-, ``-13`` = mu+; PDG ids, not electric
    charge values), and ``w`` to ``1.0`` (no production-weight
    interpretation). The input array is never mutated.
    """

    physical = _validate_physical_array(physical)
    pdg_id = _validate_pdg_id(pdg_id)
    plane_z = float(plane_z)
    if not math.isfinite(plane_z):
        raise ControlledTargetDomainError("plane_z must be finite")

    n_rows = physical.shape[0]
    raw = np.empty((n_rows, schema.N_COLUMNS), dtype=np.float64)
    raw[:, 0:N_PHYSICAL_DIMS] = physical
    raw[:, schema.COLUMN_INDEX["z"]] = plane_z
    raw[:, schema.COLUMN_INDEX["id"]] = float(pdg_id)
    raw[:, schema.COLUMN_INDEX["w"]] = 1.0
    return raw


def _diag_covariance(std: Sequence[float]) -> np.ndarray:
    std_array = np.asarray(std, dtype=np.float64)
    return np.diag(std_array**2)


# --- D0: five-dimensional diagonal Gaussian ---------------------------------

_D0_V0_MEAN: Tuple[float, ...] = (0.0, 0.0, 50.0, 0.0, 0.0)
_D0_V0_STD: Tuple[float, ...] = (3.0, 3.0, 4.0, 0.5, 0.5)


def _build_d0() -> ControlledTarget:
    component = GaussianComponent(
        mean=np.asarray(_D0_V0_MEAN, dtype=np.float64),
        covariance=_diag_covariance(_D0_V0_STD),
        weight=1.0,
    )
    components_by_pdg_id = {13: (component,), -13: (component,)}
    return ControlledTarget(
        target_id="D0",
        description=(
            "D0 v0: five-dimensional diagonal Gaussian in physical coordinates "
            "[px, py, pz, x, y]."
        ),
        pdg_id_parameterization="shared_across_pdg_ids",
        components_by_pdg_id=components_by_pdg_id,
    )


# --- D1: five-dimensional full-covariance Gaussian --------------------------

_D1_V0_MEAN: Tuple[float, ...] = (0.0, 0.0, 50.0, 0.0, 0.0)
_D1_V0_COVARIANCE: Tuple[Tuple[float, ...], ...] = (
    (9.00, 1.80, 1.20, 0.75, 0.00),
    (1.80, 9.00, 1.20, 0.00, 0.75),
    (1.20, 1.20, 16.00, 0.10, 0.10),
    (0.75, 0.00, 0.10, 0.25, 0.05),
    (0.00, 0.75, 0.10, 0.05, 0.25),
)


def _build_d1() -> ControlledTarget:
    component = GaussianComponent(
        mean=np.asarray(_D1_V0_MEAN, dtype=np.float64),
        covariance=np.asarray(_D1_V0_COVARIANCE, dtype=np.float64),
        weight=1.0,
    )
    components_by_pdg_id = {13: (component,), -13: (component,)}
    return ControlledTarget(
        target_id="D1",
        description=(
            "D1 v0: five-dimensional full-covariance Gaussian in physical "
            "coordinates [px, py, pz, x, y] with non-trivial correlations."
        ),
        pdg_id_parameterization="shared_across_pdg_ids",
        components_by_pdg_id=components_by_pdg_id,
    )


# --- D2: PDG-id-conditioned two-component Gaussian mixture ------------------

_D2_V0_PARAMS: Dict[int, Tuple[Dict[str, Any], ...]] = {
    13: (
        {
            "mean": (2.0, 1.0, 45.0, 0.3, 0.2),
            "std": (3.0, 3.0, 4.0, 0.5, 0.5),
            "weight": 0.6,
        },
        {
            "mean": (-2.0, -1.0, 60.0, -0.3, -0.2),
            "std": (4.0, 4.0, 5.0, 0.6, 0.6),
            "weight": 0.4,
        },
    ),
    -13: (
        {
            "mean": (-1.5, 2.5, 55.0, 0.1, -0.4),
            "std": (3.5, 3.0, 4.5, 0.5, 0.5),
            "weight": 0.55,
        },
        {
            "mean": (3.0, -2.0, 48.0, -0.2, 0.3),
            "std": (4.0, 3.5, 4.0, 0.6, 0.6),
            "weight": 0.45,
        },
    ),
}


def _build_d2() -> ControlledTarget:
    components_by_pdg_id = {}
    for pdg_id, specs in _D2_V0_PARAMS.items():
        components_by_pdg_id[pdg_id] = tuple(
            GaussianComponent(
                mean=np.asarray(spec["mean"], dtype=np.float64),
                covariance=_diag_covariance(spec["std"]),
                weight=spec["weight"],
            )
            for spec in specs
        )
    return ControlledTarget(
        target_id="D2",
        description=(
            "D2 v0: PDG-id-conditioned two-component Gaussian mixture in "
            "physical coordinates [px, py, pz, x, y]."
        ),
        pdg_id_parameterization="pdg_id_conditioned_independent_mixtures",
        components_by_pdg_id=components_by_pdg_id,
    )


_TARGET_FACTORIES = {
    "D0": _build_d0,
    "D1": _build_d1,
    "D2": _build_d2,
}


def make_controlled_target(target_id: str) -> ControlledTarget:
    """Construct the exact controlled target named ``target_id``."""

    try:
        factory = _TARGET_FACTORIES[target_id]
    except (KeyError, TypeError) as exc:
        raise ControlledTargetConfigError(
            "unknown controlled target_id {!r}; expected one of {}".format(
                target_id, SUPPORTED_TARGET_IDS
            )
        ) from exc
    return factory()
