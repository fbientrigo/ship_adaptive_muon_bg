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
SUPPORTED_TARGET_IDS: Tuple[str, ...] = ("D0", "D1", "D2", "D3", "D4", "D5")

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
    """Numerically stable ``log(sum(exp(log_terms)))`` along ``axis``.

    Rows where every term is ``-inf`` (all mixture components underflow at
    a far-tail point) are shifted by ``0`` instead of ``max_term``: shifting
    by ``max_term`` there would compute ``-inf - (-inf) = nan``, when the
    mathematically exact result is ``-inf``.
    """

    max_term = np.max(log_terms, axis=axis, keepdims=True)
    shift = np.where(np.isneginf(max_term), 0.0, max_term)
    with np.errstate(divide="ignore"):
        summed = np.sum(np.exp(log_terms - shift), axis=axis, keepdims=True)
        result = shift + np.log(summed)
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
        # Validate each key's type strictly (same rule as `_validate_pdg_id`)
        # before comparing the key set, so a key that is merely `==` to a
        # supported id but not actually an int (e.g. `13.9`, `"13"`) cannot
        # be silently coerced by `int(pdg_id)` into overwriting a real entry.
        coerced_keys: Dict[Any, int] = {}
        for pdg_id in components_by_pdg_id.keys():
            if isinstance(pdg_id, bool) or not isinstance(pdg_id, (int, np.integer)):
                raise ControlledTargetConfigError(
                    "components_by_pdg_id must declare exactly the supported "
                    "PDG ids {}, got a non-integer key {!r}".format(
                        sorted(SUPPORTED_PDG_IDS), pdg_id
                    )
                )
            coerced_keys[pdg_id] = int(pdg_id)

        if set(coerced_keys.values()) != set(SUPPORTED_PDG_IDS):
            raise ControlledTargetConfigError(
                "components_by_pdg_id must declare exactly the supported PDG "
                "ids {}, got {}".format(
                    sorted(SUPPORTED_PDG_IDS), sorted(coerced_keys.values())
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
            normalized[coerced_keys[pdg_id]] = components

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

    def _component_log_terms(
        self, physical: np.ndarray, pdg_id: int
    ) -> np.ndarray:
        components = self._components_for(pdg_id)
        return np.stack(
            [
                math.log(component.weight) + component.log_prob(physical)
                for component in components
            ],
            axis=-1,
        )

    def component_log_prob(
        self, physical: np.ndarray, *, pdg_id: int
    ) -> np.ndarray:
        """Return the weighted per-component log density, shape ``(N, K)``.

        Column ``k`` is ``log(weight_k) + log N(x; mean_k, cov_k)``; the
        mixture ``log_prob`` is ``logsumexp`` across the columns.
        """

        pdg_id = _validate_pdg_id(pdg_id)
        physical = _validate_physical_array(physical)
        return self._component_log_terms(physical, pdg_id)

    def component_posterior(
        self, physical: np.ndarray, *, pdg_id: int
    ) -> np.ndarray:
        """Return normalized component responsibilities, shape ``(N, K)``.

        Each row sums to 1 (a valid categorical posterior over components).
        """

        log_terms = self.component_log_prob(physical, pdg_id=pdg_id)
        log_norm = _logsumexp(log_terms, axis=-1)
        return np.exp(log_terms - log_norm[:, np.newaxis])

    def declared_regions(self) -> Tuple[Any, ...]:
        """Return declared architecture-independent regions (none for D0-D2)."""

        return ()

    def region_mask(
        self, physical: np.ndarray, *, pdg_id: int, region_id: str
    ) -> np.ndarray:
        """Return a boolean membership mask for the named declared region."""

        raise ControlledTargetConfigError(
            "target {} declares no region {!r}".format(self.target_id, region_id)
        )

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


# ===========================================================================
# Transformed targets: D3 (curved), D4 (heteroscedastic+skew), D5 (rare tail)
# ===========================================================================


def _cov_from_std_corr(
    std: Sequence[float], corr_pairs: Mapping[Tuple[int, int], float]
) -> np.ndarray:
    std_array = np.asarray(std, dtype=np.float64)
    cov = np.diag(std_array**2)
    for (i, j), rho in corr_pairs.items():
        value = rho * std_array[i] * std_array[j]
        cov[i, j] = value
        cov[j, i] = value
    return cov


class TransformedControlledTarget:
    """An exact controlled target: a base mixture wrapped by exact transforms.

    ``sample`` draws from the base mixture (retaining exact component labels)
    and applies the transform forward map. ``log_prob`` maps a physical point
    back through the exact inverse and adds the exact inverse log-Jacobian, so
    the density is exact. ``pz`` is preserved by every transform in the chain.
    """

    def __init__(
        self,
        *,
        target_id: str,
        target_variant: str,
        description: str,
        base: ControlledTarget,
        transform: Any,  # ExactTransform (imported lazily to avoid a cycle)
        rare_component_id_by_pdg_id: Mapping[int, int] | None = None,
        regions_by_pdg_id: Mapping[int, Mapping[str, Any]] | None = None,
        rare_mass: float | None = None,
        calibration: Mapping[str, Any] | None = None,
    ) -> None:
        self.target_id = target_id
        self.target_variant = target_variant
        self.description = description
        self._base = base
        self._transform = transform
        self._rare_component_id_by_pdg_id = dict(rare_component_id_by_pdg_id or {})
        self._regions_by_pdg_id = {
            int(k): dict(v) for k, v in (regions_by_pdg_id or {}).items()
        }
        self.rare_mass = None if rare_mass is None else float(rare_mass)
        self._calibration = dict(calibration or {})

    # -- sampling / density --------------------------------------------------

    def sample(self, n: int, *, pdg_id: int, seed: int) -> SampleBatch:
        base_batch = self._base.sample(n, pdg_id=pdg_id, seed=seed)
        transformed = self._transform.forward(base_batch.physical)
        transformed = np.ascontiguousarray(transformed, dtype=np.float64)
        return SampleBatch(
            physical=transformed,
            component_id=base_batch.component_id,
            pdg_id=base_batch.pdg_id,
            target_id=self.target_id,
            seed=base_batch.seed,
        )

    def base_of(self, physical: np.ndarray, *, pdg_id: int) -> np.ndarray:
        """Map transformed physical coordinates back to base coordinates."""

        _validate_pdg_id(pdg_id)
        physical = _validate_physical_array(physical)
        return self._transform.inverse(physical)

    def log_prob(self, physical: np.ndarray, *, pdg_id: int) -> np.ndarray:
        pdg_id = _validate_pdg_id(pdg_id)
        physical = _validate_physical_array(physical)
        base = self._transform.inverse(physical)
        return self._base.log_prob(
            base, pdg_id=pdg_id
        ) + self._transform.inverse_log_abs_det_jacobian(physical)

    def component_log_prob(
        self, physical: np.ndarray, *, pdg_id: int
    ) -> np.ndarray:
        pdg_id = _validate_pdg_id(pdg_id)
        physical = _validate_physical_array(physical)
        base = self._transform.inverse(physical)
        jac = self._transform.inverse_log_abs_det_jacobian(physical)
        return self._base.component_log_prob(base, pdg_id=pdg_id) + jac[:, np.newaxis]

    def component_posterior(
        self, physical: np.ndarray, *, pdg_id: int
    ) -> np.ndarray:
        # The inverse Jacobian is common to every component and cancels, so the
        # posterior equals the base posterior at the mapped-back point.
        pdg_id = _validate_pdg_id(pdg_id)
        physical = _validate_physical_array(physical)
        base = self._transform.inverse(physical)
        return self._base.component_posterior(base, pdg_id=pdg_id)

    # -- regions -------------------------------------------------------------

    def declared_regions(self) -> Tuple[str, ...]:
        region_ids: Tuple[str, ...] = ()
        for regions in self._regions_by_pdg_id.values():
            region_ids = tuple(sorted(regions.keys()))
            break
        return region_ids

    def rare_component_id(self, *, pdg_id: int) -> int:
        pdg_id = _validate_pdg_id(pdg_id)
        return int(self._rare_component_id_by_pdg_id[pdg_id])

    def region_mask(
        self, physical: np.ndarray, *, pdg_id: int, region_id: str
    ) -> np.ndarray:
        pdg_id = _validate_pdg_id(pdg_id)
        physical = _validate_physical_array(physical)
        try:
            region = self._regions_by_pdg_id[pdg_id][region_id]
        except KeyError as exc:
            raise ControlledTargetConfigError(
                "target {} declares no region {!r} for pdg_id {}".format(
                    self.target_id, region_id, pdg_id
                )
            ) from exc
        base = self._transform.inverse(physical)
        return region.contains(base)

    # -- manifest ------------------------------------------------------------

    def manifest(self) -> Dict[str, Any]:
        regions_manifest: Dict[str, Any] = {}
        for pdg_id, regions in self._regions_by_pdg_id.items():
            regions_manifest[str(pdg_id)] = {
                rid: region.manifest() for rid, region in regions.items()
            }
        payload = {
            "target_schema_version": TARGET_SCHEMA_VERSION,
            "target_id": self.target_id,
            "target_variant": self.target_variant,
            "target_description": self.description,
            "density_coordinate": "physical_px_py_pz_x_y",
            "physical_columns": list(PHYSICAL_COLUMNS),
            "supported_pdg_ids": list(SUPPORTED_PDG_IDS),
            "base_target": self._base.manifest(),
            "transform": self._transform.manifest(),
            "exact_sample": True,
            "exact_log_prob": True,
            "physics_claim": False,
            "event_level_conservation_applied": False,
        }
        if self.rare_mass is not None:
            payload["rare_mass"] = self.rare_mass
            payload["rare_component_id"] = {
                str(k): int(v) for k, v in self._rare_component_id_by_pdg_id.items()
            }
            payload["rare_region_definition"] = regions_manifest
            payload["rare_region_calibration"] = self._calibration
        return payload

    def config_hash(self) -> str:
        return _canonical_json_hash(self.manifest())


# --- D3: curved nonlinear correlations (banana transform) -------------------

_D3_BANANA = {
    "scale_px": 3.0,
    "scale_py": 3.0,
    "scale_x": 0.5,
    "curvature": 0.9,
    "shear": 0.6,
}

_D3_BASE_PARAMS: Dict[int, Tuple[Dict[str, Any], ...]] = {
    13: (
        {
            "mean": (1.5, 0.5, 48.0, 0.1, -0.1),
            "std": (3.0, 3.0, 4.0, 0.5, 0.5),
            "corr": {(0, 1): 0.3},
            "weight": 0.6,
        },
        {
            "mean": (-2.0, -1.0, 55.0, -0.2, 0.2),
            "std": (3.5, 3.0, 4.5, 0.5, 0.5),
            "corr": {(0, 1): -0.2},
            "weight": 0.4,
        },
    ),
    -13: (
        {
            "mean": (-1.0, 1.5, 52.0, -0.1, 0.1),
            "std": (3.0, 3.2, 4.2, 0.5, 0.5),
            "corr": {(0, 1): 0.25},
            "weight": 0.55,
        },
        {
            "mean": (2.5, -1.5, 47.0, 0.2, -0.2),
            "std": (3.5, 3.0, 4.0, 0.5, 0.5),
            "corr": {(0, 1): -0.15},
            "weight": 0.45,
        },
    ),
}


def _components_from_specs(
    specs: Sequence[Dict[str, Any]]
) -> Tuple[GaussianComponent, ...]:
    return tuple(
        GaussianComponent(
            mean=np.asarray(spec["mean"], dtype=np.float64),
            covariance=_cov_from_std_corr(spec["std"], spec.get("corr", {})),
            weight=spec["weight"],
        )
        for spec in specs
    )


def _build_d3() -> TransformedControlledTarget:
    from .target_transforms import TriangularBananaTransform

    base = ControlledTarget(
        target_id="D3_base",
        description="D3 base: PDG-id-conditioned correlated Gaussian mixture.",
        pdg_id_parameterization="pdg_id_conditioned_independent_mixtures",
        components_by_pdg_id={
            pdg_id: _components_from_specs(specs)
            for pdg_id, specs in _D3_BASE_PARAMS.items()
        },
    )
    transform = TriangularBananaTransform(**_D3_BANANA)
    return TransformedControlledTarget(
        target_id="D3",
        target_variant="default",
        description=(
            "D3 v0: curved nonlinear correlations from a smooth triangular "
            "banana transform of a Gaussian mixture (unit Jacobian; pz "
            "unchanged). Falsifies a full Gaussian; learnable by a small "
            "coupling flow."
        ),
        base=base,
        transform=transform,
    )


# --- D4: asymmetric heteroscedastic multimodality + skew --------------------

_D4_SKEW_SCALES = (4.0, 4.0, 5.0, 0.6, 0.6)
_D4_SKEW_SKEWS = (0.6, -0.4, 0.0, 0.3, -0.3)
_D4_BANANA = {
    "scale_px": 4.0,
    "scale_py": 4.0,
    "scale_x": 0.6,
    "curvature": 0.7,
    "shear": 0.5,
}

_D4_BASE_PARAMS: Dict[int, Tuple[Dict[str, Any], ...]] = {
    13: (
        {
            "mean": (3.0, 1.0, 45.0, 0.3, 0.1),
            "std": (2.5, 2.5, 4.0, 0.4, 0.4),
            "corr": {(0, 1): 0.4, (0, 3): 0.3},
            "weight": 0.5,
        },
        {
            "mean": (-4.0, -2.0, 58.0, -0.4, 0.3),
            "std": (5.0, 3.0, 5.0, 0.7, 0.5),
            "corr": {(0, 1): -0.3, (1, 4): 0.35},
            "weight": 0.35,
        },
        {
            "mean": (0.0, 5.0, 52.0, 0.0, -0.5),
            "std": (3.0, 4.5, 4.5, 0.5, 0.7),
            "corr": {(1, 4): -0.4},
            "weight": 0.15,
        },
    ),
    -13: (
        {
            "mean": (-2.5, 2.0, 50.0, -0.2, 0.2),
            "std": (2.8, 3.5, 4.5, 0.45, 0.55),
            "corr": {(0, 1): -0.35, (1, 4): 0.3},
            "weight": 0.55,
        },
        {
            "mean": (4.0, -3.0, 46.0, 0.4, -0.3),
            "std": (4.5, 2.8, 4.0, 0.65, 0.45),
            "corr": {(0, 3): 0.4},
            "weight": 0.3,
        },
        {
            "mean": (0.5, -1.0, 60.0, 0.1, 0.4),
            "std": (3.2, 3.2, 5.0, 0.5, 0.6),
            "corr": {(0, 1): 0.2},
            "weight": 0.15,
        },
    ),
}


def _build_d4_transform() -> Any:
    from .target_transforms import (
        ComposedTransform,
        SinhArcsinhSkewTransform,
        TriangularBananaTransform,
    )

    skew = SinhArcsinhSkewTransform(scales=_D4_SKEW_SCALES, skews=_D4_SKEW_SKEWS)
    banana = TriangularBananaTransform(**_D4_BANANA)
    return ComposedTransform([skew, banana])


def _build_d4() -> TransformedControlledTarget:
    base = ControlledTarget(
        target_id="D4_base",
        description="D4 base: asymmetric heteroscedastic Gaussian mixture.",
        pdg_id_parameterization="pdg_id_conditioned_independent_mixtures",
        components_by_pdg_id={
            pdg_id: _components_from_specs(specs)
            for pdg_id, specs in _D4_BASE_PARAMS.items()
        },
    )
    return TransformedControlledTarget(
        target_id="D4",
        target_variant="default",
        description=(
            "D4 v0: asymmetric heteroscedastic multimodal mixture under a "
            "sinh-arcsinh skew composed with a triangular banana transform "
            "(exact inverse and Jacobian; pz unchanged). Harder than D3 for "
            "an affine-only family."
        ),
        base=base,
        transform=_build_d4_transform(),
    )


# --- D5: labelled rare tail mode --------------------------------------------

D5_VARIANTS: Dict[str, float] = {
    "rare_1e-2": 1e-2,
    "rare_1e-3": 1e-3,
}
_D5_DEFAULT_VARIANT = "rare_1e-2"

# Deterministic Monte Carlo calibration provenance for the rare region.
D5_CALIBRATION_SEED = 987654321
D5_CALIBRATION_N = 1_000_000

# Rare component means per pdg id (well separated from the main modes so the
# rare-region ellipsoid captures it cleanly with low main-mode contamination).
_D5_RARE_MEAN = {
    13: (18.0, 16.0, 50.0, 3.0, -3.0),
    -13: (-18.0, -16.0, 50.0, -3.0, 3.0),
}
_D5_RARE_STD = (2.0, 2.0, 4.0, 0.4, 0.4)
_D5_RARE_REGION_RADIUS_SQ = 16.0  # 4 standardized-sigma Mahalanobis ellipsoid

# Precomputed rare-region calibration (target mass + main-mode contamination),
# obtained once via the deterministic MC above. Recomputed and checked in
# tests/test_controlled_targets_d3_d5.py; never recomputed during training.
_D5_RARE_CALIBRATION: Dict[str, Dict[str, Dict[str, float]]] = {
    "rare_1e-2": {
        "13": {
            "target_probability_in_rare_region": 0.009812,
            "target_probability_stderr": 9.856837553698447e-05,
            "main_component_contamination_in_rare_region": 0.0,
        },
        "-13": {
            "target_probability_in_rare_region": 0.009812,
            "target_probability_stderr": 9.856837553698447e-05,
            "main_component_contamination_in_rare_region": 0.0,
        },
    },
    "rare_1e-3": {
        "13": {
            "target_probability_in_rare_region": 0.000979,
            "target_probability_stderr": 3.1273655990305963e-05,
            "main_component_contamination_in_rare_region": 0.0,
        },
        "-13": {
            "target_probability_in_rare_region": 0.000979,
            "target_probability_stderr": 3.1273655990305963e-05,
            "main_component_contamination_in_rare_region": 0.0,
        },
    },
}


def _d5_base_params(rare_mass: float) -> Dict[int, Tuple[Dict[str, Any], ...]]:
    main_mass = 1.0 - rare_mass
    params: Dict[int, Tuple[Dict[str, Any], ...]] = {}
    for pdg_id, main_specs in _D4_BASE_PARAMS.items():
        rescaled_main = []
        for spec in main_specs:
            new_spec = dict(spec)
            new_spec["weight"] = spec["weight"] * main_mass
            rescaled_main.append(new_spec)
        rare_spec = {
            "mean": _D5_RARE_MEAN[pdg_id],
            "std": _D5_RARE_STD,
            "corr": {},
            "weight": rare_mass,
        }
        params[pdg_id] = tuple(rescaled_main) + (rare_spec,)
    return params


def _build_d5(variant: str) -> TransformedControlledTarget:
    from .target_regions import MahalanobisRegion

    if variant not in D5_VARIANTS:
        raise ControlledTargetConfigError(
            "unknown D5 variant {!r}; expected one of {}".format(
                variant, sorted(D5_VARIANTS)
            )
        )
    rare_mass = D5_VARIANTS[variant]
    base_params = _d5_base_params(rare_mass)
    base = ControlledTarget(
        target_id="D5_base",
        description="D5 base: heteroscedastic mixture plus a rare tail mode.",
        pdg_id_parameterization="pdg_id_conditioned_independent_mixtures",
        components_by_pdg_id={
            pdg_id: _components_from_specs(specs)
            for pdg_id, specs in base_params.items()
        },
    )
    transform = _build_d4_transform()

    rare_precision = np.linalg.inv(_cov_from_std_corr(_D5_RARE_STD, {}))
    rare_component_id_by_pdg_id: Dict[int, int] = {}
    regions_by_pdg_id: Dict[int, Dict[str, MahalanobisRegion]] = {}
    for pdg_id in SUPPORTED_PDG_IDS:
        rare_component_id_by_pdg_id[pdg_id] = len(base_params[pdg_id]) - 1
        regions_by_pdg_id[pdg_id] = {
            "rare_tail": MahalanobisRegion(
                region_id="rare_tail",
                center=np.asarray(_D5_RARE_MEAN[pdg_id], dtype=np.float64),
                precision=rare_precision,
                radius_sq=_D5_RARE_REGION_RADIUS_SQ,
                description=(
                    "Fixed 4-sigma Mahalanobis ellipsoid around the rare mode "
                    "in base physical coordinates."
                ),
            )
        }

    calibration = {
        "calibration_seed": D5_CALIBRATION_SEED,
        "calibration_n": D5_CALIBRATION_N,
        "per_pdg_id": _D5_RARE_CALIBRATION[variant],
    }
    return TransformedControlledTarget(
        target_id="D5",
        target_variant=variant,
        description=(
            "D5 v0 ({}): labelled rare tail mode (rare_mass={}) added to the "
            "D4 heteroscedastic family with the same transform. Rare region "
            "declared as a fixed Mahalanobis ellipsoid in base coordinates."
        ).format(variant, rare_mass),
        base=base,
        transform=transform,
        rare_component_id_by_pdg_id=rare_component_id_by_pdg_id,
        regions_by_pdg_id=regions_by_pdg_id,
        rare_mass=rare_mass,
        calibration=calibration,
    )


def calibrate_d5_rare_region(
    target: TransformedControlledTarget,
    *,
    pdg_id: int,
    seed: int = D5_CALIBRATION_SEED,
    n_samples: int = D5_CALIBRATION_N,
) -> Dict[str, float]:
    """Deterministically Monte-Carlo the rare-region mass and contamination.

    Returns the target probability inside the rare region, its binomial
    standard error, and the fraction of in-region samples that are *not* from
    the rare component (main-mode contamination). Deterministic in
    ``(seed, n_samples)``; used to produce and verify the versioned manifest
    calibration constants (never called during training).
    """

    pdg_id = _validate_pdg_id(pdg_id)
    batch = target.sample(n_samples, pdg_id=pdg_id, seed=seed)
    mask = target.region_mask(batch.physical, pdg_id=pdg_id, region_id="rare_tail")
    n_region = int(np.count_nonzero(mask))
    prob = n_region / float(n_samples)
    stderr = float(math.sqrt(max(prob * (1.0 - prob), 0.0) / n_samples))
    rare_id = target.rare_component_id(pdg_id=pdg_id)
    if n_region > 0:
        contamination = float(np.mean(batch.component_id[mask] != rare_id))
    else:
        contamination = 0.0
    return {
        "target_probability_in_rare_region": prob,
        "target_probability_stderr": stderr,
        "main_component_contamination_in_rare_region": contamination,
        "observed_rare_region_count": n_region,
    }


_TARGET_FACTORIES = {
    "D0": _build_d0,
    "D1": _build_d1,
    "D2": _build_d2,
    "D3": _build_d3,
    "D4": _build_d4,
}


def make_controlled_target(
    target_id: str, *, variant: str | None = None
) -> Any:
    """Construct the exact controlled target named ``target_id``.

    ``variant`` selects a parameterized variant (currently only D5 uses it:
    ``"rare_1e-2"`` or ``"rare_1e-3"``). For D0-D4 ``variant`` must be ``None``
    or ``"default"``.
    """

    if target_id == "D5":
        return _build_d5(variant or _D5_DEFAULT_VARIANT)

    if variant not in (None, "default"):
        raise ControlledTargetConfigError(
            "target {!r} does not accept variant {!r}".format(target_id, variant)
        )
    try:
        factory = _TARGET_FACTORIES[target_id]
    except (KeyError, TypeError) as exc:
        raise ControlledTargetConfigError(
            "unknown controlled target_id {!r}; expected one of {}".format(
                target_id, SUPPORTED_TARGET_IDS
            )
        ) from exc
    return factory()
