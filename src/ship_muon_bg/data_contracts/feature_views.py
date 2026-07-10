"""Deterministic feature views for post-shield muon density modelling.

This module implements the two invertible five-dimensional views fixed by
``docs/contracts/density_problem_contract_v0.md``:

``cartesian_log_v0``
    ``[px, py, log(pz / pz_unit), x, y]``

``slope_log_v0``
    ``[px / pz, py / pz, log(pz / pz_unit), x, y]``

The raw on-disk schema remains ``[px, py, pz, x, y, z, id, w]``. The scoring
plane ``z``, particle id and production weight are deliberately excluded from
the generated feature vector. No physical label, FairShip dependency or model
backend is introduced here.

The Jacobians in this module cover only the deterministic physical-state to
feature-view transform. A later train-fitted standardization layer must account
for its own affine Jacobian separately.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Dict, Tuple

import numpy as np

from . import schema
from .errors import (
    FeatureViewConfigError,
    FeatureViewDomainError,
    FeatureViewShapeError,
)

FEATURE_VIEW_SCHEMA_VERSION = "0"

CARTESIAN_LOG_VIEW_ID = "cartesian_log_v0"
SLOPE_LOG_VIEW_ID = "slope_log_v0"
SUPPORTED_FEATURE_VIEW_IDS = (
    CARTESIAN_LOG_VIEW_ID,
    SLOPE_LOG_VIEW_ID,
)

PHYSICAL_STATE_COLUMNS = ("px", "py", "pz", "x", "y")
CARTESIAN_LOG_FEATURES = ("px", "py", "log_pz", "x", "y")
SLOPE_LOG_FEATURES = ("tx", "ty", "log_pz", "x", "y")
N_DENSITY_FEATURES = len(PHYSICAL_STATE_COLUMNS)

_DEFAULT_PZ_UNIT_GEV = 1.0


def _canonical_json_hash(payload: Dict[str, Any]) -> str:
    """Return a stable SHA-256 hash for a JSON-serializable mapping."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _coerce_raw_array(raw_array: np.ndarray) -> np.ndarray:
    """Validate and return a float64 raw ``(N, 8)`` array.

    Only shape and finiteness of the five modelled physical coordinates are
    checked here. Full raw-contract validation, including ``id`` and ``w``,
    remains the responsibility of :mod:`ship_muon_bg.data_contracts.validation`.
    """

    try:
        array = np.asarray(raw_array, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise FeatureViewShapeError(
            f"raw_array is not coercible to float64: {exc}"
        ) from exc

    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] != schema.N_COLUMNS:
        raise FeatureViewShapeError(
            f"expected raw shape (N, {schema.N_COLUMNS}) with N >= 1, "
            f"got {array.shape}"
        )

    physical_indices = schema.column_indices(PHYSICAL_STATE_COLUMNS)
    physical = np.ascontiguousarray(array[:, physical_indices], dtype=np.float64)
    if not np.isfinite(physical).all():
        raise FeatureViewDomainError(
            "modelled physical coordinates contain NaN or inf"
        )
    return physical


def _coerce_feature_array(features: np.ndarray) -> np.ndarray:
    """Validate and return a float64 transformed ``(N, 5)`` array."""

    try:
        array = np.asarray(features, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise FeatureViewShapeError(
            f"features are not coercible to float64: {exc}"
        ) from exc

    if (
        array.ndim != 2
        or array.shape[0] < 1
        or array.shape[1] != N_DENSITY_FEATURES
    ):
        raise FeatureViewShapeError(
            f"expected feature shape (N, {N_DENSITY_FEATURES}) with N >= 1, "
            f"got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise FeatureViewDomainError("features contain NaN or inf")
    return np.ascontiguousarray(array, dtype=np.float64)


def _coerce_log_prob(log_prob: np.ndarray, n_rows: int) -> np.ndarray:
    """Validate a per-row log-probability vector."""

    try:
        values = np.asarray(log_prob, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise FeatureViewShapeError(
            f"log_prob is not coercible to float64: {exc}"
        ) from exc
    if values.shape != (n_rows,):
        raise FeatureViewShapeError(
            f"expected log_prob shape ({n_rows},), got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise FeatureViewDomainError("log_prob contains NaN or inf")
    return values


@dataclass(frozen=True)
class FeatureView:
    """An immutable deterministic feature-view configuration.

    Parameters
    ----------
    view_id
        One of :data:`SUPPORTED_FEATURE_VIEW_IDS`.
    pz_unit_gev
        Positive finite numerical value, in GeV/c, used to make
        ``log(pz / pz_unit)`` dimensionless. The raw contract stores momentum
        numerically in GeV/c.
    """

    view_id: str
    pz_unit_gev: float = _DEFAULT_PZ_UNIT_GEV

    def __post_init__(self) -> None:
        if self.view_id not in SUPPORTED_FEATURE_VIEW_IDS:
            raise FeatureViewConfigError(
                f"unsupported feature view {self.view_id!r}; "
                f"expected one of {SUPPORTED_FEATURE_VIEW_IDS}"
            )
        try:
            pz_unit = float(self.pz_unit_gev)
        except (TypeError, ValueError) as exc:
            raise FeatureViewConfigError("pz_unit_gev must be numeric") from exc
        if not np.isfinite(pz_unit) or pz_unit <= 0.0:
            raise FeatureViewConfigError(
                "pz_unit_gev must be finite and strictly positive"
            )
        object.__setattr__(self, "pz_unit_gev", pz_unit)

    @property
    def feature_names(self) -> Tuple[str, ...]:
        """Return the ordered output feature names."""

        if self.view_id == CARTESIAN_LOG_VIEW_ID:
            return CARTESIAN_LOG_FEATURES
        return SLOPE_LOG_FEATURES

    @property
    def jacobian_pz_power(self) -> int:
        """Power ``k`` in ``|det dT/dx| = pz**(-k)``."""

        if self.view_id == CARTESIAN_LOG_VIEW_ID:
            return 1
        return 3

    def manifest(self) -> Dict[str, Any]:
        """Return the JSON-serializable normative configuration manifest."""

        if self.view_id == CARTESIAN_LOG_VIEW_ID:
            forward_definition = [
                "px",
                "py",
                "log(pz / pz_unit)",
                "x",
                "y",
            ]
            inverse_definition = [
                "px = feature[0]",
                "py = feature[1]",
                "pz = pz_unit * exp(feature[2])",
                "x = feature[3]",
                "y = feature[4]",
            ]
        else:
            forward_definition = [
                "tx = px / pz",
                "ty = py / pz",
                "log(pz / pz_unit)",
                "x",
                "y",
            ]
            inverse_definition = [
                "pz = pz_unit * exp(feature[2])",
                "px = feature[0] * pz",
                "py = feature[1] * pz",
                "x = feature[3]",
                "y = feature[4]",
            ]

        return {
            "schema_version": FEATURE_VIEW_SCHEMA_VERSION,
            "feature_view_id": self.view_id,
            "raw_contract_version": schema.CONTRACT_VERSION,
            "raw_columns": list(schema.COLUMNS),
            "physical_input_columns": list(PHYSICAL_STATE_COLUMNS),
            "feature_output_columns": list(self.feature_names),
            "excluded_raw_columns": ["z", "id", "w"],
            "pz_unit_gev": self.pz_unit_gev,
            "domain": {"pz": "strictly_positive"},
            "forward_definition": forward_definition,
            "inverse_definition": inverse_definition,
            "physical_to_feature_log_abs_det_jacobian": (
                f"-{self.jacobian_pz_power} * log(pz)"
            ),
            "includes_train_fitted_standardization": False,
        }

    def config_hash(self) -> str:
        """Return a deterministic hash of :meth:`manifest`."""

        return _canonical_json_hash(self.manifest())

    def forward(self, raw_array: np.ndarray) -> np.ndarray:
        """Map validated raw rows into the configured five-dimensional view.

        Parameters
        ----------
        raw_array
            Raw ``(N, 8)`` rows in fixed schema order.

        Returns
        -------
        numpy.ndarray
            C-contiguous ``float64`` array with shape ``(N, 5)``.

        Raises
        ------
        FeatureViewDomainError
            If any modelled coordinate is non-finite or ``pz <= 0``.
        """

        physical = _coerce_raw_array(raw_array)
        px, py, pz, x_pos, y_pos = physical.T
        if np.any(pz <= 0.0):
            bad_count = int(np.count_nonzero(pz <= 0.0))
            raise FeatureViewDomainError(
                f"{self.view_id} requires pz > 0; found {bad_count} invalid rows"
            )

        log_pz = np.log(pz / self.pz_unit_gev)
        if self.view_id == CARTESIAN_LOG_VIEW_ID:
            transformed = np.column_stack((px, py, log_pz, x_pos, y_pos))
        else:
            transformed = np.column_stack(
                (px / pz, py / pz, log_pz, x_pos, y_pos)
            )

        if not np.isfinite(transformed).all():
            raise FeatureViewDomainError(
                "feature transformation produced NaN or inf"
            )
        return np.ascontiguousarray(transformed, dtype=np.float64)

    def inverse(self, features: np.ndarray) -> np.ndarray:
        """Map feature rows back to physical ``[px, py, pz, x, y]`` rows.

        The excluded raw values ``z``, ``id`` and ``w`` cannot be reconstructed
        from the density view and are intentionally not returned.
        """

        feature_array = _coerce_feature_array(features)
        first, second, log_pz, x_pos, y_pos = feature_array.T
        with np.errstate(over="ignore", invalid="ignore"):
            pz = self.pz_unit_gev * np.exp(log_pz)
            if self.view_id == CARTESIAN_LOG_VIEW_ID:
                px = first
                py = second
            else:
                px = first * pz
                py = second * pz

        physical = np.column_stack((px, py, pz, x_pos, y_pos))
        if not np.isfinite(physical).all() or np.any(pz <= 0.0):
            raise FeatureViewDomainError(
                "inverse feature transformation left the finite pz > 0 domain"
            )
        return np.ascontiguousarray(physical, dtype=np.float64)

    def forward_log_abs_det_jacobian(self, raw_array: np.ndarray) -> np.ndarray:
        """Return ``log|det d(feature)/d(physical)|`` for each raw row."""

        physical = _coerce_raw_array(raw_array)
        pz = physical[:, 2]
        if np.any(pz <= 0.0):
            raise FeatureViewDomainError(
                f"{self.view_id} Jacobian requires pz > 0"
            )
        return -float(self.jacobian_pz_power) * np.log(pz)

    def inverse_log_abs_det_jacobian(self, features: np.ndarray) -> np.ndarray:
        """Return ``log|det d(physical)/d(feature)|`` for each feature row."""

        physical = self.inverse(features)
        pz = physical[:, 2]
        return float(self.jacobian_pz_power) * np.log(pz)

    def physical_log_prob_from_feature(
        self,
        feature_log_prob: np.ndarray,
        raw_array: np.ndarray,
    ) -> np.ndarray:
        """Convert feature-space log density to physical-coordinate log density.

        This applies only the deterministic feature-view Jacobian:

        ``log q_physical(x) = log q_feature(T(x)) + log|det dT/dx|``.

        Any train-fitted standardization Jacobian must already have been
        included in ``feature_log_prob`` or added by its owning layer.
        """

        physical = _coerce_raw_array(raw_array)
        values = _coerce_log_prob(feature_log_prob, physical.shape[0])
        return values + self.forward_log_abs_det_jacobian(raw_array)
