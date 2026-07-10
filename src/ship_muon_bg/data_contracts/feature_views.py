"""Controlled feature-view experiment for post-shield muon density modelling.

The module implements one untransformed reference arm and two invertible
candidate arms. No candidate is assumed to be superior. All three arms are
intended to run against matched raw rows, splits, seeds, model configurations
and evaluation budgets.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Dict, Optional, Tuple

import numpy as np

from . import schema
from .errors import (
    FeatureViewConfigError,
    FeatureViewDomainError,
    FeatureViewShapeError,
)

FEATURE_VIEW_SCHEMA_VERSION = "0"
FEATURE_VIEW_EXPERIMENT_ID = "feature_view_ab_v0"

IDENTITY_CARTESIAN_VIEW_ID = "identity_cartesian_v0"
CARTESIAN_LOGPZ_VIEW_ID = "cartesian_logpz_v0"
SLOPE_LOGPZ_VIEW_ID = "slope_logpz_v0"

SUPPORTED_FEATURE_VIEW_IDS = (
    IDENTITY_CARTESIAN_VIEW_ID,
    CARTESIAN_LOGPZ_VIEW_ID,
    SLOPE_LOGPZ_VIEW_ID,
)

PHYSICAL_STATE_COLUMNS = ("px", "py", "pz", "x", "y")
IDENTITY_CARTESIAN_FEATURES = PHYSICAL_STATE_COLUMNS
CARTESIAN_LOGPZ_FEATURES = ("px", "py", "log_pz", "x", "y")
SLOPE_LOGPZ_FEATURES = ("tx", "ty", "log_pz", "x", "y")
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
    """Validate and return modelled physical coordinates from raw ``(N, 8)`` rows."""

    try:
        array = np.asarray(raw_array, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise FeatureViewShapeError(
            "raw_array is not coercible to float64: {}".format(exc)
        ) from exc

    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] != schema.N_COLUMNS:
        raise FeatureViewShapeError(
            "expected raw shape (N, {}) with N >= 1, got {}".format(
                schema.N_COLUMNS, array.shape
            )
        )

    physical_indices = schema.column_indices(PHYSICAL_STATE_COLUMNS)
    physical = np.ascontiguousarray(array[:, physical_indices], dtype=np.float64)
    if not np.isfinite(physical).all():
        raise FeatureViewDomainError(
            "modelled physical coordinates contain NaN or inf"
        )
    return physical


def _coerce_feature_array(features: np.ndarray) -> np.ndarray:
    """Validate and return a finite ``float64`` transformed ``(N, 5)`` array."""

    try:
        array = np.asarray(features, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise FeatureViewShapeError(
            "features are not coercible to float64: {}".format(exc)
        ) from exc

    if (
        array.ndim != 2
        or array.shape[0] < 1
        or array.shape[1] != N_DENSITY_FEATURES
    ):
        raise FeatureViewShapeError(
            "expected feature shape (N, {}) with N >= 1, got {}".format(
                N_DENSITY_FEATURES, array.shape
            )
        )
    if not np.isfinite(array).all():
        raise FeatureViewDomainError("features contain NaN or inf")
    return np.ascontiguousarray(array, dtype=np.float64)


def _coerce_log_prob(log_prob: np.ndarray, n_rows: int) -> np.ndarray:
    """Validate a finite per-row log-probability vector."""

    try:
        values = np.asarray(log_prob, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise FeatureViewShapeError(
            "log_prob is not coercible to float64: {}".format(exc)
        ) from exc
    if values.shape != (n_rows,):
        raise FeatureViewShapeError(
            "expected log_prob shape ({},), got {}".format(n_rows, values.shape)
        )
    if not np.isfinite(values).all():
        raise FeatureViewDomainError("log_prob contains NaN or inf")
    return values


def feature_view_experiment_manifest() -> Dict[str, Any]:
    """Return the normative matched-arm experiment definition."""

    return {
        "schema_version": FEATURE_VIEW_SCHEMA_VERSION,
        "experiment_id": FEATURE_VIEW_EXPERIMENT_ID,
        "comparison_type": "matched_multi_arm_ab",
        "baseline_arm_id": IDENTITY_CARTESIAN_VIEW_ID,
        "candidate_arm_ids": [
            CARTESIAN_LOGPZ_VIEW_ID,
            SLOPE_LOGPZ_VIEW_ID,
        ],
        "required_match_keys": [
            "raw_dataset_hash",
            "weight_target_id",
            "charge",
            "split_manifest_hash",
            "seed",
            "model_name",
            "model_config_hash",
            "training_budget_id",
            "metric_config_hash",
        ],
        "selection_rule": (
            "No transformed arm is promoted by default. Compare physical-space "
            "metrics and minimum passing model cost under matched conditions."
        ),
    }


@dataclass(frozen=True)
class FeatureView:
    """Immutable configuration for one feature-view experiment arm.

    Parameters
    ----------
    view_id
        One of :data:`SUPPORTED_FEATURE_VIEW_IDS`.
    pz_unit_gev
        Positive finite value used only by logarithmic arms to make
        ``log(pz / pz_unit)`` dimensionless. It must be omitted for the
        identity reference arm so an irrelevant parameter cannot alter its
        configuration hash.
    """

    view_id: str
    pz_unit_gev: Optional[float] = None

    def __post_init__(self) -> None:
        if self.view_id not in SUPPORTED_FEATURE_VIEW_IDS:
            raise FeatureViewConfigError(
                "unsupported feature view {!r}; expected one of {}".format(
                    self.view_id, SUPPORTED_FEATURE_VIEW_IDS
                )
            )

        if self.view_id == IDENTITY_CARTESIAN_VIEW_ID:
            if self.pz_unit_gev is not None:
                raise FeatureViewConfigError(
                    "identity_cartesian_v0 does not accept pz_unit_gev"
                )
            return

        pz_unit = _DEFAULT_PZ_UNIT_GEV if self.pz_unit_gev is None else self.pz_unit_gev
        try:
            pz_unit_float = float(pz_unit)
        except (TypeError, ValueError) as exc:
            raise FeatureViewConfigError("pz_unit_gev must be numeric") from exc
        if not np.isfinite(pz_unit_float) or pz_unit_float <= 0.0:
            raise FeatureViewConfigError(
                "pz_unit_gev must be finite and strictly positive"
            )
        object.__setattr__(self, "pz_unit_gev", pz_unit_float)

    @property
    def experiment_role(self) -> str:
        """Return ``baseline`` for identity and ``candidate`` otherwise."""

        if self.view_id == IDENTITY_CARTESIAN_VIEW_ID:
            return "baseline"
        return "candidate"

    @property
    def feature_names(self) -> Tuple[str, ...]:
        """Return ordered output feature names."""

        if self.view_id == IDENTITY_CARTESIAN_VIEW_ID:
            return IDENTITY_CARTESIAN_FEATURES
        if self.view_id == CARTESIAN_LOGPZ_VIEW_ID:
            return CARTESIAN_LOGPZ_FEATURES
        return SLOPE_LOGPZ_FEATURES

    @property
    def jacobian_pz_power(self) -> int:
        """Return ``k`` in ``|det dT/dx| = pz**(-k)`` for numeric GeV units."""

        if self.view_id == IDENTITY_CARTESIAN_VIEW_ID:
            return 0
        if self.view_id == CARTESIAN_LOGPZ_VIEW_ID:
            return 1
        return 3

    @property
    def requires_positive_pz(self) -> bool:
        """Whether the arm has the hard domain constraint ``pz > 0``."""

        return self.view_id != IDENTITY_CARTESIAN_VIEW_ID

    def manifest(self) -> Dict[str, Any]:
        """Return a JSON-serializable arm manifest."""

        if self.view_id == IDENTITY_CARTESIAN_VIEW_ID:
            forward_definition = ["px", "py", "pz", "x", "y"]
            inverse_definition = [
                "px = feature[0]",
                "py = feature[1]",
                "pz = feature[2]",
                "x = feature[3]",
                "y = feature[4]",
            ]
            hypothesis = "Untransformed Cartesian reference; no simplification assumed."
            domain = {"modelled_coordinates": "finite"}
            jacobian_definition = "0"
        elif self.view_id == CARTESIAN_LOGPZ_VIEW_ID:
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
            hypothesis = "Log longitudinal momentum may reduce scale asymmetry."
            domain = {"modelled_coordinates": "finite", "pz": "strictly_positive"}
            jacobian_definition = "-log(pz_numeric_in_GeV_per_c)"
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
            hypothesis = (
                "Slopes may simplify directional correlations, but can amplify "
                "small-pz behaviour."
            )
            domain = {"modelled_coordinates": "finite", "pz": "strictly_positive"}
            jacobian_definition = "-3 * log(pz_numeric_in_GeV_per_c)"

        return {
            "schema_version": FEATURE_VIEW_SCHEMA_VERSION,
            "feature_view_experiment_id": FEATURE_VIEW_EXPERIMENT_ID,
            "feature_view_id": self.view_id,
            "experiment_role": self.experiment_role,
            "hypothesis": hypothesis,
            "raw_contract_version": schema.CONTRACT_VERSION,
            "raw_columns": list(schema.COLUMNS),
            "physical_input_columns": list(PHYSICAL_STATE_COLUMNS),
            "feature_output_columns": list(self.feature_names),
            "excluded_raw_columns": ["z", "id", "w"],
            "pz_unit_gev": self.pz_unit_gev,
            "domain": domain,
            "forward_definition": forward_definition,
            "inverse_definition": inverse_definition,
            "physical_to_feature_log_abs_det_jacobian": jacobian_definition,
            "includes_train_fitted_standardization": False,
            "promotion_status": (
                "reference"
                if self.experiment_role == "baseline"
                else "unvalidated_candidate"
            ),
        }

    def config_hash(self) -> str:
        """Return a deterministic hash of :meth:`manifest`."""

        return _canonical_json_hash(self.manifest())

    def forward(self, raw_array: np.ndarray) -> np.ndarray:
        """Map raw rows into the configured five-dimensional arm."""

        physical = _coerce_raw_array(raw_array)
        px, py, pz, x_pos, y_pos = physical.T

        if self.requires_positive_pz and np.any(pz <= 0.0):
            bad_count = int(np.count_nonzero(pz <= 0.0))
            raise FeatureViewDomainError(
                "{} requires pz > 0; found {} invalid rows".format(
                    self.view_id, bad_count
                )
            )

        if self.view_id == IDENTITY_CARTESIAN_VIEW_ID:
            transformed = physical.copy()
        else:
            assert self.pz_unit_gev is not None
            log_pz = np.log(pz / self.pz_unit_gev)
            if self.view_id == CARTESIAN_LOGPZ_VIEW_ID:
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
        """Map feature rows back to physical ``[px, py, pz, x, y]`` rows."""

        feature_array = _coerce_feature_array(features)
        first, second, third, x_pos, y_pos = feature_array.T

        if self.view_id == IDENTITY_CARTESIAN_VIEW_ID:
            physical = feature_array.copy()
        else:
            assert self.pz_unit_gev is not None
            with np.errstate(over="ignore", invalid="ignore"):
                pz = self.pz_unit_gev * np.exp(third)
                if self.view_id == CARTESIAN_LOGPZ_VIEW_ID:
                    px = first
                    py = second
                else:
                    px = first * pz
                    py = second * pz
            physical = np.column_stack((px, py, pz, x_pos, y_pos))

        if not np.isfinite(physical).all():
            raise FeatureViewDomainError(
                "inverse feature transformation produced NaN or inf"
            )
        if self.requires_positive_pz and np.any(physical[:, 2] <= 0.0):
            raise FeatureViewDomainError(
                "inverse feature transformation left the pz > 0 domain"
            )
        return np.ascontiguousarray(physical, dtype=np.float64)

    def forward_log_abs_det_jacobian(self, raw_array: np.ndarray) -> np.ndarray:
        """Return ``log|det d(feature)/d(physical)|`` for each raw row."""

        physical = _coerce_raw_array(raw_array)
        if self.view_id == IDENTITY_CARTESIAN_VIEW_ID:
            return np.zeros(physical.shape[0], dtype=np.float64)

        pz = physical[:, 2]
        if np.any(pz <= 0.0):
            raise FeatureViewDomainError(
                "{} Jacobian requires pz > 0".format(self.view_id)
            )
        return -float(self.jacobian_pz_power) * np.log(pz)

    def inverse_log_abs_det_jacobian(self, features: np.ndarray) -> np.ndarray:
        """Return ``log|det d(physical)/d(feature)|`` for each feature row."""

        physical = self.inverse(features)
        if self.view_id == IDENTITY_CARTESIAN_VIEW_ID:
            return np.zeros(physical.shape[0], dtype=np.float64)
        pz = physical[:, 2]
        return float(self.jacobian_pz_power) * np.log(pz)

    def physical_log_prob_from_feature(
        self,
        feature_log_prob: np.ndarray,
        raw_array: np.ndarray,
    ) -> np.ndarray:
        """Convert feature-space density to physical-coordinate density.

        This adds only the deterministic feature-view Jacobian. A later
        train-fitted standardization layer must account for its own affine
        Jacobian separately.
        """

        physical = _coerce_raw_array(raw_array)
        values = _coerce_log_prob(feature_log_prob, physical.shape[0])
        return values + self.forward_log_abs_det_jacobian(raw_array)
