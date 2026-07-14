"""Train-only per-view feature pipeline.

A :class:`FittedFeaturePipeline` combines an existing
:class:`~ship_muon_bg.data_contracts.FeatureView` with a five-dimensional
standardization fitted **only on train rows**. It does not reuse the existing
six-dimensional raw normalization helper (which includes ``z``): the density
lab models exactly the five modelled coordinates.

Density accounting (physical-space likelihood):

    physical_log_prob
        = normalized_model_log_prob
        + normalization forward log-Jacobian   (sum(-log std))
        + FeatureView forward log-Jacobian      (view.forward_log_abs_det_jacobian)

All fitted arrays are immutable; hashes are deterministic. Pure NumPy.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict

import numpy as np

from ..data_contracts import schema
from ..data_contracts.feature_views import FeatureView, N_DENSITY_FEATURES

PIPELINE_SCHEMA_VERSION = "0"
_PZ_RAW_INDEX = schema.COLUMN_INDEX["pz"]
_PZ_PHYSICAL_INDEX = 2


class FeaturePipelineError(ValueError):
    """A feature-pipeline configuration or usage error."""


def _canonical_json_hash(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class FittedFeaturePipeline:
    """View + train-fitted 5D standardization, with exact density accounting."""

    def __init__(
        self,
        *,
        feature_view: FeatureView,
        mean: np.ndarray,
        std: np.ndarray,
        n_train_rows: int,
        zero_variance_policy: str,
    ) -> None:
        self.feature_view = feature_view
        mean = np.array(mean, dtype=np.float64, copy=True)
        std = np.array(std, dtype=np.float64, copy=True)
        if mean.shape != (N_DENSITY_FEATURES,) or std.shape != (N_DENSITY_FEATURES,):
            raise FeaturePipelineError("mean/std must have shape (5,)")
        mean.flags.writeable = False
        std.flags.writeable = False
        self._mean = mean
        self._std = std
        self.n_train_rows = int(n_train_rows)
        self.zero_variance_policy = zero_variance_policy
        # normalization forward log-Jacobian is constant per row: sum(-log std)
        self._norm_forward_logjac = float(-np.sum(np.log(std)))

    # -- construction --------------------------------------------------------

    @classmethod
    def fit(
        cls,
        raw_train: np.ndarray,
        feature_view: FeatureView,
        *,
        zero_variance_policy: str = "error",
    ) -> "FittedFeaturePipeline":
        """Fit standardization on the training rows of one feature-view arm.

        ``zero_variance_policy`` is ``"error"`` (raise on a zero-variance
        feature) or ``"unit"`` (documented safe fallback: use std = 1 for that
        feature, recorded in the manifest).
        """

        if zero_variance_policy not in ("error", "unit"):
            raise FeaturePipelineError(
                "zero_variance_policy must be 'error' or 'unit'"
            )
        features = feature_view.forward(raw_train)  # validates + applies view
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        zero_mask = std <= 0.0
        if np.any(zero_mask):
            if zero_variance_policy == "error":
                bad = [
                    feature_view.feature_names[i]
                    for i in np.flatnonzero(zero_mask)
                ]
                raise FeaturePipelineError(
                    "zero-variance train features {} under policy 'error'".format(bad)
                )
            std = np.where(zero_mask, 1.0, std)
        return cls(
            feature_view=feature_view,
            mean=mean,
            std=std,
            n_train_rows=int(features.shape[0]),
            zero_variance_policy=zero_variance_policy,
        )

    # -- transforms ----------------------------------------------------------

    def transform_raw(self, raw: np.ndarray) -> np.ndarray:
        """Raw rows -> standardized model features (N, 5)."""

        features = self.feature_view.forward(raw)
        return (features - self._mean) / self._std

    def inverse_to_physical(self, normalized_features: np.ndarray) -> np.ndarray:
        """Standardized features -> physical [px, py, pz, x, y] rows."""

        array = np.asarray(normalized_features, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != N_DENSITY_FEATURES:
            raise FeaturePipelineError("expected normalized features shape (N, 5)")
        features = array * self._std + self._mean
        return self.feature_view.inverse(features)

    # -- density accounting --------------------------------------------------

    def _view_forward_logjac(self, pz: np.ndarray) -> np.ndarray:
        power = self.feature_view.jacobian_pz_power
        if power == 0:
            return np.zeros(pz.shape[0], dtype=np.float64)
        if np.any(pz <= 0.0):
            raise FeaturePipelineError("feature-view Jacobian requires pz > 0")
        return -float(power) * np.log(pz)

    def normalized_to_physical_log_prob(
        self, normalized_log_prob: np.ndarray, raw_or_physical: np.ndarray
    ) -> np.ndarray:
        """Convert normalized-space log density to physical-space log density.

        ``raw_or_physical`` may be raw ``(N, 8)`` rows or physical ``(N, 5)``
        rows; only ``pz`` is needed for the feature-view Jacobian.
        """

        values = np.asarray(normalized_log_prob, dtype=np.float64)
        rows = np.asarray(raw_or_physical, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] not in (schema.N_COLUMNS, N_DENSITY_FEATURES):
            raise FeaturePipelineError(
                "raw_or_physical must be (N, 8) raw or (N, 5) physical rows"
            )
        pz_index = _PZ_RAW_INDEX if rows.shape[1] == schema.N_COLUMNS else _PZ_PHYSICAL_INDEX
        pz = rows[:, pz_index]
        if values.shape != (rows.shape[0],):
            raise FeaturePipelineError("normalized_log_prob must have shape (N,)")
        return values + self._norm_forward_logjac + self._view_forward_logjac(pz)

    # -- provenance ----------------------------------------------------------

    @property
    def dimension(self) -> int:
        return N_DENSITY_FEATURES

    def manifest(self) -> Dict[str, Any]:
        return {
            "schema_version": PIPELINE_SCHEMA_VERSION,
            "feature_view_id": self.feature_view.view_id,
            "feature_view_config_hash": self.feature_view.config_hash(),
            "feature_names": list(self.feature_view.feature_names),
            "standardization": {
                "fit_on": "train",
                "n_train_rows": self.n_train_rows,
                "mean": self._mean.tolist(),
                "std": self._std.tolist(),
                "zero_variance_policy": self.zero_variance_policy,
            },
            "normalization_forward_log_jacobian": self._norm_forward_logjac,
            "density_accounting": (
                "physical = normalized + sum(-log std) + view_forward_log_jac"
            ),
        }

    def config_hash(self) -> str:
        return _canonical_json_hash(self.manifest())
