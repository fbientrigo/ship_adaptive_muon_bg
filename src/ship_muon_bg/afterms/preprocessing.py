"""Phase C: Preprocessing A/B variants (§6).

Implements three preprocessing variants:
1. identity_standardized_v0
2. quantile_normal_v0
3. cartesian_log1p_pz_v0
"""

from __future__ import annotations

import pickle
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from ship_muon_bg.afterms.log1p_pz import (
    transform_rows,
    inverse_transform_rows,
    forward_log_abs_det_jacobian,
    validate_pz_domain,
)

class PreprocessingPipeline:
    def __init__(self, variant_id: str, seed: int = 20260720):
        self.variant_id = variant_id
        self.seed = seed
        self.mean = None
        self.std = None
        self.qt = None

    def fit(self, raw_train: np.ndarray) -> PreprocessingPipeline:
        # raw_train is (N, 8) with columns: px, py, pz, x, y, z, id, w
        # We model the 5 features px, py, pz, x, y
        features = raw_train[:, :5]

        # Domain checks for negative pz
        pz_col = features[:, 2]
        negative_count, _ = validate_pz_domain(pz_col, raise_on_negative=False)
        if negative_count > 0 and self.variant_id == "cartesian_log1p_pz_v0":
            # For cartesian_log1p_pz_v0, we must raise if negative pz rows exist
            # but allow the caller to catch it (represented as blocked).
            validate_pz_domain(pz_col, raise_on_negative=True)

        if self.variant_id == "identity_standardized_v0":
            self.mean = np.mean(features, axis=0)
            self.std = np.std(features, axis=0)
            self.std = np.where(self.std <= 0.0, 1.0, self.std)

        elif self.variant_id == "quantile_normal_v0":
            self.qt = QuantileTransformer(
                output_distribution="normal",
                random_state=self.seed,
                subsample=None,
            )
            self.qt.fit(features)

        elif self.variant_id == "cartesian_log1p_pz_v0":
            transformed = transform_rows(features, s_pz=1.0)
            self.mean = np.mean(transformed, axis=0)
            self.std = np.std(transformed, axis=0)
            self.std = np.where(self.std <= 0.0, 1.0, self.std)

        else:
            raise ValueError(f"Unknown preprocessing variant: {self.variant_id}")

        return self

    def transform(self, raw_data: np.ndarray) -> np.ndarray:
        features = raw_data[:, :5]
        if self.variant_id == "identity_standardized_v0":
            return (features - self.mean) / self.std
        elif self.variant_id == "quantile_normal_v0":
            return self.qt.transform(features)
        elif self.variant_id == "cartesian_log1p_pz_v0":
            transformed = transform_rows(features, s_pz=1.0)
            return (transformed - self.mean) / self.std
        else:
            raise ValueError(f"Unknown variant: {self.variant_id}")

    def inverse(self, transformed_data: np.ndarray) -> np.ndarray:
        if self.variant_id == "identity_standardized_v0":
            features = transformed_data * self.std + self.mean
            return features
        elif self.variant_id == "quantile_normal_v0":
            return self.qt.inverse_transform(transformed_data)
        elif self.variant_id == "cartesian_log1p_pz_v0":
            features = transformed_data * self.std + self.mean
            return inverse_transform_rows(features, s_pz=1.0)
        else:
            raise ValueError(f"Unknown variant: {self.variant_id}")

    def forward_log_abs_det_jacobian(self, raw_data: np.ndarray) -> np.ndarray:
        n = raw_data.shape[0]
        if self.variant_id == "identity_standardized_v0":
            return np.full(n, -np.sum(np.log(self.std)))
        elif self.variant_id == "quantile_normal_v0":
            raise NotImplementedError("QuantileTransformer does not support an exact analytical Jacobian.")
        elif self.variant_id == "cartesian_log1p_pz_v0":
            pz = raw_data[:, 2]
            view_jac = -np.log(1.0 + pz)
            std_jac = -np.sum(np.log(self.std))
            return std_jac + view_jac
        else:
            raise ValueError(f"Unknown variant: {self.variant_id}")

    def to_dict(self) -> dict:
        state = {
            "variant_id": self.variant_id,
            "seed": self.seed,
        }
        if self.variant_id in ("identity_standardized_v0", "cartesian_log1p_pz_v0"):
            state["mean"] = self.mean.tolist() if self.mean is not None else None
            state["std"] = self.std.tolist() if self.std is not None else None
        elif self.variant_id == "quantile_normal_v0":
            # Knots can be serialized via pickle bytes or similar
            state["qt_pkl"] = pickle.dumps(self.qt).hex() if self.qt is not None else None
        return state

    @classmethod
    def from_dict(cls, state: dict) -> PreprocessingPipeline:
        obj = cls(state["variant_id"], state["seed"])
        if obj.variant_id in ("identity_standardized_v0", "cartesian_log1p_pz_v0"):
            obj.mean = np.array(state["mean"]) if state.get("mean") is not None else None
            obj.std = np.array(state["std"]) if state.get("std") is not None else None
        elif obj.variant_id == "quantile_normal_v0":
            obj.qt = pickle.loads(bytes.fromhex(state["qt_pkl"])) if state.get("qt_pkl") is not None else None
        return obj
