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
            return self._quantile_forward_log_abs_det_jacobian(raw_data)
        elif self.variant_id == "cartesian_log1p_pz_v0":
            pz = raw_data[:, 2]
            view_jac = -np.log(1.0 + pz)
            std_jac = -np.sum(np.log(self.std))
            return std_jac + view_jac
        else:
            raise ValueError(f"Unknown variant: {self.variant_id}")

    def _quantile_forward_log_abs_det_jacobian(self, raw_data: np.ndarray) -> np.ndarray:
        """Exact log-abs-det-Jacobian of the fitted QuantileTransformer's
        feature-wise map x -> z = norm.ppf(interp(x, quantiles_, references_)).

        QuantileTransformer acts independently per feature via a monotone
        piecewise-linear interpolant between the fitted quantile knots
        (``quantiles_``) and reference quantile levels (``references_``),
        followed by the standard-normal inverse CDF. This is analytically
        exact for that interpolant -- not an empirical/finite-difference
        approximation -- but only where the interpolant is genuinely
        piecewise-linear: rows outside the fitted knot range, on a
        zero-width knot interval (duplicate training values), or close
        enough to the outer knots that sklearn's internal probability
        clipping (before ``norm.ppf``) makes the map locally
        non-differentiable, are rejected via ``ValueError`` rather than
        silently misreported.
        """
        from scipy.stats import norm

        features = raw_data[:, :5]
        n = features.shape[0]
        quantiles = self.qt.quantiles_
        references = self.qt.references_
        z = self.transform(raw_data)

        total = np.zeros(n)
        for j in range(features.shape[1]):
            x_col = features[:, j]
            knots_x = quantiles[:, j]
            lo, hi = knots_x[0], knots_x[-1]
            out_of_domain = (x_col < lo) | (x_col > hi)
            if np.any(out_of_domain):
                raise ValueError(
                    "quantile_normal_v0 Jacobian is undefined for {} row(s) in "
                    "feature index {} outside the fitted quantile range "
                    "[{}, {}]".format(int(np.count_nonzero(out_of_domain)), j, lo, hi)
                )

            idx = np.searchsorted(knots_x, x_col, side="right") - 1
            idx = np.clip(idx, 0, len(knots_x) - 2)
            x0, x1 = knots_x[idx], knots_x[idx + 1]
            u0, u1 = references[idx], references[idx + 1]
            dx = x1 - x0
            if np.any(dx <= 0):
                raise ValueError(
                    "quantile_normal_v0 Jacobian is undefined for feature "
                    "index {}: fitted quantile knots contain a zero-width "
                    "interval (duplicate values in the training data)".format(j)
                )
            u_raw = np.clip(u0 + (u1 - u0) * (x_col - x0) / dx, 0.0, 1.0)

            # Detect sklearn's internal boundary clipping of u before
            # norm.ppf (distinct from the x-range check above): if the
            # unclipped z we'd compute here doesn't match what qt.transform()
            # actually returned, this row was clipped internally and this
            # closed-form Jacobian does not account for that.
            z_expected = norm.ppf(u_raw)
            mismatched = ~np.isclose(z_expected, z[:, j], atol=1e-6, rtol=1e-6)
            if np.any(mismatched):
                raise ValueError(
                    "quantile_normal_v0 Jacobian is undefined for {} row(s) "
                    "in feature index {}: too close to the outer quantile "
                    "knots, where QuantileTransformer's internal probability "
                    "clipping makes the map locally non-differentiable"
                    .format(int(np.count_nonzero(mismatched)), j)
                )

            log_slope = np.log((u1 - u0) / dx)
            log_norm_term = 0.5 * np.log(2.0 * np.pi) + 0.5 * z[:, j] ** 2
            total += log_slope + log_norm_term

        return total

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
