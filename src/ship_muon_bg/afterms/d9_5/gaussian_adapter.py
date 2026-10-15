"""D9-5 Gaussian control adapters: GAUSS_DIAG_d05 / GAUSS_FULL_d05 (Sec 9).

Deterministic, single-fit controls. The model itself is fit in the same
standardized (``identity_standardized_v0``) feature space every other D9-5
family uses (Sec 7); ``validation_log_prob``/``test_log_prob`` return
feature-space log p(z) only -- the shared evaluation layer adds the
preprocessing Jacobian once (Sec 8/Sec 13), so no family can silently
double-count or drop that constant.

Sufficient statistics are computed by a two-pass, chunk-bounded streaming
accumulator (Sec 9.1/9.2) rather than one ``np.mean``/covariance call over the
whole array, and are algebraically identical to
``Nflow.baselines.gaussian.{DiagonalGaussian,FullGaussian}.fit`` on the same
rows (required tests 13/14) -- both paths route through the same
``variance_floor``/``covariance_regularization`` + Cholesky logic via
``from_moments`` so there is only one copy of that regularization formula.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from Nflow.baselines.gaussian import DiagonalGaussian, FullGaussian
from ship_muon_bg.afterms.preprocessing import PreprocessingPipeline

from . import model_adapter as ma

DEFAULT_CHUNK_ROWS = 200_000


def _chunks(features: np.ndarray, chunk_rows: int):
    n = features.shape[0]
    for start in range(0, n, chunk_rows):
        yield features[start:start + chunk_rows]


def streaming_diagonal_statistics(
    features: np.ndarray, *, chunk_rows: int = DEFAULT_CHUNK_ROWS
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Two-pass, chunk-bounded (mean, variance, n) -- pass 1 accumulates the
    sum for the mean, pass 2 accumulates centered sum-of-squares from that
    exact mean. Equivalent to ``variance = mean(centered**2, axis=0)``."""

    dimension = features.shape[1]
    total_n = 0
    sum_x = np.zeros(dimension, dtype=np.float64)
    for chunk in _chunks(features, chunk_rows):
        sum_x += chunk.sum(axis=0, dtype=np.float64)
        total_n += chunk.shape[0]
    if total_n == 0:
        raise ValueError("streaming_diagonal_statistics: no rows to fit")
    mean = sum_x / total_n

    sum_sq_centered = np.zeros(dimension, dtype=np.float64)
    for chunk in _chunks(features, chunk_rows):
        centered = chunk - mean
        sum_sq_centered += np.sum(centered ** 2, axis=0)
    variance = sum_sq_centered / total_n
    return mean, variance, total_n


def streaming_full_statistics(
    features: np.ndarray, *, chunk_rows: int = DEFAULT_CHUNK_ROWS
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Two-pass, chunk-bounded (mean, covariance, n). Equivalent to
    ``covariance = centered.T @ centered / n``."""

    dimension = features.shape[1]
    total_n = 0
    sum_x = np.zeros(dimension, dtype=np.float64)
    for chunk in _chunks(features, chunk_rows):
        sum_x += chunk.sum(axis=0, dtype=np.float64)
        total_n += chunk.shape[0]
    if total_n == 0:
        raise ValueError("streaming_full_statistics: no rows to fit")
    mean = sum_x / total_n

    cov_sum = np.zeros((dimension, dimension), dtype=np.float64)
    for chunk in _chunks(features, chunk_rows):
        centered = chunk - mean
        cov_sum += centered.T @ centered
    covariance = cov_sum / total_n
    return mean, covariance, total_n


class _BaseGaussianAdapter(ma.ModelAdapter):
    _estimator_cls = None  # set by subclass

    def __init__(self, **estimator_kwargs: Any) -> None:
        self._estimator_kwargs = dict(estimator_kwargs)
        self.pipeline: PreprocessingPipeline = None
        self._estimator = None
        self._fitting_log: Dict[str, Any] = None
        self._validation_metrics: Dict[str, Any] = None

    def _standardize(self, raw: np.ndarray) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("model is not fitted")
        return self.pipeline.transform(raw)

    def validation_log_prob(self, validation_raw: np.ndarray) -> np.ndarray:
        return self._estimator.log_prob(self._standardize(validation_raw))

    def test_log_prob(self, test_raw: np.ndarray) -> np.ndarray:
        return self._estimator.log_prob(self._standardize(test_raw))

    def sample(self, n: int, *, seed: int) -> np.ndarray:
        standardized = self._estimator.sample(int(n), seed=int(seed))
        return self.pipeline.inverse(standardized)

    def capability_metadata(self) -> Dict[str, bool]:
        capabilities = {
            "supports_exact_log_prob": True,
            "supports_physical_log_prob": True,
            "stochastic_fit": False,
            "supports_cuda": False,
            "supports_resume": False,
            "deterministic_sampling_given_seed": True,
        }
        ma.assert_capability_metadata(capabilities)
        return capabilities

    def model_metadata(self) -> Dict[str, Any]:
        return {
            "model_family_id": self.model_family_id,
            "model_config_id": self.model_config_id,
            "estimator_manifest": self._estimator.manifest(),
            "preprocessing_name": self.pipeline.variant_id if self.pipeline is not None else None,
        }

    def save_bundle(self, output_dir: Path) -> Dict[str, Any]:
        output_dir = Path(output_dir)
        self._estimator.save(output_dir)
        ma.atomic_write_json(output_dir / "metadata.json", self.model_metadata())
        ma.atomic_write_json(output_dir / "preprocessing.json", self.pipeline.to_dict())
        ma.atomic_write_json(output_dir / "fitting_log.json", self._fitting_log)
        if self._validation_metrics is not None:
            ma.atomic_write_json(output_dir / "validation_metrics.json", self._validation_metrics)
        return {"output_dir": str(output_dir)}

    @classmethod
    def load_bundle(cls, input_dir: Path) -> "_BaseGaussianAdapter":
        input_dir = Path(input_dir)
        estimator = cls._estimator_cls.load(input_dir)
        adapter = cls()
        adapter._estimator = estimator
        adapter.pipeline = PreprocessingPipeline.from_dict(ma.read_json(input_dir / "preprocessing.json"))
        adapter._fitting_log = ma.read_json(input_dir / "fitting_log.json")
        return adapter

    def set_validation_metrics(self, metrics: Dict[str, Any]) -> None:
        self._validation_metrics = metrics


class DiagonalGaussianAdapter(_BaseGaussianAdapter):
    model_family_id = "GAUSS_DIAG"
    model_config_id = "GAUSS_DIAG_d05"
    _estimator_cls = DiagonalGaussian

    def __init__(self, *, variance_floor: float = 1e-6) -> None:
        super().__init__(variance_floor=variance_floor)
        self.variance_floor = variance_floor

    def fit(self, train_raw: np.ndarray, validation_raw: np.ndarray, *, seed: int = 0) -> Dict[str, Any]:
        start = time.perf_counter()
        self.pipeline = PreprocessingPipeline("identity_standardized_v0", seed=seed)
        self.pipeline.fit(train_raw)
        standardized = self.pipeline.transform(train_raw)
        mean, variance, n = streaming_diagonal_statistics(standardized)
        self._estimator = DiagonalGaussian.from_moments(
            dimension=5, mean=mean, variance=variance, variance_floor=self.variance_floor,
        )
        validation_feature_nll = float(-np.mean(self.validation_log_prob(validation_raw)))
        self._fitting_log = {
            "model_config_id": self.model_config_id,
            "fitting_policy": "deterministic_single_fit",
            "sufficient_statistics_method": "two_pass_streaming",
            "train_row_count": int(n),
            "wall_time_seconds": time.perf_counter() - start,
            "variance_floor": self.variance_floor,
            "validation_feature_nll": validation_feature_nll,
        }
        return self._fitting_log


class FullGaussianAdapter(_BaseGaussianAdapter):
    model_family_id = "GAUSS_FULL"
    model_config_id = "GAUSS_FULL_d05"
    _estimator_cls = FullGaussian

    def __init__(self, *, covariance_regularization: float = 1e-6) -> None:
        super().__init__(covariance_regularization=covariance_regularization)
        self.covariance_regularization = covariance_regularization

    def fit(self, train_raw: np.ndarray, validation_raw: np.ndarray, *, seed: int = 0) -> Dict[str, Any]:
        start = time.perf_counter()
        self.pipeline = PreprocessingPipeline("identity_standardized_v0", seed=seed)
        self.pipeline.fit(train_raw)
        standardized = self.pipeline.transform(train_raw)
        mean, covariance, n = streaming_full_statistics(standardized)

        pre_reg_eigenvalues = np.linalg.eigvalsh(covariance)
        pre_reg_condition_number = (
            float(pre_reg_eigenvalues[-1] / pre_reg_eigenvalues[0]) if pre_reg_eigenvalues[0] > 0 else float("inf")
        )

        self._estimator = FullGaussian.from_moments(
            dimension=5, mean=mean, covariance=covariance,
            covariance_regularization=self.covariance_regularization,
        )
        post_reg_eigenvalues = np.linalg.eigvalsh(self._estimator._covariance)
        cholesky_success = True
        try:
            np.linalg.cholesky(self._estimator._covariance)
        except np.linalg.LinAlgError:
            cholesky_success = False

        validation_feature_nll = float(-np.mean(self.validation_log_prob(validation_raw)))
        self._fitting_log = {
            "model_config_id": self.model_config_id,
            "fitting_policy": "deterministic_single_fit",
            "sufficient_statistics_method": "two_pass_streaming",
            "train_row_count": int(n),
            "wall_time_seconds": time.perf_counter() - start,
            "covariance_regularization": self.covariance_regularization,
            "pre_regularization_eigenvalues": pre_reg_eigenvalues.tolist(),
            "pre_regularization_condition_number": pre_reg_condition_number,
            "post_regularization_eigenvalues": post_reg_eigenvalues.tolist(),
            "cholesky_success": cholesky_success,
            "validation_feature_nll": validation_feature_nll,
        }
        return self._fitting_log
