"""Diagonal and full-covariance Gaussian baselines (analytic MLE).

NumPy-only. Both expose the :class:`~Nflow.interfaces.DensityEstimator`
boundary: fit on ``(n, d)`` float64 arrays, exact ``log_prob``/``sample``,
explicit-parameter ``save``/``load`` (NPZ + JSON, never a pickled estimator),
and a versioned ``manifest``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from Nflow.interfaces import FIT_STATUS_OK, FitResult

_LOG_TWO_PI = float(np.log(2.0 * np.pi))


def _validate_fit_array(x: Any, dimension: int) -> np.ndarray:
    array = np.asarray(x, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != dimension:
        raise ValueError(
            "expected (n, {}) training array, got {}".format(dimension, array.shape)
        )
    if array.shape[0] < 2:
        raise ValueError("need at least 2 training rows")
    if not np.isfinite(array).all():
        raise ValueError("training array contains NaN or inf")
    return array


def _weighted_moments(x: np.ndarray, weights: Optional[np.ndarray]):
    if weights is None:
        mean = x.mean(axis=0)
        centered = x - mean
        return mean, centered, x.shape[0]
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (x.shape[0],) or not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("sample_weight must be finite, non-negative, shape (n,)")
    total = w.sum()
    if total <= 0:
        raise ValueError("sample_weight must have positive sum")
    mean = (w[:, None] * x).sum(axis=0) / total
    centered = x - mean
    return mean, centered, total


def _reject_unsupported_fit_arguments(family: str, **arguments: Any) -> None:
    for name, value in arguments.items():
        if value is not None:
            raise NotImplementedError(
                "{} does not support {}; pass None".format(family, name)
            )


class DiagonalGaussian:
    """Diagonal-covariance Gaussian with a configurable, recorded variance floor."""

    family = "diagonal_gaussian"

    def __init__(self, *, dimension: int, variance_floor: float = 1e-6) -> None:
        self.dimension = int(dimension)
        self.variance_floor = float(variance_floor)
        if not np.isfinite(self.variance_floor) or self.variance_floor <= 0:
            raise ValueError("variance_floor must be finite and > 0")
        self._mean: Optional[np.ndarray] = None
        self._variance: Optional[np.ndarray] = None

    def fit(
        self,
        x_train: np.ndarray,
        *,
        x_validation: Optional[np.ndarray] = None,
        seed: int = 0,
        sample_weight: Optional[np.ndarray] = None,
        validation_sample_weight: Optional[np.ndarray] = None,
        component_id: Optional[np.ndarray] = None,
        validation_component_id: Optional[np.ndarray] = None,
        rare_component_id: Optional[int] = None,
    ) -> FitResult:
        _reject_unsupported_fit_arguments(
            self.family,
            validation_sample_weight=validation_sample_weight,
            component_id=component_id,
            validation_component_id=validation_component_id,
            rare_component_id=rare_component_id,
        )
        start = time.perf_counter()
        x = _validate_fit_array(x_train, self.dimension)
        mean, centered, total = _weighted_moments(x, sample_weight)
        if sample_weight is None:
            variance = np.mean(centered**2, axis=0)
        else:
            w = np.asarray(sample_weight, dtype=np.float64)
            variance = (w[:, None] * centered**2).sum(axis=0) / total
        floored = np.maximum(variance, self.variance_floor)
        self._mean = mean
        self._variance = floored
        val_nll = None
        if x_validation is not None:
            val = _validate_fit_array(x_validation, self.dimension)
            val_nll = float(-np.mean(self.log_prob(val)))
        return FitResult(
            status=FIT_STATUS_OK,
            seed=int(seed),
            train_history=[{"step": 0, "train_nll": float(-np.mean(self.log_prob(x)))}],
            best_step=0,
            best_validation_nll=val_nll,
            wall_time_seconds=time.perf_counter() - start,
        )

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        if self._mean is None:
            raise RuntimeError("model is not fitted")
        array = np.asarray(x, dtype=np.float64)
        diff = array - self._mean
        return -0.5 * (
            self.dimension * _LOG_TWO_PI
            + np.sum(np.log(self._variance))
            + np.sum(diff**2 / self._variance, axis=1)
        )

    def sample(self, n: int, *, seed: int) -> np.ndarray:
        if self._mean is None:
            raise RuntimeError("model is not fitted")
        rng = np.random.default_rng(int(seed))
        eps = rng.standard_normal((int(n), self.dimension))
        return self._mean + eps * np.sqrt(self._variance)

    def parameter_count(self) -> int:
        return 2 * self.dimension

    def manifest(self) -> Dict[str, Any]:
        return {
            "family": self.family,
            "dimension": self.dimension,
            "variance_floor": self.variance_floor,
            "parameter_count": self.parameter_count(),
            "fitted": self._mean is not None,
        }

    def save(self, output_dir: Path) -> Dict[str, Any]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        params_path = output_dir / "model_parameters.npz"
        np.savez(params_path, mean=self._mean, variance=self._variance)
        config = {
            "family": self.family,
            "dimension": self.dimension,
            "variance_floor": self.variance_floor,
        }
        (output_dir / "model_config.json").write_text(json.dumps(config, indent=2))
        return {"family": self.family, "parameters_file": params_path.name}

    @classmethod
    def load(cls, input_dir: Path) -> "DiagonalGaussian":
        input_dir = Path(input_dir)
        config = json.loads((input_dir / "model_config.json").read_text())
        model = cls(
            dimension=config["dimension"], variance_floor=config["variance_floor"]
        )
        data = np.load(input_dir / "model_parameters.npz")
        model._mean = data["mean"]
        model._variance = data["variance"]
        return model


class FullGaussian:
    """Full-covariance Gaussian with configurable, recorded covariance regularization."""

    family = "full_gaussian"

    def __init__(
        self, *, dimension: int, covariance_regularization: float = 1e-6
    ) -> None:
        self.dimension = int(dimension)
        self.covariance_regularization = float(covariance_regularization)
        if (
            not np.isfinite(self.covariance_regularization)
            or self.covariance_regularization < 0
        ):
            raise ValueError("covariance_regularization must be finite and >= 0")
        self._mean: Optional[np.ndarray] = None
        self._covariance: Optional[np.ndarray] = None
        self._cholesky: Optional[np.ndarray] = None
        self._log_det: Optional[float] = None

    def _set_covariance(self, covariance: np.ndarray) -> None:
        cov = covariance + self.covariance_regularization * np.eye(self.dimension)
        chol = np.linalg.cholesky(cov)
        self._covariance = cov
        self._cholesky = chol
        self._log_det = 2.0 * float(np.sum(np.log(np.diag(chol))))

    def fit(
        self,
        x_train: np.ndarray,
        *,
        x_validation: Optional[np.ndarray] = None,
        seed: int = 0,
        sample_weight: Optional[np.ndarray] = None,
        validation_sample_weight: Optional[np.ndarray] = None,
        component_id: Optional[np.ndarray] = None,
        validation_component_id: Optional[np.ndarray] = None,
        rare_component_id: Optional[int] = None,
    ) -> FitResult:
        _reject_unsupported_fit_arguments(
            self.family,
            validation_sample_weight=validation_sample_weight,
            component_id=component_id,
            validation_component_id=validation_component_id,
            rare_component_id=rare_component_id,
        )
        start = time.perf_counter()
        x = _validate_fit_array(x_train, self.dimension)
        mean, centered, total = _weighted_moments(x, sample_weight)
        if sample_weight is None:
            covariance = centered.T @ centered / x.shape[0]
        else:
            w = np.asarray(sample_weight, dtype=np.float64)
            covariance = (centered * w[:, None]).T @ centered / total
        self._mean = mean
        self._set_covariance(covariance)
        val_nll = None
        if x_validation is not None:
            val = _validate_fit_array(x_validation, self.dimension)
            val_nll = float(-np.mean(self.log_prob(val)))
        return FitResult(
            status=FIT_STATUS_OK,
            seed=int(seed),
            train_history=[{"step": 0, "train_nll": float(-np.mean(self.log_prob(x)))}],
            best_step=0,
            best_validation_nll=val_nll,
            wall_time_seconds=time.perf_counter() - start,
        )

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        if self._mean is None:
            raise RuntimeError("model is not fitted")
        array = np.asarray(x, dtype=np.float64)
        diff = array - self._mean
        whitened = np.linalg.solve(self._cholesky, diff.T)
        mahalanobis = np.sum(whitened**2, axis=0)
        return -0.5 * (
            self.dimension * _LOG_TWO_PI + self._log_det + mahalanobis
        )

    def sample(self, n: int, *, seed: int) -> np.ndarray:
        if self._mean is None:
            raise RuntimeError("model is not fitted")
        rng = np.random.default_rng(int(seed))
        eps = rng.standard_normal((int(n), self.dimension))
        return self._mean + eps @ self._cholesky.T

    def parameter_count(self) -> int:
        return self.dimension + self.dimension * (self.dimension + 1) // 2

    def manifest(self) -> Dict[str, Any]:
        return {
            "family": self.family,
            "dimension": self.dimension,
            "covariance_regularization": self.covariance_regularization,
            "parameter_count": self.parameter_count(),
            "fitted": self._mean is not None,
        }

    def save(self, output_dir: Path) -> Dict[str, Any]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        params_path = output_dir / "model_parameters.npz"
        np.savez(params_path, mean=self._mean, covariance=self._covariance)
        config = {
            "family": self.family,
            "dimension": self.dimension,
            "covariance_regularization": self.covariance_regularization,
        }
        (output_dir / "model_config.json").write_text(json.dumps(config, indent=2))
        return {"family": self.family, "parameters_file": params_path.name}

    @classmethod
    def load(cls, input_dir: Path) -> "FullGaussian":
        input_dir = Path(input_dir)
        config = json.loads((input_dir / "model_config.json").read_text())
        model = cls(
            dimension=config["dimension"],
            covariance_regularization=config["covariance_regularization"],
        )
        data = np.load(input_dir / "model_parameters.npz")
        model._mean = data["mean"]
        # regularization already baked into the saved covariance; restore chol.
        cov = data["covariance"]
        model._covariance = cov
        model._cholesky = np.linalg.cholesky(cov)
        model._log_det = 2.0 * float(np.sum(np.log(np.diag(model._cholesky))))
        return model
