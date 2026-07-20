"""Gaussian-mixture baseline behind the common estimator interface.

Fitting uses ``sklearn.mixture.GaussianMixture`` (an optional ``lab``
dependency, imported lazily only when ``fit`` runs). The canonical artifact is
NOT a pickled sklearn estimator: explicit weights, means and covariances are
written to NPZ + JSON and the inference wrapper is reconstructed from those,
so densities and samples are computed by NumPy code we control.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from Nflow.interfaces import FIT_STATUS_FAILED, FIT_STATUS_OK, FitResult

_LOG_TWO_PI = float(np.log(2.0 * np.pi))


class GaussianMixtureEstimator:
    """Full-covariance Gaussian mixture (analytic inference from saved params)."""

    family = "gaussian_mixture"

    def __init__(
        self,
        *,
        dimension: int,
        n_components: int = 2,
        covariance_regularization: float = 1e-6,
        n_init: int = 1,
        max_iter: int = 200,
    ) -> None:
        self.dimension = int(dimension)
        self.n_components = int(n_components)
        if self.n_components < 1:
            raise ValueError("n_components must be >= 1")
        self.covariance_regularization = float(covariance_regularization)
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self._weights: Optional[np.ndarray] = None
        self._means: Optional[np.ndarray] = None
        self._covariances: Optional[np.ndarray] = None
        self._cholesky: Optional[np.ndarray] = None
        self._log_det: Optional[np.ndarray] = None
        self._converged: Optional[bool] = None

    def _prepare_inference(self) -> None:
        self._cholesky = np.linalg.cholesky(self._covariances)
        self._log_det = 2.0 * np.sum(
            np.log(np.diagonal(self._cholesky, axis1=1, axis2=2)), axis=1
        )

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
        for name, value in (
            ("sample_weight", sample_weight),
            ("validation_sample_weight", validation_sample_weight),
            ("component_id", component_id),
            ("validation_component_id", validation_component_id),
            ("rare_component_id", rare_component_id),
        ):
            if value is not None:
                raise NotImplementedError(
                    "{} does not support {}; pass None".format(self.family, name)
                )
        start = time.perf_counter()
        x = np.asarray(x_train, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.dimension:
            raise ValueError(
                "expected (n, {}) training array, got {}".format(
                    self.dimension, x.shape
                )
            )
        from sklearn.mixture import GaussianMixture  # lazy optional import

        gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type="full",
            reg_covar=self.covariance_regularization,
            n_init=self.n_init,
            max_iter=self.max_iter,
            random_state=int(seed),
        )
        warnings_list = []
        try:
            gmm.fit(x)
        except Exception as exc:  # pragma: no cover - defensive
            return FitResult(
                status=FIT_STATUS_FAILED,
                seed=int(seed),
                warnings=["sklearn GaussianMixture.fit raised: {}".format(exc)],
                wall_time_seconds=time.perf_counter() - start,
            )
        self._weights = np.asarray(gmm.weights_, dtype=np.float64)
        self._means = np.asarray(gmm.means_, dtype=np.float64)
        self._covariances = np.asarray(gmm.covariances_, dtype=np.float64)
        self._converged = bool(gmm.converged_)
        if not self._converged:
            warnings_list.append("sklearn GaussianMixture did not converge")
        self._prepare_inference()
        val_nll = None
        if x_validation is not None:
            val = np.asarray(x_validation, dtype=np.float64)
            val_nll = float(-np.mean(self.log_prob(val)))
        return FitResult(
            status=FIT_STATUS_OK,
            seed=int(seed),
            train_history=[
                {
                    "step": int(gmm.n_iter_),
                    "train_nll": float(-np.mean(self.log_prob(x))),
                    "converged": self._converged,
                }
            ],
            best_step=int(gmm.n_iter_),
            best_validation_nll=val_nll,
            wall_time_seconds=time.perf_counter() - start,
            warnings=warnings_list,
        )

    def _component_log_prob(self, x: np.ndarray) -> np.ndarray:
        # returns (n, k): log(weight_k) + log N_k(x)
        n = x.shape[0]
        out = np.empty((n, self.n_components), dtype=np.float64)
        for k in range(self.n_components):
            diff = x - self._means[k]
            whitened = np.linalg.solve(self._cholesky[k], diff.T)
            mahalanobis = np.sum(whitened**2, axis=0)
            out[:, k] = np.log(self._weights[k]) - 0.5 * (
                self.dimension * _LOG_TWO_PI + self._log_det[k] + mahalanobis
            )
        return out

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        if self._weights is None:
            raise RuntimeError("model is not fitted")
        array = np.asarray(x, dtype=np.float64)
        return np.logaddexp.reduce(self._component_log_prob(array), axis=1)

    def sample(self, n: int, *, seed: int) -> np.ndarray:
        if self._weights is None:
            raise RuntimeError("model is not fitted")
        rng = np.random.default_rng(int(seed))
        n = int(n)
        labels = rng.choice(self.n_components, size=n, p=self._weights)
        out = np.empty((n, self.dimension), dtype=np.float64)
        for k in range(self.n_components):
            mask = labels == k
            count = int(np.count_nonzero(mask))
            if count:
                eps = rng.standard_normal((count, self.dimension))
                out[mask] = self._means[k] + eps @ self._cholesky[k].T
        return out

    def parameter_count(self) -> int:
        per_cov = self.dimension * (self.dimension + 1) // 2
        return self.n_components * (1 + self.dimension + per_cov) - 1

    def manifest(self) -> Dict[str, Any]:
        return {
            "family": self.family,
            "dimension": self.dimension,
            "n_components": self.n_components,
            "covariance_regularization": self.covariance_regularization,
            "n_init": self.n_init,
            "max_iter": self.max_iter,
            "converged": self._converged,
            "parameter_count": self.parameter_count(),
            "fitted": self._weights is not None,
        }

    def save(self, output_dir: Path) -> Dict[str, Any]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        params_path = output_dir / "model_parameters.npz"
        np.savez(
            params_path,
            weights=self._weights,
            means=self._means,
            covariances=self._covariances,
        )
        config = {
            "family": self.family,
            "dimension": self.dimension,
            "n_components": self.n_components,
            "covariance_regularization": self.covariance_regularization,
            "n_init": self.n_init,
            "max_iter": self.max_iter,
            "converged": self._converged,
        }
        (output_dir / "model_config.json").write_text(json.dumps(config, indent=2))
        return {"family": self.family, "parameters_file": params_path.name}

    @classmethod
    def load(cls, input_dir: Path) -> "GaussianMixtureEstimator":
        input_dir = Path(input_dir)
        config = json.loads((input_dir / "model_config.json").read_text())
        model = cls(
            dimension=config["dimension"],
            n_components=config["n_components"],
            covariance_regularization=config["covariance_regularization"],
            n_init=config["n_init"],
            max_iter=config["max_iter"],
        )
        data = np.load(input_dir / "model_parameters.npz")
        model._weights = data["weights"]
        model._means = data["means"]
        model._covariances = data["covariances"]
        model._converged = config.get("converged")
        model._prepare_inference()
        return model
