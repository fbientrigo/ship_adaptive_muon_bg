"""D9-5 GMM control adapter: GMM_k04_covFULL_d05 (Sec 10).

One adapter instance = one (track, seed) fit, matching the NF_AC adapter's
per-seed granularity so multi-seed aggregation (Sec 12) treats both
stochastic families the same way. Drives ``sklearn.mixture.GaussianMixture``
one EM step at a time via ``warm_start`` to record a genuine per-iteration
lower-bound curve, then wraps the fitted result behind
``Nflow.baselines.gmm.GaussianMixtureEstimator`` (``from_fitted_sklearn``) for
log_prob/sample/save/load -- EM itself is never reimplemented.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from Nflow.baselines.gmm import GaussianMixtureEstimator
from ship_muon_bg.afterms.preprocessing import PreprocessingPipeline

from . import model_adapter as ma

DEFAULT_N_COMPONENTS = 4
DEFAULT_COVARIANCE_TYPE = "full"
DEFAULT_COVARIANCE_REGULARIZATION = 1e-6
DEFAULT_N_INIT = 1
DEFAULT_MAX_ITER = 200
DEFAULT_TOL = 1e-3
DEFAULT_OCCUPIED_WEIGHT_THRESHOLD = 0.01


class GmmAdapter(ma.ModelAdapter):
    model_family_id = "GMM"
    model_config_id = "GMM_k04_covFULL_d05"

    def __init__(
        self,
        *,
        n_components: int = DEFAULT_N_COMPONENTS,
        covariance_type: str = DEFAULT_COVARIANCE_TYPE,
        covariance_regularization: float = DEFAULT_COVARIANCE_REGULARIZATION,
        n_init: int = DEFAULT_N_INIT,
        max_iter: int = DEFAULT_MAX_ITER,
        tol: float = DEFAULT_TOL,
        occupied_weight_threshold: float = DEFAULT_OCCUPIED_WEIGHT_THRESHOLD,
    ) -> None:
        if covariance_type != "full":
            raise ValueError("D9-5 GMM control is frozen to covariance_type='full'")
        self.n_components = int(n_components)
        self.covariance_type = covariance_type
        self.covariance_regularization = float(covariance_regularization)
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.occupied_weight_threshold = float(occupied_weight_threshold)
        self.pipeline: PreprocessingPipeline = None
        self._estimator: GaussianMixtureEstimator = None
        self._fitting_log: Dict[str, Any] = None
        self._validation_metrics: Dict[str, Any] = None

    def fit(self, train_raw: np.ndarray, validation_raw: np.ndarray, *, seed: int) -> Dict[str, Any]:
        from sklearn.mixture import GaussianMixture

        start = time.perf_counter()
        self.pipeline = PreprocessingPipeline("identity_standardized_v0", seed=seed)
        self.pipeline.fit(train_raw)
        standardized = self.pipeline.transform(train_raw).astype(np.float64)

        gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            reg_covar=self.covariance_regularization,
            n_init=self.n_init,
            max_iter=1,
            tol=self.tol,
            warm_start=True,
            random_state=int(seed),
        )

        lower_bound_curve = []
        converged = False
        warnings_list = []
        try:
            for iteration in range(1, self.max_iter + 1):
                gmm.fit(standardized)
                lower_bound_curve.append({"iteration": iteration, "lower_bound": float(gmm.lower_bound_)})
                if gmm.converged_:
                    converged = True
                    break
        except Exception as exc:  # pragma: no cover - defensive, e.g. singular covariance
            self._fitting_log = {
                "model_config_id": self.model_config_id,
                "seed": int(seed),
                "status": "failed",
                "error": str(exc),
                "wall_time_seconds": time.perf_counter() - start,
                "n_em_iterations_completed": len(lower_bound_curve),
                "lower_bound_curve": lower_bound_curve,
            }
            return self._fitting_log

        if not converged:
            warnings_list.append(
                f"GaussianMixture did not converge within max_iter={self.max_iter} (tol={self.tol})"
            )

        gmm.max_iter = self.max_iter  # restore declared config value for the persisted manifest
        self._estimator = GaussianMixtureEstimator.from_fitted_sklearn(gmm, dimension=5)

        weights = self._estimator._weights
        occupied = int(np.count_nonzero(weights >= self.occupied_weight_threshold))
        cov_eigs = [np.linalg.eigvalsh(cov) for cov in self._estimator._covariances]
        condition_numbers = [
            float(eig[-1] / eig[0]) if eig[0] > 0 else float("inf") for eig in cov_eigs
        ]
        validation_feature_nll = float(-np.mean(self.validation_log_prob(validation_raw)))

        self._fitting_log = {
            "model_config_id": self.model_config_id,
            "seed": int(seed),
            "status": "ok",
            "converged": converged,
            "n_em_iterations_completed": len(lower_bound_curve),
            "max_iter": self.max_iter,
            "tol": self.tol,
            "lower_bound_curve": lower_bound_curve,
            "final_lower_bound": lower_bound_curve[-1]["lower_bound"] if lower_bound_curve else None,
            "component_weights": weights.tolist(),
            "effective_occupied_components": occupied,
            "occupied_weight_threshold": self.occupied_weight_threshold,
            "covariance_condition_numbers": condition_numbers,
            "train_row_count": int(standardized.shape[0]),
            "wall_time_seconds": time.perf_counter() - start,
            "warnings": warnings_list,
            "validation_feature_nll": validation_feature_nll,
        }
        return self._fitting_log

    def validation_log_prob(self, validation_raw: np.ndarray) -> np.ndarray:
        return self._estimator.log_prob(self.pipeline.transform(validation_raw))

    def test_log_prob(self, test_raw: np.ndarray) -> np.ndarray:
        return self._estimator.log_prob(self.pipeline.transform(test_raw))

    def sample(self, n: int, *, seed: int) -> np.ndarray:
        standardized = self._estimator.sample(int(n), seed=int(seed))
        return self.pipeline.inverse(standardized)

    def capability_metadata(self) -> Dict[str, bool]:
        capabilities = {
            "supports_exact_log_prob": True,
            "supports_physical_log_prob": True,
            "stochastic_fit": True,
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
            "estimator_manifest": self._estimator.manifest() if self._estimator is not None else None,
            "preprocessing_name": self.pipeline.variant_id if self.pipeline is not None else None,
        }

    def save_bundle(self, output_dir: Path) -> Dict[str, Any]:
        output_dir = Path(output_dir)
        self._estimator.save(output_dir)
        ma.atomic_write_json(output_dir / "metadata.json", self.model_metadata())
        ma.atomic_write_json(output_dir / "preprocessing.json", self.pipeline.to_dict())
        ma.atomic_write_json(output_dir / "fitting_history.json", self._fitting_log)
        if self._validation_metrics is not None:
            ma.atomic_write_json(output_dir / "validation_metrics.json", self._validation_metrics)
        return {"output_dir": str(output_dir)}

    @classmethod
    def load_bundle(cls, input_dir: Path) -> "GmmAdapter":
        input_dir = Path(input_dir)
        estimator = GaussianMixtureEstimator.load(input_dir)
        adapter = cls(
            n_components=estimator.n_components,
            covariance_regularization=estimator.covariance_regularization,
            n_init=estimator.n_init,
            max_iter=estimator.max_iter,
        )
        adapter._estimator = estimator
        adapter.pipeline = PreprocessingPipeline.from_dict(ma.read_json(input_dir / "preprocessing.json"))
        adapter._fitting_log = ma.read_json(input_dir / "fitting_history.json")
        return adapter

    def set_validation_metrics(self, metrics: Dict[str, Any]) -> None:
        self._validation_metrics = metrics
