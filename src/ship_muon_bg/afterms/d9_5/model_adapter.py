"""Common D9-5 model-adapter contract (Sec 8).

The smallest useful shared surface across NF_AC/GAUSS_DIAG/GAUSS_FULL/GMM --
not a generic ML framework. Every adapter exposes the same seven operations
and a capability dict so the evaluation/aggregation/report layers never branch
on model family.
"""

from __future__ import annotations

import json
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

REQUIRED_CAPABILITY_KEYS = (
    "supports_exact_log_prob",
    "supports_physical_log_prob",
    "stochastic_fit",
    "supports_cuda",
    "supports_resume",
    "deterministic_sampling_given_seed",
)


class ModelAdapter(ABC):
    """Base class for one (track, model_config_id[, seed]) fit."""

    model_family_id: str
    model_config_id: str

    @abstractmethod
    def fit(self, train_raw: np.ndarray, validation_raw: np.ndarray, *, seed: int) -> Dict[str, Any]:
        """Fit on raw (n, 8) rows already filtered to this track's PDG value.
        ``validation_raw`` is required by every adapter for a uniform
        signature; NF_AC uses it functionally for early stopping, the
        deterministic/stochastic Gaussian and GMM controls use it only to
        record a train-time validation NLL alongside the fit. Returns a
        JSON-safe fitting-log dict."""

    @abstractmethod
    def validation_log_prob(self, validation_raw: np.ndarray) -> np.ndarray:
        """Feature-space log p(x) for raw (n, 8) validation rows."""

    @abstractmethod
    def test_log_prob(self, test_raw: np.ndarray) -> np.ndarray:
        """Feature-space log p(x) for raw (n, 8) test rows."""

    @abstractmethod
    def sample(self, n: int, *, seed: int) -> np.ndarray:
        """Deterministic (given seed) draws in raw physical (n, 5) space."""

    @abstractmethod
    def save_bundle(self, output_dir: Path) -> Dict[str, Any]:
        """Persist metadata.json / parameters.npz / fitting_log.json (+
        validation_metrics.json if already computed) atomically."""

    @classmethod
    @abstractmethod
    def load_bundle(cls, input_dir: Path) -> "ModelAdapter":
        pass

    @abstractmethod
    def model_metadata(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def capability_metadata(self) -> Dict[str, bool]:
        pass


def assert_capability_metadata(capabilities: Dict[str, bool]) -> None:
    missing = [k for k in REQUIRED_CAPABILITY_KEYS if k not in capabilities]
    if missing:
        raise ValueError(f"capability_metadata is missing required keys: {missing}")


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".tmp_{path.name}_", suffix=".json")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        tmp_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def physical_log_prob(feature_log_prob: np.ndarray, forward_log_abs_det_jacobian: np.ndarray) -> np.ndarray:
    """Sec 7: log p_x(x) = log p_z(T(x)) + log|det J_T|."""

    return feature_log_prob + forward_log_abs_det_jacobian
