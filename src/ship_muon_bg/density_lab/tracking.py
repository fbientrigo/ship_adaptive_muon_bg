"""Optional experiment-tracking adapter.

The local artifact store is canonical and mandatory. This module adds a
minimal optional MLflow tracker selected by ``tracking_mode = "local" |
"mlflow"``. Remote credentials/URIs come exclusively from environment
variables (``MLFLOW_TRACKING_URI``, ``MLFLOW_TRACKING_USERNAME``,
``MLFLOW_TRACKING_PASSWORD``) -- never hardcoded. MLflow is imported lazily.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional


def _flatten_scalars(payload: Any, prefix: str = "") -> Dict[str, float]:
    """Collect numeric leaves into a flat ``dotted.key -> float`` mapping."""

    out: Dict[str, float] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            out.update(_flatten_scalars(value, "{}.{}".format(prefix, key) if prefix else str(key)))
    elif isinstance(payload, bool):
        out[prefix] = 1.0 if payload else 0.0
    elif isinstance(payload, (int, float)):
        import math

        if math.isfinite(float(payload)):
            out[prefix] = float(payload)
    return out


class LocalTracker:
    """No-op tracker: the local artifact store is already canonical."""

    mode = "local"

    def log_run(self, *, run_id: str, params: Dict[str, Any], metrics: Dict[str, Any], run_dir: Path) -> None:
        return None


class MlflowTracker:
    """Minimal MLflow tracker (params, scalar metrics, artifact directory)."""

    mode = "mlflow"

    def __init__(self, experiment_name: str, *, tracking_uri: Optional[str] = None) -> None:
        import mlflow  # lazy optional import

        self._mlflow = mlflow
        uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
        if uri:
            mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment_name)

    def log_run(self, *, run_id: str, params: Dict[str, Any], metrics: Dict[str, Any], run_dir: Path) -> None:
        mlflow = self._mlflow
        with mlflow.start_run(run_name=run_id):
            flat_params = {
                k: str(v) for k, v in _flatten_params(params).items()
            }
            mlflow.log_params(flat_params)
            for key, value in _flatten_scalars(metrics).items():
                mlflow.log_metric(key.replace("/", "."), value)
            if Path(run_dir).exists():
                mlflow.log_artifacts(str(run_dir))


def _flatten_params(payload: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            new_prefix = "{}.{}".format(prefix, key) if prefix else str(key)
            if isinstance(value, dict):
                out.update(_flatten_params(value, new_prefix))
            else:
                out[new_prefix] = value
    return out


def make_tracker(tracking_spec, *, tracking_uri: Optional[str] = None):
    """Construct the tracker for ``tracking_spec.mode`` (default local)."""

    mode = getattr(tracking_spec, "mode", "local")
    if mode == "local":
        return LocalTracker()
    if mode == "mlflow":
        return MlflowTracker(
            getattr(tracking_spec, "experiment_name", "controlled_density_lab"),
            tracking_uri=tracking_uri,
        )
    raise ValueError("unknown tracking mode {!r}".format(mode))
