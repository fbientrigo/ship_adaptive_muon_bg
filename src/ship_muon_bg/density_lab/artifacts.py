"""Local artifact store and run-id derivation.

The local store is canonical and mandatory. ``run_id`` is derived from the
canonical run configuration hash (not a timestamp), so an identical config
resumes/skips and a changed config produces a new run directory. Each run
writes the full artifact set under
``artifacts/density_lab/<experiment_id>/<run_id>/``.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

ARTIFACT_ROOT = Path("artifacts") / "density_lab"

STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_RUNNING = "running"


def _sanitize(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "-", str(text))


def derive_run_id(run_spec) -> str:
    """Deterministic, descriptive run id derived from the config hash."""

    h = run_spec.config_hash()[:12]
    variant = run_spec.target.variant or "default"
    pdg = "m13" if run_spec.pdg_id == -13 else "p13"
    label = "_".join(
        _sanitize(part)
        for part in (
            run_spec.target.target_id,
            variant,
            pdg,
            run_spec.feature_view.view_id,
            run_spec.model.name,
            "seed{}".format(run_spec.seed),
        )
    )
    return "{}_{}".format(label, h)


@dataclass
class RunPaths:
    root: Path
    experiment_dir: Path
    run_dir: Path

    @property
    def run_status(self) -> Path:
        return self.run_dir / "run_status.json"

    @property
    def metrics(self) -> Path:
        return self.run_dir / "metrics.json"

    @property
    def training_history(self) -> Path:
        return self.run_dir / "training_history.jsonl"

    @property
    def samples(self) -> Path:
        return self.run_dir / "samples.npz"


class ArtifactStore:
    """Filesystem artifact store rooted at ``<root>/<experiment_id>/``."""

    def __init__(self, experiment_id: str, *, root: Optional[Path] = None) -> None:
        self.experiment_id = experiment_id
        self.root = Path(root) if root is not None else ARTIFACT_ROOT
        self.experiment_dir = self.root / _sanitize(experiment_id)

    def run_paths(self, run_spec) -> RunPaths:
        run_dir = self.experiment_dir / derive_run_id(run_spec)
        return RunPaths(
            root=self.root, experiment_dir=self.experiment_dir, run_dir=run_dir
        )

    def is_complete(self, run_spec) -> bool:
        status_path = self.run_paths(run_spec).run_status
        if not status_path.exists():
            return False
        try:
            status = json.loads(status_path.read_text())
        except (OSError, json.JSONDecodeError):
            return False
        return status.get("status") == STATUS_COMPLETED

    def write_run(
        self,
        run_spec,
        *,
        environment: Dict[str, Any],
        dataset_manifest: Dict[str, Any],
        feature_pipeline_manifest: Dict[str, Any],
        model_manifest: Dict[str, Any],
        fit_result: Dict[str, Any],
        metrics: Dict[str, Any],
        training_history,
        samples: Dict[str, np.ndarray],
        status: str,
        run_id: str,
        save_manifest: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        hashes: Optional[Dict[str, Any]] = None,
        scientific_status: Optional[str] = None,
    ) -> RunPaths:
        paths = self.run_paths(run_spec)
        paths.run_dir.mkdir(parents=True, exist_ok=True)
        # On a re-run (e.g. --force) that does not reproduce an optional
        # artifact, remove the stale one so a later reader can never mix a
        # previous run's samples/checkpoint with this attempt's metadata.
        if not samples:
            (paths.samples).unlink(missing_ok=True)
        if save_manifest is None:
            (paths.run_dir / "model_parameters.npz").unlink(missing_ok=True)
            (paths.run_dir / "model_config.json").unlink(missing_ok=True)
            shutil.rmtree(paths.run_dir / "checkpoint", ignore_errors=True)
        _write_json(paths.run_dir / "experiment_config.json", run_spec.to_dict())
        _write_json(paths.run_dir / "environment.json", environment)
        _write_json(paths.run_dir / "dataset_manifest.json", dataset_manifest)
        _write_json(
            paths.run_dir / "feature_pipeline_manifest.json",
            feature_pipeline_manifest,
        )
        _write_json(paths.run_dir / "model_manifest.json", model_manifest)
        _write_json(paths.run_dir / "fit_result.json", fit_result)
        _write_json(paths.metrics, metrics)
        _write_jsonl(paths.training_history, training_history or [])
        if samples:
            np.savez(paths.samples, **samples)
        # "status" is retained for backward compatibility and keeps its original
        # meaning: technical execution status (completed | failed). We add an
        # explicit "technical_status" mirror plus a separate "scientific_status"
        # so a technically completed run can still be scientifically
        # catastrophic (or inconclusive) without being reported as a technical
        # failure. scientific_status is None for technically failed runs and for
        # runs that were not scientifically evaluated.
        status_payload = {
            "run_id": run_id,
            "status": status,
            "technical_status": status,
            "scientific_status": scientific_status,
            "config_hash": run_spec.config_hash(),
            "hashes": hashes or {},
            "save_manifest": save_manifest,
            "error": error,
        }
        _write_json(paths.run_status, status_payload)
        return paths


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_default))


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=_default) + "\n")


def _default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError("not JSON serializable: {!r}".format(type(obj)))
