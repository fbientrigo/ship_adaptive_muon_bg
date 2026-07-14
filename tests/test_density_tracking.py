"""Tests for the optional MLflow tracking adapter (temporary local file store).

Marked ``tracking``; auto-skips when mlflow is absent. The local artifact store
remains canonical -- this only exercises the optional adapter.
"""

from __future__ import annotations

import numpy as np
import pytest

from ship_muon_bg.density_lab.tracking import LocalTracker, make_tracker
from ship_muon_bg.density_lab.config import TrackingSpec

pytestmark = pytest.mark.tracking


def test_local_tracker_is_noop(tmp_path):
    tracker = make_tracker(TrackingSpec(mode="local"))
    assert isinstance(tracker, LocalTracker)
    # no-op: must not raise
    tracker.log_run(run_id="r", params={"a": 1}, metrics={"m": 0.5}, run_dir=tmp_path)


def test_mlflow_tracker_logs_to_temp_file_store(tmp_path, monkeypatch):
    # mlflow >= 3 gates the maintenance-mode file store behind this flag; the
    # task explicitly permits a temporary local file-based tracking URI.
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    uri = "file://{}".format(tmp_path / "mlruns")
    tracker = make_tracker(
        TrackingSpec(mode="mlflow", experiment_name="unit_test"),
        tracking_uri=uri,
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text("{}")
    tracker.log_run(
        run_id="run_0",
        params={"model": {"family": "full_gaussian"}, "seed": 11},
        metrics={
            "forward_kl": {"forward_kl": 0.12},
            "importance_ess": {"ess_over_n": 0.8, "catastrophic": False},
        },
        run_dir=run_dir,
    )
    import mlflow

    mlflow.set_tracking_uri(uri)
    exp = mlflow.get_experiment_by_name("unit_test")
    assert exp is not None
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    assert len(runs) == 1
    assert runs.iloc[0]["metrics.forward_kl.forward_kl"] == pytest.approx(0.12)
