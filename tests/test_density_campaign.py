"""Integration tests for the campaign runner (small CPU runs).

Marked ``lab``/``slow`` where they train a flow or fit a GMM. Covers a small
matched-view campaign, failure isolation, resume/skip, force, and that failed
runs remain visible.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ship_muon_bg.density_lab import (
    ArtifactStore,
    ExperimentConfig,
    run_campaign,
    run_single,
)
from ship_muon_bg.density_lab.config import (
    DatasetSpec,
    EvaluationSpec,
    FeatureViewSpec,
    ModelSpec,
    RunSpec,
    TargetSpec,
)

pytestmark = pytest.mark.lab


def _small_eval():
    return EvaluationSpec(ess_sample_count=1500, c2st_sample_count=600)


def _run_spec(model, view="identity_cartesian_v0", target="D2", variant=None, pdg=13):
    return RunSpec(
        experiment_id="itest",
        target=TargetSpec(target, variant),
        pdg_id=pdg,
        feature_view=FeatureViewSpec(view),
        model=model,
        seed=11,
        dataset=DatasetSpec(n_train=1200, n_validation=600, n_test=1200),
        evaluation=_small_eval(),
        device="cpu",
    )


def test_baseline_run_writes_physical_metrics(tmp_path):
    store = ArtifactStore("itest", root=tmp_path)
    rs = _run_spec(ModelSpec("full_gaussian", "full_gaussian", {}))
    rec = run_single(rs, store, device="cpu")
    assert rec["status"] == "completed"
    metrics = json.loads((store.run_paths(rs).metrics).read_text())
    assert metrics["physical_space"] is True
    assert "forward_kl" in metrics and "importance_ess" in metrics


def test_failed_run_is_recorded_and_visible(tmp_path):
    store = ArtifactStore("itest", root=tmp_path)
    # An unknown model family raises inside run_single -> recorded as failed.
    rs = _run_spec(ModelSpec("broken", "nonexistent_family", {}))
    rec = run_single(rs, store, device="cpu")
    assert rec["status"] == "failed"
    status = json.loads((store.run_paths(rs).run_status).read_text())
    assert status["status"] == "failed"
    assert status["error"]  # traceback preserved


def test_resume_skips_completed_and_force_reruns(tmp_path):
    store = ArtifactStore("itest", root=tmp_path)
    rs = _run_spec(ModelSpec("diagonal_gaussian", "diagonal_gaussian", {}))
    assert run_single(rs, store, device="cpu")["status"] == "completed"
    assert run_single(rs, store, device="cpu")["status"] == "skipped_completed"
    assert run_single(rs, store, force=True, device="cpu")["status"] == "completed"


def test_matched_rows_identical_across_views(tmp_path):
    store = ArtifactStore("itest", root=tmp_path)
    model = ModelSpec("diagonal_gaussian", "diagonal_gaussian", {})
    cache = {}
    samples = {}
    for view in ("identity_cartesian_v0", "cartesian_logpz_v0"):
        rs = _run_spec(model, view=view)
        run_single(rs, store, device="cpu", dataset_cache=cache)
    # both runs drew from the same cached dataset (same physical rows)
    assert len(cache) == 1


def test_small_campaign_continues_after_failure(tmp_path):
    config = ExperimentConfig(
        experiment_id="mini",
        targets=[TargetSpec("D2")],
        pdg_ids=[13],
        feature_views=[FeatureViewSpec("identity_cartesian_v0")],
        models=[
            ModelSpec("diagonal_gaussian", "diagonal_gaussian", {}),
            ModelSpec("broken", "nonexistent_family", {}),
            ModelSpec("full_gaussian", "full_gaussian", {}),
        ],
        seeds=[11],
        dataset=DatasetSpec(n_train=1000, n_validation=500, n_test=1000),
        evaluation=_small_eval(),
    )
    summary = run_campaign(config, root=tmp_path)
    assert summary["n_completed"] == 2
    assert summary["n_failed"] == 1
    assert summary["n_runs"] == 3
