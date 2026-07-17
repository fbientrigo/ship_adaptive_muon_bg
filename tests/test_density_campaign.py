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


def test_real_evaluator_bundle_satisfies_the_finiteness_gate(tmp_path):
    """A real, healthy run supplies the finiteness evidence the gate requires.

    The gate only passes when the non-finite counters/rate are present and zero.
    This pins that a bundle produced by the actual evaluator carries them, so the
    stricter gate cannot silently turn healthy runs inconclusive.
    """

    store = ArtifactStore("itest", root=tmp_path)
    rs = _run_spec(ModelSpec("full_gaussian", "full_gaussian", {}))
    rec = run_single(rs, store, device="cpu")
    assert rec["status"] == "completed"

    metrics = json.loads((store.run_paths(rs).metrics).read_text())
    # the evidence the gate consumes is actually written by the evaluator
    assert metrics["held_out"]["non_finite_count"] == 0
    assert metrics["forward_kl"]["non_finite_count"] == 0
    assert metrics["non_finite_density"]["non_finite_count"] == 0
    assert metrics["non_finite_density"]["non_finite_density_rate"] == 0.0

    gate = next(
        g
        for g in metrics["scientific_gates"]["gate_results"]
        if g["gate_id"] == "density_finiteness"
    )
    assert gate["outcome"] == "pass"
    assert gate["threshold_class"] == "mathematical_invariant"


def test_decision_scope_is_recorded_in_artifacts(tmp_path):
    """`pass` is scoped in the artifact itself, not just in the docs."""

    store = ArtifactStore("itest", root=tmp_path)
    rs = _run_spec(ModelSpec("full_gaussian", "full_gaussian", {}))
    rec = run_single(rs, store, device="cpu")
    assert rec["status"] == "completed"

    status = json.loads((store.run_paths(rs).run_status).read_text())
    assert status["technical_status"] == "completed"
    assert status["decision_scope"] == "active_gates_v0"

    metrics = json.loads((store.run_paths(rs).metrics).read_text())
    assert metrics["scientific_gates"]["decision_scope"] == "active_gates_v0"


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


# --- regression: resumed runs must not lose their scientific status (Codex P2)
#
# The skip path used to return only {run_id, status}, so a no-op resume wrote
# scientific_status_counts={} even though every run_status.json on disk carries
# a real verdict. These tests pin that the resumed campaign summary describes
# the selected matrix, not just work executed in this invocation.


def _mini_config(tmp_models, **overrides):
    kwargs = dict(
        experiment_id="resume-mini",
        targets=[TargetSpec("D2")],
        pdg_ids=[13],
        feature_views=[FeatureViewSpec("identity_cartesian_v0")],
        models=tmp_models,
        seeds=[11],
        dataset=DatasetSpec(n_train=1000, n_validation=500, n_test=1000),
        evaluation=_small_eval(),
    )
    kwargs.update(overrides)
    return ExperimentConfig(**kwargs)


def test_resume_preserves_scientific_status_counts(tmp_path):
    config = _mini_config(
        [
            ModelSpec("diagonal_gaussian", "diagonal_gaussian", {}),
            ModelSpec("full_gaussian", "full_gaussian", {}),
        ]
    )
    first = run_campaign(config, root=tmp_path)
    assert first["n_completed"] == 2
    assert first["n_skipped"] == 0
    assert first["scientific_status_counts"]  # non-empty: real gate verdicts

    second = run_campaign(config, root=tmp_path)
    assert second["n_completed"] == 0
    assert second["n_skipped"] == 2
    assert second["scientific_status_counts"] == first["scientific_status_counts"]

    for record in second["runs"]:
        assert record["status"] == "skipped_completed"
        assert record["technical_status"] == "completed"
        assert record["scientific_status"] is not None


def test_resume_preserves_reasons_and_decision_scope(tmp_path):
    config = _mini_config([ModelSpec("diagonal_gaussian", "diagonal_gaussian", {})])
    run_campaign(config, root=tmp_path)
    resumed = run_campaign(config, root=tmp_path)

    record = resumed["runs"][0]
    assert record["status"] == "skipped_completed"
    assert record["decision_scope"] == "active_gates_v0"
    assert isinstance(record["scientific_failure_reasons"], list)


def test_resume_with_corrupted_stored_status_is_unavailable_not_pass(tmp_path):
    store = ArtifactStore("resume-corrupt", root=tmp_path)
    rs = _run_spec(ModelSpec("diagonal_gaussian", "diagonal_gaussian", {}))
    rec = run_single(rs, store, device="cpu")
    assert rec["status"] == "completed"

    status_path = store.run_paths(rs).run_status
    payload = json.loads(status_path.read_text())
    assert payload["status"] == "completed"  # still resumable
    assert payload["config_hash"] == rs.config_hash()  # identity preserved
    del payload["scientific_status"]
    status_path.write_text(json.dumps(payload))

    resumed = run_single(rs, store, device="cpu")
    assert resumed["status"] == "skipped_completed"
    assert resumed["scientific_status"] == "unavailable"


def test_resume_with_malformed_scientific_status_is_unavailable(tmp_path):
    store = ArtifactStore("resume-malformed", root=tmp_path)
    rs = _run_spec(ModelSpec("diagonal_gaussian", "diagonal_gaussian", {}))
    run_single(rs, store, device="cpu")

    status_path = store.run_paths(rs).run_status
    payload = json.loads(status_path.read_text())
    payload["scientific_status"] = 42  # not a string; not a real gate verdict
    status_path.write_text(json.dumps(payload))

    resumed = run_single(rs, store, device="cpu")
    assert resumed["status"] == "skipped_completed"
    assert resumed["scientific_status"] == "unavailable"


def test_resume_does_not_construct_a_model(tmp_path, monkeypatch):
    store = ArtifactStore("resume-no-fit", root=tmp_path)
    rs = _run_spec(ModelSpec("diagonal_gaussian", "diagonal_gaussian", {}))
    assert run_single(rs, store, device="cpu")["status"] == "completed"

    calls = {"n": 0}

    def _spy(*args, **kwargs):
        calls["n"] += 1
        raise AssertionError("create_density_estimator must not be called on resume")

    import Nflow.registry

    monkeypatch.setattr(Nflow.registry, "create_density_estimator", _spy)

    resumed = run_single(rs, store, device="cpu")
    assert resumed["status"] == "skipped_completed"
    assert calls["n"] == 0


def test_resume_counts_mixed_pass_and_catastrophic_separately(tmp_path):
    store = ArtifactStore("resume-mixed", root=tmp_path)
    config = _mini_config(
        [
            ModelSpec("diagonal_gaussian", "diagonal_gaussian", {}),
            ModelSpec("full_gaussian", "full_gaussian", {}),
        ],
        experiment_id="resume-mixed",
    )
    first = run_campaign(config, root=tmp_path)
    assert first["n_completed"] == 2

    # Tamper one stored run to a genuinely different verdict, preserving the
    # identity fields (run_id, config_hash) so it still resolves as complete.
    tampered_run_id = first["runs"][0]["run_id"]
    tampered_path = None
    for run_dir in store.experiment_dir.iterdir():
        if run_dir.name == tampered_run_id:
            tampered_path = run_dir / "run_status.json"
            break
    assert tampered_path is not None
    payload = json.loads(tampered_path.read_text())
    payload["scientific_status"] = "catastrophic"
    payload["scientific_failure_reasons"] = [
        {"gate_id": "importance_ess_catastrophic", "outcome": "catastrophic"}
    ]
    tampered_path.write_text(json.dumps(payload))

    resumed = run_campaign(config, root=tmp_path)
    assert resumed["n_skipped"] == 2
    counts = resumed["scientific_status_counts"]
    assert counts.get("catastrophic") == 1
    assert sum(counts.values()) == 2
    non_tampered = [r for r in resumed["runs"] if r["run_id"] != tampered_run_id][0]
    # the two stored verdicts are counted as distinct buckets, not merged
    assert non_tampered["scientific_status"] != "catastrophic"
