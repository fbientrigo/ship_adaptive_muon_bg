import json
import os

import numpy as np
import pytest
import torch

from scripts.run_afterms_nightly_queue import (
    NIGHTLY_JOB_NAMES,
    build_final_nightly_report,
    checkpoint_file_hash,
    evaluate_generated_samples,
    generated_sample_hash,
    run_build_nightly_report_job,
    train_and_validation_nll_from_fit_result,
)
from Nflow.interfaces import FitResult


def _write_job(artifact_dir, job_name, status=None, metrics=None):
    job_dir = os.path.join(artifact_dir, "jobs", job_name)
    os.makedirs(job_dir, exist_ok=True)
    if status is not None:
        with open(os.path.join(job_dir, "status.json"), "w") as f:
            json.dump(status, f)
    if metrics is not None:
        with open(os.path.join(job_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f)


class _Args:
    def __init__(self, artifact_dir):
        self.artifact_dir = artifact_dir


def test_job_queue_includes_all_fourteen_jobs():
    assert len(NIGHTLY_JOB_NAMES) == 14
    assert "13_build_nightly_report" in NIGHTLY_JOB_NAMES
    assert NIGHTLY_JOB_NAMES[-1] == "13_build_nightly_report"


def test_missing_physical_nll_renders_na_not_zero(tmp_path):
    artifact_dir = str(tmp_path)
    # Only job 04 (single-run shape) is populated; physical_space_nll is
    # explicitly null (no Jacobian for QuantileTransformer), as job 04's
    # actual code now serializes it.
    _write_job(
        artifact_dir,
        "04_legacy_available_code_realnvp_quantile",
        status={"status": "completed"},
        metrics={
            "history": [],
            "metrics": {
                "test_feature_space_nll": -2.3,
                "physical_space_nll": None,
                "wall_time_seconds": 1.0,
                "parameter_count": 100,
            },
        },
    )
    build_final_nightly_report(_Args(artifact_dir), "deadbeef", {"dataset_hash": "abc"})

    md = open(os.path.join(artifact_dir, "report", "nightly_summary.md")).read()
    assert "N/A (No Jac)" in md
    assert "0.0000" not in md

    with open(os.path.join(artifact_dir, "report", "nightly_results.csv")) as f:
        csv_text = f.read()
    # csv.writer serializes None as an empty field, never as "0.0"
    assert ",0.0," not in csv_text


def test_status_code_reflects_actual_completion_not_hardcoded(tmp_path):
    artifact_dir = str(tmp_path)
    _write_job(artifact_dir, "00_environment_and_dataset_smoke", status={"status": "completed"})
    # Everything else is left "missing" -> must not be NIGHTLY_SMOKES_COMPLETE
    build_final_nightly_report(_Args(artifact_dir), "deadbeef", {"dataset_hash": "abc"})

    summary = json.load(open(os.path.join(artifact_dir, "report", "nightly_summary.json")))
    assert summary["status_code"] == "NIGHTLY_SMOKES_PARTIAL"

    md = open(os.path.join(artifact_dir, "report", "nightly_summary.md")).read()
    assert "NIGHTLY_SMOKES_PARTIAL" in md
    assert "NIGHTLY_SMOKES_COMPLETE" not in md


def test_status_code_complete_when_smoke_jobs_done_despite_job13_missing(tmp_path):
    artifact_dir = str(tmp_path)
    # Jobs 00-12 are all completed; job 13 (the report builder) can never
    # observe its own status.json while building this very report, so it
    # must be excluded from the completeness gate.
    for name in NIGHTLY_JOB_NAMES:
        if name == "13_build_nightly_report":
            continue
        _write_job(artifact_dir, name, status={"status": "completed"})
    build_final_nightly_report(_Args(artifact_dir), "deadbeef", {"dataset_hash": "abc"})

    summary = json.load(open(os.path.join(artifact_dir, "report", "nightly_summary.json")))
    assert summary["status_code"] == "NIGHTLY_SMOKES_COMPLETE"
    assert summary["job_statuses"]["13_build_nightly_report"] == "missing"


def test_report_builder_job_excluded_from_its_own_performance_table(tmp_path):
    artifact_dir = str(tmp_path)
    for name in NIGHTLY_JOB_NAMES:
        _write_job(artifact_dir, name, status={"status": "completed"})
    # Stray/orphaned metrics.json under the report-builder job's own dir
    # (as observed in production artifacts) must never be rendered as a
    # model-performance row.
    _write_job(
        artifact_dir,
        "13_build_nightly_report",
        metrics={
            "some_stray_run": {
                "metrics": {
                    "test_feature_space_nll": 1.0,
                    "physical_space_nll": 1.0,
                    "wall_time_seconds": 1.0,
                    "parameter_count": 1,
                }
            }
        },
    )
    build_final_nightly_report(_Args(artifact_dir), "deadbeef", {"dataset_hash": "abc"})

    md = open(os.path.join(artifact_dir, "report", "nightly_summary.md")).read()
    assert "some_stray_run" not in md
    csv_text = open(os.path.join(artifact_dir, "report", "nightly_results.csv")).read()
    assert "some_stray_run" not in csv_text


def test_report_regeneration_from_existing_artifacts_is_deterministic(tmp_path):
    artifact_dir = str(tmp_path)
    _write_job(
        artifact_dir,
        "04_legacy_available_code_realnvp_quantile",
        status={"status": "completed"},
        metrics={
            "history": [],
            "metrics": {
                "test_feature_space_nll": -2.3,
                "physical_space_nll": None,
                "wall_time_seconds": 1.0,
                "parameter_count": 100,
            },
        },
    )
    args = _Args(artifact_dir)
    build_final_nightly_report(args, "deadbeef", {"dataset_hash": "abc"})
    first = open(os.path.join(artifact_dir, "report", "nightly_summary.md")).read()
    build_final_nightly_report(args, "deadbeef", {"dataset_hash": "abc"})
    second = open(os.path.join(artifact_dir, "report", "nightly_summary.md")).read()
    assert first == second


def test_train_nll_not_conflated_with_validation_nll():
    fit_res = FitResult(
        status="ok",
        seed=1,
        train_history=[{"step": 0, "train_nll": 1.111}],
        best_validation_nll=2.222,
    )
    train_nll, val_nll = train_and_validation_nll_from_fit_result(fit_res)
    assert train_nll == pytest.approx(1.111)
    assert val_nll == pytest.approx(2.222)
    assert train_nll != val_nll


def test_train_nll_is_null_when_fitter_did_not_record_it():
    fit_res = FitResult(status="ok", seed=1, train_history=[], best_validation_nll=2.222)
    train_nll, val_nll = train_and_validation_nll_from_fit_result(fit_res)
    assert train_nll is None
    assert val_nll == pytest.approx(2.222)


def test_generated_domain_violations_counts_negative_pz():
    rng = np.random.default_rng(0)
    n = 1000
    test_data = np.column_stack([
        rng.normal(size=n), rng.normal(size=n),
        rng.uniform(1.0, 50.0, size=n),
        rng.normal(size=n), rng.normal(size=n),
    ])
    q_samples = test_data.copy()
    # Force a known fraction of generated pz negative (an out-of-domain flow
    # inversion), distinct from the reference sample.
    q_samples[:50, 2] = -np.abs(q_samples[:50, 2]) - 1.0

    result = evaluate_generated_samples(test_data, q_samples)
    viol = result["generated_domain_violations"]
    assert viol["generated_domain_violation_count"] == 50
    assert viol["generated_domain_violation_rate"] == pytest.approx(0.05)
    assert viol["min_generated_pz"] < 0.0


def test_generated_domain_violations_zero_when_support_respected():
    rng = np.random.default_rng(1)
    n = 500
    test_data = np.column_stack([
        rng.normal(size=n), rng.normal(size=n),
        rng.uniform(1.0, 50.0, size=n),
        rng.normal(size=n), rng.normal(size=n),
    ])
    q_samples = test_data.copy()

    result = evaluate_generated_samples(test_data, q_samples)
    viol = result["generated_domain_violations"]
    assert viol["generated_domain_violation_count"] == 0
    assert viol["generated_domain_violation_rate"] == 0.0


def test_generated_domain_violations_reports_quantiles_without_clipping():
    rng = np.random.default_rng(2)
    n = 1000
    test_data = np.column_stack([
        rng.normal(size=n), rng.normal(size=n),
        rng.uniform(1.0, 50.0, size=n),
        rng.normal(size=n), rng.normal(size=n),
    ])
    q_samples = test_data.copy()
    q_samples[:10, 2] = -5.0  # out-of-domain, must survive un-clipped

    result = evaluate_generated_samples(test_data, q_samples)
    viol = result["generated_domain_violations"]
    quantiles = viol["quantiles_of_generated_pz"]
    assert quantiles["q001"] <= quantiles["q01"] <= quantiles["q05"] <= quantiles["q50"]
    # The negative values must still be present in the generated column
    # itself (nothing clipped/repaired them before this diagnostic ran).
    assert np.min(q_samples[:, 2]) == -5.0
    assert viol["min_generated_pz"] == -5.0


def test_fixed_seed_generator_produces_deterministic_sample_hash():
    def draw(seed):
        gen = torch.Generator()
        gen.manual_seed(seed)
        return torch.randn(200, 5, generator=gen).numpy()

    first = draw(20260720)
    second = draw(20260720)
    assert generated_sample_hash(first) == generated_sample_hash(second)


def test_different_generation_seed_changes_sample_hash():
    def draw(seed):
        gen = torch.Generator()
        gen.manual_seed(seed)
        return torch.randn(200, 5, generator=gen).numpy()

    a = draw(20260720)
    b = draw(1)
    assert generated_sample_hash(a) != generated_sample_hash(b)


def test_sample_hash_is_a_stable_hex_digest():
    arr = np.arange(20, dtype=np.float64).reshape(4, 5)
    first = generated_sample_hash(arr)
    second = generated_sample_hash(arr)
    assert first == second
    assert len(first) == 64
    int(first, 16)  # raises ValueError if not valid hex


def test_checkpoint_hash_changes_when_file_content_changes(tmp_path):
    path = tmp_path / "ckpt.pt"
    path.write_bytes(b"checkpoint-bytes-v1")
    first = checkpoint_file_hash(str(path))
    second = checkpoint_file_hash(str(path))
    assert first == second

    path.write_bytes(b"checkpoint-bytes-v2-different")
    third = checkpoint_file_hash(str(path))
    assert third != first


def test_physical_nll_zero_is_distinguishable_from_missing(tmp_path):
    artifact_dir = str(tmp_path)
    # A real, computed physical NLL of exactly 0.0 must render as "0.0000",
    # never fall into the "N/A (No Jac)" missing-value branch.
    _write_job(
        artifact_dir,
        "04_legacy_available_code_realnvp_quantile",
        status={"status": "completed"},
        metrics={
            "history": [],
            "metrics": {
                "test_feature_space_nll": -1.0,
                "physical_space_nll": 0.0,
                "wall_time_seconds": 1.0,
                "parameter_count": 100,
            },
        },
    )
    build_final_nightly_report(_Args(artifact_dir), "deadbeef", {"dataset_hash": "abc"})

    md = open(os.path.join(artifact_dir, "report", "nightly_summary.md")).read()
    assert "0.0000" in md
    assert "N/A (No Jac)" not in md

    csv_text = open(os.path.join(artifact_dir, "report", "nightly_results.csv")).read()
    assert ",0.0," in csv_text


def test_json_csv_markdown_agree_on_physical_and_test_nll(tmp_path):
    artifact_dir = str(tmp_path)
    metrics = {
        "test_feature_space_nll": -2.3456,
        "physical_space_nll": 1.2345,
        "wall_time_seconds": 1.0,
        "parameter_count": 100,
    }
    _write_job(
        artifact_dir,
        "05_affine_preprocessing_ab_pdg13",
        status={"status": "completed"},
        metrics={"identity_standardized_v0_affine_small_unweighted": {"metrics": metrics}},
    )
    build_final_nightly_report(_Args(artifact_dir), "deadbeef", {"dataset_hash": "abc"})

    # The job's own metrics.json is the source of truth; it is never
    # rewritten by the report builder.
    source = json.load(
        open(os.path.join(artifact_dir, "jobs", "05_affine_preprocessing_ab_pdg13", "metrics.json"))
    )
    source_metrics = source["identity_standardized_v0_affine_small_unweighted"]["metrics"]
    assert source_metrics == metrics

    csv_text = open(os.path.join(artifact_dir, "report", "nightly_results.csv")).read()
    assert str(metrics["physical_space_nll"]) in csv_text
    assert str(metrics["test_feature_space_nll"]) in csv_text

    md = open(os.path.join(artifact_dir, "report", "nightly_summary.md")).read()
    assert f"{metrics['physical_space_nll']:.4f}" in md
    assert f"{metrics['test_feature_space_nll']:.4f}" in md


# --- Job 13 self-consistency: this guards the "Job 13 failed: [Errno 2] No
# such file or directory: .../13_build_nightly_report/status.json" incident.
# Root cause was never a self-status read dependency (build_final_nightly_report
# already treats its own status.json as gracefully "missing"); it was that
# nothing created jobs/13_build_nightly_report/ before the caller tried to
# write status.json into it. run_build_nightly_report_job now owns both the
# report build and that status write, for the queue loop and a standalone
# `--run-job 13_build_nightly_report` invocation alike.


class _QueueArgs:
    def __init__(self, artifact_dir, shard_dir=None):
        self.artifact_dir = str(artifact_dir)
        self.shard_dir = str(shard_dir) if shard_dir is not None else str(artifact_dir)


def _write_all_smoke_jobs_completed(artifact_dir):
    for name in NIGHTLY_JOB_NAMES:
        if name == "13_build_nightly_report":
            continue
        _write_job(str(artifact_dir), name, status={"status": "completed"})


def test_run_build_nightly_report_job_succeeds_with_no_preexisting_job13_status(tmp_path):
    # This is the exact incident scenario: jobs/13_build_nightly_report/ does
    # not exist anywhere on disk yet when the report job first runs.
    args = _QueueArgs(tmp_path)
    assert not os.path.exists(os.path.join(tmp_path, "jobs", "13_build_nightly_report"))

    run_build_nightly_report_job(args, "deadbeef", {"dataset_hash": "abc"})

    status = json.load(open(os.path.join(tmp_path, "jobs", "13_build_nightly_report", "status.json")))
    assert status["status"] == "completed"


def test_report_generation_success_marks_job13_completed(tmp_path):
    args = _QueueArgs(tmp_path)
    _write_all_smoke_jobs_completed(tmp_path)

    run_build_nightly_report_job(args, "deadbeef", {"dataset_hash": "abc"})

    status = json.load(open(os.path.join(tmp_path, "jobs", "13_build_nightly_report", "status.json")))
    assert status["status"] == "completed"
    summary = json.load(open(os.path.join(tmp_path, "report", "nightly_summary.json")))
    assert summary["status_code"] == "NIGHTLY_SMOKES_COMPLETE"


def test_report_generation_failure_marks_job13_failed(tmp_path):
    args = _QueueArgs(tmp_path)
    # A metrics.json missing a required numeric key makes the markdown
    # renderer's `:.4f` format spec raise on a real value (None), the same
    # class of failure a malformed real job could produce.
    _write_job(
        str(tmp_path),
        "05_affine_preprocessing_ab_pdg13",
        status={"status": "completed"},
        metrics={"identity_standardized_v0_affine_small_unweighted": {"metrics": {
            "physical_space_nll": 0.5, "wall_time_seconds": 1.0, "parameter_count": 10,
        }}},
    )

    with pytest.raises(TypeError):
        run_build_nightly_report_job(args, "deadbeef", {"dataset_hash": "abc"})

    status = json.load(open(os.path.join(tmp_path, "jobs", "13_build_nightly_report", "status.json")))
    assert status["status"] == "failed"
    assert "error" in status


def test_no_stale_report_committed_when_report_generation_fails(tmp_path):
    args = _QueueArgs(tmp_path)
    job_dir = os.path.join(tmp_path, "jobs", "05_affine_preprocessing_ab_pdg13")
    good_metrics = {"identity_standardized_v0_affine_small_unweighted": {"metrics": {
        "test_feature_space_nll": -1.0, "physical_space_nll": 0.5,
        "wall_time_seconds": 1.0, "parameter_count": 10,
    }}}
    _write_job(str(tmp_path), "05_affine_preprocessing_ab_pdg13", status={"status": "completed"}, metrics=good_metrics)
    run_build_nightly_report_job(args, "deadbeef", {"dataset_hash": "abc"})

    good_json = open(os.path.join(tmp_path, "report", "nightly_summary.json")).read()
    good_md = open(os.path.join(tmp_path, "report", "nightly_summary.md")).read()
    good_csv = open(os.path.join(tmp_path, "report", "nightly_results.csv")).read()

    # Corrupt the same job's metrics.json (drop a required key) so the next
    # report build fails partway through rendering.
    bad_metrics = {"identity_standardized_v0_affine_small_unweighted": {"metrics": {
        "physical_space_nll": 0.5, "wall_time_seconds": 1.0, "parameter_count": 10,
    }}}
    _write_job(str(tmp_path), "05_affine_preprocessing_ab_pdg13", metrics=bad_metrics)

    with pytest.raises(TypeError):
        run_build_nightly_report_job(args, "deadbeef", {"dataset_hash": "abc"})

    # None of the three previously-good report files may be silently
    # replaced with new-but-inconsistent content when generation fails.
    assert open(os.path.join(tmp_path, "report", "nightly_summary.json")).read() == good_json
    assert open(os.path.join(tmp_path, "report", "nightly_summary.md")).read() == good_md
    assert open(os.path.join(tmp_path, "report", "nightly_results.csv")).read() == good_csv


def test_report_job_never_invokes_training(tmp_path, monkeypatch):
    import scripts.run_afterms_nightly_queue as queue_mod

    def _boom(*a, **kw):
        raise AssertionError("report-only job 13 must never invoke neural training")

    monkeypatch.setattr(queue_mod, "run_neural_training_subprocess", _boom)

    args = _QueueArgs(tmp_path)
    _write_all_smoke_jobs_completed(tmp_path)
    run_build_nightly_report_job(args, "deadbeef", {"dataset_hash": "abc"})  # must not raise


def test_report_job_deterministic_once_its_own_status_stabilizes(tmp_path):
    args = _QueueArgs(tmp_path)
    _write_all_smoke_jobs_completed(tmp_path)

    # The very first run transitions job 13's own displayed status from
    # "missing" to "completed" -- that transition is expected to change the
    # rendered report once. From the second run onward, with nothing else
    # changing on disk, output must be byte-for-byte identical.
    run_build_nightly_report_job(args, "deadbeef", {"dataset_hash": "abc"})  # missing -> completed
    run_build_nightly_report_job(args, "deadbeef", {"dataset_hash": "abc"})
    second = open(os.path.join(tmp_path, "report", "nightly_summary.md")).read()
    run_build_nightly_report_job(args, "deadbeef", {"dataset_hash": "abc"})
    third = open(os.path.join(tmp_path, "report", "nightly_summary.md")).read()
    assert second == third


def test_job13_appears_exactly_once_in_all_three_report_outputs(tmp_path):
    args = _QueueArgs(tmp_path)
    _write_all_smoke_jobs_completed(tmp_path)
    run_build_nightly_report_job(args, "deadbeef", {"dataset_hash": "abc"})

    summary = json.load(open(os.path.join(tmp_path, "report", "nightly_summary.json")))
    assert list(summary["job_statuses"]).count("13_build_nightly_report") == 1

    md = open(os.path.join(tmp_path, "report", "nightly_summary.md")).read()
    assert md.count("13_build_nightly_report") == 1  # Job Statuses table only

    csv_text = open(os.path.join(tmp_path, "report", "nightly_results.csv")).read()
    assert "13_build_nightly_report" not in csv_text  # never a performance row


def test_json_csv_markdown_agree_on_job_count_and_final_status(tmp_path):
    args = _QueueArgs(tmp_path)
    _write_all_smoke_jobs_completed(tmp_path)
    run_build_nightly_report_job(args, "deadbeef", {"dataset_hash": "abc"})

    summary = json.load(open(os.path.join(tmp_path, "report", "nightly_summary.json")))
    assert len(summary["job_statuses"]) == 14
    assert summary["status_code"] == "NIGHTLY_SMOKES_COMPLETE"

    md = open(os.path.join(tmp_path, "report", "nightly_summary.md")).read()
    assert "NIGHTLY_SMOKES_COMPLETE" in md
    # 14 job rows in the Job Statuses table (header + separator + 14 rows).
    assert sum(1 for name in NIGHTLY_JOB_NAMES if f"| {name} |" in md) == 14

    status = json.load(open(os.path.join(tmp_path, "jobs", "13_build_nightly_report", "status.json")))
    assert status["status"] == "completed"
