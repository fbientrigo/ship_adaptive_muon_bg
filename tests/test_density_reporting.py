"""Tests for the report builder (reads artifacts only, never retrains).

The summary/limitations builders are NumPy-only; the plot builder is marked
``lab`` (matplotlib).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ship_muon_bg.density_lab.reporting import (
    PlotSeriesKey,
    _plot_series_label,
    _series_metric_stats,
    build_report,
    build_scientific_gate_summary,
    build_summary_tables,
    load_run_records,
)


def _write_run(
    campaign_dir,
    run_id,
    *,
    device,
    seed,
    status="completed",
    fkl=0.1,
    ess=0.8,
    target_id="D3",
    variant=None,
    scientific_status="pass",
    reasons=None,
    observed_rare=None,
    target_stage="transformed",
    sampling_regime="iid_target",
    diagnostic_only=False,
    feature_view="identity_cartesian_v0",
    parameter_count=37416,
):
    run_dir = campaign_dir / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "run_status.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": status,
                "technical_status": status,
                "scientific_status": scientific_status if status == "completed" else None,
            }
        )
    )
    (run_dir / "experiment_config.json").write_text(
        json.dumps(
            {
                "target": {
                    "target_id": target_id,
                    "variant": variant,
                    "stage": target_stage,
                },
                "pdg_id": 13,
                "feature_view": {"view_id": feature_view},
                "model": {"name": "affine_small"},
                "seed": seed,
                "device": device,
            }
        )
    )
    if status == "completed":
        metrics = {
            "forward_kl": {"forward_kl": fkl},
            "held_out": {"held_out_nll": 1.0},
            "importance_ess": {"ess_over_n": ess, "catastrophic": ess < 0.01},
            "c2st": {"c2st_accuracy": 0.6},
            "parameter_count": parameter_count,
            "sampling_regime": sampling_regime,
            "diagnostic_only": diagnostic_only,
            "scientific_gates": {
                "gate_schema_version": "0",
                "scientific_status": scientific_status,
                "scientific_failure_reasons": reasons or [],
                "gate_results": [],
                "gate_config_hash": "deadbeef",
            },
        }
        if observed_rare is not None:
            metrics["rare_mode"] = {
                "observed_q_rare_sample_count": observed_rare,
                "rare_region_mass_ratio": 0.0 if observed_rare == 0 else 0.9,
                "target_rare_mass": 1e-3,
                "q_rare_region_mass": 0.0 if observed_rare == 0 else 9e-4,
            }
        (run_dir / "metrics.json").write_text(json.dumps(metrics))


def test_load_run_records_carries_device(tmp_path):
    _write_run(tmp_path, "run_cpu", device="cpu", seed=11)
    records = load_run_records(tmp_path)
    assert records[0]["device"] == "cpu"


def test_device_runs_are_not_collapsed_in_aggregate(tmp_path):
    # Same config on two backends must not be counted as 2 seeds of one group.
    _write_run(tmp_path, "run_cpu", device="cpu", seed=11, fkl=0.10)
    _write_run(tmp_path, "run_cuda", device="cuda", seed=11, fkl=0.40)
    records = load_run_records(tmp_path)
    summary = build_summary_tables(records, tmp_path / "report")
    # two distinct device groups, each with a single seed -- not one 2-seed group
    assert len(summary["aggregate"]) == 2
    devices = {row["device"] for row in summary["aggregate"]}
    assert devices == {"cpu", "cuda"}
    for row in summary["aggregate"]:
        assert row["n_seeds"] == 1


def test_same_device_multiple_seeds_group_together(tmp_path):
    _write_run(tmp_path, "run_s11", device="cpu", seed=11, fkl=0.10)
    _write_run(tmp_path, "run_s22", device="cpu", seed=22, fkl=0.20)
    records = load_run_records(tmp_path)
    summary = build_summary_tables(records, tmp_path / "report")
    assert len(summary["aggregate"]) == 1
    assert summary["aggregate"][0]["n_seeds"] == 2
    assert summary["aggregate"][0]["forward_kl_mean_noncatastrophic"] == pytest.approx(0.15)


def test_failed_runs_remain_visible(tmp_path):
    _write_run(tmp_path, "ok", device="cpu", seed=11)
    _write_run(tmp_path, "bad", device="cpu", seed=22, status="failed")
    records = load_run_records(tmp_path)
    summary = build_summary_tables(records, tmp_path / "report")
    assert summary["n_failed"] == 1
    csv_text = (tmp_path / "report" / "benchmark_summary.csv").read_text()
    assert "failed" in csv_text  # failed run row is present, not dropped


def test_build_report_reads_only_without_plots(tmp_path):
    _write_run(tmp_path, "run_cpu", device="cpu", seed=11)
    result = build_report(tmp_path, with_plots=False)
    report_dir = tmp_path / "report"
    for name in (
        "benchmark_summary.json",
        "benchmark_summary.csv",
        "benchmark_summary.md",
        "limitations.md",
        "scientific_gate_summary.json",
        "scientific_gate_summary.md",
    ):
        assert (report_dir / name).exists(), name
    assert result["summary"]["n_completed"] == 1


def test_technical_and_scientific_status_are_separate_columns(tmp_path):
    # A technically completed but scientifically catastrophic run.
    _write_run(
        tmp_path,
        "cat",
        device="cpu",
        seed=11,
        target_id="D5",
        variant="rare_1e-3",
        scientific_status="catastrophic",
        reasons=[{"gate_id": "d5_zero_rare_samples", "threshold_class": "catastrophic_guard"}],
        observed_rare=0,
    )
    records = load_run_records(tmp_path)
    assert records[0]["technical_status"] == "completed"
    assert records[0]["scientific_status"] == "catastrophic"
    build_summary_tables(records, tmp_path / "report")
    csv_text = (tmp_path / "report" / "benchmark_summary.csv").read_text()
    header = csv_text.splitlines()[0]
    assert "technical_status" in header and "scientific_status" in header
    assert "completed" in csv_text and "catastrophic" in csv_text


def test_scientific_gate_summary_counts_all_statuses(tmp_path):
    _write_run(tmp_path, "p1", device="cpu", seed=11, scientific_status="pass")
    _write_run(
        tmp_path,
        "c1",
        device="cpu",
        seed=22,
        target_id="D5",
        variant="rare_1e-3",
        scientific_status="catastrophic",
        reasons=[{"gate_id": "d5_zero_rare_samples", "threshold_class": "catastrophic_guard"}],
        observed_rare=0,
    )
    _write_run(tmp_path, "i1", device="cpu", seed=33, scientific_status="inconclusive")
    records = load_run_records(tmp_path)
    summary = build_scientific_gate_summary(records, tmp_path / "report")
    assert summary["scientific_status_counts"]["pass"] == 1
    assert summary["scientific_status_counts"]["catastrophic"] == 1
    assert summary["scientific_status_counts"]["inconclusive"] == 1
    assert summary["n_d5_zero_rare"] == 1
    md = (tmp_path / "report" / "scientific_gate_summary.md").read_text()
    for token in ("pass", "catastrophic", "inconclusive", "d5_zero_rare_samples"):
        assert token in md


def test_catastrophic_runs_not_removed_from_summary(tmp_path):
    # Two runs in the SAME aggregate group (same target/model/device): one
    # passing, one catastrophic. The catastrophic run must stay visible and must
    # not contaminate the non-catastrophic mean.
    _write_run(
        tmp_path,
        "ok",
        device="cpu",
        seed=11,
        fkl=0.1,
        target_id="D5",
        variant="rare_1e-3",
        scientific_status="pass",
        observed_rare=30,
    )
    _write_run(
        tmp_path,
        "bad",
        device="cpu",
        seed=22,
        fkl=5.0,
        ess=0.001,
        target_id="D5",
        variant="rare_1e-3",
        scientific_status="catastrophic",
        reasons=[{"gate_id": "importance_ess_catastrophic", "threshold_class": "catastrophic_guard"}],
        observed_rare=0,
    )
    records = load_run_records(tmp_path)
    summary = build_summary_tables(records, tmp_path / "report")
    # The catastrophic run is present in the CSV (not silently dropped)...
    csv_text = (tmp_path / "report" / "benchmark_summary.csv").read_text()
    assert "bad" not in csv_text  # run_id not a column, but the row exists:
    assert csv_text.count("cpu") == 2  # two data rows
    # ...and its metrics never contaminate the clean (non-catastrophic) mean.
    row = summary["aggregate"][0]
    assert row["n_scientific_catastrophic"] == 1
    assert row["forward_kl_mean_noncatastrophic"] == pytest.approx(0.1)


def test_gate_config_hash_carried_into_records(tmp_path):
    _write_run(tmp_path, "r", device="cpu", seed=11)
    records = load_run_records(tmp_path)
    assert records[0]["gate_config_hash"] == "deadbeef"


def test_plot_series_key_separates_scope_and_aggregates_only_matching_seeds(tmp_path):
    common = {"device": "cpu", "target_id": "D5", "variant": "rare_1e-3"}
    _write_run(tmp_path, "base_s11", seed=11, fkl=0.1, **common)
    _write_run(tmp_path, "base_s22", seed=22, fkl=0.3, **common)
    _write_run(
        tmp_path, "before_d4", seed=11, fkl=0.4,
        target_stage="base_before_d4", **common
    )
    _write_run(
        tmp_path, "stratified", seed=11, fkl=0.5,
        sampling_regime="exact_stratified", **common
    )
    _write_run(
        tmp_path, "diagnostic", seed=11, fkl=0.6,
        diagnostic_only=True, **common
    )

    records = load_run_records(tmp_path)
    summary = build_summary_tables(records, tmp_path / "report")
    assert len(summary["aggregate"]) == 4

    base_key = PlotSeriesKey("D5-1e-3", "transformed", "iid_target", False)
    stats = {key: (mean, std) for key, mean, std in _series_metric_stats(records, "forward_kl")}
    assert len(stats) == 4
    assert stats[base_key] == pytest.approx((0.2, 0.1))
    assert "DIAGNOSTIC ONLY" in _plot_series_label(
        PlotSeriesKey("D5-1e-3", "transformed", "iid_target", True)
    )

    base_aggregate = [
        row for row in summary["aggregate"]
        if (
            row["target_label"], row["target_stage"], row["sampling_regime"],
            row["diagnostic_only"],
        ) == base_key
    ]
    assert len(base_aggregate) == 1
    assert base_aggregate[0]["n_seeds"] == 2
    assert base_aggregate[0]["forward_kl_mean_noncatastrophic"] == pytest.approx(0.2)


@pytest.mark.lab
def test_build_report_plots_all_use_scoped_series_without_error(tmp_path):
    pytest.importorskip("matplotlib")
    common = {
        "device": "cpu",
        "target_id": "D5",
        "variant": "rare_1e-3",
        "observed_rare": 10,
    }
    _write_run(tmp_path, "identity", seed=11, parameter_count=100, **common)
    _write_run(
        tmp_path, "cylindrical", seed=22, parameter_count=200,
        feature_view="cylindrical_v0", diagnostic_only=True, **common
    )

    result = build_report(tmp_path, with_plots=True)
    assert {Path(path).name for path in result["plots"]} == {
        "quality_by_target.png",
        "rare_mode_recovery.png",
        "feature_view_comparison.png",
        "capacity_frontier.png",
    }
    assert not (tmp_path / "report" / "plots_error.txt").exists()
