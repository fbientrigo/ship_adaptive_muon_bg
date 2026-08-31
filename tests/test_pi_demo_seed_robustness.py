"""Tests for ``ship_muon_bg.density_lab.pi_demo_seed_robustness``.

Pure-function tests against small, hand-written arena_summary.json fixtures
-- never touches the real 40k dataset or NF training, runs in well under a
second.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from ship_muon_bg.density_lab import pi_demo_seed_robustness as sr


def _write_arena_report(tmp_path: Path, name: str, rows) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps({"rows": rows}))
    return path


def test_aggregate_reports_mean_std_and_side_consistency(tmp_path):
    variant = "UA_d0p9_a01"
    # Same p0/pU every seed; q oscillates around p0 -- deliberately NOT
    # all on the same side, so all_seeds_same_side_of_p0 must be False.
    reports = {}
    for seed, q in ((11, 0.02), (12, 0.04), (13, 0.03)):
        reports[(13, seed)] = _write_arena_report(
            tmp_path, "pdg13_seed{}.json".format(seed),
            [{
                "variant_id": variant, "nominal_b_toy_prevalence": 0.03,
                "theoretical_target_b_toy_prevalence": 0.035, "generated_b_toy_occupancy": q,
            }],
        )

    rows = sr.aggregate_seed_robustness(
        pdg_ids=[13], variant_ids=[variant], arena_report_path_by_pdg_and_seed=reports,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["n_seeds"] == 3
    assert row["seeds"] == "11;12;13"
    assert row["q_theta_mean"] == pytest.approx((0.02 + 0.04 + 0.03) / 3)
    assert row["q_theta_min"] == pytest.approx(0.02)
    assert row["q_theta_max"] == pytest.approx(0.04)
    assert row["all_seeds_same_side_of_p0"] is False


def test_aggregate_detects_consistent_side(tmp_path):
    variant = "UP_d0p9_a01"
    reports = {}
    for seed, q in ((11, 0.005), (12, 0.006), (13, 0.004)):
        reports[(13, seed)] = _write_arena_report(
            tmp_path, "seed{}.json".format(seed),
            [{
                "variant_id": variant, "nominal_b_toy_prevalence": 0.03,
                "theoretical_target_b_toy_prevalence": 0.032, "generated_b_toy_occupancy": q,
            }],
        )
    rows = sr.aggregate_seed_robustness(
        pdg_ids=[13], variant_ids=[variant], arena_report_path_by_pdg_and_seed=reports,
    )
    assert rows[0]["all_seeds_same_side_of_p0"] is True  # every q < p0


def test_aggregate_rejects_missing_pdg(tmp_path):
    reports = {
        (13, 11): _write_arena_report(tmp_path, "a.json", [{
            "variant_id": "UA_d0p9_a01", "nominal_b_toy_prevalence": 0.03,
            "theoretical_target_b_toy_prevalence": 0.03, "generated_b_toy_occupancy": 0.03,
        }]),
    }
    with pytest.raises(ValueError, match="no arena reports supplied"):
        sr.aggregate_seed_robustness(
            pdg_ids=[13, -13], variant_ids=["UA_d0p9_a01"],
            arena_report_path_by_pdg_and_seed=reports,
        )


def test_aggregate_rejects_uneven_seed_counts(tmp_path):
    row = {
        "variant_id": "UA_d0p9_a01", "nominal_b_toy_prevalence": 0.03,
        "theoretical_target_b_toy_prevalence": 0.03, "generated_b_toy_occupancy": 0.03,
    }
    reports = {
        (13, 11): _write_arena_report(tmp_path, "a.json", [row]),
        (13, 12): _write_arena_report(tmp_path, "b.json", [row]),
        (-13, 11): _write_arena_report(tmp_path, "c.json", [row]),
    }
    with pytest.raises(ValueError, match="uneven seed counts"):
        sr.aggregate_seed_robustness(
            pdg_ids=[13, -13], variant_ids=["UA_d0p9_a01"],
            arena_report_path_by_pdg_and_seed=reports,
        )


def test_aggregate_rejects_missing_variant(tmp_path):
    reports = {
        (13, 11): _write_arena_report(tmp_path, "a.json", [{
            "variant_id": "UA_d0p9_a01", "nominal_b_toy_prevalence": 0.03,
            "theoretical_target_b_toy_prevalence": 0.03, "generated_b_toy_occupancy": 0.03,
        }]),
    }
    with pytest.raises(ValueError, match="missing from"):
        sr.aggregate_seed_robustness(
            pdg_ids=[13], variant_ids=["UP_d0p1_a04"],
            arena_report_path_by_pdg_and_seed=reports,
        )


def test_write_seed_robustness_csv_round_trips(tmp_path):
    reports = {
        (13, 11): _write_arena_report(tmp_path, "a.json", [{
            "variant_id": "UA_d0p9_a01", "nominal_b_toy_prevalence": 0.03,
            "theoretical_target_b_toy_prevalence": 0.032, "generated_b_toy_occupancy": 0.028,
        }]),
    }
    rows = sr.aggregate_seed_robustness(
        pdg_ids=[13], variant_ids=["UA_d0p9_a01"], arena_report_path_by_pdg_and_seed=reports,
    )
    out_path = tmp_path / "seed_robustness_summary.csv"
    sr.write_seed_robustness_csv(rows, out_path)

    with out_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(sr.CSV_FIELDNAMES)
        written = list(reader)
    assert len(written) == 1
    assert written[0]["variant_id"] == "UA_d0p9_a01"
    assert written[0]["n_seeds"] == "1"
