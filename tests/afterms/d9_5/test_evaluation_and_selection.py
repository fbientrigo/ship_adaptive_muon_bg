"""Validation-only selection freeze + evaluation-budget tests.

Covers required tests 26, 27, 28, 29, 31, 32.
"""

from __future__ import annotations

import pytest

from ship_muon_bg.afterms.d9_5 import aggregation, config as d9_5config, report


def test_validation_primary_metric_is_physical_nll():
    record = aggregation.aggregate_seeds(
        "TRK_PDG13_UW_ID", "GMM_k04_covFULL_d05", [{"seed": 1, "status": "ok", "physical_nll": 1.0}],
    )
    assert record["metric_key"] == "physical_nll"


def test_freeze_manifest_states_validation_nll_is_primary_but_not_only_diagnostic():
    manifest = report.build_family_selection_manifest({"TRK_PDG13_UW_ID": []})
    assert "physical-space mean NLL is primary" in manifest["primary_metric_statement"]
    assert "not the only scientific diagnostic" in manifest["primary_metric_statement"]


def test_selection_is_a_pure_function_of_validation_aggregates_only():
    """Required test 27 (structural): the freeze manifest is built from exactly
    the aggregate records passed in -- there is no code path here that reads a
    test_evaluation directory or any other test-labeled artifact."""

    records = [
        aggregation.deterministic_fit_record("TRK_PDG13_UW_ID", "GAUSS_DIAG_d05", {"physical_nll": 5.0}),
        aggregation.deterministic_fit_record("TRK_PDG13_UW_ID", "GAUSS_FULL_d05", {"physical_nll": 4.0}),
    ]
    manifest = report.build_family_selection_manifest({"TRK_PDG13_UW_ID": records})
    ranking = manifest["tracks"]["TRK_PDG13_UW_ID"]["ranked_by_validation_physical_nll"]
    assert [r["model_config_id"] for r in ranking] == ["GAUSS_FULL_d05", "GAUSS_DIAG_d05"]
    assert [r["value"] for r in ranking] == [4.0, 5.0]


def test_evaluate_test_refuses_before_freeze(tmp_path):
    with pytest.raises(report.SelectionNotFrozenError):
        report.require_frozen_selection(tmp_path / "validation")


def test_evaluate_test_succeeds_after_freeze(tmp_path):
    manifest = report.build_family_selection_manifest({"TRK_PDG13_UW_ID": []})
    out_dir = tmp_path / "validation"
    report.write_freeze_manifest(out_dir, manifest)
    loaded = report.require_frozen_selection(out_dir)
    assert loaded["content_hash"] == manifest["content_hash"]


def test_same_evaluation_budget_used_across_families():
    arena_config = d9_5config.load_model_family_arena_config()
    for family_key in ("gauss_diag", "gauss_full", "gmm", "nf_ac"):
        assert "evaluation_policy" not in arena_config[family_key]
    assert "quick_validation_budget" in arena_config["evaluation_policy"]
    assert "final_candidate_budget" in arena_config["evaluation_policy"]


def test_no_global_cross_track_ranking_key_exists():
    manifest = report.build_family_selection_manifest({
        "TRK_PDG13_UW_ID": [aggregation.deterministic_fit_record("TRK_PDG13_UW_ID", "GAUSS_DIAG_d05", {"physical_nll": 5.0})],
        "TRK_PDGM13_UW_ID": [aggregation.deterministic_fit_record("TRK_PDGM13_UW_ID", "GAUSS_DIAG_d05", {"physical_nll": 1.0})],
    })
    assert set(manifest["tracks"].keys()) == {"TRK_PDG13_UW_ID", "TRK_PDGM13_UW_ID"}
    assert "global_ranking" not in manifest
    assert "ranked_by_validation_physical_nll" not in manifest  # only nested per-track, never at top level


def test_failed_technical_fit_is_reported_as_failed_not_as_a_poor_finite_metric():
    record = aggregation.aggregate_seeds(
        "TRK_PDG13_UW_ID", "GMM_k04_covFULL_d05",
        [{"seed": 20260720, "status": "failed", "physical_nll": None}],
    )
    assert record["completed_seed_count"] == 0
    assert record["failed_seed_count"] == 1
    assert record["mean"] is None
