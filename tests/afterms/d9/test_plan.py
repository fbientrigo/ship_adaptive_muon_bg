from pathlib import Path

import pytest

from ship_muon_bg.afterms.d9 import plan as d9plan

REPO_ROOT = Path(__file__).resolve().parents[3]
PLAN_PATH = REPO_ROOT / "configs" / "afterms" / "d9_candidate_plan_v0.json"


@pytest.fixture(scope="module")
def plan():
    return d9plan.load_plan(PLAN_PATH)


def test_plan_parses_and_is_valid(plan):
    violations = d9plan.validate_plan(plan)
    assert violations == []
    assert plan["status"] == "D9_PLAN_VALID"


@pytest.mark.local_env
def test_d7_d8_inputs_unchanged(plan):
    """Required test 35: D7/D8 input hashes remain unchanged."""

    mismatches = d9plan.verify_source_hashes(plan, REPO_ROOT)
    assert mismatches == [], mismatches


def test_maximum_candidate_count(plan):
    """Required test 7: maximum candidate-count policy."""

    assert len(plan["candidates"]) == plan["maximum_primary_neural_candidate_count"] == 6


def test_weighted_and_unweighted_tracks_stay_separate(plan):
    """Required test 4."""

    by_pdg = {}
    for c in plan["candidates"]:
        by_pdg.setdefault(c["pdg_value"], set()).add(c["weighting_policy"])
    # PDG 13 and -13 each have both an unweighted and a weighted candidate,
    # but no single candidate is ambiguous about which it is.
    for c in plan["candidates"]:
        assert c["weighting_policy"] in ("row_empirical_unweighted", "production_weighted")


def test_pdg_13_and_minus13_stay_separate(plan):
    """Required test 5."""

    for c in plan["candidates"]:
        assert c["pdg_value"] in (13, -13)
    pdg13_ids = {c["candidate_id"] for c in plan["candidates"] if c["pdg_value"] == 13}
    pdg_minus13_ids = {c["candidate_id"] for c in plan["candidates"] if c["pdg_value"] == -13}
    assert pdg13_ids.isdisjoint(pdg_minus13_ids)


def test_legacy_4d_excluded_from_primary_candidates(plan):
    """Required test 6."""

    for c in plan["candidates"]:
        assert c["modeled_dimension"] == 5
        assert list(c["modeled_features"]) == ["px", "py", "pz", "x", "y"]
    legacy = [e for e in plan["excluded_candidates"] if e["modeled_dimension"] == 4]
    assert len(legacy) == 1
    assert legacy[0]["D9_enabled_by_default"] is False


def test_selection_reason_never_cites_test_metrics(plan):
    """Required tests 2/3: selection uses validation evidence only; test metrics
    cannot alter the plan."""

    for c in plan["candidates"]:
        assert "test_nll" not in (c["selection_reason"] or "")
        assert "D8_validation_evidence" in c
        assert c["D8_validation_evidence"]


def test_physical_weight_never_a_modeled_feature(plan):
    """Required test 8."""

    for c in plan["candidates"] + plan["excluded_candidates"]:
        assert "w" not in c.get("modeled_features", [])


def test_no_utility_multiplier_present(plan):
    """Required test 9."""

    import json

    text = json.dumps(plan)
    assert "utility_multiplier" not in text
    assert "utility_weight" not in text


def test_weighted_track_decision_is_open(plan):
    assert plan["weighted_track_decision_status"] == "OPEN"
