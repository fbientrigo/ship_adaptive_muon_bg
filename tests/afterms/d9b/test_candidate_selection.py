import json
from pathlib import Path

import pytest

from ship_muon_bg.afterms.d9b import candidate_selection as cs

REPO_ROOT = Path(__file__).resolve().parents[3]
PLAN_PATH = REPO_ROOT / "configs" / "afterms" / "d9_candidate_plan_v0.json"
TRAINING_CONFIG_PATH = REPO_ROOT / "configs" / "afterms" / "d9_training_v0.json"


def _real_plan_and_config():
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    training_config = json.loads(TRAINING_CONFIG_PATH.read_text(encoding="utf-8"))
    return plan, training_config


def _synthetic_candidate(candidate_id, *, params, enabled=True, reconstructible=True,
                          unweighted=True, dimension=5):
    return {
        "candidate_id": candidate_id,
        "parameter_count": params,
        "D9_enabled_by_default": enabled,
        "reconstruction_status": "RECONSTRUCTIBLE" if reconstructible else "MISSING_HISTORICAL_CHECKPOINT",
        "weighting_policy": "row_empirical_unweighted" if unweighted else "production_weighted",
        "modeled_dimension": dimension,
    }


def test_gpu_candidate_derived_from_real_plan_is_smallest_unweighted_modern_5d():
    """Required test 1: GPU candidate derived from the plan, not hardcoded."""

    plan, training_config = _real_plan_and_config()
    candidate_config, report = cs.select_gpu_gate_candidate(plan, training_config)
    assert candidate_config["candidate_id"] == report["selected_candidate_id"]
    assert candidate_config["weighting_policy"] == "row_empirical_unweighted"
    assert candidate_config["modeled_features"] == ["px", "py", "pz", "x", "y"]
    assert report["selected_candidate_id"] == min(
        report["eligible_parameter_counts"], key=lambda cid: (report["eligible_parameter_counts"][cid], cid)
    )


def test_selection_never_hardcodes_a_single_id_it_recomputes_from_plan_fields():
    plan, training_config = _real_plan_and_config()
    eligible = cs.eligible_plan_candidates(plan)
    assert len(eligible) >= 2, "test assumes the real plan has more than one eligible candidate"
    ids = {c["candidate_id"] for c in eligible}
    assert ids == {
        "A1_capacity_medium_identity_pdg13_unweighted",
        "A2_capacity_small_cartesian_pdg13_unweighted",
        "B1_capacity_small_cartesian_pdg_minus13_unweighted",
        "B2_capacity_small_identity_pdg_minus13_unweighted",
    }


def test_three_way_tie_breaks_on_candidate_id_ascending():
    plan = {"candidates": [
        _synthetic_candidate("Z_small", params=100),
        _synthetic_candidate("A_small", params=100),
        _synthetic_candidate("M_small", params=100),
        _synthetic_candidate("tiny_but_bigger", params=200),
    ]}
    selected, report = cs.select_gpu_gate_plan_candidate(plan)
    assert selected["candidate_id"] == "A_small"
    assert report["tie_break_applied"] is True
    assert set(report["tied_at_minimum_candidate_ids"]) == {"Z_small", "A_small", "M_small"}


def test_excludes_disabled_candidate():
    plan = {"candidates": [
        _synthetic_candidate("disabled_smallest", params=10, enabled=False),
        _synthetic_candidate("enabled_larger", params=50),
    ]}
    selected, _ = cs.select_gpu_gate_plan_candidate(plan)
    assert selected["candidate_id"] == "enabled_larger"


def test_excludes_non_reconstructible_candidate():
    plan = {"candidates": [
        _synthetic_candidate("broken_smallest", params=10, reconstructible=False),
        _synthetic_candidate("ok_larger", params=50),
    ]}
    selected, _ = cs.select_gpu_gate_plan_candidate(plan)
    assert selected["candidate_id"] == "ok_larger"


def test_excludes_weighted_candidate():
    plan = {"candidates": [
        _synthetic_candidate("weighted_smallest", params=10, unweighted=False),
        _synthetic_candidate("unweighted_larger", params=50),
    ]}
    selected, _ = cs.select_gpu_gate_plan_candidate(plan)
    assert selected["candidate_id"] == "unweighted_larger"


def test_excludes_legacy_4d_candidate():
    plan = {"candidates": [
        _synthetic_candidate("legacy_4d_smallest", params=10, dimension=4),
        _synthetic_candidate("modern_5d_larger", params=50),
    ]}
    selected, _ = cs.select_gpu_gate_plan_candidate(plan)
    assert selected["candidate_id"] == "modern_5d_larger"


def test_no_eligible_candidate_raises():
    plan = {"candidates": [_synthetic_candidate("disabled_only", params=10, enabled=False)]}
    with pytest.raises(cs.NoEligibleCandidateError):
        cs.select_gpu_gate_plan_candidate(plan)


def test_selected_plan_candidate_must_exist_in_training_config():
    plan = {"candidates": [_synthetic_candidate("not_in_training_config", params=10)]}
    training_config = {"candidates": []}
    with pytest.raises(cs.NoEligibleCandidateError):
        cs.select_gpu_gate_candidate(plan, training_config)
