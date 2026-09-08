import json
from pathlib import Path

import pytest

from ship_muon_bg.benchmarks.fairship_contract import (
    ARM_ORDER,
    SCHEMA_VERSION,
    BenchmarkConfig,
    build_report,
    load_config,
    summarize_fairship_outcomes,
    validate_report,
)


CONFIG = Path(__file__).parents[1] / "configs" / "utility_guided_fairship_benchmark_v0.json"


def test_versioned_config_and_identical_three_arm_shape() -> None:
    config = load_config(CONFIG)
    report = build_report(config)
    validate_report(report)
    assert tuple(report["arms"]) == ARM_ORDER
    assert {tuple(value) for value in report["arms"].values()} == {
        ("arm_id", "generation", "utility_tilt", "proposal_fidelity", "fairship_outcomes")
    }
    assert report["arms"]["PU_DIRECT"]["generation"]["sampling_method"] == "direct_pu_sampling"
    assert report["arms"]["Q_THETA"]["generation"]["sampling_method"] == "learned_q_theta_generation"
    assert report["weight_policy"]["distinct_objects"] is True
    assert report["lineage_axes"]["intermediate_Y_k_is_not_endpoint_B"] is True


def test_predeclared_cohort_and_fairship_outcome_censoring() -> None:
    config = BenchmarkConfig("b", "fs", "geo", "state")
    records = [
        {"candidate_id": "a", "execution_status": "technical_failure", "failure_reason": "timeout"},
        {"candidate_id": "b", "execution_status": "succeeded", "physics_outcome": "physics_rejection"},
        {"candidate_id": "c", "execution_status": "succeeded", "physics_outcome": "accepted_candidate"},
    ]
    outcomes = summarize_fairship_outcomes(records)
    report = build_report(config, {
        "P0": {
            "cohort": {"cohort_id": "p0", "candidate_ids": ["a", "b", "c"], "predeclared": True},
            "fairship_outcomes": outcomes,
        }
    })
    assert report["arms"]["P0"]["fairship_outcomes"]["valid_execution_count"] == 2
    assert report["arms"]["P0"]["fairship_outcomes"]["technical_failure_count"] == 1
    assert report["arms"]["P0"]["fairship_outcomes"]["accepted_candidate_count"] == 1
    assert report["arms"]["P0"]["generation"]["candidate_count"] == 3


def test_technical_failure_cannot_be_a_physics_negative() -> None:
    with pytest.raises(ValueError, match="must not carry"):
        summarize_fairship_outcomes([{
            "candidate_id": "a",
            "execution_status": "technical_failure",
            "physics_outcome": "physics_rejection",
        }])


def test_no_redraw_and_no_universal_ess_threshold() -> None:
    config = load_config(CONFIG)
    with pytest.raises(ValueError, match="redraw"):
        build_report(config, {"P0": {"cohort": {"cohort_id": "x", "candidate_ids": ["a"], "predeclared": True, "redraw_until_success": True}}})
    report = build_report(config)
    assert report["arms"]["P0"]["utility_tilt"]["ess"] is None
    assert report["arms"]["P0"]["utility_tilt"]["ess_is_diagnostic_only"] is True
    assert "threshold" not in json.dumps(report).lower()
