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
    def refs(candidate, decision):
        return {"source_state_ref": "source-" + candidate, "execution_ref": "exec-" + candidate,
                "interaction_realization_refs": ["interaction-" + candidate],
                "observation_refs": ["observation-" + candidate], "decision_ref": decision}
    records = [
        {"candidate_id": "a", "execution_status": "technical_failure", "failure_reason": "timeout",
         "lineage_refs": refs("a", None)},
        {"candidate_id": "b", "execution_status": "succeeded", "physics_outcome": "physics_rejection",
         "lineage_refs": refs("b", "decision-b")},
        {"candidate_id": "c", "execution_status": "succeeded", "physics_outcome": "accepted_candidate",
         "lineage_refs": refs("c", "decision-c")},
    ]
    outcomes = summarize_fairship_outcomes(records)
    report = build_report(config, {
        "P0": {
            "cohort": {"cohort_id": "p0", "candidate_ids": ["a", "b", "c"], "predeclared": True,
                       "manifest_ref": "cohorts/p0.json", "manifest_sha256": "hash-p0",
                       "candidate_provenance": {"source_state_definition_id": "state", "dataset_hash": "data-hash", "candidate_table_ref": "candidates-p0.csv"},
                       "proposal_provenance": {"proposal_id": "p0", "proposal_version": "v0", "checkpoint_id": None}},
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
            "lineage_refs": {"source_state_ref": "s", "execution_ref": "e", "interaction_realization_refs": [], "observation_refs": [], "decision_ref": None},
        }])


def test_no_redraw_and_no_universal_ess_threshold() -> None:
    config = load_config(CONFIG)
    with pytest.raises(ValueError, match="redraw"):
        build_report(config, {"P0": {"cohort": {"cohort_id": "x", "candidate_ids": ["a"], "predeclared": True, "redraw_until_success": True}}})
    report = build_report(config)
    assert report["arms"]["P0"]["utility_tilt"]["ess"] is None
    assert report["arms"]["P0"]["utility_tilt"]["ess_is_diagnostic_only"] is True
    assert "threshold" not in json.dumps(report).lower()


def test_computed_outcomes_need_cohort_and_counts_cannot_lie() -> None:
    config = BenchmarkConfig("b", "fs", "geo", "state")
    outcomes = {"status": "computed", "candidate_count": 2, "valid_execution_count": 2,
                "technical_failure_count": 0, "physics_rejection_count": 1,
                "accepted_candidate_count": 0, "technical_failures": [],
                "physics_outcomes": [{"candidate_id": "a", "outcome": "physics_rejection", "lineage_refs": {}}]}
    with pytest.raises(ValueError, match="predeclared cohort"):
        build_report(config, {"P0": {"fairship_outcomes": outcomes}})
    with pytest.raises(ValueError, match="valid_execution_count"):
        build_report(config, {"P0": {"cohort": {"cohort_id": "p0", "candidate_ids": ["a", "b"], "predeclared": True,
                                                   "manifest_ref": "p0.json", "manifest_sha256": "h",
                                                   "candidate_provenance": {"source_state_definition_id": "s", "dataset_hash": "d", "candidate_table_ref": "c"},
                                                   "proposal_provenance": {"proposal_id": "p", "proposal_version": "v", "checkpoint_id": None}},
                                      "fairship_outcomes": outcomes}})


def test_unknown_report_fields_and_qtheta_target_are_rejected() -> None:
    config = BenchmarkConfig("b", "fs", "geo", "state")
    with pytest.raises(ValueError, match="unknown Q_THETA proposal_fidelity"):
        build_report(config, {"Q_THETA": {"proposal_fidelity": {"invented": 1}}})
    report = build_report(config)
    report["arms"]["Q_THETA"]["proposal_fidelity"]["comparison_target"] = "Q_THETA"
    with pytest.raises(ValueError, match="declared PU_DIRECT"):
        validate_report(report)


def test_technical_and_physics_buckets_cannot_overlap() -> None:
    config = BenchmarkConfig("b", "fs", "geo", "state")
    cohort = {"cohort_id": "p0", "candidate_ids": ["a"], "predeclared": True,
              "manifest_ref": "p0.json", "manifest_sha256": "h",
              "candidate_provenance": {"source_state_definition_id": "s", "dataset_hash": "d", "candidate_table_ref": "c"},
              "proposal_provenance": {"proposal_id": "p", "proposal_version": "v", "checkpoint_id": None}}
    outcomes = {"status": "computed", "candidate_count": 1, "valid_execution_count": 1,
                "technical_failure_count": 0, "physics_rejection_count": 1,
                "accepted_candidate_count": 0,
                "technical_failures": [{"candidate_id": "a", "reason": "timeout", "lineage_refs": {}}],
                "physics_outcomes": [{"candidate_id": "a", "outcome": "physics_rejection", "lineage_refs": {}}]}
    with pytest.raises(ValueError, match="disjoint"):
        build_report(config, {"P0": {"cohort": cohort, "fairship_outcomes": outcomes}})
