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
HASH = "a" * 64


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
    config = BenchmarkConfig("b", "fs", "geo", "state", downstream_endpoint_definition_id="endpoint-B-v0")
    def refs(candidate, decision):
        return {"source_state_ref": "source-" + candidate, "execution_ref": "exec-" + candidate,
                "interaction_realization_refs": [] if decision is None else ["interaction-" + candidate],
                "observation_refs": [] if decision is None else ["observation-" + candidate], "decision_ref": decision}
    records = [
        {"candidate_id": "aa", "execution_status": "technical_failure", "failure_reason": "timeout",
         "lineage_refs": refs("aa", None)},
        {"candidate_id": "bb", "execution_status": "succeeded", "physics_outcome": "physics_rejection", "endpoint_status": "evaluated", "endpoint_definition_id": "endpoint-B-v0",
         "lineage_refs": refs("bb", "decision-b")},
        {"candidate_id": "cc", "execution_status": "succeeded", "physics_outcome": "accepted_candidate", "endpoint_status": "evaluated", "endpoint_definition_id": "endpoint-B-v0",
         "lineage_refs": refs("cc", "decision-c")},
    ]
    outcomes = summarize_fairship_outcomes(records)
    report = build_report(config, {
        "P0": {
            "cohort": {"cohort_id": "p0", "candidate_ids": ["aa", "bb", "cc"], "predeclared": True,
                       "manifest_ref": "cohorts/p0.json", "manifest_sha256": HASH,
                       "candidate_provenance": {"source_state_definition_id": "state", "dataset_hash": HASH, "candidate_table_ref": "candidates-p0-csv"},
                       "proposal_provenance": {"proposal_id": "p0", "proposal_version": "v0", "checkpoint_id": None}},
            "fairship_outcomes": outcomes,
        }
    })
    assert report["arms"]["P0"]["fairship_outcomes"]["valid_execution_count"] == 2
    assert report["arms"]["P0"]["fairship_outcomes"]["technical_failure_count"] == 1
    assert report["arms"]["P0"]["fairship_outcomes"]["accepted_candidate_count"] == 1
    assert report["arms"]["P0"]["generation"]["candidate_count"] == 3
    assert report["arms"]["PU_DIRECT"]["proposal_fidelity"]["declared_target_measure"] == "PU"
    assert report["arms"]["Q_THETA"]["proposal_fidelity"]["declared_target_measure"] == "PU"


def test_technical_failure_cannot_be_a_physics_negative() -> None:
    with pytest.raises(ValueError, match="must not carry"):
        summarize_fairship_outcomes([{
            "candidate_id": "a",
            "execution_status": "technical_failure",
            "physics_outcome": "physics_rejection",
            "lineage_refs": {"source_state_ref": "s", "execution_ref": "e", "interaction_realization_refs": [], "observation_refs": [], "decision_ref": None},
        }])


def test_successful_execution_can_have_unavailable_endpoint() -> None:
    refs = {"source_state_ref": "source-a", "execution_ref": "exec-a",
            "interaction_realization_refs": ["interaction-a"], "observation_refs": ["Y-k-a"], "decision_ref": None}
    result = summarize_fairship_outcomes([{"candidate_id": "aa", "execution_status": "succeeded",
                                           "endpoint_status": "unavailable", "lineage_refs": refs}])
    assert result["valid_execution_count"] == 1
    assert result["endpoint_evaluated_count"] == 0
    assert result["unevaluated_executions"][0]["lineage_refs"]["observation_refs"] == ["Y-k-a"]


def test_technical_failure_cannot_carry_child_refs_and_healthy_execution_may_have_zero_children() -> None:
    with pytest.raises(ValueError, match="technical_failure must not carry"):
        summarize_fairship_outcomes([{"candidate_id": "aa", "execution_status": "technical_failure",
                                      "lineage_refs": {"source_state_ref": "source-aa", "execution_ref": "exec-aa",
                                                       "interaction_realization_refs": ["interaction-aa"],
                                                       "observation_refs": [], "decision_ref": None}}])
    result = summarize_fairship_outcomes([{"candidate_id": "aa", "execution_status": "succeeded",
                                           "lineage_refs": {"source_state_ref": "source-aa", "execution_ref": "exec-aa",
                                                            "interaction_realization_refs": [], "observation_refs": [], "decision_ref": None}}])
    assert result["valid_execution_count"] == 1


def test_no_redraw_and_no_universal_ess_threshold() -> None:
    config = load_config(CONFIG)
    with pytest.raises(ValueError, match="redraw"):
        build_report(config, {"P0": {"cohort": {"cohort_id": "x", "candidate_ids": ["a"], "predeclared": True, "redraw_until_success": True}}})
    report = build_report(config)
    assert report["arms"]["P0"]["utility_tilt"]["ess"] is None
    assert report["arms"]["P0"]["utility_tilt"]["ess_is_diagnostic_only"] is True
    assert "threshold" not in json.dumps(report).lower()
    with pytest.raises(ValueError, match="not_run"):
        build_report(config, {"P0": {"fairship_outcomes": {"candidate_count": 0}}})


def test_computed_outcomes_need_cohort_and_counts_cannot_lie() -> None:
    config = BenchmarkConfig("b", "fs", "geo", "state", downstream_endpoint_definition_id="endpoint-B-v0")
    outcomes = {"status": "computed", "candidate_count": 2, "valid_execution_count": 2,
                "technical_failure_count": 0, "physics_rejection_count": 1,
                "accepted_candidate_count": 0, "technical_failures": [],
                "endpoint_evaluated_count": 1, "unevaluated_executions": [],
                "physics_outcomes": [{"candidate_id": "aa", "outcome": "physics_rejection", "endpoint_definition_id": "endpoint-B-v0",
                                       "lineage_refs": {"source_state_ref": "s", "execution_ref": "e", "interaction_realization_refs": ["i"], "observation_refs": ["o"], "decision_ref": "d"}}]}
    with pytest.raises(ValueError, match="predeclared cohort"):
        build_report(config, {"P0": {"fairship_outcomes": outcomes}})
    with pytest.raises(ValueError, match="valid_execution_count"):
        build_report(config, {"P0": {"cohort": {"cohort_id": "p0", "candidate_ids": ["aa", "bb"], "predeclared": True,
                                                   "manifest_ref": "p0.json", "manifest_sha256": HASH,
                                                   "candidate_provenance": {"source_state_definition_id": "state", "dataset_hash": HASH, "candidate_table_ref": "candidate-table"},
                                                   "proposal_provenance": {"proposal_id": "pp", "proposal_version": "vv", "checkpoint_id": None}},
                                      "fairship_outcomes": outcomes}})


def test_unknown_report_fields_and_qtheta_target_are_rejected() -> None:
    config = BenchmarkConfig("b", "fs", "geo", "state")
    with pytest.raises(ValueError, match="unknown Q_THETA proposal_fidelity"):
        build_report(config, {"Q_THETA": {"proposal_fidelity": {"invented": 1}}})
    report = build_report(config)
    report["arms"]["Q_THETA"]["proposal_fidelity"]["comparison_target"] = "Q_THETA"
    with pytest.raises(ValueError, match="declared PU"):
        validate_report(report)


def test_technical_and_physics_buckets_cannot_overlap() -> None:
    config = BenchmarkConfig("b", "fs", "geo", "state", downstream_endpoint_definition_id="endpoint-B-v0")
    cohort = {"cohort_id": "p0", "candidate_ids": ["aa"], "predeclared": True,
              "manifest_ref": "p0.json", "manifest_sha256": HASH,
              "candidate_provenance": {"source_state_definition_id": "state", "dataset_hash": HASH, "candidate_table_ref": "candidate-table"},
              "proposal_provenance": {"proposal_id": "pp", "proposal_version": "vv", "checkpoint_id": None}}
    outcomes = {"status": "computed", "candidate_count": 2, "valid_execution_count": 1,
                "technical_failure_count": 1, "endpoint_evaluated_count": 1, "physics_rejection_count": 1,
                "accepted_candidate_count": 0,
                "technical_failures": [{"candidate_id": "aa", "reason": "timeout", "lineage_refs": {}}], "unevaluated_executions": [],
                "physics_outcomes": [{"candidate_id": "aa", "outcome": "physics_rejection", "endpoint_definition_id": "endpoint-B-v0", "lineage_refs": {}}]}
    with pytest.raises(ValueError, match="disjoint"):
        build_report(config, {"P0": {"cohort": cohort, "fairship_outcomes": outcomes}})


def test_predeclared_generation_count_must_match_ids() -> None:
    report = build_report(load_config(CONFIG))
    generation = report["arms"]["P0"]["generation"]
    generation.update({"predeclared_cohort": True, "cohort_manifest_ref": "manifest-v0", "cohort_manifest_sha256": HASH,
                       "candidate_ids": ["aa"], "candidate_count": 2,
                       "candidate_provenance": {"source_state_definition_id": "afterms_5d_nf_v0", "dataset_hash": HASH, "candidate_table_ref": "candidate-table"},
                       "proposal_provenance": {"proposal_id": "p0", "proposal_version": "v0", "checkpoint_id": None}})
    with pytest.raises(ValueError, match="candidate_count"):
        validate_report(report)
