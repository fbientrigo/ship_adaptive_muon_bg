import csv
from pathlib import Path

from ship_muon_bg.benchmarks.fairship_contract import load_config
from ship_muon_bg.benchmarks.fairship_endpoint import (
    SBT_OBSERVATION_DEFINITION_ID,
    SBT_STAGE,
    SBT_STAGE_DEFINITION_ID,
)
from ship_muon_bg.entities.decision import DecisionEvaluationStatus
from ship_muon_bg.entities.lineage import ExecutionStatus, FSSimExecution
from ship_muon_bg.entities.observation import (
    ObservationEnvelope,
    ObservationEvaluationStatus,
    ScalarObservationPayload,
)
from ship_muon_bg.entities.subject import TagSubject
from ship_muon_bg.tagging.evaluator import StageEvaluator


AUDIT = Path(__file__).parents[1] / "artifacts" / "fairship_tgeo_audit_v0" / "candidate_audit.csv"
CONFIG = Path(__file__).parents[1] / "configs" / "utility_guided_fairship_benchmark_v0.json"
PINNED_STAGE_ID = "fairship_sbt_selection_v0@sha256:1ac14cb360d7500eb652ede256b17b7cf12aa384e43fc3101e6e49eaa0eb07f2"
PINNED_OBSERVATION_ID = "fairship_sbt_qualifying_hit_count_v0@sha256:669ffed16b93494731888d6e19be37482cfed12708bf0150c729af205fdcb42e"


def test_fixed_audit_is_candidate_level_evidence_only() -> None:
    rows = list(csv.DictReader(AUDIT.open(newline="", encoding="utf-8")))
    by_cohort = {}
    for row in rows:
        by_cohort.setdefault(row["cohort"], []).append(row)

    assert SBT_OBSERVATION_DEFINITION_ID == PINNED_OBSERVATION_ID
    assert SBT_STAGE_DEFINITION_ID == PINNED_STAGE_ID == SBT_STAGE.stage_definition_id
    assert load_config(CONFIG).downstream_endpoint_definition_id == PINNED_STAGE_ID
    assert len(rows) == 14
    assert len({row["candidate_id"] for row in rows}) == 14
    nf = by_cohort["genuine_nf_pilot"]
    controls = by_cohort["empirical_sbt_positive_control"]
    assert len(nf) == 12 and len(controls) == 2
    assert all(row["sbt_qualifying_hit_count"] == "0" for row in nf)
    assert all(row["current_preprocessing_decision"] == "NOT_SBT_SELECTED" for row in nf)
    assert all(float(row["sbt_qualifying_hit_count"]) > 0 for row in controls)
    assert all(row["current_preprocessing_decision"] == "SBT_SELECTED" for row in controls)


def _synthetic_decision(execution: FSSimExecution, hits: float):
    observation = ObservationEnvelope(
        observation_id="obs:" + execution.execution_id,
        subject_ref=execution.execution_id,
        observation_definition_id=SBT_OBSERVATION_DEFINITION_ID,
        units="count",
        evaluation_status=ObservationEvaluationStatus.COMPUTED,
        evidence_reference="synthetic-cbmsim.vetoPoint",
        payload=ScalarObservationPayload(hits),
    )
    return StageEvaluator().evaluate(SBT_STAGE, (observation,), subject_ref=execution.execution_id)


def test_synthetic_execution_contract_groups_repeats_by_source_subject() -> None:
    source = TagSubject("synthetic-source", "synthetic_candidate", "synthetic_state_v0")
    execution = FSSimExecution("exec:synthetic:0", source.subject_id, "fairship-test-v0", ExecutionStatus.SUCCEEDED)
    repeat = FSSimExecution("exec:synthetic:1", source.subject_id, "fairship-test-v0", ExecutionStatus.SUCCEEDED)
    decisions = (_synthetic_decision(execution, 0.0), _synthetic_decision(repeat, 1.0))
    valid = [decision for decision in decisions if decision.evaluation_status is DecisionEvaluationStatus.EVALUATED]

    assert execution.subject_id == repeat.subject_id == source.subject_id
    assert execution.execution_id != execution.subject_id
    assert len(valid) == 2
    assert decisions[0].decision is False and decisions[1].decision is True
    assert {execution.subject_id} == {source.subject_id}


def test_sbt_technical_unavailability_is_not_a_negative() -> None:
    execution = FSSimExecution(
        execution_id="exec:technical",
        subject_id="synthetic-source",
        fs_sim_configuration_id="fairship-test-v0",
        execution_status=ExecutionStatus.TECHNICAL_FAILURE,
        failure_reason="missing cbmsim vetoPoint output",
    )
    observation = ObservationEnvelope(
        observation_id="obs:technical",
        subject_ref=execution.execution_id,
        observation_definition_id=SBT_OBSERVATION_DEFINITION_ID,
        units="count",
        evaluation_status=ObservationEvaluationStatus.TECHNICALLY_UNAVAILABLE,
        evidence_reference="synthetic-missing-cbmsim",
        payload=None,
    )
    decision = StageEvaluator().evaluate(SBT_STAGE, (observation,), subject_ref=execution.execution_id)
    assert decision.decision is None
    assert decision.evaluation_status is not DecisionEvaluationStatus.EVALUATED
