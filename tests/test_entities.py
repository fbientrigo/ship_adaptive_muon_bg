"""Guard-rail tests for the canonical ``ship_muon_bg.entities`` layer.

Covers cardinality preservation, the execution/decision censoring
invariants, typed-observation-payload support, and semantic-definition-id
determinism. Requires no FairShip, no ROOT, no GPU.
"""

from __future__ import annotations

import pytest

from ship_muon_bg.entities import (
    DecisionEvaluationStatus,
    ExecutionStatus,
    FSSimExecution,
    InteractionRealization,
    ObservationEnvelope,
    ObservationEvaluationStatus,
    ReconstructedCandidate,
    ScalarObservationPayload,
    SequenceObservationPayload,
    StageDecision,
    TagSubject,
    content_hash,
    definition_id,
)


# ---------------------------------------------------------------------------
# Cardinality: 0..N is not structurally collapsed anywhere in the lineage.
# ---------------------------------------------------------------------------


def test_one_subject_can_be_referenced_by_multiple_executions():
    subject = TagSubject(
        subject_id="s0", subject_type="after_ms_row_v0", state_definition_id="state@v0"
    )
    executions = [
        FSSimExecution(
            execution_id=f"e{i}",
            subject_id=subject.subject_id,
            fs_sim_configuration_id="cfg@v0",
            execution_status=ExecutionStatus.SUCCEEDED,
        )
        for i in range(3)
    ]
    assert len({e.execution_id for e in executions}) == 3
    assert all(e.subject_id == subject.subject_id for e in executions)


def test_one_execution_can_reference_multiple_interaction_realizations():
    realizations = [
        InteractionRealization(
            realization_id=f"r{i}",
            execution_id="e0",
            interaction_type="muon_dis",
            interaction_definition_id="muon_dis@v0",
        )
        for i in range(5)
    ]
    assert len(realizations) == 5
    assert all(r.execution_id == "e0" for r in realizations)


def test_zero_realizations_is_a_valid_execution_outcome():
    # No structural requirement forces at least one InteractionRealization
    # per execution; an execution with none is representable by simply not
    # constructing any, which does not raise.
    execution = FSSimExecution(
        execution_id="e0",
        subject_id="s0",
        fs_sim_configuration_id="cfg@v0",
        execution_status=ExecutionStatus.SUCCEEDED,
    )
    assert execution.execution_status is ExecutionStatus.SUCCEEDED


def test_candidate_association_does_not_require_exactly_one_realization():
    execution_level = ReconstructedCandidate(candidate_id="c0", execution_id="e0")
    realization_level = ReconstructedCandidate(
        candidate_id="c1", execution_id="e0", realization_id="r0"
    )
    assert execution_level.realization_id is None
    assert realization_level.realization_id == "r0"


def test_execution_status_enum_excludes_physics_decision_values():
    values = {status.value for status in ExecutionStatus}
    assert "accepted_candidate" not in values
    assert "physics_rejection" not in values
    assert values == {"succeeded", "technical_failure"}


# ---------------------------------------------------------------------------
# Censoring: technical failure / not-evaluated never becomes an implicit
# negative decision.
# ---------------------------------------------------------------------------


def test_not_evaluated_stage_decision_carries_no_decision_value():
    decision = StageDecision(
        decision_id="d0",
        subject_ref="e0",
        stage_definition_id="operational_selection@v0",
        evaluation_status=DecisionEvaluationStatus.NOT_EVALUATED,
        decision=None,
    )
    assert decision.decision is None
    assert decision.evaluation_status is DecisionEvaluationStatus.NOT_EVALUATED


def test_not_evaluated_stage_decision_rejects_a_smuggled_false():
    with pytest.raises(ValueError):
        StageDecision(
            decision_id="d0",
            subject_ref="e0",
            stage_definition_id="operational_selection@v0",
            evaluation_status=DecisionEvaluationStatus.NOT_EVALUATED,
            decision=False,
        )


def test_technically_unavailable_stage_decision_rejects_a_smuggled_false():
    with pytest.raises(ValueError):
        StageDecision(
            decision_id="d0",
            subject_ref="e0",
            stage_definition_id="operational_selection@v0",
            evaluation_status=DecisionEvaluationStatus.TECHNICALLY_UNAVAILABLE,
            decision=False,
        )


def test_evaluated_stage_decision_requires_a_decision_value():
    with pytest.raises(ValueError):
        StageDecision(
            decision_id="d0",
            subject_ref="e0",
            stage_definition_id="operational_selection@v0",
            evaluation_status=DecisionEvaluationStatus.EVALUATED,
            decision=None,
        )


def test_evaluated_stage_decision_accepts_a_real_false():
    # Once a stage was genuinely evaluated, False is a legitimate physics
    # decision (e.g. "not accepted") — only NOT_EVALUATED/TECHNICALLY_UNAVAILABLE
    # forbid it.
    decision = StageDecision(
        decision_id="d0",
        subject_ref="e0",
        stage_definition_id="operational_selection@v0",
        evaluation_status=DecisionEvaluationStatus.EVALUATED,
        decision=False,
    )
    assert decision.decision is False


def test_stage_decision_module_never_imports_execution_status():
    # Structural check that DECISION-04's separation is not just documented:
    # entities.decision does not import entities.lineage at all, so a
    # StageDecision can never be auto-derived from an ExecutionStatus.
    import ship_muon_bg.entities.decision as decision_module

    assert "lineage" not in vars(decision_module)
    assert not hasattr(decision_module, "ExecutionStatus")


# ---------------------------------------------------------------------------
# ObservationEnvelope: typed, non-scalar-safe payloads.
# ---------------------------------------------------------------------------


def test_observation_envelope_accepts_a_scalar_typed_payload():
    obs = ObservationEnvelope(
        observation_id="o0",
        subject_ref="e0",
        observation_definition_id="path_length@v0",
        units="m",
        evaluation_status=ObservationEvaluationStatus.COMPUTED,
        evidence_reference="raw_output.root#tree0",
        payload=ScalarObservationPayload(value=12.5),
    )
    assert isinstance(obs.payload, ScalarObservationPayload)
    assert obs.payload.value == 12.5


def test_observation_envelope_accepts_a_non_scalar_typed_payload():
    obs = ObservationEnvelope(
        observation_id="o1",
        subject_ref="e0",
        observation_definition_id="traversal_profile@v0",
        units="m",
        evaluation_status=ObservationEvaluationStatus.COMPUTED,
        evidence_reference="raw_output.root#tree0",
        payload=SequenceObservationPayload(values=(1.0, 2.5, 3.75)),
    )
    assert isinstance(obs.payload, SequenceObservationPayload)
    assert obs.payload.values == (1.0, 2.5, 3.75)


def test_computed_observation_requires_a_payload():
    with pytest.raises(ValueError):
        ObservationEnvelope(
            observation_id="o0",
            subject_ref="e0",
            observation_definition_id="path_length@v0",
            units="m",
            evaluation_status=ObservationEvaluationStatus.COMPUTED,
            evidence_reference="raw_output.root#tree0",
            payload=None,
        )


def test_technically_unavailable_observation_rejects_a_sentinel_payload():
    with pytest.raises(ValueError):
        ObservationEnvelope(
            observation_id="o0",
            subject_ref="e0",
            observation_definition_id="path_length@v0",
            units="m",
            evaluation_status=ObservationEvaluationStatus.TECHNICALLY_UNAVAILABLE,
            evidence_reference="raw_output.root#tree0",
            payload=ScalarObservationPayload(value=0.0),
        )


def test_not_applicable_observation_carries_no_payload():
    obs = ObservationEnvelope(
        observation_id="o0",
        subject_ref="e0",
        observation_definition_id="veto_path_length@v0",
        units="m",
        evaluation_status=ObservationEvaluationStatus.NOT_APPLICABLE,
        evidence_reference="raw_output.root#tree0",
        payload=None,
    )
    assert obs.payload is None


# ---------------------------------------------------------------------------
# Semantic definition identifiers: deterministic, content-sensitive.
# ---------------------------------------------------------------------------


def test_definition_id_is_deterministic_for_identical_content():
    content = {"cut": "pt > 1.0", "version": 0}
    first = definition_id("operational_selection_v0", content)
    second = definition_id("operational_selection_v0", dict(content))
    assert first == second


def test_definition_id_changes_with_changed_semantic_content():
    base = definition_id("operational_selection_v0", {"cut": "pt > 1.0"})
    changed = definition_id("operational_selection_v0", {"cut": "pt > 1.5"})
    assert base != changed


def test_definition_id_label_alone_is_not_load_bearing_for_identity():
    # Same content, different human-readable label: different string
    # overall (label is a real prefix), but the embedded content hash
    # segment is identical — proving identity rides on content, not label.
    content = {"cut": "pt > 1.0"}
    a = definition_id("label_a", content)
    b = definition_id("label_b", content)
    assert a != b
    assert content_hash(content) in a
    assert content_hash(content) in b


def test_definition_id_rejects_empty_label():
    with pytest.raises(ValueError):
        definition_id("", {"cut": "pt > 1.0"})


def test_content_hash_is_order_insensitive_over_mapping_keys():
    assert content_hash({"a": 1, "b": 2}) == content_hash({"b": 2, "a": 1})


# ---------------------------------------------------------------------------
# Immutability.
# ---------------------------------------------------------------------------


def test_tag_subject_is_frozen():
    subject = TagSubject(subject_id="s0", subject_type="t", state_definition_id="d@v0")
    with pytest.raises(Exception):
        subject.subject_id = "s1"  # type: ignore[misc]


def test_stage_decision_is_frozen():
    decision = StageDecision(
        decision_id="d0",
        subject_ref="e0",
        stage_definition_id="operational_selection@v0",
        evaluation_status=DecisionEvaluationStatus.NOT_EVALUATED,
        decision=None,
    )
    with pytest.raises(Exception):
        decision.decision = True  # type: ignore[misc]
