"""Focused tests for the generic, non-physical tagging core."""

from __future__ import annotations

import pytest

from ship_muon_bg.entities import (
    ObservationEnvelope,
    ObservationEvaluationStatus,
    ScalarObservationPayload,
)
from ship_muon_bg.tagging import (
    StageDefinition,
    StageEvaluator,
    scalar_above_threshold_stage,
)
from ship_muon_bg.entities.decision import DecisionEvaluationStatus


def _scalar_observation(
    *,
    observation_id: str,
    observation_definition_id: str = "fixture.scalar@v0",
    subject_ref: str = "subject-0",
    value: float = 0.75,
    status: ObservationEvaluationStatus = ObservationEvaluationStatus.COMPUTED,
) -> ObservationEnvelope:
    return ObservationEnvelope(
        observation_id=observation_id,
        subject_ref=subject_ref,
        observation_definition_id=observation_definition_id,
        units="arbitrary_fixture_units",
        evaluation_status=status,
        evidence_reference=f"fixture://{observation_id}",
        payload=ScalarObservationPayload(value=value) if status is ObservationEvaluationStatus.COMPUTED else None,
    )


def _stage(*, observation_definition_id: str = "fixture.scalar@v0", threshold: float = 0.5):
    return scalar_above_threshold_stage(
        observation_definition_id=observation_definition_id,
        threshold=threshold,
    )


def test_equivalent_stage_definition_content_has_identical_id_even_when_reordered():
    first = StageDefinition(
        stage_name="fixture.scalar_above_threshold",
        semantic_content={
            "operation": "greater_than",
            "nested": {"b": 2, "a": 1},
            "threshold": 0.5,
        },
        required_observation_definition_ids=("fixture.scalar@v0",),
    )
    equivalent = StageDefinition(
        stage_name="fixture.scalar_above_threshold",
        semantic_content={
            "threshold": 0.5,
            "nested": {"a": 1, "b": 2},
            "operation": "greater_than",
        },
        required_observation_definition_ids=("fixture.scalar@v0",),
    )
    assert first.stage_definition_id == equivalent.stage_definition_id


def test_threshold_changes_stage_definition_id():
    assert _stage(threshold=0.5).stage_definition_id != _stage(threshold=0.75).stage_definition_id


def test_required_observation_definition_changes_stage_definition_id():
    content = {"operation": "greater_than", "threshold": 0.5}
    first = StageDefinition(
        stage_name="fixture.scalar_above_threshold",
        semantic_content=content,
        required_observation_definition_ids=("fixture.scalar_a@v0",),
    )
    changed = StageDefinition(
        stage_name="fixture.scalar_above_threshold",
        semantic_content=content,
        required_observation_definition_ids=("fixture.scalar_b@v0",),
    )
    assert first.stage_definition_id != changed.stage_definition_id


def test_required_observation_definition_membership_is_canonicalized_as_unordered():
    content = {"operation": "not_evaluated"}
    first = StageDefinition(
        stage_name="fixture.availability",
        semantic_content=content,
        required_observation_definition_ids=("obs_A", "obs_B"),
    )
    reordered = StageDefinition(
        stage_name="fixture.availability",
        semantic_content=content,
        required_observation_definition_ids=("obs_B", "obs_A"),
    )

    assert first.stage_definition_id == reordered.stage_definition_id
    assert first.required_observation_definition_ids == ("obs_A", "obs_B")


def test_required_observation_definition_membership_changes_stage_definition_id():
    content = {"operation": "not_evaluated"}
    first = StageDefinition(
        stage_name="fixture.availability",
        semantic_content=content,
        required_observation_definition_ids=("obs_A", "obs_B"),
    )
    changed = StageDefinition(
        stage_name="fixture.availability",
        semantic_content=content,
        required_observation_definition_ids=("obs_A", "obs_C"),
    )

    assert first.stage_definition_id != changed.stage_definition_id


def test_duplicate_required_observation_definitions_are_rejected():
    with pytest.raises(ValueError, match="must be unique"):
        StageDefinition(
            stage_name="fixture.duplicate_dependencies",
            semantic_content={"operation": "not_evaluated"},
            required_observation_definition_ids=("obs_A", "obs_A"),
        )


def test_explicit_role_order_remains_semantically_distinguishable():
    first = StageDefinition(
        stage_name="fixture.ordered_roles",
        semantic_content={"lhs": "obs_A", "rhs": "obs_B"},
        required_observation_definition_ids=("obs_A", "obs_B"),
    )
    reversed_roles = StageDefinition(
        stage_name="fixture.ordered_roles",
        semantic_content={"lhs": "obs_B", "rhs": "obs_A"},
        required_observation_definition_ids=("obs_B", "obs_A"),
    )

    assert first.stage_definition_id != reversed_roles.stage_definition_id


def test_stage_definition_rejects_opaque_executable_semantics():
    with pytest.raises(TypeError):
        StageDefinition(
            stage_name="fixture.opaque",
            semantic_content={"predicate": lambda value: value > 0.5},
        )


def test_stage_definition_content_is_immutable():
    definition = _stage()
    with pytest.raises(TypeError):
        definition.semantic_content["threshold"] = 0.9  # type: ignore[index]


def test_evaluated_path_returns_decision_and_exact_evidence_reference():
    definition = _stage()
    observation = _scalar_observation(observation_id="obs-0", value=0.75)

    result = StageEvaluator().evaluate(
        definition,
        [observation],
        subject_ref="subject-0",
    )

    assert result.evaluation_status is DecisionEvaluationStatus.EVALUATED
    assert result.decision is True
    assert result.stage_definition_id == definition.stage_definition_id
    assert result.evidence_references == ("obs-0",)


def test_decision_identity_depends_on_consumed_evidence_identity():
    definition = _stage()
    first = StageEvaluator().evaluate(
        definition,
        [_scalar_observation(observation_id="obs-A", value=0.75)],
        subject_ref="subject-0",
    )
    second = StageEvaluator().evaluate(
        definition,
        [_scalar_observation(observation_id="obs-B", value=0.75)],
        subject_ref="subject-0",
    )

    assert first.stage_definition_id == second.stage_definition_id
    assert first.evidence_references != second.evidence_references
    assert first.decision_id != second.decision_id


def test_decision_identity_is_deterministic_and_input_order_insensitive():
    definition = StageDefinition(
        stage_name="fixture.censored_pair",
        semantic_content={"operation": "not_evaluated"},
        required_observation_definition_ids=("obs_A", "obs_B"),
    )
    not_applicable = _scalar_observation(
        observation_id="evidence-A",
        observation_definition_id="obs_A",
        status=ObservationEvaluationStatus.NOT_APPLICABLE,
    )
    unavailable = _scalar_observation(
        observation_id="evidence-B",
        observation_definition_id="obs_B",
        status=ObservationEvaluationStatus.TECHNICALLY_UNAVAILABLE,
    )

    first = StageEvaluator().evaluate(
        definition, [not_applicable, unavailable], subject_ref="subject-0"
    )
    repeated = StageEvaluator().evaluate(
        definition, [not_applicable, unavailable], subject_ref="subject-0"
    )
    reordered = StageEvaluator().evaluate(
        definition, [unavailable, not_applicable], subject_ref="subject-0"
    )

    assert first.evaluation_status is DecisionEvaluationStatus.TECHNICALLY_UNAVAILABLE
    assert first.decision is None
    assert first.decision_id == repeated.decision_id == reordered.decision_id
    assert first.evidence_references == ("evidence-A", "evidence-B")


def test_not_evaluated_decision_identity_is_deterministic_and_not_false():
    definition = _stage()
    missing = StageEvaluator().evaluate(definition, [], subject_ref="subject-0")
    repeated_missing = StageEvaluator().evaluate(definition, [], subject_ref="subject-0")

    assert missing.evaluation_status is DecisionEvaluationStatus.NOT_EVALUATED
    assert missing.decision is None
    assert missing.decision_id == repeated_missing.decision_id


def test_computed_physical_zero_or_false_remains_an_evaluated_false():
    result = StageEvaluator().evaluate(
        _stage(threshold=0.5),
        [_scalar_observation(observation_id="obs-zero", value=0.5)],
        subject_ref="subject-0",
    )

    assert result.evaluation_status is DecisionEvaluationStatus.EVALUATED
    assert result.decision is False


def test_missing_required_observation_is_not_evaluated_not_false():
    result = StageEvaluator().evaluate(_stage(), [], subject_ref="subject-0")

    assert result.evaluation_status is DecisionEvaluationStatus.NOT_EVALUATED
    assert result.decision is None
    assert result.evidence_references == ()


def test_technically_unavailable_required_observation_is_censored():
    unavailable = _scalar_observation(
        observation_id="obs-unavailable",
        status=ObservationEvaluationStatus.TECHNICALLY_UNAVAILABLE,
    )
    result = StageEvaluator().evaluate(_stage(), [unavailable], subject_ref="subject-0")

    assert result.evaluation_status is DecisionEvaluationStatus.TECHNICALLY_UNAVAILABLE
    assert result.decision is None
    assert result.evidence_references == ("obs-unavailable",)


def test_conflicting_availability_uses_technical_unavailable_precedence():
    definition = StageDefinition(
        stage_name="fixture.availability_precedence",
        semantic_content={"operation": "not_evaluated"},
        required_observation_definition_ids=("fixture.a@v0", "fixture.b@v0"),
    )
    not_applicable = _scalar_observation(
        observation_id="obs-a",
        observation_definition_id="fixture.a@v0",
        status=ObservationEvaluationStatus.NOT_APPLICABLE,
    )
    unavailable = _scalar_observation(
        observation_id="obs-b",
        observation_definition_id="fixture.b@v0",
        status=ObservationEvaluationStatus.TECHNICALLY_UNAVAILABLE,
    )

    result = StageEvaluator().evaluate(
        definition,
        [not_applicable, unavailable],
        subject_ref="subject-0",
    )

    assert result.evaluation_status is DecisionEvaluationStatus.TECHNICALLY_UNAVAILABLE
    assert result.decision is None
    assert result.evidence_references == ("obs-a", "obs-b")


def test_unused_observations_are_not_evidence_dependencies():
    required = _scalar_observation(observation_id="obs-required")
    unused = _scalar_observation(
        observation_id="obs-unused",
        observation_definition_id="fixture.unused@v0",
    )

    result = StageEvaluator().evaluate(
        _stage(),
        [required, unused],
        subject_ref="subject-0",
    )

    assert result.evaluation_status is DecisionEvaluationStatus.EVALUATED
    assert result.evidence_references == ("obs-required",)


def test_observations_from_other_entity_references_are_not_aggregated():
    other_entity = _scalar_observation(
        observation_id="obs-other",
        subject_ref="subject-1",
        value=1.0,
    )

    result = StageEvaluator().evaluate(
        _stage(),
        [other_entity],
        subject_ref="subject-0",
    )

    assert result.evaluation_status is DecisionEvaluationStatus.NOT_EVALUATED
    assert result.decision is None
    assert result.evidence_references == ()


def test_duplicate_required_observations_are_not_silently_reduced():
    first = _scalar_observation(observation_id="obs-1")
    second = _scalar_observation(observation_id="obs-2")

    with pytest.raises(ValueError, match="does not aggregate"):
        StageEvaluator().evaluate(_stage(), [first, second], subject_ref="subject-0")


def test_controlled_fixture_is_explicitly_non_physical():
    definition = _stage()

    assert definition.stage_name == "fixture.scalar_above_threshold"
    assert definition.semantic_content["fixture"] == "controlled_non_physical"
    assert definition.semantic_content["is_physical"] is False
    assert "DIS" not in definition.stage_name
