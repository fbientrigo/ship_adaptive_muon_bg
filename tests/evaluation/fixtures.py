"""Hand-computable fixtures shared by the evaluation-boundary contract tests.

Everything here is a *test* construct. No value in this file is a SHiP
physics quantity, and the one stage definition used is the repository's
explicitly non-physical ``fixture.scalar_above_threshold`` rule.
"""

from __future__ import annotations

from ship_muon_bg.entities import (
    DecisionEvaluationStatus,
    ExecutionStatus,
    FSSimExecution,
    InteractionRealization,
    StageDecision,
    stage_decision_id,
    ObservationEnvelope,
    ObservationEvaluationStatus,
    ReconstructedCandidate,
    ScalarObservationPayload,
    TagSubject,
)
from ship_muon_bg.entities.identifiers import content_hash
from ship_muon_bg.entities.lineage import OPTIONS_DIGEST_PROVENANCE_KEY
from ship_muon_bg.simulation.evaluation import EvaluationBundle, EvaluationRequest

#: The digest every fixture request carries, since they all use empty options.
EMPTY_OPTIONS_DIGEST = content_hash({})

CONFIG_A = "fs_sim_config_a@notahash:aaaa"
CONFIG_B = "fs_sim_config_b@notahash:bbbb"
STATE_DEFINITION = "post_shield_state_v0@notahash:cccc"
MOMENTUM_OBSERVATION = "fixture.scalar_momentum_v0@notahash:dddd"


def subject(subject_id: str, subject_type: str = "muon_state") -> TagSubject:
    return TagSubject(
        subject_id=subject_id,
        subject_type=subject_type,
        state_definition_id=STATE_DEFINITION,
    )


def scalar_observation(
    observation_id: str,
    subject_ref: str,
    value: float,
    *,
    observation_definition_id: str = MOMENTUM_OBSERVATION,
    units: str = "GeV",
) -> ObservationEnvelope:
    return ObservationEnvelope(
        observation_id=observation_id,
        subject_ref=subject_ref,
        observation_definition_id=observation_definition_id,
        units=units,
        evaluation_status=ObservationEvaluationStatus.COMPUTED,
        evidence_reference=f"fixture:{observation_id}",
        payload=ScalarObservationPayload(value=value),
    )


def unavailable_observation(
    observation_id: str,
    subject_ref: str,
    *,
    observation_definition_id: str = MOMENTUM_OBSERVATION,
    units: str = "GeV",
) -> ObservationEnvelope:
    return ObservationEnvelope(
        observation_id=observation_id,
        subject_ref=subject_ref,
        observation_definition_id=observation_definition_id,
        units=units,
        evaluation_status=ObservationEvaluationStatus.TECHNICALLY_UNAVAILABLE,
        evidence_reference=f"fixture:{observation_id}",
        payload=None,
    )


def request(
    *,
    request_id: str = "req-1",
    subject_ids=("s1",),
    fs_sim_configuration_id: str = CONFIG_A,
    seed: int = 11,
    replications_per_subject: int = 1,
    subject_observations=(),
) -> EvaluationRequest:
    return EvaluationRequest(
        request_id=request_id,
        subjects=tuple(subject(sid) for sid in subject_ids),
        fs_sim_configuration_id=fs_sim_configuration_id,
        seed=seed,
        replications_per_subject=replications_per_subject,
        subject_observations=tuple(subject_observations),
    )


def execution(
    execution_id: str,
    subject_id: str,
    *,
    fs_sim_configuration_id: str = CONFIG_A,
    status: ExecutionStatus = ExecutionStatus.SUCCEEDED,
    failure_reason: str = "",
    seed: int = 11,
    options_digest: str = EMPTY_OPTIONS_DIGEST,
) -> FSSimExecution:
    return FSSimExecution(
        execution_id=execution_id,
        subject_id=subject_id,
        fs_sim_configuration_id=fs_sim_configuration_id,
        execution_status=status,
        seed=seed,
        failure_reason=failure_reason,
        provenance={OPTIONS_DIGEST_PROVENANCE_KEY: options_digest},
    )


def realization(
    realization_id: str,
    execution_id: str,
    *,
    interaction_type: str = "fixture_interaction",
) -> InteractionRealization:
    return InteractionRealization(
        realization_id=realization_id,
        execution_id=execution_id,
        interaction_type=interaction_type,
        interaction_definition_id="fixture.interaction_v0@notahash:eeee",
    )


def candidate(
    candidate_id: str,
    execution_id: str,
    *,
    realization_id=None,
    candidate_index=None,
) -> ReconstructedCandidate:
    return ReconstructedCandidate(
        candidate_id=candidate_id,
        execution_id=execution_id,
        realization_id=realization_id,
        candidate_index=candidate_index,
    )


def bundle(
    request_id: str = "req-1",
    backend_name: str = "fixture",
    is_physical: bool = False,
    **kwargs,
):
    return EvaluationBundle(
        request_id=request_id,
        backend_name=backend_name,
        is_physical=is_physical,
        **kwargs,
    )


# --- backend fixtures ------------------------------------------------------

SCORE_A = "fixture.score_a_v0@notahash:1111"
SCORE_B = "fixture.score_b_v0@notahash:2222"
THRESHOLD = 0.5


def stage_definition(observation_definition_id=SCORE_A, threshold=THRESHOLD):
    """A real ``StageDefinition`` from the tagging layer.

    Built in the *test*, never inside the backend: ``simulation/`` may import
    ``entities/`` only, so a backend receives stage identity as an opaque
    string exactly as a real adapter would.
    """
    from ship_muon_bg.tagging import scalar_above_threshold_stage

    return scalar_above_threshold_stage(
        observation_definition_id=observation_definition_id, threshold=threshold
    )


def stage_spec(observation_definition_id=SCORE_A, threshold=THRESHOLD):
    from ship_muon_bg.simulation.fake_fairship import FakeStageSpec

    definition = stage_definition(observation_definition_id, threshold)
    return definition, FakeStageSpec(
        stage_definition_id=definition.stage_definition_id,
        observation_definition_id=observation_definition_id,
        threshold=threshold,
    )


FIXTURE_STAGE = "fixture.stage_v0@notahash:ffff"


def decision(
    subject_ref: str,
    evaluation_status: DecisionEvaluationStatus,
    value,
    *,
    stage_definition_id: str = FIXTURE_STAGE,
    evidence_references=(),
) -> StageDecision:
    references = tuple(evidence_references)
    return StageDecision(
        decision_id=stage_decision_id(
            stage_definition_id=stage_definition_id,
            subject_ref=subject_ref,
            evidence_references=references,
            evaluation_status=evaluation_status,
        ),
        subject_ref=subject_ref,
        stage_definition_id=stage_definition_id,
        evaluation_status=evaluation_status,
        decision=value,
        evidence_references=references,
    )
