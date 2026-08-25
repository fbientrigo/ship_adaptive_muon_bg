"""``MinimalStubBackend`` — a second, deliberately different fake backend.

Its only purpose is to prove that the evaluation boundary is a real boundary.
:class:`~ship_muon_bg.simulation.fake_fairship.FakeFairShipBackend` derives its
structure from sha256 draws; this one uses no hashing whatsoever — it walks a
short cycle of explicit integers. The two share no code path beyond the
canonical entity constructors.

If the same aggregation, tagging, proxy, and flow code produces correct results
against both, then backend choice really is dependency injection rather than a
scientific assumption baked into the downstream stack. That is the mission's
substitution criterion, and this class is the control arm of that experiment.

**Not a physics model.** ``is_physical`` is ``False``.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

from ship_muon_bg.entities.decision import (
    DecisionEvaluationStatus,
    StageDecision,
    stage_decision_id,
)
from ship_muon_bg.entities.lineage import (
    OPTIONS_DIGEST_PROVENANCE_KEY,
    ExecutionStatus,
    FSSimExecution,
    InteractionRealization,
    ReconstructedCandidate,
)
from ship_muon_bg.entities.observation import (
    ObservationEnvelope,
    ObservationEvaluationStatus,
    ScalarObservationPayload,
)
from ship_muon_bg.simulation.evaluation import EvaluationBundle, EvaluationRequest

STUB_INTERACTION_TYPE = "stub_interaction"
STUB_INTERACTION_DEFINITION_ID = "stub.interaction_v0@notahash:stub"


class MinimalStubBackend:
    """Emit one execution per (subject, replication) from an explicit cycle.

    ``candidate_count_cycle`` is walked by a running run-ordinal, so the
    ``0 / 1 / many`` candidate cases all appear without any randomness.
    ``failure_every`` marks every n-th run a technical failure (``0``
    disables); ``score_cycle`` supplies the scalar each candidate reports for
    every stage, so positives and negatives are fully predictable.
    """

    is_physical = False
    version = "v0"

    def __init__(
        self,
        *,
        stage_definition_ids: Sequence[str],
        observation_definition_ids: Sequence[str],
        candidate_count_cycle: Tuple[int, ...] = (0, 1, 2),
        score_cycle: Tuple[float, ...] = (0.9, 0.1),
        threshold: float = 0.5,
        failure_every: int = 0,
        name: str = "minimal_stub",
        units: str = "fixture_units",
    ) -> None:
        stage_ids = tuple(stage_definition_ids)
        observation_ids = tuple(observation_definition_ids)
        if len(stage_ids) != len(observation_ids):
            raise ValueError(
                "each stage needs exactly one observation definition of its own"
            )
        if not stage_ids:
            raise ValueError("at least one stage is required")
        if len(set(stage_ids)) != len(stage_ids):
            raise ValueError("stage_definition_ids must be distinct")
        if len(set(observation_ids)) != len(observation_ids):
            raise ValueError("observation_definition_ids must be distinct")
        if not candidate_count_cycle:
            raise ValueError("candidate_count_cycle must be non-empty")
        if not score_cycle:
            raise ValueError("score_cycle must be non-empty")
        if isinstance(failure_every, bool) or not isinstance(failure_every, int):
            raise TypeError("failure_every must be an int")
        if failure_every < 0:
            raise ValueError("failure_every must be non-negative")

        self.name = name
        self._stage_ids = stage_ids
        self._observation_ids = observation_ids
        self._candidate_count_cycle = tuple(candidate_count_cycle)
        self._score_cycle = tuple(float(value) for value in score_cycle)
        self._threshold = float(threshold)
        self._failure_every = failure_every
        self._units = units

    def evaluate(self, request: EvaluationRequest) -> EvaluationBundle:
        if not isinstance(request, EvaluationRequest):
            raise TypeError("request must be an EvaluationRequest")

        executions = []
        realizations = []
        candidates = []
        observations = []
        decisions = []
        run_ordinal = 0
        score_ordinal = 0

        for subject in request.subjects:
            for replication_index in range(request.replications_per_subject):
                execution_id = (
                    f"stub-exec:{request.request_id}:{subject.subject_id}:"
                    f"{replication_index}"
                )
                failed = bool(
                    self._failure_every and run_ordinal % self._failure_every == 0
                )
                executions.append(
                    FSSimExecution(
                        execution_id=execution_id,
                        subject_id=subject.subject_id,
                        fs_sim_configuration_id=request.fs_sim_configuration_id,
                        execution_status=(
                            ExecutionStatus.TECHNICAL_FAILURE
                            if failed
                            else ExecutionStatus.SUCCEEDED
                        ),
                        seed=request.seed + run_ordinal,
                        failure_reason=(
                            "stub_backend: scheduled technical failure" if failed else ""
                        ),
                        provenance={
                            OPTIONS_DIGEST_PROVENANCE_KEY: request.options_digest,
                            "backend_name": self.name,
                            "backend_version": self.version,
                            "is_physical": "false",
                            "request_id": request.request_id,
                            "replication_index": str(replication_index),
                        },
                    )
                )
                if failed:
                    run_ordinal += 1
                    continue

                candidate_count = self._candidate_count_cycle[
                    run_ordinal % len(self._candidate_count_cycle)
                ]
                realization_id: Optional[str] = None
                if candidate_count:
                    realization_id = f"stub-real:{execution_id}:0"
                    realizations.append(
                        InteractionRealization(
                            realization_id=realization_id,
                            execution_id=execution_id,
                            interaction_type=STUB_INTERACTION_TYPE,
                            interaction_definition_id=STUB_INTERACTION_DEFINITION_ID,
                        )
                    )
                for candidate_index in range(candidate_count):
                    candidate_id = f"stub-cand:{execution_id}:{candidate_index}"
                    candidates.append(
                        ReconstructedCandidate(
                            candidate_id=candidate_id,
                            execution_id=execution_id,
                            realization_id=realization_id,
                            candidate_index=candidate_index,
                        )
                    )
                    value = self._score_cycle[score_ordinal % len(self._score_cycle)]
                    score_ordinal += 1
                    for stage_id, observation_id_def in zip(
                        self._stage_ids, self._observation_ids
                    ):
                        observation_id = f"stub-obs:{candidate_id}:{observation_id_def}"
                        observations.append(
                            ObservationEnvelope(
                                observation_id=observation_id,
                                subject_ref=candidate_id,
                                observation_definition_id=observation_id_def,
                                units=self._units,
                                evaluation_status=(
                                    ObservationEvaluationStatus.COMPUTED
                                ),
                                evidence_reference=f"{self.name}:{execution_id}",
                                payload=ScalarObservationPayload(value=value),
                            )
                        )
                        status = DecisionEvaluationStatus.EVALUATED
                        decisions.append(
                            StageDecision(
                                decision_id=stage_decision_id(
                                    stage_definition_id=stage_id,
                                    subject_ref=candidate_id,
                                    evidence_references=(observation_id,),
                                    evaluation_status=status,
                                ),
                                subject_ref=candidate_id,
                                stage_definition_id=stage_id,
                                evaluation_status=status,
                                decision=bool(value > self._threshold),
                                evidence_references=(observation_id,),
                            )
                        )
                if candidate_count == 0:
                    # Nothing else can attest that the stage was applied to a
                    # run that reconstructed nothing.
                    for stage_id in self._stage_ids:
                        status = DecisionEvaluationStatus.EVALUATED
                        decisions.append(
                            StageDecision(
                                decision_id=stage_decision_id(
                                    stage_definition_id=stage_id,
                                    subject_ref=execution_id,
                                    evidence_references=(),
                                    evaluation_status=status,
                                ),
                                subject_ref=execution_id,
                                stage_definition_id=stage_id,
                                evaluation_status=status,
                                decision=False,
                            )
                        )
                run_ordinal += 1

        return EvaluationBundle(
            request_id=request.request_id,
            backend_name=self.name,
            is_physical=self.is_physical,
            executions=tuple(executions),
            realizations=tuple(realizations),
            candidates=tuple(candidates),
            observations=tuple(observations),
            decisions=tuple(decisions),
        )
