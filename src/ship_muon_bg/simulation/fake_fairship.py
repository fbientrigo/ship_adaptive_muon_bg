"""``FakeFairShipBackend`` — deterministic test infrastructure, not physics.

**This is not a physics model.** It contains no transport, no geometry, no
detector response, and no SHiP selection. Every number it produces is a
deterministic function of a hash. Its outputs must never be cited, plotted,
or aggregated as SHiP physics evidence; ``is_physical`` is ``False`` so that
statement is machine-checkable rather than a naming convention.

What it *is*: a reference implementation of
:class:`~ship_muon_bg.simulation.evaluation.EvaluationBackend` rich enough to
exercise every structural situation a real FairShip adapter will have to
represent —

- a subject that produced **no execution at all**;
- an execution that **failed technically** (and therefore has no physics
  outcome of any kind);
- an execution with **zero, one, or many** interaction realizations;
- a realization with **zero, one, or many** reconstructed candidates;
- candidates **not scoped to any realization** (``EXEC-04``);
- stage evidence that is **computed**, **technically unavailable**, or
  **absent entirely** — three distinct forms of missingness, none of which
  may become a negative decision (``CENSOR-01``/``CENSOR-02``).

Determinism: outcomes depend only on ``seed``, ``fs_sim_configuration_id``,
the subject's identity and state definition, and the replication index. They
deliberately do **not** depend on ``request_id`` or ``options``, so relabelling
a request or moving an output directory cannot change the physics-shaped
answer — only the identifiers change. Nothing consults wall-clock time, the
process id, or ``random``.

Hand-computability: pass explicit :class:`FakeRunPlan` objects via ``plans``
and the emitted records follow the plan exactly, so a test can state the
expected counts in advance. Unscripted subjects fall back to the hash-derived
structure, which is reproducible but not meant to be predicted on paper.

Backend independence: imports ``entities`` and ``simulation`` only — never
``tagging``, ROOT, or FairShip. Stage rules arrive as opaque
``stage_definition_id`` strings supplied by the caller, exactly as a real
adapter would report the outcome of a rule it did not itself define.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Tuple

from ship_muon_bg.entities.decision import (
    DecisionEvaluationStatus,
    StageDecision,
    stage_decision_id,
)
from ship_muon_bg.entities.identifiers import content_hash
from ship_muon_bg.entities.lineage import (
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
from ship_muon_bg.entities.subject import TagSubject
from ship_muon_bg.simulation.evaluation import EvaluationBundle, EvaluationRequest

BACKEND_NAME = "fake_fairship"
BACKEND_VERSION = "v0"

#: Interaction type emitted by this backend. Deliberately not ``"muon_dis"``:
#: nothing here models deep inelastic scattering, and a fixture must not be
#: mistakable for one.
FIXTURE_INTERACTION_TYPE = "fixture_interaction"
FIXTURE_INTERACTION_DEFINITION_ID = "fixture.interaction_v0@sha256:fake"


def _require_nonempty_str(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_non_negative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


@dataclass(frozen=True)
class FakeStageSpec:
    """One stage this backend reports an outcome for.

    ``stage_definition_id`` is opaque here: the backend never interprets it,
    it only records *which versioned rule* an outcome belongs to. Callers
    obtain it from a real ``StageDefinition`` built in the tagging layer.

    ``threshold`` is the fixture's internal cut on the emitted scalar. It
    stands in for a black-box simulator's own reconstruction/selection logic:
    a real adapter likewise reports an outcome it cannot re-derive from the
    observables it exports. Tests that want the two to agree must construct
    the ``StageDefinition`` with this same threshold.

    Each stage carries its **own** ``observation_definition_id`` and its own
    independently derived value. Stages therefore have no logical implications
    between them — passing one never implies passing another (mission
    invariant 3.4), which is exactly the placeholder semantics the thesis
    requires while the real ``G -> D -> R -> S -> V -> B`` chain is unfixed.
    """

    stage_definition_id: str
    observation_definition_id: str
    threshold: float
    units: str = "fixture_units"

    def __post_init__(self) -> None:
        _require_nonempty_str(self.stage_definition_id, "stage_definition_id")
        _require_nonempty_str(
            self.observation_definition_id, "observation_definition_id"
        )
        _require_nonempty_str(self.units, "units")
        if isinstance(self.threshold, bool) or not isinstance(
            self.threshold, (int, float)
        ):
            raise TypeError("threshold must be numeric")


@dataclass(frozen=True)
class FakeCandidatePlan:
    """Scripted stage evidence for one candidate, keyed by observation id.

    The three keys are the three distinct missingness modes, kept separate on
    purpose: a *scored* observation, one that is ``TECHNICALLY_UNAVAILABLE``
    (the detector-level analogue of a crash), and one that is simply never
    emitted. Only the first can yield a decision value at all.
    """

    scores: Mapping[str, float] = field(default_factory=dict)
    unavailable_observations: Tuple[str, ...] = ()
    missing_observations: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.scores, Mapping):
            raise TypeError("scores must be a mapping")
        for key, value in self.scores.items():
            _require_nonempty_str(key, "scores key")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("scores values must be numeric")
        object.__setattr__(
            self, "unavailable_observations", tuple(self.unavailable_observations)
        )
        object.__setattr__(
            self, "missing_observations", tuple(self.missing_observations)
        )
        overlap = set(self.unavailable_observations) & set(self.missing_observations)
        if overlap:
            raise ValueError(
                "an observation cannot be both technically unavailable and absent: "
                f"{sorted(overlap)}"
            )


@dataclass(frozen=True)
class FakeRunPlan:
    """Scripted structure of one execution attempt.

    ``emit_execution=False`` models a subject for which no run record exists
    at all. That is a real and different situation from a failed run, and the
    aggregation layer must be able to tell them apart.

    ``technical_failure=True`` models a run that crashed. Such a run emits no
    realizations, no candidates, no evidence and no decisions — there is
    nothing trustworthy to report — and its only mark is
    ``ExecutionStatus.TECHNICAL_FAILURE`` plus a free-text reason on the health
    axis (mission invariant 3.2).

    ``realization_candidate_counts`` has one entry per interaction realization;
    the value is how many candidates that realization yields. ``()`` means an
    execution that ran cleanly and produced nothing — a genuine physics zero,
    not a failure.
    """

    emit_execution: bool = True
    technical_failure: bool = False
    failure_reason: str = "fake_backend: injected technical failure"
    realization_candidate_counts: Tuple[int, ...] = (1,)
    unattached_candidate_count: int = 0
    candidate_plans: Tuple[FakeCandidatePlan, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.emit_execution, bool):
            raise TypeError("emit_execution must be a bool")
        if not isinstance(self.technical_failure, bool):
            raise TypeError("technical_failure must be a bool")
        counts = tuple(self.realization_candidate_counts)
        for count in counts:
            _require_non_negative_int(count, "realization_candidate_counts entry")
        _require_non_negative_int(
            self.unattached_candidate_count, "unattached_candidate_count"
        )
        plans = tuple(self.candidate_plans)
        for plan in plans:
            if not isinstance(plan, FakeCandidatePlan):
                raise TypeError("candidate_plans must contain FakeCandidatePlan objects")
        if self.technical_failure and (
            counts or self.unattached_candidate_count or plans
        ):
            raise ValueError(
                "a technically failed run has no trustworthy output: it must not "
                "also script realizations, candidates, or evidence"
            )
        object.__setattr__(self, "realization_candidate_counts", counts)
        object.__setattr__(self, "candidate_plans", plans)

    @property
    def total_candidate_count(self) -> int:
        return sum(self.realization_candidate_counts) + self.unattached_candidate_count


class FakeFairShipBackend:
    """A deterministic, FairShip-shaped fake. Not a physics model.

    ``plans`` maps ``(subject_id, replication_index)`` to an explicit
    :class:`FakeRunPlan`. Anything unscripted falls back to a hash-derived
    structure bounded by ``max_realizations`` and
    ``max_candidates_per_realization``, with a derived technical failure
    whenever ``technical_failure_modulus`` divides the structural draw
    (``0`` disables derived failures entirely).
    """

    is_physical = False
    version = BACKEND_VERSION

    def __init__(
        self,
        *,
        stages: Sequence[FakeStageSpec],
        plans: Optional[Mapping[Tuple[str, int], FakeRunPlan]] = None,
        max_realizations: int = 2,
        max_candidates_per_realization: int = 2,
        technical_failure_modulus: int = 0,
        name: str = BACKEND_NAME,
    ) -> None:
        stage_specs = tuple(stages)
        for stage in stage_specs:
            if not isinstance(stage, FakeStageSpec):
                raise TypeError("stages must contain FakeStageSpec objects")
        stage_ids = [stage.stage_definition_id for stage in stage_specs]
        if len(set(stage_ids)) != len(stage_ids):
            raise ValueError("stages must have distinct stage_definition_ids")
        observation_ids = [stage.observation_definition_id for stage in stage_specs]
        if len(set(observation_ids)) != len(observation_ids):
            # Sharing one observable across stages would make the stages
            # logically dependent, which mission invariant 3.4 forbids
            # assuming.
            raise ValueError("stages must have distinct observation_definition_ids")

        scripted = dict(plans or {})
        for key, plan in scripted.items():
            if (
                not isinstance(key, tuple)
                or len(key) != 2
                or not isinstance(key[0], str)
                or isinstance(key[1], bool)
                or not isinstance(key[1], int)
            ):
                raise TypeError("plans keys must be (subject_id, replication_index)")
            if not isinstance(plan, FakeRunPlan):
                raise TypeError("plans values must be FakeRunPlan objects")

        _require_nonempty_str(name, "name")
        self.name = name
        self._stages = stage_specs
        self._plans = scripted
        self._max_realizations = _require_non_negative_int(
            max_realizations, "max_realizations"
        )
        self._max_candidates_per_realization = _require_non_negative_int(
            max_candidates_per_realization, "max_candidates_per_realization"
        )
        self._technical_failure_modulus = _require_non_negative_int(
            technical_failure_modulus, "technical_failure_modulus"
        )

    # -- deterministic draws ------------------------------------------------

    def _draw(
        self,
        request: EvaluationRequest,
        subject: TagSubject,
        replication_index: int,
        purpose: str,
    ) -> int:
        """A reproducible integer for one (subject, replication, purpose).

        ``request_id`` and ``options`` are excluded on purpose: relabelling a
        request or changing an output path must not change the answer, only
        the identifiers. ``fs_sim_configuration_id`` *is* included, because a
        configuration change genuinely changes the estimand (``COMPAT-01``)
        and the fake must reflect that rather than hide it.
        """
        digest = content_hash(
            {
                "backend": self.name,
                "version": self.version,
                "seed": request.seed,
                "fs_sim_configuration_id": request.fs_sim_configuration_id,
                "subject_id": subject.subject_id,
                "state_definition_id": subject.state_definition_id,
                "replication_index": replication_index,
                "purpose": purpose,
            }
        )
        return int(digest[:16], 16)

    def _derived_plan(
        self,
        request: EvaluationRequest,
        subject: TagSubject,
        replication_index: int,
    ) -> FakeRunPlan:
        draw = self._draw(request, subject, replication_index, "structure")
        if self._technical_failure_modulus and draw % self._technical_failure_modulus == 0:
            return FakeRunPlan(
                technical_failure=True,
                failure_reason="fake_backend: derived technical failure",
                realization_candidate_counts=(),
            )
        realization_count = (draw >> 16) % (self._max_realizations + 1)
        counts = tuple(
            (draw >> (24 + 8 * index)) % (self._max_candidates_per_realization + 1)
            for index in range(realization_count)
        )
        return FakeRunPlan(realization_candidate_counts=counts)

    def _derived_score(
        self,
        request: EvaluationRequest,
        subject: TagSubject,
        replication_index: int,
        candidate_ordinal: int,
        observation_definition_id: str,
    ) -> float:
        draw = self._draw(
            request,
            subject,
            replication_index,
            f"score:{candidate_ordinal}:{observation_definition_id}",
        )
        return (draw % 1_000_000) / 1_000_000.0

    # -- the boundary contract ---------------------------------------------

    def evaluate(self, request: EvaluationRequest) -> EvaluationBundle:
        if not isinstance(request, EvaluationRequest):
            raise TypeError("request must be an EvaluationRequest")

        executions = []
        realizations = []
        candidates = []
        observations = []
        decisions = []

        for subject in request.subjects:
            for replication_index in range(request.replications_per_subject):
                plan = self._plans.get((subject.subject_id, replication_index))
                if plan is None:
                    plan = self._derived_plan(request, subject, replication_index)
                if not plan.emit_execution:
                    continue

                execution_id = (
                    f"exec:{request.request_id}:{subject.subject_id}:{replication_index}"
                )
                failed = plan.technical_failure
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
                        seed=self._draw(request, subject, replication_index, "seed")
                        % (2**31),
                        failure_reason=plan.failure_reason if failed else "",
                        provenance={
                            "backend_name": self.name,
                            "backend_version": self.version,
                            "is_physical": "false",
                            "request_id": request.request_id,
                            "request_seed": str(request.seed),
                            "replication_index": str(replication_index),
                            "fs_sim_configuration_id": request.fs_sim_configuration_id,
                            "state_definition_id": subject.state_definition_id,
                        },
                    )
                )
                if failed:
                    # Nothing else is emitted. A crashed run has no physics
                    # outcome to report, not even a negative one.
                    continue

                candidate_ordinal = 0
                for realization_index, candidate_count in enumerate(
                    plan.realization_candidate_counts
                ):
                    realization_id = f"real:{execution_id}:{realization_index}"
                    realizations.append(
                        InteractionRealization(
                            realization_id=realization_id,
                            execution_id=execution_id,
                            interaction_type=FIXTURE_INTERACTION_TYPE,
                            interaction_definition_id=FIXTURE_INTERACTION_DEFINITION_ID,
                        )
                    )
                    for within_realization in range(candidate_count):
                        self._emit_candidate(
                            request=request,
                            subject=subject,
                            replication_index=replication_index,
                            plan=plan,
                            execution_id=execution_id,
                            realization_id=realization_id,
                            candidate_ordinal=candidate_ordinal,
                            candidate_index=within_realization,
                            candidates=candidates,
                            observations=observations,
                            decisions=decisions,
                        )
                        candidate_ordinal += 1

                for within_execution in range(plan.unattached_candidate_count):
                    self._emit_candidate(
                        request=request,
                        subject=subject,
                        replication_index=replication_index,
                        plan=plan,
                        execution_id=execution_id,
                        realization_id=None,
                        candidate_ordinal=candidate_ordinal,
                        candidate_index=within_execution,
                        candidates=candidates,
                        observations=observations,
                        decisions=decisions,
                    )
                    candidate_ordinal += 1

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

    def _emit_candidate(
        self,
        *,
        request: EvaluationRequest,
        subject: TagSubject,
        replication_index: int,
        plan: FakeRunPlan,
        execution_id: str,
        realization_id: Optional[str],
        candidate_ordinal: int,
        candidate_index: int,
        candidates: list,
        observations: list,
        decisions: list,
    ) -> None:
        candidate_id = f"cand:{execution_id}:{candidate_ordinal}"
        candidates.append(
            ReconstructedCandidate(
                candidate_id=candidate_id,
                execution_id=execution_id,
                realization_id=realization_id,
                candidate_index=candidate_index,
            )
        )
        candidate_plan = (
            plan.candidate_plans[candidate_ordinal]
            if candidate_ordinal < len(plan.candidate_plans)
            else FakeCandidatePlan()
        )

        for stage in self._stages:
            observation_definition_id = stage.observation_definition_id
            if observation_definition_id in candidate_plan.missing_observations:
                decisions.append(
                    self._decision(
                        stage,
                        candidate_id,
                        DecisionEvaluationStatus.NOT_EVALUATED,
                        None,
                        (),
                        f"required_observation_missing:{observation_definition_id}",
                    )
                )
                continue

            observation_id = f"obs:{candidate_id}:{observation_definition_id}"
            if observation_definition_id in candidate_plan.unavailable_observations:
                observations.append(
                    ObservationEnvelope(
                        observation_id=observation_id,
                        subject_ref=candidate_id,
                        observation_definition_id=observation_definition_id,
                        units=stage.units,
                        evaluation_status=(
                            ObservationEvaluationStatus.TECHNICALLY_UNAVAILABLE
                        ),
                        evidence_reference=f"{self.name}:{execution_id}",
                        payload=None,
                        config_provenance=self._config_provenance(request),
                    )
                )
                decisions.append(
                    self._decision(
                        stage,
                        candidate_id,
                        DecisionEvaluationStatus.TECHNICALLY_UNAVAILABLE,
                        None,
                        (observation_id,),
                        "required_observation_technically_unavailable:"
                        f"{observation_definition_id}",
                    )
                )
                continue

            if observation_definition_id in candidate_plan.scores:
                value = float(candidate_plan.scores[observation_definition_id])
            else:
                value = self._derived_score(
                    request,
                    subject,
                    replication_index,
                    candidate_ordinal,
                    observation_definition_id,
                )
            observations.append(
                ObservationEnvelope(
                    observation_id=observation_id,
                    subject_ref=candidate_id,
                    observation_definition_id=observation_definition_id,
                    units=stage.units,
                    evaluation_status=ObservationEvaluationStatus.COMPUTED,
                    evidence_reference=f"{self.name}:{execution_id}",
                    payload=ScalarObservationPayload(value=value),
                    config_provenance=self._config_provenance(request),
                )
            )
            decisions.append(
                self._decision(
                    stage,
                    candidate_id,
                    DecisionEvaluationStatus.EVALUATED,
                    bool(value > float(stage.threshold)),
                    (observation_id,),
                    "",
                )
            )

    def _config_provenance(self, request: EvaluationRequest) -> Mapping[str, str]:
        return {
            "fs_sim_configuration_id": request.fs_sim_configuration_id,
            "backend_name": self.name,
            "backend_version": self.version,
        }

    def _decision(
        self,
        stage: FakeStageSpec,
        subject_ref: str,
        evaluation_status: DecisionEvaluationStatus,
        decision: Optional[Any],
        evidence_references: Tuple[str, ...],
        reason: str,
    ) -> StageDecision:
        return StageDecision(
            decision_id=stage_decision_id(
                stage_definition_id=stage.stage_definition_id,
                subject_ref=subject_ref,
                evidence_references=evidence_references,
                evaluation_status=evaluation_status,
            ),
            subject_ref=subject_ref,
            stage_definition_id=stage.stage_definition_id,
            evaluation_status=evaluation_status,
            decision=decision,
            reason=reason,
            evidence_references=evidence_references,
        )
