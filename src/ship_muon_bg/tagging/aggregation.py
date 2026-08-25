"""State-level aggregation: canonical evaluation records -> proxy-ready table.

This is a pure transformation. It takes the flat canonical records any backend
produces — executions, candidates, stage decisions — and returns per-source-state
empirical summaries. It imports ``entities`` only: it never sees an
``EvaluationBundle``, a backend, FairShip, or ROOT, so swapping the simulator
cannot reach it (``tagging -> entities`` is the documented dependency
direction, ``scientific_architecture_v2.md`` §2a).

What this layer is careful about
--------------------------------

**The statistical unit is the execution, not the source state.** Several
executions of one ``TagSubject`` are conditional repetitions of that one
post-shield state; they are grouped under it, never spread out as independent
draws from the source measure (mission invariant 3.1).

**Three ways of having no answer are kept apart** (``CENSOR-01``/``CENSOR-02``):
a run that crashed (``TECHNICAL_FAILURE``), a run that finished but whose stage
evidence was unavailable (``TECHNICALLY_CENSORED``), and a run the stage was
never applied to (``NOT_EVALUATED``). None of them is a negative, and none of
them enters the denominator of ``eta_hat``. A negative is only ever a run that
was evaluated and came out negative.

``NOT_EVALUATED`` currently merges two things ``CENSOR-02`` distinguishes:
``NOT_APPLICABLE`` evidence and evidence that is simply missing. Excluding both
from the denominator is this layer's conservative default; ``CENSOR-04`` makes
that choice target-specific and leaves it [OPEN], and ``not_evaluated_count`` is
retained separately so a target that wants to include them can.

**Multiplicity is retained, not collapsed.** Every aggregate carries the full
distribution of candidate counts and of stage-positive candidate counts, so
both ``N`` and ``Y = 1{N >= 1}`` remain derivable after aggregation (mission
invariant 3.3).

**Configurations are never pooled by this layer.** ``fs_sim_configuration_id``
*and* the options digest recorded in each execution's provenance are both part
of the grouping key, so results obtained under different configurations — or
under one configuration label with different options — land in different rows
(``COMPAT-01``). Note the precise claim: the rows preserve the axis, they do
not defend it. A caller can still average a mixed table, which is what
``require_single_configuration`` exists to prevent and why any step that
concatenates, averages, or fits across rows must call it first.

**A negative must be attested, never inferred from silence.** An execution with
zero candidates and no decision saying the stage was applied is
``NOT_EVALUATED``, not a physics zero. The commonest real partial failure — a
job that exits 0 but whose reconstruction output is empty or truncated — is
otherwise indistinguishable from a clean zero, and resolving that ambiguity in
favour of a countable negative is precisely the corruption this layer exists to
prevent.

**No weights are applied anywhere in this module.** It emits counts and an
empirical frequency. A source's physical weight ``w_i`` and any utility
multiplier ``h(U_i)`` are separate, later, explicit decisions (``WEIGHT-01``).

What this layer explicitly does *not* claim
-------------------------------------------

``eta_hat`` is the observed frequency ``positive / valid``. It is **not**
presented as an unbiased estimator of anything, no independence between
repeated executions of one state is asserted, and no standard error is
computed — precisely because the correct uncertainty treatment for repeated
conditional simulations of one state is an open question. The raw counts are
retained so a different uncertainty method can be applied later without
re-running anything.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from ship_muon_bg.entities.decision import DecisionEvaluationStatus, StageDecision
from ship_muon_bg.entities.identifiers import definition_id
from ship_muon_bg.entities.lineage import (
    OPTIONS_DIGEST_PROVENANCE_KEY,
    ExecutionStatus,
    FSSimExecution,
    ReconstructedCandidate,
)
from ship_muon_bg.entities.subject import TagSubject

#: The one implemented policy for turning candidate-level records into one
#: outcome per execution, written out as data and content-addressed so that
#: changing the precedence necessarily changes the id stored on every row. A
#: bare version string would let two vintages of rows pool under one label —
#: the "names are not evidence" hazard the rest of the design avoids by
#: hashing (``COMPAT-02``, ``CONF-01``).
ROLLUP_RULE_CONTENT = {
    "unit_of_analysis": "execution",
    "precedence": [
        "a technically failed execution is TECHNICAL_FAILURE, whatever else is present",
        "an execution-level decision for the stage, when present, is authoritative, "
        "except that an identified positive candidate still wins over an "
        "execution-level censoring marker",
        "an execution-level decision contradicting an EVALUATED candidate decision "
        "under the same stage definition is refused, in both directions",
        "otherwise a single EVALUATED-true candidate makes the execution positive, "
        "regardless of what happened to its siblings",
        "otherwise technically unavailable candidate evidence censors the execution",
        "otherwise absent or not-applicable candidate evidence leaves it not evaluated",
        "an execution with zero candidates is NEGATIVE only when an execution-level "
        "decision attests the stage was applied; silence is NOT_EVALUATED",
        "all remaining evaluated executions are NEGATIVE",
    ],
    "negative_requires_attestation": True,
}

ROLLUP_RULE = definition_id("execution_stage_rollup_v0", ROLLUP_RULE_CONTENT)


class DecisionLevel(str, enum.Enum):
    """Which records decided one (execution, stage) pair.

    Recorded because ``Y`` and ``1{N >= 1}`` genuinely diverge on the
    ``EXECUTION`` path: a stage whose positive condition is the *absence* of a
    candidate (a veto) is legitimately positive with zero candidates. A
    consumer reconstructing ``Y`` from the multiplicity distribution needs to
    know which executions were decided that way rather than silently
    disagreeing with ``positive_count``.
    """

    EXECUTION = "execution"
    CANDIDATES = "candidates"
    NONE = "none"


class ExecutionStageOutcome(str, enum.Enum):
    """The outcome of one stage for one execution. Exactly one applies.

    The four non-positive values are deliberately not merged: collapsing any
    of the last three into ``NEGATIVE`` is the single failure mode this whole
    layer exists to prevent (mission invariant 3.2).
    """

    POSITIVE = "positive"
    NEGATIVE = "negative"
    TECHNICAL_FAILURE = "technical_failure"
    TECHNICALLY_CENSORED = "technically_censored"
    NOT_EVALUATED = "not_evaluated"


VALID_OUTCOMES = (ExecutionStageOutcome.POSITIVE, ExecutionStageOutcome.NEGATIVE)


@dataclass(frozen=True)
class ExecutionStageResult:
    """One execution's outcome for one stage, before grouping.

    Kept as an explicit intermediate rather than folded straight into counts:
    a reviewer can inspect exactly which execution was classified how, and the
    aggregate below is provably just a tally of these.
    """

    execution_id: str
    subject_id: str
    fs_sim_configuration_id: str
    stage_definition_id: str
    outcome: ExecutionStageOutcome
    candidate_count: int
    positive_candidate_count: int
    decision_level: DecisionLevel = DecisionLevel.NONE

    @property
    def is_valid(self) -> bool:
        return self.outcome in VALID_OUTCOMES


@dataclass(frozen=True)
class StateStageAggregate:
    """Empirical summary for one (source state, configuration, stage).

    ``eta_hat`` is ``positive_count / valid_count``, or ``None`` when
    ``valid_count`` is zero. ``None`` is used rather than ``0.0`` on purpose: a
    state with nothing evaluable has an *unknown* rate, and writing zero there
    would fabricate a confident negative out of missing data.

    The two distributions map a count to how many valid executions had that
    count, so multiplicity survives aggregation.
    """

    subject_id: str
    fs_sim_configuration_id: str
    options_digest: Optional[str]
    stage_definition_id: str
    state_definition_id: Optional[str]
    rollup_rule: str
    execution_count: int
    execution_decided_count: int
    valid_count: int
    positive_count: int
    negative_count: int
    technical_failure_count: int
    technically_censored_count: int
    not_evaluated_count: int
    candidate_count_distribution: Mapping[int, int]
    positive_candidate_count_distribution: Mapping[int, int]

    def __post_init__(self) -> None:
        if self.execution_decided_count > self.execution_count:
            raise ValueError("execution_decided_count cannot exceed execution_count")
        if self.valid_count != self.positive_count + self.negative_count:
            raise ValueError("valid_count must equal positive_count + negative_count")
        total = (
            self.valid_count
            + self.technical_failure_count
            + self.technically_censored_count
            + self.not_evaluated_count
        )
        if total != self.execution_count:
            raise ValueError(
                "every execution must fall into exactly one outcome class; "
                f"{self.execution_count} executions but {total} classified"
            )

    @property
    def eta_hat(self) -> Optional[float]:
        """Observed positive frequency among evaluable executions.

        Not an estimator with a claimed sampling distribution: repeated
        executions of one source state are conditional repetitions, and this
        module asserts nothing about their independence.
        """
        if self.valid_count == 0:
            return None
        return self.positive_count / self.valid_count

    @property
    def censored_count(self) -> int:
        """All executions with no trustworthy answer, of any kind."""
        return (
            self.technical_failure_count
            + self.technically_censored_count
            + self.not_evaluated_count
        )

    def as_record(self) -> Dict[str, Any]:
        """A flat, JSON-serializable row for the proxy-ready tagging table.

        Deliberately carries no training weight: choosing how to weight a
        state is a separate, explicit scientific decision, and emitting a
        single opaque weight here is exactly how ``w_i`` and ``h(U_i)`` get
        multiplied together by accident (``WEIGHT-01``).

        The two distributions are ``int``-keyed dicts. They survive
        ``json.dumps`` but come back with string keys, and they have no flat
        CSV/Parquet column representation — so whichever writer persists this
        table is where multiplicity is most likely to actually get dropped.
        Whatever that writer is, it must keep them.
        """
        return {
            "subject_id": self.subject_id,
            "state_definition_id": self.state_definition_id,
            "fs_sim_configuration_id": self.fs_sim_configuration_id,
            "options_digest": self.options_digest,
            "stage_definition_id": self.stage_definition_id,
            "rollup_rule": self.rollup_rule,
            "execution_count": self.execution_count,
            "execution_decided_count": self.execution_decided_count,
            "valid_count": self.valid_count,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "technical_failure_count": self.technical_failure_count,
            "technically_censored_count": self.technically_censored_count,
            "not_evaluated_count": self.not_evaluated_count,
            "eta_hat": self.eta_hat,
            "candidate_count_distribution": dict(self.candidate_count_distribution),
            "positive_candidate_count_distribution": dict(
                self.positive_candidate_count_distribution
            ),
        }

    @property
    def key(self) -> Tuple[str, str, Optional[str], str]:
        """The row's primary key: one state, one configuration, one stage."""
        return (
            self.subject_id,
            self.fs_sim_configuration_id,
            self.options_digest,
            self.stage_definition_id,
        )


@dataclass(frozen=True)
class TaggingDataset:
    """The proxy-ready artifact: aggregates plus the provenance to read them.

    A consumer needs no knowledge of FairShip, of which backend ran, or of how
    the records were produced — only the definition ids, which are
    content-addressed and therefore self-describing.
    """

    rows: Tuple[StateStageAggregate, ...]
    results: Tuple[ExecutionStageResult, ...] = ()
    unevaluated_subject_ids: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        rows = tuple(self.rows)
        keys = [row.key for row in rows]
        if len(set(keys)) != len(keys):
            duplicated = sorted({key for key in keys if keys.count(key) > 1})
            raise ValueError(
                "duplicate (subject, configuration, options, stage) rows: "
                f"{duplicated}. One source state must appear once; two rows for it "
                "make every row-level mean and every summed exposure count it "
                "twice. Combine batches by aggregating over the concatenated "
                "records, not by concatenating datasets"
            )
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "results", tuple(self.results))
        object.__setattr__(
            self, "unevaluated_subject_ids", tuple(self.unevaluated_subject_ids)
        )

    @property
    def configuration_ids(self) -> Tuple[str, ...]:
        return tuple(sorted({row.fs_sim_configuration_id for row in self.rows}))

    @property
    def stage_definition_ids(self) -> Tuple[str, ...]:
        return tuple(sorted({row.stage_definition_id for row in self.rows}))

    @property
    def subject_ids(self) -> Tuple[str, ...]:
        """Subjects that produced at least one row.

        Deliberately *not* the declared population: see
        ``unevaluated_subject_ids`` for subjects that produced no execution at
        all. Reading this as the population is how a selection-biased sample
        becomes invisible.
        """
        return tuple(sorted({row.subject_id for row in self.rows}))

    def for_stage(self, stage_definition_id: str) -> Tuple[StateStageAggregate, ...]:
        return tuple(
            row for row in self.rows if row.stage_definition_id == stage_definition_id
        )

    def for_configuration(
        self, fs_sim_configuration_id: str
    ) -> Tuple[StateStageAggregate, ...]:
        return tuple(
            row
            for row in self.rows
            if row.fs_sim_configuration_id == fs_sim_configuration_id
        )

    def as_records(self) -> Tuple[Dict[str, Any], ...]:
        return tuple(row.as_record() for row in self.rows)

    @property
    def state_definition_ids(self) -> Tuple[Optional[str], ...]:
        return tuple(sorted({row.state_definition_id for row in self.rows}, key=str))

    def require_single_state_definition(self) -> str:
        """Return the one state definition id, or refuse.

        ``COMPAT-01`` lists the state definition alongside geometry and seed as
        a compatibility axis: coordinates that mean different things must not
        be pooled. Raises when the aggregation was run without subject records,
        because the axis was then never checked at all.
        """
        definitions = self.state_definition_ids
        if len(definitions) != 1:
            raise ValueError(
                "these aggregates span multiple state_definition_ids and must not "
                f"be pooled: {list(definitions)}"
            )
        if definitions[0] is None:
            raise ValueError(
                "state definition unknown: this aggregation was run without "
                "subjects=, so this axis was never checked. A guard that passes on "
                "unverified data is worse than no guard"
            )
        return definitions[0]

    def require_single_configuration(self) -> str:
        """Return the one configuration id, or refuse.

        Any downstream step that would average, concatenate, or fit across
        rows must call this first: pooling incompatible configurations is not
        a rounding error, it silently changes the estimand (``COMPAT-01``).
        """
        configurations = self.configuration_ids
        if len(configurations) != 1:
            raise ValueError(
                "these aggregates span multiple fs_sim_configuration_ids and must "
                f"not be pooled: {list(configurations)}"
            )
        return configurations[0]


def _index_decisions(
    decisions: Sequence[StageDecision],
) -> Dict[Tuple[str, str], StageDecision]:
    index: Dict[Tuple[str, str], StageDecision] = {}
    for decision in decisions:
        if not isinstance(decision, StageDecision):
            raise TypeError("decisions must contain StageDecision objects")
        key = (decision.subject_ref, decision.stage_definition_id)
        existing = index.get(key)
        if existing is not None:
            if existing == decision:
                continue
            raise ValueError(
                "two different decisions for the same entity and stage are "
                f"ambiguous: {key!r}"
            )
        index[key] = decision
    return index


def classify_executions(
    *,
    executions: Sequence[FSSimExecution],
    candidates: Sequence[ReconstructedCandidate],
    decisions: Sequence[StageDecision],
    stage_definition_ids: Sequence[str],
) -> Tuple[ExecutionStageResult, ...]:
    """Classify every (execution, stage) pair into exactly one outcome.

    Raises rather than guessing when the records are inconsistent: duplicate
    execution ids (which would double-count a repetition), candidates whose
    parent execution is absent (broken lineage), or an execution-level decision
    that contradicts its own candidates for the same stage.
    """
    execution_list = tuple(executions)
    for execution in execution_list:
        if not isinstance(execution, FSSimExecution):
            raise TypeError("executions must contain FSSimExecution objects")
    execution_ids = [execution.execution_id for execution in execution_list]
    if len(set(execution_ids)) != len(execution_ids):
        raise ValueError(
            "duplicate execution_id: the same run counted twice would inflate "
            "the denominator of every rate derived from it"
        )
    known_executions = set(execution_ids)

    candidates_by_execution: Dict[str, list] = {
        execution_id: [] for execution_id in execution_ids
    }
    seen_candidate_ids = set()
    for candidate in candidates:
        if not isinstance(candidate, ReconstructedCandidate):
            raise TypeError("candidates must contain ReconstructedCandidate objects")
        if candidate.candidate_id in seen_candidate_ids:
            raise ValueError(f"duplicate candidate_id: {candidate.candidate_id!r}")
        seen_candidate_ids.add(candidate.candidate_id)
        if candidate.execution_id not in known_executions:
            raise ValueError(
                f"candidate {candidate.candidate_id!r} has no execution in this "
                f"record set (execution_id {candidate.execution_id!r})"
            )
        candidates_by_execution[candidate.execution_id].append(candidate)

    # Per-bundle integrity does not compose. Records concatenated from several
    # bundles can reuse one id in two roles, which would let a single decision
    # be consumed once as a candidate outcome and once as an execution outcome.
    collisions = known_executions & seen_candidate_ids
    if collisions:
        raise ValueError(
            f"identifiers {sorted(collisions)} name both an execution and a "
            "candidate in this record set; a decision referencing one would be "
            "consumed twice in two different roles"
        )

    # Materialize once: the record set is walked twice below, and a caller
    # passing a generator would otherwise have the second pass see nothing.
    decision_list = tuple(decisions)
    decision_index = _index_decisions(decision_list)
    stages = tuple(stage_definition_ids)
    if len(set(stages)) != len(stages):
        raise ValueError("stage_definition_ids must be distinct")

    # A decision this layer cannot interpret must never be silently dropped.
    # StageDecision.subject_ref is an untyped reference and a backend may
    # legitimately attach a decision to an InteractionRealization or to the
    # TagSubject itself (EXEC-02a names the realization as the canonical
    # lineage node, which is exactly where a DIS selection would sit). This
    # layer only knows how to roll up execution- and candidate-level records,
    # so anything else is refused loudly rather than ignored — ignoring it
    # turns an EVALUATED positive into a counted physics negative, and the
    # right rollup semantics for realization-scoped stages is an open
    # scientific question this module does not get to answer by omission.
    requested_stages = set(stages)
    interpretable = known_executions | seen_candidate_ids
    unhandled = sorted(
        {
            decision.subject_ref
            for decision in decision_list
            if decision.stage_definition_id in requested_stages
            and decision.subject_ref not in interpretable
        }
    )
    if unhandled:
        raise ValueError(
            "these stage decisions attach to entities this aggregation cannot "
            f"interpret: {unhandled}. Only execution-level and candidate-level "
            "decisions are rolled up; a realization- or subject-scoped decision "
            "needs rollup semantics that have not been defined, and a reference "
            "that matches nothing is broken lineage. Neither may be dropped"
        )

    results = []
    for execution in execution_list:
        execution_candidates = candidates_by_execution[execution.execution_id]
        for stage_definition_id in stages:
            results.append(
                _classify_one(
                    execution=execution,
                    execution_candidates=execution_candidates,
                    stage_definition_id=stage_definition_id,
                    decision_index=decision_index,
                )
            )
    return tuple(results)


def _classify_one(
    *,
    execution: FSSimExecution,
    execution_candidates: Sequence[ReconstructedCandidate],
    stage_definition_id: str,
    decision_index: Mapping[Tuple[str, str], StageDecision],
) -> ExecutionStageResult:
    candidate_count = len(execution_candidates)

    positive_candidates = 0
    evaluated_candidates = 0
    saw_technically_unavailable = False
    saw_not_evaluated = False
    for candidate in execution_candidates:
        decision = decision_index.get((candidate.candidate_id, stage_definition_id))
        if decision is None:
            # A candidate exists but this stage was never applied to it. That
            # is missing information, never evidence of a negative.
            saw_not_evaluated = True
            continue
        if decision.evaluation_status is DecisionEvaluationStatus.EVALUATED:
            evaluated_candidates += 1
            if decision.decision is True:
                positive_candidates += 1
            elif decision.decision is not False:
                raise ValueError(
                    "this aggregation supports boolean stage decisions only; "
                    f"got {decision.decision!r} for {decision.decision_id!r}"
                )
        elif (
            decision.evaluation_status
            is DecisionEvaluationStatus.TECHNICALLY_UNAVAILABLE
        ):
            saw_technically_unavailable = True
        else:
            saw_not_evaluated = True

    def _result(
        outcome: ExecutionStageOutcome,
        decision_level: DecisionLevel = DecisionLevel.NONE,
    ) -> ExecutionStageResult:
        return ExecutionStageResult(
            execution_id=execution.execution_id,
            subject_id=execution.subject_id,
            fs_sim_configuration_id=execution.fs_sim_configuration_id,
            stage_definition_id=stage_definition_id,
            outcome=outcome,
            candidate_count=candidate_count,
            positive_candidate_count=positive_candidates,
            decision_level=decision_level,
        )

    # 1. Run health dominates everything. A crashed run has no physics outcome.
    if execution.execution_status is ExecutionStatus.TECHNICAL_FAILURE:
        return _result(ExecutionStageOutcome.TECHNICAL_FAILURE)

    # 2. An execution-level decision, where a backend reports one, is the
    #    authoritative answer for that stage.
    execution_decision = decision_index.get((execution.execution_id, stage_definition_id))
    if execution_decision is not None:
        status = execution_decision.evaluation_status
        if status is DecisionEvaluationStatus.EVALUATED:
            _reject_contradiction(
                execution=execution,
                stage_definition_id=stage_definition_id,
                execution_decision=execution_decision,
                positive_candidates=positive_candidates,
                evaluated_candidates=evaluated_candidates,
            )
            if execution_decision.decision is True:
                # Y and 1{N >= 1} may legitimately diverge here: a stage whose
                # positive condition is the *absence* of a candidate is positive
                # with zero candidates. decision_level records which path this
                # took so a consumer never has to guess.
                return _result(ExecutionStageOutcome.POSITIVE, DecisionLevel.EXECUTION)
            return _result(ExecutionStageOutcome.NEGATIVE, DecisionLevel.EXECUTION)
        # The execution-level record says the stage could not be evaluated —
        # but an identified positive candidate is still an identified positive,
        # and discarding it would abandon the partial-identification logic
        # applied to sibling candidates four lines below.
        if positive_candidates:
            return _result(ExecutionStageOutcome.POSITIVE, DecisionLevel.CANDIDATES)
        if status is DecisionEvaluationStatus.TECHNICALLY_UNAVAILABLE:
            return _result(
                ExecutionStageOutcome.TECHNICALLY_CENSORED, DecisionLevel.EXECUTION
            )
        return _result(ExecutionStageOutcome.NOT_EVALUATED, DecisionLevel.EXECUTION)

    # 3. Roll up the candidates. A positive candidate settles the question
    #    regardless of what happened to its siblings; censoring only matters
    #    when it could still have changed the answer.
    if positive_candidates:
        return _result(ExecutionStageOutcome.POSITIVE, DecisionLevel.CANDIDATES)
    if saw_technically_unavailable:
        return _result(
            ExecutionStageOutcome.TECHNICALLY_CENSORED, DecisionLevel.CANDIDATES
        )
    if saw_not_evaluated:
        return _result(ExecutionStageOutcome.NOT_EVALUATED, DecisionLevel.CANDIDATES)
    if candidate_count == 0:
        # Nothing attests that this stage was ever applied. "Ran and found
        # nothing" and "never looked" are indistinguishable from an empty
        # record set, and calling the ambiguity a negative would manufacture a
        # confident physics zero out of a silent reconstruction failure. A
        # backend that really did look reports it with an execution-level
        # decision.
        return _result(ExecutionStageOutcome.NOT_EVALUATED, DecisionLevel.NONE)
    # Candidates exist and every one of them was evaluated negative: the stage
    # demonstrably ran and found nothing. This is the only inferred NEGATIVE.
    return _result(ExecutionStageOutcome.NEGATIVE, DecisionLevel.CANDIDATES)


def _reject_contradiction(
    *,
    execution: FSSimExecution,
    stage_definition_id: str,
    execution_decision: StageDecision,
    positive_candidates: int,
    evaluated_candidates: int,
) -> None:
    """Refuse two opposite conclusions under one versioned rule, either way round.

    Symmetric on purpose. Preferring the execution-level record would overwrite
    candidate-level stage semantics; preferring the candidates would overwrite
    the backend's own report. A rule that can conclude both things about one run
    is not one rule, and a veto needs its own ``stage_definition_id``.
    """
    value = execution_decision.decision
    if value is not True and value is not False:
        raise ValueError(
            "this aggregation supports boolean stage decisions only; got "
            f"{value!r} for {execution_decision.decision_id!r}"
        )
    if value is False and positive_candidates:
        raise ValueError(
            f"execution {execution.execution_id!r} reports stage "
            f"{stage_definition_id!r} negative while {positive_candidates} of its "
            "candidates report positive; a veto must be a distinct stage "
            "definition, not a contradiction under the same one"
        )
    if value is True and evaluated_candidates and not positive_candidates:
        raise ValueError(
            f"execution {execution.execution_id!r} reports stage "
            f"{stage_definition_id!r} positive while all {evaluated_candidates} of "
            "its evaluated candidates report negative; a stage whose positive "
            "condition is the absence of a candidate must be a distinct stage "
            "definition, not a contradiction under the same one"
        )


def aggregate_state_stage_evaluations(
    *,
    executions: Sequence[FSSimExecution],
    candidates: Sequence[ReconstructedCandidate],
    decisions: Sequence[StageDecision],
    stage_definition_ids: Sequence[str],
    subjects: Optional[Iterable[TagSubject]] = None,
    allowed_configuration_ids: Optional[Iterable[str]] = None,
) -> TaggingDataset:
    """Group canonical records into one aggregate per state/configuration/stage.

    ``allowed_configuration_ids`` filters executions before grouping, for the
    common case of deliberately restricting an analysis to one configuration.
    It never *merges* configurations: grouping is always keyed on the
    configuration id, so even an unfiltered call keeps incompatible settings in
    separate rows (``COMPAT-01``).

    ``subjects``, when supplied, is used both to attach each row's
    ``state_definition_id`` and to reject executions attributed to a source
    state the caller never declared.
    """
    subject_index = None
    if subjects is not None:
        subject_index = {}
        for subject in subjects:
            if not isinstance(subject, TagSubject):
                raise TypeError("subjects must contain TagSubject objects")
            if subject.subject_id in subject_index:
                raise ValueError(f"duplicate subject_id: {subject.subject_id!r}")
            subject_index[subject.subject_id] = subject

    selected = tuple(executions)
    if allowed_configuration_ids is not None:
        if isinstance(allowed_configuration_ids, (str, bytes)):
            # set("cfg_a") is a set of characters, which would silently select
            # nothing and hand back an empty table.
            raise TypeError(
                "allowed_configuration_ids must be a collection of ids, not a "
                "single string"
            )
        allowed = set(allowed_configuration_ids)
        if not allowed:
            raise ValueError("allowed_configuration_ids must not be empty")
        selected = tuple(
            execution
            for execution in selected
            if execution.fs_sim_configuration_id in allowed
        )
    # Dropping a candidate whose execution was filtered out by configuration is
    # correct; silently dropping one whose execution does not exist at all
    # would hide broken lineage and understate multiplicity. The two cases are
    # separated explicitly rather than collapsed into one filter.
    all_execution_ids = {execution.execution_id for execution in executions}
    for candidate in candidates:
        if not isinstance(candidate, ReconstructedCandidate):
            raise TypeError("candidates must contain ReconstructedCandidate objects")
        if candidate.execution_id not in all_execution_ids:
            raise ValueError(
                f"candidate {candidate.candidate_id!r} has no execution in this "
                f"record set (execution_id {candidate.execution_id!r})"
            )
    kept_execution_ids = {execution.execution_id for execution in selected}
    kept_candidates = tuple(
        candidate
        for candidate in candidates
        if candidate.execution_id in kept_execution_ids
    )

    if subject_index is not None:
        for execution in selected:
            if execution.subject_id not in subject_index:
                raise ValueError(
                    f"execution {execution.execution_id!r} is attributed to subject "
                    f"{execution.subject_id!r}, which was not declared"
                )

    results = classify_executions(
        executions=selected,
        candidates=kept_candidates,
        decisions=decisions,
        stage_definition_ids=stage_definition_ids,
    )

    options_digest_by_execution = {
        execution.execution_id: execution.provenance.get(
            OPTIONS_DIGEST_PROVENANCE_KEY
        )
        for execution in selected
    }

    grouped: Dict[Tuple[str, str, str, str], list] = {}
    for result in results:
        key = (
            result.subject_id,
            result.fs_sim_configuration_id,
            # Part of the key, not decoration: a configuration *label* cannot
            # cover a free-form options mapping, so two runs whose options
            # differ must not share a row even under one configuration id.
            options_digest_by_execution.get(result.execution_id) or "",
            result.stage_definition_id,
        )
        grouped.setdefault(key, []).append(result)

    rows = []
    for key in sorted(grouped):
        subject_id, configuration_id, options_digest, stage_definition_id = key
        group = grouped[key]
        tally = {outcome: 0 for outcome in ExecutionStageOutcome}
        candidate_distribution: Dict[int, int] = {}
        positive_distribution: Dict[int, int] = {}
        for result in group:
            tally[result.outcome] += 1
            if result.is_valid:
                candidate_distribution[result.candidate_count] = (
                    candidate_distribution.get(result.candidate_count, 0) + 1
                )
                positive_distribution[result.positive_candidate_count] = (
                    positive_distribution.get(result.positive_candidate_count, 0) + 1
                )
        state_definition_id = None
        if subject_index is not None:
            state_definition_id = subject_index[subject_id].state_definition_id
        rows.append(
            StateStageAggregate(
                subject_id=subject_id,
                fs_sim_configuration_id=configuration_id,
                options_digest=options_digest or None,
                stage_definition_id=stage_definition_id,
                state_definition_id=state_definition_id,
                rollup_rule=ROLLUP_RULE,
                execution_count=len(group),
                execution_decided_count=sum(
                    1
                    for result in group
                    if result.decision_level is DecisionLevel.EXECUTION
                ),
                valid_count=tally[ExecutionStageOutcome.POSITIVE]
                + tally[ExecutionStageOutcome.NEGATIVE],
                positive_count=tally[ExecutionStageOutcome.POSITIVE],
                negative_count=tally[ExecutionStageOutcome.NEGATIVE],
                technical_failure_count=tally[ExecutionStageOutcome.TECHNICAL_FAILURE],
                technically_censored_count=tally[
                    ExecutionStageOutcome.TECHNICALLY_CENSORED
                ],
                not_evaluated_count=tally[ExecutionStageOutcome.NOT_EVALUATED],
                candidate_count_distribution=dict(sorted(candidate_distribution.items())),
                positive_candidate_count_distribution=dict(
                    sorted(positive_distribution.items())
                ),
            )
        )
    covered = {row.subject_id for row in rows}
    unevaluated = ()
    if subject_index is not None:
        # A declared state that produced no execution vanishes from a table
        # built only from execution results. If dropped work correlates with
        # anything physical - long jobs, high-energy muons, timeouts - the
        # proxy then trains on a selection-biased population with nothing in
        # the artifact to reveal it.
        unevaluated = tuple(
            sorted(subject_id for subject_id in subject_index if subject_id not in covered)
        )
    return TaggingDataset(
        rows=tuple(rows), results=results, unevaluated_subject_ids=unevaluated
    )
