"""The canonical evaluation boundary: request in, lineage-preserving records out.

This module is the *stable contract* that lets the scientific core treat an
expensive simulator (FairShip today, anything else tomorrow) as an
interchangeable backend. It defines three things and nothing more:

``EvaluationRequest``
    What the core asks for, expressed only in canonical identifiers.
``EvaluationBundle``
    What a backend returns: a flat, normalized set of the canonical records
    of ``ship_muon_bg.entities`` — never a denormalized nested tree.
``EvaluationBackend``
    The structural protocol both a fake backend and a future real FairShip
    adapter satisfy.

Deliberate non-duplication (mission §5 "prefer relationships and identifiers"):
this module declares **no** field for source kinematics, physical weights,
detector observables, or stage semantics. Those already have canonical homes
(``ObservationEnvelope`` with its own ``observation_definition_id`` and
provenance; ``StageDefinition``/``StageDecision``). The request and the bundle
carry those canonical records and the identifiers linking them; they never
restate their content in new fields. In particular there is no
``weight``/``final_weight`` field anywhere here (``WEIGHT-01``): a source's
physical weight is an ``ObservationEnvelope`` like any other evidence, so it
can never be silently multiplied into a utility multiplier ``h(U)``.

Replication is expressed as ``EvaluationRequest.replications_per_subject``,
not by listing the same subject twice — the request rejects a repeated
``subject_id``. Be precise about what that buys: it is a check on the
*identifier*, so it stops the accidental case (a state appended twice to a
list) but not the case where one physical state is minted under two different
``subject_id``s. ``subject_id`` is a caller-supplied opaque string with no
relation to the state's coordinates, so guaranteeing one id per physical state
remains the request builder's obligation. Whether ``subject_id`` should instead
be content-addressed over the state coordinates is genuinely unresolved — two
distinct muons can share rounded coordinates — and is recorded as an open
question rather than decided here (mission invariant 3.1, ``COMPAT-04``).

Backend-independent: imports neither ROOT, FairShip, nor
``ship_muon_bg.adapters.fairship`` (guarded by
``tests/test_architecture_boundaries.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, Tuple, runtime_checkable

from ship_muon_bg.entities.decision import DecisionEvaluationStatus, StageDecision
from ship_muon_bg.entities.lineage import (
    ExecutionStatus,
    FSSimExecution,
    InteractionRealization,
    ReconstructedCandidate,
)
from ship_muon_bg.entities.identifiers import content_hash
from ship_muon_bg.entities.observation import ObservationEnvelope
from ship_muon_bg.entities.subject import TagSubject


def _require_nonempty_str(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int")
    return value


def _require_tuple_of(value: object, kind: type, field_name: str) -> Tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise TypeError(f"{field_name} must be a tuple of {kind.__name__}")
    items = tuple(value)
    for item in items:
        if not isinstance(item, kind):
            raise TypeError(f"{field_name} must contain only {kind.__name__} objects")
    return items


def _reject_duplicates(ids: Tuple[str, ...], field_name: str) -> None:
    seen = set()
    for value in ids:
        if value in seen:
            raise ValueError(f"duplicate {field_name}: {value!r}")
        seen.add(value)


@dataclass(frozen=True)
class EvaluationRequest:
    """One evaluation order placed against a backend.

    ``subjects`` carries canonical ``TagSubject`` identity records (three
    fields — an id, a declared type, and the versioned state definition that
    gives its coordinates meaning). It does **not** carry the subjects'
    kinematics: those travel as ``subject_observations``, keeping the
    existing observation/provenance model as the single home for physical
    content (``OBS-01``, ``WEIGHT-01``).

    ``replications_per_subject`` is how many times the backend is asked to
    re-evaluate *each* subject. It is the only sanctioned way to express
    repetition; ``subjects`` must not contain the same ``subject_id`` twice,
    so a repeat can never be mistaken for an independent source draw
    (mission invariant 3.1).

    ``fs_sim_configuration_id`` pins the configuration the whole request runs
    under. Results produced under different configuration ids estimate
    different quantities and must not be pooled downstream (``COMPAT-01``);
    this field is the immutable provenance that makes that check possible.

    ``options`` is free backend-specific configuration (paths, environment
    profiles, resource limits). It is deliberately opaque to the core and must
    never carry scientific meaning: anything the core reasons about belongs in
    a versioned definition id, not in this mapping. Because "must never" is a
    rule and not a mechanism, ``options_digest`` content-addresses it, and
    :func:`assert_requests_compatible` uses that digest to catch the specific
    hazard this affords — two runs sharing one ``fs_sim_configuration_id``
    while their options differ in something that actually changed the physics
    (a geometry file, a physics list). Without the digest that drift is
    undetectable downstream and the two pool as one estimand (``COMPAT-01``).

    Subjects with different ``state_definition_id``s may share one request:
    evaluating a mixed population in one batch is legitimate, and the
    per-subject definition is preserved on every ``TagSubject``. What is *not*
    legitimate is pooling across them afterwards, since ``COMPAT-01`` lists the
    state definition as a compatibility axis. The bundle records executions by
    ``subject_id`` only, so a persisted bundle is self-describing on that axis
    only in company with its request or the subject records — see
    ``state_definition_ids`` and ``TaggingDataset.require_single_state_definition``.
    """

    request_id: str
    subjects: Tuple[TagSubject, ...]
    fs_sim_configuration_id: str
    seed: int
    replications_per_subject: int = 1
    subject_observations: Tuple[ObservationEnvelope, ...] = ()
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_nonempty_str(self.request_id, "request_id")
        _require_nonempty_str(self.fs_sim_configuration_id, "fs_sim_configuration_id")

        subjects = _require_tuple_of(self.subjects, TagSubject, "subjects")
        subject_ids = tuple(subject.subject_id for subject in subjects)
        # Two entries for one state would silently become two independent
        # nominal draws downstream. Repetition must go through
        # ``replications_per_subject`` instead (mission invariant 3.1).
        _reject_duplicates(subject_ids, "subject_id in subjects")

        observations = _require_tuple_of(
            self.subject_observations, ObservationEnvelope, "subject_observations"
        )
        _reject_duplicates(
            tuple(observation.observation_id for observation in observations),
            "observation_id in subject_observations",
        )
        known_subjects = set(subject_ids)
        for observation in observations:
            if observation.subject_ref not in known_subjects:
                raise ValueError(
                    "subject_observations may only describe subjects declared in this "
                    f"request; {observation.observation_id!r} references unknown "
                    f"subject_ref {observation.subject_ref!r}"
                )

        seed = _require_int(self.seed, "seed")
        if seed < 0:
            raise ValueError("seed must be non-negative")
        replications = _require_int(
            self.replications_per_subject, "replications_per_subject"
        )
        if replications < 1:
            raise ValueError("replications_per_subject must be at least 1")
        if not isinstance(self.options, Mapping):
            raise TypeError("options must be a mapping")

        object.__setattr__(self, "subjects", subjects)
        object.__setattr__(self, "subject_observations", observations)

    @property
    def subject_ids(self) -> Tuple[str, ...]:
        """Declared subject ids, in request order."""
        return tuple(subject.subject_id for subject in self.subjects)

    @property
    def state_definition_ids(self) -> Tuple[str, ...]:
        """Distinct state definitions represented in this request, sorted."""
        return tuple(sorted({subject.state_definition_id for subject in self.subjects}))

    @property
    def options_digest(self) -> str:
        """Content hash of ``options`` — the handle for detecting silent drift."""
        return content_hash(dict(self.options))


@dataclass(frozen=True)
class EvaluationBundle:
    """A backend's answer: the canonical records it produced, flat and linked.

    This is a *normalized record set*, not an aggregate and not a nested
    object graph. Each collection holds the canonical entity type already
    defined in ``ship_muon_bg.entities``; parents and children are joined by
    identifier, exactly as they would be in the eventual on-disk tables.
    Nothing here is derived: there are no counts, no rates, and no
    ``eta_hat``. Deriving those is the aggregation layer's job, and it must
    stay separable so a different estimator can be swapped in later.

    ``__post_init__`` enforces the integrity that can be checked without
    knowing the request: identifiers are unique across *all* record kinds in
    the bundle (so a downstream ``subject_ref`` lookup can never resolve to
    two different kinds of thing), and every child points at a parent present
    in this same bundle. Conformance to the request that produced it —
    subject coverage and configuration agreement — is checked separately by
    :func:`verify_evaluation_bundle`.

    ``is_physical`` restates the producing backend's own declaration *inside
    the record set*, because the records are what get persisted, pooled and
    eventually plotted — while the backend object does not travel with them.
    A downstream report generator can therefore refuse to cite fake output
    without having to recognise a backend by name. It is a self-declaration
    and nothing more: it stops an honest mistake, not a mislabelled backend.

    Cardinalities stay ``0..N`` everywhere by construction: a bundle may hold
    zero executions for a subject, zero realizations for an execution, and
    zero, one, or many candidates (``EXEC-01``, ``EXEC-02``). Multiplicity is
    always *the number of records present*, never a stored count.

    A technically failed execution may carry evidence and censoring markers but
    **never** realizations, candidates, or an evaluated decision: a run with no
    trustworthy answer must not contribute countable physics objects that a
    later ``groupby`` could tally. Partial or truncated output from such a run
    belongs in ``FSSimExecution.provenance``, not in the candidate table.
    """

    request_id: str
    backend_name: str
    is_physical: bool
    executions: Tuple[FSSimExecution, ...] = ()
    realizations: Tuple[InteractionRealization, ...] = ()
    candidates: Tuple[ReconstructedCandidate, ...] = ()
    observations: Tuple[ObservationEnvelope, ...] = ()
    decisions: Tuple[StageDecision, ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty_str(self.request_id, "request_id")
        _require_nonempty_str(self.backend_name, "backend_name")
        if not isinstance(self.is_physical, bool):
            raise TypeError("is_physical must be a bool")

        executions = _require_tuple_of(self.executions, FSSimExecution, "executions")
        realizations = _require_tuple_of(
            self.realizations, InteractionRealization, "realizations"
        )
        candidates = _require_tuple_of(
            self.candidates, ReconstructedCandidate, "candidates"
        )
        observations = _require_tuple_of(
            self.observations, ObservationEnvelope, "observations"
        )
        decisions = _require_tuple_of(self.decisions, StageDecision, "decisions")

        execution_ids = tuple(item.execution_id for item in executions)
        realization_ids = tuple(item.realization_id for item in realizations)
        candidate_ids = tuple(item.candidate_id for item in candidates)
        observation_ids = tuple(item.observation_id for item in observations)
        decision_ids = tuple(item.decision_id for item in decisions)

        _reject_duplicates(execution_ids, "execution_id")
        _reject_duplicates(realization_ids, "realization_id")
        _reject_duplicates(candidate_ids, "candidate_id")
        _reject_duplicates(observation_ids, "observation_id")
        _reject_duplicates(decision_ids, "decision_id")

        # Cross-kind uniqueness over *every* id in the bundle.
        # ``ObservationEnvelope.subject_ref``, ``StageDecision.subject_ref``
        # and ``StageDecision.evidence_references`` are untyped id references
        # by design, so an id reused for two kinds makes attribution
        # ambiguous: a candidate-level decision could be tallied as an
        # execution-level one, or a cited piece of evidence could resolve to
        # either an observation or the run that produced it.
        _reject_duplicates(
            execution_ids
            + realization_ids
            + candidate_ids
            + observation_ids
            + decision_ids,
            "identifier reused across record kinds",
        )

        known_executions = set(execution_ids)
        realization_owner = {
            item.realization_id: item.execution_id for item in realizations
        }
        for realization in realizations:
            if realization.execution_id not in known_executions:
                raise ValueError(
                    f"realization {realization.realization_id!r} references "
                    f"unknown execution_id {realization.execution_id!r}"
                )
        for candidate in candidates:
            if candidate.execution_id not in known_executions:
                raise ValueError(
                    f"candidate {candidate.candidate_id!r} references unknown "
                    f"execution_id {candidate.execution_id!r}"
                )
            if candidate.realization_id is None:
                continue
            if candidate.realization_id not in realization_owner:
                raise ValueError(
                    f"candidate {candidate.candidate_id!r} references unknown "
                    f"realization_id {candidate.realization_id!r}"
                )
            if realization_owner[candidate.realization_id] != candidate.execution_id:
                raise ValueError(
                    f"candidate {candidate.candidate_id!r} claims execution "
                    f"{candidate.execution_id!r} but its realization belongs to "
                    f"execution {realization_owner[candidate.realization_id]!r}"
                )

        # ``candidate_index`` is scoped to ``(execution_id, realization_id)``.
        # Left unchecked it is the first key in the schema on which an ordinary
        # dict/groupby/primary-key operation would silently collapse N
        # candidates to one — the exact loss mission invariant 3.3 forbids.
        _reject_duplicates(
            tuple(
                f"{item.execution_id}|{item.realization_id}|{item.candidate_index}"
                for item in candidates
                if item.candidate_index is not None
            ),
            "(execution_id, realization_id, candidate_index)",
        )

        failed_executions = {
            item.execution_id
            for item in executions
            if item.execution_status is ExecutionStatus.TECHNICAL_FAILURE
        }
        for realization in realizations:
            if realization.execution_id in failed_executions:
                raise ValueError(
                    f"realization {realization.realization_id!r} belongs to "
                    f"technically failed execution {realization.execution_id!r}; a "
                    "run with no trustworthy answer must not contribute countable "
                    "physics objects (record partial output in provenance instead)"
                )
        for candidate in candidates:
            if candidate.execution_id in failed_executions:
                raise ValueError(
                    f"candidate {candidate.candidate_id!r} belongs to technically "
                    f"failed execution {candidate.execution_id!r}; counting it would "
                    "turn an untrusted run into a physics result"
                )
        for decision in decisions:
            if (
                decision.subject_ref in failed_executions
                and decision.evaluation_status is DecisionEvaluationStatus.EVALUATED
            ):
                raise ValueError(
                    f"decision {decision.decision_id!r} reports an evaluated outcome "
                    f"for technically failed execution {decision.subject_ref!r}; a "
                    "crashed run is censored, never evaluated"
                )

        object.__setattr__(self, "executions", executions)
        object.__setattr__(self, "realizations", realizations)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "decisions", decisions)

    @property
    def referenceable_ids(self) -> frozenset:
        """Every in-bundle id a ``subject_ref`` may legitimately point at.

        Excludes subject ids, which live in the request rather than the
        bundle; :func:`verify_evaluation_bundle` adds them.
        """
        return frozenset(
            tuple(item.execution_id for item in self.executions)
            + tuple(item.realization_id for item in self.realizations)
            + tuple(item.candidate_id for item in self.candidates)
        )


def verify_evaluation_bundle(
    bundle: EvaluationBundle, request: EvaluationRequest
) -> None:
    """Raise ``ValueError`` unless ``bundle`` is a valid answer to ``request``.

    This is the request-relative half of the boundary contract, and the check
    a substituted backend must also pass — it is written against the canonical
    types only, so it constrains a future FairShip adapter exactly as much as
    it constrains the fake backend:

    - the bundle answers *this* request;
    - every execution belongs to a subject the request declared (no invented
      subjects, and therefore no invented source states);
    - every execution carries the request's ``fs_sim_configuration_id``, so a
      backend cannot quietly relabel the configuration and have results pooled
      across incompatible settings downstream (``COMPAT-01``);
    - no subject id collides with a record id, and no observation id is reused
      between the request and the bundle, so an untyped reference can never
      resolve to two different kinds of thing — in particular an observation
      cannot be ambiguous between a source state and one execution *of* that
      state, which would make its statistical unit undefined;
    - every observation and stage decision attaches to something that exists —
      a declared subject or a record in this bundle — and every piece of
      evidence a decision cites resolves to a real observation, so the audit
      trail from a conclusion back to its evidence is never broken;
    - no subject receives more executions than were asked for, which would mean
      the backend invented repetitions the request never authorized.

    It deliberately does **not** require that every subject produced the full
    ``replications_per_subject`` count, or any execution at all. A shortfall is
    legitimate — a scheduler may drop work — but it is *not* self-describing:
    a backend that omits crashed runs instead of reporting them as
    ``TECHNICAL_FAILURE`` looks identical to one that was never scheduled, and
    the difference matters because only the first is censoring. Callers that
    care must compare against the request; :func:`execution_shortfall` does
    exactly that and is why the request must be retained alongside the bundle.
    """
    if not isinstance(bundle, EvaluationBundle):
        raise TypeError("bundle must be an EvaluationBundle")
    if not isinstance(request, EvaluationRequest):
        raise TypeError("request must be an EvaluationRequest")

    if bundle.request_id != request.request_id:
        raise ValueError(
            f"bundle answers request {bundle.request_id!r}, not {request.request_id!r}"
        )

    subject_ids = set(request.subject_ids)
    collisions = subject_ids & bundle.referenceable_ids
    if collisions:
        raise ValueError(
            "subject ids must not collide with record ids; an untyped reference "
            f"to {sorted(collisions)} would be ambiguous between a source state "
            "and a record about it"
        )
    request_observation_ids = {
        observation.observation_id for observation in request.subject_observations
    }
    bundle_observation_ids = {
        observation.observation_id for observation in bundle.observations
    }
    reused = request_observation_ids & bundle_observation_ids
    if reused:
        raise ValueError(
            f"observation ids {sorted(reused)} are declared both on the request and "
            "in the bundle; an id-keyed merge would keep one of them arbitrarily"
        )

    executions_per_subject: dict = {}
    for execution in bundle.executions:
        executions_per_subject[execution.subject_id] = (
            executions_per_subject.get(execution.subject_id, 0) + 1
        )
    for subject_id, count in sorted(executions_per_subject.items()):
        if count > request.replications_per_subject:
            raise ValueError(
                f"subject {subject_id!r} received {count} executions but the request "
                f"authorized {request.replications_per_subject}; the backend invented "
                "repetitions of a source state"
            )

    for execution in bundle.executions:
        if execution.subject_id not in subject_ids:
            raise ValueError(
                f"execution {execution.execution_id!r} references subject "
                f"{execution.subject_id!r}, which this request did not declare"
            )
        if execution.fs_sim_configuration_id != request.fs_sim_configuration_id:
            raise ValueError(
                f"execution {execution.execution_id!r} reports configuration "
                f"{execution.fs_sim_configuration_id!r} but the request specified "
                f"{request.fs_sim_configuration_id!r}"
            )

    attachable = bundle.referenceable_ids | subject_ids
    for observation in bundle.observations:
        if observation.subject_ref not in attachable:
            raise ValueError(
                f"observation {observation.observation_id!r} attaches to unknown "
                f"reference {observation.subject_ref!r}"
            )
    resolvable_evidence = request_observation_ids | bundle_observation_ids
    for decision in bundle.decisions:
        if decision.subject_ref not in attachable:
            raise ValueError(
                f"decision {decision.decision_id!r} attaches to unknown reference "
                f"{decision.subject_ref!r}"
            )
        for reference in decision.evidence_references:
            if reference not in resolvable_evidence:
                raise ValueError(
                    f"decision {decision.decision_id!r} cites evidence "
                    f"{reference!r}, which is not an observation in this bundle or "
                    "request; a conclusion whose evidence cannot be resolved is "
                    "not auditable"
                )


def execution_shortfall(
    bundle: EvaluationBundle, request: EvaluationRequest
) -> Mapping[str, int]:
    """Executions asked for but not reported, per subject (omitting zeros).

    A non-empty result is not automatically an error, but it is always a
    question: the missing runs are either genuinely never scheduled or crashes
    the backend failed to report, and only the second is censoring that belongs
    in a denominator discussion.
    """
    counts: dict = {subject_id: 0 for subject_id in request.subject_ids}
    for execution in bundle.executions:
        if execution.subject_id in counts:
            counts[execution.subject_id] += 1
    return {
        subject_id: request.replications_per_subject - count
        for subject_id, count in sorted(counts.items())
        if count < request.replications_per_subject
    }


def assert_requests_compatible(requests: Sequence[EvaluationRequest]) -> None:
    """Refuse a set of requests whose configuration labels hide a real difference.

    Two requests carrying the same ``fs_sim_configuration_id`` must agree on
    everything that could change the estimand. ``options`` is the channel the
    configuration id does not cover by construction, so its digest is compared
    here: same label plus different options means the label is not, in fact, a
    complete configuration identity (``CONF-01``, ``COMPAT-01``).
    """
    digests: dict = {}
    for request in requests:
        if not isinstance(request, EvaluationRequest):
            raise TypeError("requests must contain EvaluationRequest objects")
        known = digests.setdefault(
            request.fs_sim_configuration_id, (request.options_digest, request.request_id)
        )
        if known[0] != request.options_digest:
            raise ValueError(
                f"requests {known[1]!r} and {request.request_id!r} share "
                f"fs_sim_configuration_id {request.fs_sim_configuration_id!r} but "
                "their options differ; results under them must not be pooled until "
                "the configuration id covers the difference"
            )


def evaluate_verified(
    backend: "EvaluationBackend", request: EvaluationRequest
) -> EvaluationBundle:
    """Run ``backend`` and verify its answer before anyone can consume it.

    The checked path should be the easy path. Calling ``backend.evaluate``
    directly is still legitimate — a bundle read back from disk has no request
    to verify against — but new code has no reason to skip the check.
    """
    bundle = backend.evaluate(request)
    verify_evaluation_bundle(bundle, request)
    return bundle


@runtime_checkable
class EvaluationBackend(Protocol):
    """The one interface the scientific core knows about.

    Implementations live behind this boundary and may be anything: a
    deterministic fake used in tests, a minimal stub, or a future adapter that
    shells out to real FairShip. Selecting one is dependency injection — never
    a branch in scientific code on which backend is in use.

    Requirements on every implementation:

    - ``evaluate`` is deterministic given ``request`` alone. All randomness
      derives from ``request.seed``; never from wall-clock time, process id,
      dict iteration order, or ambient environment state.
    - The returned bundle passes :func:`verify_evaluation_bundle` against the
      request.
    - A run that could not produce a trustworthy answer is reported as an
      ``FSSimExecution`` with ``ExecutionStatus.TECHNICAL_FAILURE``. It is
      never reported as a negative physics outcome, and never as a
      ``StageDecision`` carrying ``False`` (mission invariant 3.2).
    - Candidate multiplicity is reported by emitting that many
      ``ReconstructedCandidate`` records — zero, one, or many. It is never
      collapsed to a boolean (mission invariant 3.3).
    - No CERN/EOS absolute paths and no environment assumptions are hardcoded;
      everything site-specific arrives through ``request.options``.

    ``is_physical`` states whether this backend's output may be used as
    physics evidence. A fake or stub backend sets it ``False``, so downstream
    code and reports can assert on it structurally instead of relying on a
    naming convention to keep test output out of scientific claims.
    """

    name: str
    is_physical: bool

    def evaluate(self, request: EvaluationRequest) -> EvaluationBundle:
        """Evaluate every subject in ``request`` and return canonical records."""
        ...
