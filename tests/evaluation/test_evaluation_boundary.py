"""Contract tests for the canonical evaluation boundary (Unit A2).

These tests constrain the *types*, not any backend. They are the guarantees a
future real FairShip adapter inherits for free by returning canonical records.
"""

from __future__ import annotations

import dataclasses

import pytest

from ship_muon_bg.entities import (
    DecisionEvaluationStatus,
    ExecutionStatus,
    TagSubject,
)
from ship_muon_bg.simulation.evaluation import (
    EvaluationBackend,
    EvaluationBundle,
    EvaluationRequest,
    assert_requests_compatible,
    evaluate_verified,
    execution_shortfall,
    verify_evaluation_bundle,
)

from tests.evaluation import fixtures as fx


# --------------------------------------------------------------------------
# EvaluationRequest
# --------------------------------------------------------------------------


def test_request_rejects_the_same_subject_listed_twice():
    """Mission invariant 3.1: one post-shield state is one source state.

    Listing a subject twice is the exact mistake that turns a conditional
    repetition into two independent nominal draws. This closes the accidental
    case only: the check is on the identifier, so minting one physical state
    under two ids still gets through, and keeping ids one-per-state remains the
    request builder's obligation (recorded as OPEN, see the module docstring).
    """
    duplicated = fx.subject("s1")
    with pytest.raises(ValueError, match="duplicate subject_id"):
        EvaluationRequest(
            request_id="req-1",
            subjects=(duplicated, duplicated),
            fs_sim_configuration_id=fx.CONFIG_A,
            seed=1,
        )


def test_repetition_is_expressed_as_replications_not_duplicate_subjects():
    req = fx.request(subject_ids=("s1",), replications_per_subject=4)
    assert req.replications_per_subject == 4
    assert req.subject_ids == ("s1",)


@pytest.mark.parametrize("bad_seed", [True, False, 1.0, "1", None])
def test_request_rejects_non_int_seeds(bad_seed):
    """``True`` is an ``int`` in Python; a bool seed must still be rejected."""
    with pytest.raises(TypeError):
        fx.request(seed=bad_seed)


def test_request_rejects_negative_seed_and_zero_replications():
    with pytest.raises(ValueError):
        fx.request(seed=-1)
    with pytest.raises(ValueError):
        fx.request(replications_per_subject=0)


def test_request_rejects_observations_about_undeclared_subjects():
    with pytest.raises(ValueError, match="unknown subject_ref"):
        fx.request(
            subject_ids=("s1",),
            subject_observations=(fx.scalar_observation("o1", "s_other", 5.0),),
        )


def test_request_carries_no_kinematics_or_weight_fields():
    """WEIGHT-01 / mission §5: physical content stays in observations.

    The request must not grow its own ``weight``/``momentum``/``px`` fields,
    which is how a physical source weight ``w_i`` and a utility multiplier
    ``h(U_i)`` end up multiplied into one opaque number.
    """
    forbidden_substrings = ("weight", "momentum", "energy", "kinemat")
    offenders = [
        name
        for name in EvaluationRequest.__dataclass_fields__
        if any(token in name.lower() for token in forbidden_substrings)
        or name.lower() in {"px", "py", "pz", "e", "p", "w"}
    ]
    assert offenders == []


def test_request_is_immutable():
    req = fx.request()
    with pytest.raises(dataclasses.FrozenInstanceError):
        req.seed = 99


# --------------------------------------------------------------------------
# EvaluationBundle: intrinsic integrity
# --------------------------------------------------------------------------


def test_empty_bundle_is_valid():
    """Zero executions is a legitimate answer and must stay representable."""
    empty = fx.bundle()
    assert empty.executions == ()
    assert empty.referenceable_ids == frozenset()


def test_bundle_rejects_duplicate_identifiers_within_a_kind():
    with pytest.raises(ValueError, match="duplicate execution_id"):
        fx.bundle(executions=(fx.execution("e1", "s1"), fx.execution("e1", "s1")))


def test_bundle_rejects_identifier_reused_across_record_kinds():
    """An id shared by an execution and a candidate makes every untyped
    ``subject_ref`` lookup ambiguous, which is how a candidate-level decision
    silently becomes an execution-level one."""
    with pytest.raises(ValueError, match="reused across record kinds"):
        fx.bundle(
            executions=(fx.execution("x1", "s1"),),
            candidates=(fx.candidate("x1", "x1"),),
        )


def test_bundle_rejects_orphaned_realization_and_candidate():
    with pytest.raises(ValueError, match="unknown execution_id"):
        fx.bundle(realizations=(fx.realization("r1", "e_missing"),))
    with pytest.raises(ValueError, match="unknown execution_id"):
        fx.bundle(candidates=(fx.candidate("c1", "e_missing"),))


def test_bundle_rejects_candidate_whose_realization_belongs_to_another_execution():
    with pytest.raises(ValueError, match="belongs to execution"):
        fx.bundle(
            executions=(fx.execution("e1", "s1"), fx.execution("e2", "s1")),
            realizations=(fx.realization("r1", "e1"),),
            candidates=(fx.candidate("c1", "e2", realization_id="r1"),),
        )


def test_bundle_preserves_zero_one_and_many_candidates():
    """Mission invariant 3.3: multiplicity survives as record count.

    ``N`` is read off the records, and ``Y = 1{N>=1}`` is derived from ``N`` —
    never the other way round. The failed execution here is the important part:
    a naive count would score it ``N = 0`` and therefore ``Y = 0``, which is
    exactly the corruption invariant 3.2 forbids. The correct derivation
    partitions on ``execution_status`` first and never assigns it a ``Y`` at
    all.
    """
    b = fx.bundle(
        executions=(
            fx.execution("e0", "s1"),
            fx.execution("e1", "s1"),
            fx.execution("e2", "s1"),
            fx.execution("e3", "s1", status=ExecutionStatus.TECHNICAL_FAILURE,
                         failure_reason="fixture: injected crash"),
        ),
        candidates=(
            fx.candidate("c1", "e1", candidate_index=0),
            fx.candidate("c2", "e2", candidate_index=0),
            fx.candidate("c3", "e2", candidate_index=1),
            fx.candidate("c4", "e2", candidate_index=2),
        ),
    )
    counts = {}
    for cand in b.candidates:
        counts[cand.execution_id] = counts.get(cand.execution_id, 0) + 1

    evaluable = [
        e for e in b.executions if e.execution_status is ExecutionStatus.SUCCEEDED
    ]
    censored = [
        e
        for e in b.executions
        if e.execution_status is ExecutionStatus.TECHNICAL_FAILURE
    ]
    assert [e.execution_id for e in censored] == ["e3"]
    assert [counts.get(e.execution_id, 0) for e in evaluable] == [0, 1, 3]
    assert [int(counts.get(e.execution_id, 0) >= 1) for e in evaluable] == [0, 1, 1]
    # The crashed run contributes to neither numerator nor denominator.
    assert len(evaluable) == 3


def test_bundle_holds_no_derived_aggregate_fields():
    """The bundle is a record set. Counts and rates belong to aggregation, so
    a different estimator can be swapped in without touching backends."""
    forbidden_substrings = ("eta", "count", "rate", "summary", "mean", "total", "frac")
    offenders = [
        name
        for name in EvaluationBundle.__dataclass_fields__
        if any(token in name.lower() for token in forbidden_substrings)
    ]
    assert offenders == []


# --------------------------------------------------------------------------
# verify_evaluation_bundle: request-relative conformance
# --------------------------------------------------------------------------


def test_verify_accepts_a_well_formed_answer():
    req = fx.request(subject_ids=("s1", "s2"), replications_per_subject=2)
    b = fx.bundle(
        executions=(
            fx.execution("e1", "s1"),
            fx.execution("e2", "s1"),
            fx.execution("e3", "s2"),
        ),
        realizations=(fx.realization("r1", "e1"),),
        candidates=(fx.candidate("c1", "e1", realization_id="r1", candidate_index=0),),
        observations=(fx.scalar_observation("o1", "c1", 3.0),),
    )
    verify_evaluation_bundle(b, req)


def test_verify_rejects_a_bundle_answering_a_different_request():
    with pytest.raises(ValueError, match="answers request"):
        verify_evaluation_bundle(fx.bundle(request_id="other"), fx.request())


def test_verify_rejects_an_invented_subject():
    """A backend must not manufacture source states the core never declared."""
    req = fx.request(subject_ids=("s1",))
    b = fx.bundle(executions=(fx.execution("e1", "s_invented"),))
    with pytest.raises(ValueError, match="did not declare"):
        verify_evaluation_bundle(b, req)


def test_verify_rejects_a_relabelled_configuration():
    """COMPAT-01: a backend cannot silently answer under another config and
    have those results pooled with the requested one."""
    req = fx.request(fs_sim_configuration_id=fx.CONFIG_A)
    b = fx.bundle(
        executions=(fx.execution("e1", "s1", fs_sim_configuration_id=fx.CONFIG_B),)
    )
    with pytest.raises(ValueError, match="reports configuration"):
        verify_evaluation_bundle(b, req)


def test_verify_rejects_orphaned_evidence_and_decisions():
    req = fx.request()
    b = fx.bundle(observations=(fx.scalar_observation("o1", "nowhere", 1.0),))
    with pytest.raises(ValueError, match="attaches to unknown reference"):
        verify_evaluation_bundle(b, req)


def test_verify_allows_a_subject_with_no_executions():
    """Not-evaluated is a real, reportable outcome — not an error, and not a
    negative physics result (mission invariant 3.2)."""
    req = fx.request(subject_ids=("s1", "s2"))
    b = fx.bundle(executions=(fx.execution("e1", "s1"),))
    verify_evaluation_bundle(b, req)


def test_verify_allows_evidence_attached_to_a_declared_subject():
    req = fx.request(subject_ids=("s1",))
    b = fx.bundle(observations=(fx.scalar_observation("o1", "s1", 2.0),))
    verify_evaluation_bundle(b, req)


# --------------------------------------------------------------------------
# EvaluationBackend protocol
# --------------------------------------------------------------------------


class _MinimalConformingBackend:
    name = "minimal"
    is_physical = False

    def evaluate(self, request: EvaluationRequest) -> EvaluationBundle:
        return EvaluationBundle(
            request_id=request.request_id,
            backend_name=self.name,
            is_physical=self.is_physical,
        )


def test_protocol_is_structural_and_requires_no_inheritance():
    backend = _MinimalConformingBackend()
    assert isinstance(backend, EvaluationBackend)
    req = fx.request()
    verify_evaluation_bundle(backend.evaluate(req), req)


def test_protocol_rejects_an_object_without_evaluate():
    class NotABackend:
        name = "nope"
        is_physical = False

    assert not isinstance(NotABackend(), EvaluationBackend)


def test_a_failed_execution_may_not_carry_countable_physics_objects():
    """Mission invariant 3.2, enforced rather than merely intended.

    A truncated ROOT read from a timed-out run must not deposit candidates or
    realizations into the record set, where a later ``groupby`` would tally
    them as real. The failure is recorded on the health axis; any partial
    output belongs in provenance.
    """
    failed = fx.execution(
        "e1",
        "s1",
        status=ExecutionStatus.TECHNICAL_FAILURE,
        failure_reason="fixture: injected transport crash",
    )
    with pytest.raises(ValueError, match="technically failed execution"):
        fx.bundle(executions=(failed,), candidates=(fx.candidate("c1", "e1"),))
    with pytest.raises(ValueError, match="technically failed execution"):
        fx.bundle(executions=(failed,), realizations=(fx.realization("r1", "e1"),))


def test_a_failed_execution_may_not_carry_an_evaluated_decision():
    failed = fx.execution(
        "e1", "s1", status=ExecutionStatus.TECHNICAL_FAILURE, failure_reason="crash"
    )
    censoring = fx.decision("e1", DecisionEvaluationStatus.TECHNICALLY_UNAVAILABLE, None)
    evaluated = fx.decision("e1", DecisionEvaluationStatus.EVALUATED, False)
    # Censoring on a crashed run is exactly right and stays representable...
    fx.bundle(executions=(failed,), decisions=(censoring,))
    # ...but an evaluated outcome for a run that produced nothing trustworthy
    # is the corruption itself, whichever value it carries.
    with pytest.raises(ValueError, match="crashed run is censored, never evaluated"):
        fx.bundle(executions=(failed,), decisions=(evaluated,))


# --------------------------------------------------------------------------
# Guards added after independent review
# --------------------------------------------------------------------------


def test_a_backend_may_not_return_more_executions_than_were_authorized():
    """Invented repetitions of a source state would inflate every denominator."""
    req = fx.request(subject_ids=("s1",), replications_per_subject=2)
    b = fx.bundle(
        executions=(
            fx.execution("e1", "s1"),
            fx.execution("e2", "s1"),
            fx.execution("e3", "s1"),
        )
    )
    with pytest.raises(ValueError, match="invented repetitions"):
        verify_evaluation_bundle(b, req)


def test_a_shortfall_is_allowed_but_is_reported_rather_than_hidden():
    """A backend that omits crashed runs looks identical to one that was never
    scheduled. Both are legitimate; only the first is censoring, so the
    difference has to be visible rather than inferred."""
    req = fx.request(subject_ids=("s1", "s2"), replications_per_subject=3)
    b = fx.bundle(executions=(fx.execution("e1", "s1"),))
    verify_evaluation_bundle(b, req)
    assert execution_shortfall(b, req) == {"s1": 2, "s2": 3}


def test_no_shortfall_reported_when_the_request_was_fully_answered():
    req = fx.request(subject_ids=("s1",), replications_per_subject=2)
    b = fx.bundle(executions=(fx.execution("e1", "s1"), fx.execution("e2", "s1")))
    assert execution_shortfall(b, req) == {}


def test_a_subject_id_may_not_collide_with_a_record_id():
    """Otherwise an observation on that id is ambiguous between the source
    state and one execution of it — its statistical unit becomes undefined."""
    req = fx.request(subject_ids=("e1",))
    b = fx.bundle(
        executions=(fx.execution("e1", "e1"),),
        observations=(fx.scalar_observation("o1", "e1", 42.0),),
    )
    with pytest.raises(ValueError, match="must not collide with record ids"):
        verify_evaluation_bundle(b, req)


def test_observation_and_decision_ids_join_the_cross_kind_uniqueness_pool():
    with pytest.raises(ValueError, match="reused across record kinds"):
        fx.bundle(
            executions=(fx.execution("e1", "s1"),),
            observations=(fx.scalar_observation("e1", "e1", 1.0),),
        )


def test_request_and_bundle_may_not_reuse_one_observation_id():
    req = fx.request(
        subject_ids=("s1",),
        subject_observations=(fx.scalar_observation("o1", "s1", 73.5),),
    )
    b = fx.bundle(
        executions=(fx.execution("e1", "s1"),),
        observations=(fx.scalar_observation("o1", "e1", -999.0),),
    )
    with pytest.raises(ValueError, match="declared both on the request and"):
        verify_evaluation_bundle(b, req)


def test_a_decision_may_not_cite_evidence_that_does_not_exist():
    """A conclusion whose evidence cannot be resolved is an unfalsifiable
    claim; DECISION-01 makes evidence_references the audit trail."""
    req = fx.request(subject_ids=("s1",))
    b = fx.bundle(
        executions=(fx.execution("e1", "s1"),),
        decisions=(
            fx.decision(
                "e1",
                DecisionEvaluationStatus.EVALUATED,
                True,
                evidence_references=("o_does_not_exist",),
            ),
        ),
    )
    with pytest.raises(ValueError, match="not an observation in this bundle"):
        verify_evaluation_bundle(b, req)


def test_a_decision_may_cite_evidence_declared_on_the_request():
    req = fx.request(
        subject_ids=("s1",),
        subject_observations=(fx.scalar_observation("o1", "s1", 4.0),),
    )
    b = fx.bundle(
        executions=(fx.execution("e1", "s1"),),
        decisions=(
            fx.decision(
                "e1",
                DecisionEvaluationStatus.EVALUATED,
                True,
                evidence_references=("o1",),
            ),
        ),
    )
    verify_evaluation_bundle(b, req)


def test_candidate_index_may_not_repeat_within_one_parent():
    """The first key in the schema on which an ordinary groupby would collapse
    N candidates to one."""
    with pytest.raises(ValueError, match="candidate_index"):
        fx.bundle(
            executions=(fx.execution("e1", "s1"),),
            realizations=(fx.realization("r1", "e1"),),
            candidates=(
                fx.candidate("c1", "e1", realization_id="r1", candidate_index=0),
                fx.candidate("c2", "e1", realization_id="r1", candidate_index=0),
            ),
        )


def test_candidate_index_zero_repeats_across_sibling_realizations():
    """Index is scoped to (execution, realization), so this is two candidates,
    not a contradiction."""
    b = fx.bundle(
        executions=(fx.execution("e1", "s1"),),
        realizations=(fx.realization("r1", "e1"), fx.realization("r2", "e1")),
        candidates=(
            fx.candidate("c1", "e1", realization_id="r1", candidate_index=0),
            fx.candidate("c2", "e1", realization_id="r2", candidate_index=0),
        ),
    )
    assert len(b.candidates) == 2


def test_the_bundle_carries_the_physicality_declaration_into_the_records():
    """The backend object does not travel with persisted records; this does."""
    b = fx.bundle(is_physical=False)
    assert b.is_physical is False
    with pytest.raises(TypeError, match="is_physical must be a bool"):
        fx.bundle(is_physical="false")


def test_options_drift_under_one_configuration_id_is_detectable():
    """COMPAT-01: options is the channel the configuration id does not cover,
    so two runs can differ in geometry while sharing one label."""
    first = fx.request(request_id="run-a", fs_sim_configuration_id=fx.CONFIG_A)
    second = EvaluationRequest(
        request_id="run-b",
        subjects=first.subjects,
        fs_sim_configuration_id=fx.CONFIG_A,
        seed=first.seed,
        options={"geometry_file": "geo_2024_new_magnet.root"},
    )
    assert first.options_digest != second.options_digest
    assert_requests_compatible([first, first])
    with pytest.raises(ValueError, match="their options differ"):
        assert_requests_compatible([first, second])


def test_different_configuration_ids_with_different_options_are_fine():
    first = fx.request(request_id="run-a", fs_sim_configuration_id=fx.CONFIG_A)
    second = EvaluationRequest(
        request_id="run-b",
        subjects=first.subjects,
        fs_sim_configuration_id=fx.CONFIG_B,
        seed=first.seed,
        options={"geometry_file": "other.root"},
    )
    assert_requests_compatible([first, second])


def test_state_definition_ids_are_exposed_for_compatibility_checks():
    req = fx.request(subject_ids=("s1", "s2"))
    assert req.state_definition_ids == (fx.STATE_DEFINITION,)


def test_evaluate_verified_makes_the_checked_path_the_easy_path():
    backend = _MinimalConformingBackend()
    req = fx.request()
    assert evaluate_verified(backend, req).request_id == req.request_id

    class LiarBackend:
        name = "liar"
        is_physical = False

        def evaluate(self, request):
            return EvaluationBundle(
                request_id=request.request_id,
                backend_name=self.name,
                is_physical=False,
                executions=(fx.execution("e1", "not_a_declared_subject"),),
            )

    with pytest.raises(ValueError, match="did not declare"):
        evaluate_verified(LiarBackend(), req)


def test_execution_provenance_must_be_a_mapping():
    from ship_muon_bg.entities import ExecutionStatus as _Status
    from ship_muon_bg.entities import FSSimExecution

    with pytest.raises(TypeError, match="provenance must be a mapping"):
        FSSimExecution(
            execution_id="e1",
            subject_id="s1",
            fs_sim_configuration_id="cfg",
            execution_status=_Status.SUCCEEDED,
            provenance="not a mapping",
        )
