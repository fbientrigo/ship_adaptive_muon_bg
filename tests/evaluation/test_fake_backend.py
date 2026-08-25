"""Unit B contract tests for the deterministic fake backends.

The fake backends are test infrastructure. These tests check that they are
*honest* infrastructure: deterministic, lineage-preserving, and incapable of
turning a technical failure into a physics negative. Where a claim must be
hand-checkable, the test scripts an explicit ``FakeRunPlan`` and states the
expected record counts in advance.
"""

from __future__ import annotations

from collections import Counter

import pytest

from ship_muon_bg.entities import (
    DecisionEvaluationStatus,
    ExecutionStatus,
    ObservationEvaluationStatus,
)
from ship_muon_bg.simulation.evaluation import (
    EvaluationBackend,
    verify_evaluation_bundle,
)
from ship_muon_bg.simulation.fake_fairship import (
    FakeCandidatePlan,
    FakeFairShipBackend,
    FakeRunPlan,
    FakeStageSpec,
)
from ship_muon_bg.simulation.stub_backend import MinimalStubBackend
from ship_muon_bg.tagging import StageEvaluator

from tests.evaluation import fixtures as fx


def _backend(**kwargs):
    definition, spec = fx.stage_spec()
    return definition, FakeFairShipBackend(stages=(spec,), **kwargs)


def _candidates_per_execution(bundle):
    counts = Counter()
    for candidate in bundle.candidates:
        counts[candidate.execution_id] += 1
    return {execution.execution_id: counts[execution.execution_id] for execution in bundle.executions}


# --------------------------------------------------------------------------
# Protocol conformance and self-declaration
# --------------------------------------------------------------------------


def test_both_backends_satisfy_the_canonical_protocol():
    _definition, fake = _backend()
    stub = MinimalStubBackend(
        stage_definition_ids=("stage-1",), observation_definition_ids=(fx.SCORE_A,)
    )
    assert isinstance(fake, EvaluationBackend)
    assert isinstance(stub, EvaluationBackend)


def test_neither_fake_backend_claims_to_be_physical():
    """Machine-checkable, so a report generator can refuse to cite fake output."""
    _definition, fake = _backend()
    stub = MinimalStubBackend(
        stage_definition_ids=("stage-1",), observation_definition_ids=(fx.SCORE_A,)
    )
    assert fake.is_physical is False
    assert stub.is_physical is False


def test_fake_backend_names_no_physics_interaction():
    """``muon_dis`` is a real physics process; the fixture must not claim it."""
    from ship_muon_bg.simulation import fake_fairship

    assert "dis" not in fake_fairship.FIXTURE_INTERACTION_TYPE.lower()


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def test_repeated_evaluation_is_byte_for_byte_identical():
    _definition, backend = _backend(technical_failure_modulus=5)
    request = fx.request(
        subject_ids=tuple(f"s{i}" for i in range(5)), replications_per_subject=4
    )
    assert backend.evaluate(request) == backend.evaluate(request)


def test_a_fresh_backend_instance_reproduces_the_same_bundle():
    """Determinism must come from the request, not from instance state."""
    _d1, first = _backend(technical_failure_modulus=5)
    _d2, second = _backend(technical_failure_modulus=5)
    request = fx.request(subject_ids=("s1", "s2", "s3"), replications_per_subject=3)
    assert first.evaluate(request) == second.evaluate(request)


def test_outcomes_do_not_depend_on_the_request_label():
    """Relabelling a request must change identifiers only, never outcomes.

    A physics answer that moves when you rename the job is not reproducible.
    """
    _definition, backend = _backend(technical_failure_modulus=5)
    subjects = tuple(f"s{i}" for i in range(4))
    first = backend.evaluate(
        fx.request(request_id="run-a", subject_ids=subjects, replications_per_subject=3)
    )
    second = backend.evaluate(
        fx.request(request_id="run-b", subject_ids=subjects, replications_per_subject=3)
    )
    assert len(first.executions) == len(second.executions)
    assert len(first.candidates) == len(second.candidates)
    assert [e.execution_status for e in first.executions] == [
        e.execution_status for e in second.executions
    ]
    assert [d.decision for d in first.decisions] == [d.decision for d in second.decisions]
    # ...but the identifiers genuinely differ, so the two runs stay distinguishable.
    assert first.executions[0].execution_id != second.executions[0].execution_id


def test_a_different_seed_changes_outcomes():
    """Guards against a fake that ignores the seed and only looks deterministic."""
    _definition, backend = _backend(technical_failure_modulus=5)
    subjects = tuple(f"s{i}" for i in range(6))
    first = backend.evaluate(fx.request(subject_ids=subjects, seed=1, replications_per_subject=3))
    second = backend.evaluate(fx.request(subject_ids=subjects, seed=2, replications_per_subject=3))
    assert first.candidates != second.candidates


def test_a_different_configuration_changes_outcomes():
    """COMPAT-01: a configuration change changes the estimand, and the fake
    must reflect that rather than return identical numbers under a new label."""
    _definition, backend = _backend(technical_failure_modulus=5)
    subjects = tuple(f"s{i}" for i in range(6))
    first = backend.evaluate(
        fx.request(subject_ids=subjects, fs_sim_configuration_id=fx.CONFIG_A)
    )
    second = backend.evaluate(
        fx.request(subject_ids=subjects, fs_sim_configuration_id=fx.CONFIG_B)
    )
    assert [c.candidate_id for c in first.candidates] != [
        c.candidate_id for c in second.candidates
    ] or [d.decision for d in first.decisions] != [d.decision for d in second.decisions]


# --------------------------------------------------------------------------
# Lineage and cardinality, hand-computed from explicit plans
# --------------------------------------------------------------------------


def test_scripted_plan_produces_exactly_the_stated_records():
    """Hand-computable: 1 subject, 3 replications, scripted structure.

    replication 0: 2 realizations yielding 1 and 3 candidates -> 4 candidates
    replication 1: no realizations, 0 candidates      -> clean physics zero
    replication 2: technical failure                  -> no output at all
    """
    definition, _spec = fx.stage_spec()
    backend = FakeFairShipBackend(
        stages=(
            FakeStageSpec(
                stage_definition_id=definition.stage_definition_id,
                observation_definition_id=fx.SCORE_A,
                threshold=fx.THRESHOLD,
            ),
        ),
        plans={
            ("s1", 0): FakeRunPlan(realization_candidate_counts=(1, 3)),
            ("s1", 1): FakeRunPlan(realization_candidate_counts=()),
            ("s1", 2): FakeRunPlan(technical_failure=True, realization_candidate_counts=()),
        },
    )
    request = fx.request(subject_ids=("s1",), replications_per_subject=3)
    bundle = backend.evaluate(request)
    verify_evaluation_bundle(bundle, request)

    assert len(bundle.executions) == 3
    assert len(bundle.realizations) == 2
    assert len(bundle.candidates) == 4
    assert len(bundle.observations) == 4
    assert len(bundle.decisions) == 4

    per_execution = _candidates_per_execution(bundle)
    assert sorted(per_execution.values()) == [0, 0, 4]

    statuses = [e.execution_status for e in bundle.executions]
    assert statuses.count(ExecutionStatus.TECHNICAL_FAILURE) == 1
    assert statuses.count(ExecutionStatus.SUCCEEDED) == 2


def test_a_subject_can_produce_no_execution_at_all():
    """Distinct from a failed run, and the aggregation layer must see both."""
    definition, spec = fx.stage_spec()
    backend = FakeFairShipBackend(
        stages=(spec,), plans={("s1", 0): FakeRunPlan(emit_execution=False)}
    )
    request = fx.request(subject_ids=("s1", "s2"))
    bundle = backend.evaluate(request)
    verify_evaluation_bundle(bundle, request)
    assert [e.subject_id for e in bundle.executions] == ["s2"]


def test_candidate_multiplicity_survives_as_records_not_a_flag():
    """Mission invariant 3.3: N and Y = 1{N>=1} both derivable."""
    definition, spec = fx.stage_spec()
    backend = FakeFairShipBackend(
        stages=(spec,),
        plans={
            ("s1", 0): FakeRunPlan(realization_candidate_counts=(0,)),
            ("s1", 1): FakeRunPlan(realization_candidate_counts=(1,)),
            ("s1", 2): FakeRunPlan(realization_candidate_counts=(5,)),
        },
    )
    request = fx.request(subject_ids=("s1",), replications_per_subject=3)
    bundle = backend.evaluate(request)
    counts = sorted(_candidates_per_execution(bundle).values())
    assert counts == [0, 1, 5]
    assert [int(n >= 1) for n in counts] == [0, 1, 1]


def test_unattached_candidates_are_supported():
    """EXEC-04: a candidate need not be scoped to one interaction realization."""
    definition, spec = fx.stage_spec()
    backend = FakeFairShipBackend(
        stages=(spec,),
        plans={
            ("s1", 0): FakeRunPlan(
                realization_candidate_counts=(1,), unattached_candidate_count=2
            )
        },
    )
    request = fx.request(subject_ids=("s1",))
    bundle = backend.evaluate(request)
    verify_evaluation_bundle(bundle, request)
    attached = [c for c in bundle.candidates if c.realization_id is not None]
    unattached = [c for c in bundle.candidates if c.realization_id is None]
    assert len(attached) == 1
    assert len(unattached) == 2


def test_every_record_traces_to_exactly_one_declared_subject():
    _definition, backend = _backend(technical_failure_modulus=5)
    request = fx.request(
        subject_ids=tuple(f"s{i}" for i in range(6)), replications_per_subject=3
    )
    bundle = backend.evaluate(request)
    verify_evaluation_bundle(bundle, request)

    owner = {e.execution_id: e.subject_id for e in bundle.executions}
    assert set(owner.values()) <= set(request.subject_ids)
    for realization in bundle.realizations:
        assert realization.execution_id in owner
    for candidate in bundle.candidates:
        assert candidate.execution_id in owner
    candidate_owner = {c.candidate_id: owner[c.execution_id] for c in bundle.candidates}
    for decision in bundle.decisions:
        assert decision.subject_ref in candidate_owner


def test_the_generated_bundle_always_verifies_against_its_request():
    _definition, backend = _backend(technical_failure_modulus=3)
    for seed in range(8):
        request = fx.request(
            subject_ids=tuple(f"s{i}" for i in range(4)),
            seed=seed,
            replications_per_subject=3,
        )
        verify_evaluation_bundle(backend.evaluate(request), request)


# --------------------------------------------------------------------------
# Failure and censoring semantics
# --------------------------------------------------------------------------


def test_a_technically_failed_run_emits_no_decision_of_any_kind():
    """Mission invariant 3.2. A crash has no physics outcome, not even False."""
    definition, spec = fx.stage_spec()
    backend = FakeFairShipBackend(
        stages=(spec,),
        plans={("s1", 0): FakeRunPlan(technical_failure=True, realization_candidate_counts=())},
    )
    request = fx.request(subject_ids=("s1",))
    bundle = backend.evaluate(request)
    assert bundle.candidates == ()
    assert bundle.decisions == ()
    assert bundle.observations == ()
    failed = bundle.executions[0]
    assert failed.execution_status is ExecutionStatus.TECHNICAL_FAILURE
    assert failed.failure_reason


def test_no_generated_decision_is_ever_false_without_being_evaluated():
    """The exact corruption to prevent: censoring rendered as a negative."""
    _definition, backend = _backend(technical_failure_modulus=3)
    request = fx.request(
        subject_ids=tuple(f"s{i}" for i in range(8)), replications_per_subject=4
    )
    bundle = backend.evaluate(request)
    for decision in bundle.decisions:
        if decision.evaluation_status is not DecisionEvaluationStatus.EVALUATED:
            assert decision.decision is None


def test_unavailable_evidence_censors_rather_than_rejects():
    definition, spec = fx.stage_spec()
    backend = FakeFairShipBackend(
        stages=(spec,),
        plans={
            ("s1", 0): FakeRunPlan(
                realization_candidate_counts=(1,),
                candidate_plans=(
                    FakeCandidatePlan(unavailable_observations=(fx.SCORE_A,)),
                ),
            )
        },
    )
    request = fx.request(subject_ids=("s1",))
    bundle = backend.evaluate(request)
    verify_evaluation_bundle(bundle, request)

    assert bundle.executions[0].execution_status is ExecutionStatus.SUCCEEDED
    assert len(bundle.candidates) == 1
    observation = bundle.observations[0]
    assert (
        observation.evaluation_status
        is ObservationEvaluationStatus.TECHNICALLY_UNAVAILABLE
    )
    assert observation.payload is None
    decision = bundle.decisions[0]
    assert decision.evaluation_status is DecisionEvaluationStatus.TECHNICALLY_UNAVAILABLE
    assert decision.decision is None


def test_absent_evidence_yields_not_evaluated_and_no_observation_record():
    definition, spec = fx.stage_spec()
    backend = FakeFairShipBackend(
        stages=(spec,),
        plans={
            ("s1", 0): FakeRunPlan(
                realization_candidate_counts=(1,),
                candidate_plans=(FakeCandidatePlan(missing_observations=(fx.SCORE_A,)),),
            )
        },
    )
    bundle = backend.evaluate(fx.request(subject_ids=("s1",)))
    assert bundle.observations == ()
    decision = bundle.decisions[0]
    assert decision.evaluation_status is DecisionEvaluationStatus.NOT_EVALUATED
    assert decision.decision is None


def test_a_failed_plan_cannot_also_script_output():
    with pytest.raises(ValueError, match="no trustworthy output"):
        FakeRunPlan(technical_failure=True, realization_candidate_counts=(2,))


def test_scripted_scores_give_exact_positives_and_negatives():
    """Hand-computable: 0.9 > 0.5 is True, 0.1 > 0.5 is False."""
    definition, spec = fx.stage_spec()
    backend = FakeFairShipBackend(
        stages=(spec,),
        plans={
            ("s1", 0): FakeRunPlan(
                realization_candidate_counts=(2,),
                candidate_plans=(
                    FakeCandidatePlan(scores={fx.SCORE_A: 0.9}),
                    FakeCandidatePlan(scores={fx.SCORE_A: 0.1}),
                ),
            )
        },
    )
    bundle = backend.evaluate(fx.request(subject_ids=("s1",)))
    assert [d.decision for d in bundle.decisions] == [True, False]


def test_threshold_is_strict_greater_than():
    definition, spec = fx.stage_spec()
    backend = FakeFairShipBackend(
        stages=(spec,),
        plans={
            ("s1", 0): FakeRunPlan(
                realization_candidate_counts=(1,),
                candidate_plans=(FakeCandidatePlan(scores={fx.SCORE_A: fx.THRESHOLD}),),
            )
        },
    )
    bundle = backend.evaluate(fx.request(subject_ids=("s1",)))
    assert bundle.decisions[0].decision is False


# --------------------------------------------------------------------------
# Agreement with the independent tagging evaluator
# --------------------------------------------------------------------------


def test_backend_decisions_match_the_independent_stage_evaluator_exactly():
    """The strongest available check that the fake is not quietly inventing
    its own semantics: the tagging layer, which shares no code with the
    backend, re-derives every decision from the emitted evidence and produces
    identical records — identity, status, value and evidence references."""
    definition, spec = fx.stage_spec()
    backend = FakeFairShipBackend(stages=(spec,), technical_failure_modulus=4)
    request = fx.request(
        subject_ids=tuple(f"s{i}" for i in range(6)), replications_per_subject=4
    )
    bundle = backend.evaluate(request)
    evaluator = StageEvaluator()

    assert bundle.decisions, "fixture must generate decisions to be meaningful"
    for decision in bundle.decisions:
        re_derived = evaluator.evaluate(
            definition, bundle.observations, subject_ref=decision.subject_ref
        )
        assert re_derived == decision


def test_multiple_stages_are_independent_and_carry_their_own_evidence():
    """Mission invariant 3.4: no logical implication between stages."""
    definition_a, spec_a = fx.stage_spec(fx.SCORE_A, 0.5)
    definition_b, spec_b = fx.stage_spec(fx.SCORE_B, 0.5)
    backend = FakeFairShipBackend(stages=(spec_a, spec_b))
    request = fx.request(
        subject_ids=tuple(f"s{i}" for i in range(10)), replications_per_subject=3
    )
    bundle = backend.evaluate(request)

    by_candidate = {}
    for decision in bundle.decisions:
        by_candidate.setdefault(decision.subject_ref, {})[
            decision.stage_definition_id
        ] = decision.decision
    pairs = {
        (
            values.get(definition_a.stage_definition_id),
            values.get(definition_b.stage_definition_id),
        )
        for values in by_candidate.values()
    }
    # All four combinations occur, so neither stage implies the other.
    assert {(True, False), (False, True)} <= pairs
    assert len(bundle.observations) == 2 * len(bundle.candidates)


def test_stages_must_not_share_one_observable():
    """Sharing an observable across stages would manufacture an implication."""
    _definition, spec = fx.stage_spec()
    shadow = FakeStageSpec(
        stage_definition_id="a-different-stage@sha256:0000",
        observation_definition_id=spec.observation_definition_id,
        threshold=0.25,
    )
    with pytest.raises(ValueError, match="distinct observation_definition_ids"):
        FakeFairShipBackend(stages=(spec, shadow))
