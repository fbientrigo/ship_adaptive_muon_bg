"""Unit C tests: state-level aggregation with hand-computed expected values.

Every count asserted here is written out by hand in the test's docstring
before the code runs, so a reader can check the arithmetic without trusting
the implementation.
"""

from __future__ import annotations

import pytest

from ship_muon_bg.entities import DecisionEvaluationStatus, ExecutionStatus
from ship_muon_bg.tagging.aggregation import (
    ROLLUP_RULE,
    DecisionLevel,
    ExecutionStageOutcome,
    TaggingDataset,
    aggregate_state_stage_evaluations,
    classify_executions,
)

from tests.evaluation import fixtures as fx

STAGE = fx.FIXTURE_STAGE
OTHER_STAGE = "fixture.other_stage_v0@sha256:9999"


def _aggregate(executions, candidates, decisions, stages=(STAGE,), **kwargs):
    return aggregate_state_stage_evaluations(
        executions=executions,
        candidates=candidates,
        decisions=decisions,
        stage_definition_ids=stages,
        **kwargs,
    )


def _positive(subject_ref):
    return fx.decision(subject_ref, DecisionEvaluationStatus.EVALUATED, True)


def _negative(subject_ref):
    return fx.decision(subject_ref, DecisionEvaluationStatus.EVALUATED, False)


def _unavailable(subject_ref):
    return fx.decision(subject_ref, DecisionEvaluationStatus.TECHNICALLY_UNAVAILABLE, None)


def _not_evaluated(subject_ref):
    return fx.decision(subject_ref, DecisionEvaluationStatus.NOT_EVALUATED, None)


# --------------------------------------------------------------------------
# The core arithmetic, fully hand-computed
# --------------------------------------------------------------------------


def test_hand_computed_aggregate_for_one_state():
    """One state, six executions of it, one stage.

    e1 succeeded, 1 candidate, decision True          -> POSITIVE
    e2 succeeded, 2 candidates, both False            -> NEGATIVE
    e3 succeeded, 0 candidates, execution attests it
       applied the stage and found nothing            -> NEGATIVE  (clean zero)
    e4 crashed                                        -> TECHNICAL_FAILURE
    e5 succeeded, 1 candidate, evidence unavailable   -> TECHNICALLY_CENSORED
    e6 succeeded, 1 candidate, stage never applied    -> NOT_EVALUATED

    valid = 2 negatives + 1 positive = 3
    eta_hat = 1 / 3
    candidate counts over valid executions: {1: 1, 2: 1, 0: 1}
    positive-candidate counts over valid executions: {1: 1, 0: 2}
    exactly one execution was decided at execution level: e3
    """
    executions = (
        fx.execution("e1", "s1"),
        fx.execution("e2", "s1"),
        fx.execution("e3", "s1"),
        fx.execution("e4", "s1", status=ExecutionStatus.TECHNICAL_FAILURE,
                     failure_reason="fixture: crash"),
        fx.execution("e5", "s1"),
        fx.execution("e6", "s1"),
    )
    candidates = (
        fx.candidate("c1", "e1", candidate_index=0),
        fx.candidate("c2", "e2", candidate_index=0),
        fx.candidate("c3", "e2", candidate_index=1),
        fx.candidate("c5", "e5", candidate_index=0),
        fx.candidate("c6", "e6", candidate_index=0),
    )
    decisions = (
        _positive("c1"),
        _negative("c2"),
        _negative("c3"),
        _negative("e3"),
        _unavailable("c5"),
    )
    dataset = _aggregate(executions, candidates, decisions)
    assert len(dataset.rows) == 1
    row = dataset.rows[0]

    assert row.execution_count == 6
    assert row.positive_count == 1
    assert row.negative_count == 2
    assert row.valid_count == 3
    assert row.technical_failure_count == 1
    assert row.technically_censored_count == 1
    assert row.not_evaluated_count == 1
    assert row.censored_count == 3
    assert row.eta_hat == pytest.approx(1 / 3)
    assert row.candidate_count_distribution == {0: 1, 1: 1, 2: 1}
    assert row.positive_candidate_count_distribution == {0: 2, 1: 1}
    assert row.execution_decided_count == 1
    assert row.rollup_rule == ROLLUP_RULE


def test_technical_failures_never_enter_the_denominator():
    """The single most important arithmetic property in this layer.

    Nine of ten executions crash. eta_hat must be 1/1, not 1/10 and not 0.1.
    """
    executions = tuple(
        fx.execution(f"e{i}", "s1", status=ExecutionStatus.TECHNICAL_FAILURE,
                     failure_reason="fixture: crash")
        for i in range(9)
    ) + (fx.execution("e9", "s1"),)
    candidates = (fx.candidate("c1", "e9", candidate_index=0),)
    row = _aggregate(executions, candidates, (_positive("c1"),)).rows[0]
    assert row.execution_count == 10
    assert row.technical_failure_count == 9
    assert row.valid_count == 1
    assert row.positive_count == 1
    assert row.eta_hat == 1.0


def test_a_state_with_nothing_evaluable_has_an_unknown_rate_not_zero():
    """Writing 0.0 here would fabricate a confident negative out of no data."""
    executions = (
        fx.execution("e1", "s1", status=ExecutionStatus.TECHNICAL_FAILURE,
                     failure_reason="fixture: crash"),
    )
    row = _aggregate(executions, (), ()).rows[0]
    assert row.valid_count == 0
    assert row.eta_hat is None
    assert row.positive_count == 0


def test_every_execution_lands_in_exactly_one_outcome_class():
    """Exercised with all five outcomes present, so the partition arithmetic is
    actually under load rather than summing one bucket."""
    executions = (
        fx.execution("e1", "s1"),
        fx.execution("e2", "s1"),
        fx.execution("e3", "s1"),
        fx.execution("e4", "s1", status=ExecutionStatus.TECHNICAL_FAILURE,
                     failure_reason="fixture: crash"),
        fx.execution("e5", "s1"),
    )
    candidates = (
        fx.candidate("c1", "e1", candidate_index=0),
        fx.candidate("c2", "e2", candidate_index=0),
        fx.candidate("c3", "e3", candidate_index=0),
    )
    decisions = (_positive("c1"), _negative("c2"), _unavailable("c3"))
    dataset = _aggregate(executions, candidates, decisions)
    row = dataset.rows[0]
    assert {r.outcome for r in dataset.results} == set(ExecutionStageOutcome)
    assert (
        row.valid_count
        + row.technical_failure_count
        + row.technically_censored_count
        + row.not_evaluated_count
        == row.execution_count
    )


# --------------------------------------------------------------------------
# Censoring semantics
# --------------------------------------------------------------------------


def test_a_positive_candidate_settles_the_execution_despite_a_censored_sibling():
    """Partial identification done right: censoring only matters when it could
    still have changed the answer."""
    executions = (fx.execution("e1", "s1"),)
    candidates = (
        fx.candidate("c1", "e1", candidate_index=0),
        fx.candidate("c2", "e1", candidate_index=1),
    )
    decisions = (_positive("c1"), _unavailable("c2"))
    row = _aggregate(executions, candidates, decisions).rows[0]
    assert row.positive_count == 1
    assert row.technically_censored_count == 0


def test_a_censored_sibling_blocks_a_negative():
    """Here the censored candidate could have been the positive one, so the
    execution is not evaluable and must not be counted as a negative."""
    executions = (fx.execution("e1", "s1"),)
    candidates = (
        fx.candidate("c1", "e1", candidate_index=0),
        fx.candidate("c2", "e1", candidate_index=1),
    )
    decisions = (_negative("c1"), _unavailable("c2"))
    row = _aggregate(executions, candidates, decisions).rows[0]
    assert row.negative_count == 0
    assert row.technically_censored_count == 1
    assert row.valid_count == 0
    assert row.eta_hat is None


def test_a_candidate_with_no_decision_for_the_stage_is_not_a_negative():
    executions = (fx.execution("e1", "s1"),)
    candidates = (fx.candidate("c1", "e1", candidate_index=0),)
    row = _aggregate(executions, candidates, ()).rows[0]
    assert row.not_evaluated_count == 1
    assert row.negative_count == 0


def test_zero_candidates_is_a_negative_only_when_something_attests_it():
    """The most dangerous ambiguity in the layer.

    "Ran and found nothing" and "never looked" are indistinguishable from an
    empty record set. Resolving that in favour of a countable negative would
    turn the commonest real partial failure -- a job that exits 0 but whose
    reconstruction output is empty or truncated -- into a confident physics
    zero. So a negative has to be asserted by someone who knows.
    """
    attested = _aggregate(
        (fx.execution("e1", "s1"),), (), (_negative("e1"),)
    ).rows[0]
    assert attested.negative_count == 1
    assert attested.valid_count == 1
    assert attested.eta_hat == 0.0
    assert attested.candidate_count_distribution == {0: 1}
    assert attested.execution_decided_count == 1


def test_silence_about_a_stage_is_never_a_physics_zero():
    silent = _aggregate((fx.execution("e1", "s1"),), (), ()).rows[0]
    assert silent.negative_count == 0
    assert silent.not_evaluated_count == 1
    assert silent.valid_count == 0
    assert silent.eta_hat is None


def test_a_stage_nobody_ever_implemented_yields_no_rate_at_all():
    """Aggregating for a stage no record mentions must not return 0.0."""
    executions = tuple(fx.execution(f"e{i}", "s1") for i in range(3))
    row = _aggregate(executions, (), (), stages=("stage.never_built@sha256:z",)).rows[0]
    assert row.eta_hat is None
    assert row.not_evaluated_count == 3
    assert row.valid_count == 0


def test_not_evaluated_and_technically_censored_stay_distinct():
    executions = (fx.execution("e1", "s1"), fx.execution("e2", "s1"))
    candidates = (
        fx.candidate("c1", "e1", candidate_index=0),
        fx.candidate("c2", "e2", candidate_index=0),
    )
    decisions = (_unavailable("c1"), _not_evaluated("c2"))
    row = _aggregate(executions, candidates, decisions).rows[0]
    assert row.technically_censored_count == 1
    assert row.not_evaluated_count == 1


# --------------------------------------------------------------------------
# Statistical unit and grouping
# --------------------------------------------------------------------------


def test_repeated_executions_group_under_one_state_not_as_separate_states():
    """Mission invariant 3.1: the unit of analysis is the execution, but the
    unit of *grouping* is the source state."""
    executions = tuple(fx.execution(f"e{i}", "s1") for i in range(7))
    dataset = _aggregate(executions, (), ())
    assert len(dataset.rows) == 1
    assert dataset.rows[0].execution_count == 7
    assert dataset.subject_ids == ("s1",)


def test_configurations_are_never_pooled():
    """COMPAT-01. Same state, same stage, two configurations -> two rows."""
    executions = (
        fx.execution("e1", "s1", fs_sim_configuration_id=fx.CONFIG_A),
        fx.execution("e2", "s1", fs_sim_configuration_id=fx.CONFIG_B),
    )
    dataset = _aggregate(executions, (), ())
    assert len(dataset.rows) == 2
    assert dataset.configuration_ids == tuple(sorted((fx.CONFIG_A, fx.CONFIG_B)))
    with pytest.raises(ValueError, match="must not be pooled"):
        dataset.require_single_configuration()


def test_a_configuration_filter_selects_without_merging():
    executions = (
        fx.execution("e1", "s1", fs_sim_configuration_id=fx.CONFIG_A),
        fx.execution("e2", "s1", fs_sim_configuration_id=fx.CONFIG_B),
    )
    dataset = _aggregate(
        executions, (), (), allowed_configuration_ids=(fx.CONFIG_A,)
    )
    assert dataset.require_single_configuration() == fx.CONFIG_A
    assert len(dataset.rows) == 1


def test_state_definitions_are_a_compatibility_axis_too():
    from ship_muon_bg.entities import TagSubject

    subjects = (
        fx.subject("s1"),
        TagSubject(
            subject_id="s2", subject_type="muon_state", state_definition_id="other@sha256:z"
        ),
    )
    executions = (fx.execution("e1", "s1"), fx.execution("e2", "s2"))
    dataset = _aggregate(executions, (), (), subjects=subjects)
    assert len(dataset.state_definition_ids) == 2
    with pytest.raises(ValueError, match="must not be pooled"):
        dataset.require_single_state_definition()


def test_stages_are_aggregated_independently():
    """Mission invariant 3.4: no stage implies any other."""
    executions = (fx.execution("e1", "s1"),)
    candidates = (fx.candidate("c1", "e1", candidate_index=0),)
    decisions = (
        fx.decision("c1", DecisionEvaluationStatus.EVALUATED, True, stage_definition_id=STAGE),
        fx.decision(
            "c1", DecisionEvaluationStatus.EVALUATED, False, stage_definition_id=OTHER_STAGE
        ),
    )
    dataset = _aggregate(executions, candidates, decisions, stages=(STAGE, OTHER_STAGE))
    assert len(dataset.rows) == 2
    by_stage = {row.stage_definition_id: row for row in dataset.rows}
    assert by_stage[STAGE].positive_count == 1
    assert by_stage[OTHER_STAGE].positive_count == 0
    assert by_stage[OTHER_STAGE].negative_count == 1


# --------------------------------------------------------------------------
# Multiplicity survives aggregation
# --------------------------------------------------------------------------


def test_multiplicity_survives_into_the_aggregate():
    """Mission invariant 3.3: N is still recoverable after summarising."""
    executions = tuple(fx.execution(f"e{i}", "s1") for i in range(3))
    candidates = (
        fx.candidate("c1", "e1", candidate_index=0),
        fx.candidate("c2", "e2", candidate_index=0),
        fx.candidate("c3", "e2", candidate_index=1),
        fx.candidate("c4", "e2", candidate_index=2),
    )
    # e0 reconstructed nothing and says so, so it is a real zero rather than a
    # silence.
    decisions = tuple(_positive(f"c{i}") for i in range(1, 5)) + (_negative("e0"),)
    row = _aggregate(executions, candidates, decisions).rows[0]
    assert row.candidate_count_distribution == {0: 1, 1: 1, 3: 1}
    assert row.positive_candidate_count_distribution == {0: 1, 1: 1, 3: 1}
    # Y = 1{N >= 1} is derivable from the distribution; N is not derivable
    # from Y, which is why the distribution and not just eta_hat is retained.
    assert row.positive_count == 2
    assert row.eta_hat == pytest.approx(2 / 3)


# --------------------------------------------------------------------------
# Integrity: refuse rather than guess
# --------------------------------------------------------------------------


def test_duplicate_executions_are_refused():
    """Counting one run twice would inflate every rate derived from it."""
    executions = (fx.execution("e1", "s1"), fx.execution("e1", "s1"))
    with pytest.raises(ValueError, match="duplicate execution_id"):
        _aggregate(executions, (), ())


def test_a_candidate_without_its_execution_is_refused():
    with pytest.raises(ValueError, match="no execution in this record set"):
        _aggregate((fx.execution("e1", "s1"),), (fx.candidate("c1", "e_other"),), ())


def test_contradictory_decisions_for_one_entity_and_stage_are_refused():
    executions = (fx.execution("e1", "s1"),)
    candidates = (fx.candidate("c1", "e1", candidate_index=0),)
    with pytest.raises(ValueError, match="ambiguous"):
        _aggregate(executions, candidates, (_positive("c1"), _negative("c1")))


def test_an_execution_level_veto_contradicting_its_candidates_is_refused():
    """A veto is a different rule and needs its own stage definition; silently
    letting it overwrite candidate-level semantics is how stage meaning gets
    rewritten without anyone noticing."""
    executions = (fx.execution("e1", "s1"),)
    candidates = (fx.candidate("c1", "e1", candidate_index=0),)
    decisions = (_positive("c1"), _negative("e1"))
    with pytest.raises(ValueError, match="veto must be a distinct stage definition"):
        _aggregate(executions, candidates, decisions)


def test_an_execution_level_decision_is_authoritative_when_consistent():
    executions = (fx.execution("e1", "s1"),)
    candidates = (fx.candidate("c1", "e1", candidate_index=0),)
    decisions = (_negative("c1"), _negative("e1"))
    row = _aggregate(executions, candidates, decisions).rows[0]
    assert row.negative_count == 1
    assert row.valid_count == 1


def test_non_boolean_decisions_are_refused_rather_than_coerced():
    executions = (fx.execution("e1", "s1"),)
    candidates = (fx.candidate("c1", "e1", candidate_index=0),)
    decisions = (fx.decision("c1", DecisionEvaluationStatus.EVALUATED, "yes"),)
    with pytest.raises(ValueError, match="boolean stage decisions only"):
        _aggregate(executions, candidates, decisions)


def test_an_execution_attributed_to_an_undeclared_subject_is_refused():
    with pytest.raises(ValueError, match="was not declared"):
        _aggregate(
            (fx.execution("e1", "ghost"),), (), (), subjects=(fx.subject("s1"),)
        )


# --------------------------------------------------------------------------
# The proxy-ready artifact
# --------------------------------------------------------------------------


def test_the_tagging_table_is_plain_serializable_rows_with_no_weight():
    """WEIGHT-01: choosing how to weight a state is a separate, explicit
    decision. Emitting one opaque number here is how w_i and h(U_i) get
    multiplied together by accident."""
    import json

    executions = (fx.execution("e1", "s1"),)
    dataset = _aggregate(executions, (), (), subjects=(fx.subject("s1"),))
    records = dataset.as_records()
    json.dumps(records)  # must round-trip without custom encoders
    assert records[0]["state_definition_id"] == fx.STATE_DEFINITION
    for record in records:
        for key in record:
            assert "weight" not in key.lower()


def test_the_tagging_table_retains_raw_counts_not_only_eta_hat():
    """So a different uncertainty treatment can be applied later without
    re-running the simulator."""
    executions = (fx.execution("e1", "s1"), fx.execution("e2", "s1"))
    candidates = (fx.candidate("c1", "e1", candidate_index=0),)
    record = _aggregate(executions, candidates, (_positive("c1"),)).as_records()[0]
    for key in (
        "valid_count",
        "positive_count",
        "negative_count",
        "technical_failure_count",
        "technically_censored_count",
        "not_evaluated_count",
        "candidate_count_distribution",
        "positive_candidate_count_distribution",
        "fs_sim_configuration_id",
        "rollup_rule",
    ):
        assert key in record


def test_per_execution_results_are_retained_for_audit():
    executions = (fx.execution("e1", "s1"), fx.execution("e2", "s1"))
    dataset = _aggregate(executions, (), (_negative("e1"),))
    assert len(dataset.results) == 2
    by_id = {r.execution_id: r for r in dataset.results}
    assert by_id["e1"].outcome is ExecutionStageOutcome.NEGATIVE
    assert by_id["e2"].outcome is ExecutionStageOutcome.NOT_EVALUATED


def test_classify_executions_is_usable_on_its_own():
    results = classify_executions(
        executions=(fx.execution("e1", "s1"),),
        candidates=(),
        decisions=(_negative("e1"),),
        stage_definition_ids=(STAGE,),
    )
    assert len(results) == 1
    assert results[0].outcome is ExecutionStageOutcome.NEGATIVE
    assert results[0].is_valid is True
    assert results[0].decision_level is DecisionLevel.EXECUTION


# --------------------------------------------------------------------------
# Guards added after red-team and senior review
# --------------------------------------------------------------------------


def test_a_realization_scoped_decision_is_refused_not_dropped():
    """The defect both reviewers found independently.

    ``EXEC-02a`` names ``InteractionRealization`` as the canonical lineage node
    between an execution and its candidates — exactly where a DIS selection
    would attach. This layer only knows how to roll up execution- and
    candidate-level records. Ignoring a realization-scoped decision turned an
    ``EVALUATED`` positive into a counted physics negative, silently, with
    ``not_evaluated_count == 0`` so the censoring audit showed nothing.
    """
    executions = (fx.execution("e1", "s1"),)
    decisions = (_positive("r1"),)
    with pytest.raises(ValueError, match="cannot interpret"):
        _aggregate(executions, (), decisions)


def test_a_subject_scoped_decision_is_refused():
    with pytest.raises(ValueError, match="cannot interpret"):
        _aggregate((fx.execution("e1", "s1"),), (), (_positive("s1"),))


def test_a_decision_referencing_nothing_is_refused():
    """A typo in a subject_ref is broken lineage, not a missing evaluation."""
    executions = (fx.execution("e1", "s1"),)
    candidates = (fx.candidate("c1", "e1", candidate_index=0),)
    with pytest.raises(ValueError, match="cannot interpret"):
        _aggregate(executions, candidates, (_positive("c1_TYPO"),))


def test_decisions_for_other_stages_are_left_alone():
    """The refusal is scoped to the stages actually being aggregated, so a
    record set carrying decisions for stages this call ignores still works."""
    executions = (fx.execution("e1", "s1"),)
    candidates = (fx.candidate("c1", "e1", candidate_index=0),)
    decisions = (
        _positive("c1"),
        fx.decision("r1", DecisionEvaluationStatus.EVALUATED, True,
                    stage_definition_id=OTHER_STAGE),
    )
    row = _aggregate(executions, candidates, decisions, stages=(STAGE,)).rows[0]
    assert row.positive_count == 1


def test_an_id_naming_both_an_execution_and_a_candidate_is_refused():
    """Per-bundle integrity does not compose. Records concatenated from several
    bundles can reuse one id in two roles, and a single decision would then be
    consumed once as a candidate outcome and once as an execution outcome."""
    executions = (fx.execution("SHARED", "s1"), fx.execution("e2", "s1"))
    candidates = (fx.candidate("SHARED", "e2", candidate_index=0),)
    with pytest.raises(ValueError, match="both an execution and a candidate"):
        _aggregate(executions, candidates, ())


def test_an_execution_level_positive_contradicting_its_candidates_is_refused():
    """The contradiction guard is symmetric. Preferring the execution-level
    record would overwrite candidate-level stage semantics just as surely as
    the reverse, which the guard already refused."""
    executions = (fx.execution("e1", "s1"),)
    candidates = (
        fx.candidate("c1", "e1", candidate_index=0),
        fx.candidate("c2", "e1", candidate_index=1),
    )
    decisions = (_negative("c1"), _negative("c2"), _positive("e1"))
    with pytest.raises(ValueError, match="absence of a candidate must be a distinct"):
        _aggregate(executions, candidates, decisions)


def test_an_identified_positive_survives_an_execution_level_censoring_marker():
    """Partial identification applies at both levels. Discarding an identified
    positive because the run-level record says "could not evaluate" abandons the
    same logic applied to sibling candidates."""
    executions = (fx.execution("e1", "s1"),)
    candidates = (fx.candidate("c1", "e1", candidate_index=0),)
    decisions = (_positive("c1"), _unavailable("e1"))
    dataset = _aggregate(executions, candidates, decisions)
    row = dataset.rows[0]
    assert row.positive_count == 1
    assert row.eta_hat == 1.0
    assert dataset.results[0].decision_level is DecisionLevel.CANDIDATES


def test_an_execution_level_positive_with_no_candidates_is_recorded_as_such():
    """A stage whose positive condition is the *absence* of a candidate — a veto
    — is legitimately positive with N = 0. So Y and 1{N>=1} genuinely diverge
    here, and ``decision_level`` records which executions took that path rather
    than leaving a consumer to discover the disagreement."""
    executions = (fx.execution("e1", "s1"),)
    dataset = _aggregate(executions, (), (_positive("e1"),))
    row = dataset.rows[0]
    assert row.positive_count == 1
    assert row.candidate_count_distribution == {0: 1}
    assert row.execution_decided_count == 1
    assert dataset.results[0].decision_level is DecisionLevel.EXECUTION


def test_runs_differing_only_in_options_do_not_share_a_row():
    """A configuration label cannot cover a free-form options mapping, so the
    digest each execution records is part of the grouping key. Without it eight
    executions under two shield geometries pooled into one eta_hat and every
    guard returned green."""
    executions = (
        fx.execution("e1", "s1", options_digest="digest_geometry_a"),
        fx.execution("e2", "s1", options_digest="digest_geometry_b"),
    )
    candidates = (fx.candidate("c1", "e1", candidate_index=0),)
    dataset = _aggregate(executions, candidates, (_positive("c1"), _negative("e2")))
    assert len(dataset.rows) == 2
    assert {row.eta_hat for row in dataset.rows} == {1.0, 0.0}
    assert {row.options_digest for row in dataset.rows} == {
        "digest_geometry_a",
        "digest_geometry_b",
    }


def test_duplicate_rows_for_one_state_are_refused():
    """Aggregating batches separately and concatenating the datasets makes one
    source state two population members: a naive mean over rows returns 0.5
    where the truth is 0.25, and summed exposure double-counts."""
    executions = tuple(fx.execution(f"e{i}", "s1") for i in range(4))
    candidates = (fx.candidate("c1", "e0", candidate_index=0),)
    decisions = (_positive("c1"),) + tuple(_negative(f"e{i}") for i in range(1, 4))
    whole = _aggregate(executions, candidates, decisions)
    assert whole.rows[0].eta_hat == pytest.approx(0.25)

    first = _aggregate(executions[:1], candidates, decisions[:1])
    rest = _aggregate(executions[1:], (), decisions[1:])
    with pytest.raises(ValueError, match="duplicate .subject, configuration"):
        TaggingDataset(rows=first.rows + rest.rows)


def test_a_declared_state_that_never_ran_is_reported_not_dropped():
    """If dropped work correlates with anything physical — long jobs,
    high-energy muons, timeouts — the proxy trains on a selection-biased
    population, and a table built only from execution results shows nothing."""
    subjects = (fx.subject("s1"), fx.subject("s2"), fx.subject("s3"))
    executions = (fx.execution("e1", "s1"), fx.execution("e3", "s3"))
    dataset = _aggregate(executions, (), (), subjects=subjects)
    assert dataset.subject_ids == ("s1", "s3")
    assert dataset.unevaluated_subject_ids == ("s2",)


def test_the_state_definition_gate_refuses_to_pass_on_unverified_data():
    """A require_* guard that silently no-ops is worse than no guard, because it
    reads as verification."""
    dataset = _aggregate((fx.execution("e1", "s1"),), (), ())
    with pytest.raises(ValueError, match="state definition unknown"):
        dataset.require_single_state_definition()


def test_a_bare_string_configuration_filter_is_refused():
    """set("cfg_a") is a set of characters, which silently selected nothing."""
    with pytest.raises(TypeError, match="not a single string"):
        _aggregate(
            (fx.execution("e1", "s1"),), (), (), allowed_configuration_ids=fx.CONFIG_A
        )


def test_the_rollup_rule_is_content_addressed():
    """A bare version string would let two vintages of rows pool under one
    label after the precedence changed."""
    from ship_muon_bg.entities.identifiers import definition_id
    from ship_muon_bg.tagging.aggregation import ROLLUP_RULE_CONTENT

    assert ROLLUP_RULE == definition_id("execution_stage_rollup_v0", ROLLUP_RULE_CONTENT)
    altered = dict(ROLLUP_RULE_CONTENT, negative_requires_attestation=False)
    assert definition_id("execution_stage_rollup_v0", altered) != ROLLUP_RULE


def test_a_generator_of_decisions_is_not_half_consumed():
    """The record set is walked twice internally; a generator would otherwise
    make the second pass see nothing, silently disabling the refusal above."""
    executions = (fx.execution("e1", "s1"),)
    with pytest.raises(ValueError, match="cannot interpret"):
        _aggregate(executions, (), (d for d in (_positive("r1"),)))
