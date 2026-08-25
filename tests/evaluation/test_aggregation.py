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
    ExecutionStageOutcome,
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

    e1 succeeded, 1 candidate, decision True         -> POSITIVE
    e2 succeeded, 2 candidates, both False           -> NEGATIVE
    e3 succeeded, 0 candidates                       -> NEGATIVE  (clean zero)
    e4 crashed                                       -> TECHNICAL_FAILURE
    e5 succeeded, 1 candidate, evidence unavailable  -> TECHNICALLY_CENSORED
    e6 succeeded, 1 candidate, stage never applied   -> NOT_EVALUATED

    valid = 2 negatives + 1 positive = 3
    eta_hat = 1 / 3
    candidate counts over valid executions: {1: 1, 2: 1, 0: 1}
    positive-candidate counts over valid executions: {1: 1, 0: 2}
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
    """Enforced by the aggregate itself, so no execution can be double- or
    un-counted whatever the classification logic does."""
    executions = tuple(fx.execution(f"e{i}", "s1") for i in range(5))
    row = _aggregate(executions, (), ()).rows[0]
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


def test_zero_candidates_is_a_genuine_negative():
    """A run that finished cleanly and reconstructed nothing really did fail to
    produce a candidate. This is the only path to a negative."""
    row = _aggregate((fx.execution("e1", "s1"),), (), ()).rows[0]
    assert row.negative_count == 1
    assert row.valid_count == 1
    assert row.eta_hat == 0.0
    assert row.candidate_count_distribution == {0: 1}


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
    decisions = tuple(_positive(f"c{i}") for i in range(1, 5))
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
    dataset = _aggregate(executions, (), ())
    assert len(dataset.results) == 2
    assert {r.outcome for r in dataset.results} == {ExecutionStageOutcome.NEGATIVE}


def test_classify_executions_is_usable_on_its_own():
    results = classify_executions(
        executions=(fx.execution("e1", "s1"),),
        candidates=(),
        decisions=(),
        stage_definition_ids=(STAGE,),
    )
    assert len(results) == 1
    assert results[0].outcome is ExecutionStageOutcome.NEGATIVE
    assert results[0].is_valid is True
