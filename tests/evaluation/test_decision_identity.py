"""Decision ids are content-addressed and persisted, so they must never drift.

``stage_decision_id`` was extracted from ``tagging/evaluator.py`` into
``entities/decision.py`` so the tagging evaluator and any backend reporting its
own opaque stage outcomes cannot disagree. The refactor had to be
value-preserving: a changed formula would silently orphan every decision id
already written to disk.

The golden values below were computed with the pre-extraction implementation at
commit ``2965ddd`` and are frozen here. If a change to the identity formula is
ever deliberate, this test failing is the moment to decide what happens to
existing records — not a nuisance to update.
"""

from __future__ import annotations

import pytest

from ship_muon_bg.entities import DecisionEvaluationStatus, stage_decision_id

# (stage_definition_id, subject_ref, evidence_references, status) -> id
GOLDEN = (
    (
        "stage@sha256:a",
        "c1",
        ("o1",),
        DecisionEvaluationStatus.EVALUATED,
        "decision@sha256:367d9838eac082e95ccd83bb",
    ),
    (
        "stage@sha256:a",
        "c1",
        (),
        DecisionEvaluationStatus.NOT_EVALUATED,
        "decision@sha256:eab9287e3276f127f339696f",
    ),
    (
        "stage@sha256:b",
        "e9",
        ("o1", "o2"),
        DecisionEvaluationStatus.TECHNICALLY_UNAVAILABLE,
        "decision@sha256:2fe76ecfa9cdf17beb2b528a",
    ),
)


@pytest.mark.parametrize("stage,subject,evidence,status,expected_prefix", GOLDEN)
def test_decision_identity_is_unchanged_since_the_extraction(
    stage, subject, evidence, status, expected_prefix
):
    produced = stage_decision_id(
        stage_definition_id=stage,
        subject_ref=subject,
        evidence_references=evidence,
        evaluation_status=status,
    )
    assert produced.startswith(expected_prefix)


def test_evidence_order_is_part_of_identity():
    """The evaluator canonicalizes evidence order through the definition's
    sorted dependency tuple, so two ids differing only in order mean the caller
    bypassed that canonicalization — which is worth surfacing, not smoothing."""
    first = stage_decision_id(
        stage_definition_id="s@sha256:1",
        subject_ref="c1",
        evidence_references=("a", "b"),
        evaluation_status=DecisionEvaluationStatus.EVALUATED,
    )
    second = stage_decision_id(
        stage_definition_id="s@sha256:1",
        subject_ref="c1",
        evidence_references=("b", "a"),
        evaluation_status=DecisionEvaluationStatus.EVALUATED,
    )
    assert first != second


def test_evaluation_status_separates_a_censored_id_from_an_evaluated_one():
    """Otherwise a censored record and a real conclusion about the same entity
    and evidence would share an identity."""
    common = dict(
        stage_definition_id="s@sha256:1", subject_ref="c1", evidence_references=("a",)
    )
    ids = {
        stage_decision_id(evaluation_status=status, **common)
        for status in DecisionEvaluationStatus
    }
    assert len(ids) == len(DecisionEvaluationStatus)
