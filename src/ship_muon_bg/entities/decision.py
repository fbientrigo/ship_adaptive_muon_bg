"""``StageDecision`` — immutable, versioned stage interpretations.

Implements ``docs/contracts/tagging_contract_v0.md`` ``DECISION-01``-``DECISION-05``.
A ``StageDecision`` is never derived from ``ExecutionStatus``
(``entities.lineage``) automatically (``DECISION-04``): the two live on
different objects, and this module never imports ``entities.lineage`` to
keep that separation structural, not just documented.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional, Tuple

from ship_muon_bg.entities.identifiers import content_hash


class DecisionEvaluationStatus(str, enum.Enum):
    """Whether a stage was actually evaluated for this subject/entity.

    ``NOT_EVALUATED`` must remain distinct from a negative ``decision``
    value (e.g. ``False``) — a stage that never ran is not evidence of a
    negative outcome (``CENSOR-01``/``CENSOR-02``).
    """

    EVALUATED = "evaluated"
    NOT_EVALUATED = "not_evaluated"
    TECHNICALLY_UNAVAILABLE = "technically_unavailable"


def _require_nonempty_str(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")


def stage_decision_id(
    *,
    stage_definition_id: str,
    subject_ref: str,
    evidence_references: Iterable[str],
    evaluation_status: "DecisionEvaluationStatus",
) -> str:
    """The canonical content-addressed identity of one stage decision.

    Identity is *what was decided about*, not *what was concluded*: the
    versioned rule, the entity it was applied to, the exact evidence
    consumed, and whether it could be evaluated at all. The concluded value
    is deliberately absent — for a well-formed rule it is a function of those
    inputs, so including it would let two contradictory records about the
    same evidence coexist under different ids instead of colliding and being
    rejected as the contradiction they are.

    Lives here, in ``entities``, rather than in either producer, because both
    the tagging evaluator and any simulation backend that reports its own
    opaque stage outcomes must agree on it exactly. Two independent copies of
    this formula would drift silently.
    """
    _require_nonempty_str(stage_definition_id, "stage_definition_id")
    _require_nonempty_str(subject_ref, "subject_ref")
    if not isinstance(evaluation_status, DecisionEvaluationStatus):
        raise TypeError("evaluation_status must be a DecisionEvaluationStatus")
    references = tuple(evidence_references)
    for reference in references:
        _require_nonempty_str(reference, "evidence_reference")
    identity = {
        "stage_definition_id": stage_definition_id,
        "subject_ref": subject_ref,
        "evidence_references": list(references),
        "evaluation_status": evaluation_status.value,
    }
    return f"decision@sha256:{content_hash(identity)}"


@dataclass(frozen=True)
class StageDecision:
    """One immutable, versioned interpretation at a named stage.

    ``decision`` is populated only when ``evaluation_status is EVALUATED``;
    it is always ``None`` otherwise — enforced below, not left to caller
    discipline, so a technical failure or a not-evaluated stage can never be
    silently read as a negative (e.g. ``False``) decision value.
    """

    decision_id: str
    subject_ref: str
    stage_definition_id: str
    evaluation_status: DecisionEvaluationStatus
    decision: Optional[Any]
    reason: str = ""
    evidence_references: Tuple[str, ...] = ()
    config_provenance: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("decision_id", "subject_ref", "stage_definition_id"):
            _require_nonempty_str(getattr(self, field_name), field_name)
        if not isinstance(self.evaluation_status, DecisionEvaluationStatus):
            raise TypeError("evaluation_status must be a DecisionEvaluationStatus")
        if self.evaluation_status is DecisionEvaluationStatus.EVALUATED:
            if self.decision is None:
                raise ValueError("an EVALUATED StageDecision must carry a decision value")
        elif self.decision is not None:
            raise ValueError(
                f"a {self.evaluation_status.value} StageDecision must not carry a decision "
                "value (got a non-None value where NOT_EVALUATED/TECHNICALLY_UNAVAILABLE "
                "must stay distinct from any concrete decision, including False)"
            )
        if not isinstance(self.config_provenance, Mapping):
            raise TypeError("config_provenance must be a mapping")
        object.__setattr__(
            self, "config_provenance", MappingProxyType(dict(self.config_provenance))
        )
