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
from typing import Any, Mapping, Optional, Tuple


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
