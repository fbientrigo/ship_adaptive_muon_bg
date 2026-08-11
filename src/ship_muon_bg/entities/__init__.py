"""Canonical entity layer — backend-independent, imports neither FairShip nor ROOT.

Implements the entity schemas of ``docs/contracts/tagging_contract_v0.md``
and ``docs/architecture/scientific_architecture_v2.md``: ``TagSubject``,
execution/interaction/candidate lineage, typed ``Observation``s, and
``StageDecision``. This package is the one thing both
``src/ship_muon_bg/adapters/fairship/`` (FairShip-specific, not implemented
yet) and the tagging/proxy/proposal layers depend on; it never depends on
either.

``TrainingTarget`` is deliberately not implemented here yet — it is the next
semantic layer and stays a separate slice (see
``docs/architecture/scientific_architecture_v2.md`` §11 migration path).
"""

from __future__ import annotations

from ship_muon_bg.entities.decision import DecisionEvaluationStatus, StageDecision
from ship_muon_bg.entities.identifiers import canonical_json, content_hash, definition_id
from ship_muon_bg.entities.lineage import (
    ExecutionStatus,
    FSSimExecution,
    InteractionRealization,
    ReconstructedCandidate,
)
from ship_muon_bg.entities.observation import (
    ObservationEnvelope,
    ObservationEvaluationStatus,
    ObservationPayload,
    ScalarObservationPayload,
    SequenceObservationPayload,
)
from ship_muon_bg.entities.subject import TagSubject

__all__ = [
    "TagSubject",
    "ExecutionStatus",
    "FSSimExecution",
    "InteractionRealization",
    "ReconstructedCandidate",
    "ObservationEvaluationStatus",
    "ObservationPayload",
    "ScalarObservationPayload",
    "SequenceObservationPayload",
    "ObservationEnvelope",
    "DecisionEvaluationStatus",
    "StageDecision",
    "canonical_json",
    "content_hash",
    "definition_id",
]
