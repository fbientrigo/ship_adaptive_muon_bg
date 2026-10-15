"""Execution / interaction / candidate lineage.

Implements ``docs/contracts/tagging_contract_v0.md`` ``EXEC-01``-``EXEC-04``
and the corrected ``OPEN-09`` resolution in
``docs/architecture/scientific_architecture_v2.md``: the lineage entity
between an execution and a reconstructed candidate is interaction-neutral
(``InteractionRealization``, not the DIS-specific ``DISRealization`` of the
first draft). Current Muon DIS is one ``interaction_type`` value, not a
hardcoded entity name.

Cardinalities are ``0..N`` throughout by construction: nothing here forces
exactly one execution per subject, exactly one realization per execution, or
exactly one candidate per realization (``EXEC-02``). ``ExecutionStatus`` is
health-only and must never contain a physics decision value such as
``accepted_candidate``/``physics_rejection`` (``EXEC-03``, ``DECISION-04``);
those live on ``StageDecision`` (see ``entities.decision``), a separate
object entirely.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Optional

#: Reserved key in ``FSSimExecution.provenance``. Backends must record the
#: content hash of the ``options`` their run was launched with, because a
#: configuration *label* cannot cover a free-form options mapping and a
#: persisted execution would otherwise carry no trace of what actually varied
#: (``CONF-01``, ``COMPAT-01``). Aggregation groups on it, so two runs whose
#: options differ can never pool even under one configuration id.
OPTIONS_DIGEST_PROVENANCE_KEY = "options_digest"


class ExecutionStatus(str, enum.Enum):
    """Run health only — never a physics outcome (``EXEC-03``)."""

    SUCCEEDED = "succeeded"
    TECHNICAL_FAILURE = "technical_failure"


def _require_nonempty_str(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")


@dataclass(frozen=True)
class FSSimExecution:
    """One FairShip (or FairShip-like backend) execution attempt against a
    ``TagSubject``. A subject may have ``0..N`` of these (``EXEC-01``);
    repeated executions of the same subject are conditional repetitions of
    that subject, never new draws from the nominal source measure
    (``COMPAT-04`` — enforced by aggregation code downstream, not by this
    dataclass, which only records one execution's own identity).

    ``seed``, ``failure_reason`` and ``provenance`` are additive
    reproducibility/health provenance (``PROV-01``). ``failure_reason`` stays
    free text on the health axis only: it explains *why the run is untrusted*,
    never what the physics concluded (``EXEC-03``), so a non-empty reason is
    rejected on a ``SUCCEEDED`` execution rather than becoming a second,
    informal outcome channel alongside ``StageDecision``.
    """

    execution_id: str
    subject_id: str
    fs_sim_configuration_id: str
    execution_status: ExecutionStatus
    seed: Optional[int] = None
    failure_reason: str = ""
    provenance: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_nonempty_str(self.execution_id, "execution_id")
        _require_nonempty_str(self.subject_id, "subject_id")
        _require_nonempty_str(self.fs_sim_configuration_id, "fs_sim_configuration_id")
        if not isinstance(self.execution_status, ExecutionStatus):
            raise TypeError("execution_status must be an ExecutionStatus")
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise TypeError("seed must be None or an int")
        if not isinstance(self.failure_reason, str):
            raise TypeError("failure_reason must be a string")
        if not isinstance(self.provenance, Mapping):
            raise TypeError("provenance must be a mapping")
        # Copy, then freeze. Storing the caller's mapping by reference would
        # let a verified record's recorded configuration be rewritten
        # afterwards by anyone still holding the dict (``PROV-01``).
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))
        if self.failure_reason and self.execution_status is not ExecutionStatus.TECHNICAL_FAILURE:
            raise ValueError(
                "failure_reason describes run health and is only meaningful on a "
                "TECHNICAL_FAILURE execution; a succeeded run's physics outcome "
                "belongs on a StageDecision, not in this field"
            )


@dataclass(frozen=True)
class InteractionRealization:
    """One interaction realization produced within an ``FSSimExecution``.

    ``interaction_type`` names *what kind* of interaction this is (e.g.
    ``"muon_dis"``); ``interaction_definition_id`` names the exact versioned
    rule that classified it as such. An execution may produce ``0..N`` of
    these (``EXEC-02``) — a scalar boolean DIS tag is exactly the 1:1
    assumption this entity replaces. A DIS-specific typed view/specialization
    may be layered on top later if useful; this base entity stays neutral.
    """

    realization_id: str
    execution_id: str
    interaction_type: str
    interaction_definition_id: str

    def __post_init__(self) -> None:
        for field_name in (
            "realization_id",
            "execution_id",
            "interaction_type",
            "interaction_definition_id",
        ):
            _require_nonempty_str(getattr(self, field_name), field_name)


@dataclass(frozen=True)
class ReconstructedCandidate:
    """One reconstructed detector-level candidate. Always tied to an
    ``FSSimExecution``; ``realization_id`` is explicitly optional
    (``None`` = an execution-level candidate not scoped to one
    realization) rather than defaulting to an invented 1:1 association
    (``EXEC-02``, ``EXEC-04``).

    ``candidate_index`` is this candidate's 0-based position within the
    candidate collection of its parent, where the parent is the pair
    ``(execution_id, realization_id)`` — so index ``0`` under one realization
    and index ``0`` under a sibling realization of the same execution are two
    different candidates, not a contradiction. It preserves multiplicity
    *ordering* as data; it never implies how many siblings exist, and nothing
    may infer a count from it (``EXEC-02``, ``OPEN-03``). The multiplicity
    itself is always the number of ``ReconstructedCandidate`` records actually
    present. Uniqueness of the triple is not checkable on one record in
    isolation; it is enforced where a collection exists, in
    ``simulation.evaluation.EvaluationBundle``.
    """

    candidate_id: str
    execution_id: str
    realization_id: Optional[str] = None
    candidate_index: Optional[int] = None

    def __post_init__(self) -> None:
        _require_nonempty_str(self.candidate_id, "candidate_id")
        _require_nonempty_str(self.execution_id, "execution_id")
        if self.realization_id is not None:
            _require_nonempty_str(self.realization_id, "realization_id")
        if self.candidate_index is not None:
            if isinstance(self.candidate_index, bool) or not isinstance(
                self.candidate_index, int
            ):
                raise TypeError("candidate_index must be None or an int")
            if self.candidate_index < 0:
                raise ValueError("candidate_index must be non-negative")
