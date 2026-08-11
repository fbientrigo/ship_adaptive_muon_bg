"""``ObservationEnvelope`` — typed, provenanced evidence records.

Implements ``docs/contracts/tagging_contract_v0.md`` ``OBS-01``-``OBS-04``
using the two-level design fixed during this continuation: a common
``ObservationEnvelope`` (provenance/units/status metadata) wrapping a typed
``ObservationPayload``, rather than an unrestricted name/value/units EAV
system. This is deliberately not a generic dynamic framework: concrete
payload types are plain frozen dataclasses inheriting from the
``ObservationPayload`` marker, so ``isinstance`` checks stay meaningful
without a registry or plugin discovery.

Only two payload shapes are implemented in this slice, to prove — not to
enumerate — that scalar and non-scalar payloads are both supported without
flattening (``OBS-02``): ``ScalarObservationPayload`` and
``SequenceObservationPayload``. Future structured payloads (e.g. a
``PathObservation`` carrying per-step traversal records) are additional
``ObservationPayload`` subclasses, not a redesign of the envelope.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Mapping, Optional, Tuple


class ObservationEvaluationStatus(str, enum.Enum):
    """``OBS-03``/``CENSOR-02``: three distinct reasons a payload may or may
    not be present. Never coerced to a sentinel value in the payload itself.
    """

    COMPUTED = "computed"
    NOT_APPLICABLE = "not_applicable"
    TECHNICALLY_UNAVAILABLE = "technically_unavailable"


class ObservationPayload:
    """Marker base for typed observation payloads. Concrete payloads are
    frozen dataclasses inheriting from this class; the base itself carries
    no fields or behavior.
    """

    __slots__ = ()


@dataclass(frozen=True)
class ScalarObservationPayload(ObservationPayload):
    """A single scalar measurement (e.g. a path length in meters)."""

    value: float


@dataclass(frozen=True)
class SequenceObservationPayload(ObservationPayload):
    """An ordered sequence measurement (e.g. per-step traversal distances).
    Proves ``OBS-02``: a payload need not be a single flat-table scalar.
    """

    values: Tuple[float, ...]


def _require_nonempty_str(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")


@dataclass(frozen=True)
class ObservationEnvelope:
    """One provenanced observation attached to a subject/execution/
    realization/candidate (``subject_ref`` holds whichever id applies).

    Enforces ``OBS-03``: a ``COMPUTED`` envelope must carry a payload; a
    ``NOT_APPLICABLE``/``TECHNICALLY_UNAVAILABLE`` envelope must carry
    ``None`` — never a sentinel value written into the payload slot.
    """

    observation_id: str
    subject_ref: str
    observation_definition_id: str
    units: str
    evaluation_status: ObservationEvaluationStatus
    evidence_reference: str
    payload: Optional[ObservationPayload]
    config_provenance: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "observation_id",
            "subject_ref",
            "observation_definition_id",
            "units",
            "evidence_reference",
        ):
            _require_nonempty_str(getattr(self, field_name), field_name)
        if not isinstance(self.evaluation_status, ObservationEvaluationStatus):
            raise TypeError("evaluation_status must be an ObservationEvaluationStatus")
        if self.payload is not None and not isinstance(self.payload, ObservationPayload):
            raise TypeError("payload must be None or an ObservationPayload")
        if self.evaluation_status is ObservationEvaluationStatus.COMPUTED:
            if self.payload is None:
                raise ValueError("a COMPUTED observation must carry a payload")
        elif self.payload is not None:
            raise ValueError(
                f"a {self.evaluation_status.value} observation must not carry a payload"
            )
