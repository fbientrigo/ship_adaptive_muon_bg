"""``TagSubject`` — the neutral operational object tracked through lineage.

Implements ``docs/contracts/tagging_contract_v0.md`` ``SUBJ-01``-``SUBJ-03``.
Deliberately not named ``SourceState``: no ``subject_type`` asserted here may
be read as the final thesis source-state definition (``SUBJ-02``, ``OPEN-01``
— that remains an open scientific question this package does not resolve).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TagSubject:
    """A declared muon state (or other tracked object) identified by
    ``subject_id``, whose coordinates are given meaning only by
    ``state_definition_id`` — never assumed from ``subject_type`` alone.
    """

    subject_id: str
    subject_type: str
    state_definition_id: str

    def __post_init__(self) -> None:
        for field_name in ("subject_id", "subject_type", "state_definition_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must be a non-empty string")
