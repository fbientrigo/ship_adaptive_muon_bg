"""Immutable, declarative stage definitions for the generic tagging layer.

``StageDefinition`` deliberately stores semantic rule content rather than an
arbitrary callable.  The content is recursively frozen and its identity is
computed with the canonical ``entities.definition_id`` helper, so a reviewer
can inspect the rule represented by a ``stage_definition_id`` without relying
on Python object identity or function representations.
"""

from __future__ import annotations

import math
from collections.abc import Mapping as ABCMapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Tuple

from ship_muon_bg.entities.identifiers import definition_id


def _freeze_semantic_value(value: Any) -> Any:
    """Return a recursively immutable JSON-compatible semantic value."""
    if isinstance(value, ABCMapping):
        frozen = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("semantic-content mapping keys must be strings")
            frozen[key] = _freeze_semantic_value(item)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_semantic_value(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("semantic-content floats must be finite")
        return value
    raise TypeError(
        "semantic content must contain only JSON-compatible values; "
        f"got {type(value).__name__}"
    )


def _plain_semantic_value(value: Any) -> Any:
    """Convert frozen mappings/tuples back to JSON-compatible containers."""
    if isinstance(value, ABCMapping):
        return {key: _plain_semantic_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_semantic_value(item) for item in value]
    return value


@dataclass(frozen=True)
class StageDefinition:
    """An immutable, declarative interpretation rule.

    ``required_observation_definition_ids`` declares the evidence definitions
    the evaluator may consume for one entity reference.  ``semantic_content``
    describes the rule itself; it must contain data, not executable callables.
    The required definitions are included in the content-addressed identity,
    so changing either the rule or its declared evidence changes the ID.
    """

    stage_name: str
    semantic_content: Mapping[str, Any]
    required_observation_definition_ids: Tuple[str, ...] = ()
    stage_definition_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.stage_name, str) or not self.stage_name:
            raise ValueError("stage_name must be a non-empty string")
        if not isinstance(self.semantic_content, ABCMapping):
            raise TypeError("semantic_content must be a mapping")
        if isinstance(self.required_observation_definition_ids, str):
            raise TypeError("required observation definitions must be a sequence")

        required = tuple(self.required_observation_definition_ids)
        for observation_definition_id in required:
            if not isinstance(observation_definition_id, str) or not observation_definition_id:
                raise ValueError(
                    "required observation definition IDs must be non-empty strings"
                )
        if len(set(required)) != len(required):
            raise ValueError("required observation definition IDs must be unique")

        frozen_content = _freeze_semantic_value(self.semantic_content)
        identity_content = {
            "stage_name": self.stage_name,
            "semantic_content": _plain_semantic_value(frozen_content),
            "required_observation_definition_ids": list(required),
        }

        object.__setattr__(self, "semantic_content", frozen_content)
        object.__setattr__(self, "required_observation_definition_ids", required)
        object.__setattr__(
            self,
            "stage_definition_id",
            definition_id(self.stage_name, identity_content),
        )


def scalar_above_threshold_stage(
    *, observation_definition_id: str, threshold: float
) -> StageDefinition:
    """Build the one controlled, explicitly non-physical fixture stage.

    This is a test fixture for the tagging machinery, not a SHiP tag or a
    physics-selection definition.  Its declarative rule is ``value >
    threshold`` for one required scalar observation.
    """
    if not isinstance(observation_definition_id, str) or not observation_definition_id:
        raise ValueError("observation_definition_id must be a non-empty string")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise TypeError("threshold must be a finite numeric value")
    threshold_value = float(threshold)
    if not math.isfinite(threshold_value):
        raise ValueError("threshold must be a finite numeric value")

    return StageDefinition(
        stage_name="fixture.scalar_above_threshold",
        semantic_content={
            "fixture": "controlled_non_physical",
            "is_physical": False,
            "observation_definition_id": observation_definition_id,
            "operation": "greater_than",
            "threshold": threshold_value,
        },
        required_observation_definition_ids=(observation_definition_id,),
    )
