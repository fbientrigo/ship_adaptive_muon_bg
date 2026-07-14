"""Exact controlled density benchmarks for the nominal density track (v0).

Numerical benchmark distributions in canonical physical coordinates
``[px, py, pz, x, y]``; not SHiP physics models. See
``docs/contracts/controlled_targets_v0.md``.
"""

from __future__ import annotations

from .controlled_targets import (
    ControlledTarget,
    ControlledTargetConfigError,
    ControlledTargetDomainError,
    ControlledTargetError,
    ControlledTargetShapeError,
    GaussianComponent,
    N_PHYSICAL_DIMS,
    PHYSICAL_COLUMNS,
    SUPPORTED_PDG_IDS,
    SUPPORTED_TARGET_IDS,
    SampleBatch,
    TARGET_SCHEMA_VERSION,
    embed_physical_to_raw,
    make_controlled_target,
)

__all__ = [
    "ControlledTarget",
    "ControlledTargetError",
    "ControlledTargetConfigError",
    "ControlledTargetShapeError",
    "ControlledTargetDomainError",
    "GaussianComponent",
    "SampleBatch",
    "TARGET_SCHEMA_VERSION",
    "SUPPORTED_TARGET_IDS",
    "SUPPORTED_PDG_IDS",
    "PHYSICAL_COLUMNS",
    "N_PHYSICAL_DIMS",
    "make_controlled_target",
    "embed_physical_to_raw",
]
