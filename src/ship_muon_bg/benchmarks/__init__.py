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
    D5_VARIANTS,
    GaussianComponent,
    N_PHYSICAL_DIMS,
    PHYSICAL_COLUMNS,
    SUPPORTED_PDG_IDS,
    SUPPORTED_TARGET_IDS,
    SampleBatch,
    TARGET_SCHEMA_VERSION,
    TransformedControlledTarget,
    calibrate_d5_rare_region,
    embed_physical_to_raw,
    make_controlled_target,
)
from .target_regions import MahalanobisRegion
from .target_transforms import (
    ComposedTransform,
    ExactTransform,
    SinhArcsinhSkewTransform,
    TriangularBananaTransform,
    numerical_forward_log_abs_det_jacobian,
)

__all__ = [
    "ControlledTarget",
    "ControlledTargetError",
    "ControlledTargetConfigError",
    "ControlledTargetShapeError",
    "ControlledTargetDomainError",
    "GaussianComponent",
    "SampleBatch",
    "TransformedControlledTarget",
    "TARGET_SCHEMA_VERSION",
    "SUPPORTED_TARGET_IDS",
    "SUPPORTED_PDG_IDS",
    "PHYSICAL_COLUMNS",
    "N_PHYSICAL_DIMS",
    "D5_VARIANTS",
    "make_controlled_target",
    "calibrate_d5_rare_region",
    "embed_physical_to_raw",
    "ExactTransform",
    "TriangularBananaTransform",
    "SinhArcsinhSkewTransform",
    "ComposedTransform",
    "numerical_forward_log_abs_det_jacobian",
    "MahalanobisRegion",
]
