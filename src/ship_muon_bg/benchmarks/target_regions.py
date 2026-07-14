"""Architecture-independent region descriptors for controlled targets (v0).

A :class:`MahalanobisRegion` is a fixed ellipsoid in *base* physical
coordinates (i.e. the exact mixture coordinates, before any target transform).
It is used to declare a rare-mode tail region for D5 that does not depend on
any model's internal component labels: membership is evaluated by mapping a
transformed physical sample back through the target's exact inverse and testing
a fixed Mahalanobis radius against a fixed center and precision matrix.

Pure NumPy, deterministic, JSON-serializable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from .controlled_targets import N_PHYSICAL_DIMS


@dataclass(frozen=True)
class MahalanobisRegion:
    """A fixed Mahalanobis ellipsoid in base physical coordinates.

    ``contains(base)`` returns a boolean mask selecting rows whose squared
    Mahalanobis distance to ``center`` (under ``precision``) is at most
    ``radius_sq``.
    """

    region_id: str
    center: np.ndarray
    precision: np.ndarray
    radius_sq: float
    coordinate_frame: str = "base_physical"
    description: str = ""

    def __post_init__(self) -> None:
        center = np.array(self.center, dtype=np.float64, copy=True)
        precision = np.array(self.precision, dtype=np.float64, copy=True)
        if center.shape != (N_PHYSICAL_DIMS,):
            raise ValueError("region center must have shape (5,)")
        if precision.shape != (N_PHYSICAL_DIMS, N_PHYSICAL_DIMS):
            raise ValueError("region precision must have shape (5, 5)")
        if not np.allclose(precision, precision.T, atol=1e-12):
            raise ValueError("region precision must be symmetric")
        radius_sq = float(self.radius_sq)
        if not np.isfinite(radius_sq) or radius_sq <= 0.0:
            raise ValueError("region radius_sq must be finite and > 0")
        center.flags.writeable = False
        precision.flags.writeable = False
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "precision", precision)
        object.__setattr__(self, "radius_sq", radius_sq)

    def mahalanobis_sq(self, base_physical: np.ndarray) -> np.ndarray:
        diff = np.asarray(base_physical, dtype=np.float64) - self.center
        return np.einsum("ij,jk,ik->i", diff, self.precision, diff)

    def contains(self, base_physical: np.ndarray) -> np.ndarray:
        return self.mahalanobis_sq(base_physical) <= self.radius_sq

    def manifest(self) -> Dict[str, Any]:
        return {
            "region_id": self.region_id,
            "kind": "mahalanobis_ellipsoid",
            "coordinate_frame": self.coordinate_frame,
            "center": self.center.tolist(),
            "precision": self.precision.tolist(),
            "radius_sq": self.radius_sq,
            "description": self.description,
        }
