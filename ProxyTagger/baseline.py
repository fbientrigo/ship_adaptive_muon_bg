"""Dummy baseline scorer — the v0 deliverable of the proxy track.

``DummyProxy`` exists purely so the acquisition interface and score-artifact
schema can be exercised and tested before real labels exist. It is
explicitly **non-physical** and must never be reported as a result.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ship_muon_bg.data_contracts.schema import MOMENTUM_COLUMNS, column_indices


class DummyProxy:
    """A trivially-derived, deterministic ``U(x)`` placeholder.

    Scores a candidate by a monotone squashing of its momentum magnitude:
    ``|p| / (|p| + scale)``, which maps ``[0, inf)`` into ``[0, 1)``. There
    is no physics in this choice; it only gives downstream code a score
    vector with the right shape, range, and determinism.
    """

    name = "dummy_momentum_monotone"
    is_physical = False

    def __init__(self, scale: float = 100.0):
        if scale <= 0:
            raise ValueError("scale must be positive")
        self.scale = float(scale)

    def fit(
        self,
        x: np.ndarray,
        labels: np.ndarray,
        *,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        """No-op: the dummy learns nothing; it only honors the interface."""

    def score(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        momentum = x[:, column_indices(MOMENTUM_COLUMNS)]
        p_mag = np.linalg.norm(momentum, axis=1)
        return p_mag / (p_mag + self.scale)
