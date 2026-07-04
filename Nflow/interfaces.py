"""Interfaces for the density/proposal track.

``DensityModel`` is the protocol from
``docs/architecture/ml_skeleton_local_pkl_v0.md`` (section 5): any proposal
model — the placeholder baseline, the legacy RealNVP once it is adapted and
tested, or a future conditional flow — plugs in behind it without touching
the acquisition or diagnostics layers.

``BiasStrategy`` is the A/B seam for module 2's open question: *how* should
``U(x)`` scores bias the proposal toward DIS-likely regions? Candidate
mechanisms (data aggregation, modified loss, both, or something else) each
implement this protocol, so campaigns can compare them under identical
seeds, data, and artifacts.

No implementation here trains anything; these are contracts only.
"""

from __future__ import annotations

from typing import Protocol, Tuple, runtime_checkable

import numpy as np


@runtime_checkable
class DensityModel(Protocol):
    """A proposal/density model over post-shield muon feature space."""

    def fit(self, x_train: np.ndarray) -> None:
        """Fit the model on training rows (already validated + normalized)."""
        ...

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        """Per-row log-density under the model, shape ``(n,)``."""
        ...

    def sample(self, n: int, *, seed: int) -> np.ndarray:
        """Draw ``n`` proposal rows deterministically for a given ``seed``."""
        ...


@runtime_checkable
class BiasStrategy(Protocol):
    """How ``U(x)`` scores bias the proposal model toward DIS-likely regions.

    A strategy may act on the data side, the loss side, or both:

    - ``resample`` implements the *data aggregation* family: given training
      rows, their event weights, and their ``U(x)`` scores, it returns a
      possibly reweighted/augmented ``(x, w)`` pair to train on.
    - ``loss_weights`` implements the *modified loss* family: a per-row
      multiplier applied to the negative-log-likelihood terms.

    A pure data-aggregation strategy returns all-ones from ``loss_weights``;
    a pure modified-loss strategy returns ``(x, w)`` unchanged from
    ``resample``; a combined strategy overrides both. ``name`` is recorded
    in every artifact so A/B comparisons stay attributable.
    """

    name: str

    def resample(
        self, x: np.ndarray, w: np.ndarray, scores: np.ndarray, *, seed: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return the (possibly reweighted/augmented) training set ``(x, w)``."""
        ...

    def loss_weights(self, x: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Per-row loss multiplier, shape ``(n,)`` (all ones = unmodified loss)."""
        ...
