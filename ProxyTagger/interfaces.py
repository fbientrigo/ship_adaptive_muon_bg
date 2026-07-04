"""The ``U(x)`` contract.

Semantics (fixed by ``docs/architecture/ml_skeleton_local_pkl_v0.md``
section 6 and the ``fairship_adapter`` contract):

- ``U(x)`` is an **operational danger / worth-simulating score** in
  ``[0, 1]``: 0 means "never produces DIS", 1 means "always produces DIS".
  It is treated as a noisy measurement of the DIS boundary, not a
  likelihood and not a physics prediction.
- **Labels come only from the ``simulation_backend``** (today a future
  ``toy_simulator``; eventually the FairShip base simulator via the
  ``fairship_adapter``). Labels are produced, never assumed from legacy
  files.
- **``technical_failure`` is never a training label.** Only
  ``physics_rejection`` vs ``accepted_candidate`` outcomes may be used;
  technical failures must be excluded, never folded into the negative
  class (see ``ship_muon_bg.simulation.types.OutcomeCategory``).
- If ``U(x)`` is used as a probability it must be calibrated, and the
  primary metric is **tail false-negative rate in the dangerous region**,
  not global accuracy/AUC.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import numpy as np

# Version of the score-artifact schema. Bump on any breaking change to the
# fields below so downstream consumers can detect incompatible artifacts.
SCORE_SCHEMA_VERSION = "0"

# Valid range of a U(x) score.
SCORE_MIN = 0.0
SCORE_MAX = 1.0

# Required fields of any persisted score artifact (JSON side of an NPZ score
# array). Provenance-first: a score without these fields is not trusted.
SCORE_ARTIFACT_FIELDS = (
    "schema_version",
    "proxy_name",
    "is_physical",  # DummyProxy and other placeholders must set False
    "dataset_hash",
    "seed",
    "n_rows",
    "scores_path",  # NPZ file holding the (n,) float array
)


@runtime_checkable
class ProxyScorer(Protocol):
    """A ``U(x)`` model: fit on simulation-produced labels, score candidates."""

    name: str

    # Placeholders/baselines must declare themselves non-physical so their
    # scores can never be reported as results.
    is_physical: bool

    def fit(
        self,
        x: np.ndarray,
        labels: np.ndarray,
        *,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        """Fit on candidate rows ``x`` and binary/ordinal DIS ``labels``.

        ``labels`` must be derived from non-technical-failure simulation
        outcomes only; the caller filters ``technical_failure`` out before
        calling.
        """
        ...

    def score(self, x: np.ndarray) -> np.ndarray:
        """Return ``U(x)`` in ``[SCORE_MIN, SCORE_MAX]``, shape ``(n,)``."""
        ...
