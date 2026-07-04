"""ProxyTagger — the ``U(x)`` module (module 3 of the adaptive loop).

Consumes labeled outcomes produced by the ``simulation_backend`` (module 1)
and maintains ``U(x)``: a continuous score in ``[0, 1]`` over the full
simulation input space ``x``, from 0 (never DIS) to 1 (DIS always). ``U(x)``
is an *operational worth-simulating score* — a noisy boundary estimate, not
a likelihood and not a physics prediction. It steers the ``Nflow`` proposal
bias and supports visualization of the hyperparameter landscape.

Pure Python + NumPy; imports neither FairShip nor ROOT, and never runs
physics itself.
"""

from __future__ import annotations

from ProxyTagger.baseline import DummyProxy
from ProxyTagger.interfaces import SCORE_MAX, SCORE_MIN, SCORE_SCHEMA_VERSION, ProxyScorer

__all__ = [
    "ProxyScorer",
    "DummyProxy",
    "SCORE_SCHEMA_VERSION",
    "SCORE_MIN",
    "SCORE_MAX",
]
