"""Nflow — normalizing-flow proposal module (module 2 of the adaptive loop).

Learns the post-shield muon distribution and proposes new candidate points
``x``, biased toward regions the ``ProxyTagger`` score ``U(x)`` marks as
likely to produce Deep Inelastic Scattering (DIS). The biasing mechanism
(data aggregation, modified loss, or both) is an open research question, so
it stays behind the :class:`~Nflow.interfaces.BiasStrategy` seam and every
candidate mechanism is an interchangeable, A/B-testable implementation.

This top-level package imports only NumPy-typed interfaces. Heavy
dependencies (torch, h5py, scikit-learn, ...) are confined to
``Nflow.legacy`` (the untested NFlow-fork code) and to future concrete
model modules — never to this ``__init__``.
"""

from __future__ import annotations

from Nflow.interfaces import BiasStrategy, DensityModel

__all__ = ["DensityModel", "BiasStrategy"]
