"""The ``simulation_backend`` boundary (project side of module 1).

Everything that runs physics — the future ``fairship_adapter`` wrapping the
FairShip base simulator, a ``toy_simulator``, HTCondor batch execution —
lives *behind* the :class:`SimulationBackend` protocol. The core (and the
``Nflow`` / ``ProxyTagger`` modules) only ever see
:class:`FlowProposalRecord` in and :class:`SimulationResult` out.

This package is pure Python and imports neither FairShip nor ROOT. The
CERN FairShip repository itself is referenced at the top-level ``FairShip/``
directory (gitignored); see ``docs/contracts/fairship_adapter_contract_v0.md``
for the full boundary contract.
"""

from __future__ import annotations

from ship_muon_bg.simulation.backend import SimulationBackend
from ship_muon_bg.simulation.types import (
    FlowProposalRecord,
    OutcomeCategory,
    SimulationResult,
)

__all__ = [
    "SimulationBackend",
    "FlowProposalRecord",
    "SimulationResult",
    "OutcomeCategory",
]
