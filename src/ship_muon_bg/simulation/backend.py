"""The ``SimulationBackend`` protocol.

Any simulator — ``toy_simulator``, dry-run simulator, or the FairShip base
simulator wrapped by the future ``fairship_adapter`` — satisfies this
interface. Callers never know which backend they are talking to.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Protocol, Sequence, runtime_checkable

from ship_muon_bg.simulation.types import FlowProposalRecord, SimulationResult


@runtime_checkable
class SimulationBackend(Protocol):
    """Simulate proposed candidates and classify every outcome.

    Requirements on implementations (from the ``fairship_adapter`` contract):

    - Deterministic for a given ``seed`` (never seeded from wall-clock time).
    - Every input candidate yields exactly one :class:`SimulationResult`,
      classified into exactly one ``OutcomeCategory``.
    - No hardcoded CERN/EOS paths; all paths and environment profiles come
      from ``config``.
    """

    name: str

    def simulate(
        self,
        candidates: Sequence[FlowProposalRecord],
        *,
        seed: int,
        config: Mapping[str, Any],
    ) -> List[SimulationResult]:
        """Run the backend on ``candidates`` and return one result each."""
        ...
