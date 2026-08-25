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
from ship_muon_bg.simulation.evaluation import (
    EvaluationBackend,
    EvaluationBundle,
    EvaluationRequest,
    assert_requests_compatible,
    evaluate_verified,
    execution_shortfall,
    verify_evaluation_bundle,
)
from ship_muon_bg.simulation.fake_fairship import (
    FakeCandidatePlan,
    FakeFairShipBackend,
    FakeRunPlan,
    FakeStageSpec,
)
from ship_muon_bg.simulation.stub_backend import MinimalStubBackend
from ship_muon_bg.simulation.types import (
    FlowProposalRecord,
    OutcomeCategory,
    SimulationResult,
)

# The two fake backends are importable from their own modules but deliberately
# absent from ``__all__``: they are test infrastructure, and a star-import of
# the production simulation namespace should not hand anyone a non-physical
# simulator.
__all__ = [
    "SimulationBackend",
    "FlowProposalRecord",
    "SimulationResult",
    "OutcomeCategory",
    "EvaluationRequest",
    "EvaluationBundle",
    "EvaluationBackend",
    "verify_evaluation_bundle",
    "evaluate_verified",
    "execution_shortfall",
    "assert_requests_compatible",
]
