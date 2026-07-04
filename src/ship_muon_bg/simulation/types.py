"""Project-level record types crossing the ``simulation_backend`` boundary.

Implements the data-flow types of
``docs/contracts/fairship_adapter_contract_v0.md``:

    FlowProposalRecord[] -> simulation_backend -> SimulationResult[]

Pure Python + NumPy-free dataclasses; no FairShip, no ROOT.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Mapping, Optional, Tuple


class OutcomeCategory(str, enum.Enum):
    """Mutually exclusive outcome of one simulated candidate.

    Invariant (from the ``fairship_adapter`` contract): ``TECHNICAL_FAILURE``
    is about *whether the simulation is trustworthy*; the other two are about
    *what a trustworthy simulation concluded*. A technical problem must never
    be counted as a physics result, and never used as a proxy training label.
    """

    TECHNICAL_FAILURE = "technical_failure"
    PHYSICS_REJECTION = "physics_rejection"
    ACCEPTED_CANDIDATE = "accepted_candidate"


@dataclass(frozen=True)
class FlowProposalRecord:
    """One proposed muon candidate handed to a ``simulation_backend``.

    Kinematic fields mirror the ``(N, 8)`` data contract
    (``ship_muon_bg.data_contracts.schema``): momenta in GeV/c, positions in
    meters, ``pdg_id`` a PDG code, ``weight`` dimensionless.
    """

    candidate_id: str
    px: float
    py: float
    pz: float
    x: float
    y: float
    z: float
    pdg_id: int
    weight: float
    # Provenance: which dataset/model/campaign produced this proposal.
    dataset_hash: Optional[str] = None
    campaign_id: Optional[str] = None
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SimulationResult:
    """Parsed, project-level outcome for one simulated candidate.

    ``dis`` is the operational DIS tag from the backend: ``True``/``False``
    only when ``outcome`` is a valid physics outcome, ``None`` when the run
    was a ``TECHNICAL_FAILURE`` (no trustworthy physics answer exists).
    """

    candidate_id: str
    outcome: OutcomeCategory
    dis: Optional[bool] = None
    # Human-readable reason (e.g. which taxonomy cause: timeout, ROOT crash,
    # selection cut) — for reports, never for branching logic.
    detail: str = ""
    raw_output_paths: Tuple[str, ...] = ()
    # Reproducibility metadata per the adapter contract (seed, geometry_tag,
    # backend_name/version, git_commit, config_hash, ...).
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.outcome is OutcomeCategory.TECHNICAL_FAILURE and self.dis is not None:
            raise ValueError(
                "a technical_failure has no trustworthy physics answer; dis must be None"
            )
