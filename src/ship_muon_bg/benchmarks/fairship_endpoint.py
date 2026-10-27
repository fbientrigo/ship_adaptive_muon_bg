"""The first empirical FairShip intermediate endpoint (not final B)."""

from ship_muon_bg.entities.identifiers import definition_id
from ship_muon_bg.tagging.definitions import StageDefinition


SBT_DETID_RANGE = (1000, 999999)  # exclusive; PROVISIONAL for TRY_2026 (queue item 6)
SBT_PDG_ABS = 13
SBT_THRESHOLD_GEV = 3.0
# Byte-identical to the historical literal: it is hashed into the pinned stage ids.
SBT_SELECTION_RULE = (
    f"{SBT_DETID_RANGE[0]} < detector_id < {SBT_DETID_RANGE[1]} AND abs(pdg_id) == {SBT_PDG_ABS} "
    f"AND momentum_gev > {SBT_THRESHOLD_GEV}"
)


def is_sbt_hit(detector_id: int, pdg_id: int, momentum_gev: float, threshold_gev: float = SBT_THRESHOLD_GEV) -> bool:
    """The one SBT predicate; `SBT_SELECTION_RULE` is its text."""
    return SBT_DETID_RANGE[0] < detector_id < SBT_DETID_RANGE[1] and abs(pdg_id) == SBT_PDG_ABS and momentum_gev > threshold_gev

_SBT_OBSERVATION_DEFINITION_CONTENT = {
    "event_field": "cbmsim.vetoPoint",
    "metric": "qualifying_hit_count",
    "selection_rule": SBT_SELECTION_RULE,
    "source": "FairShip/muonDIS/make_nTuple_SBT.py",
    "units": "count",
}
SBT_OBSERVATION_DEFINITION_ID = definition_id(
    "fairship_sbt_qualifying_hit_count_v0", _SBT_OBSERVATION_DEFINITION_CONTENT
)

SBT_STAGE = StageDefinition(
    stage_name="fairship_sbt_selection_v0",
    semantic_content={
        "is_physical": True,
        "operation": "greater_than",
        "observation_definition_id": SBT_OBSERVATION_DEFINITION_ID,
        "selection_rule": SBT_SELECTION_RULE,
        "source": "FairShip/muonDIS/make_nTuple_SBT.py",
        "threshold": 0.0,
        "intermediate_Y_k_is_not_endpoint_B": True,
    },
    required_observation_definition_ids=(SBT_OBSERVATION_DEFINITION_ID,),
)

SBT_STAGE_DEFINITION_ID = SBT_STAGE.stage_definition_id

__all__ = [
    "SBT_DETID_RANGE",
    "SBT_PDG_ABS",
    "SBT_THRESHOLD_GEV",
    "is_sbt_hit",
    "SBT_OBSERVATION_DEFINITION_ID",
    "SBT_SELECTION_RULE",
    "SBT_STAGE",
    "SBT_STAGE_DEFINITION_ID",
]
