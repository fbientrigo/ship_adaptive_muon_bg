"""The first empirical FairShip intermediate endpoint (not final B)."""

from ship_muon_bg.entities.identifiers import definition_id
from ship_muon_bg.tagging.definitions import StageDefinition


SBT_SELECTION_RULE = (
    "1000 < detector_id < 999999 AND abs(pdg_id) == 13 "
    "AND momentum_gev > 3.0"
)
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
    "SBT_OBSERVATION_DEFINITION_ID",
    "SBT_SELECTION_RULE",
    "SBT_STAGE",
    "SBT_STAGE_DEFINITION_ID",
]
