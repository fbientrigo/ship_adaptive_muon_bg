"""Generic, backend-independent stage tagging core.

This package stops at ``StageDecision``.  Its controlled fixture is explicitly
non-physical; no SHiP tag, TrainingTarget, proxy, or FairShip integration is
defined here.
"""

from ship_muon_bg.tagging.definitions import StageDefinition, scalar_above_threshold_stage
from ship_muon_bg.tagging.evaluator import StageEvaluator

__all__ = ["StageDefinition", "StageEvaluator", "scalar_above_threshold_stage"]
