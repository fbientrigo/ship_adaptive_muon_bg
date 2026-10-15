"""D9 per-candidate training configuration (§6).

Every training choice a candidate needs is an explicit, versioned JSON field
here -- never a code literal buried in the runner. ``configs/afterms/d9_training_v0.json``
is the frozen instance of this schema for the 6 primary candidates in
``configs/afterms/d9_candidate_plan_v0.json``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

REQUIRED_FIELDS = (
    "campaign_id",
    "candidate_id",
    "model_family",
    "architecture",
    "modeled_features",
    "feature_order",
    "pdg_policy",
    "pdg_value",
    "preprocessing_name",
    "target_measure",
    "weighting_policy",
    "weighting_estimator_version",
    "train_shards",
    "validation_shards",
    "test_shards",
    "seed_set",
    "max_epochs",
    "minimum_epochs",
    "early_stopping_patience",
    "optimizer",
    "learning_rate",
    "batch_size",
    "weight_decay",
    "gradient_clipping",
    "dtype",
    "device_policy",
    "checkpoint_policy",
    "evaluation_policy",
)

# §6: "Three seeds are an engineering starting point, not a universal
# sufficiency claim." Kept as an explicit default, never hardcoded downstream.
DEFAULT_SEED_SET: List[int] = [20260720, 20260721, 20260722]


class TrainingConfigError(ValueError):
    pass


def load_training_config(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_candidate_config(config: Dict[str, Any]) -> List[str]:
    violations = []
    missing = [f for f in REQUIRED_FIELDS if f not in config]
    if missing:
        violations.append(f"missing required fields: {missing}")
    if not missing:
        if not isinstance(config["seed_set"], list) or not config["seed_set"]:
            violations.append("seed_set must be a non-empty list")
        if config["max_epochs"] < config["minimum_epochs"]:
            violations.append("max_epochs must be >= minimum_epochs")
        if "w" in config["modeled_features"]:
            violations.append("physical weight 'w' must never be a modeled feature")
        if "utility_multiplier" in config or "utility_weight" in config:
            violations.append("no utility multiplier may appear in a D9 training config")
        eval_policy = config.get("evaluation_policy", {})
        if "quick_validation_budget" not in eval_policy or "final_candidate_budget" not in eval_policy:
            violations.append(
                "evaluation_policy must define both quick_validation_budget and final_candidate_budget"
            )
    return violations


def config_hash(config: Dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def candidates_by_id(training_config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {c["candidate_id"]: c for c in training_config["candidates"]}
