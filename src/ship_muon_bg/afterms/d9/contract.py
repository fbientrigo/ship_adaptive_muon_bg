"""D9 training-contract identity (§10).

``semantic_training_hash`` is the resume/compatibility key. It is deliberately
NOT the full git HEAD: a commit that touches an unrelated file (docs, a
different module) must not force every in-flight run to be treated as
incompatible and refused resume, while a commit that changes a module this
run actually depends on (the model family, the preprocessing contract, the
weighted-objective estimator, this training loop itself) MUST invalidate
resume. So the hash is built from the semantic training fingerprint --
candidate config + preprocessing contract + dataset/shard identities +
modeled feature order + target measure + weighting estimator name +
optimizer/training settings + content hashes of the specific source modules
those choices depend on -- with the producer git commit recorded alongside it
for provenance, never substituted for it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Sequence

# Modules whose source content directly determines training semantics for a
# D9 run. Adding a new dependency here is a deliberate, reviewed decision --
# this list is not auto-derived from imports, so that dependents outside the
# training-relevant surface (e.g. plotting, reporting) never force spurious
# resume invalidation.
TRAINING_RELEVANT_MODULES: Sequence[str] = (
    "src/ship_muon_bg/afterms/preprocessing.py",
    "src/ship_muon_bg/afterms/log1p_pz.py",
    "src/ship_muon_bg/afterms/d9/weighted_objective.py",
    "src/ship_muon_bg/afterms/d9/runner.py",
    "src/ship_muon_bg/afterms/d9/checkpoint.py",
    "Nflow/torch_models/affine_coupling.py",
)


def _canonical_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def module_source_fingerprint(repo_root: Path, relative_paths: Sequence[str] = TRAINING_RELEVANT_MODULES) -> Dict[str, str]:
    """sha256 of each training-relevant module's current source bytes."""

    fingerprints = {}
    for rel in relative_paths:
        path = Path(repo_root, rel)
        fingerprints[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return fingerprints


def semantic_training_hash(
    *,
    candidate_config: Dict[str, Any],
    preprocessing_contract: Dict[str, Any],
    dataset_identity: Dict[str, Any],
    modeled_feature_order: Sequence[str],
    target_measure: str,
    weighting_estimator_version: str,
    optimizer_settings: Dict[str, Any],
    module_fingerprints: Dict[str, str],
) -> str:
    """The single hash resume compatibility is keyed on.

    Every argument here is a semantic/content fact, never a raw git ref --
    this is what lets an unrelated commit leave the hash unchanged while a
    relevant code or config change alters it (required tests 19-21).
    """

    payload = {
        "candidate_config": candidate_config,
        "preprocessing_contract": preprocessing_contract,
        "dataset_identity": dataset_identity,
        "modeled_feature_order": list(modeled_feature_order),
        "target_measure": target_measure,
        "weighting_estimator_version": weighting_estimator_version,
        "optimizer_settings": optimizer_settings,
        "module_fingerprints": module_fingerprints,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def semantic_training_hash_from_repo(
    repo_root: Path,
    *,
    candidate_config: Dict[str, Any],
    preprocessing_contract: Dict[str, Any],
    dataset_identity: Dict[str, Any],
    modeled_feature_order: Sequence[str],
    target_measure: str,
    weighting_estimator_version: str,
    optimizer_settings: Dict[str, Any],
) -> str:
    fingerprints = module_source_fingerprint(repo_root)
    return semantic_training_hash(
        candidate_config=candidate_config,
        preprocessing_contract=preprocessing_contract,
        dataset_identity=dataset_identity,
        modeled_feature_order=modeled_feature_order,
        target_measure=target_measure,
        weighting_estimator_version=weighting_estimator_version,
        optimizer_settings=optimizer_settings,
        module_fingerprints=fingerprints,
    )


def execution_policy_hash(
    *,
    minimum_epochs: int,
    early_stopping_patience: int,
    checkpoint_policy: Dict[str, Any],
    scheduler_config: Any = None,
    validation_frequency: str = "every_epoch",
) -> str:
    """Resume-gating hash for training-DURATION/scheduling knobs (deliberately
    excludes max_epochs, which has its own explicit-extension contract in
    runner.py -- see MaxEpochsExtensionError)."""
    payload = {
        "minimum_epochs": minimum_epochs,
        "early_stopping_patience": early_stopping_patience,
        "checkpoint_policy": checkpoint_policy,
        "scheduler_config": scheduler_config,
        "validation_frequency": validation_frequency,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def evaluation_policy_hash(*, evaluation_policy: Dict[str, Any]) -> str:
    """Descriptive hash of evaluation budgets/settings. NEVER used to gate
    checkpoint/resume compatibility -- changing evaluation policy must not
    invalidate a trained model. Every evaluation output must record which
    evaluation_policy_hash produced it (see evaluate.py)."""
    payload = {"evaluation_policy": evaluation_policy}
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
