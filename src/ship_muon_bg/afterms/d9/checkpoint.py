"""Self-describing D9 checkpoint bundles (§9) and atomic writes.

Each run writes three named files: ``best_checkpoint.pt``,
``final_checkpoint.pt``, ``last_resumable_checkpoint.pt``. Every bundle is a
single ``torch.save``-d dict containing everything needed to reload and
verify compatibility without any out-of-band context. ``best`` and ``final``
are never the same object identity-checked-in as one another: a final
checkpoint is never mislabeled as best.
"""

from __future__ import annotations

import os
import platform
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "d9_checkpoint_bundle_v1"

SCOPE_BEST = "best"
SCOPE_FINAL = "final"
SCOPE_LAST_RESUMABLE = "last_resumable"
VALID_SCOPES = (SCOPE_BEST, SCOPE_FINAL, SCOPE_LAST_RESUMABLE)

FILENAME_BY_SCOPE = {
    SCOPE_BEST: "best_checkpoint.pt",
    SCOPE_FINAL: "final_checkpoint.pt",
    SCOPE_LAST_RESUMABLE: "last_resumable_checkpoint.pt",
}

# Fields checked for exact compatibility before a resume/reload is permitted.
# Deliberately excludes epoch/history/optimizer state (those are expected to
# differ across checkpoints of the *same* run) and excludes producer_git_commit
# (recorded for provenance but never used to gate compatibility, per §10).
_COMPATIBILITY_FIELDS = (
    "schema_version",
    "candidate_id",
    "architecture_config",
    "feature_order",
    "pdg_policy",
    "preprocessing_hash",
    "target_measure",
    "weighting_policy",
    "weighting_estimator_version",
    "seed",
    "semantic_training_hash",
    "execution_policy_hash",
)


class CheckpointCompatibilityError(RuntimeError):
    """A checkpoint bundle does not match the expected training identity."""


def capture_runtime_metadata() -> Dict[str, Any]:
    metadata = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch

        # str(...): torch.__version__ is a `TorchVersion` subclass of str, not
        # a plain str, which `torch.load(weights_only=True)` refuses to
        # unpickle as an unrecognized global. Casting here keeps the bundle
        # entirely within weights_only's safelist (tensors, dict, list, str,
        # int, float, bool, None).
        metadata["torch_version"] = str(torch.__version__)
        metadata["cuda_available"] = bool(torch.cuda.is_available())
        metadata["cuda_version"] = torch.version.cuda if torch.cuda.is_available() else None
    except Exception:
        metadata["torch_version"] = None
        metadata["cuda_available"] = False
        metadata["cuda_version"] = None
    return metadata


def build_bundle(
    *,
    campaign_id: str,
    run_id: str,
    candidate_id: str,
    seed: int,
    epoch: int,
    checkpoint_scope: str,
    model_family: str,
    architecture_config: Dict[str, Any],
    feature_order: List[str],
    pdg_policy: str,
    preprocessing_reference: str,
    preprocessing_hash: str,
    target_measure: str,
    weighting_policy: str,
    weighting_estimator_version: str,
    model_state_dict: Dict[str, Any],
    optimizer_state_dict: Optional[Dict[str, Any]],
    scheduler_state_dict: Optional[Dict[str, Any]],
    training_history: List[Dict[str, Any]],
    best_validation_metric: Optional[float],
    best_validation_epoch: Optional[int],
    rng_states: Dict[str, Any],
    dataset_hash: str,
    split_hashes: Dict[str, str],
    shard_manifest_hash: str,
    training_config_hash: str,
    semantic_training_hash: str,
    execution_policy_hash: str,
    evaluation_policy_hash: str,
    max_epochs: int,
    training_code_fingerprint: Dict[str, str],
    producer_git_commit: str,
    execution_policy_revision: int = 0,
) -> Dict[str, Any]:
    if checkpoint_scope not in VALID_SCOPES:
        raise ValueError(f"unknown checkpoint_scope {checkpoint_scope!r}, expected one of {VALID_SCOPES}")

    return {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "run_id": run_id,
        "candidate_id": candidate_id,
        "seed": int(seed),
        "epoch": int(epoch),
        "checkpoint_scope": checkpoint_scope,
        "model_family": model_family,
        "architecture_config": architecture_config,
        "feature_order": list(feature_order),
        "pdg_policy": pdg_policy,
        "preprocessing_reference": preprocessing_reference,
        "preprocessing_hash": preprocessing_hash,
        "target_measure": target_measure,
        "weighting_policy": weighting_policy,
        "weighting_estimator_version": weighting_estimator_version,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "scheduler_state_dict": scheduler_state_dict,
        "training_history": training_history,
        "best_validation_metric": best_validation_metric,
        "best_validation_epoch": best_validation_epoch,
        "rng_states": rng_states,
        "dataset_hash": dataset_hash,
        "split_hashes": split_hashes,
        "shard_manifest_hash": shard_manifest_hash,
        "training_config_hash": training_config_hash,
        "semantic_training_hash": semantic_training_hash,
        "execution_policy_hash": execution_policy_hash,
        "evaluation_policy_hash": evaluation_policy_hash,
        "max_epochs": int(max_epochs),
        "execution_policy_revision": int(execution_policy_revision),
        "training_code_fingerprint": training_code_fingerprint,
        "producer_git_commit": producer_git_commit,
        "runtime_metadata": capture_runtime_metadata(),
    }


def save_bundle(directory: Path, scope: str, bundle: Dict[str, Any]) -> Path:
    """Atomically write ``bundle`` to ``directory/<scope filename>``.

    Writes to a temp file in the same directory, then ``os.replace`` swaps it
    into place -- either the old file or the fully-written new file is
    observable, never a partially-written one.
    """

    import torch

    if scope not in VALID_SCOPES:
        raise ValueError(f"unknown checkpoint_scope {scope!r}")
    if bundle.get("checkpoint_scope") != scope:
        raise ValueError(
            f"bundle.checkpoint_scope={bundle.get('checkpoint_scope')!r} does not match "
            f"requested scope {scope!r} -- refusing to write a mislabeled checkpoint"
        )

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / FILENAME_BY_SCOPE[scope]

    fd, tmp_name = tempfile.mkstemp(dir=str(directory), prefix=f".tmp_{scope}_", suffix=".pt")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        torch.save(bundle, tmp_path)
        os.replace(tmp_path, target)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    return target


def load_bundle(path: Path) -> Dict[str, Any]:
    """Load a bundle written by ``save_bundle``.

    ``weights_only=True``: every bundle field is a tensor, dict, list, or
    JSON-safe primitive (see ``build_bundle``) -- there is never a reason to
    unpickle arbitrary Python objects from a checkpoint file, even one this
    codebase wrote itself.
    """

    import torch

    return torch.load(Path(path), map_location="cpu", weights_only=True)


def verify_compatibility(bundle: Dict[str, Any], expected: Dict[str, Any]) -> List[str]:
    """Return a list of mismatch descriptions (empty if fully compatible).

    ``expected`` must supply every key in ``_COMPATIBILITY_FIELDS`` that the
    caller wants checked; keys absent from ``expected`` are skipped.
    """

    violations = []
    for field in _COMPATIBILITY_FIELDS:
        if field not in expected:
            continue
        actual = bundle.get(field)
        want = expected[field]
        if actual != want:
            violations.append(f"{field}: bundle has {actual!r}, expected {want!r}")
    return violations


def load_and_verify(path: Path, expected: Dict[str, Any]) -> Dict[str, Any]:
    bundle = load_bundle(path)
    violations = verify_compatibility(bundle, expected)
    if violations:
        raise CheckpointCompatibilityError(
            f"checkpoint at {path} is incompatible with the expected training identity: {violations}"
        )
    return bundle
