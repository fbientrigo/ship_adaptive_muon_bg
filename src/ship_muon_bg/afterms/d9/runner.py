"""D9 one-candidate/one-seed training runner (§11-12).

Structural test-set isolation (§12, required tests 22/23): ``train_candidate_seed``
has NO parameter that accepts test-split data at all -- it is not merely
"unused", the function signature cannot receive it. Evaluating the test split
is only possible through ``evaluate_candidate_seed`` in ``evaluate.py``, called
strictly after training completes.

This module implements its own minibatch loop (it does not call
``AffineCouplingFlow.fit()`` / ``Nflow.torch_models.trainer.train_flow``)
because that trainer's weighted-loss form is the historical
self-normalized-minibatch ratio D8 flagged and D9 must not silently inherit
(§7); the unweighted path here is otherwise the same manual-loop shape as the
frozen D7 producer script.
"""

from __future__ import annotations

import inspect
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ship_muon_bg.afterms.preprocessing import PreprocessingPipeline
from ship_muon_bg.data_contracts import dataset_hash, schema

from . import checkpoint as ckpt
from . import contract as d9contract
from . import sampling as d9sampling
from . import weighted_objective as wo

STATUS_PLANNED = "planned"
STATUS_RUNNING = "running"
STATUS_INTERRUPTED = "interrupted"
STATUS_COMPLETED = "completed"
STATUS_FAILED_TECHNICAL = "failed_technical"
STATUS_EXCLUDED_CONTRACT_MISMATCH = "excluded_contract_mismatch"


class TrainingInterrupted(RuntimeError):
    """Raised internally to unwind cleanly after a KeyboardInterrupt."""


class MaxEpochsExtensionError(RuntimeError):
    """max_epochs changed without going through the explicit extend_max_epochs
    contract, or the extension did not strictly increase the value."""


def pdg_filter(raw: np.ndarray, pdg_value: Optional[int]) -> np.ndarray:
    if pdg_value is None:
        return raw
    mask = np.rint(raw[:, schema.COLUMN_INDEX["id"]]) == pdg_value
    return raw[mask]


def load_concatenated_shards(shard_dir: Path, shard_names: List[str]) -> np.ndarray:
    arrays = [np.load(Path(shard_dir, f"{name}.npy")) for name in shard_names]
    return np.concatenate(arrays, axis=0) if len(arrays) > 1 else arrays[0]


def run_id_for(candidate_id: str, seed: int) -> str:
    return f"{candidate_id}__seed{seed}"


def run_directory(artifact_root: Path, candidate_id: str, seed: int) -> Path:
    return Path(artifact_root, "runs", candidate_id, f"seed_{seed}")


def _write_status(run_dir: Path, payload: Dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp = run_dir / ".status.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(run_dir / "status.json")


def _append_execution_policy_event(run_dir: Path, event: Dict[str, Any]) -> None:
    """Append an immutable record to this run's execution-policy event log.

    Never rewrites or removes prior entries -- only appends, so a max-epochs
    extension or an explicit execution-policy revision leaves a permanent,
    reviewable trail (§3.3/§3.2).
    """

    from datetime import datetime, timezone

    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "execution_policy_log.json"
    existing = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else []
    existing.append(dict(event, recorded_at=datetime.now(timezone.utc).isoformat()))
    tmp = run_dir / ".execution_policy_log.json.tmp"
    tmp.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")
    tmp.replace(log_path)


def _preprocessing_contract(pipeline: PreprocessingPipeline, training_split_hash: str, pdg_policy: str) -> Dict[str, Any]:
    state = pipeline.to_dict()
    serialization_hash = __import__("hashlib").sha256(
        json.dumps(state, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "preprocessing_name": pipeline.variant_id,
        "preprocessing_version": "v0",
        "feature_order": ["px", "py", "pz", "x", "y"],
        "fitted_state": state,
        "training_split_hash": training_split_hash,
        "training_PDG_policy": pdg_policy,
        "serialization_hash": serialization_hash,
    }


def _weight_column(raw: np.ndarray) -> np.ndarray:
    return raw[:, schema.COLUMN_INDEX["w"]].astype(np.float64)


def train_candidate_seed(
    candidate_config: Dict[str, Any],
    seed: int,
    train_raw: np.ndarray,
    validation_raw: np.ndarray,
    *,
    artifact_root: Path,
    repo_root: Path,
    device: str = "cpu",
    resume: bool = False,
    extend_max_epochs: Optional[int] = None,
    execution_policy_revision_reason: Optional[str] = None,
    interrupt_flag: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """Train exactly one (candidate, seed). Never touches test data -- there is
    no parameter through which it could.

    ``interrupt_flag``: an optional zero-arg callable polled once per epoch;
    if it returns True, the run stops as cleanly as a KeyboardInterrupt would
    (used by tests, since raising real SIGINT mid-test is impractical).
    """

    import torch
    from Nflow.registry import create_density_estimator

    candidate_id = candidate_config["candidate_id"]
    run_id = run_id_for(candidate_id, seed)
    run_dir = run_directory(artifact_root, candidate_id, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = run_dir / "checkpoints"
    histories_dir = run_dir / "histories"
    histories_dir.mkdir(parents=True, exist_ok=True)

    pdg_value = candidate_config["pdg_value"]
    train_filtered = pdg_filter(train_raw, pdg_value)
    validation_filtered = pdg_filter(validation_raw, pdg_value)

    train_hash = dataset_hash(train_filtered)
    validation_hash = dataset_hash(validation_filtered)

    pipeline = PreprocessingPipeline(candidate_config["preprocessing_name"], seed=seed)
    pipeline.fit(train_filtered)
    preprocessing_contract = _preprocessing_contract(pipeline, train_hash, candidate_config["pdg_policy"])
    preprocessing_hash = preprocessing_contract["serialization_hash"]

    optimizer_settings = {
        "optimizer": candidate_config["optimizer"],
        "learning_rate": candidate_config["learning_rate"],
        "batch_size": candidate_config["batch_size"],
        "weight_decay": candidate_config["weight_decay"],
        "gradient_clipping": candidate_config["gradient_clipping"],
    }
    dataset_identity = {
        "shard_dir": candidate_config.get("shard_dir"),
        "train_shards": candidate_config["train_shards"],
        "validation_shards": candidate_config["validation_shards"],
        "test_shards": candidate_config["test_shards"],
        "train_split_hash": train_hash,
        "validation_split_hash": validation_hash,
    }
    semantic_training_hash = d9contract.semantic_training_hash_from_repo(
        repo_root,
        candidate_config=_identity_relevant_config(candidate_config),
        preprocessing_contract={"preprocessing_name": preprocessing_contract["preprocessing_name"], "serialization_hash": preprocessing_hash},
        dataset_identity=dataset_identity,
        modeled_feature_order=candidate_config["feature_order"],
        target_measure=candidate_config["target_measure"],
        weighting_estimator_version=candidate_config["weighting_estimator_version"],
        optimizer_settings=optimizer_settings,
    )
    execution_policy_hash_value = d9contract.execution_policy_hash(
        minimum_epochs=candidate_config["minimum_epochs"],
        early_stopping_patience=candidate_config["early_stopping_patience"],
        checkpoint_policy=candidate_config["checkpoint_policy"],
    )
    evaluation_policy_hash_value = d9contract.evaluation_policy_hash(
        evaluation_policy=candidate_config["evaluation_policy"],
    )

    from . import training_config as tc

    training_config_hash = tc.config_hash(candidate_config)
    module_fingerprint = d9contract.module_source_fingerprint(repo_root)

    minimum_epochs = candidate_config["minimum_epochs"]
    patience = candidate_config["early_stopping_patience"]

    resumable_path = checkpoints_dir / ckpt.FILENAME_BY_SCOPE[ckpt.SCOPE_LAST_RESUMABLE]
    start_epoch = 0
    history: List[Dict[str, Any]] = []
    best_val = None
    best_epoch = None
    module = None
    optimizer = None
    estimator = None

    declared_max_epochs = candidate_config["max_epochs"]
    execution_policy_revision = 0
    effective_max_epochs = extend_max_epochs or declared_max_epochs

    if resume and resumable_path.exists():
        bundle = ckpt.load_bundle(resumable_path)

        expected = {
            "semantic_training_hash": semantic_training_hash,
            "sampling_contract_version": d9sampling.SAMPLING_CONTRACT_VERSION,
        }
        prior_execution_policy_hash = bundle.get("execution_policy_hash")
        execution_policy_changed = (
            prior_execution_policy_hash is not None
            and prior_execution_policy_hash != execution_policy_hash_value
        )
        if execution_policy_changed and not execution_policy_revision_reason:
            expected["execution_policy_hash"] = execution_policy_hash_value

        violations = ckpt.verify_compatibility(bundle, expected)
        if violations:
            raise ckpt.CheckpointCompatibilityError(
                f"checkpoint at {resumable_path} is incompatible with the expected training identity: {violations}"
            )

        prior_max_epochs = bundle.get("max_epochs")
        execution_policy_revision = bundle.get("execution_policy_revision", 0)
        if prior_max_epochs is not None:
            if extend_max_epochs is not None:
                if extend_max_epochs <= prior_max_epochs:
                    raise MaxEpochsExtensionError(
                        f"extend_max_epochs={extend_max_epochs} must be strictly greater than the "
                        f"previously recorded max_epochs={prior_max_epochs}"
                    )
                effective_max_epochs = extend_max_epochs
                _append_execution_policy_event(run_dir, {
                    "event": "max_epochs_extended",
                    "previous_max_epochs": prior_max_epochs,
                    "new_max_epochs": extend_max_epochs,
                })
            else:
                if declared_max_epochs != prior_max_epochs:
                    raise MaxEpochsExtensionError(
                        f"candidate_config max_epochs={declared_max_epochs} does not match the "
                        f"previously recorded max_epochs={prior_max_epochs} for this run, and no "
                        "extend_max_epochs was given. Pass extend_max_epochs=<new_value> to explicitly "
                        "extend an existing run, or restore the previous max_epochs value."
                    )
                effective_max_epochs = prior_max_epochs

        if execution_policy_changed and execution_policy_revision_reason:
            execution_policy_revision += 1
            _append_execution_policy_event(run_dir, {
                "event": "execution_policy_revised",
                "previous_execution_policy_hash": prior_execution_policy_hash,
                "new_execution_policy_hash": execution_policy_hash_value,
                "reason": execution_policy_revision_reason,
            })

        estimator = create_density_estimator(
            {"family": candidate_config["model_family"], "params": _architecture_params(candidate_config)},
            dimension=5, device=device,
        )
        estimator._build_module(seed=int(seed))
        module = estimator._module
        module.load_state_dict(bundle["model_state_dict"])
        optimizer = torch.optim.Adam(
            module.parameters(), lr=candidate_config["learning_rate"], weight_decay=candidate_config["weight_decay"],
        )
        if bundle.get("optimizer_state_dict"):
            optimizer.load_state_dict(bundle["optimizer_state_dict"])
        start_epoch = bundle["epoch"]
        history = list(bundle["training_history"])
        best_val = bundle["best_validation_metric"]
        best_epoch = bundle["best_validation_epoch"]

    if module is None:
        estimator = create_density_estimator(
            {"family": candidate_config["model_family"], "params": _architecture_params(candidate_config)},
            dimension=5, device=device,
        )
        estimator._build_module(seed=int(seed))
        module = estimator._module
        optimizer = torch.optim.Adam(
            module.parameters(), lr=candidate_config["learning_rate"], weight_decay=candidate_config["weight_decay"],
        )

    torch_dtype = getattr(torch, candidate_config["dtype"])
    train_norm = pipeline.transform(train_filtered)
    val_norm = pipeline.transform(validation_filtered)
    n_train = train_norm.shape[0]
    batch_size = min(candidate_config["batch_size"], n_train)

    weighted = candidate_config["weighting_policy"] == "production_weighted"
    if weighted:
        w_train = _weight_column(train_filtered)
        w_val = _weight_column(validation_filtered)
        w_train_total = wo.split_weight_total(w_train)
    else:
        w_train = np.ones(n_train, dtype=np.float64)
        w_val = np.ones(val_norm.shape[0], dtype=np.float64)
        w_train_total = float(n_train)

    t_train = torch.tensor(train_norm, dtype=torch_dtype, device=device)
    t_val = torch.tensor(val_norm, dtype=torch_dtype, device=device)
    t_w_train = torch.tensor(w_train, dtype=torch_dtype, device=device)
    t_w_val = torch.tensor(w_val, dtype=torch_dtype, device=device)

    gen = torch.Generator(device=device)

    _write_status(run_dir, {"status": STATUS_RUNNING, "run_id": run_id, "candidate_id": candidate_id, "seed": seed})

    stale = 0
    epoch = start_epoch
    interrupted = False
    try:
        for epoch in range(start_epoch + 1, effective_max_epochs + 1):
            if interrupt_flag is not None and interrupt_flag():
                raise TrainingInterrupted()
            module.train()
            # Reseeded every epoch from (seed, epoch) alone (§Gate B.1 sampling
            # contract) -- NOT a continuous draw from one process-lifetime
            # generator -- so epoch N's minibatch order is identical whether
            # this process trained epochs 1..N-1 first or was resumed fresh at
            # epoch N.
            gen.manual_seed(d9sampling.epoch_permutation_seed(seed, epoch))
            perm = torch.randperm(n_train, generator=gen, device=device)
            wall_start = time.perf_counter()
            train_loss_sum = 0.0
            n_batches = 0
            grad_norm_last = None
            for start in range(0, n_train, batch_size):
                idx = perm[start:start + batch_size]
                optimizer.zero_grad()
                nll_batch = -module.log_prob(t_train[idx])
                if weighted:
                    loss = wo.weighted_batch_loss_torch(
                        nll_batch, t_w_train[idx], n_train=n_train, w_train_total=w_train_total,
                    )
                else:
                    loss = nll_batch.mean()
                loss.backward()
                if candidate_config["gradient_clipping"]:
                    grad_norm_last = float(
                        torch.nn.utils.clip_grad_norm_(module.parameters(), candidate_config["gradient_clipping"])
                    )
                optimizer.step()
                train_loss_sum += float(loss.item())
                n_batches += 1
            train_loss = train_loss_sum / max(n_batches, 1)
            wall_time = time.perf_counter() - wall_start

            module.eval()
            with torch.no_grad():
                val_nll_full = (-module.log_prob(t_val)).detach().cpu().numpy().astype(np.float64)
            if weighted:
                validation_loss = wo.weighted_validation_nll(val_nll_full, w_val)
            else:
                validation_loss = wo.unweighted_validation_nll(val_nll_full)

            finite_loss = bool(np.isfinite(train_loss) and np.isfinite(validation_loss))
            improved = best_val is None or validation_loss < best_val
            if improved:
                best_val = validation_loss
                best_epoch = epoch
                stale = 0
            else:
                stale += 1

            epoch_record = {
                "epoch": epoch,
                "train_feature_nll": train_loss,
                "validation_feature_nll": validation_loss,
                "weighted_or_unweighted_estimator": (
                    wo.ESTIMATOR_NAME if weighted else "row_empirical_unweighted_mean"
                ),
                "learning_rate": candidate_config["learning_rate"],
                "gradient_norm": grad_norm_last,
                "finite_loss": finite_loss,
                "wall_time_seconds": wall_time,
                "gpu_peak_memory_bytes": (
                    int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
                ),
                "best_so_far": improved,
                "checkpoint_written": True,
            }
            history.append(epoch_record)

            common_bundle_kwargs = dict(
                campaign_id=candidate_config["campaign_id"],
                run_id=run_id,
                candidate_id=candidate_id,
                seed=seed,
                model_family=candidate_config["model_family"],
                architecture_config=_architecture_params(candidate_config),
                feature_order=candidate_config["feature_order"],
                pdg_policy=candidate_config["pdg_policy"],
                preprocessing_reference="preprocessing.json",
                preprocessing_hash=preprocessing_hash,
                target_measure=candidate_config["target_measure"],
                weighting_policy=candidate_config["weighting_policy"],
                weighting_estimator_version=candidate_config["weighting_estimator_version"],
                model_state_dict=module.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=None,
                training_history=history,
                best_validation_metric=best_val,
                best_validation_epoch=best_epoch,
                rng_states={"torch_manual_seed": int(seed)},
                sampling_contract_version=d9sampling.SAMPLING_CONTRACT_VERSION,
                dataset_hash=train_hash,
                split_hashes={"train": train_hash, "validation": validation_hash},
                shard_manifest_hash=train_hash,
                training_config_hash=training_config_hash,
                semantic_training_hash=semantic_training_hash,
                execution_policy_hash=execution_policy_hash_value,
                evaluation_policy_hash=evaluation_policy_hash_value,
                max_epochs=effective_max_epochs,
                execution_policy_revision=execution_policy_revision,
                training_code_fingerprint=module_fingerprint,
                producer_git_commit=_current_git_commit(repo_root),
            )
            ckpt.save_bundle(
                checkpoints_dir, ckpt.SCOPE_LAST_RESUMABLE,
                ckpt.build_bundle(epoch=epoch, checkpoint_scope=ckpt.SCOPE_LAST_RESUMABLE, **common_bundle_kwargs),
            )
            if improved:
                ckpt.save_bundle(
                    checkpoints_dir, ckpt.SCOPE_BEST,
                    ckpt.build_bundle(epoch=epoch, checkpoint_scope=ckpt.SCOPE_BEST, **common_bundle_kwargs),
                )
            (histories_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

            if epoch >= minimum_epochs and stale >= patience:
                break
    except (TrainingInterrupted, KeyboardInterrupt):
        interrupted = True

    if interrupted:
        _write_status(run_dir, {
            "status": STATUS_INTERRUPTED, "run_id": run_id, "candidate_id": candidate_id, "seed": seed,
            "last_completed_epoch": epoch - 1 if epoch == start_epoch + 1 else epoch,
        })
        return {"status": STATUS_INTERRUPTED, "run_id": run_id, "last_epoch": epoch}

    final_bundle = ckpt.build_bundle(
        campaign_id=candidate_config["campaign_id"], run_id=run_id, candidate_id=candidate_id, seed=seed,
        epoch=epoch, checkpoint_scope=ckpt.SCOPE_FINAL,
        model_family=candidate_config["model_family"], architecture_config=_architecture_params(candidate_config),
        feature_order=candidate_config["feature_order"], pdg_policy=candidate_config["pdg_policy"],
        preprocessing_reference="preprocessing.json", preprocessing_hash=preprocessing_hash,
        target_measure=candidate_config["target_measure"], weighting_policy=candidate_config["weighting_policy"],
        weighting_estimator_version=candidate_config["weighting_estimator_version"],
        model_state_dict=module.state_dict(), optimizer_state_dict=optimizer.state_dict(),
        scheduler_state_dict=None, training_history=history, best_validation_metric=best_val,
        best_validation_epoch=best_epoch, rng_states={"torch_manual_seed": int(seed)},
        sampling_contract_version=d9sampling.SAMPLING_CONTRACT_VERSION,
        dataset_hash=train_hash, split_hashes={"train": train_hash, "validation": validation_hash},
        shard_manifest_hash=train_hash, training_config_hash=training_config_hash,
        semantic_training_hash=semantic_training_hash,
        execution_policy_hash=execution_policy_hash_value,
        evaluation_policy_hash=evaluation_policy_hash_value,
        max_epochs=effective_max_epochs,
        execution_policy_revision=execution_policy_revision,
        training_code_fingerprint=module_fingerprint,
        producer_git_commit=_current_git_commit(repo_root),
    )
    ckpt.save_bundle(checkpoints_dir, ckpt.SCOPE_FINAL, final_bundle)

    preprocessing_dir = run_dir / "preprocessing"
    preprocessing_dir.mkdir(parents=True, exist_ok=True)
    (preprocessing_dir / "preprocessing.json").write_text(
        json.dumps(preprocessing_contract, indent=2, default=str), encoding="utf-8"
    )
    (run_dir / "training_config.json").write_text(json.dumps(candidate_config, indent=2), encoding="utf-8")

    _write_status(run_dir, {
        "status": STATUS_COMPLETED, "run_id": run_id, "candidate_id": candidate_id, "seed": seed,
        "best_validation_metric": best_val, "best_validation_epoch": best_epoch, "final_epoch": epoch,
        "semantic_training_hash": semantic_training_hash,
        "execution_policy_hash": execution_policy_hash_value,
        "evaluation_policy_hash": evaluation_policy_hash_value,
        "max_epochs": effective_max_epochs,
    })
    return {
        "status": STATUS_COMPLETED, "run_id": run_id, "best_validation_metric": best_val,
        "best_validation_epoch": best_epoch, "final_epoch": epoch,
        "semantic_training_hash": semantic_training_hash,
        "execution_policy_hash": execution_policy_hash_value,
        "evaluation_policy_hash": evaluation_policy_hash_value,
        "max_epochs": effective_max_epochs,
    }


# Fields that are training-duration/scheduling knobs, not part of the
# semantic training identity. They are not silently dropped: minimum_epochs,
# early_stopping_patience and checkpoint_policy feed execution_policy_hash;
# evaluation_policy feeds evaluation_policy_hash; max_epochs has its own
# explicit-extension contract (see MaxEpochsExtensionError). seed_set is
# excluded from all three hashes because it declares which seeds are
# *permitted* for this candidate, not this run's actual behavior -- the
# actual seed a run used is tracked as its own checkpoint field (already
# outside all three hashes) and disambiguates runs via the seed_<seed>/
# directory path, not the hash.
_IDENTITY_IRRELEVANT_FIELDS = (
    "max_epochs", "minimum_epochs", "early_stopping_patience",
    "checkpoint_policy", "evaluation_policy", "seed_set",
)


def _identity_relevant_config(candidate_config: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in candidate_config.items() if k not in _IDENTITY_IRRELEVANT_FIELDS}


def _architecture_params(candidate_config: Dict[str, Any]) -> Dict[str, Any]:
    arch = dict(candidate_config["architecture"])
    arch.pop("capacity_label", None)
    arch["max_epochs"] = candidate_config["max_epochs"]
    arch["batch_size"] = candidate_config["batch_size"]
    return arch


def _current_git_commit(repo_root: Path) -> str:
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True).strip()
    except Exception:
        return "unknown"


# Structural proof (required tests 22/23): this function's signature has no
# test-data parameter, so no code path through it can read a test shard.
assert "test_raw" not in inspect.signature(train_candidate_seed).parameters
assert "test_data" not in inspect.signature(train_candidate_seed).parameters
