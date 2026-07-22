"""Explicit, isolated, read-only compatibility adapter for the frozen D7
``afterms_nightly_v1`` campaign (D8 spec §4).

Every historical-format fact this module encodes (checkpoint file naming,
hash method, feature order, architecture literals, PDG dispatch) was verified
against the campaign's *producer* commit -- ``f9d8246`` -- not the current
HEAD, because the reusable pipeline changed checkpoint/preprocessing
persistence after this campaign ran (HEAD 08ee572/56355d0). Nothing here may
be inferred from current-HEAD behavior or from tensor shapes.

This is the only place the legacy contract may live (D8 spec §4, §4.2): the
generic D7 checkpoint loaders (``AffineCouplingFlow.load`` etc.) are not
taught this format.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ship_muon_bg.data_contracts import schema
from ship_muon_bg.afterms.preprocessing import PreprocessingPipeline

# --- Frozen campaign identity (verified against artifacts on disk) --------

PRODUCER_GIT_COMMIT = "f9d8246f281a0e80a8583f8c70b4a0c2e9fc0870"

JOB_NAMES: Tuple[str, ...] = (
    "00_environment_and_dataset_smoke",
    "01_build_afterms_shards",
    "02_validate_afterms_shards",
    "03_preprocessing_roundtrip_and_plots",
    "04_legacy_available_code_realnvp_quantile",
    "05_affine_preprocessing_ab_pdg13",
    "06_affine_preprocessing_ab_pdg_minus13",
    "07_affine_weight_ab_pdg13",
    "08_affine_weight_ab_pdg_minus13",
    "09_affine_capacity_smoke_pdg13",
    "10_gaussian_controls_pdg13",
    "11_gaussian_controls_pdg_minus13",
    "12_memory_release_repeat_smoke",
    "13_build_nightly_report",
)

# Jobs 04-12: the only jobs that produced model runs consumed by the registry.
NEURAL_JOB_NAMES: Tuple[str, ...] = JOB_NAMES[4:13]

TRAINING_SEED = 20260720
GENERATION_SEED = 20260720

# Every job 04-12 trains/validates/tests on shard index 0 only
# (`run_neural_training_subprocess` hardcodes `train_shard_000.npy` etc.) --
# this is the single split identity shared by every run in the registry.
SPLIT_SHARD_NAMES = {
    "train": "train_shard_000",
    "validation": "validation_shard_000",
    "test": "test_shard_000",
}

RAW_PKL_RELPATH = ("data", "raw", "nflow_releases", "muonsFullMC_afterMS.pkl")

MODELED_FEATURES_5D: Tuple[str, ...] = ("px", "py", "pz", "x", "y")
MODELED_FEATURES_4D: Tuple[str, ...] = ("px", "py", "pz", "E")

# job_name -> (pdg_value_or_None, pdg_policy_label). Mirrors the producer's
# `"pdg13" in job_name` / `"pdg_minus13" in job_name` substring dispatch
# exactly (job_name, not job id, is what the producer script switched on).
JOB_PDG: Dict[str, Tuple[Optional[int], str]] = {
    "04_legacy_available_code_realnvp_quantile": (None, "combined_unfiltered"),
    "05_affine_preprocessing_ab_pdg13": (13, "pdg13"),
    "06_affine_preprocessing_ab_pdg_minus13": (-13, "pdg_minus13"),
    "07_affine_weight_ab_pdg13": (13, "pdg13"),
    "08_affine_weight_ab_pdg_minus13": (-13, "pdg_minus13"),
    "09_affine_capacity_smoke_pdg13": (13, "pdg13"),
    "10_gaussian_controls_pdg13": (13, "pdg13"),
    "11_gaussian_controls_pdg_minus13": (-13, "pdg_minus13"),
    "12_memory_release_repeat_smoke": (None, "combined_unfiltered"),
}

# job_name -> preprocessing variant ids exercised in that job (producer's
# `variants = [...]` branch keyed off job_name substrings).
JOB_VARIANTS: Dict[str, Tuple[str, ...]] = {
    "05_affine_preprocessing_ab_pdg13": (
        "identity_standardized_v0", "quantile_normal_v0", "cartesian_log1p_pz_v0",
    ),
    "06_affine_preprocessing_ab_pdg_minus13": (
        "identity_standardized_v0", "quantile_normal_v0", "cartesian_log1p_pz_v0",
    ),
    "07_affine_weight_ab_pdg13": ("identity_standardized_v0",),
    "08_affine_weight_ab_pdg_minus13": ("identity_standardized_v0",),
    "09_affine_capacity_smoke_pdg13": ("identity_standardized_v0",),
    "10_gaussian_controls_pdg13": ("identity_standardized_v0",),
    "11_gaussian_controls_pdg_minus13": ("identity_standardized_v0",),
    "12_memory_release_repeat_smoke": ("identity_standardized_v0",),
}

# job_name -> (model_names, family_for_each_name).
JOB_MODEL_NAMES: Dict[str, Tuple[str, ...]] = {
    "05_affine_preprocessing_ab_pdg13": ("affine_small",),
    "06_affine_preprocessing_ab_pdg_minus13": ("affine_small",),
    "07_affine_weight_ab_pdg13": ("affine_small",),
    "08_affine_weight_ab_pdg_minus13": ("affine_small",),
    "09_affine_capacity_smoke_pdg13": ("affine_tiny", "affine_small", "affine_medium"),
    "10_gaussian_controls_pdg13": ("diagonal_gaussian", "full_gaussian", "gaussian_mixture"),
    "11_gaussian_controls_pdg_minus13": ("diagonal_gaussian", "full_gaussian", "gaussian_mixture"),
    "12_memory_release_repeat_smoke": ("affine_tiny",),
}

# job_name -> whether that job trains both an unweighted and a weighted
# variant per (preprocessing, model) pair (producer's `weighted = True` jobs).
JOB_WEIGHTED: Dict[str, bool] = {
    "07_affine_weight_ab_pdg13": True,
    "08_affine_weight_ab_pdg_minus13": True,
}

MODEL_FAMILY_OF_NAME: Dict[str, str] = {
    "affine_tiny": "affine_coupling",
    "affine_small": "affine_coupling",
    "affine_medium": "affine_coupling",
    "diagonal_gaussian": "diagonal_gaussian",
    "full_gaussian": "full_gaussian",
    "gaussian_mixture": "gaussian_mixture",
}

# Exact architecture literals from the producer's per-`m_name` `params` dict
# (`scripts/run_afterms_nightly_queue.py` @ f9d8246). `dimension` is always 5
# for these; `device`, `activation`, `max_log_scale`, `dtype`, `mixing_mode`,
# `initializer_mode` were left at `AffineCouplingFlow.__init__` defaults
# (unchanged between f9d8246 and HEAD 56355d0 -- verified via `git diff`).
AFFINE_TRAIN_PARAMS: Dict[str, Dict[str, Any]] = {
    "affine_tiny": {"number_of_blocks": 2, "hidden_width": 32, "hidden_depth": 1, "max_epochs": 5, "batch_size": 256},
    "affine_small": {"number_of_blocks": 4, "hidden_width": 64, "hidden_depth": 2, "max_epochs": 5, "batch_size": 256},
    "affine_medium": {"number_of_blocks": 8, "hidden_width": 128, "hidden_depth": 2, "max_epochs": 5, "batch_size": 256},
}

BASELINE_TRAIN_PARAMS: Dict[str, Dict[str, Any]] = {
    "diagonal_gaussian": {},
    "full_gaussian": {},
    "gaussian_mixture": {"n_components": 4, "n_init": 1},
}

# Job 04's frozen legacy config, read directly from the producer's
# `job_name == "04_legacy_available_code_realnvp_quantile"` branch (no
# `legacy_model_config.json` sidecar exists on disk for this campaign --
# that sidecar was only added by the later commit that introduced
# `_write_preprocessing_sidecar`/config persistence, 08ee572).
JOB04_LEGACY_CONFIG: Dict[str, Any] = {
    "family": "legacy_normalizing_flow",
    "input_dim": 4,
    "hidden_dim": 160,
    "n_layers": 10,
    "init_zero": True,
    "mass_muon": 0.1134289259,
    "feature_order": MODELED_FEATURES_4D,
    "dtype": "float32",
    "quantile_transformer_random_state": TRAINING_SEED,
    "spec_source": "producer f9d8246 scripts/run_afterms_nightly_queue.py::run_neural_training_subprocess, job_name=='04_legacy_available_code_realnvp_quantile' branch",
}


class LegacyContractError(RuntimeError):
    """A frozen artifact fact could not be verified against source."""


def run_label(variant_id: str, model_name: str, weighted: bool) -> str:
    """Reproduce the producer's exact run-label string (job_dir filenames key on this)."""

    suffix = "weighted" if weighted else "unweighted"
    return f"{variant_id}_{model_name}_{suffix}"


# --- §4.1 historical hash normalization -------------------------------------

def recompute_raw_file_sha256(repo_root: Path) -> str:
    """Streamed SHA-256 of the raw pkl bytes on disk today."""

    path = Path(repo_root, *RAW_PKL_RELPATH)
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def legacy_raw_file_sha256_prefix16(nightly_summary: Dict[str, Any]) -> str:
    """The frozen report's truncated raw-file hash.

    Historically named ``dataset_hash`` in ``report/nightly_summary.json`` --
    NOT the same quantity as the shard manifest's same-named field (see
    ``shard_manifest_content_hash``).
    """

    value = nightly_summary.get("dataset_hash")
    if not isinstance(value, str) or len(value) != 16:
        raise LegacyContractError(
            "nightly_summary.json.dataset_hash is not a 16-hex-char legacy raw-file prefix: {!r}".format(value)
        )
    return value


def audit_raw_file_sha256(afterms_audit: Dict[str, Any]) -> str:
    """The frozen dataset audit's full raw-file SHA-256 (``file_sha256``)."""

    value = afterms_audit.get("file_sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise LegacyContractError("afterms_audit.json.file_sha256 is not a 64-hex-char sha256: {!r}".format(value))
    return value


def audit_content_dataset_hash(afterms_audit: Dict[str, Any]) -> str:
    value = afterms_audit.get("content_dataset_hash")
    if not isinstance(value, str) or len(value) != 64:
        raise LegacyContractError(
            "afterms_audit.json.content_dataset_hash is not a 64-hex-char sha256: {!r}".format(value)
        )
    return value


def shard_manifest_content_hash(shard_manifest: Dict[str, Any]) -> str:
    """The shard manifest's content-array hash.

    Historically named ``dataset_hash`` in ``shard_manifest.json`` -- the SAME
    quantity as ``afterms_audit.json``'s ``content_dataset_hash`` field (both
    hash the canonicalized in-memory array), just recorded under a different
    key by a different producer. Never compare this to the raw-file hash.
    """

    value = shard_manifest.get("dataset_hash")
    if not isinstance(value, str) or len(value) != 64:
        raise LegacyContractError("shard_manifest.json.dataset_hash is not a 64-hex-char sha256: {!r}".format(value))
    return value


def verify_raw_file_hash_relation(prefix16: str, recomputed_full_sha256: str) -> bool:
    """§4.1's required relation: legacy prefix16 == recomputed[:16]."""

    return recomputed_full_sha256[:16] == prefix16


# --- PDG-filtered shard access ----------------------------------------------

def load_pdg_filtered_shard(shard_dir: Path, split: str, pdg_value: Optional[int]) -> np.ndarray:
    """Load ``<split>_shard_000.npy`` and apply the same PDG mask the producer applied.

    ``np.rint(...) == pdg_value`` mirrors
    ``run_neural_training_subprocess``'s exact filter; ``pdg_value=None``
    means no filter (the row order producer trained on).
    """

    shard_name = SPLIT_SHARD_NAMES[split]
    array = np.load(Path(shard_dir, f"{shard_name}.npy"))
    if pdg_value is None:
        return array
    mask = np.rint(array[:, schema.COLUMN_INDEX["id"]]) == pdg_value
    return array[mask]


# --- §4.3 historical preprocessing reconstruction ---------------------------

RECONSTRUCTION_SCOPE_MODERN = "deterministically_refit_from_frozen_d7_training_shard"
RECONSTRUCTION_SCOPE_JOB04 = (
    "deterministically_refit_from_frozen_d7_training_shard_sklearn_version_sensitive_subsample"
)


def reconstruct_preprocessing_pipeline(
    variant_id: str, shard_dir: Path, pdg_value: Optional[int], seed: int = TRAINING_SEED
) -> PreprocessingPipeline:
    """Refit the modern ``PreprocessingPipeline`` on the exact frozen train shard + PDG filter.

    Allowed by D8 spec §4.3: this is a deterministic refit from frozen inputs,
    not generative-model training, and its output is scoped as
    ``RECONSTRUCTION_SCOPE_MODERN``, never written back into D7.
    """

    train = load_pdg_filtered_shard(shard_dir, "train", pdg_value)
    pipeline = PreprocessingPipeline(variant_id, seed=seed)
    pipeline.fit(train)
    return pipeline


def job04_legacy_feature_transform(raw_rows: np.ndarray, mass_muon: float = 0.1134289259) -> np.ndarray:
    """px, py, pz, E=sqrt(px^2+py^2+pz^2+m^2) -- job 04's inline feature engineering."""

    px, py, pz = raw_rows[:, 0], raw_rows[:, 1], raw_rows[:, 2]
    energy = np.sqrt(px**2 + py**2 + pz**2 + mass_muon**2)
    return np.column_stack((px, py, pz, energy))


def reconstruct_job04_quantile_transformer(shard_dir: Path, seed: int = TRAINING_SEED):
    """Refit job 04's inline ``QuantileTransformer`` (no PreprocessingPipeline involved).

    Unlike ``quantile_normal_v0`` in ``PreprocessingPipeline`` (which pins
    ``subsample=None``), job 04's inline construction
    (``QuantileTransformer(output_distribution="normal", random_state=seed)``)
    left ``subsample`` at whatever the installed scikit-learn's default was --
    a genuine, documented reconstruction limitation
    (``RECONSTRUCTION_SCOPE_JOB04``), not a clean bit-for-bit reconstruction.
    """

    from sklearn.preprocessing import QuantileTransformer

    train = load_pdg_filtered_shard(shard_dir, "train", None)
    feats = job04_legacy_feature_transform(train)
    qt = QuantileTransformer(output_distribution="normal", random_state=seed)
    qt.fit(feats)
    return qt


# --- §4.2 historical checkpoint contract ------------------------------------

@dataclass(frozen=True)
class CheckpointInfo:
    checkpoint_path: Optional[str]  # relative to job_dir, or None
    checkpoint_kind: str  # bare_state_dict_affine_module | bare_state_dict_legacy_flow | none
    reconstruction_status: str  # RECONSTRUCTIBLE | MISSING_HISTORICAL_CHECKPOINT | AMBIGUOUS_CONFIGURATION
    exclusion_reasons: Tuple[str, ...] = ()


def resolve_checkpoint_info(job_name: str, run_label_str: str) -> CheckpointInfo:
    """Where (if anywhere) a run's checkpoint lives in its frozen job_dir, and why."""

    if job_name == "04_legacy_available_code_realnvp_quantile":
        return CheckpointInfo("legacy_model.pt", "bare_state_dict_legacy_flow", "RECONSTRUCTIBLE")

    family = MODEL_FAMILY_OF_NAME.get(run_label_str.rsplit("_", 1)[0], None)  # unused fallback path

    if job_name in ("10_gaussian_controls_pdg13", "11_gaussian_controls_pdg_minus13"):
        # Baseline (Gaussian/GMM) family: producer never called `.save()` for
        # these; `checkpoint_hash` stayed `None` in every recorded metric.
        return CheckpointInfo(
            None, "none", "MISSING_HISTORICAL_CHECKPOINT",
            ("frozen D7 jobs 10/11 did not persist Gaussian/GMM checkpoints (producer never called estimator.save())",),
        )

    if job_name == "12_memory_release_repeat_smoke":
        # Both repeat subprocesses write to the SAME path
        # (`job_dir/f"{run_label}_model.pt"`); run2 runs second and silently
        # overwrites run1's file. Only run2's checkpoint survives on disk.
        return CheckpointInfo(
            f"{run_label_str}_model.pt", "bare_state_dict_affine_module", "RECONSTRUCTIBLE",
        )

    return CheckpointInfo(f"{run_label_str}_model.pt", "bare_state_dict_affine_module", "RECONSTRUCTIBLE")


def job12_run_checkpoint_info(repeat_index: int) -> CheckpointInfo:
    """Job 12 ran the identical tiny-affine smoke twice, sharing one checkpoint path.

    ``repeat_index`` is 1 or 2. Only run 2 (written last) has a checkpoint
    that actually corresponds to its own weights; run 1's file was
    overwritten before D8 ever ran.
    """

    label = run_label("identity_standardized_v0", "affine_tiny", weighted=False)
    if repeat_index == 2:
        return CheckpointInfo(f"{label}_model.pt", "bare_state_dict_affine_module", "RECONSTRUCTIBLE")
    return CheckpointInfo(
        None, "none", "MISSING_HISTORICAL_CHECKPOINT",
        (
            "job 12 runs the same tiny-affine smoke twice in separate subprocesses that both write to "
            f"job_dir/{label}_model.pt; run 2 (written second) silently overwrote run 1's checkpoint, so "
            "run 1's exact trained weights no longer exist on disk",
        ),
    )


def checkpoint_file_sha256(path: Path) -> str:
    """Raw file-bytes SHA-256 -- the ONLY hash method the producer used for
    checkpoints in this campaign (verified: recomputing this for a frozen
    affine run matches its recorded ``metrics.json`` ``checkpoint_hash``
    exactly). Do NOT use ``AffineCouplingFlow.checkpoint_hash()`` (the
    functional-fingerprint hash introduced later, 56355d0) here -- it hashes
    a different payload and will never match these frozen values."""

    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# --- Historical model reconstruction (§8) -----------------------------------

def build_frozen_affine_estimator(model_name: str, device: str = "cpu"):
    """Construct an ``AffineCouplingFlow`` with the exact frozen architecture,
    seeded exactly as the producer seeded it, WITHOUT loading any checkpoint."""

    from Nflow.torch_models.affine_coupling import AffineCouplingFlow

    params = AFFINE_TRAIN_PARAMS[model_name]
    estimator = AffineCouplingFlow(dimension=5, device=device, **params)
    estimator._build_module(seed=TRAINING_SEED)
    return estimator


def load_frozen_affine_checkpoint(model_name: str, checkpoint_path: Path, device: str = "cpu"):
    """Reconstruct + load a frozen bare-state_dict affine checkpoint.

    The frozen checkpoint is `torch.save(module.state_dict(), path)` -- not
    the modern `AffineCouplingFlow.save()` bundle -- so this bypasses
    `AffineCouplingFlow.load()` entirely and loads directly into `._module`.
    """

    import torch

    estimator = build_frozen_affine_estimator(model_name, device=device)
    state = torch.load(checkpoint_path, map_location=estimator.device, weights_only=True)
    estimator._module.load_state_dict(state, strict=True)
    return estimator


def load_frozen_legacy_flow_checkpoint(checkpoint_path: Path, device: str = "cpu"):
    """Reconstruct + load job 04's frozen legacy `NormalizingFlow` checkpoint."""

    import torch
    from Nflow.legacy.utils.flow_models import NormalizingFlow

    model = NormalizingFlow(
        input_dim=JOB04_LEGACY_CONFIG["input_dim"],
        hidden_dim=JOB04_LEGACY_CONFIG["hidden_dim"],
        n_layers=JOB04_LEGACY_CONFIG["n_layers"],
        init_zero=JOB04_LEGACY_CONFIG["init_zero"],
    )
    model.to(device)
    model.base_dist = torch.distributions.MultivariateNormal(
        torch.zeros(4, device=device), torch.eye(4, device=device)
    )
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    return model


# --- §4.6 Phase A audit assembly --------------------------------------------

INPUT_VALID = "D8_INPUT_VALID"
INPUT_PARTIAL = "D8_INPUT_PARTIAL"
INPUT_INCONSISTENT = "D8_INPUT_INCONSISTENT"


def run_phase_a_audit(repo_root: Path, input_artifact_dir: Path, shard_dir: Path) -> Dict[str, Any]:
    """Run the §4/§4.6 checks and return the audit result + classification.

    Never raises on a partial/degraded input; only raises (or returns
    `INPUT_INCONSISTENT`) when the §4.1 hash relation itself fails -- that is
    the one hard stop this function enforces directly.
    """

    input_artifact_dir = Path(input_artifact_dir)
    shard_dir = Path(shard_dir)
    findings: List[str] = []
    classification = INPUT_VALID

    afterms_audit_path = input_artifact_dir / "afterms_audit.json"
    nightly_summary_path = input_artifact_dir / "report" / "nightly_summary.json"
    queue_state_path = input_artifact_dir / "queue_state.json"
    shard_manifest_path = shard_dir / "shard_manifest.json"

    afterms_audit = json.loads(afterms_audit_path.read_text()) if afterms_audit_path.exists() else None
    nightly_summary = json.loads(nightly_summary_path.read_text()) if nightly_summary_path.exists() else None
    queue_state = json.loads(queue_state_path.read_text()) if queue_state_path.exists() else None
    shard_manifest = json.loads(shard_manifest_path.read_text()) if shard_manifest_path.exists() else None

    if afterms_audit is None or nightly_summary is None or shard_manifest is None:
        return {
            "classification": INPUT_INCONSISTENT,
            "findings": ["missing one or more of afterms_audit.json / nightly_summary.json / shard_manifest.json"],
        }

    # Check 1-3: exactly 14 jobs, all completed, final report says complete.
    job_dirs = sorted(p.name for p in (input_artifact_dir / "jobs").iterdir() if p.is_dir())
    if tuple(job_dirs) != JOB_NAMES:
        classification = INPUT_PARTIAL
        findings.append(f"job directory set does not exactly match the expected 14 jobs: found {job_dirs}")

    status_code = nightly_summary.get("status_code")
    if status_code != "NIGHTLY_SMOKES_COMPLETE":
        classification = INPUT_PARTIAL
        findings.append(f"nightly_summary.json.status_code is {status_code!r}, not NIGHTLY_SMOKES_COMPLETE")

    # Known, non-blocking discrepancy: queue_state.json can list job 13 as
    # failed from an earlier attempt whose status.json/report were later
    # overwritten by a successful standalone rerun; queue_state.json's
    # completed/failed lists are not re-synced by that rerun path. Per-job
    # status.json + nightly_summary.json are authoritative.
    if queue_state and "13_build_nightly_report" in queue_state.get("failed_jobs", []):
        job13_status_path = input_artifact_dir / "jobs" / "13_build_nightly_report" / "status.json"
        job13_status = json.loads(job13_status_path.read_text()) if job13_status_path.exists() else {}
        if job13_status.get("status") == "completed" and status_code == "NIGHTLY_SMOKES_COMPLETE":
            findings.append(
                "queue_state.json lists 13_build_nightly_report in failed_jobs, but its status.json is "
                "'completed' and nightly_summary.json reports NIGHTLY_SMOKES_COMPLETE -- queue_state.json is "
                "stale from an earlier attempt and was never re-synced by the successful standalone rerun; "
                "not treated as a failure"
            )
        else:
            classification = INPUT_PARTIAL
            findings.append("queue_state.json reports job 13 failed and no completed status.json confirms otherwise")

    # §4.1 hash relation -- the one hard stop.
    try:
        legacy_prefix16 = legacy_raw_file_sha256_prefix16(nightly_summary)
        recomputed_full = recompute_raw_file_sha256(repo_root)
        if not verify_raw_file_hash_relation(legacy_prefix16, recomputed_full):
            return {
                "classification": INPUT_INCONSISTENT,
                "findings": findings + [
                    f"raw-file hash relation failed: legacy prefix16={legacy_prefix16!r}, "
                    f"recomputed[:16]={recomputed_full[:16]!r}"
                ],
                "legacy_raw_file_sha256_prefix16": legacy_prefix16,
                "recomputed_raw_file_sha256": recomputed_full,
            }
    except LegacyContractError as exc:
        return {"classification": INPUT_INCONSISTENT, "findings": findings + [str(exc)]}

    content_hash_audit = audit_content_dataset_hash(afterms_audit)
    content_hash_shard = shard_manifest_content_hash(shard_manifest)
    if content_hash_audit != content_hash_shard:
        classification = INPUT_PARTIAL
        findings.append(
            "afterms_audit.json.content_dataset_hash != shard_manifest.json.dataset_hash "
            f"({content_hash_audit!r} != {content_hash_shard!r}); both are supposed to be the same "
            "canonicalized-array content hash recorded by two different producers"
        )

    return {
        "classification": classification,
        "findings": findings,
        "legacy_raw_file_sha256_prefix16": legacy_prefix16,
        "recomputed_raw_file_sha256": recomputed_full,
        "audit_raw_file_sha256": audit_raw_file_sha256(afterms_audit),
        "content_dataset_hash": content_hash_audit,
        "shard_manifest_hash": content_hash_shard,
        "producer_git_commit": nightly_summary.get("git_commit"),
        "job_directories": job_dirs,
        "status_code": status_code,
    }


def build_legacy_contract_table() -> Dict[str, Any]:
    """A JSON-serializable snapshot of every frozen-contract fact this module
    encodes, for `audit/legacy_d7_contract.json` (§4.6)."""

    return {
        "producer_git_commit": PRODUCER_GIT_COMMIT,
        "training_seed": TRAINING_SEED,
        "generation_seed": GENERATION_SEED,
        "split_shard_names": SPLIT_SHARD_NAMES,
        "job_pdg": {k: {"pdg_value": v[0], "pdg_policy": v[1]} for k, v in JOB_PDG.items()},
        "job_variants": {k: list(v) for k, v in JOB_VARIANTS.items()},
        "job_model_names": {k: list(v) for k, v in JOB_MODEL_NAMES.items()},
        "job_weighted": JOB_WEIGHTED,
        "affine_train_params": AFFINE_TRAIN_PARAMS,
        "baseline_train_params": BASELINE_TRAIN_PARAMS,
        "job04_legacy_config": {**JOB04_LEGACY_CONFIG, "feature_order": list(JOB04_LEGACY_CONFIG["feature_order"])},
        "notes": [
            "affine_coupling.py and Nflow/baselines/*.py are byte-identical between the producer commit "
            "(f9d8246) and the current HEAD (56355d0); only preprocessing.py (+Jacobian) and "
            "scripts/run_afterms_nightly_queue.py (checkpoint persistence, hash naming) changed.",
            "checkpoint_hash in this campaign's metrics.json is always a raw-file-bytes SHA-256 "
            "(checkpoint_file_hash at f9d8246), never AffineCouplingFlow.checkpoint_hash()'s functional "
            "fingerprint (introduced later, 56355d0). Verified empirically against every affine/legacy "
            "checkpoint in the campaign.",
            "Every job 04-12 trains/validates/tests on shard index 0 only "
            "(train_shard_000/validation_shard_000/test_shard_000).",
            "Job 12 runs the same tiny-affine smoke twice in-process to the same checkpoint path; only "
            "the second run's checkpoint survives on disk.",
            "Jobs 10/11 (Gaussian/GMM) never called estimator.save(); no historical checkpoint exists.",
        ],
    }


def write_phase_a_outputs(audit_result: Dict[str, Any], output_dir: Path) -> None:
    """Write `audit/immutable_input_manifest.json`, `audit/immutable_input_audit.md`
    and `audit/legacy_d7_contract.json` (§4.6)."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "immutable_input_manifest.json").write_text(json.dumps(audit_result, indent=2))
    (output_dir / "legacy_d7_contract.json").write_text(json.dumps(build_legacy_contract_table(), indent=2))

    md = []
    md.append("# D8 Immutable Input Audit\n\n")
    md.append(f"- **Classification**: `{audit_result['classification']}`\n")
    md.append(f"- **Producer git commit**: `{audit_result.get('producer_git_commit')}`\n")
    md.append(f"- **Legacy raw-file SHA-256 prefix16**: `{audit_result.get('legacy_raw_file_sha256_prefix16')}`\n")
    md.append(f"- **Recomputed raw-file SHA-256**: `{audit_result.get('recomputed_raw_file_sha256')}`\n")
    md.append(f"- **Content dataset hash**: `{audit_result.get('content_dataset_hash')}`\n")
    md.append(f"- **Shard manifest hash**: `{audit_result.get('shard_manifest_hash')}`\n\n")
    md.append("## Findings\n\n")
    for finding in audit_result.get("findings", []):
        md.append(f"- {finding}\n")
    if not audit_result.get("findings"):
        md.append("- none\n")
    (output_dir / "immutable_input_audit.md").write_text("".join(md))
