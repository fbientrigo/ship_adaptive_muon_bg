"""Phase D (D8 spec §8-§10): historical model reconstruction + deterministic samples.

Reconstruction is attempted ONLY for runs the registry already marked
`RECONSTRUCTIBLE` (adapter-verified checkpoint contract). Nothing here infers
missing configuration or refits a generative model -- baselines with
`MISSING_HISTORICAL_CHECKPOINT` are never silently substituted.

Bit-matching the original D7 in-memory generated sample is explicitly NOT
required (§9): that sample was never persisted and its RNG state cannot be
recovered. This module's generated samples are a FRESH, D8-owned,
deterministic generation from the reconstructed checkpoint + a fixed
generation seed -- reproducible run-to-run within D8, not against D7.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ship_muon_bg.afterms.d8 import legacy_d7_adapter as adapter
from ship_muon_bg.afterms.d8.registry import RunRecord


class ReconstructionError(RuntimeError):
    pass


def _job_dir_name(run_id: str) -> str:
    # run_id is "<job_name>__<run_key>[__repeatN]"; job_name itself contains
    # no "__", so splitting once recovers it exactly.
    return run_id.split("__", 1)[0]


def _model_name_from_run_id(record: RunRecord) -> Optional[str]:
    if record.model_family != "affine_coupling":
        return None
    for name in adapter.AFFINE_TRAIN_PARAMS:
        if name in record.run_id:
            return name
    return None


def sample_hash(samples: np.ndarray) -> str:
    arr = np.ascontiguousarray(samples, dtype=np.float64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def reconstruct_and_generate(
    record: RunRecord,
    *,
    jobs_dir: Path,
    shard_dir: Path,
    sample_count: int,
    generation_seed: int = adapter.GENERATION_SEED,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Reconstruct `record`'s historical model + preprocessing, generate
    `sample_count` deterministic physical-space samples, and return a
    provenance-complete result dict. Raises `ReconstructionError` if
    `record.reconstruction_status != "RECONSTRUCTIBLE"`."""

    if record.reconstruction_status != "RECONSTRUCTIBLE":
        raise ReconstructionError(
            f"{record.run_id} is not reconstructible (status={record.reconstruction_status})"
        )

    import torch

    job_dir = Path(jobs_dir) / _job_dir_name(record.run_id)
    checkpoint_path = job_dir / record.checkpoint_path

    if not checkpoint_path.exists():
        raise ReconstructionError(
            f"{record.run_id}: registry marked this run RECONSTRUCTIBLE but its checkpoint file is "
            f"missing on disk at {checkpoint_path}"
        )

    recomputed_checkpoint_hash = adapter.checkpoint_file_sha256(checkpoint_path)
    if record.checkpoint_hash is not None and recomputed_checkpoint_hash != record.checkpoint_hash:
        raise ReconstructionError(
            f"{record.run_id}: checkpoint file hash changed on disk since the registry was built "
            f"(recorded {record.checkpoint_hash!r}, now {recomputed_checkpoint_hash!r})"
        )

    if record.model_family == "legacy_normalizing_flow":
        model = adapter.load_frozen_legacy_flow_checkpoint(checkpoint_path, device=device)
        qt = adapter.reconstruct_job04_quantile_transformer(shard_dir)
        preprocessing_hash = _sklearn_object_hash(qt)

        torch.manual_seed(generation_seed)
        model.eval()
        with torch.no_grad():
            z = model.base_dist.sample((sample_count,))
            gen_scaled = model.inverse(z).cpu().numpy()
        samples = qt.inverse_transform(gen_scaled)
        feature_order = list(adapter.MODELED_FEATURES_4D)
        pz_index = 2

    elif record.model_family == "affine_coupling":
        model_name = _model_name_from_run_id(record)
        if model_name is None:
            raise ReconstructionError(f"{record.run_id}: could not resolve affine model_name from run_id")
        estimator = adapter.load_frozen_affine_checkpoint(model_name, checkpoint_path, device=device)
        pipeline = adapter.reconstruct_preprocessing_pipeline(record.preprocessing_name, shard_dir, record.pdg_value)
        preprocessing_hash = _preprocessing_hash(pipeline)

        generator = torch.Generator(device=estimator.device)
        generator.manual_seed(generation_seed)
        estimator._module.eval()
        with torch.no_grad():
            z = torch.randn(
                sample_count, estimator.dimension, dtype=estimator.torch_dtype,
                device=estimator.device, generator=generator,
            )
            gen_scaled = estimator._module(z).cpu().numpy()
        samples = pipeline.inverse(gen_scaled)
        feature_order = list(adapter.MODELED_FEATURES_5D)
        pz_index = 2

    else:
        raise ReconstructionError(f"{record.run_id}: reconstruction not implemented for family {record.model_family!r}")

    pz = samples[:, pz_index]
    domain_violation_mask = pz < 0.0
    domain_violation_count = int(np.count_nonzero(domain_violation_mask))

    provenance = {
        "run_id": record.run_id,
        "sample_count": int(sample_count),
        "generation_seed": generation_seed,
        "historical_checkpoint_path": str(checkpoint_path),
        "historical_checkpoint_hash": recomputed_checkpoint_hash,
        "preprocessing_reconstruction_hash": preprocessing_hash,
        "feature_order": feature_order,
        "sample_hash": sample_hash(samples),
        "domain_violation_count": domain_violation_count,
        "domain_violation_rate": domain_violation_count / sample_count if sample_count else None,
        "minimum_generated_pz": float(np.min(pz)),
        "generated_pz_quantiles": {
            q: float(np.quantile(pz, q)) for q in (0.001, 0.01, 0.05, 0.5, 0.95, 0.99)
        },
        "bit_match_to_original_d7_sample": "not_applicable_d7_sample_never_persisted_rng_state_unrecoverable",
    }

    if record.preprocessing_name == "quantile_normal_v0" or record.model_family == "legacy_normalizing_flow":
        provenance.update(_quantile_saturation_diagnostics(gen_scaled, qt if record.model_family == "legacy_normalizing_flow" else pipeline.qt))

    return {"samples": samples, "provenance": provenance}


def _quantile_saturation_diagnostics(transformed: np.ndarray, qt) -> Dict[str, Any]:
    quantiles = qt.quantiles_
    lo = quantiles[0, :]
    hi = quantiles[-1, :]
    inv = qt.inverse_transform(transformed)
    at_min = np.any(inv <= lo[None, :], axis=1)
    at_max = np.any(inv >= hi[None, :], axis=1)
    n = transformed.shape[0]
    unique_rows = np.unique(np.ascontiguousarray(inv), axis=0)
    return {
        "fraction_at_training_minimum": float(np.mean(at_min)),
        "fraction_at_training_maximum": float(np.mean(at_max)),
        "fraction_at_outer_knots": float(np.mean(at_min | at_max)),
        "duplicate_rate": float(1.0 - unique_rows.shape[0] / n) if n else None,
    }


def _preprocessing_hash(pipeline) -> str:
    payload = json.dumps(pipeline.to_dict(), sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sklearn_object_hash(obj) -> str:
    import pickle

    return hashlib.sha256(pickle.dumps(obj)).hexdigest()


def write_generated_samples(result: Dict[str, Any], output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = result["provenance"]["run_id"]
    np.save(output_dir / f"{run_id}.npy", result["samples"])
    (output_dir / f"{run_id}.json").write_text(json.dumps(result["provenance"], indent=2), encoding='utf-8')


# --- §10 reference sample contract ------------------------------------------

def select_reference_indices(
    shard_dir: Path, pdg_value: Optional[int], sample_count: int, seed: int
) -> np.ndarray:
    """Deterministic indices into the PDG-filtered `test_shard_000` array
    (held-out rows only -- never train/validation). Indices are positions in
    the filtered array, not the raw unfiltered shard, and must be read back
    against the SAME filter to mean anything."""

    n_available = adapter.load_pdg_filtered_shard(shard_dir, "test", pdg_value).shape[0]
    n = min(sample_count, n_available)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_available, size=n, replace=False))


def write_reference_sample(arena_id: str, pdg_value: Optional[int], shard_dir: Path, sample_count: int, seed: int, output_dir: Path) -> Dict[str, Any]:
    indices = select_reference_indices(shard_dir, pdg_value, sample_count, seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"{arena_id}.indices.npy", indices)
    provenance = {
        "arena_id": arena_id,
        "pdg_value": pdg_value,
        "split": "test_shard_000",
        "sample_count": int(indices.shape[0]),
        "seed": seed,
        "indices_hash": sample_hash(indices.astype(np.float64)),
        "note": (
            "indices are positions in the PDG-filtered test_shard_000 array, held-out rows only; "
            "the D7 split is row-disjoint from train/validation but has no source-lineage/group "
            "identifier, so this does NOT establish source-muon independence"
        ),
    }
    (output_dir / f"{arena_id}.json").write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    return provenance


def load_reference_rows(shard_dir: Path, pdg_value: Optional[int], indices: np.ndarray) -> np.ndarray:
    """Physical-space (px, py, pz, x, y) held-out rows at `indices` (positions
    into the PDG-filtered `test_shard_000` array -- see `select_reference_indices`)."""

    filtered = adapter.load_pdg_filtered_shard(shard_dir, "test", pdg_value)
    return filtered[indices][:, :5]


def real_vs_real_disjoint_subsets(
    shard_dir: Path, pdg_value: Optional[int], sample_count: int, seed: int
) -> Dict[str, np.ndarray]:
    """Two disjoint deterministic held-out subsets of `test_shard_000` for a
    real-vs-real baseline (§10, §12: every statistical test needs one)."""

    n_available = adapter.load_pdg_filtered_shard(shard_dir, "test", pdg_value).shape[0]
    n_each = min(sample_count, n_available // 2)
    rng = np.random.default_rng(seed)
    permuted = rng.permutation(n_available)
    return {"subset_a": np.sort(permuted[:n_each]), "subset_b": np.sort(permuted[n_each:2 * n_each])}
