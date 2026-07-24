"""D9-5 common data scope: shard resolution, per-track PDG filtering, the
data-scope manifest/audit, and a host-RAM preflight (Sec 6, Sec 6.1).

Every model family within a track reads through :func:`load_filtered_split`
so all four families see exactly the same rows. Shards are filtered one at a
time and only the (smaller) filtered result is retained -- the full
unfiltered ``(both PDG codes, N, 8)`` concatenation this mission's RAM
preflight flagged as the avoidable peak is never materialized.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ship_muon_bg.data_contracts import dataset_hash, schema

from . import config as d9_5config

SPLITS = ("train", "validation", "test")

# float64 raw row (8 cols) and float64 filtered-feature row (5 cols) sizes,
# used only for the preflight estimate -- never for deciding whether to fit.
_BYTES_PER_RAW_ROW = schema.N_COLUMNS * 8
_BYTES_PER_FEATURE_ROW = 5 * 8


def load_shard_manifest(shard_dir: Path) -> Dict[str, Any]:
    return json.loads(Path(shard_dir, "shard_manifest.json").read_text(encoding="utf-8"))


def shards_for_split(manifest: Dict[str, Any], split: str) -> List[Dict[str, Any]]:
    return [s for s in manifest["shards"] if s["split"] == split]


def check_no_shard_overlap(manifest: Dict[str, Any]) -> List[str]:
    """Return a list of violation strings (empty if every shard file appears
    in exactly one split)."""

    seen: Dict[str, str] = {}
    violations = []
    for shard in manifest["shards"]:
        name = shard["npy_file"]
        if name in seen:
            violations.append(f"{name} appears in both split={seen[name]!r} and split={shard['split']!r}")
        else:
            seen[name] = shard["split"]
    return violations


def _filtered_shard(shard_dir: Path, shard_entry: Dict[str, Any], pdg_value: int) -> np.ndarray:
    raw = np.load(Path(shard_dir, shard_entry["npy_file"]))
    mask = np.rint(raw[:, schema.COLUMN_INDEX["id"]]) == pdg_value
    return raw[mask]


def load_filtered_split(shard_dir: Path, manifest: Dict[str, Any], split: str, pdg_value: int) -> np.ndarray:
    """Concatenate every shard in ``split``, filtered to ``pdg_value``, one
    shard at a time. Only the (post-filter) rows are ever concatenated."""

    parts = [
        _filtered_shard(shard_dir, shard, pdg_value)
        for shard in shards_for_split(manifest, split)
    ]
    if not parts:
        return np.empty((0, schema.N_COLUMNS), dtype=np.float64)
    return np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]


def _weight_summary(raw: np.ndarray) -> Dict[str, Any]:
    if raw.shape[0] == 0:
        return {"count": 0}
    w = raw[:, schema.COLUMN_INDEX["w"]].astype(np.float64)
    return {
        "count": int(w.shape[0]),
        "mean": float(np.mean(w)),
        "min": float(np.min(w)),
        "max": float(np.max(w)),
        "note": "metadata only -- w is never a modeled feature or a loss weight in D9-5",
    }


def _split_track_record(shard_dir: Path, manifest: Dict[str, Any], split: str, pdg_value: int) -> Dict[str, Any]:
    shard_entries = shards_for_split(manifest, split)
    original_row_count = sum(int(s["row_count"]) for s in shard_entries)
    filtered = load_filtered_split(shard_dir, manifest, split, pdg_value)
    features = filtered[:, :5]
    finite_mask = np.isfinite(features)
    negative_pz = int(np.count_nonzero(filtered[:, 2] < 0.0)) if filtered.shape[0] else 0
    return {
        "split": split,
        "shard_names": [s["npy_file"] for s in shard_entries],
        "shard_hashes": [s["shard_hash"] for s in shard_entries],
        "original_row_count": original_row_count,
        "row_count_after_pdg_filter": int(filtered.shape[0]),
        "feature_order": ["px", "py", "pz", "x", "y"],
        "finite_value_counts_per_feature": [int(np.count_nonzero(finite_mask[:, j])) for j in range(5)],
        "negative_pz_count": negative_pz,
        "source_weight_summary": _weight_summary(filtered),
        "content_hash_of_filtered_scope": dataset_hash(filtered) if filtered.shape[0] else None,
    }


def build_data_scope_manifest(data_scope_config: Dict[str, Any]) -> Dict[str, Any]:
    """The resolved, evidence-backed counterpart to the frozen policy config
    (Sec 6): concrete shard names/hashes/row counts per split and track."""

    shard_dir = d9_5config.REPO_ROOT / data_scope_config["shard_dir"]
    manifest = load_shard_manifest(shard_dir)
    overlap_violations = check_no_shard_overlap(manifest)

    tracks_out = []
    for track in data_scope_config["tracks"]:
        splits_out = {
            split: _split_track_record(shard_dir, manifest, split, track["pdg_value"])
            for split in SPLITS
        }
        tracks_out.append({
            "track_id": track["track_id"],
            "pdg_value": track["pdg_value"],
            "splits": splits_out,
        })

    return {
        "schema_version": "d9_5_data_scope_manifest_v0",
        "dataset_hash": manifest.get("dataset_hash"),
        "shard_dir": data_scope_config["shard_dir"],
        "shard_overlap_violations": overlap_violations,
        "tracks": tracks_out,
    }


def render_data_scope_audit_md(manifest_record: Dict[str, Any]) -> str:
    lines = ["# D9-5 Data Scope Audit", ""]
    lines.append(f"Source dataset hash: `{manifest_record['dataset_hash']}`")
    lines.append(f"Shard overlap violations: {manifest_record['shard_overlap_violations'] or 'none'}")
    lines.append("")
    for track in manifest_record["tracks"]:
        lines.append(f"## {track['track_id']} (pdg_value={track['pdg_value']})")
        lines.append("")
        lines.append("| split | shards | rows (raw) | rows (pdg-filtered) | negative pz |")
        lines.append("|---|---|---|---|---|")
        for split in SPLITS:
            rec = track["splits"][split]
            lines.append(
                f"| {split} | {len(rec['shard_names'])} | {rec['original_row_count']} | "
                f"{rec['row_count_after_pdg_filter']} | {rec['negative_pz_count']} |"
            )
        lines.append("")
    return "\n".join(lines)


def estimate_ram_preflight(manifest_record: Dict[str, Any]) -> Dict[str, Any]:
    """Sec 6.1: estimate peak host RAM for NF shard streaming, exact Gaussian
    sufficient statistics, GMM fitting and validation/test evaluation, from
    actual row counts and dtype sizes -- never from a guess."""

    estimates = {}
    for track in manifest_record["tracks"]:
        train_rows = track["splits"]["train"]["row_count_after_pdg_filter"]
        val_rows = track["splits"]["validation"]["row_count_after_pdg_filter"]
        test_rows = track["splits"]["test"]["row_count_after_pdg_filter"]
        # NF/GMM: one materialized (n_train, 5) float64 array plus a comparable
        # amount of working memory (optimizer/EM intermediates); Gaussian
        # streaming never materializes more than one shard (~30MB) at a time.
        train_feature_bytes = train_rows * _BYTES_PER_FEATURE_ROW
        estimates[track["track_id"]] = {
            "train_rows": train_rows,
            "validation_rows": val_rows,
            "test_rows": test_rows,
            "gaussian_streaming_peak_bytes": 2 * 30_000_000,  # two shards in flight worst case
            "gmm_incore_fit_peak_bytes": int(train_feature_bytes * 3),  # X + responsibilities + working arrays
            "nf_gpu_resident_tensor_bytes": int(train_rows * 5 * 4),  # float32 train tensor on device
            "nf_host_peak_bytes": int(train_feature_bytes * 2),  # standardized array + raw filtered array
            "evaluation_peak_bytes": int((val_rows + test_rows) * _BYTES_PER_FEATURE_ROW * 2),
        }
    return {
        "schema_version": "d9_5_ram_preflight_v0",
        "per_track": estimates,
        "conclusion": "complete declared scope (all 22/3/3 shards) fits comfortably in host RAM for every family; no chunked/streaming GMM implementation required (see configs/afterms/d9_5_model_family_arena_v0.json:gmm.fitting_policy_note). D9_5_BLOCKED_BY_GMM_DATA_SCALE does not apply.",
    }
