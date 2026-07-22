#!/usr/bin/env python3
"""build_afterms_shards.py: Phase A descriptive audit and Phase B sharding of the after-MS dataset.

Usage:
    python scripts/build_afterms_shards.py [--seed SEED] [--target-rows TARGET_ROWS]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import numpy as np

# Add src/ to path to allow importing ship_muon_bg
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ship_muon_bg.data_contracts import (
    load_muon_pkl,
    dataset_hash,
    schema,
    run_checks,
)
from ship_muon_bg.afterms import audit, split, stratify

DEFAULT_JOB_NAME = "01_build_afterms_shards"


def compute_hash(data_bytes: bytes) -> str:
    return hashlib.sha256(data_bytes).hexdigest()


def _atomic_save_npy(path, array, **save_kwargs):
    """Write ``array`` to ``path`` via a same-directory temp file + os.replace,
    so a killed process never leaves a half-written file at the final name."""
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as handle:
        np.save(handle, array, **save_kwargs)
    os.replace(tmp_path, path)


def _atomic_write_json(path, obj, **dump_kwargs):
    """Write ``obj`` as JSON to ``path`` via a same-directory temp file +
    os.replace, so a completed manifest never references a partially written
    sibling file (and never appears itself half-written)."""
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, **dump_kwargs)
        handle.write("\n")
    os.replace(tmp_path, path)


def _atomic_write_text(path, text):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(tmp_path, path)


def _write_progress(artifact_dir, job_name, phase, *, started_at, extra=None):
    """Update the existing jobs/<job_name>/progress.json heartbeat contract
    that scripts/watch_afterms_nightly_queue.py already reads, so the shard
    build reports meaningful phase boundaries instead of going silent for
    minutes at a time."""
    job_dir = os.path.join(artifact_dir, "jobs", job_name)
    os.makedirs(job_dir, exist_ok=True)
    payload = {
        "phase": phase,
        "elapsed_seconds": time.perf_counter() - started_at,
        "heartbeat": time.time(),
    }
    if extra:
        payload.update(extra)
    try:
        _atomic_write_json(os.path.join(job_dir, "progress.json"), payload, indent=2, sort_keys=True)
    except OSError:
        pass


def standardized_mean_difference(shard_col, split_col):
    split_std = np.std(split_col)
    if split_std <= 0.0:
        return 0.0
    return (np.mean(shard_col) - np.mean(split_col)) / split_std


def calculate_representativeness(shard_data, split_data, shard_indices, split_indices, schema_index):
    """Numerically compare a shard's distribution against its parent split."""
    report = {}
    
    # Row fractions
    report["row_count_shard"] = int(shard_data.shape[0])
    report["row_count_split"] = int(split_data.shape[0])
    report["row_fraction"] = float(shard_data.shape[0] / split_data.shape[0]) if split_data.shape[0] > 0 else 0.0
    
    # PDG fractions
    id_idx = schema_index["id"]
    shard_pdgs, shard_pdg_counts = np.unique(np.rint(shard_data[:, id_idx]).astype(np.int64), return_counts=True)
    split_pdgs, split_pdg_counts = np.unique(np.rint(split_data[:, id_idx]).astype(np.int64), return_counts=True)
    
    shard_pdg_fracs = {str(p): float(c / shard_data.shape[0]) for p, c in zip(shard_pdgs, shard_pdg_counts)}
    split_pdg_fracs = {str(p): float(c / split_data.shape[0]) for p, c in zip(split_pdgs, split_pdg_counts)}
    
    report["pdg_fractions"] = {
        "shard": shard_pdg_fracs,
        "split": split_pdg_fracs,
        "difference": {p: shard_pdg_fracs.get(p, 0.0) - split_pdg_fracs.get(p, 0.0) for p in split_pdg_fracs}
    }
    
    # Weight summaries and sign categories
    w_idx = schema_index["w"]
    shard_w = shard_data[:, w_idx]
    split_w = split_data[:, w_idx]
    
    def sign_fracs(col):
        total = col.shape[0]
        if total == 0:
            return {"positive": 0.0, "zero": 0.0, "negative": 0.0}
        return {
            "positive": float(np.count_nonzero(col > 0) / total),
            "zero": float(np.count_nonzero(col == 0) / total),
            "negative": float(np.count_nonzero(col < 0) / total),
        }
        
    report["weight_fractions"] = {
        "shard": sign_fracs(shard_w),
        "split": sign_fracs(split_w),
    }
    
    # Feature quantiles & standardized mean differences
    features = ["px", "py", "pz", "x", "y"]
    q_levels = [0.01, 0.1, 0.5, 0.9, 0.99]
    feature_diffs = {}
    
    for feat in features:
        feat_idx = schema_index[feat]
        shard_col = shard_data[:, feat_idx]
        split_col = split_data[:, feat_idx]
        
        shard_qs = np.quantile(shard_col, q_levels)
        split_qs = np.quantile(split_col, q_levels)
        
        q_diff = {str(ql): float(sq - spq) for ql, sq, spq in zip(q_levels, shard_qs, split_qs)}
        smd = standardized_mean_difference(shard_col, split_col)
        
        feature_diffs[feat] = {
            "quantile_differences": q_diff,
            "standardized_mean_difference": smd,
            "shard_mean": float(np.mean(shard_col)),
            "split_mean": float(np.mean(split_col)),
            "shard_std": float(np.std(shard_col)),
            "split_std": float(np.std(split_col)),
        }
    report["features"] = feature_diffs
    
    # Correlation differences
    # Only compute correlations for px, py, pz, x, y (the modeled features)
    feat_indices = [schema_index[f] for f in features]
    shard_corr = np.corrcoef(shard_data[:, feat_indices], rowvar=False)
    split_corr = np.corrcoef(split_data[:, feat_indices], rowvar=False)
    
    # Handle possible 1-element or NaN correlations
    if isinstance(shard_corr, np.ndarray) and isinstance(split_corr, np.ndarray):
        corr_diff = shard_corr - split_corr
        report["max_abs_correlation_diff"] = float(np.max(np.abs(corr_diff)))
    else:
        report["max_abs_correlation_diff"] = 0.0
        
    # Edge bucket fractions
    shard_tails = stratify.compute_tail_buckets(shard_data, schema_index)
    split_tails = stratify.compute_tail_buckets(split_data, schema_index)
    
    tail_fracs = {}
    for name in shard_tails:
        shard_frac = float(np.mean(shard_tails[name]))
        split_frac = float(np.mean(split_tails[name]))
        tail_fracs[name] = {
            "shard_fraction": shard_frac,
            "split_fraction": split_frac,
            "difference": shard_frac - split_frac
        }
    report["edge_buckets"] = tail_fracs
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Build representative after-MS shards")
    parser.add_argument("--seed", type=int, default=20260720, help="Seed for split and sharding")
    parser.add_argument("--target-rows", type=int, default=500000, help="Target rows per training/validation/test shard")
    parser.add_argument("--raw-file", type=str, default="data/raw/nflow_releases/muonsFullMC_afterMS.pkl", help="Input PKL path")
    parser.add_argument("--shard-dir", type=str, default="data/shards/afterms_nightly_v0", help="Output directory for shards")
    parser.add_argument("--artifact-dir", type=str, default="artifacts/afterms_nightly_v0", help="Output directory for audits")
    parser.add_argument("--job-name", type=str, default=DEFAULT_JOB_NAME, help="Job name under artifact-dir/jobs/ for progress heartbeat")
    args = parser.parse_args()

    os.makedirs(args.shard_dir, exist_ok=True)
    os.makedirs(args.artifact_dir, exist_ok=True)

    t_start = time.perf_counter()

    def progress(phase, **extra):
        _write_progress(args.artifact_dir, args.job_name, phase, started_at=t_start, extra=extra or None)

    print(f"Loading raw dataset: {args.raw_file}", flush=True)
    progress("raw_dataset_load_started")
    array = load_muon_pkl(args.raw_file)
    n_rows, n_cols = array.shape
    print(f"Loaded dataset: {n_rows} rows, {n_cols} columns (elapsed {time.perf_counter() - t_start:.1f}s)", flush=True)
    progress("raw_dataset_load_completed", n_rows=int(n_rows), n_cols=int(n_cols))

    # 1. Descriptive Audit (Phase A)
    print("Performing Phase A descriptive audit...", flush=True)
    audit_data = audit.build_afterms_audit(args.raw_file, sample_seed=args.seed)
    audit_path = audit.write_afterms_audit(audit_data, args.artifact_dir)
    print(f"Audit written to: {audit_path} (elapsed {time.perf_counter() - t_start:.1f}s)", flush=True)
    raw_hash = audit_data["content_dataset_hash"]
    print(f"Dataset Content Hash: {raw_hash}", flush=True)
    progress("descriptive_audit_completed", content_dataset_hash=raw_hash)

    # 2. Deterministic split (Phase B, §5.1)
    print("Generating deterministic global train/validation/test split...", flush=True)
    progress("split_assignment_started")
    row_indices = np.arange(n_rows, dtype=np.uint64)
    labels = split.assign_split(row_indices, dataset_hash=raw_hash, seed=args.seed)
    print(f"Split assignment completed (elapsed {time.perf_counter() - t_start:.1f}s)", flush=True)
    progress("split_assignment_completed")

    # 3. Create shards for each split (Phase B, §5.2-5.3)
    shards_written = []
    representativeness_reports = {}
    
    global_manifest = {
        # Content hash of the canonicalized in-memory array (data_contracts
        # .dataset_hash), distinct from a raw-file-bytes hash -- named
        # explicitly so it is never confused with (or joined against) the
        # nightly queue's file-level "raw_file_sha256".
        "content_dataset_hash": raw_hash,
        "source_file": os.path.abspath(args.raw_file),
        "seed": args.seed,
        "splits": {}
    }

    for split_name in ["train", "validation", "test"]:
        split_mask = (labels == split_name)
        split_indices = row_indices[split_mask]
        split_array = array[split_mask]

        split_n_rows = split_array.shape[0]
        n_shards = stratify.n_shards_for_split(split_n_rows, target_rows_per_shard=args.target_rows)
        print(f"Split '{split_name}' started: {split_n_rows} rows, assigning to {n_shards} shards "
              f"(elapsed {time.perf_counter() - t_start:.1f}s)", flush=True)
        progress(f"split_{split_name}_started", row_count=int(split_n_rows), n_shards=int(n_shards))

        print(f"Split '{split_name}': computing stratification summaries...", flush=True)
        stratum_ids, stratum_table = stratify.compute_strata(split_array, schema.COLUMN_INDEX)
        shard_of_row, understaffed_report = stratify.assign_shards(
            split_indices, stratum_ids, n_shards=n_shards, shard_seed=args.seed
        )
        print(f"Split '{split_name}': stratification summaries completed "
              f"(elapsed {time.perf_counter() - t_start:.1f}s)", flush=True)

        global_manifest["splits"][split_name] = {
            "row_count": int(split_n_rows),
            "n_shards": int(n_shards),
            "understaffed_strata": understaffed_report["understaffed_strata"]
        }

        for s in range(n_shards):
            shard_mask = (shard_of_row == s)
            shard_data = split_array[shard_mask]
            shard_row_indices = split_indices[shard_mask]
            
            # Shard filenames
            shard_name = f"{split_name}_shard_{s:03d}"
            npy_path = os.path.join(args.shard_dir, f"{shard_name}.npy")
            idx_path = os.path.join(args.shard_dir, f"{shard_name}.indices.npy")
            json_path = os.path.join(args.shard_dir, f"{shard_name}.json")

            # Save arrays (preserve float64 format) atomically: a killed
            # process must never leave a shard file that looks complete but
            # isn't.
            _atomic_save_npy(npy_path, shard_data, allow_pickle=False)
            _atomic_save_npy(idx_path, shard_row_indices, allow_pickle=False)

            # Compute shard file hashes
            with open(npy_path, "rb") as f:
                shard_hash = compute_hash(f.read())
            with open(idx_path, "rb") as f:
                idx_hash = compute_hash(f.read())

            # Stratification counts/checks
            pdg_col = shard_data[:, schema.COLUMN_INDEX["id"]]
            pdg_vals, pdg_counts = np.unique(np.rint(pdg_col).astype(np.int64), return_counts=True)
            pdg_props = {str(int(pv)): float(pc / shard_data.shape[0]) for pv, pc in zip(pdg_vals, pdg_counts)}

            w_col = shard_data[:, schema.COLUMN_INDEX["w"]]
            weight_summary = {
                "min": float(np.min(w_col)),
                "max": float(np.max(w_col)),
                "mean": float(np.mean(w_col)),
                "std": float(np.std(w_col)),
                "sum": float(np.sum(w_col)),
            }

            # Quantiles for PX, PY, PZ, X, Y
            feat_quantiles = {}
            for name in ["px", "py", "pz", "x", "y"]:
                col = shard_data[:, schema.COLUMN_INDEX[name]]
                feat_quantiles[name] = {
                    str(q): float(np.quantile(col, q)) for q in [0.0, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 1.0]
                }

            # Edge buckets
            tail_masks = stratify.compute_tail_buckets(shard_data, schema.COLUMN_INDEX)
            edge_counts = {name: int(np.count_nonzero(mask)) for name, mask in tail_masks.items()}

            shard_manifest = {
                "content_dataset_hash": raw_hash,
                "source_file": os.path.abspath(args.raw_file),
                "split": split_name,
                "shard_number": int(s),
                "row_count": int(shard_data.shape[0]),
                "source_index_checksum": int(np.sum(shard_row_indices)),
                "pdg_proportions": pdg_props,
                "weight_summaries": weight_summary,
                "feature_quantiles": feat_quantiles,
                "edge_bucket_counts": edge_counts,
                "construction_seed": int(args.seed),
                "stratification_definition": stratum_table,
                "shard_hash": shard_hash,
                "indices_hash": idx_hash,
            }

            _atomic_write_json(json_path, shard_manifest, indent=2, sort_keys=True)

            shards_written.append({
                "split": split_name,
                "shard_number": s,
                "npy_file": os.path.basename(npy_path),
                "indices_file": os.path.basename(idx_path),
                "manifest_file": os.path.basename(json_path),
                "row_count": shard_data.shape[0],
                "shard_hash": shard_hash,
            })
            print(f"Split '{split_name}': shard {s + 1}/{n_shards} written ({shard_data.shape[0]} rows, "
                  f"elapsed {time.perf_counter() - t_start:.1f}s)", flush=True)
            progress(f"split_{split_name}_shard_written", shard_number=int(s), row_count=int(shard_data.shape[0]))

            # Calculate representativeness report entry
            rep_report = calculate_representativeness(
                shard_data, split_array, shard_row_indices, split_indices, schema.COLUMN_INDEX
            )
            representativeness_reports[shard_name] = rep_report

    global_manifest["shards"] = shards_written
    progress("manifest_validation_started")

    # Write global shard_manifest.json. Atomic (tmp + os.replace) so that
    # 02_validate_afterms_shards can never observe a manifest referencing
    # shards that were only partially written.
    manifest_path = os.path.join(args.shard_dir, "shard_manifest.json")
    _atomic_write_json(manifest_path, global_manifest, indent=2, sort_keys=True)
    print(f"Global shard manifest written to: {manifest_path} (elapsed {time.perf_counter() - t_start:.1f}s)", flush=True)

    # Write global shard_validation_report.json
    report_json_path = os.path.join(args.shard_dir, "shard_validation_report.json")
    validation_status = "generated"

    # Simple check for completeness/validity
    for s_name, s_rep in representativeness_reports.items():
        if s_rep["row_count_shard"] == 0:
            validation_status = "incomplete"

    report_envelope = {
        "validation_status": validation_status,
        "content_dataset_hash": raw_hash,
        "seed": args.seed,
        "shards": representativeness_reports
    }
    _atomic_write_json(report_json_path, report_envelope, indent=2, sort_keys=True)
    print(f"Validation report JSON written to: {report_json_path}", flush=True)
    progress("manifest_validation_completed")

    # Write global shard_validation_report.md
    report_md_path = os.path.join(args.shard_dir, "shard_validation_report.md")
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# SHARD REPRESENTATIVENESS VALIDATION REPORT\n\n")
        f.write(f"- **Dataset Content Hash**: `{raw_hash}`\n")
        f.write(f"- **Seed**: `{args.seed}`\n")
        f.write(f"- **Status**: `{validation_status.upper()}`\n\n")
        
        f.write("## Numeric Representativeness Summary\n\n")
        f.write("| Shard Name | Row Count | Split Row Count | Fraction | Max Abs Corr Diff | PDG +13 Diff | PDG -13 Diff |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
        for s_name in sorted(representativeness_reports.keys()):
            rep = representativeness_reports[s_name]
            p_diff = rep["pdg_fractions"]["difference"]
            diff_13 = p_diff.get("13", 0.0)
            diff_minus13 = p_diff.get("-13", 0.0)
            f.write(f"| {s_name} | {rep['row_count_shard']} | {rep['row_count_split']} | {rep['row_fraction']:.4f} | {rep['max_abs_correlation_diff']:.4f} | {diff_13:+.4f} | {diff_minus13:+.4f} |\n")
            
        f.write("\n## Standardized Mean Differences (SMD) against Parent Split\n\n")
        f.write("| Shard Name | px SMD | py SMD | pz SMD | x SMD | y SMD |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        for s_name in sorted(representativeness_reports.keys()):
            rep = representativeness_reports[s_name]
            px_smd = rep["features"]["px"]["standardized_mean_difference"]
            py_smd = rep["features"]["py"]["standardized_mean_difference"]
            pz_smd = rep["features"]["pz"]["standardized_mean_difference"]
            x_smd = rep["features"]["x"]["standardized_mean_difference"]
            y_smd = rep["features"]["y"]["standardized_mean_difference"]
            f.write(f"| {s_name} | {px_smd:+.4f} | {py_smd:+.4f} | {pz_smd:+.4f} | {x_smd:+.4f} | {y_smd:+.4f} |\n")
            
        f.write("\n## Edge Bucket Fractions Comparison\n\n")
        for s_name in sorted(representativeness_reports.keys()):
            f.write(f"### {s_name}\n\n")
            f.write("| Bucket Name | Shard Fraction | Split Fraction | Difference |\n")
            f.write("| --- | --- | --- | --- |\n")
            rep = representativeness_reports[s_name]
            for b_name in sorted(rep["edge_buckets"].keys()):
                b_info = rep["edge_buckets"][b_name]
                f.write(f"| {b_name} | {b_info['shard_fraction']:.6f} | {b_info['split_fraction']:.6f} | {b_info['difference']:+.6f} |\n")
            f.write("\n")
            
        f.write("## Leakage limitations\n\n")
        f.write("> [!IMPORTANT]\n")
        f.write("> No source-lineage/group identifier is currently available. Row-disjoint splitting does not prove source-muon independence.\n")
        
    print(f"Validation report MD written to: {report_md_path}", flush=True)
    total_elapsed = time.perf_counter() - t_start
    print(f"Shard construction completed successfully. Total elapsed: {total_elapsed:.1f}s", flush=True)
    progress("shard_construction_completed", total_elapsed_seconds=total_elapsed)


if __name__ == "__main__":
    main()
