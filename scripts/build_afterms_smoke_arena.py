#!/usr/bin/env python3
"""build_afterms_smoke_arena.py: D8 read-only after-MS smoke arena CLI.

Consumes the frozen D7 campaign (`artifacts/afterms_nightly_v1`,
`data/shards/afterms_nightly_v1`) read-only and writes evaluation-only
outputs under `--output-dir`. Never calls D7 queue jobs, optimizer code,
generative `fit`, Gaussian/GMM fit, shard construction, or raw-data split
construction. Fitting the diagnostic C2ST classifier is the one exception
(§13).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))  # Nflow/ lives at repo root, not under src/

from ship_muon_bg.afterms.d8 import legacy_d7_adapter as adapter
from ship_muon_bg.afterms.d8 import registry as d8_registry
from ship_muon_bg.afterms.d8 import arenas as d8_arenas
from ship_muon_bg.afterms.d8 import plotting as d8_plotting
from ship_muon_bg.afterms.d8 import reconstruction as d8_reconstruction
from ship_muon_bg.afterms.d8 import statistics as d8_statistics
from ship_muon_bg.afterms.d8 import report as d8_report

DEFAULT_SEED = 20260720
DEFAULT_SAMPLE_SIZE = 20000
DEFAULT_ENERGY_SAMPLE_SIZE = 2000
DEFAULT_PERMUTATIONS = 199
DEFAULT_BOOTSTRAP_REPETITIONS = 300
DEFAULT_C2ST_SAMPLE_SIZE = 10000


def _infer_raw_pkl_repo_root(input_artifact_dir: Path) -> Path:
    """The raw pkl lives at `<repo_root>/data/raw/nflow_releases/...`; the
    campaign's artifacts live at `<repo_root>/artifacts/<campaign_name>`. This
    walks back from `--input-artifact-dir` instead of assuming the script's
    own on-disk location, so a caller pointing at a different checkout (or a
    test fixture root) still resolves the raw file relative to ITS OWN tree."""

    return Path(input_artifact_dir).resolve().parent.parent


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="D8 read-only after-MS smoke arena builder")
    parser.add_argument("--input-artifact-dir", default=str(REPO_ROOT / "artifacts" / "afterms_nightly_v1"))
    parser.add_argument("--shard-dir", default=str(REPO_ROOT / "data" / "shards" / "afterms_nightly_v1"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "artifacts" / "afterms_d8_evaluation_v0"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="overwrite an existing non-empty --output-dir")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--energy-sample-size", type=int, default=DEFAULT_ENERGY_SAMPLE_SIZE)
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--bootstrap-repetitions", type=int, default=DEFAULT_BOOTSTRAP_REPETITIONS)
    parser.add_argument("--c2st-sample-size", type=int, default=DEFAULT_C2ST_SAMPLE_SIZE)
    return parser.parse_args(argv)


def _dry_run_report(args: argparse.Namespace) -> int:
    input_artifact_dir = Path(args.input_artifact_dir)
    shard_dir = Path(args.shard_dir)

    audit_result = adapter.run_phase_a_audit(_infer_raw_pkl_repo_root(input_artifact_dir), input_artifact_dir, shard_dir)
    report: dict = {"phase_a_audit": audit_result}

    if audit_result["classification"] == adapter.INPUT_INCONSISTENT:
        report["status"] = "AFTERMS_SMOKE_ARENA_BLOCKED_BY_INPUTS"
        print(json.dumps(report, indent=2))
        return 2

    records = d8_registry.build_registry(input_artifact_dir, shard_dir)
    arena_result = d8_arenas.build_arenas(records, shard_dir)

    report["discovered_runs"] = [r.run_id for r in records]
    report["legacy_adapter_mapping"] = {
        r.run_id: {
            "model_family": r.model_family,
            "checkpoint_path": r.checkpoint_path,
            "checkpoint_kind": r.checkpoint_kind,
            "reconstruction_status": r.reconstruction_status,
        }
        for r in records
    }
    report["arenas"] = [
        {"arena_id": a["arena_id"], "n_candidates": a["n_candidates"], "champion": (a["arena_champion"] or {}).get("run_id")}
        for a in arena_result["arenas"]
    ]
    report["exclusions"] = arena_result["unassigned_runs"] + [
        {"run_id": r.run_id, "reasons": r.exclusion_reasons} for r in records if r.exclusion_reasons
    ]
    report["reconstruction_plan"] = [
        {"run_id": r.run_id, "reconstruction_status": r.reconstruction_status, "reconstruction_scope": r.reconstruction_scope}
        for r in records
    ]
    report["visualization_candidates"] = [
        (a["visualization_candidate"] or {}).get("run_id") for a in arena_result["arenas"] if a.get("visualization_candidate")
    ]
    report["evaluation_budgets"] = {
        "sample_size": args.sample_size,
        "energy_sample_size": args.energy_sample_size,
        "permutations": args.permutations,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "c2st_sample_size": args.c2st_sample_size,
    }
    report["expected_outputs"] = [
        "audit/immutable_input_manifest.json", "audit/immutable_input_audit.md", "audit/legacy_d7_contract.json",
        "registry/run_registry.{json,csv,md}", "arenas/model_arena.{json,csv,md}",
        "training_curves/loss_curve__<run_id>.png", "generated_samples/<run_id>.{npy,json}",
        "reference_samples/<arena_id>.{indices.npy,json}", "sample_matrices/sample_matrix__<run_id>.png",
        "pz_diagnostics/pz_diagnostics__<run_id>.png", "statistics/*.{json,csv}",
        "report/afterms_smoke_arena.{md,json,csv}",
    ]
    report["status"] = (
        "AFTERMS_SMOKE_ARENA_PARTIAL"
        if audit_result["classification"] == adapter.INPUT_PARTIAL
        else "DRY_RUN_OK"
    )
    print(json.dumps(report, indent=2))
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.dry_run:
        return _dry_run_report(args)

    input_artifact_dir = Path(args.input_artifact_dir)
    shard_dir = Path(args.shard_dir)
    output_dir = Path(args.output_dir)

    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        print(f"refusing to write into non-empty {output_dir} without --force", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Phase A ---
    audit_result = adapter.run_phase_a_audit(_infer_raw_pkl_repo_root(input_artifact_dir), input_artifact_dir, shard_dir)
    adapter.write_phase_a_outputs(audit_result, output_dir / "audit")
    if audit_result["classification"] == adapter.INPUT_INCONSISTENT:
        print("AFTERMS_SMOKE_ARENA_BLOCKED_BY_INPUTS")
        return 2

    # --- Phase B ---
    records = d8_registry.build_registry(input_artifact_dir, shard_dir)
    d8_registry.write_registry(records, output_dir / "registry")

    # --- Phase C ---
    arena_result = d8_arenas.build_arenas(records, shard_dir)
    d8_arenas.write_arenas(arena_result, output_dir / "arenas")

    # --- Phase D: reconstruction + deterministic samples (§8-§10) ---
    records_by_id = {r.run_id: r for r in records}
    jobs_dir = input_artifact_dir / "jobs"
    reconstruction_outcomes = []
    one_d_results, two_d_results, ndim_results, real_vs_real_baselines = {}, {}, {}, {}

    for arena in arena_result["arenas"]:
        arena_id = arena["arena_id"]
        viz = arena.get("visualization_candidate")
        if not viz:
            continue
        record = records_by_id[viz["run_id"]]

        try:
            result = d8_reconstruction.reconstruct_and_generate(
                record, jobs_dir=jobs_dir, shard_dir=shard_dir,
                sample_count=args.sample_size, device="cpu",
            )
        except d8_reconstruction.ReconstructionError as exc:
            print(f"WARNING: reconstruction failed for {record.run_id}: {exc}", file=sys.stderr)
            reconstruction_outcomes.append({"run_id": record.run_id, "arena_id": arena_id, "status": "failed", "error": str(exc)})
            continue

        d8_reconstruction.write_generated_samples(result, output_dir / "generated_samples")
        d8_reconstruction.write_reference_sample(
            arena_id, record.pdg_value, shard_dir, args.sample_size, args.seed,
            output_dir / "reference_samples",
        )
        outcome = {"run_id": record.run_id, "arena_id": arena_id, "status": "reconstructed"}
        reconstruction_outcomes.append(outcome)

        if record.modeled_dimension != 5:
            outcome["note"] = "modeled_dimension != 5 (legacy px,py,pz,E); x/y-involving diagnostics do not apply and are skipped"
            continue

        # --- Phase E: statistics + visual diagnostics (only for 5D runs) ---
        indices = d8_reconstruction.select_reference_indices(shard_dir, record.pdg_value, args.sample_size, args.seed)
        reference = d8_reconstruction.load_reference_rows(shard_dir, record.pdg_value, indices)
        generated_samples = result["samples"]

        weights = None
        if record.weighting_policy:
            filtered = adapter.load_pdg_filtered_shard(shard_dir, "test", record.pdg_value)
            weights = filtered[indices][:, 7]  # schema COLUMN_INDEX["w"]

        one_d_results[record.run_id] = d8_statistics.run_1d_suite(
            reference, generated_samples, weighted=record.weighting_policy, weights=weights,
            n_boot=args.bootstrap_repetitions, seed=args.seed,
        )
        two_d_results[record.run_id] = d8_statistics.run_2d_suite(
            reference, generated_samples, weighted=record.weighting_policy,
            energy_sample_size=args.energy_sample_size, n_permutations=args.permutations, seed=args.seed,
        )
        ndim_results[record.run_id] = d8_statistics.ndim_c2st(
            reference, generated_samples, sample_size=args.c2st_sample_size, seed=args.seed,
            n_boot=min(30, args.bootstrap_repetitions), n_permutations=min(30, args.permutations),
        )

        subsets = d8_reconstruction.real_vs_real_disjoint_subsets(shard_dir, record.pdg_value, args.sample_size, args.seed)
        subset_a = d8_reconstruction.load_reference_rows(shard_dir, record.pdg_value, subsets["subset_a"])
        subset_b = d8_reconstruction.load_reference_rows(shard_dir, record.pdg_value, subsets["subset_b"])
        real_vs_real_baselines[record.run_id] = {
            "one_dimensional": {
                var: d8_statistics.real_vs_real_baseline_1d(
                    subset_a[:, d8_statistics.FEATURE_INDEX[var]], subset_b[:, d8_statistics.FEATURE_INDEX[var]],
                    n_boot=min(30, args.bootstrap_repetitions), seed=args.seed,
                )
                for var in d8_statistics.VARIABLES_1D
            },
            "ndimensional_c2st": d8_statistics.real_vs_real_baseline_ndim_c2st(
                subset_a, subset_b, sample_size=args.c2st_sample_size, seed=args.seed,
                n_boot=min(30, args.bootstrap_repetitions), n_permutations=min(30, args.permutations),
            ),
        }

        sample_matrix_path = d8_plotting.plot_sample_matrix(
            record.run_id, reference, generated_samples, output_dir / "sample_matrices" / f"sample_matrix__{record.run_id}.png",
        )
        pz_path = d8_plotting.plot_pz_diagnostics(
            record.run_id, reference, generated_samples, output_dir / "pz_diagnostics" / f"pz_diagnostics__{record.run_id}.png",
        )
        outcome["sample_matrix_path"] = str(sample_matrix_path)
        outcome["pz_diagnostics_path"] = str(pz_path)

    (output_dir / "statistics").mkdir(parents=True, exist_ok=True)
    (output_dir / "statistics" / "one_dimensional_tests.json").write_text(json.dumps(one_d_results, indent=2))
    (output_dir / "statistics" / "two_dimensional_tests.json").write_text(json.dumps(two_d_results, indent=2))
    (output_dir / "statistics" / "ndimensional_c2st.json").write_text(json.dumps(ndim_results, indent=2))

    curves_manifest = d8_plotting.write_all_curves(records, output_dir / "training_curves")

    final_status = (
        "AFTERMS_SMOKE_ARENA_PARTIAL" if audit_result["classification"] == adapter.INPUT_PARTIAL
        else "AFTERMS_SMOKE_ARENA_COMPLETE"
    )
    report = d8_report.build_report(
        audit_result=audit_result,
        records=records,
        arena_result=arena_result,
        loss_semantics_notes=[
            "Test-set NLL is always row-empirical, even for weighted-training runs (the producer's "
            "test evaluation never applies sample weights).",
            "Per-epoch validation_loss for weighted runs is a self_normalized_minibatch_ratio "
            "quantity; validation_nll for those runs is weighted even though test_nll is not.",
            "physical_validation_nll (arenas A/B) is derived as validation_nll minus the mean "
            "log-Jacobian on the validation shard -- verified against the recorded test physical "
            "NLL to 7 significant figures.",
        ],
        curves_manifest=curves_manifest,
        reconstruction_results=reconstruction_outcomes,
        one_d_results=one_d_results,
        two_d_results=two_d_results,
        ndim_results=ndim_results,
        real_vs_real_baselines=real_vs_real_baselines,
        evaluation_budgets={
            "sample_size": args.sample_size, "energy_sample_size": args.energy_sample_size,
            "permutations": args.permutations, "bootstrap_repetitions": args.bootstrap_repetitions,
            "c2st_sample_size": args.c2st_sample_size,
        },
        final_status=final_status,
    )
    d8_report.write_report(report, output_dir / "report")

    print(final_status)
    return 0


if __name__ == "__main__":
    sys.exit(main())
