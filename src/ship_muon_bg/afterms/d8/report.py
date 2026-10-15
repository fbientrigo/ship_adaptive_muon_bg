"""Phase F (D8 spec §14): final report assembly.

Pure aggregation over already-computed Phase A-E results -- this module
performs no I/O against D7 and runs no additional computation of its own.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any, Dict, List

NON_CLAIMS = [
    "Five-epoch smoke results establish plausible wiring, not convergence.",
    "No claim of final model superiority: champions are provisional_smoke_champion, scoped to "
    "five_epoch_single_seed_smoke, never production or downstream-acceptance claims.",
    "Production readiness is not established.",
    "Physical equivalence of generated and held-out distributions is not established: non-rejection "
    "of a distribution test never implies equality.",
    "Downstream FairShip acceptance is out of scope.",
    "Dangerous-background enrichment is out of scope.",
    "Proxy validity is out of scope.",
    "Utility-guided sampling performance is out of scope.",
    "Background rates are out of scope.",
    "Row-disjoint train/validation/test splits do not establish source-muon independence: the D7 "
    "split has no source-lineage/group identifier.",
    "Bit-matching the original D7 in-memory generated sample is not established or required: that "
    "sample was never persisted and its RNG state cannot be recovered.",
    "quantile_normal_v0 NLL is feature-space-only and is never placed on the same numerical ranking "
    "axis as physical-space NLL from identity/log1p runs.",
]

D9_RECOMMENDATIONS = [
    "Persist per-run checkpoints for Gaussian/GMM controls in future campaigns so metric-only "
    "champions in those families can also become visualization candidates.",
    "Record a source-lineage/group identifier in future shard builds so row-disjoint splits can "
    "support an actual source-muon-independence claim.",
    "Extend the weighted two-sample statistics with a verified weighted energy-distance estimator "
    "(deferred here, D8 spec §12.2) before relying on 2D weighted diagnostics.",
    "If quantile-space physical NLL is ever needed for historical runs, it must come from a NEW "
    "campaign using the forward-looking Jacobian, never retrofitted onto frozen D7 metrics.",
]


def build_report(
    *,
    audit_result: Dict[str, Any],
    records: List[Any],
    arena_result: Dict[str, Any],
    loss_semantics_notes: List[str],
    curves_manifest: Dict[str, Any],
    reconstruction_results: List[Dict[str, Any]],
    one_d_results: Dict[str, Any],
    two_d_results: Dict[str, Any],
    ndim_results: Dict[str, Any],
    real_vs_real_baselines: Dict[str, Any],
    evaluation_budgets: Dict[str, Any],
    final_status: str,
) -> Dict[str, Any]:
    return {
        "frozen_d7_identity": {
            "producer_git_commit": audit_result.get("producer_git_commit"),
            "recomputed_raw_file_sha256": audit_result.get("recomputed_raw_file_sha256"),
            "legacy_raw_file_sha256_prefix16": audit_result.get("legacy_raw_file_sha256_prefix16"),
            "content_dataset_hash": audit_result.get("content_dataset_hash"),
        },
        "historical_compatibility_contract": {
            "checkpoint_hash_method": "raw_file_bytes_sha256 (not the functional-fingerprint hash introduced at HEAD 56355d0)",
            "quantile_policy": "quantile_normal_v0 is FEATURE_SPACE_ONLY for this campaign; never converted to physical-space NLL",
            "gaussian_gmm_policy": "no historical checkpoint persisted; metrics included, samples/reconstruction not attempted",
        },
        "input_audit": audit_result,
        "run_registry": [r.to_json_dict() for r in records],
        "loss_semantics_notes": loss_semantics_notes,
        "arenas_and_exclusions": arena_result,
        "training_curves": curves_manifest,
        "provisional_champions": [
            {"arena_id": a["arena_id"], "champion": a.get("arena_champion")} for a in arena_result["arenas"]
        ],
        "reconstruction_outcomes": reconstruction_results,
        "deterministic_sample_provenance": [r["provenance"] for r in reconstruction_results if "provenance" in r],
        "visual_diagnostics": {
            "sample_matrices": [r.get("sample_matrix_path") for r in reconstruction_results if r.get("sample_matrix_path")],
            "pz_diagnostics": [r.get("pz_diagnostics_path") for r in reconstruction_results if r.get("pz_diagnostics_path")],
        },
        "one_dimensional_tests": one_d_results,
        "two_dimensional_tests": two_d_results,
        "ndimensional_c2st": ndim_results,
        "real_vs_real_baselines": real_vs_real_baselines,
        "evaluation_budgets": evaluation_budgets,
        "d9_recommendations": D9_RECOMMENDATIONS,
        "non_claims": NON_CLAIMS,
        "status": final_status,
    }


def write_report(report: Dict[str, Any], output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "afterms_smoke_arena.json").write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')

    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["run_id", "arena_id", "reconstruction_status", "validation_nll", "test_nll", "physical_space_nll"])
    arena_by_run: Dict[str, str] = {}
    for arena in report["arenas_and_exclusions"]["arenas"]:
        for c in arena.get("ranked_candidates", []):
            arena_by_run[c["run_id"]] = arena["arena_id"]
    for row in report["run_registry"]:
        writer.writerow([
            row["run_id"], arena_by_run.get(row["run_id"], ""), row["reconstruction_status"],
            row["validation_nll"], row["test_nll"], row["physical_space_nll"],
        ])
    (output_dir / "afterms_smoke_arena.csv").write_text(csv_buffer.getvalue(), encoding='utf-8')

    md = io.StringIO()
    md.write("# D8 After-MS Smoke Arena Report\n\n")
    md.write(f"**Status**: `{report['status']}`\n\n")

    md.write("## 1. Frozen D7 identity\n\n")
    for k, v in report["frozen_d7_identity"].items():
        md.write(f"- **{k}**: `{v}`\n")

    md.write("\n## 2. Historical compatibility contract\n\n")
    for k, v in report["historical_compatibility_contract"].items():
        md.write(f"- **{k}**: {v}\n")

    md.write("\n## 3. Input audit\n\n")
    md.write(f"- classification: `{report['input_audit']['classification']}`\n")
    for f in report["input_audit"].get("findings", []):
        md.write(f"- finding: {f}\n")

    md.write(f"\n## 4. Run registry\n\n{len(report['run_registry'])} runs. See `registry/run_registry.md`.\n")

    md.write("\n## 5. Loss semantics\n\n")
    for note in report["loss_semantics_notes"]:
        md.write(f"- {note}\n")

    md.write("\n## 6. Arenas and exclusions\n\nSee `arenas/model_arena.md`.\n")

    md.write("\n## 7. Training curves\n\n")
    md.write(f"- written: {len(report['training_curves'].get('written', []))}\n")
    md.write(f"- skipped (no multi-epoch history): {report['training_curves'].get('skipped_no_history', [])}\n")

    md.write("\n## 8. Provisional champions\n\n")
    for entry in report["provisional_champions"]:
        champ = entry["champion"]
        md.write(f"- **{entry['arena_id']}**: {champ['run_id'] if champ else 'none'}\n")

    md.write("\n## 9. Reconstruction outcomes\n\n")
    for r in report["reconstruction_outcomes"]:
        md.write(f"- {r.get('run_id')}: {r.get('status')}\n")

    md.write("\n## 10. Deterministic sample provenance\n\nSee `generated_samples/*.json`.\n")
    md.write("\n## 11. Visual diagnostics\n\nSee `sample_matrices/` and `pz_diagnostics/`.\n")
    md.write("\n## 12. 1D tests\n\nSee `statistics/one_dimensional_tests.json`.\n")
    md.write("\n## 13. 2D tests\n\nSee `statistics/two_dimensional_tests.json`.\n")
    md.write("\n## 14. N-D C2ST\n\nSee `statistics/ndimensional_c2st.json`.\n")
    md.write("\n## 15. Real-vs-real baselines\n\nSee `real_vs_real_baselines` in the JSON report.\n")

    md.write("\n## 16. D9 recommendations\n\n")
    for rec in report["d9_recommendations"]:
        md.write(f"- {rec}\n")

    md.write("\n## 17. Non-claims\n\n")
    for claim in report["non_claims"]:
        md.write(f"- {claim}\n")

    md.write(f"\n{report['status']}\n")
    (output_dir / "afterms_smoke_arena.md").write_text(md.getvalue(), encoding='utf-8')
