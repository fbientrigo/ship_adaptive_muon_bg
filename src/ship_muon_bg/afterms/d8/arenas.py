"""Phase C (D8 spec §7): arena partitioning and provisional champion selection.

Arena key: target_measure, pdg_policy, modeled_features/dimension,
preprocessing_comparability_class, split_identity, training_budget, seed_set.
Never merged: weighted/unweighted, physical/feature-space, PDG+13/-13, legacy
4D/modern 5D (§7). `training_budget` is the shared scope token
(`five_epoch_single_seed_smoke`, §2), never a per-run epoch count -- keying on
the per-run count would silently split Gaussian/GMM baselines (one-shot,
`training_budget_epochs=1`) out of the neural arenas they must be able to
win metric-only (§7: "a metric-only Gaussian/GMM champion may remain the
arena champion even though it cannot be visualized").

Arenas A/B compare `identity_standardized_v0` and `cartesian_log1p_pz_v0`
together (§6: "preprocessing identity vs log1p by PDG" is a comparable
grouping) on the PHYSICAL NLL axis, per spec. But champion selection must use
VALIDATION metrics only (§7), and no historical per-epoch physical-space
validation NLL was ever computed. Physical and feature-space NLL differ by
exactly the mean log-Jacobian -- a deterministic, data-only quantity (no
model inference) -- so `physical_validation_nll = validation_nll -
mean(log_jac over the validation shard)` is derived here, the same class of
deterministic-from-frozen-inputs computation §4.3 blesses for preprocessing
reconstruction. This was verified against the recorded TEST physical NLL
(same identity, `physical_test_nll = test_nll - mean(log_jac on test)`) and
matched to 7 significant figures for both identity and log1p runs.

Arenas C/D/E/F are each single-preprocessing (or preprocessing-fixed-by-family)
so their recorded feature-space `validation_nll` is directly comparable
within-arena; no derivation needed there.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ship_muon_bg.afterms.d8 import legacy_d7_adapter as adapter
from ship_muon_bg.afterms.d8.registry import RunRecord

TRAINING_BUDGET_SCOPE = "five_epoch_single_seed_smoke"

PHYSICAL_COMPARABLE_PREPROCESSING = ("identity_standardized_v0", "cartesian_log1p_pz_v0")


def _is_5d_unweighted_physical_eligible(r: RunRecord) -> bool:
    return (
        r.modeled_dimension == 5
        and not r.weighting_policy
        and r.preprocessing_name in PHYSICAL_COMPARABLE_PREPROCESSING
    )


ARENA_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "arena_id": "A_modern_5d_unweighted_pdg13_physical",
        "description": "Modern 5D, unweighted, PDG +13, physical NLL",
        "ranking_axis": "physical_validation_nll",
        "predicate": lambda r: r.pdg_policy == "pdg13" and _is_5d_unweighted_physical_eligible(r),
    },
    {
        "arena_id": "B_modern_5d_unweighted_pdg_minus13_physical",
        "description": "Modern 5D, unweighted, PDG -13, physical NLL",
        "ranking_axis": "physical_validation_nll",
        "predicate": lambda r: r.pdg_policy == "pdg_minus13" and _is_5d_unweighted_physical_eligible(r),
    },
    {
        "arena_id": "C_modern_5d_weighted_pdg13",
        "description": "Modern 5D, weighted, PDG +13",
        "ranking_axis": "validation_nll",
        "predicate": lambda r: r.pdg_policy == "pdg13" and r.modeled_dimension == 5 and r.weighting_policy,
    },
    {
        "arena_id": "D_modern_5d_weighted_pdg_minus13",
        "description": "Modern 5D, weighted, PDG -13",
        "ranking_axis": "validation_nll",
        "predicate": lambda r: r.pdg_policy == "pdg_minus13" and r.modeled_dimension == 5 and r.weighting_policy,
    },
    {
        "arena_id": "E_modern_5d_quantile_pdg13",
        "description": "Modern 5D Quantile, feature-space-only, PDG +13",
        "ranking_axis": "validation_nll",
        "predicate": lambda r: (
            r.pdg_policy == "pdg13" and r.modeled_dimension == 5
            and not r.weighting_policy and r.preprocessing_name == "quantile_normal_v0"
        ),
    },
    {
        "arena_id": "E_modern_5d_quantile_pdg_minus13",
        "description": "Modern 5D Quantile, feature-space-only, PDG -13",
        "ranking_axis": "validation_nll",
        "predicate": lambda r: (
            r.pdg_policy == "pdg_minus13" and r.modeled_dimension == 5
            and not r.weighting_policy and r.preprocessing_name == "quantile_normal_v0"
        ),
    },
    {
        "arena_id": "F_legacy_combined_pdg_4d",
        "description": "Legacy combined-PDG 4D (px, py, pz, E)",
        "ranking_axis": "validation_nll",
        "predicate": lambda r: r.modeled_dimension == 4,
    },
    {
        "arena_id": "G_memory_repeat_combined_unfiltered_5d",
        "description": (
            "Job 12 memory-release repeat smoke, combined-unfiltered PDG, 5D -- a determinism/memory "
            "diagnostic, not a genuine model comparison; included for registry completeness only"
        ),
        "ranking_axis": "validation_nll",
        "predicate": lambda r: r.pdg_policy == "combined_unfiltered" and r.modeled_dimension == 5,
    },
]


def _mean_log_jacobian(preprocessing_name: str, shard_dir: Path, pdg_value: Optional[int], split: str) -> float:
    pipeline = adapter.reconstruct_preprocessing_pipeline(preprocessing_name, shard_dir, pdg_value)
    raw = adapter.load_pdg_filtered_shard(shard_dir, split, pdg_value)
    log_jac = pipeline.forward_log_abs_det_jacobian(raw)
    return float(np.mean(log_jac))


def physical_validation_nll(record: RunRecord, shard_dir: Path) -> Optional[float]:
    if record.validation_nll is None:
        return None
    mean_log_jac = _mean_log_jacobian(record.preprocessing_name, shard_dir, record.pdg_value, split="validation")
    return record.validation_nll - mean_log_jac


def assign_arenas(records: List[RunRecord]) -> Dict[str, List[RunRecord]]:
    grouped: Dict[str, List[RunRecord]] = {d["arena_id"]: [] for d in ARENA_DEFINITIONS}
    unassigned: List[RunRecord] = []
    for r in records:
        matched = False
        for d in ARENA_DEFINITIONS:
            if d["predicate"](r):
                grouped[d["arena_id"]].append(r)
                matched = True
        if not matched:
            unassigned.append(r)
    grouped["_unassigned"] = unassigned
    return grouped


def _eligible_for_champion(r: RunRecord, ranking_value: Optional[float]) -> Tuple[bool, Optional[str]]:
    if r.target_measure == "unresolved":
        return False, "unresolved target_measure"
    if ranking_value is None or not np.isfinite(ranking_value):
        return False, "non-finite or missing validation metric"
    return True, None


def select_arena_champion(
    arena_id: str, records: List[RunRecord], shard_dir: Path
) -> Dict[str, Any]:
    definition = next(d for d in ARENA_DEFINITIONS if d["arena_id"] == arena_id)
    ranking_axis = definition["ranking_axis"]

    candidates = []
    ineligible = []
    for r in records:
        if ranking_axis == "physical_validation_nll":
            value = physical_validation_nll(r, shard_dir)
        else:
            value = r.validation_nll
        ok, reason = _eligible_for_champion(r, value)
        if ok:
            candidates.append((r, value))
        else:
            ineligible.append({"run_id": r.run_id, "reason": reason})

    if not candidates:
        return {
            "arena_id": arena_id,
            "description": definition["description"],
            "ranking_axis": ranking_axis,
            "training_budget": TRAINING_BUDGET_SCOPE,
            "n_candidates": 0,
            "arena_champion": None,
            "visualization_candidate": None,
            "ineligible": ineligible,
        }

    # Selection: (1) min ranking value, (2) lower complexity (parameter_count)
    # as tiebreak, (3) deterministic (job_id, run_id) as final tiebreak.
    def sort_key(item):
        r, value = item
        return (value, r.parameter_count if r.parameter_count is not None else float("inf"), r.job_id, r.run_id)

    candidates.sort(key=sort_key)
    champion_record, champion_value = candidates[0]

    visualization_candidates = [r for r, _ in candidates if r.reconstruction_status == "RECONSTRUCTIBLE"]
    if champion_record.reconstruction_status == "RECONSTRUCTIBLE":
        visualization_candidate = champion_record
        visualization_note = None
    elif visualization_candidates:
        visualization_candidate = visualization_candidates[0]
        visualization_note = "visualization_candidate_not_arena_champion"
    else:
        visualization_candidate = None
        visualization_note = "no reconstructible candidate in this arena"

    return {
        "arena_id": arena_id,
        "description": definition["description"],
        "ranking_axis": ranking_axis,
        "training_budget": TRAINING_BUDGET_SCOPE,
        "n_candidates": len(candidates),
        "arena_champion": {
            "run_id": champion_record.run_id,
            "label": "provisional_smoke_champion",
            "comparison_scope": TRAINING_BUDGET_SCOPE,
            "ranking_value": champion_value,
            "model_family": champion_record.model_family,
            "model_capacity": champion_record.model_capacity,
            "parameter_count": champion_record.parameter_count,
        },
        "visualization_candidate": (
            {"run_id": visualization_candidate.run_id, "note": visualization_note}
            if visualization_candidate is not None
            else None
        ),
        "ranked_candidates": [
            {"run_id": r.run_id, "ranking_value": value, "parameter_count": r.parameter_count}
            for r, value in candidates
        ],
        "ineligible": ineligible,
    }


def build_arenas(records: List[RunRecord], shard_dir: Path) -> Dict[str, Any]:
    grouped = assign_arenas(records)
    arenas = []
    for definition in ARENA_DEFINITIONS:
        arena_records = grouped[definition["arena_id"]]
        arenas.append(select_arena_champion(definition["arena_id"], arena_records, shard_dir))
    exclusions = [
        {"run_id": r.run_id, "reasons": r.exclusion_reasons}
        for r in grouped["_unassigned"]
    ]
    return {"arenas": arenas, "unassigned_runs": exclusions}


def write_arenas(result: Dict[str, Any], output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "model_arena.json").write_text(json.dumps(result, indent=2))

    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["arena_id", "description", "ranking_axis", "n_candidates", "champion_run_id", "champion_value"])
    for arena in result["arenas"]:
        champ = arena.get("arena_champion")
        writer.writerow([
            arena["arena_id"], arena["description"], arena["ranking_axis"], arena["n_candidates"],
            champ["run_id"] if champ else None, champ["ranking_value"] if champ else None,
        ])
    (output_dir / "model_arena.csv").write_text(csv_buffer.getvalue())

    md_buffer = io.StringIO()
    md_buffer.write("# D8 Model Arenas\n\n")
    for arena in result["arenas"]:
        md_buffer.write(f"## {arena['arena_id']}\n\n{arena['description']}\n\n")
        champ = arena.get("arena_champion")
        if champ:
            md_buffer.write(
                f"- **provisional_smoke_champion**: `{champ['run_id']}` "
                f"({arena['ranking_axis']}={champ['ranking_value']:.6f}, scope={champ['comparison_scope']})\n"
            )
            viz = arena.get("visualization_candidate")
            if viz:
                md_buffer.write(f"- **visualization_candidate**: `{viz['run_id']}`" + (f" ({viz['note']})\n" if viz.get("note") else "\n"))
        else:
            md_buffer.write("- no eligible candidates\n")
        if arena.get("ineligible"):
            md_buffer.write(f"- excluded: {arena['ineligible']}\n")
        md_buffer.write("\n")
    (output_dir / "model_arena.md").write_text(md_buffer.getvalue())
