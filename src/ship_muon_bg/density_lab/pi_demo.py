"""Analysis layer for the PI-ready utility-guided sampling demonstration.

Pure NumPy + stdlib (no matplotlib import here -- plotting is a separate,
downstream module). This module reads the already-built Table A / Table B
reference tables produced by ``scripts/run_utility_tilt_campaign.py`` (see
``density_lab.utility_tilt``) and extracts one summary CSV row plus a run
manifest per ``(pdg_id, tilt_id)`` pair.

``B_toy`` is a project-defined synthetic diagnostic region (thresholds on
``pT`` and ``R_xy``), never a validated FairShip danger metric. ``U_A`` and
``U_P`` are synthetic utility scores, never calibrated probabilities. Nothing
in this module introduces a pass/fail threshold: every quantity computed here
is diagnostic-only, matching ``utility_tilt.py``'s own contract.

Terminology, reused verbatim from ``utility_tilt.py`` and never renamed:
``w_mc`` is a physical MC weight; ``r_utility`` is a synthetic utility
multiplier ``h(U)`` and must never be called or labeled a physical weight;
``pi_tilt`` is the tilted sampling probability.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import environment
from .artifacts import _default
from .utility_tilt import (
    AliasSampler,
    NominalTable,
    TiltConfig,
    TiltTable,
    build_tilt_grid,
    empirical_draw_diagnostics,
    nominal_prevalence,
    theoretical_concentration_diagnostics,
    theoretical_tilted_prevalence,
)

SCHEMA_VERSION = "0"

# Single source of truth for the NF-arena caveat, reused by both the
# manifest writer and any downstream plotting/README text -- never
# hand-copied. Revised per review round 1 (BLOCKER 2): the flow does not
# reliably track the mild tilts at the sample budget used here, so the
# caveat says so explicitly rather than implying uniform agreement.
NF_ARENA_CAVEAT = (
    "Bounded CPU-scale affine-coupling flow (8 blocks, width 96, <=40 epochs, "
    "early stopping), not a converged production model. generated_b_toy_occupancy "
    "is measured from a small sample (see generated_sample_count per run, typically "
    "4000) drawn from the trained flow and mapped back to physical space. For the "
    "strong tilts it approaches but does not exactly reach the declared "
    "theoretical_target_b_toy_prevalence, which is expected at this training budget. "
    "For the mild tilts, the occupancy measurement is not reliably distinguishable "
    "from sampling noise at this sample budget (binomial standard error is a "
    "sizeable fraction of the signal) and does not consistently track the tilt's "
    "declared direction -- this is a stated NF-training-budget limitation, not "
    "evidence of a sampler or mathematics bug (those are validated separately and "
    "exactly via the closed-form/direct-summation/alias-sampler path)."
)

# Frozen CSV schema. A downstream plotting module depends on this exact
# name/order -- never rename or reorder these columns.
CSV_FIELDNAMES: Tuple[str, ...] = (
    "pdg_id", "tilt_id", "mode", "delta", "alpha", "n_rows",
    "p0", "pU_closed_form", "pU_direct", "empirical_pU", "enrichment",
    "n_eff", "n_eff_fraction", "expected_n_unique", "expected_unique_fraction",
    "max_row_probability", "top_10_mass", "top_100_mass",
    "draw_budget", "observed_unique_rows", "max_reuse_count",
    "nf_run_id", "nf_b_toy_fraction", "nf_minus_target",
    "nominal_validation_nll", "tilted_validation_nll",
)

# NF-facing columns are not populated by this (tables-only) analysis layer.
# They are set to Python ``None`` -- never the string "NA" -- so that
# ``csv.DictWriter`` renders them as an empty field, per house convention
# (see ``reporting.build_summary_tables``, which follows the same idiom).
_NF_PLACEHOLDER_COLUMNS: Tuple[str, ...] = (
    "nf_run_id", "nf_b_toy_fraction", "nf_minus_target",
    "nominal_validation_nll", "tilted_validation_nll",
)


def load_pdg_tables(tables_dir: Path) -> Tuple[NominalTable, TiltTable]:
    """Load Table A and Table B for one PDG track from the same directory."""

    tables_dir = Path(tables_dir)
    table_a = NominalTable.load(tables_dir)
    table_b = TiltTable.load(tables_dir)
    return table_a, table_b


def compute_summary_row(
    table_a: NominalTable,
    table_b: TiltTable,
    tilt_config: TiltConfig,
    *,
    draw_budget: int,
    sampler_seed: int,
    top_k: Sequence[int] = (10, 100),
) -> Tuple[Dict[str, Any], float]:
    """One frozen-schema CSV row for one ``(pdg_id, tilt_id)`` pair.

    Returns ``(row_dict, abs_closed_vs_direct)`` -- the second element is a
    validation-only diagnostic, not part of the CSV schema; callers collect
    it across rows to drive the closed-form-vs-direct validation gate.
    """

    n_rows = table_a.n_rows
    block_mask = table_b.block(tilt_config.tilt_id)
    pi_tilt_block = np.asarray(table_b.pi_tilt, dtype=np.float64)[block_mask]

    p0 = nominal_prevalence(table_a.B_toy, table_a.w_mc)
    pU_closed_form = theoretical_tilted_prevalence(p0, tilt_config.r_positive(), tilt_config.r_negative())
    pU_direct = float(np.sum(pi_tilt_block * table_a.B_toy))
    enrichment = (pU_direct / p0) if p0 != 0.0 else float("nan")

    alias_sampler = AliasSampler.from_probabilities(pi_tilt_block)
    drawn_indices = alias_sampler.draw(int(draw_budget), seed=int(sampler_seed))
    draw_diag = empirical_draw_diagnostics(drawn_indices, b_toy=table_a.B_toy)
    empirical_pU = draw_diag["empirical_b_toy_fraction"]

    concentration = theoretical_concentration_diagnostics(
        pi_tilt_block, draw_budget=int(draw_budget), top_k=top_k
    )
    n_eff = float(concentration["n_eff"])
    expected_n_unique = float(concentration["expected_n_unique"])
    n_eff_fraction = (n_eff / n_rows) if n_rows else float("nan")
    expected_unique_fraction = (expected_n_unique / n_rows) if n_rows else float("nan")

    row: Dict[str, Any] = {
        "pdg_id": int(table_a.pdg_id),
        "tilt_id": tilt_config.tilt_id,
        "mode": tilt_config.mode,
        "delta": float(tilt_config.delta),
        "alpha": float(tilt_config.alpha),
        "n_rows": int(n_rows),
        "p0": float(p0),
        "pU_closed_form": float(pU_closed_form),
        "pU_direct": float(pU_direct),
        "empirical_pU": empirical_pU,
        "enrichment": float(enrichment),
        "n_eff": n_eff,
        "n_eff_fraction": float(n_eff_fraction),
        "expected_n_unique": expected_n_unique,
        "expected_unique_fraction": float(expected_unique_fraction),
        "max_row_probability": float(concentration["max_row_probability"]),
        "top_10_mass": float(concentration["top_10_mass"]),
        "top_100_mass": float(concentration["top_100_mass"]),
        "draw_budget": int(draw_budget),
        "observed_unique_rows": int(draw_diag["unique_rows_drawn"]),
        "max_reuse_count": int(draw_diag["max_reuse_count"]),
        "nf_run_id": None,
        "nf_b_toy_fraction": None,
        "nf_minus_target": None,
        "nominal_validation_nll": None,
        "tilted_validation_nll": None,
    }
    abs_closed_vs_direct = abs(pU_closed_form - pU_direct)
    return row, abs_closed_vs_direct


def build_summary_table(
    tables_root: Path,
    pdg_ids: Sequence[int],
    *,
    draw_budget: int,
    sampler_seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Summary rows for every ``(pdg_id, tilt_id)`` pair, PDGs never pooled.

    Each ``pdg_id`` in ``pdg_ids`` is processed independently: its own
    directory is loaded, its own rows are computed and appended, and no
    array is ever concatenated or averaged across PDGs. This is why
    ``validation_gate["all_pdg_kept_separate"]`` is unconditionally ``True``
    -- it documents an invariant of this function's control flow, not a
    runtime check.
    """

    tables_root = Path(tables_root)
    all_rows: List[Dict[str, Any]] = []
    pi_nominal_sums_to_one: Dict[Any, bool] = {}
    pi_tilt_sums_to_one: Dict[Any, Dict[str, bool]] = {}
    abs_closed_vs_direct_values: List[float] = []

    for pdg_id in pdg_ids:
        pdg_dir = tables_root / "pdg_{}".format(pdg_id)
        table_a, table_b = load_pdg_tables(pdg_dir)

        pi_nominal_sums_to_one[pdg_id] = bool(
            abs(float(np.sum(table_a.pi_nominal)) - 1.0) < 1e-9
        )

        manifest_b_path = pdg_dir / "table_b_tilt.manifest.json"
        pi_sum_per_tilt = None
        if manifest_b_path.exists():
            manifest_b = json.loads(manifest_b_path.read_text())
            pi_sum_per_tilt = manifest_b.get("pi_tilt_sum_per_tilt")

        tilt_sums: Dict[str, bool] = {}
        for tilt_config in build_tilt_grid():
            row, abs_diff = compute_summary_row(
                table_a, table_b, tilt_config,
                draw_budget=draw_budget, sampler_seed=sampler_seed,
            )
            all_rows.append(row)
            abs_closed_vs_direct_values.append(abs_diff)

            if pi_sum_per_tilt is not None and tilt_config.tilt_id in pi_sum_per_tilt:
                pi_sum = float(pi_sum_per_tilt[tilt_config.tilt_id])
            else:
                block_mask = table_b.block(tilt_config.tilt_id)
                pi_sum = float(np.sum(np.asarray(table_b.pi_tilt, dtype=np.float64)[block_mask]))
            tilt_sums[tilt_config.tilt_id] = bool(abs(pi_sum - 1.0) < 1e-6)

        pi_tilt_sums_to_one[pdg_id] = tilt_sums

    validation_gate: Dict[str, Any] = {
        "pi_nominal_sums_to_one": pi_nominal_sums_to_one,
        "pi_tilt_sums_to_one": pi_tilt_sums_to_one,
        "max_abs_closed_vs_direct": (
            float(max(abs_closed_vs_direct_values)) if abs_closed_vs_direct_values else float("nan")
        ),
        "all_pdg_kept_separate": True,
    }
    return all_rows, validation_gate


def write_summary_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    """Write ``rows`` using the frozen :data:`CSV_FIELDNAMES` schema."""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(CSV_FIELDNAMES), extrasaction="ignore", lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_run_manifest(
    *,
    out_path: Path,
    dataset_path: str,
    dataset_hash: str,
    dataset_scope: str,
    pdg_ids: Sequence[int],
    tables_root: Path,
    seed: int,
    sampler_seed: int,
    draw_budget_per_pdg: Mapping[Any, int],
    validation_gate: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build and write ``run_manifest.json``; returns the payload written.

    ``draw_budget_per_pdg`` is required as a per-track mapping, never a
    single scalar: PDG tracks have different ``n_rows`` (this dataset's two
    tracks differ by 272 rows) and the draw budget follows ``n_train``, so a
    single top-level "draw_budget" field would silently misreport every
    track but one (review round 1, finding O1).
    """

    if dataset_scope != "sample":
        raise ValueError(
            "dataset_scope must be the literal string 'sample' (this is a "
            "committed sample, never the full parent dataset); got {!r}".format(dataset_scope)
        )
    if not pdg_ids:
        raise ValueError("pdg_ids must be non-empty (an empty run has no validation_gate to report)")

    tables_root = Path(tables_root)
    per_pdg: Dict[str, Any] = {}
    for pdg_id in pdg_ids:
        pdg_dir = tables_root / "pdg_{}".format(pdg_id)
        manifest_a = json.loads((pdg_dir / "table_a_nominal.manifest.json").read_text())
        manifest_b = json.loads((pdg_dir / "table_b_tilt.manifest.json").read_text())
        per_pdg[str(pdg_id)] = {
            "n_rows": manifest_a["n_rows"],
            "table_a_hash": manifest_a["table_hash"],
            "table_b_hash": manifest_b["table_hash"],
            "split_hash": manifest_a["split_hash"],
        }

    env = environment.capture_environment()
    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset_path": dataset_path,
        "dataset_hash": dataset_hash,
        "dataset_scope": dataset_scope,
        "pdg_ids": list(pdg_ids),
        "per_pdg": per_pdg,
        "seed": int(seed),
        "sampler_seed": int(sampler_seed),
        "draw_budget_per_pdg": {str(k): int(v) for k, v in draw_budget_per_pdg.items()},
        "code_commit": env.get("git_commit"),
        "environment": env,
        "validation_gate": validation_gate,
        "generated_at": environment.utc_timestamp(),
        "terminology": {
            "w_mc": "physical MC weight",
            "r_utility": "utility multiplier h(U), never a physical weight",
            "pi_tilt": "tilted sampling probability",
        },
    }
    if extra:
        payload["extra"] = extra

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_default))
    return payload


def _run_pi_demo_v0() -> None:
    """One-off integration run against the already-built ``pi_demo_v0`` tables.

    Not a CLI (that is a separate, downstream ``scripts/run_pi_demo.py``
    task). This just wires the functions above together against real data
    and writes the two artifacts under ``artifacts/pi_demo/``.
    """

    repo_root = Path(__file__).resolve().parents[3]
    tables_root = repo_root / "artifacts" / "density_lab" / "pi_demo_v0" / "tables"
    out_dir = repo_root / "artifacts" / "pi_demo"
    pdg_ids = [13, -13]
    sampler_seed = 7
    seed = 11
    dataset_path = "data/samples/muonsFullMC_afterMS_sample.npz"

    dataset_hashes = set()
    all_rows: List[Dict[str, Any]] = []
    merged_gate: Dict[str, Any] = {
        "pi_nominal_sums_to_one": {},
        "pi_tilt_sums_to_one": {},
        "max_abs_closed_vs_direct": float("-inf"),
        "all_pdg_kept_separate": True,
    }
    per_pdg_draw_budget: Dict[Any, int] = {}

    for pdg_id in pdg_ids:
        pdg_dir = tables_root / "pdg_{}".format(pdg_id)
        manifest_a = json.loads((pdg_dir / "table_a_nominal.manifest.json").read_text())
        dataset_hashes.add(manifest_a["source_dataset_hash"])
        draw_budget = int(manifest_a["n_rows"])
        per_pdg_draw_budget[pdg_id] = draw_budget

        rows, gate = build_summary_table(
            tables_root, [pdg_id], draw_budget=draw_budget, sampler_seed=sampler_seed,
        )
        all_rows.extend(rows)
        merged_gate["pi_nominal_sums_to_one"].update(gate["pi_nominal_sums_to_one"])
        merged_gate["pi_tilt_sums_to_one"].update(gate["pi_tilt_sums_to_one"])
        merged_gate["max_abs_closed_vs_direct"] = max(
            merged_gate["max_abs_closed_vs_direct"], gate["max_abs_closed_vs_direct"]
        )

    if len(dataset_hashes) != 1:
        raise ValueError(
            "source_dataset_hash disagrees across PDG table manifests: {}".format(sorted(dataset_hashes))
        )
    dataset_hash = next(iter(dataset_hashes))

    csv_path = out_dir / "utility_guided_summary.csv"
    manifest_path = out_dir / "run_manifest.json"
    write_summary_csv(all_rows, csv_path)
    write_run_manifest(
        out_path=manifest_path,
        dataset_path=dataset_path,
        dataset_hash=dataset_hash,
        dataset_scope="sample",
        pdg_ids=pdg_ids,
        tables_root=tables_root,
        seed=seed,
        sampler_seed=sampler_seed,
        draw_budget_per_pdg=per_pdg_draw_budget,
        validation_gate=merged_gate,
    )

    print("wrote {} rows to {}".format(len(all_rows), csv_path))
    print("pi_nominal_sums_to_one:", merged_gate["pi_nominal_sums_to_one"])
    print("max_abs_closed_vs_direct:", merged_gate["max_abs_closed_vs_direct"])
    print("manifest:", manifest_path)


def backfill_nf_arena_results(
    *,
    csv_path: Path,
    manifest_path: Path,
    arena_report_paths: Mapping[Any, Path],
    nf_config_path: str,
) -> Dict[str, Any]:
    """Merge trained-NF results from arena reports into an existing summary CSV/manifest.

    This is the ONE committed mechanism that populates the CSV's ``nf_*``
    columns and the manifest's ``nf_arena`` block -- there is no other path
    that does this (review round 1, BLOCKER 1: those values were previously
    inserted by an uncommitted ad-hoc script). Idempotent: rerunning against
    the same arena reports overwrites the same rows with the same values.

    Only rows whose ``variant_id`` is not ``"NOMINAL_PHYSICAL"`` are used --
    that variant is the untilted arm and is not one of the 20 tilt_id rows
    in the summary CSV.
    """

    csv_path = Path(csv_path)
    manifest_path = Path(manifest_path)

    nf_by_pdg_tilt: Dict[Tuple[int, str], Dict[str, Any]] = {}
    trained_variants: List[str] = []
    for pdg_id, arena_path in arena_report_paths.items():
        payload = json.loads(Path(arena_path).read_text())
        for row in payload["rows"]:
            variant_id = row["variant_id"]
            if variant_id not in trained_variants:
                trained_variants.append(variant_id)
            if variant_id == "NOMINAL_PHYSICAL":
                continue
            nf_by_pdg_tilt[(int(pdg_id), variant_id)] = {
                "nf_run_id": row["run_id"],
                "nf_b_toy_fraction": row["generated_b_toy_occupancy"],
                "nf_minus_target": row["generated_b_toy_occupancy"] - row["theoretical_target_b_toy_prevalence"],
                "nominal_validation_nll": row["nominal_validation_nll_weighted_by_w"],
                "tilted_validation_nll": row["tilted_validation_nll_weighted_by_w_times_r"],
            }

    with csv_path.open("r", newline="") as handle:
        rows = list(csv.DictReader(handle))

    n_backfilled = 0
    for row in rows:
        key = (int(row["pdg_id"]), row["tilt_id"])
        if key in nf_by_pdg_tilt:
            row.update(nf_by_pdg_tilt[key])
            n_backfilled += 1

    expected = len(nf_by_pdg_tilt)
    if n_backfilled != expected:
        raise ValueError(
            "expected to backfill exactly {} rows (one per (pdg, trained tilt) pair "
            "found in the arena reports), matched {} -- summary CSV and arena reports "
            "disagree on pdg_id/tilt_id, refusing a silent partial backfill".format(
                expected, n_backfilled
            )
        )

    write_summary_csv(rows, csv_path)

    manifest = json.loads(manifest_path.read_text())
    manifest["nf_arena"] = {
        "status": "SUPPORTED",
        "config": nf_config_path,
        "trained_variants": trained_variants,
        "arena_report_paths": {str(k): str(v) for k, v in arena_report_paths.items()},
        "caveat": NF_ARENA_CAVEAT,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=_default))

    return {"n_backfilled": n_backfilled, "trained_variants": trained_variants}


if __name__ == "__main__":
    # Both halves of this entry point resolve paths the same way -- via
    # _REPO_ROOT, not the process cwd -- so `python -m
    # ship_muon_bg.density_lab.pi_demo` behaves identically regardless of
    # where it is invoked from (review round 1, finding NEW-2: the two
    # halves previously disagreed, cwd-independent vs cwd-relative).
    _REPO_ROOT = Path(__file__).resolve().parents[3]
    _run_pi_demo_v0()
    backfill_nf_arena_results(
        csv_path=_REPO_ROOT / "artifacts/pi_demo/utility_guided_summary.csv",
        manifest_path=_REPO_ROOT / "artifacts/pi_demo/run_manifest.json",
        arena_report_paths={
            13: _REPO_ROOT / "artifacts/density_lab/pi_demo_v0/arena_report/pdg_13/arena_summary.json",
            -13: _REPO_ROOT / "artifacts/density_lab/pi_demo_v0/arena_report/pdg_-13/arena_summary.json",
        },
        nf_config_path="configs/density_lab/utility_tilt/pi_demo_cpu_v0.json",
    )
    print("backfilled NF arena results into the summary CSV and run manifest")
