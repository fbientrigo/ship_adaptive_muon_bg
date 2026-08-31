"""Multi-seed robustness check for the PI-demo NF arena (Figure 05's data).

Answers one question: is an observed per-seed NF result (e.g. one PDG
track's mild-tilt ``generated_b_toy_occupancy`` undershooting its target)
a real, seed-independent signal, or an artifact of that one training run's
particular initialization/optimization trajectory ("seed luck")? A single
seed cannot distinguish these; this module aggregates the SAME arena
(NOMINAL_PHYSICAL + 4 named tilts, both PDG tracks) already trained
independently at several seeds and reports the seed-to-seed spread.

This is a diagnostic, not a P0 deliverable: it does not replace, retrain,
or modify any figure/CSV/manifest under ``artifacts/pi_demo/`` -- it reads
additional arena reports (one per seed, produced by the unmodified
``scripts/run_utility_tilt_campaign.py --arena-report`` CLI) and writes its
own small, separate summary under ``artifacts/pi_demo/seed_robustness/``.

Pure NumPy + stdlib (no matplotlib, no torch) -- like ``pi_demo.py``, this
module only reads already-produced arena JSON and reports it; it contains
no training code and no tilt mathematics of its own.

``B_toy`` is a project-defined SYNTHETIC diagnostic region, never a
validated FairShip danger metric; ``U_A``/``U_P`` are synthetic utility
scores, never calibrated probabilities.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

SCHEMA_VERSION = "0"

# Matches the existing multi-seed convention already used elsewhere in this
# repository (``conditional_utility.CONDITIONAL_UTILITY_MODEL_SEEDS``) --
# reused rather than inventing a new seed set.
DEFAULT_SEEDS: Tuple[int, ...] = (11, 12, 13)

CSV_FIELDNAMES: Tuple[str, ...] = (
    "pdg_id", "variant_id", "n_seeds", "seeds",
    "p0_mean", "p0_std",
    "pU_mean", "pU_std",
    "q_theta_mean", "q_theta_std", "q_theta_min", "q_theta_max",
    "q_theta_values_by_seed",
    "all_seeds_same_side_of_p0",
)


def load_arena_rows(arena_report_path: Path) -> Dict[str, Dict[str, Any]]:
    """One ``arena_summary.json`` -> ``{variant_id: row}``."""

    payload = json.loads(Path(arena_report_path).read_text())
    return {row["variant_id"]: row for row in payload["rows"]}


def aggregate_seed_robustness(
    *,
    pdg_ids: Sequence[int],
    variant_ids: Sequence[str],
    arena_report_path_by_pdg_and_seed: Mapping[Tuple[int, int], Path],
) -> List[Dict[str, Any]]:
    """One row per ``(pdg_id, variant_id)``, aggregated across every seed present.

    ``arena_report_path_by_pdg_and_seed`` maps ``(pdg_id, seed) ->
    arena_summary.json path``; every (pdg, seed) pair must be present for
    every requested ``pdg_ids`` -- a partial seed set for one PDG track
    would silently understate its spread, so this raises rather than
    proceeding with an uneven seed count per track.
    """

    seeds_by_pdg: Dict[int, List[int]] = {}
    for pdg_id, seed in arena_report_path_by_pdg_and_seed:
        seeds_by_pdg.setdefault(pdg_id, []).append(seed)
    for pdg_id in pdg_ids:
        if pdg_id not in seeds_by_pdg:
            raise ValueError("no arena reports supplied for pdg_id={}".format(pdg_id))
    n_seeds_by_pdg = {pdg_id: len(seeds) for pdg_id, seeds in seeds_by_pdg.items()}
    if len(set(n_seeds_by_pdg.values())) > 1:
        raise ValueError(
            "uneven seed counts across PDG tracks would understate one track's "
            "spread relative to the other: {}".format(n_seeds_by_pdg)
        )

    rows_by_pdg_seed: Dict[Tuple[int, int], Dict[str, Dict[str, Any]]] = {
        key: load_arena_rows(path) for key, path in arena_report_path_by_pdg_and_seed.items()
    }

    out_rows: List[Dict[str, Any]] = []
    for pdg_id in pdg_ids:
        seeds = sorted(seeds_by_pdg[pdg_id])
        for variant_id in variant_ids:
            p0_vals: List[float] = []
            pU_vals: List[float] = []
            q_vals: List[float] = []
            for seed in seeds:
                row = rows_by_pdg_seed[(pdg_id, seed)].get(variant_id)
                if row is None:
                    raise ValueError(
                        "variant_id={!r} missing from pdg_id={} seed={} arena report".format(
                            variant_id, pdg_id, seed
                        )
                    )
                p0_vals.append(float(row["nominal_b_toy_prevalence"]))
                pU_vals.append(float(row["theoretical_target_b_toy_prevalence"]))
                q_vals.append(float(row["generated_b_toy_occupancy"]))

            p0_arr, q_arr = np.asarray(p0_vals), np.asarray(q_vals)
            sides = np.sign(q_arr - p0_arr)
            all_same_side = bool(np.all(sides == sides[0])) if sides.size else True

            out_rows.append({
                "pdg_id": int(pdg_id),
                "variant_id": variant_id,
                "n_seeds": len(seeds),
                "seeds": ";".join(str(s) for s in seeds),
                "p0_mean": float(p0_arr.mean()),
                "p0_std": float(p0_arr.std()),
                "pU_mean": float(np.mean(pU_vals)),
                "pU_std": float(np.std(pU_vals)),
                "q_theta_mean": float(q_arr.mean()),
                "q_theta_std": float(q_arr.std()),
                "q_theta_min": float(q_arr.min()),
                "q_theta_max": float(q_arr.max()),
                "q_theta_values_by_seed": ";".join("{:.6f}".format(v) for v in q_vals),
                "all_seeds_same_side_of_p0": all_same_side,
            })
    return out_rows


def write_seed_robustness_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(CSV_FIELDNAMES), extrasaction="ignore", lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
