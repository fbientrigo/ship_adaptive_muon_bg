"""Run the corrected v1 conditional-charge NF fixture study across model seeds.

Bounded multiseed fixture sanity study (task section 7): the same v1 config
(architecture, per-epoch draw budget) is run once per model seed, each run
writing its own artifact directory; this script then aggregates the
per-charge metrics that matter for a capacity sanity check (median, min, max
across seeds) without defining any universal pass/fail threshold.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List

from ship_muon_bg.density_lab.conditional_charge import run_fixture_pilot

DEFAULT_SEEDS = (11, 12, 13)


def _aggregate(values: List[float]) -> Dict[str, Any]:
    finite = [v for v in values if v is not None]
    if not finite:
        return {"median": None, "min": None, "max": None, "values": values}
    return {
        "median": statistics.median(finite),
        "min": min(finite),
        "max": max(finite),
        "values": values,
    }


def run_study(config_path: Path, output_root: Path, seeds=DEFAULT_SEEDS) -> Dict[str, Any]:
    base_config = json.loads(config_path.read_text(encoding="utf-8"))
    per_seed: List[Dict[str, Any]] = []
    for seed in seeds:
        config = dict(base_config)
        config["seed"] = int(seed)
        config["experiment_id"] = "{}_seed{}".format(base_config["experiment_id"], seed)
        output_dir = output_root / "seed_{}".format(seed)
        summary = run_fixture_pilot(config, output_dir=output_dir)
        per_charge_by_pdg = {row["pdg_id"]: row for row in summary["per_charge_validation"]}
        generated_by_pdg = {row["pdg_id"]: row for row in summary["per_charge_generated_summary"]}
        record = {
            "seed": int(seed),
            "output_dir": str(output_dir),
            "macro_validation_nll": summary["validation"]["macro_nll"],
            "worst_charge_validation_nll": summary["validation"]["worst_charge_nll"],
            "best_validation_nll": summary["fit"]["best_validation_nll"],
            "per_charge": {},
        }
        for pdg_id in (13, -13):
            val_row = per_charge_by_pdg[pdg_id]
            gen_row = generated_by_pdg[pdg_id]
            charge_manifest = summary["sampling_manifest"]["charges"][str(pdg_id)]
            record["per_charge"][str(pdg_id)] = {
                "condition": val_row["condition"],
                "final_weighted_validation_nll": val_row["physical_validation_nll_weighted"],
                "nominal_b_toy_occupancy": gen_row["nominal_b_toy_occupancy"],
                "generated_b_toy_occupancy": gen_row["generated_b_toy_occupancy"],
                "generated_over_nominal_b_toy_ratio": gen_row["generated_over_nominal_b_toy_ratio"],
                "generated_sample_finite_fraction": gen_row["generated_sample_finite_fraction"],
                "generated_log_prob_finite_fraction": gen_row["generated_log_prob_finite_fraction"],
                "cumulative_unique_source_rows": charge_manifest["cumulative_unique_source_rows"],
                "cumulative_unique_source_rows_fraction": charge_manifest["cumulative_unique_source_rows_fraction"],
            }
        per_seed.append(record)

    aggregate: Dict[str, Any] = {"macro_validation_nll": _aggregate([r["macro_validation_nll"] for r in per_seed])}
    for pdg_id in (13, -13):
        key = str(pdg_id)
        aggregate[key] = {
            metric: _aggregate([r["per_charge"][key][metric] for r in per_seed])
            for metric in (
                "final_weighted_validation_nll",
                "generated_b_toy_occupancy",
                "generated_over_nominal_b_toy_ratio",
            )
        }
    study_summary = {"seeds": list(seeds), "per_seed": per_seed, "aggregate": aggregate}
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "multiseed_study_summary.json").write_text(
        json.dumps(study_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return study_summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    args = parser.parse_args()
    study_summary = run_study(args.config, args.output_root, seeds=tuple(args.seeds))
    print(json.dumps(study_summary["aggregate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
