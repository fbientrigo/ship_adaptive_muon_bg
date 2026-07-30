#!/usr/bin/env python
"""D9 -- direct-sampling utility-tilt campaign entry point.

Builds and validates the 20-configuration utility-tilt sampling grid
(``ship_muon_bg.density_lab.utility_tilt``) on a D7 (empirical after-MS)
dataset, and optionally trains selected tilt configurations by drawing
training rows directly (with replacement) from the tilted law and training
with ordinary unweighted NLL.

Table generation is always bounded to what is requested; training never
launches more than the tilt IDs explicitly passed via ``--train-tilt-ids``.
The single external-data configuration seam is ``--dataset``: the repository
afterMS fixture and a full local dataset run through the exact same code,
differing only in that path plus bounded operational settings (``--max-rows``,
``--artifact-root``).

Examples
--------
Build and validate all 20 tables for both PDG tracks, no training:

    python scripts/run_utility_tilt_campaign.py \\
        --dataset data/samples/muonsFullMC_afterMS_sample.npz \\
        --pdg-ids 13 -13 --max-rows 2000 --seed 11 \\
        --build-tables --tables-only

Run exactly the four bounded cloud tilt configurations on PDG 13:

    python scripts/run_utility_tilt_campaign.py \\
        --dataset data/samples/muonsFullMC_afterMS_sample.npz \\
        --pdg-ids 13 --max-rows 2000 --seed 11 --sampler-seed 7 \\
        --model-config configs/density_lab/utility_tilt/d9_fixture_smoke_v0.json \\
        --train-tilt-ids UA_d0p9_a01 UA_d0p1_a04 UP_d0p9_a01 UP_d0p1_a04 \\
        --epochs 1 --device cpu

Rerun one selected tilt locally, skipping if already completed:

    python scripts/run_utility_tilt_campaign.py \\
        --dataset data/samples/muonsFullMC_afterMS_sample.npz \\
        --pdg-ids 13 --max-rows 2000 --seed 11 --sampler-seed 7 \\
        --model-config configs/density_lab/utility_tilt/d9_fixture_smoke_v0.json \\
        --train-tilt-ids UA_d0p9_a01
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from ship_muon_bg.density_lab import utility_tilt as ut  # noqa: E402
from ship_muon_bg.density_lab.artifacts import ArtifactStore  # noqa: E402
from ship_muon_bg.density_lab.config import EvaluationSpec, FeatureViewSpec, ModelSpec  # noqa: E402
from ship_muon_bg.density_lab.empirical import EmpiricalDataError, EmpiricalDatasetSpec  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", required=True, help="dataset path (env vars/~ expanded); the only flag that usually differs between the repository fixture and a full local run")
    parser.add_argument("--pdg-ids", nargs="+", type=int, required=True, help="PDG track(s) to process, e.g. 13 -13")
    parser.add_argument("--seed", type=int, required=True, help="deterministic dataset/split/model-init seed")
    parser.add_argument("--sampler-seed", type=int, default=None, help="deterministic alias-sampler draw seed (default: --seed)")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--max-rows", type=int, default=None, help="per-PDG row budget applied after PDG filtering, before the split")
    parser.add_argument("--allow-zero-weight", action="store_true", default=True)
    parser.add_argument("--experiment-id", default="d9_utility_tilt_v0")
    parser.add_argument("--artifact-root", default=None, help="override the artifact root directory (default: artifacts/density_lab/)")

    parser.add_argument("--build-tables", action="store_true", help="build Table A (nominal) and Table B (utility-tilt) for the requested PDG track(s)")
    parser.add_argument("--tilt-ids", nargs="*", default=None, help="restrict Table B to these tilt IDs (default: all 20)")
    parser.add_argument("--tables-only", action="store_true", help="build/validate tables and exit; never trains")
    parser.add_argument("--tables-dir", default=None, help="directory to write table artifacts under (default: <artifact-root>/<experiment-id>/tables/pdg_<id>/)")

    parser.add_argument("--train-tilt-ids", nargs="*", default=None, help="exactly the tilt IDs to train (direct-sampling, ordinary unweighted NLL); pass the literal 'NOMINAL_PHYSICAL' to train the untilted direct-from-pi_nominal arm; none trained if omitted")
    parser.add_argument("--arena", action="store_true", help="shorthand: set --train-tilt-ids to the bounded D9 arena (NOMINAL_PHYSICAL + 8 tilts, ut.ARENA_VARIANT_IDS) unless --train-tilt-ids was already given explicitly")
    parser.add_argument("--arena-report", action="store_true", help="after training, validate --train-tilt-ids as an arena variant set (rejects duplicates/unknown ids) and write a deterministic arena_summary.{json,csv,md} report per PDG track under <artifact-root>/<experiment-id>/arena_report/pdg_<id>/")
    parser.add_argument("--model-config", default=None, help="JSON file with a top-level 'model' (ModelSpec) and 'feature_view' (FeatureViewSpec) block; required if --train-tilt-ids/--arena is given")
    parser.add_argument("--n-draws", type=int, default=None, help="training draw budget T (default: n_train, i.e. identical optimizer-step count to the un-tilted baseline)")
    parser.add_argument("--epochs", type=int, default=None, help="override the model config's max_epochs")
    parser.add_argument("--device", default="cpu", help="cpu | cuda | auto")
    parser.add_argument("--force", action="store_true", help="re-run completed table builds/training runs")

    args = parser.parse_args()

    if args.arena and not args.train_tilt_ids:
        args.train_tilt_ids = list(ut.ARENA_VARIANT_IDS)

    import os

    dataset_path = os.path.expanduser(os.path.expandvars(args.dataset))
    sampler_seed = args.sampler_seed if args.sampler_seed is not None else args.seed
    root = Path(args.artifact_root) if args.artifact_root else None

    summary = {"experiment_id": args.experiment_id, "dataset_path": dataset_path, "pdg_results": {}}

    if args.build_tables or args.tables_only:
        for pdg_id in args.pdg_ids:
            try:
                result = ut.build_and_validate_pdg_tables(
                    dataset_path, pdg_id, seed=args.seed, val_fraction=args.val_fraction,
                    test_fraction=args.test_fraction, max_rows=args.max_rows,
                    allow_zero_weight=args.allow_zero_weight, tilt_ids=args.tilt_ids,
                )
            except EmpiricalDataError as exc:
                print("pdg_id={}: BLOCKED building tables: {}".format(pdg_id, exc))
                summary["pdg_results"][str(pdg_id)] = {"status": "blocked", "error": str(exc)}
                continue

            tables_dir = (
                Path(args.tables_dir) if args.tables_dir else
                (root or Path("artifacts") / "density_lab") / args.experiment_id / "tables" / "pdg_{}".format(pdg_id)
            )
            manifest_a = result["table_a"].save(tables_dir, seed=args.seed)
            manifest_b = result["table_b"].save(
                tables_dir, [ut.TILT_CONFIG_BY_ID[t] for t in (args.tilt_ids or list(ut.TILT_CONFIG_BY_ID))],
                source_dataset_hash=result["table_a"].source_dataset_hash,
                split_hash=result["table_a"].split_hash, seed=args.seed,
            )
            bad_tilts = [
                tid for tid, check in result["checks"]["per_tilt"].items()
                if not (check["pi_tilt_sums_to_one"] and check["agrees_within_tolerance"])
            ]
            print(
                "pdg_id={}: tables written to {} (table_a_hash={} table_b_hash={} n_tilts_validated={} bad_tilts={})".format(
                    pdg_id, tables_dir, manifest_a["table_hash"][:16], manifest_b["table_hash"][:16],
                    len(result["checks"]["per_tilt"]), bad_tilts,
                )
            )
            summary["pdg_results"][str(pdg_id)] = {
                "status": "tables_built", "tables_dir": str(tables_dir),
                "table_a_hash": manifest_a["table_hash"], "table_b_hash": manifest_b["table_hash"],
                "pi_nominal_sums_to_one": result["checks"]["pi_nominal_sums_to_one"], "bad_tilts": bad_tilts,
            }

    if args.tables_only:
        print(json.dumps(summary, indent=2, default=str))
        return 0

    if args.arena_report and not args.train_tilt_ids:
        parser.error("--arena-report requires --train-tilt-ids (or --arena)")

    if args.train_tilt_ids:
        if not args.model_config:
            parser.error("--model-config is required when --train-tilt-ids is given")
        if args.arena_report:
            # Fail fast on a malformed arena definition before spending any
            # training time (task section 5E requirement 4).
            ut.validate_arena_variant_ids(args.train_tilt_ids)
        model_payload = json.loads(Path(args.model_config).read_text())
        model_dict = model_payload["model"]
        if args.epochs is not None:
            model_dict = dict(model_dict)
            model_dict["params"] = dict(model_dict.get("params", {}))
            model_dict["params"]["max_epochs"] = int(args.epochs)
        model = ModelSpec(
            name=model_dict["name"], family=model_dict["family"],
            params=dict(model_dict.get("params", {})),
            training_budget_id=model_dict.get("training_budget_id", "default"),
        )
        epochs_value = model_dict.get("params", {}).get("max_epochs")
        fv_dict = model_payload.get("feature_view", {"view_id": "identity_cartesian_v0"})
        feature_view = FeatureViewSpec(fv_dict["view_id"], fv_dict.get("pz_unit_gev"))
        evaluation = EvaluationSpec(**model_payload.get("evaluation", {})) if model_payload.get("evaluation") else EvaluationSpec()

        store = ArtifactStore(args.experiment_id, root=root)
        training_records = []
        records_by_pdg: dict = {}
        for pdg_id in args.pdg_ids:
            dataset_spec = EmpiricalDatasetSpec(
                dataset_path=dataset_path, pdg_id=int(pdg_id), seed=args.seed,
                val_fraction=args.val_fraction, test_fraction=args.test_fraction,
                max_rows=args.max_rows, allow_zero_weight=args.allow_zero_weight,
            )
            for tilt_id in args.train_tilt_ids:
                record = ut.run_direct_sampling_training(
                    dataset_spec, store, experiment_id=args.experiment_id, tilt_id=tilt_id,
                    feature_view=feature_view, model=model, sampler_seed=sampler_seed,
                    n_draws=args.n_draws, evaluation=evaluation, device=args.device, force=args.force,
                )
                training_records.append(record)
                records_by_pdg.setdefault(pdg_id, []).append(record)
                print(
                    "pdg_id={} tilt_id={}: status={} run_id={}".format(
                        pdg_id, tilt_id, record["status"], record.get("run_id")
                    )
                )
        summary["training_records"] = [
            {k: v for k, v in r.items() if k != "metrics"} for r in training_records
        ]
        summary["n_training_jobs"] = len(training_records)

        if args.arena_report:
            arena_root = (root or Path("artifacts") / "density_lab") / args.experiment_id / "arena_report"
            summary["arena_reports"] = {}
            for pdg_id, pdg_records in records_by_pdg.items():
                out_dir = arena_root / "pdg_{}".format(pdg_id)
                report = ut.build_arena_report(
                    pdg_records, store, out_dir=out_dir, pdg_id=pdg_id,
                    experiment_id=args.experiment_id, epochs=epochs_value,
                )
                summary["arena_reports"][str(pdg_id)] = {
                    "out_dir": str(out_dir), "n_rows": len(report["rows"]),
                }
                print("pdg_id={}: arena report written to {} ({} rows)".format(pdg_id, out_dir, len(report["rows"])))

    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
