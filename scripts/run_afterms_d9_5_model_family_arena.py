#!/usr/bin/env python3
"""run_afterms_d9_5_model_family_arena.py: D9-5 model family arena CLI (Sec 17).

Subcommands:
    audit             Resolve the data-scope manifest + RAM preflight. Never fits.
    plan              Show resolved config (scout-promoted NF_AC config per
                       track, frozen family arena config); --dry-run prints
                       the intended (track, family[, seed]) workload table.
                       Never fits.
    fit               Fit exactly ONE (track, model-family[, seed]). Requires
                       --execute to do anything; otherwise prints the intended
                       action and exits. NF_AC and GMM require --seed;
                       GAUSS_DIAG/GAUSS_FULL are deterministic (no --seed).
                       There is no "fit everything" default.
    status            Report per-run status from disk. Never fits.
    validate          Evaluate one already-fitted run on its VALIDATION split
                       only. Never reads a test shard, never fits.
    freeze-selection  Aggregate validation results into
                       family_selection_manifest.json. Never reads test data.
    evaluate-test     Evaluate one already-fitted run on its TEST split.
                       Refuses to run unless freeze-selection has already
                       written a manifest for this track. Never fits.
    summarize         Assemble the final cross-family report from persisted
                       test_evaluation results. Never fits, never trains.

Every subcommand supports --dry-run. No command fits every model without an
explicit --campaign flag naming every (track, family) pair it will touch.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from ship_muon_bg.afterms.d9 import checkpoint as d9ckpt  # noqa: E402
from ship_muon_bg.afterms.d9 import runner as d9runner  # noqa: E402
from ship_muon_bg.afterms.d9_5 import (  # noqa: E402
    aggregation as d9_5agg,
    config as d9_5config,
    data_scope,
    evaluation as d9_5eval,
    gaussian_adapter,
    gmm_adapter,
    model_adapter as ma,
    nf_ac_adapter,
    report as d9_5report,
)

DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "afterms_d9_5_model_family_arena_v0"
MODEL_FAMILIES = ("NF_AC", "GAUSS_DIAG", "GAUSS_FULL", "GMM")
STOCHASTIC_FAMILIES = ("NF_AC", "GMM")


def _run_dir(artifact_root: Path, track_id: str, model_config_id: str, seed) -> Path:
    seed_segment = f"seed_{seed}" if seed is not None else "deterministic"
    return Path(artifact_root, "runs", track_id, model_config_id, seed_segment)


def _resolve_nf_config(track_id: str) -> dict:
    return nf_ac_adapter.select_scout_promoted_nf_config(track_id)


def _build_adapter(track_id: str, model_family: str, *, arena_config: dict, artifact_root: Path, device: str, seed=None):
    pdg_value = 13 if track_id == "TRK_PDG13_UW_ID" else -13
    if model_family == "GAUSS_DIAG":
        return gaussian_adapter.DiagonalGaussianAdapter(variance_floor=arena_config["gauss_diag"]["variance_floor"])
    if model_family == "GAUSS_FULL":
        return gaussian_adapter.FullGaussianAdapter(covariance_regularization=arena_config["gauss_full"]["covariance_regularization"])
    if model_family == "GMM":
        gmm_cfg = arena_config["gmm"]
        return gmm_adapter.GmmAdapter(
            n_components=gmm_cfg["n_components"], covariance_type=gmm_cfg["covariance_type"],
            covariance_regularization=gmm_cfg["covariance_regularization"], n_init=gmm_cfg["n_init"],
            max_iter=gmm_cfg["max_iter"], tol=gmm_cfg["convergence_tol"],
            occupied_weight_threshold=gmm_cfg["effective_component_weight_threshold"],
        )
    if model_family == "NF_AC":
        nf_cfg = arena_config["nf_ac"]
        scout = _resolve_nf_config(track_id)
        data_scope_config = d9_5config.load_data_scope_config()
        shard_dir = REPO_ROOT / data_scope_config["shard_dir"]
        shard_manifest = data_scope.load_shard_manifest(shard_dir)
        return nf_ac_adapter.NfAcAdapter(
            track_id=track_id, pdg_value=pdg_value, architecture=scout["architecture"],
            execution_policy=nf_cfg["execution_policy"],
            optimizer_settings={
                "optimizer": nf_cfg["optimizer"], "learning_rate": nf_cfg["learning_rate"],
                "batch_size": nf_cfg["batch_size"], "weight_decay": nf_cfg["weight_decay"],
                "gradient_clipping": nf_cfg["gradient_clipping"], "dtype": nf_cfg["dtype"],
                "device_policy": nf_cfg["device_policy"],
            },
            evaluation_policy=arena_config["evaluation_policy"],
            artifact_root=artifact_root, repo_root=REPO_ROOT, device=device,
            shard_dir=data_scope_config["shard_dir"],
            train_shard_names=[s["npy_file"] for s in data_scope.shards_for_split(shard_manifest, "train")],
            validation_shard_names=[s["npy_file"] for s in data_scope.shards_for_split(shard_manifest, "validation")],
            test_shard_names=[s["npy_file"] for s in data_scope.shards_for_split(shard_manifest, "test")],
        )
    raise ValueError(f"unknown model_family {model_family!r}; expected one of {MODEL_FAMILIES}")


def cmd_audit(args) -> int:
    data_scope_config = d9_5config.load_data_scope_config()

    if args.dry_run:
        # A full audit reads every declared shard to filter/hash/count rows
        # (Sec 6); --dry-run previews the resolved shard list only, without
        # that read, so it stays cheap regardless of dataset size.
        shard_dir = REPO_ROOT / data_scope_config["shard_dir"]
        manifest = data_scope.load_shard_manifest(shard_dir)
        preview = {
            "shard_dir": data_scope_config["shard_dir"],
            "tracks": [t["track_id"] for t in data_scope_config["tracks"]],
            "shard_counts_by_split": {
                split: len(data_scope.shards_for_split(manifest, split)) for split in data_scope.SPLITS
            },
            "executed": False,
        }
        print(json.dumps(preview, indent=2, default=str))
        return 0

    manifest_record = data_scope.build_data_scope_manifest(data_scope_config)
    preflight = data_scope.estimate_ram_preflight(manifest_record)

    out_dir = args.artifact_root / "data_scope"
    out_dir.mkdir(parents=True, exist_ok=True)
    ma.atomic_write_json(out_dir / "data_scope_manifest.json", manifest_record)
    (out_dir / "data_scope_audit.md").write_text(
        data_scope.render_data_scope_audit_md(manifest_record), encoding="utf-8",
    )
    ma.atomic_write_json(out_dir / "ram_preflight.json", preflight)
    print(json.dumps({"manifest": manifest_record, "ram_preflight": preflight}, indent=2, default=str))
    return 0


def cmd_plan(args) -> int:
    arena_config = d9_5config.load_model_family_arena_config()
    data_scope_config = d9_5config.load_data_scope_config()
    tracks = [t["track_id"] for t in data_scope_config["tracks"]]

    plan = {"tracks": {}}
    for track_id in tracks:
        scout = _resolve_nf_config(track_id)
        plan["tracks"][track_id] = {
            "nf_ac_scout_promoted_config": scout,
            "gauss_diag": arena_config["gauss_diag"],
            "gauss_full": arena_config["gauss_full"],
            "gmm": arena_config["gmm"],
        }

    if not args.dry_run:
        print(json.dumps(plan, indent=2, default=str))
        return 0

    print(f"{'track_id':<20} {'family':<12} {'model_config_id':<24} {'seed':>10} run_dir")
    for track_id in tracks:
        scout = _resolve_nf_config(track_id)
        for seed in arena_config["nf_ac"]["seeds"]:
            run_dir = _run_dir(args.artifact_root, track_id, scout["model_config_id"], seed)
            print(f"{track_id:<20} {'NF_AC':<12} {scout['model_config_id']:<24} {seed:>10} {run_dir}")
        for seed in arena_config["gmm"]["seeds"]:
            run_dir = _run_dir(args.artifact_root, track_id, arena_config["gmm"]["model_config_id"], seed)
            print(f"{track_id:<20} {'GMM':<12} {arena_config['gmm']['model_config_id']:<24} {seed:>10} {run_dir}")
        for family_key in ("gauss_diag", "gauss_full"):
            cfg = arena_config[family_key]
            run_dir = _run_dir(args.artifact_root, track_id, cfg["model_config_id"], None)
            print(f"{track_id:<20} {cfg['model_family']:<12} {cfg['model_config_id']:<24} {'n/a':>10} {run_dir}")
    print("\nNo fitting was executed by this dry-run.")
    return 0


def cmd_fit(args) -> int:
    if args.model_family in STOCHASTIC_FAMILIES and args.seed is None:
        print(f"ERROR: --model-family {args.model_family} requires --seed", file=sys.stderr)
        return 2
    if args.model_family not in STOCHASTIC_FAMILIES and args.seed is not None:
        print(f"ERROR: --model-family {args.model_family} is deterministic; do not pass --seed", file=sys.stderr)
        return 2

    arena_config = d9_5config.load_model_family_arena_config()
    data_scope_config = d9_5config.load_data_scope_config()
    shard_dir = REPO_ROOT / data_scope_config["shard_dir"]
    manifest = data_scope.load_shard_manifest(shard_dir)
    pdg_value = 13 if args.track_id == "TRK_PDG13_UW_ID" else -13

    adapter = _build_adapter(
        args.track_id, args.model_family, arena_config=arena_config,
        artifact_root=args.artifact_root, device=args.device, seed=args.seed,
    )
    run_dir = _run_dir(args.artifact_root, args.track_id, adapter.model_config_id, args.seed)

    if not args.execute:
        print(json.dumps({
            "action": "fit", "track_id": args.track_id, "model_family": args.model_family,
            "model_config_id": adapter.model_config_id, "seed": args.seed, "run_dir": str(run_dir),
            "executed": False, "note": "pass --execute to actually fit",
        }, indent=2, default=str))
        return 0

    train_raw = data_scope.load_filtered_split(shard_dir, manifest, "train", pdg_value)
    validation_raw = data_scope.load_filtered_split(shard_dir, manifest, "validation", pdg_value)

    if args.model_family == "NF_AC":
        interrupt_flag = None
        if args.deadline_timestamp is not None:
            deadline_timestamp = args.deadline_timestamp
            interrupt_flag = lambda: time.time() >= deadline_timestamp
        result = adapter.fit(
            train_raw, validation_raw, seed=args.seed, resume=args.resume,
            extend_max_epochs=args.extend_max_epochs, interrupt_flag=interrupt_flag,
        )
    elif args.model_family == "GMM":
        result = adapter.fit(train_raw, validation_raw, seed=args.seed)
    else:
        result = adapter.fit(train_raw, validation_raw, seed=0)

    if args.model_family != "NF_AC":
        # NF_AC persists its own bundle atomically inside train_candidate_seed.
        adapter.save_bundle(run_dir)
    print(json.dumps(result, indent=2, default=str))
    return 0


def cmd_status(args) -> int:
    arena_config = d9_5config.load_model_family_arena_config()
    data_scope_config = d9_5config.load_data_scope_config()
    tracks = [args.track_id] if args.track_id else [t["track_id"] for t in data_scope_config["tracks"]]

    for track_id in tracks:
        scout = _resolve_nf_config(track_id)
        for seed in arena_config["nf_ac"]["seeds"]:
            run_dir = d9runner.run_directory(args.artifact_root, f"{track_id}/{scout['model_config_id']}", seed)
            status_path = run_dir / "status.json"
            status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {"status": "planned"}
            print(f"{track_id} NF_AC seed={seed}: {status.get('status')}")
        for seed in arena_config["gmm"]["seeds"]:
            run_dir = _run_dir(args.artifact_root, track_id, arena_config["gmm"]["model_config_id"], seed)
            exists = (run_dir / "fitting_history.json").exists()
            print(f"{track_id} GMM seed={seed}: {'completed' if exists else 'planned'}")
        for family_key in ("gauss_diag", "gauss_full"):
            cfg = arena_config[family_key]
            run_dir = _run_dir(args.artifact_root, track_id, cfg["model_config_id"], None)
            exists = (run_dir / "fitting_log.json").exists()
            print(f"{track_id} {cfg['model_family']}: {'completed' if exists else 'planned'}")
    return 0


def _load_adapter_for_evaluation(args, model_config_id: str):
    run_dir = _run_dir(args.artifact_root, args.track_id, model_config_id, args.seed)
    if args.model_family == "NF_AC":
        return nf_ac_adapter.NfAcAdapter.load_bundle(run_dir), run_dir
    if args.model_family == "GMM":
        return gmm_adapter.GmmAdapter.load_bundle(run_dir), run_dir
    if args.model_family == "GAUSS_DIAG":
        return gaussian_adapter.DiagonalGaussianAdapter.load_bundle(run_dir), run_dir
    if args.model_family == "GAUSS_FULL":
        return gaussian_adapter.FullGaussianAdapter.load_bundle(run_dir), run_dir
    raise ValueError(f"unknown model_family {args.model_family!r}")


def cmd_validate(args) -> int:
    arena_config = d9_5config.load_model_family_arena_config()
    data_scope_config = d9_5config.load_data_scope_config()
    shard_dir = REPO_ROOT / data_scope_config["shard_dir"]
    manifest = data_scope.load_shard_manifest(shard_dir)
    pdg_value = 13 if args.track_id == "TRK_PDG13_UW_ID" else -13

    model_config_id = (
        _resolve_nf_config(args.track_id)["model_config_id"] if args.model_family == "NF_AC"
        else arena_config["gmm"]["model_config_id"] if args.model_family == "GMM"
        else arena_config["gauss_diag"]["model_config_id"] if args.model_family == "GAUSS_DIAG"
        else arena_config["gauss_full"]["model_config_id"]
    )

    if not args.execute:
        print(json.dumps({
            "action": "validate", "track_id": args.track_id, "model_family": args.model_family,
            "model_config_id": model_config_id, "seed": args.seed, "executed": False,
        }, indent=2, default=str))
        return 0

    adapter, run_dir = _load_adapter_for_evaluation(args, model_config_id)
    validation_raw = data_scope.load_filtered_split(shard_dir, manifest, "validation", pdg_value)
    budget = arena_config["evaluation_policy"][args.budget]
    result = d9_5eval.evaluate_adapter(
        adapter, validation_raw, split_name="validation", track_id=args.track_id,
        budget=budget, budget_name=args.budget, generation_seed=args.generation_seed,
    )
    out_dir = run_dir / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    ma.atomic_write_json(out_dir / f"{args.budget}.json", result)
    print(json.dumps(result, indent=2, default=str))
    return 0


def cmd_freeze_selection(args) -> int:
    arena_config = d9_5config.load_model_family_arena_config()
    data_scope_config = d9_5config.load_data_scope_config()
    tracks = [t["track_id"] for t in data_scope_config["tracks"]]

    per_track_aggregates = {}
    for track_id in tracks:
        aggregates = []
        scout = _resolve_nf_config(track_id)
        per_seed = []
        for seed in arena_config["nf_ac"]["seeds"]:
            run_dir = d9runner.run_directory(args.artifact_root, f"{track_id}/{scout['model_config_id']}", seed)
            val_path = run_dir / "validation" / "quick_validation_budget.json"
            if val_path.exists():
                record = ma.read_json(val_path)
                per_seed.append({"seed": seed, "status": "ok", "physical_nll": record.get("physical_nll")})
            else:
                per_seed.append({"seed": seed, "status": "missing", "physical_nll": None})
        aggregates.append(d9_5agg.aggregate_seeds(track_id, scout["model_config_id"], per_seed))

        per_seed = []
        for seed in arena_config["gmm"]["seeds"]:
            run_dir = _run_dir(args.artifact_root, track_id, arena_config["gmm"]["model_config_id"], seed)
            val_path = run_dir / "validation" / "quick_validation_budget.json"
            if val_path.exists():
                record = ma.read_json(val_path)
                per_seed.append({"seed": seed, "status": "ok", "physical_nll": record.get("physical_nll")})
            else:
                per_seed.append({"seed": seed, "status": "missing", "physical_nll": None})
        aggregates.append(d9_5agg.aggregate_seeds(track_id, arena_config["gmm"]["model_config_id"], per_seed))

        for family_key in ("gauss_diag", "gauss_full"):
            cfg = arena_config[family_key]
            run_dir = _run_dir(args.artifact_root, track_id, cfg["model_config_id"], None)
            val_path = run_dir / "validation" / "quick_validation_budget.json"
            record = ma.read_json(val_path) if val_path.exists() else {}
            aggregates.append(d9_5agg.deterministic_fit_record(track_id, cfg["model_config_id"], record))

        per_track_aggregates[track_id] = aggregates

    freeze_manifest = d9_5report.build_family_selection_manifest(per_track_aggregates)
    if not args.execute:
        print(json.dumps(freeze_manifest, indent=2, default=str))
        print("\n--execute not passed: manifest not written.")
        return 0

    out_dir = args.artifact_root / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    d9_5report.write_freeze_manifest(out_dir, freeze_manifest)
    print(json.dumps(freeze_manifest, indent=2, default=str))
    return 0


def cmd_evaluate_test(args) -> int:
    try:
        d9_5report.require_frozen_selection(args.artifact_root / "validation")
    except d9_5report.SelectionNotFrozenError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    arena_config = d9_5config.load_model_family_arena_config()
    data_scope_config = d9_5config.load_data_scope_config()
    shard_dir = REPO_ROOT / data_scope_config["shard_dir"]
    manifest = data_scope.load_shard_manifest(shard_dir)
    pdg_value = 13 if args.track_id == "TRK_PDG13_UW_ID" else -13

    model_config_id = (
        _resolve_nf_config(args.track_id)["model_config_id"] if args.model_family == "NF_AC"
        else arena_config["gmm"]["model_config_id"] if args.model_family == "GMM"
        else arena_config["gauss_diag"]["model_config_id"] if args.model_family == "GAUSS_DIAG"
        else arena_config["gauss_full"]["model_config_id"]
    )

    if not args.execute:
        print(json.dumps({
            "action": "evaluate-test", "track_id": args.track_id, "model_family": args.model_family,
            "model_config_id": model_config_id, "seed": args.seed, "executed": False,
        }, indent=2, default=str))
        return 0

    adapter, run_dir = _load_adapter_for_evaluation(args, model_config_id)
    test_raw = data_scope.load_filtered_split(shard_dir, manifest, "test", pdg_value)
    budget = arena_config["evaluation_policy"][args.budget]
    result = d9_5eval.evaluate_adapter(
        adapter, test_raw, split_name="test", track_id=args.track_id,
        budget=budget, budget_name=args.budget, generation_seed=args.generation_seed,
    )
    out_dir = run_dir / "test_evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    ma.atomic_write_json(out_dir / f"{args.budget}.json", result)
    print(json.dumps(result, indent=2, default=str))
    return 0


def cmd_summarize(args) -> int:
    arena_config = d9_5config.load_model_family_arena_config()
    data_scope_config = d9_5config.load_data_scope_config()
    tracks = [t["track_id"] for t in data_scope_config["tracks"]]

    per_track_test_results = {}
    for track_id in tracks:
        results = []
        scout = _resolve_nf_config(track_id)
        family_configs = [
            ("NF_AC", scout["model_config_id"], arena_config["nf_ac"]["seeds"]),
            ("GMM", arena_config["gmm"]["model_config_id"], arena_config["gmm"]["seeds"]),
            ("GAUSS_DIAG", arena_config["gauss_diag"]["model_config_id"], [None]),
            ("GAUSS_FULL", arena_config["gauss_full"]["model_config_id"], [None]),
        ]
        for family, model_config_id, seeds in family_configs:
            for seed in seeds:
                run_dir = _run_dir(args.artifact_root, track_id, model_config_id, seed)
                result_path = run_dir / "test_evaluation" / f"{args.budget}.json"
                if result_path.exists():
                    results.append(ma.read_json(result_path))
        per_track_test_results[track_id] = results

    report_dir = args.artifact_root / "report"
    if not args.execute:
        print(json.dumps(per_track_test_results, indent=2, default=str))
        return 0
    report_dir.mkdir(parents=True, exist_ok=True)
    paths = d9_5report.write_final_report(report_dir, per_track_test_results)
    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)

    p_audit = sub.add_parser("audit", help="Resolve the data-scope manifest + RAM preflight. Never fits.")
    p_audit.add_argument("--dry-run", action="store_true")
    p_audit.set_defaults(func=cmd_audit)

    p_plan = sub.add_parser("plan", help="Show resolved config. Never fits.")
    p_plan.add_argument("--dry-run", action="store_true")
    p_plan.set_defaults(func=cmd_plan)

    p_fit = sub.add_parser("fit", help="Fit exactly one (track, family[, seed]).")
    p_fit.add_argument("--track-id", required=True, choices=list(d9_5config.TRACK_IDS))
    p_fit.add_argument("--model-family", required=True, choices=list(MODEL_FAMILIES))
    p_fit.add_argument("--seed", type=int, default=None)
    p_fit.add_argument("--device", default="cpu")
    p_fit.add_argument("--execute", action="store_true")
    p_fit.add_argument("--resume", action="store_true")
    p_fit.add_argument("--extend-max-epochs", type=int, default=None, dest="extend_max_epochs")
    p_fit.add_argument(
        "--deadline-timestamp", type=float, default=None, dest="deadline_timestamp",
        help="Unix epoch seconds; NF_AC only. When set, training refuses to start a new epoch "
        "once time.time() reaches this deadline and exits cleanly with status=interrupted. "
        "Absent: behavior is unchanged (no interrupt_flag).",
    )
    p_fit.set_defaults(func=cmd_fit)

    p_status = sub.add_parser("status", help="Report per-run status from disk. Never fits.")
    p_status.add_argument("--track-id", default=None, choices=list(d9_5config.TRACK_IDS))
    p_status.set_defaults(func=cmd_status)

    p_validate = sub.add_parser("validate", help="Evaluate one fitted run on its validation split. Never fits.")
    p_validate.add_argument("--track-id", required=True, choices=list(d9_5config.TRACK_IDS))
    p_validate.add_argument("--model-family", required=True, choices=list(MODEL_FAMILIES))
    p_validate.add_argument("--seed", type=int, default=None)
    p_validate.add_argument("--budget", choices=["quick_validation_budget", "final_candidate_budget"], default="quick_validation_budget")
    p_validate.add_argument("--generation-seed", type=int, default=20260720, dest="generation_seed")
    p_validate.add_argument("--execute", action="store_true")
    p_validate.set_defaults(func=cmd_validate)

    p_freeze = sub.add_parser("freeze-selection", help="Aggregate validation results into a frozen manifest.")
    p_freeze.add_argument("--execute", action="store_true")
    p_freeze.set_defaults(func=cmd_freeze_selection)

    p_evaltest = sub.add_parser("evaluate-test", help="Evaluate one fitted run on its test split. Refuses before freeze-selection.")
    p_evaltest.add_argument("--track-id", required=True, choices=list(d9_5config.TRACK_IDS))
    p_evaltest.add_argument("--model-family", required=True, choices=list(MODEL_FAMILIES))
    p_evaltest.add_argument("--seed", type=int, default=None)
    p_evaltest.add_argument("--budget", choices=["quick_validation_budget", "final_candidate_budget"], default="final_candidate_budget")
    p_evaltest.add_argument("--generation-seed", type=int, default=20260720, dest="generation_seed")
    p_evaltest.add_argument("--execute", action="store_true")
    p_evaltest.set_defaults(func=cmd_evaluate_test)

    p_summarize = sub.add_parser("summarize", help="Assemble the final cross-family report. Never fits.")
    p_summarize.add_argument("--budget", default="final_candidate_budget")
    p_summarize.add_argument("--execute", action="store_true")
    p_summarize.set_defaults(func=cmd_summarize)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
