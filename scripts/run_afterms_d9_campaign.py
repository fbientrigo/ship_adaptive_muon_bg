#!/usr/bin/env python3
"""run_afterms_d9_campaign.py: D9 directed multi-seed after-MS training CLI (D9 spec section 16).

Subcommands:
    plan        Show the candidate plan / training config; --dry-run prints
                the full intended-run workload table. Never trains.
    status      Report per-run status from disk. Never writes artifacts.
    train       Train exactly ONE (candidate, seed). Requires both
                --candidate-id and --seed; there is no "train everything"
                default.
    evaluate    Evaluate one completed (candidate, seed) run on its test
                split. Never trains.
    summarize   Aggregate multi-seed results for one or all candidates.
                Never trains.

Real training is never a default action of any subcommand: `train` always
requires --candidate-id and --seed explicitly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from ship_muon_bg.afterms.d9 import (  # noqa: E402
    aggregate as d9agg,
    checkpoint as ckpt,
    evaluate as d9eval,
    plan as d9plan,
    runner as d9runner,
    training_config as d9tc,
)

DEFAULT_PLAN_PATH = REPO_ROOT / "configs" / "afterms" / "d9_candidate_plan_v0.json"
DEFAULT_TRAINING_CONFIG_PATH = REPO_ROOT / "configs" / "afterms" / "d9_training_v0.json"
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "afterms_d9_training_v0"


def _load(plan_path: Path, training_config_path: Path):
    plan = d9plan.load_plan(plan_path)
    training_config = d9tc.load_training_config(training_config_path)
    return plan, training_config


def cmd_plan(args) -> int:
    plan, training_config = _load(args.plan_path, args.training_config_path)
    violations = d9plan.validate_plan(plan)
    if violations:
        print("PLAN INVALID:")
        for v in violations:
            print(f"  - {v}")
        return 2

    if not args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0

    print(f"{'candidate':<45} {'target':<20} {'pdg':>5} {'seed':>10} {'max_epochs':>10} {'est_runtime':>14} artifact_path")
    total_seconds = 0.0
    for candidate in training_config["candidates"]:
        seconds_per_epoch = candidate.get("estimated_seconds_per_epoch_from_d7")
        for seed in candidate["seed_set"]:
            est = None
            if seconds_per_epoch:
                est = seconds_per_epoch * candidate["max_epochs"]
                total_seconds += est
            est_str = f"{est/60.0:.1f} min" if est else "unknown"
            run_dir = d9runner.run_directory(args.artifact_root, candidate["candidate_id"], seed)
            print(
                f"{candidate['candidate_id']:<45} {candidate['target_measure']:<20} "
                f"{candidate['pdg_value']!s:>5} {seed:>10} {candidate['max_epochs']:>10} {est_str:>14} {run_dir}"
            )
    print(f"\nTotal estimated serial GPU time: {total_seconds/3600.0:.2f} hours (provisional, see non_convergence_disclaimer)")
    print("No training was executed by this dry-run.")
    return 0


def cmd_status(args) -> int:
    plan, training_config = _load(args.plan_path, args.training_config_path)
    candidates = training_config["candidates"]
    if args.candidate_id:
        candidates = [c for c in candidates if c["candidate_id"] == args.candidate_id]
    for candidate in candidates:
        for seed in candidate["seed_set"]:
            run_dir = d9runner.run_directory(args.artifact_root, candidate["candidate_id"], seed)
            status_path = run_dir / "status.json"
            if status_path.exists():
                status = json.loads(status_path.read_text(encoding="utf-8"))
            else:
                status = {"status": "planned"}
            print(f"{candidate['candidate_id']} seed={seed}: {status.get('status')}")
    return 0


def cmd_train(args) -> int:
    if not args.candidate_id or args.seed is None:
        print("ERROR: `train` requires both --candidate-id and --seed. There is no train-all default.", file=sys.stderr)
        return 2

    plan, training_config = _load(args.plan_path, args.training_config_path)
    by_id = d9tc.candidates_by_id(training_config)
    if args.candidate_id not in by_id:
        print(f"ERROR: unknown candidate_id {args.candidate_id!r}", file=sys.stderr)
        return 2
    candidate_config = by_id[args.candidate_id]
    if args.seed not in candidate_config["seed_set"]:
        print(
            f"ERROR: seed {args.seed} is not in this candidate's configured seed_set "
            f"{candidate_config['seed_set']}", file=sys.stderr,
        )
        return 2

    shard_dir = Path(training_config["shard_dir"])
    train_raw = d9runner.load_concatenated_shards(shard_dir, candidate_config["train_shards"])
    validation_raw = d9runner.load_concatenated_shards(shard_dir, candidate_config["validation_shards"])

    try:
        result = d9runner.train_candidate_seed(
            candidate_config, args.seed, train_raw, validation_raw,
            artifact_root=args.artifact_root, repo_root=REPO_ROOT, device=args.device, resume=args.resume,
            extend_max_epochs=args.extend_max_epochs,
            execution_policy_revision_reason=args.execution_policy_revision_reason,
        )
    except (d9runner.MaxEpochsExtensionError, ckpt.CheckpointCompatibilityError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["status"] in (d9runner.STATUS_COMPLETED, d9runner.STATUS_INTERRUPTED) else 1


def cmd_evaluate(args) -> int:
    plan, training_config = _load(args.plan_path, args.training_config_path)
    by_id = d9tc.candidates_by_id(training_config)
    candidate_config = by_id[args.candidate_id]
    run_dir = d9runner.run_directory(args.artifact_root, args.candidate_id, args.seed)
    shard_dir = Path(training_config["shard_dir"])
    result = d9eval.evaluate_candidate_seed(
        candidate_config, run_dir=run_dir, shard_dir=shard_dir,
        budget_name=args.budget, device=args.device,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


def cmd_summarize(args) -> int:
    plan, training_config = _load(args.plan_path, args.training_config_path)
    if args.candidate_id:
        by_id = d9tc.candidates_by_id(training_config)
        candidate = by_id[args.candidate_id]
        statuses = d9agg.load_seed_statuses(args.artifact_root, args.candidate_id, candidate["seed_set"])
        print(json.dumps(d9agg.aggregate_candidate(args.candidate_id, statuses), indent=2, default=str))
    else:
        print(json.dumps(d9agg.aggregate_campaign(training_config, args.artifact_root), indent=2, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan-path", type=Path, default=DEFAULT_PLAN_PATH)
    parser.add_argument("--training-config-path", type=Path, default=DEFAULT_TRAINING_CONFIG_PATH)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)

    p_plan = sub.add_parser("plan", help="Show the plan/training config; never trains.")
    p_plan.add_argument("--dry-run", action="store_true", help="Print the intended-run workload table.")
    p_plan.set_defaults(func=cmd_plan)

    p_status = sub.add_parser("status", help="Report per-run status from disk. Never writes artifacts.")
    p_status.add_argument("--candidate-id", default=None)
    p_status.set_defaults(func=cmd_status)

    p_train = sub.add_parser("train", help="Train exactly one (candidate, seed).")
    p_train.add_argument("--candidate-id", required=True)
    p_train.add_argument("--seed", type=int, required=True)
    p_train.add_argument("--device", default="cpu")
    p_train.add_argument("--resume", action="store_true")
    p_train.add_argument("--extend-max-epochs", type=int, default=None, dest="extend_max_epochs")
    p_train.add_argument("--execution-policy-revision-reason", default=None, dest="execution_policy_revision_reason")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", help="Evaluate one completed run. Never trains.")
    p_eval.add_argument("--candidate-id", required=True)
    p_eval.add_argument("--seed", type=int, required=True)
    p_eval.add_argument("--device", default="cpu")
    p_eval.add_argument("--budget", choices=["quick_validation_budget", "final_candidate_budget"], default="quick_validation_budget")
    p_eval.set_defaults(func=cmd_evaluate)

    p_summ = sub.add_parser("summarize", help="Aggregate multi-seed results. Never trains.")
    p_summ.add_argument("--candidate-id", default=None)
    p_summ.set_defaults(func=cmd_summarize)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
