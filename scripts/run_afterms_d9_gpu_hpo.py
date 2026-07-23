#!/usr/bin/env python3
"""run_afterms_d9_gpu_hpo.py: D9B RTX 2060 GPU-gate CLI (Gate B).

Subcommands:
    gpu-gate    Run (or, by default, only describe) the real-CUDA
                qualification gate for the smallest enabled, reconstructible,
                unweighted, modern 5D D9 candidate -- derived programmatically
                from the frozen D9 candidate plan, never hardcoded.
                `--dry-run` (the default) performs no training and reads no
                shard file. Real execution requires `--execute` explicitly.
    status      Report the gate's last recorded status from disk. Never
                trains.

This is a thin orchestrator: all training goes through the existing
`ship_muon_bg.afterms.d9.runner.train_candidate_seed` (invoked in an isolated
CUDA child process by `scripts/_d9b_gpu_gate_worker.py`), never a duplicated
training loop. Only one CUDA child process runs at a time, and there is no
"train everything"/HPO behavior in this CLI yet.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from ship_muon_bg.afterms.d9 import plan as d9plan  # noqa: E402
from ship_muon_bg.afterms.d9 import training_config as d9tc  # noqa: E402
from ship_muon_bg.afterms.d9b import candidate_selection as cs  # noqa: E402
from ship_muon_bg.afterms.d9b import gpu_gate  # noqa: E402

DEFAULT_PLAN_PATH = REPO_ROOT / "configs" / "afterms" / "d9_candidate_plan_v0.json"
DEFAULT_TRAINING_CONFIG_PATH = REPO_ROOT / "configs" / "afterms" / "d9_training_v0.json"
DEFAULT_GATE_ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "afterms_d9_gpu_hpo_v0" / "gpu_gate"


def _load(plan_path: Path, training_config_path: Path):
    plan = d9plan.load_plan(plan_path)
    training_config = d9tc.load_training_config(training_config_path)
    return plan, training_config


def cmd_gpu_gate(args) -> int:
    plan, training_config = _load(args.plan_path, args.training_config_path)

    violations = d9plan.validate_plan(plan)
    if violations:
        print("PLAN INVALID:", file=sys.stderr)
        for v in violations:
            print(f"  - {v}", file=sys.stderr)
        return 2

    candidate_config, selection_report = cs.select_gpu_gate_candidate(plan, training_config)
    seed = args.seed if args.seed is not None else candidate_config["seed_set"][0]

    if not args.execute:
        summary = gpu_gate.dry_run_plan(candidate_config, selection_report, seed, args.device)
        print(json.dumps(summary, indent=2, default=str))
        return 0

    manifest = gpu_gate.run_gpu_gate(
        repo_root=REPO_ROOT, gate_artifact_root=args.gate_artifact_root,
        plan=plan, training_config=training_config, device=args.device, seed=args.seed,
    )
    print(json.dumps({"gate_status": manifest["gate_status"], "candidate_id": manifest["candidate_id"],
                       "seed": manifest["seed"], "stop_conditions": manifest["stop_conditions"]}, indent=2))
    return 0 if manifest["gate_status"] == gpu_gate.GATE_PASSED else 1


def cmd_status(args) -> int:
    manifest_path = args.gate_artifact_root / "qualification_manifest.json"
    if not manifest_path.exists():
        print(json.dumps({"status": "not_yet_executed", "gate_artifact_root": str(args.gate_artifact_root)}, indent=2))
        return 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(json.dumps({
        "gate_status": manifest.get("gate_status"),
        "candidate_id": manifest.get("candidate_id"),
        "seed": manifest.get("seed"),
        "stop_conditions": manifest.get("stop_conditions"),
    }, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan-path", type=Path, default=DEFAULT_PLAN_PATH)
    parser.add_argument("--training-config-path", type=Path, default=DEFAULT_TRAINING_CONFIG_PATH)
    parser.add_argument("--gate-artifact-root", type=Path, default=DEFAULT_GATE_ARTIFACT_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)

    p_gate = sub.add_parser("gpu-gate", help="Run (or describe) the real-CUDA qualification gate.")
    p_gate.add_argument("--dry-run", action="store_true", help="Describe the plan only (default behavior).")
    p_gate.add_argument("--execute", action="store_true", help="Actually run the real CUDA qualification sequence.")
    p_gate.add_argument("--device", default="cuda")
    p_gate.add_argument("--seed", type=int, default=None)
    p_gate.set_defaults(func=cmd_gpu_gate)

    p_status = sub.add_parser("status", help="Report the gate's last recorded status. Never trains.")
    p_status.set_defaults(func=cmd_status)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "execute", False) and getattr(args, "dry_run", False):
        print("ERROR: --dry-run and --execute are mutually exclusive", file=sys.stderr)
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
