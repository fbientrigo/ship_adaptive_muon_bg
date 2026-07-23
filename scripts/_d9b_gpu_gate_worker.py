#!/usr/bin/env python3
"""_d9b_gpu_gate_worker.py: D9B GPU-gate CUDA child-process worker (Gate B, section 4).

Never invoked interactively -- spawned as a single subprocess at a time by
``ship_muon_bg.afterms.d9b.gpu_gate`` / ``scripts/run_afterms_d9_gpu_hpo.py``,
always via ``.venv\\Scripts\\python.exe``. Two modes:

    train    Train exactly one (candidate, seed) by calling the existing
             ``ship_muon_bg.afterms.d9.runner.train_candidate_seed`` -- this
             worker never reimplements the training loop, only wraps it with
             CUDA memory telemetry and an optional bounded interrupt hook
             (``--interrupt-after-epochs``, checked once per epoch boundary,
             the same mechanism ``train_candidate_seed`` already exposes).

    reload   Reload a completed run's checkpoints and generate a small
             deterministic physical-space sample. Never reads a test shard:
             it imports only the test-free helpers from
             ``ship_muon_bg.afterms.d9.evaluate`` (``_reconstruct_pipeline``,
             ``generate_deterministic_samples``, ``negative_pz_diagnostics``),
             never ``evaluate_candidate_seed`` or ``load_test_split``, and
             this module has no parameter through which a test shard path
             could be supplied.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))


def _cuda_memory_snapshot(device: str) -> dict:
    import torch

    if device != "cuda" or not torch.cuda.is_available():
        return {"allocated_bytes": 0, "reserved_bytes": 0}
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
    }


def _cuda_peak_snapshot(device: str) -> dict:
    import torch

    if device != "cuda" or not torch.cuda.is_available():
        return {"allocated_bytes": 0, "reserved_bytes": 0}
    return {
        "allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def cmd_train(args) -> int:
    import torch

    from ship_muon_bg.afterms.d9 import runner as d9runner

    candidate_config = json.loads(Path(args.candidate_config_json).read_text(encoding="utf-8"))
    shard_dir = Path(args.shard_dir)

    # Structural test-set isolation: only train_shards/validation_shards are
    # ever read here. There is no test_shards read path in this function.
    train_raw = d9runner.load_concatenated_shards(shard_dir, candidate_config["train_shards"])
    validation_raw = d9runner.load_concatenated_shards(shard_dir, candidate_config["validation_shards"])

    if args.device == "cuda":
        if not torch.cuda.is_available():
            print("ERROR: --device cuda requested but CUDA is not available", file=sys.stderr)
            return 4
        torch.cuda.reset_peak_memory_stats()

    initial = _cuda_memory_snapshot(args.device)

    interrupt_flag = None
    if args.interrupt_after_epochs is not None:
        calls = {"n": 0}
        threshold = args.interrupt_after_epochs

        def _interrupt_flag() -> bool:
            calls["n"] += 1
            return calls["n"] > threshold

        interrupt_flag = _interrupt_flag

    try:
        result = d9runner.train_candidate_seed(
            candidate_config,
            args.seed,
            train_raw,
            validation_raw,
            artifact_root=Path(args.artifact_root),
            repo_root=REPO_ROOT,
            device=args.device,
            resume=args.resume,
            interrupt_flag=interrupt_flag,
        )
    except Exception as exc:
        telemetry = {
            "device": args.device,
            "initial": initial,
            "peak": _cuda_peak_snapshot(args.device),
            "final": _cuda_memory_snapshot(args.device),
            "error": f"{type(exc).__name__}: {exc}",
        }
        Path(args.telemetry_out).write_text(json.dumps(telemetry, indent=2, default=str), encoding="utf-8")
        raise

    peak = _cuda_peak_snapshot(args.device)
    final = _cuda_memory_snapshot(args.device)

    telemetry = {
        "device": args.device,
        "initial": initial,
        "peak": peak,
        "final": final,
        "result": result,
    }
    Path(args.telemetry_out).write_text(json.dumps(telemetry, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["status"] in (d9runner.STATUS_COMPLETED, d9runner.STATUS_INTERRUPTED) else 1


def cmd_reload(args) -> int:
    import numpy as np

    from Nflow.registry import create_density_estimator
    from ship_muon_bg.afterms.d8.reconstruction import sample_hash
    from ship_muon_bg.afterms.d9 import checkpoint as ckpt
    from ship_muon_bg.afterms.d9 import evaluate as d9eval
    from ship_muon_bg.afterms.d9 import runner as d9runner

    run_dir = Path(args.run_dir)
    candidate_config = json.loads((run_dir / "training_config.json").read_text(encoding="utf-8"))

    best_path = run_dir / "checkpoints" / ckpt.FILENAME_BY_SCOPE[ckpt.SCOPE_BEST]
    final_path = run_dir / "checkpoints" / ckpt.FILENAME_BY_SCOPE[ckpt.SCOPE_FINAL]
    best_bundle = ckpt.load_bundle(best_path)
    final_bundle = ckpt.load_bundle(final_path)

    compat_fields = (
        "schema_version", "candidate_id", "architecture_config", "feature_order",
        "pdg_policy", "preprocessing_hash", "target_measure", "weighting_policy",
        "weighting_estimator_version", "seed", "semantic_training_hash", "execution_policy_hash",
        "sampling_contract_version",
    )
    expected = {f: best_bundle.get(f) for f in compat_fields}
    violations = ckpt.verify_compatibility(final_bundle, expected)

    pipeline = d9eval._reconstruct_pipeline(run_dir / "preprocessing" / "preprocessing.json")

    estimator = create_density_estimator(
        {"family": candidate_config["model_family"], "params": d9runner._architecture_params(candidate_config)},
        dimension=5, device=args.device,
    )
    estimator._build_module(seed=int(best_bundle["seed"]))
    estimator._module.load_state_dict(best_bundle["model_state_dict"])
    estimator._module.eval()

    generated = d9eval.generate_deterministic_samples(estimator, args.sample_count, args.generation_seed)
    generated_physical = pipeline.inverse(generated)

    all_finite = bool(np.all(np.isfinite(generated_physical)))
    negative_pz = d9eval.negative_pz_diagnostics(generated_physical[:, 2])

    result = {
        "run_dir": str(run_dir),
        "checkpoint_scope_used": ckpt.SCOPE_BEST,
        "checkpoint_reload": {
            "best_epoch": best_bundle["epoch"],
            "final_epoch": final_bundle["epoch"],
            "compatibility_violations": violations,
            "compatible": not violations,
        },
        "deterministic_sample": {
            "sample_count": int(generated_physical.shape[0]),
            "generation_seed": args.generation_seed,
            "sample_hash_sha256": sample_hash(generated_physical),
            "all_finite": all_finite,
            "negative_pz_diagnostics": negative_pz,
        },
    }
    Path(args.output_json).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))
    return 0 if (not violations and all_finite) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    p_train = sub.add_parser("train", help="Train exactly one (candidate, seed) via the existing D9 runner.")
    p_train.add_argument("--candidate-config-json", required=True)
    p_train.add_argument("--seed", type=int, required=True)
    p_train.add_argument("--shard-dir", required=True)
    p_train.add_argument("--artifact-root", required=True)
    p_train.add_argument("--device", default="cpu")
    p_train.add_argument("--resume", action="store_true")
    p_train.add_argument("--interrupt-after-epochs", type=int, default=None)
    p_train.add_argument("--telemetry-out", required=True)
    p_train.set_defaults(func=cmd_train)

    p_reload = sub.add_parser("reload", help="Reload a completed run's checkpoints; never reads a test shard.")
    p_reload.add_argument("--run-dir", required=True)
    p_reload.add_argument("--device", default="cpu")
    p_reload.add_argument("--sample-count", type=int, default=64)
    p_reload.add_argument("--generation-seed", type=int, default=20260720)
    p_reload.add_argument("--output-json", required=True)
    p_reload.set_defaults(func=cmd_reload)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
