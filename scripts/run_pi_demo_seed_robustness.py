#!/usr/bin/env python
"""PI-demo seed-robustness check -- is a per-seed NF result seed luck?

Trains the SAME bounded NF arena (NOMINAL_PHYSICAL + 4 named tilts, both
PDG tracks) already used for Figure 05, once per requested seed, then
reports the seed-to-seed spread of ``generated_b_toy_occupancy`` per
(pdg_id, variant_id). This answers: does an observed per-seed result (e.g.
one PDG track's mild-tilt occupancy undershooting its target) hold up
across independent training runs, or is it an artifact of that one run's
particular initialization/optimization trajectory?

This script does not modify anything under ``artifacts/pi_demo/`` (the
already-reviewed P0 deliverable) -- it only trains additional seeds via the
unmodified ``scripts/run_utility_tilt_campaign.py`` CLI (one subprocess per
seed not already present on disk) and writes its own summary under
``artifacts/pi_demo/seed_robustness/``.

``B_toy`` is a project-defined SYNTHETIC diagnostic region, never a
validated FairShip danger metric; ``U_A``/``U_P`` are synthetic utility
scores, never calibrated probabilities.

Example
-------
Check seeds 11 (reusing the existing pi_demo_v0 arena report), 12, 13::

    python scripts/run_pi_demo_seed_robustness.py \\
        --dataset data/samples/muonsFullMC_afterMS_sample.npz \\
        --pdg-ids 13 -13 --seeds 11 12 13 --sampler-seed 7 \\
        --model-config configs/density_lab/utility_tilt/pi_demo_cpu_v0.json \\
        --device cpu
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from ship_muon_bg.density_lab import pi_demo_seed_robustness as sr  # noqa: E402

TRAIN_TILT_IDS = ("NOMINAL_PHYSICAL", "UA_d0p9_a01", "UA_d0p1_a04", "UP_d0p9_a01", "UP_d0p1_a04")
VARIANT_IDS = TRAIN_TILT_IDS  # every trained variant, including the untilted arm


def _experiment_id_for_seed(seed: int) -> str:
    return "pi_demo_seed_robustness_seed{}".format(seed)


def _arena_report_path(artifact_root: Path, experiment_id: str, pdg_id: int) -> Path:
    return artifact_root / experiment_id / "arena_report" / "pdg_{}".format(pdg_id) / "arena_summary.json"


def _train_one_seed(args, seed: int) -> None:
    """Invoke the unmodified campaign CLI as a subprocess -- no in-process
    reuse of a script that was not designed as a library; this keeps every
    seed's run fully isolated and matches the exact command a human operator
    would type (see this script's own docstring / the README)."""

    experiment_id = _experiment_id_for_seed(seed)
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "run_utility_tilt_campaign.py"),
        "--dataset", args.dataset,
        "--pdg-ids", *[str(p) for p in args.pdg_ids],
        "--seed", str(seed),
        "--sampler-seed", str(args.sampler_seed),
        "--experiment-id", experiment_id,
        "--model-config", args.model_config,
        "--train-tilt-ids", *TRAIN_TILT_IDS,
        "--device", args.device,
        "--arena-report",
    ]
    if args.artifact_root:
        cmd += ["--artifact-root", args.artifact_root]
    print("seed {}: running {}".format(seed, " ".join(cmd)))
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pdg-ids", nargs="+", type=int, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(sr.DEFAULT_SEEDS))
    parser.add_argument("--sampler-seed", type=int, required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--artifact-root", default="artifacts/density_lab")
    parser.add_argument(
        "--reuse-seed", type=int, default=None,
        help="a seed already trained under --reuse-experiment-id (e.g. 11, "
             "matching the P0 bundle's seed) -- reused ONLY for this exact seed, "
             "never applied to any other requested --seeds value",
    )
    parser.add_argument(
        "--reuse-experiment-id", default=None,
        help="the experiment-id --reuse-seed was trained under (e.g. the P0 "
             "bundle's 'pi_demo_v0'); both --reuse-seed and --reuse-experiment-id "
             "must be given together",
    )
    parser.add_argument(
        "--out-dir", default="artifacts/pi_demo/seed_robustness",
        help="where to write seed_robustness_summary.csv/.json (never artifacts/pi_demo/ itself)",
    )
    args = parser.parse_args()
    if bool(args.reuse_seed) != bool(args.reuse_experiment_id):
        parser.error("--reuse-seed and --reuse-experiment-id must be given together")

    artifact_root = Path(args.artifact_root)
    arena_report_path_by_pdg_and_seed = {}
    for seed in args.seeds:
        experiment_id = _experiment_id_for_seed(seed)
        reused = False
        # Reuse is scoped to the single seed it was actually trained under --
        # a report built for one seed must never be silently substituted for
        # a different requested seed (that would collapse the whole point of
        # this script: measuring genuine seed-to-seed spread).
        if args.reuse_seed is not None and seed == args.reuse_seed:
            reuse_paths = {
                pdg_id: _arena_report_path(artifact_root, args.reuse_experiment_id, pdg_id)
                for pdg_id in args.pdg_ids
            }
            if all(p.exists() for p in reuse_paths.values()):
                for pdg_id, p in reuse_paths.items():
                    arena_report_path_by_pdg_and_seed[(pdg_id, seed)] = p
                reused = True
                print("seed {}: reusing existing arena report under experiment-id {!r}".format(
                    seed, args.reuse_experiment_id
                ))
        if not reused:
            _train_one_seed(args, seed)
            for pdg_id in args.pdg_ids:
                arena_report_path_by_pdg_and_seed[(pdg_id, seed)] = _arena_report_path(
                    artifact_root, experiment_id, pdg_id
                )

    rows = sr.aggregate_seed_robustness(
        pdg_ids=args.pdg_ids, variant_ids=VARIANT_IDS,
        arena_report_path_by_pdg_and_seed=arena_report_path_by_pdg_and_seed,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "seed_robustness_summary.csv"
    sr.write_seed_robustness_csv(rows, csv_path)
    manifest_path = out_dir / "seed_robustness_manifest.json"
    manifest_path.write_text(json.dumps(
        {
            "schema_version": sr.SCHEMA_VERSION,
            "seeds": args.seeds,
            "sampler_seed": args.sampler_seed,
            "pdg_ids": args.pdg_ids,
            "variant_ids": list(VARIANT_IDS),
            "arena_report_paths": {
                "{},{}".format(pdg, seed): str(path)
                for (pdg, seed), path in arena_report_path_by_pdg_and_seed.items()
            },
        },
        indent=2, sort_keys=True, default=str,
    ))

    print(json.dumps({"n_rows": len(rows), "csv": str(csv_path), "manifest": str(manifest_path)}, indent=2))
    for row in rows:
        print(
            "pdg={} variant={}: q_theta mean={:.4f} std={:.4f} min={:.4f} max={:.4f} "
            "same_side_of_p0={} (p0_mean={:.4f})".format(
                row["pdg_id"], row["variant_id"], row["q_theta_mean"], row["q_theta_std"],
                row["q_theta_min"], row["q_theta_max"], row["all_seeds_same_side_of_p0"], row["p0_mean"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
