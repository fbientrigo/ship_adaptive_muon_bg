"""Run the D9 v0 conditional utility-tilt fixture arena.

Fixture-only. Trains one shared charge-conditional affine-coupling flow per
``(variant, model seed)`` cell with deterministic per-epoch direct sampling
from the within-charge nominal or utility-tilted empirical law, then writes
the deterministic arena reports.

Seeds are the outer loop and variants the inner loop, so a truncated budget
always completes all four variants for the first seed before moving on; the
run summary records the exact completed matrix and reports
``CONDITIONAL_UTILITY_ARENA_PARTIAL`` when it is short of the request.

Examples
--------

    python scripts/run_conditional_utility_arena.py \
        configs/density_lab/conditional_utility/d9_fixture_smoke_v0.json \
        --output-root runs/d9_conditional_utility_smoke_v0

    python scripts/run_conditional_utility_arena.py \
        configs/density_lab/conditional_utility/d9_fixture_arena_v0.json \
        --output-root runs/d9_conditional_utility_arena_v0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ship_muon_bg.density_lab.conditional_utility import run_conditional_utility_arena


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="override the config's variant list (canonical variant ids)",
    )
    parser.add_argument(
        "--model-seeds",
        type=int,
        nargs="+",
        default=None,
        help="override the config's model seeds",
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    summary = run_conditional_utility_arena(
        config,
        output_dir=args.output_root,
        variants=args.variants,
        model_seeds=args.model_seeds,
    )
    print(
        json.dumps(
            {
                "completion_status": summary["completion_status"],
                "completed": len(summary["completed_matrix"]),
                "requested": len(summary["requested_matrix"]),
                "preprocessing_hash_by_model_seed": summary[
                    "preprocessing_hash_by_model_seed"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
