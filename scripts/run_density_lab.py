#!/usr/bin/env python
"""Run a controlled density-lab campaign from a JSON config.

Examples
--------
    python scripts/run_density_lab.py --config configs/density_lab/smoke_v0.json
    python scripts/run_density_lab.py --config <cfg> --targets D3 D5 --models affine_tiny
    python scripts/run_density_lab.py --config <cfg> --force --device cpu

Runs execute independently; a failed run records its status/traceback and the
campaign continues. Completed identical run hashes are skipped unless
``--force``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from ship_muon_bg.density_lab import ExperimentConfig, run_campaign  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to a JSON experiment config")
    parser.add_argument("--artifact-root", default=None, help="override the artifact root directory")
    parser.add_argument("--force", action="store_true", help="re-run completed runs")
    parser.add_argument("--targets", nargs="*", default=None, help="restrict to these target ids")
    parser.add_argument("--models", nargs="*", default=None, help="restrict to these model names")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="restrict to these seeds")
    parser.add_argument("--device", default=None, help="cpu | cuda | auto (overrides config)")
    args = parser.parse_args()

    config = ExperimentConfig.from_json_file(args.config)
    summary = run_campaign(
        config,
        root=Path(args.artifact_root) if args.artifact_root else None,
        force=args.force,
        target_ids=set(args.targets) if args.targets else None,
        model_names=set(args.models) if args.models else None,
        seeds=set(args.seeds) if args.seeds else None,
        device=args.device,
    )
    print(json.dumps(
        {k: v for k, v in summary.items() if k != "runs"}, indent=2
    ))
    print("completed={n_completed} failed={n_failed} skipped={n_skipped}".format(**summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
