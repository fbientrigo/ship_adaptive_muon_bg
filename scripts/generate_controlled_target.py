#!/usr/bin/env python
"""Thin CLI: draw a controlled-target sample and write its artifacts.

All business logic lives in ``ship_muon_bg.benchmarks``; this script only
parses arguments, calls into ``src/``, and writes:

- ``target_samples.npz``    -- physical, raw_rows, component_id, physical_log_prob
- ``target_manifest.json``  -- the target's versioned parameter/manifest record
- ``sample_manifest.json``  -- this draw's provenance (seed, hashes, filenames)

Example
-------
    python scripts/generate_controlled_target.py \
        --target D0 --pdg-id 13 --n 4096 --seed 11 --plane-z 0.0 \
        --output-dir artifacts/controlled_targets/d0_seed11

``--pdg-id`` takes a PDG particle id, not an electric charge value:
``13`` = mu-, ``-13`` = mu+.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Make ``src/`` importable without requiring an installed package.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np  # noqa: E402

from ship_muon_bg.benchmarks import make_controlled_target  # noqa: E402
from ship_muon_bg.data_contracts import dataset_hash  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, choices=["D0", "D1", "D2"])
    parser.add_argument(
        "--pdg-id",
        required=True,
        type=int,
        choices=[13, -13],
        help="PDG particle id, not electric charge: 13 = mu-, -13 = mu+.",
    )
    parser.add_argument("--n", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--plane-z", required=True, type=float)
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write target_samples.npz/target_manifest.json/sample_manifest.json into.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    target = make_controlled_target(args.target)
    batch = target.sample(n=args.n, pdg_id=args.pdg_id, seed=args.seed)
    physical_log_prob = target.log_prob(batch.physical, pdg_id=args.pdg_id)
    raw_rows = batch.to_raw(plane_z=args.plane_z)

    os.makedirs(args.output_dir, exist_ok=True)

    samples_path = os.path.join(args.output_dir, "target_samples.npz")
    np.savez(
        samples_path,
        physical=batch.physical,
        raw_rows=raw_rows,
        component_id=batch.component_id,
        physical_log_prob=physical_log_prob,
    )

    target_manifest_path = os.path.join(args.output_dir, "target_manifest.json")
    with open(target_manifest_path, "w", encoding="utf-8") as handle:
        json.dump(target.manifest(), handle, indent=2, sort_keys=True)
        handle.write("\n")

    output_filenames = {
        "samples": "target_samples.npz",
        "target_manifest": "target_manifest.json",
        "sample_manifest": "sample_manifest.json",
    }
    sample_manifest = {
        "target_id": args.target,
        "target_config_hash": target.config_hash(),
        "pdg_id": args.pdg_id,
        "n": args.n,
        "seed": args.seed,
        "plane_z": args.plane_z,
        "raw_dataset_hash": dataset_hash(raw_rows),
        "output_filenames": output_filenames,
    }
    sample_manifest_path = os.path.join(args.output_dir, "sample_manifest.json")
    with open(sample_manifest_path, "w", encoding="utf-8") as handle:
        json.dump(sample_manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print("wrote samples: {}".format(samples_path))
    print("wrote target_manifest: {}".format(target_manifest_path))
    print("wrote sample_manifest: {}".format(sample_manifest_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
