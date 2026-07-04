#!/usr/bin/env python
"""Thin CLI: build the v0 data-contract artifacts for a muon PKL file.

All business logic lives in ``ship_muon_bg.data_contracts``; this script only
parses arguments, calls into ``src/``, and writes the JSON artifacts:

- ``dataset_report.json``
- ``split_manifest.json``
- ``normalization.json``

Example
-------
    python scripts/build_dataset_report.py \
        --input tests/fixtures/muon_sample_tiny.pkl.gz \
        --output-dir artifacts/dataset_contract \
        --seed 1234
"""

from __future__ import annotations

import argparse
import os
import sys

# Make ``src/`` importable without requiring an installed package.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from ship_muon_bg.data_contracts import process_pkl, write_artifacts  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, help="Path to a trusted local gzip-PKL muon file."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write the JSON artifacts into.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Explicit deterministic split seed (no time-based seeding).",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Validation fraction in (0, 1). Default: 0.2.",
    )
    parser.add_argument(
        "--allow-zero-weight",
        action="store_true",
        help="Accept w == 0 (default rejects non-positive weights).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    artifacts = process_pkl(
        args.input,
        seed=args.seed,
        val_fraction=args.val_fraction,
        allow_zero_weight=args.allow_zero_weight,
    )
    written = write_artifacts(artifacts, args.output_dir)
    for key, path in written.items():
        print(f"wrote {key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
