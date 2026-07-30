#!/usr/bin/env python
"""Thin CLI: validate a muon dataset, or build the v0 data-contract artifacts.

All business logic lives in ``ship_muon_bg.data_contracts``; this script only
parses arguments, calls into ``src/``, and writes JSON. Works identically on
the committed repository sample and on any external full local dataset --
only the ``--dataset`` path changes; no code path differs by scale
(``--max-rows`` bounds either one the same way, via a deterministic,
envelope-preserving subsample).

Two modes:

``--validate-only`` (bounded, never trains; recommended first step on any new
dataset, including a full local file): loads, validates, and emits one
machine-readable report -- schema, PDG counts, weight-column status,
duplicate-row check, requested-vs-available rows, memory-footprint estimate,
split feasibility, and (optionally) compatibility with a campaign config.
Writes to ``--output`` if given, else prints to stdout.

Default (no ``--validate-only``): the original v0 behavior, producing
``dataset_report.json`` / ``split_manifest.json`` / ``normalization.json`` in
``--output-dir`` (a 2-way train/val split; unchanged from prior versions of
this script).

Examples
--------
Validate the committed repository sample (never trains):

    python scripts/build_dataset_report.py \\
        --dataset data/samples/muonsFullMC_afterMS_sample.npz \\
        --validate-only --allow-zero-weight --seed 1234 \\
        --output artifacts/dataset_validation/afterms_sample_report.json

Validate a full local dataset the same way (only the path changes):

    python scripts/build_dataset_report.py \\
        --dataset "$SHIP_MUON_BG_LOCAL_DATA/muonsFullMC_afterMS.pkl" \\
        --validate-only --allow-zero-weight --seed 1234 --max-rows 200000 \\
        --output artifacts/dataset_validation/afterms_full_report.json

Build the legacy report/split/normalization artifact trio:

    python scripts/build_dataset_report.py \\
        --dataset tests/fixtures/muon_sample_tiny.pkl.gz \\
        --output-dir artifacts/dataset_contract \\
        --seed 1234
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

from ship_muon_bg.data_contracts import (  # noqa: E402
    DEFAULT_DUPLICATE_CHECK_ROW_LIMIT,
    build_validation_report,
    cap_rows,
    load_muon_array,
    process_array,
    write_artifacts,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        "--input",
        dest="dataset_path",
        required=True,
        help=(
            "Path to a trusted local muon file: gzip-PKL (legacy) or NPZ "
            "(preferred). Same flag for the repository sample or a full "
            "local dataset -- only the path differs."
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate and report only; never trains, never writes a split/normalization artifact.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help=(
            "Cap the loaded array to at most this many rows via a "
            "deterministic, envelope-preserving subsample (requires --seed). "
            "Applies identically to the repository sample and a full local "
            "dataset."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="(--validate-only) Write the single combined JSON report here; default prints to stdout.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="(default mode only) Directory to write dataset_report.json / split_manifest.json / normalization.json into.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Explicit deterministic seed (no time-based seeding). Required in default mode; optional in --validate-only (enables split-feasibility and --max-rows).",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Validation fraction in (0, 1). Default: 0.2.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="(--validate-only split-feasibility only) Test fraction in (0, 1). Default: 0.2.",
    )
    parser.add_argument(
        "--allow-zero-weight",
        action="store_true",
        help="Accept w == 0 (default rejects non-positive weights).",
    )
    parser.add_argument(
        "--campaign-config",
        default=None,
        help=(
            "(--validate-only) Path to a JSON config with optional "
            "dataset.n_train/n_validation/n_test and pdg_ids keys, to check "
            "row-count sufficiency per requested PDG track."
        ),
    )
    parser.add_argument(
        "--duplicate-check-row-limit",
        type=int,
        default=DEFAULT_DUPLICATE_CHECK_ROW_LIMIT,
        help=(
            "(--validate-only) Skip the exact-duplicate-row check above this "
            "many rows (default {}); a full local afterMS file is well past "
            "it by design, so validate that with --max-rows if you need the "
            "duplicate check.".format(DEFAULT_DUPLICATE_CHECK_ROW_LIMIT)
        ),
    )
    args = parser.parse_args(argv)
    if not args.validate_only and not args.output_dir:
        parser.error("--output-dir is required unless --validate-only is given")
    if not args.validate_only and args.seed is None:
        parser.error("--seed is required unless --validate-only is given")
    if args.max_rows is not None and args.seed is None:
        parser.error("--max-rows requires --seed (the subsample is seeded)")
    return args


def main(argv=None):
    args = parse_args(argv)

    if args.validate_only:
        array = load_muon_array(args.dataset_path)
        capped, available_before_cap = cap_rows(array, args.max_rows, seed=args.seed)
        campaign_config = None
        if args.campaign_config:
            with open(args.campaign_config, encoding="utf-8") as handle:
                campaign_config = json.load(handle)
        report = build_validation_report(
            capped,
            source_path=args.dataset_path,
            allow_zero_weight=args.allow_zero_weight,
            requested_max_rows=args.max_rows,
            available_rows_before_cap=available_before_cap,
            seed=args.seed,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            campaign_config=campaign_config,
            duplicate_check_row_limit=args.duplicate_check_row_limit,
        )
        payload = json.dumps(report, indent=2, sort_keys=True)
        if args.output:
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as handle:
                handle.write(payload + "\n")
            print(f"wrote validation report: {args.output}")
        else:
            print(payload)
        all_checks_passed = all(c["passed"] for c in report["validation"])
        return 0 if all_checks_passed else 1

    # Default mode: unchanged v0 behavior (2-way split + normalization),
    # extended only to accept NPZ input and --max-rows via the shared
    # loader/cap dispatch. All business logic stays in data_contracts.report.
    array = load_muon_array(args.dataset_path)
    capped, _available_before_cap = cap_rows(array, args.max_rows, seed=args.seed)
    artifacts = process_array(
        capped,
        source_path=args.dataset_path,
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
