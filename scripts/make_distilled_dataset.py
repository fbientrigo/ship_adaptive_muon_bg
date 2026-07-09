#!/usr/bin/env python
"""Thin CLI: distill a full Muon NTuples release into a committable ~40 MB NPZ.

All business logic lives in ``ship_muon_bg.data_contracts``; this script only
parses arguments, calls into ``src/``, and writes:

- ``<stem>.npz``                       -- the distilled subset (key ``muons``)
- ``<stem>_manifest.json``             -- provenance (source tag/sha256, seed, hashes)
- ``<stem>_distribution_report.json``  -- full-vs-distilled fidelity: KS, moments, BC

The source PKL is a **trusted local** release asset; it is never committed. Only
the distilled NPZ (plus the two small JSONs) belongs in git.

Example
-------
    python scripts/make_distilled_dataset.py \
        --input /scratch/muons_FullMC.pkl \
        --output-stem data/distilled/muons_FullMC_distilled \
        --n-rows 1700000 --seed 1234 \
        --source-tag v1.0.0-fullmc \
        --source-sha256 91273fe6...232ceb9c
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
    compare_distributions,
    dataset_hash,
    load_muon_pkl,
    representative_subset,
    save_subset_npz,
    schema,
)
from ship_muon_bg.data_contracts.subsampling import NPZ_ARRAY_KEY  # noqa: E402

MANIFEST_SCHEMA_VERSION = "0"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, help="Path to a trusted local full-release muon PKL."
    )
    parser.add_argument(
        "--output-stem",
        required=True,
        help="Output path without extension; '.npz', '_manifest.json', "
        "'_distribution_report.json' are appended.",
    )
    parser.add_argument(
        "--n-rows", type=int, required=True, help="Target number of rows to distill to."
    )
    parser.add_argument(
        "--seed", type=int, required=True, help="Explicit deterministic distill seed."
    )
    parser.add_argument(
        "--source-tag",
        default=None,
        help="Release tag the source asset came from (recorded in the manifest).",
    )
    parser.add_argument(
        "--source-sha256",
        default=None,
        help="SHA-256 of the full source file (recorded in the manifest).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    array = load_muon_pkl(args.input)
    subset, selected = representative_subset(array, args.n_rows, seed=args.seed)

    stem = args.output_stem
    os.makedirs(os.path.dirname(os.path.abspath(stem)), exist_ok=True)
    npz_path = f"{stem}.npz"
    manifest_path = f"{stem}_manifest.json"
    report_path = f"{stem}_distribution_report.json"

    save_subset_npz(npz_path, subset)

    # Fidelity evidence: distilled vs the full complement (independent samples).
    report = compare_distributions(array, subset, distilled_indices=selected)
    report["source_file"] = os.path.basename(args.input)
    report["source_release_tag"] = args.source_tag
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "contract_version": schema.CONTRACT_VERSION,
        "columns": list(schema.COLUMNS),
        "units": dict(schema.UNITS),
        "npz_array_key": NPZ_ARRAY_KEY,
        "source_file": os.path.basename(args.input),
        "source_release_tag": args.source_tag,
        "source_sha256": args.source_sha256,
        "source_n_rows": int(array.shape[0]),
        "source_dataset_hash": dataset_hash(array),
        "distillation": {
            "strategy": "uniform_core_plus_range_anchors",
            "seed": int(args.seed),
            "requested_n_rows": int(args.n_rows),
        },
        "distilled_n_rows": int(subset.shape[0]),
        "distilled_fraction_of_full": float(subset.shape[0] / array.shape[0]),
        "distilled_dataset_hash": dataset_hash(subset),
        "outputs": {
            "npz": os.path.basename(npz_path),
            "distribution_report": os.path.basename(report_path),
        },
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    worst_ks = max(c["ks_statistic"] for c in report["per_column"].values())
    print(f"source rows   : {array.shape[0]:,}")
    print(f"distilled rows: {subset.shape[0]:,}  ({manifest['distilled_fraction_of_full']*100:.1f}% of full)")
    print(f"worst KS D    : {worst_ks:.5f} (distilled vs complement, over {report['columns']})")
    print(f"wrote npz     : {npz_path}")
    print(f"wrote manifest: {manifest_path}")
    print(f"wrote report  : {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
