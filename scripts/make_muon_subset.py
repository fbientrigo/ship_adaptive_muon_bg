#!/usr/bin/env python
"""Thin CLI: build a small, representative, committable muon subset.

All business logic lives in ``ship_muon_bg.data_contracts.subsampling`` and the
existing data-contract helpers; this script only parses arguments, calls into
``src/``, and writes the sample files plus a provenance manifest:

- ``<stem>.pkl.gz``       -- gzip-pickled subset (legacy-loader compatible)
- ``<stem>.npz``          -- NPZ subset (preferred; no code execution on load)
- ``<stem>_manifest.json`` -- source provenance, seed, row counts, subset hash

The source PKL is a **trusted local** Muon NTuples v1.0 release asset; it is
never committed. Only the small subset produced here belongs in git.

Example
-------
    python scripts/make_muon_subset.py \
        --input /scratch/muons_FullMC.pkl \
        --output-stem data/samples/muons_FullMC_sample \
        --n-rows 40000 --seed 1234 \
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
    dataset_hash,
    load_muon_pkl,
    representative_subset,
    save_subset_npz,
    save_subset_pkl_gz,
    schema,
)
from ship_muon_bg.data_contracts.subsampling import NPZ_ARRAY_KEY  # noqa: E402

MANIFEST_SCHEMA_VERSION = "0"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, help="Path to a trusted local source muon PKL."
    )
    parser.add_argument(
        "--output-stem",
        required=True,
        help="Output path without extension; '.pkl.gz', '.npz', '_manifest.json' appended.",
    )
    parser.add_argument(
        "--n-rows", type=int, required=True, help="Target number of rows in the subset."
    )
    parser.add_argument(
        "--seed", type=int, required=True, help="Explicit deterministic subsample seed."
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
    subset, _selected = representative_subset(array, args.n_rows, seed=args.seed)

    stem = args.output_stem
    os.makedirs(os.path.dirname(os.path.abspath(stem)), exist_ok=True)
    pkl_path = f"{stem}.pkl.gz"
    npz_path = f"{stem}.npz"
    manifest_path = f"{stem}_manifest.json"

    save_subset_pkl_gz(pkl_path, subset)
    save_subset_npz(npz_path, subset)

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
        "sampling": {
            "strategy": "uniform_core_plus_range_anchors",
            "seed": int(args.seed),
            "requested_n_rows": int(args.n_rows),
        },
        "subset_n_rows": int(subset.shape[0]),
        "subset_dataset_hash": dataset_hash(subset),
        "outputs": {
            "pkl_gz": os.path.basename(pkl_path),
            "npz": os.path.basename(npz_path),
        },
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"source rows : {array.shape[0]}")
    print(f"subset rows : {subset.shape[0]}")
    print(f"wrote pkl.gz : {pkl_path}")
    print(f"wrote npz    : {npz_path}")
    print(f"wrote manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
