#!/usr/bin/env python
"""Run a D7 (empirical after-MS) density-lab campaign from a JSON config.

The single external-data configuration seam is ``dataset_path`` (in the
config, or via ``--dataset`` which overrides it): the repository afterMS
sample and a full local dataset run through the exact same code, differing
only in that path plus bounded operational settings (``--max-rows``,
``--artifact-root``). No tracked file needs editing to point at a full local
dataset -- use an environment variable in the config
(``"${SHIP_MUON_BG_LOCAL_DATA}/muonsFullMC_afterMS.pkl"``) or pass
``--dataset`` directly; neither commits a machine-specific path.

Examples
--------
Repository-fixture smoke run (bounded, CI-safe):

    python scripts/run_empirical_campaign.py \\
        --config configs/density_lab/empirical/d7_fixture_smoke_v0.json

Full local dataset, same config, only the path overridden:

    python scripts/run_empirical_campaign.py \\
        --config configs/density_lab/empirical/d7_fixture_smoke_v0.json \\
        --dataset "$SHIP_MUON_BG_LOCAL_DATA/muonsFullMC_afterMS.pkl" \\
        --artifact-root /path/to/local/artifacts

Validate configuration and dataset compatibility without training:

    python scripts/run_empirical_campaign.py \\
        --config configs/density_lab/empirical/d7_fixture_smoke_v0.json --dry-run

Runs execute independently per PDG track; a failed track records its status
and the other track continues. Completed identical run hashes are skipped
unless ``--force``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from ship_muon_bg.density_lab import (  # noqa: E402
    EmpiricalCampaignSpec,
    EmpiricalDataError,
    EmpiricalDatasetSpec,
    build_empirical_dataset,
    run_empirical_campaign_from_spec,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="path to a JSON D7 campaign config")
    parser.add_argument(
        "--dataset",
        default=None,
        help="override the config's dataset_path (env vars/~ expanded); the only flag that usually differs between the repository fixture and a full local run",
    )
    parser.add_argument("--pdg-ids", nargs="*", type=int, default=None, help="restrict to these PDG ids (default: config's pdg_ids)")
    parser.add_argument("--max-rows", type=int, default=None, help="override the config's max_rows bound")
    parser.add_argument("--artifact-root", default=None, help="override the artifact root directory")
    parser.add_argument("--force", action="store_true", help="re-run completed runs")
    parser.add_argument("--device", default=None, help="cpu | cuda | auto (overrides config)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve configuration and validate/load the dataset for every requested PDG track; never trains",
    )
    args = parser.parse_args()

    spec = EmpiricalCampaignSpec.from_json_file(args.config)
    if args.dataset:
        import os

        spec = _with_overrides(spec, dataset_path=os.path.expanduser(os.path.expandvars(args.dataset)))
    if args.pdg_ids:
        spec = _with_overrides(spec, pdg_ids=tuple(args.pdg_ids))
    if args.max_rows is not None:
        spec = _with_overrides(spec, max_rows=args.max_rows)
    if args.device:
        spec = _with_overrides(spec, device=args.device)

    if args.dry_run:
        return _dry_run(spec)

    summary = run_empirical_campaign_from_spec(
        spec,
        root=Path(args.artifact_root) if args.artifact_root else None,
        force=args.force,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "runs"}, indent=2))
    for record in summary["runs"]:
        print(
            "  {run_id}: status={status} technical={technical_status} scientific={scientific_status}".format(
                run_id=record.get("run_id"), status=record.get("status"),
                technical_status=record.get("technical_status"),
                scientific_status=record.get("scientific_status"),
            )
        )
    return 0


def _with_overrides(spec: EmpiricalCampaignSpec, **overrides) -> EmpiricalCampaignSpec:
    import dataclasses

    return dataclasses.replace(spec, **overrides)


def _dry_run(spec: EmpiricalCampaignSpec) -> int:
    """Resolve configuration and validate the dataset for every PDG track.

    Never trains. Loads, validates, PDG-filters, and three-way-splits each
    requested track (the same code path ``run_empirical_single`` would use),
    reporting row counts and provenance so a full local dataset can be
    checked before spending any training time on it.
    """

    print("config_hash={}".format(spec.config_hash()))
    print("dataset_path={}".format(spec.dataset_path))
    exit_code = 0
    for pdg_id in spec.pdg_ids:
        dataset_spec = EmpiricalDatasetSpec(
            dataset_path=spec.dataset_path, pdg_id=pdg_id, seed=spec.seed,
            val_fraction=spec.val_fraction, test_fraction=spec.test_fraction,
            max_rows=spec.max_rows, allow_zero_weight=spec.allow_zero_weight,
        )
        try:
            dataset = build_empirical_dataset(dataset_spec)
        except EmpiricalDataError as exc:
            print("pdg_id={}: BLOCKED: {}".format(pdg_id, exc))
            exit_code = 1
            continue
        print(
            "pdg_id={}: OK train={} val={} test={} source_file_dataset_hash={}".format(
                pdg_id, dataset.train.n_rows, dataset.validation.n_rows,
                dataset.test.n_rows, dataset.source_file_dataset_hash,
            )
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
