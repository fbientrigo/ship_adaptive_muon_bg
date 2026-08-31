#!/usr/bin/env python
"""PI-demo figures -- render Figures 01-05 from already-built artifacts.

This script does not recompute the summary CSV, the Table A/Table B
reference tables, or NF training -- all of those already exist on disk
(``artifacts/pi_demo/utility_guided_summary.csv``,
``artifacts/density_lab/pi_demo_v0/tables/``,
``artifacts/density_lab/pi_demo_v0/arena_report/``). It only reads them and
calls ``ship_muon_bg.density_lab.pi_demo_plots``.

``B_toy`` is a project-defined SYNTHETIC diagnostic region, never a validated
FairShip danger metric; ``U_A``/``U_P`` are synthetic utility scores, never
calibrated probabilities. See ``pi_demo_plots.py``'s module docstring for the
full scope statement.

Example
-------
Regenerate every figure from the committed artifacts::

    python scripts/run_pi_demo.py --figures 01 02 03 04 05
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from ship_muon_bg.density_lab import pi_demo_plots as plots  # noqa: E402

ALL_FIGURE_IDS = ("01", "02", "03", "04", "05")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tables-root", default="artifacts/density_lab/pi_demo_v0/tables")
    parser.add_argument("--summary-csv", default="artifacts/pi_demo/utility_guided_summary.csv")
    parser.add_argument("--arena-report-root", default="artifacts/density_lab/pi_demo_v0/arena_report")
    parser.add_argument("--out-dir", default="artifacts/pi_demo/figures")
    parser.add_argument(
        "--figures", nargs="+", default=list(ALL_FIGURE_IDS), choices=list(ALL_FIGURE_IDS),
        help="which figure IDs to (re)generate; default: all",
    )
    args = parser.parse_args()

    tables_root = Path(args.tables_root)
    summary_csv = Path(args.summary_csv)
    arena_report_root = Path(args.arena_report_root)
    out_dir = Path(args.out_dir)

    csv_rows = plots.load_summary_rows(summary_csv) if summary_csv.exists() else []

    written: dict = {}

    if "01" in args.figures:
        written["01"] = plots.plot_01_tilt_enrichment_vs_alpha(csv_rows, tables_root, out_dir)
    if "02" in args.figures:
        written["02"] = plots.plot_02_enrichment_vs_concentration(csv_rows, out_dir)
    if "03" in args.figures:
        written["03"] = plots.plot_03_support_preservation(tables_root, out_dir)
    if "04" in args.figures:
        written["04"] = plots.plot_04_direct_sampler_validation(tables_root, out_dir)
    if "05" in args.figures:
        arena_reports = {}
        caveat = None
        run_manifest_path = summary_csv.parent / "run_manifest.json"
        if run_manifest_path.exists():
            manifest = json.loads(run_manifest_path.read_text())
            caveat = (manifest.get("nf_arena") or {}).get("caveat")
        if arena_report_root.exists():
            for pdg_dir in sorted(arena_report_root.iterdir()):
                if not pdg_dir.is_dir() or not pdg_dir.name.startswith("pdg_"):
                    continue
                pdg_id = int(pdg_dir.name[len("pdg_"):])
                report_path = pdg_dir / "arena_summary.json"
                if report_path.exists():
                    arena_reports[pdg_id] = json.loads(report_path.read_text())
        written["05"] = plots.plot_05_nf_preserves_tilt(arena_reports, out_dir, caveat=caveat)

    print(json.dumps(written, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
