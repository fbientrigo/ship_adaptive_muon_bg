#!/usr/bin/env python
"""Build report outputs from a completed campaign directory (reads only).

    python scripts/build_density_report.py --campaign-dir artifacts/density_lab/smoke_v0

Writes benchmark_summary.{json,csv,md}, limitations.md and the plot PNGs under
``<campaign-dir>/report/``. Never retrains a model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from ship_muon_bg.density_lab.reporting import build_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", required=True, help="path to artifacts/density_lab/<experiment_id>")
    parser.add_argument("--no-plots", action="store_true", help="skip plot generation")
    args = parser.parse_args()

    result = build_report(Path(args.campaign_dir), with_plots=not args.no_plots)
    print(json.dumps(result["summary"], indent=2, default=str))
    print("report_dir:", result["report_dir"])
    print("plots:", result["plots"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
