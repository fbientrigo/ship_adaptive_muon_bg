"""Run the fixture-only D9 conditional-charge NF smoke or pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ship_muon_bg.density_lab.conditional_charge import run_fixture_pilot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    summary = run_fixture_pilot(config, output_dir=args.output_dir)
    print(json.dumps({
        "status": summary["status"],
        "output_dir": str(args.output_dir),
        "macro_validation_nll": summary["validation"]["macro_nll"],
        "worst_charge_validation_nll": summary["validation"]["worst_charge_nll"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
