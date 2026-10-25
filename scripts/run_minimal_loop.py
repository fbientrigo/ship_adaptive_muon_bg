"""One iteration of the minimal utility-guided loop on the PR #1432 filter stage.

  score  (FairShip env)  U0 + TGeo initial volume for every pool row  -> <out>/pool_scores.npz + .json
  draw   (any env)       eligibility gate + reference/tilt draws        -> <out>/draws.csv + draws.json

Every path and hash the result depends on is written next to it.  See ``ship_muon_bg.minimal_loop`` for the semantics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from ship_muon_bg.adapters.current_main_mudis import _git_head
from ship_muon_bg.adapters.fairship import LEGACY_AFTERMS_TRANSFORM as TRANSFORM
from ship_muon_bg.adapters.fairship import CandidateInjectionRecord
from ship_muon_bg import minimal_loop as ml


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(a: np.ndarray, i: int) -> CandidateInjectionRecord:
    px, py, pz, x, y, z, pid, w = (float(v) for v in a[i])
    return CandidateInjectionRecord(f"row-{i}", px, py, pz, x, y, z, int(pid), w, 1.0)


def score(args) -> None:
    import audit_fairship_tgeo as A  # scripts/ sibling; needs the FairShip ROOT env

    A.configure_fairship(args.fairship)
    import ROOT  # type: ignore

    geo_file = ROOT.TFile.Open(str(args.geo))  # keep the file alive: a chained Open().Get() segfaults
    geom = geo_file.Get("FAIRGeom")
    a = ml.load_pool()
    u0 = np.full(len(a), np.nan)
    volume = np.full(len(a), "", dtype=object)
    for i in np.where(ml.valid_rows(a))[0]:
        c = record(a, i)
        x, y, z = TRANSFORM.apply(c)  # the exact placement the injection uses
        info, _ = A.trajectory(geom, dict(px=c.px, py=c.py, pz=c.pz, x=x, y=y, z=z))
        u0[i] = float(len(info["ray_sensitive_volumes"]) > 0)
        volume[i] = info["initial_volume"]
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "pool_scores.npz", u0=u0, initial_volume=volume.astype(str))
    meta = {"u0_definition": "straight TGeo ray from the transformed state crosses a LiSc* (SBT) volume; no field, no scattering",
            "pool": str(ml.POOL_NPZ), "pool_sha256": sha256(ml.POOL_NPZ), "geofile": str(args.geo), "geofile_sha256": sha256(args.geo),
            "fairship_dir": str(args.fairship), "fairship_commit": _git_head(args.fairship), "transform": TRANSFORM.manifest(),
            "n_rows": len(a), "n_scored": int(np.isfinite(u0).sum()), "n_u0_1": int((u0 == 1).sum()),
            "scores_sha256": sha256(args.out / "pool_scores.npz")}
    (args.out / "pool_scores.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta, indent=1))


def draw(args) -> None:
    a = ml.load_pool()
    s = np.load(args.scores, allow_pickle=False)
    u0, volume = s["u0"], s["initial_volume"]
    status = ml.eligibility(a, u0, volume)
    rows = ml.draw(status, u0, args.n, args.seed)
    for r in rows:
        r["status"], r["initial_volume"] = status[r["row_index"]], str(volume[r["row_index"]])
        r["p_gev"] = float(np.linalg.norm(a[r["row_index"], :3]))
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "draws.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    per_arm = {arm: Counter(r["row_index"] for r in rows if r["arm"] == arm) for arm in ml.ARMS}
    meta = {"scores": str(args.scores), "scores_sha256": sha256(args.scores), "seed": args.seed, "n_per_arm": args.n,
            "delta": ml.DELTA, "alpha": ml.ALPHA, "eligible_volumes": list(ml.ELIGIBLE_VOLUMES), "gate_status": "PROVISIONAL",
            "eligibility_counts": dict(Counter(status.tolist())),
            "ineligible_volume_counts": dict(Counter(volume[status == "INELIGIBLE_IN_SHIELD_MAGNET"].tolist())),
            "invalid_or_unscored_rows": {int(i): status[i] for i in np.where(np.isin(status, ["INVALID_ROW", "UNSCORED"]))[0]},
            "u0_1_rows": {int(i): status[i] for i in np.where(u0 == 1)[0]},
            "concentration": ml.concentration(status, u0),
            "drawn": {arm: {"executions": sum(c.values()), "distinct_rows": len(c), "multiplicity": {str(k): v for k, v in c.most_common()}}
                      for arm, c in per_arm.items()},
            "draws_sha256": sha256(args.out / "draws.csv")}
    (args.out / "draws.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps({k: meta[k] for k in ("eligibility_counts", "concentration", "drawn")}, indent=1))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("score")
    p.add_argument("--fairship", type=Path, required=True)
    p.add_argument("--geo", type=Path, required=True, help="geo_connector.root of an injection on the same checkout")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("draw")
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=20261025)
    args = ap.parse_args(argv)
    {"score": score, "draw": draw}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
