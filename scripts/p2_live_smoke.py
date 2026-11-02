"""Live P2 smoke against a FairShip checkout: small, controlled, ~1 min. Run under that checkout's pixi env; prints one JSON.

Input: two copies of one stage-1 entry whose ``PlaneHAPoint`` transverse momentum is shifted (a mechanism test, NOT a
physical sample) so that every muon of the event leaves through the side. Two stage-2 runs through ``run_stage2_p2.py``:
without ``--FollowMuon`` (P2-only events exist and the standard output loses them) and with it (the sensitive cavern keeps
them). Reports generated / kept / extracted P2 crossings, lost event ids, provenance status and the P2 placement read from
the produced geofile.

    pixi run --manifest-path <FairShip>/pixi.toml python scripts/p2_live_smoke.py --fairship <FairShip> --stage1 <file> --work <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "src"), str(HERE.parent)]

SOURCE_ENTRY, SHIFTS_GEV = 9, (-6.0, -3.0)  # entry 9 of seed-201 stage 1: 10 clones of one pion; shifts are common to the event


def make_input(src: Path, dst: Path) -> None:
    import ROOT  # type: ignore

    fin = ROOT.TFile.Open(str(src))
    tin = fin.Get("cbmsim")
    fout = ROOT.TFile(str(dst), "recreate")
    for name in ("FileSummary", "FileHeader"):  # FileSummary: absorber-field guard of run_simScript
        obj = fin.Get(name)
        if obj:
            fout.WriteObject(obj, name)
    tout = tin.CloneTree(0)
    for shift in SHIFTS_GEV:
        tin.GetEntry(SOURCE_ENTRY)
        for p in tin.PlaneHAPoint:
            p.SetMomentum(ROOT.TVector3(p.GetPx() + shift, p.GetPy(), p.GetPz()))
        tout.Fill()
    fout.cd()
    tout.Write()
    fout.Close()


def run(fairship: Path, stage1: Path, work: Path, follow: bool) -> dict:
    from ship_muon_bg.adapters.fairship_p2 import compare_with_standard, read_crossings
    from ship_muon_bg.adapters.fairship_p2_source import group_summary, read_stage1, read_stage2_tracks, reconstruct

    out = work / ("follow" if follow else "nofollow")
    out.mkdir(parents=True, exist_ok=True)
    args = ["-n", "2", "-i", "0", "-f", str(stage1), "-S", "7", "-s", "7", "--MuonBack", "--FastMuon", "--tag", "s7", "-o", "."]
    if follow:
        args.insert(args.index("--FastMuon"), "--FollowMuon")
    cmd = [sys.executable, str(HERE / "run_stage2_p2.py"), "--fairship", str(fairship), "--", *args]
    done = subprocess.run(cmd, cwd=out, capture_output=True, text=True, env={**os.environ, "QT_QPA_PLATFORM": "offscreen"})
    (out / "run.log").write_text(done.stdout + done.stderr)
    raw, std = out / "sim_s7.unfiltered.root", out / "sim_s7.root"
    row = {"returncode": done.returncode, "unfiltered_exists": raw.exists(), "standard_exists": std.exists()}
    if not (raw.exists() and std.exists()):
        return row
    raw_rows, std_rows = read_crossings(raw), read_crossings(std)
    cmp = compare_with_standard(raw_rows, std_rows)
    sha = hashlib.sha256(stage1.read_bytes()).hexdigest()
    prov = reconstruct(raw_rows, read_stage2_tracks(raw), read_stage1(stage1), stage1_sha=sha, stage2_sha=hashlib.sha256(raw.read_bytes()).hexdigest())
    row.update({"generated": cmp["generated"], "kept": cmp["kept"], "lost": len(cmp["lost"]), "lost_events": cmp["lost_events"],
                "extra_in_standard": cmp["extra_in_standard"], "detids": sorted({r["det_id"] for r in raw_rows}),
                "provenance": dict(Counter(r["status"] for r in prov)), "groups": {k: v["n_groups"] for k, v in group_summary(prov).items() if isinstance(v, dict)},
                "geofile": str(out / "geo_s7.root")})
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fairship", required=True, type=Path)
    ap.add_argument("--stage1", required=True, type=Path)
    ap.add_argument("--work", required=True, type=Path)
    ns = ap.parse_args()
    ns.work.mkdir(parents=True, exist_ok=True)
    synth = ns.work / "synth_stage1.root"
    make_input(ns.stage1, synth)
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(2) as pool:  # two FairShip processes
        nofollow, follow = pool.map(lambda f: run(ns.fairship, synth, ns.work, f), (False, True))
    from ship_muon_bg.adapters.fairship_p2 import geometry_manifest

    m = geometry_manifest(nofollow["geofile"]) if "geofile" in nofollow else {}
    print(json.dumps({"nofollow": nofollow, "follow": follow,
                      "p2": m.get("nodes", {}).get("UpstreamTaggerScoringPlane2_2"), "shield_end": m.get("shield_end_from_nodes"),
                      "p2_gap": m.get("p2_center_minus_shield_end"), "fairship_library": m.get("fairship_library")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
