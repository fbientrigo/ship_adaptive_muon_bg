"""Lossless P2 extraction: the --MuonBack filter, the driver, the proposed native option, and a live smoke.

``local_env`` tests read runs kept under ``P2_ARTIFACTS`` (see ``notes/p2_lossless_v0.md`` for how they were made).
``test_live_*`` run FairShip (about one minute); set ``FAIRSHIP_ROOT`` (a built checkout) and ``P2_ARTIFACTS``.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from scripts import run_stage2_p2 as drv
from ship_muon_bg.adapters.fairship_p2 import compare_with_standard, read_crossings

ART = Path(os.environ.get("P2_ARTIFACTS", "/nonexistent"))
FS = os.environ.get("FAIRSHIP_ROOT")
pytestmark = pytest.mark.local_env


def need(rel: str) -> Path:
    pytest.importorskip("ROOT")
    if not (ART / rel).exists():
        pytest.skip(f"artifact missing: {ART / rel}")
    return ART / rel


def cmp(raw: str, std: str):
    return compare_with_standard(read_crossings(need(raw)), read_crossings(need(std)))


def test_with_followmuon_no_p2_crossing_is_lost_and_the_driver_does_not_change_the_standard_output():
    for raw, std, stock in (("synth_p2only/u_driver/sim_s7.unfiltered.root", "synth_p2only/u_driver/sim_s7.root", "synth_p2only/u_stock/sim_s7.root"),):
        c = cmp(raw, std)
        assert (c["generated"], c["kept"], c["lost"]) == (c["generated"], c["generated"], [])
        a, b = read_crossings(need(std)), read_crossings(need(stock))
        assert [(r["mc_event_id"], r["track_id"]) for r in a] == [(r["mc_event_id"], r["track_id"]) for r in b]
        assert max(abs(x["p"] - y["p"]) for x, y in zip(a, b)) < 1e-9 and max(abs(x["x"] - y["x"]) for x, y in zip(a, b)) < 1e-9


def test_without_followmuon_the_standard_output_loses_p2_only_events_and_the_unfiltered_file_keeps_them():
    c = cmp("synth_p2only/nf_driver/sim_s7.unfiltered.root", "synth_p2only/nf_driver/sim_s7.root")
    assert (c["generated"], c["kept"], len(c["lost"])) == (117, 19, 98) and c["lost_events"] == list(range(1, 11)) and not c["extra_in_standard"]
    stock = read_crossings(need("synth_p2only/nf_stock/sim_s7.root"))
    assert len(stock) == c["kept"]  # the driver's standard file is the stock one


def test_real_stage1_without_followmuon_loses_p2_states_with_known_event_ids():
    lost = {}
    for s in (201, 202, 203):
        c = cmp(f"p2_real_nofollow/seed_{s}/sim_s{s}.unfiltered.root", f"p2_real_nofollow/seed_{s}/sim_s{s}.root")
        lost[s] = (c["generated"], c["kept"], c["lost_events"])
    assert lost == {201: (16, 9, [9, 14, 15]), 202: (2, 0, [16, 36]), 203: (1, 0, [26])}


def test_proposed_native_option_keeps_every_crossing():
    c = compare_with_standard(read_crossings(need("synth_p2only/nf_driver/sim_s7.unfiltered.root")), read_crossings(need("synth_p2only/nf_native/sim_s7.root")))
    assert (c["generated"], c["kept"]) == (117, 117)


def test_lost_crossings_keep_a_unique_stage1_origin():
    from ship_muon_bg.adapters.fairship_p2_source import read_stage1, read_stage2_tracks, reconstruct

    s1 = need("synth_p2only/synth_stage1_u.root")
    raw = need("synth_p2only/nf_driver/sim_s7.unfiltered.root")
    rows = reconstruct(read_crossings(raw), read_stage2_tracks(raw), read_stage1(s1), stage1_sha=hashlib.sha256(s1.read_bytes()).hexdigest(), stage2_sha="0" * 64)
    assert len(rows) == 117 and {r["status"] for r in rows} == {"RESOLVED"} and all(r["split_set"] == 0 for r in rows)
    assert {r["stage1_entry"] for r in rows} == set(range(12))  # one progenitor per synthetic event


@pytest.mark.skipif(FS is None, reason="FAIRSHIP_ROOT not set")
def test_installed_run_simscript_is_pinned_and_still_excludes_the_scoring_points():
    script = Path(FS) / "macro" / "run_simScript.py"
    drv.check_pinned(script)  # fails (on purpose) when FairShip changes the script: re-read the filter, then re-pin
    assert '"UpstreamTaggerScoringPoint")]' in script.read_text(encoding="utf-8")


@pytest.mark.skipif(FS is None or not ART.exists(), reason="FAIRSHIP_ROOT / P2_ARTIFACTS not set")
def test_live_smoke_p2_only_events_are_lost_by_the_standard_output_and_kept_by_the_driver(tmp_path):
    pytest.importorskip("ROOT")
    stage1 = ART / "prod_stage1_8a1cc53/seed_201/out/pythia8_evtgen_Geant4_201_1.0.root"
    if not stage1.exists():
        pytest.skip("stage-1 artifact missing")
    root = Path(__file__).resolve().parents[2]
    cmd = ["pixi", "run", "--manifest-path", f"{FS}/pixi.toml", "python", str(root / "scripts" / "p2_live_smoke.py"),
           "--fairship", FS, "--stage1", str(stage1), "--work", str(tmp_path)]
    done = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": f"{root}/src:{root}", "QT_QPA_PLATFORM": "offscreen"})
    assert done.returncode == 0, done.stderr[-2000:]
    out = json.loads(done.stdout.strip().splitlines()[-1])
    nf, fo = out["nofollow"], out["follow"]
    assert nf["returncode"] == fo["returncode"] == 0 and nf["unfiltered_exists"] and fo["unfiltered_exists"]
    assert nf["lost"] > 0 and nf["lost_events"] and not nf["extra_in_standard"]  # P2-only events vanish from the standard file...
    assert nf["generated"] == nf["kept"] + nf["lost"] and nf["detids"] == [2]  # ...but not from the unfiltered one
    assert fo["lost"] == 0  # the sensitive cavern (--FollowMuon) rescues them
    assert nf["provenance"] == {"RESOLVED": nf["generated"]} and fo["provenance"] == {"RESOLVED": fo["generated"]}
    assert out["p2"]["center"][2] == pytest.approx(out["shield_end"] + 10.0) and out["p2_gap"] == pytest.approx(10.0)
