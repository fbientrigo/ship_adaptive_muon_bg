"""P2 extraction: pure pairing semantics, and ``local_env`` characterization of FairShip 26.10 (8a1cc53) output.

The local tests are falsifiable statements about FairShip behavior: if a FairShip update changes the plane
position, the detID, the event numbering or the event filter, they fail on purpose (compatibility alarm).
Set P2_ARTIFACTS and run them under the FairShip pixi env: ``PYTHONPATH=src pixi run --manifest-path <FairShip>/pixi.toml python -m pytest ...``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ship_muon_bg.adapters.fairship_p2 import geometry_manifest, pair_crossings, read_crossings


def row(entry, tid, k=0, **kw):
    return {"entry": entry, "track_id": tid, "crossing_index": k, **kw}


def test_pairing_keeps_repeated_crossings_and_reports_unpaired():
    a = [row(0, 5), row(0, 5, 1), row(1, 7)]
    b = [row(0, 5), row(0, 5, 1), row(2, 9)]
    pairs, only_a, only_b = pair_crossings(a, b)
    assert len(pairs) == 2 and only_a == [a[2]] and only_b == [b[2]]


ART = Path(os.environ.get("P2_ARTIFACTS", "/nonexistent"))  # set P2_ARTIFACTS to the folder holding the stage1/stage2 runs
SEEDS = (201, 202, 203)


def need(path: Path) -> Path:
    pytest.importorskip("ROOT")
    if not path.exists():
        pytest.skip(f"artifact missing: {path}")
    return path


@pytest.mark.local_env
def test_p2_geometry_is_read_from_the_produced_geofile():
    m = geometry_manifest(need(ART / "stage2_prod_8a1cc53/seed_201/geo_s201.root"), extra_nodes=("exitHadronAbsorberPlaneMS",))
    assert m["fairship_library"].startswith("26.10 (8a1cc53") and m["shield_name"] == "TRY_2026"
    p2 = m["nodes"]["UpstreamTaggerScoringPlane2_2"]
    assert p2["center"] == pytest.approx([0.0, 91.8, 3228.0]) and p2["half"] == pytest.approx([456.6, 428.2, 0.05])
    assert m["shield_end_from_nodes"] == pytest.approx(m["muShield"]["z"] + m["muShield"]["length"]) == pytest.approx(3218.0)
    assert m["p2_center_minus_shield_end"] == pytest.approx(10.0)
    ms = m["nodes"]["exitHadronAbsorberPlaneMS_box_1"]
    assert ms["center"][2] == pytest.approx(3200.0) and ms["half"][:2] == pytest.approx([355.9, 169.9])
    assert ms["center"][2] < m["shield_end_from_nodes"]  # the local test plane sits inside the z-span of the last magnet


@pytest.mark.local_env
def test_p2_contains_every_planems_crossing_and_the_one_extra_is_outside_planems_extent():
    n_p2 = n_ms = 0
    extra = []
    for s in SEEDS:
        f = need(ART / f"stage2_prod_8a1cc53/seed_{s}/sim_s{s}.root")
        p2, ms = read_crossings(f), read_crossings(f, "PlaneMSPoint", None)
        pairs, only_p2, only_ms = pair_crossings(p2, ms)
        assert not only_ms and all(abs(r["pdg"]) == 13 and r["weight"] is not None for r in p2)
        assert len({(r["entry"], r["track_id"], r["crossing_index"]) for r in p2}) == len(p2)
        for a, b in pairs:
            assert abs(a["p"] - b["p"]) < 1e-3 and a["z"] - b["z"] == pytest.approx(28.05, abs=0.01)  # 3227.95 - 3199.90
        n_p2, n_ms, extra = n_p2 + len(p2), n_ms + len(ms), extra + only_p2
    assert (n_p2, n_ms) == (21, 20)
    assert len(extra) == 1 and abs(extra[0]["y"]) > 169.9  # outside PlaneMS lateral half-size


@pytest.mark.local_env
def test_muonback_filter_drops_no_p2_event_in_unfiltered_runs_and_ids_are_entry_plus_one():
    scoring = ("CaloScoringPlanePoint", "UpstreamTaggerScoringPoint")
    ROOT = pytest.importorskip("ROOT")

    seen = 0
    for s in SEEDS:
        f = ROOT.TFile.Open(str(need(ART / f"p2_nofilter_8a1cc53/seed_{s}/sim_s{s}.root")))
        t = f.Get("cbmsim")
        others = [b.GetName() for b in t.GetListOfBranches() if "Point" in b.GetName() and b.GetName() not in scoring]
        for e in range(t.GetEntries()):
            t.GetEntry(e)
            assert t.MCEventHeader.GetEventID() == e + 1
            if any(p.GetDetectorID() == 2 for p in t.UpstreamTaggerScoringPoint):
                seen += 1
                assert any(getattr(t, b).size() for b in others), f"seed {s} entry {e}: P2 crossing would be filtered out"
        f.Close()
    assert seen == 7  # P2 events in the 114 raw entries; small sample, see the note


@pytest.mark.local_env
def test_stage2_event_id_minus_one_is_the_stage1_entry_by_primary_momentum():
    ROOT = pytest.importorskip("ROOT")

    def mom(o):
        return (o.GetPx(), o.GetPy(), o.GetPz())

    checked = 0
    for s in SEEDS:
        f2 = ROOT.TFile.Open(str(need(ART / f"stage2_prod_8a1cc53/seed_{s}/sim_s{s}.root")))
        f1 = ROOT.TFile.Open(str(need(ART / f"prod_stage1_8a1cc53/seed_{s}/out/pythia8_evtgen_Geant4_{s}_1.0.root")))
        t2, t1 = f2.Get("cbmsim"), f1.Get("cbmsim")
        for e in range(t2.GetEntries()):
            t2.GetEntry(e)
            t1.GetEntry(t2.MCEventHeader.GetEventID() - 1)
            ha = [mom(p) for p in t1.PlaneHAPoint]
            for k in t2.MCTrack:
                if k.GetProcID() == 43 and abs(k.GetPdgCode()) == 13:  # primaries injected by MuonBackGenerator
                    assert any(max(abs(a - b) for a, b in zip(mom(k), h)) < 2e-3 for h in ha)
                    checked += 1
    assert checked == 21  # 1+10+2+4+1 + 1+1 + 1 primary muons in the 8 stage-2 events
