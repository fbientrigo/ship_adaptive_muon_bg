"""P2 crossing -> stage-1 origin. Synthetic tests need neither ROOT nor FairShip; ``local_env`` tests need P2_ARTIFACTS + ROOT."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from ship_muon_bg.adapters.fairship_p2_source import (K_INJECTED, classify, group_summary, reconstruct, state_definition, tag_subjects)

S1, S2 = "s1" * 32, "s2" * 32


def stage1_entry(n_clones=10, event_id=5110, p0=1.0, split=True, w=0.1):
    """Progenitor: proton -> pion clones (split set 0, role 2) -> one muon per clone; PlaneHAPoint at each muon."""
    tr = {0: dict(pdg=2212, mother=-1, w=1.0, split_set=-1, split_role=0, split_weight=0.0)}
    ha = []
    for i in range(n_clones):
        tr[1 + i] = dict(pdg=211, mother=0, w=w, split_set=0 if split else -1, split_role=2 if split else 0, split_weight=1.0 if split else 0.0)
        tr[1 + n_clones + i] = dict(pdg=-13, mother=1 + i, w=w, split_set=-1, split_role=0, split_weight=0.0)
        ha.append((1 + n_clones + i, p0 + 0.1 * i, 0.0, 10.0))
    return {"event_id": event_id, "tracks": tr, "ha": ha}


def stage2_entry(s1, w=0.1):
    """Stage 2 replays the history without split flags and injects one primary (procID 43) per HA point."""
    tr = {i: dict(pdg=t["pdg"], mother=t["mother"], w=t["w"], proc=0, px=0, py=0, pz=0) for i, t in s1["tracks"].items()}
    n = len(tr)
    for j, (tid, px, py, pz) in enumerate(s1["ha"]):
        tr[n + j] = dict(pdg=-13, mother=tid, w=w, proc=K_INJECTED, px=px, py=py, pz=pz)
    return tr, n


def crossing(entry, tid, k=0, pz=10.0, t=1.0, w=0.1, event_id=1):
    return dict(entry=entry, mc_event_id=event_id, track_id=tid, crossing_index=k, pz=pz, t=t, weight=w)


def ten_clones():
    s1 = stage1_entry()
    tr, n = stage2_entry(s1)
    cr = [crossing(0, n + j) for j in range(10)]
    return s1, tr, cr


def test_ten_clones_of_one_progenitor_are_one_group_not_ten_sources():
    s1, tr, cr = ten_clones()
    rows = reconstruct(cr, {0: tr}, {3: s1}, stage1_sha=S1, stage2_sha=S2, first_event=3)
    assert all(r["status"] == "RESOLVED" and r["eligible"] and r["split_set"] == 0 and r["split_role"] == 2 for r in rows)
    assert {r["stage1_entry"] for r in rows} == {3} and {r["stage1_event_id"] for r in rows} == {5110}
    g = group_summary(rows)
    assert (g["n_states"], g["progenitor"]["n_groups"], g["split_group"]["n_groups"], g["trajectory"]["n_groups"]) == (10, 1, 1, 10)
    (grp,) = g["split_group"]["groups"].values()
    assert grp["n_states"] == 10 and grp["sum_stored_weight"] == pytest.approx(1.0)  # one parent decay, not 10 sources
    assert "ess" not in str(g).lower() and g["weight_semantics"].startswith("OPEN")


def test_entry_is_found_by_content_and_event_id_rule_is_only_a_check_under_nonzero_first_event():
    s1, tr, cr = ten_clones()
    for c in cr:
        c["mc_event_id"] = 1  # stage 2 run with -i 3: EventID restarts at 1
    assert all(r["event_id_rule_agrees"] for r in reconstruct(cr, {0: tr}, {3: s1}, stage1_sha=S1, stage2_sha=S2, first_event=3))
    rows = reconstruct(cr, {0: tr}, {3: s1}, stage1_sha=S1, stage2_sha=S2, first_event=0)
    assert all(r["status"] == "RESOLVED" and r["stage1_entry"] == 3 and r["event_id_rule_agrees"] is False for r in rows)


def test_unsplit_muon_is_its_own_group():
    s1 = stage1_entry(n_clones=1, split=False, w=1.0)
    tr, n = stage2_entry(s1, w=1.0)
    (r,) = reconstruct([crossing(0, n, w=1.0)], {0: tr}, {0: s1}, stage1_sha=S1, stage2_sha=S2)
    assert r["split_set"] is None and r["parent_pdg"] == 211 and r["weight_agrees"]
    assert group_summary([r])["split_group"]["n_groups"] == 1


def test_repeated_and_backward_crossings_are_kept_but_only_the_first_forward_is_eligible():
    cr = [crossing(0, 5, 0, pz=-1.0, t=1.0), crossing(0, 5, 1, pz=2.0, t=2.0), crossing(0, 5, 2, pz=3.0, t=3.0), crossing(0, 6, 0, pz=1.0)]
    assert classify(cr) == ["BACKWARD", "FIRST_FORWARD", "REPEAT_FORWARD", "FIRST_FORWARD"]
    assert classify(cr[::-1]) == classify(cr)[::-1]  # decided by time, not by list order


def test_invalid_or_unrecoverable_track_ids_are_not_given_an_origin():
    s1, tr, _ = ten_clones()
    bad = [crossing(0, -2, w=None), crossing(0, 9999), crossing(0, 0)]  # not kept; unknown; a track with no injected primary
    rows = reconstruct(bad, {0: tr}, {0: s1}, stage1_sha=S1, stage2_sha=S2)
    assert [r["status"] for r in rows] == ["UNRECOVERABLE"] * 3
    assert all(r["stage1_entry"] is None and r["split_set"] is None for r in rows)
    assert tag_subjects(rows, "d") == [] and group_summary(rows)["n_states"] == 0


def test_ambiguous_or_missing_stage1_match_is_reported_not_guessed():
    s1, tr, cr = ten_clones()
    twin = {**s1, "event_id": 777}  # same momenta in a second stage-1 entry
    rows = reconstruct(cr[:1], {0: tr}, {0: s1, 1: twin}, stage1_sha=S1, stage2_sha=S2)
    assert rows[0]["status"] == "AMBIGUOUS" and rows[0]["stage1_entry"] is None and "2 stage-1" in rows[0]["reason"]
    rows = reconstruct(cr[:1], {0: tr}, {0: {**s1, "ha": []}}, stage1_sha=S1, stage2_sha=S2)
    assert rows[0]["status"] == "AMBIGUOUS" and "0 stage-1" in rows[0]["reason"]


def test_weight_disagreement_between_stages_is_flagged():
    s1, tr, cr = ten_clones()
    cr[0]["weight"] = 0.5
    rows = reconstruct(cr, {0: tr}, {0: s1}, stage1_sha=S1, stage2_sha=S2)
    assert rows[0]["weight_agrees"] is False and all(r["weight_agrees"] for r in rows[1:])


def test_only_eligible_resolved_rows_become_subjects_with_a_content_addressed_definition():
    s1, tr, cr = ten_clones()
    cr.append(crossing(0, cr[0]["track_id"], 1, t=9.0))  # a second entry of the first track
    rows = reconstruct(cr, {0: tr}, {0: s1}, stage1_sha=S1, stage2_sha=S2)
    d = state_definition({"fairship": "x", "sha256": "y", "nodes": {"UpstreamTaggerScoringPlane2_2": {"center": [0, 0, 1], "half": [1, 1, 0.05]}}})
    subs = tag_subjects(rows, d)
    assert len(rows) == 11 and len(subs) == 10 and len({s.subject_id for s in subs}) == 10
    assert all(s.subject_type == "p2_crossing_state" and s.state_definition_id == d for s in subs) and "@sha256:" in d


# --- integration with the real 8a1cc53 artifacts ------------------------------------------------------------------------

ART = Path(os.environ.get("P2_ARTIFACTS", "/nonexistent"))


def need(path: Path) -> Path:
    pytest.importorskip("ROOT")
    if not path.exists():
        pytest.skip(f"artifact missing: {path}")
    return path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_rows(stage2_dir: str, s1_seed: int, sim: str, first_event: int = 0):
    from ship_muon_bg.adapters.fairship_p2 import read_crossings
    from ship_muon_bg.adapters.fairship_p2_source import read_stage1, read_stage2_tracks

    sim_f = need(ART / stage2_dir / sim)
    s1_f = need(ART / f"prod_stage1_8a1cc53/seed_{s1_seed}/out/pythia8_evtgen_Geant4_{s1_seed}_1.0.root")
    cr = read_crossings(sim_f)
    return reconstruct(cr, read_stage2_tracks(sim_f), read_stage1(s1_f), stage1_sha=sha(s1_f), stage2_sha=sha(sim_f), first_event=first_event)


@pytest.mark.local_env
def test_real_21_crossings_resolve_to_8_progenitors_6_split_groups_and_2_unsplit_muons():
    rows = []
    for s in (201, 202, 203):
        rows += run_rows(f"stage2_prod_8a1cc53/seed_{s}", s, f"sim_s{s}.root")
    assert len(rows) == 21 and all(r["status"] == "RESOLVED" and r["eligible"] and r["weight_agrees"] and r["event_id_rule_agrees"] for r in rows)
    g = group_summary(rows)
    assert (g["progenitor"]["n_groups"], g["split_group"]["n_groups"], g["trajectory"]["n_groups"]) == (8, 8, 21)
    split = [v for k, v in g["split_group"]["groups"].items() if "unsplit" not in k]
    assert len(split) == 6 and sorted(round(v["sum_stored_weight"], 3) for v in split) == [0.1, 0.1, 0.1, 0.2, 0.4, 1.0]
    assert sum(1 for r in rows if r["split_set"] is None) == 2


@pytest.mark.local_env
def test_event_id_minus_one_fails_under_nonzero_first_event_but_content_join_still_resolves():
    for name, entries in (("p2_i8_stock", [8, 9, 13, 14]), ("p2_i8_nofilter", [8, 9, 13, 14])):
        rows = run_rows(name, 201, "sim_s201.root", first_event=8)
        assert sorted({r["stage1_entry"] for r in rows}) == entries and all(r["status"] == "RESOLVED" and r["event_id_rule_agrees"] for r in rows)
        assert not any(r["event_id_rule_agrees"] for r in run_rows(name, 201, "sim_s201.root", first_event=0))


@pytest.mark.local_env
def test_stage2_output_has_lost_the_split_flags_while_stage1_has_them():
    from ship_muon_bg.adapters.fairship_p2_source import read_stage1, read_stage2_tracks

    s1 = read_stage1(need(ART / "prod_stage1_8a1cc53/seed_201/out/pythia8_evtgen_Geant4_201_1.0.root"))
    s2 = read_stage2_tracks(need(ART / "stage2_prod_8a1cc53/seed_201/sim_s201.root"))
    assert sum(t["split_set"] >= 0 for t in s1[9]["tracks"].values()) == 10
    hist = [s2[1][i] for i in range(6, 16)]  # the same 10 pion clones, replayed as history
    assert all(t["pdg"] == 211 and t["mother"] == 5 for t in hist)
    assert all(t["split_set"] == -1 and t["split_role"] == 0 for t in s2[1].values())  # alarm: fails the day FairShip keeps the flags
