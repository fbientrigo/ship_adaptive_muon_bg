"""P2 crossing -> stage-1 origin, verified by content, never by index arithmetic alone.

A stage-2 (``--MuonBack``) file keeps the stage-1 history but drops its split flags, so the origin of a P2 crossing is
recovered by joining the injected primary muon of stage 2 with the stage-1 ``PlaneHAPoint`` it was started from
(equal momentum), then reading the split flags from the stage-1 file. ``EventID - 1`` is only a cross-check.

Levels (each one row can belong to exactly one group per level):
``run`` (stage-1 file) > ``progenitor`` (stage-1 entry) > ``split_group`` (parent pi/K whose decays were cloned, or the
unsplit muon itself) > ``trajectory`` (stage-1 muon, one clone) > ``state`` (one P2 crossing).
Counts and sums of the *stored* weight are reported per level; no effective sample size is computed, and the physical
meaning of the stored weight stays OPEN (``WEIGHT_SEMANTICS``).

Crossing policy (``Detector/ScoringPlane.h`` ProcessHits: one point per *entry* into the plane, so a track can produce several):
a crossing is ``eligible`` iff ``pz > 0`` and it is the first such crossing of its track in the entry. Later or backward
crossings are kept as rows (class ``REPEAT_FORWARD`` / ``BACKWARD``), never dropped, never a new source state.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

from ship_muon_bg.entities.identifiers import definition_id
from ship_muon_bg.entities.subject import TagSubject

WEIGHT_SEMANTICS = "OPEN: stored MCTrack weight; its physical normalisation is not settled"
K_INJECTED = 43  # procID of the primaries MuonBackGenerator injects at the scoring plane (observed on 8a1cc53)
TOL = 2e-3  # GeV per component, momentum equality between PlaneHAPoint and the injected primary


def _root():
    import ROOT  # type: ignore

    return ROOT


def read_stage2_tracks(sim_root) -> dict[int, dict[int, dict[str, Any]]]:
    """``{entry: {track_id: {pdg, mother, w, proc, px, py, pz}}}`` of the stored ``MCTrack`` of a stage-2 file."""
    f = _root().TFile.Open(str(sim_root))
    t, out = f.Get("cbmsim"), {}
    for e in range(t.GetEntries()):
        t.GetEntry(e)
        out[e] = {i: {"pdg": k.GetPdgCode(), "mother": k.GetMotherId(), "w": k.GetWeight(), "proc": k.GetProcID(),
                      "px": k.GetPx(), "py": k.GetPy(), "pz": k.GetPz(), "split_set": k.GetSplitSet(), "split_role": k.GetSplitRole()}
                      for i, k in enumerate(t.MCTrack)}
    f.Close()
    return out


def read_stage1(path) -> dict[int, dict[str, Any]]:
    """``{entry: {event_id, tracks: {id: {...split flags}}, ha: [(track_id, px, py, pz)]}}`` of a stage-1 file."""
    f = _root().TFile.Open(str(path))
    t, out = f.Get("cbmsim"), {}
    for e in range(t.GetEntries()):
        t.GetEntry(e)
        out[e] = {"event_id": t.MCEventHeader.GetEventID(),
                  "tracks": {i: {"pdg": k.GetPdgCode(), "mother": k.GetMotherId(), "w": k.GetWeight(), "split_set": k.GetSplitSet(),
                                 "split_role": k.GetSplitRole(), "split_weight": k.GetSplitWeight()} for i, k in enumerate(t.MCTrack)},
                  "ha": [(p.GetTrackID(), p.GetPx(), p.GetPy(), p.GetPz()) for p in t.PlaneHAPoint]}
    f.Close()
    return out


def _primary(tracks: Mapping[int, Mapping[str, Any]], tid: int):
    """Walk mothers from ``tid`` to the injected primary muon; None if the chain is broken or has none."""
    seen = set()
    while tid in tracks and tid not in seen:
        seen.add(tid)
        k = tracks[tid]
        if k["proc"] == K_INJECTED and abs(k["pdg"]) == 13:
            return tid, k
        tid = k["mother"]
    return None


def _split_ancestor(tracks: Mapping[int, Mapping[str, Any]], tid: int):
    seen = set()
    while tid in tracks and tid not in seen:
        seen.add(tid)
        if tracks[tid]["split_set"] >= 0:
            return tid, tracks[tid]
        tid = tracks[tid]["mother"]
    return None


def classify(crossings: Sequence[Mapping[str, Any]]) -> list[str]:
    """Class per crossing (same order): FIRST_FORWARD (eligible), REPEAT_FORWARD, BACKWARD."""
    first, out = {}, []
    for r in sorted(range(len(crossings)), key=lambda i: (crossings[i]["entry"], crossings[i]["track_id"], crossings[i]["t"])):
        c = crossings[r]
        key = (c["entry"], c["track_id"])
        if c["pz"] <= 0:
            out.append((r, "BACKWARD"))
        elif key in first:
            out.append((r, "REPEAT_FORWARD"))
        else:
            first[key] = r
            out.append((r, "FIRST_FORWARD"))
    return [cls for _, cls in sorted(out)]


def reconstruct(crossings: Sequence[Mapping[str, Any]], stage2: Mapping[int, Mapping[int, Mapping[str, Any]]],
                stage1: Mapping[int, Mapping[str, Any]], *, stage1_sha: str, stage2_sha: str, first_event: int = 0) -> list[dict[str, Any]]:
    """One row per crossing with its recovered origin. ``status``: RESOLVED, UNRECOVERABLE (no valid track or primary)
    or AMBIGUOUS (the primary matches zero or several stage-1 points); the fields of a non-RESOLVED row stay None.
    ``event_id_rule_agrees`` tests ``stage-1 entry == first_event + EventID - 1`` (``-i`` of the stage-2 run); it is a check,
    not the join: with ``-i 8`` the stage-2 ``EventID`` restarts at 1."""
    rows = []
    for c, cls in zip(crossings, classify(crossings)):
        row = {**c, "crossing_class": cls, "eligible": cls == "FIRST_FORWARD", "status": "RESOLVED", "reason": None,
               "stage1_sha": stage1_sha, "stage2_sha": stage2_sha, "stage1_entry": None, "stage1_event_id": None,
               "stage1_track": None, "parent_track": None, "parent_pdg": None, "split_set": None, "split_role": None,
               "split_weight": None, "clone_weight": None, "event_id_rule_agrees": None, "weight_agrees": None}
        rows.append(row)
        tracks = stage2.get(c["entry"], {})
        prim = _primary(tracks, c["track_id"]) if c["track_id"] >= 0 else None
        if prim is None:
            row.update(status="UNRECOVERABLE", reason="invalid track_id" if c["track_id"] < 0 else "no injected primary in the mother chain")
            continue
        _, k = prim
        hits = [(e, tid) for e, ev in stage1.items() for tid, px, py, pz in ev["ha"]
                if max(abs(px - k["px"]), abs(py - k["py"]), abs(pz - k["pz"])) < TOL]
        if len(hits) != 1:
            row.update(status="AMBIGUOUS", reason=f"{len(hits)} stage-1 PlaneHAPoint match the injected primary: {hits[:4]}")
            continue
        e1, t1 = hits[0]
        ev, tr1 = stage1[e1], stage1[e1]["tracks"]
        anc = _split_ancestor(tr1, t1)
        parent = tr1[tr1[t1]["mother"]] if tr1[t1]["mother"] in tr1 else None
        row.update(stage1_entry=e1, stage1_event_id=ev["event_id"], stage1_track=t1, clone_weight=tr1[t1]["w"],
                   parent_track=anc[0] if anc else tr1[t1]["mother"], parent_pdg=(anc[1] if anc else parent or {}).get("pdg"),
                   split_set=anc[1]["split_set"] if anc else None, split_role=anc[1]["split_role"] if anc else None,
                   split_weight=anc[1]["split_weight"] if anc else None,
                   event_id_rule_agrees=(first_event + c["mc_event_id"] - 1 == e1),
                   weight_agrees=abs(tr1[t1]["w"] - (c["weight"] if c["weight"] is not None else -1)) < 1e-6)
    return rows


def group_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict]:
    """Counts and sums of stored weight per level over ELIGIBLE, RESOLVED rows. No ESS: see ``WEIGHT_SEMANTICS``."""
    levels = {"progenitor": lambda r: (r["stage1_sha"], r["stage1_entry"]),
              "split_group": lambda r: (r["stage1_sha"], r["stage1_entry"], r["split_set"] if r["split_set"] is not None else ("unsplit", r["stage1_track"])),
              "trajectory": lambda r: (r["stage1_sha"], r["stage1_entry"], r["stage1_track"])}
    acc = {name: defaultdict(lambda: [0, 0.0]) for name in levels}
    n = 0
    for r in rows:
        if r["status"] != "RESOLVED" or not r["eligible"]:
            continue
        n += 1
        for name, key in levels.items():
            a = acc[name][key(r)]
            a[0] += 1
            a[1] += r["weight"] or 0.0
    return {"n_states": n, "weight_semantics": WEIGHT_SEMANTICS,
            **{name: {"n_groups": len(g), "groups": {str(k): {"n_states": v[0], "sum_stored_weight": v[1]} for k, v in g.items()}} for name, g in acc.items()}}


def state_definition(geometry_manifest: Mapping[str, Any]) -> str:
    """Content-addressed id of what a P2 state's coordinates mean (plane, geometry file, FairShip, units)."""
    p2 = geometry_manifest["nodes"]["UpstreamTaggerScoringPlane2_2"]
    return definition_id("p2_crossing_state_v0", {"fairship": geometry_manifest["fairship"], "geofile_sha256": geometry_manifest["sha256"],
                                                  "plane_center": p2["center"], "plane_half": p2["half"], "units": "cm, GeV/c, ns",
                                                  "frame": "FairShip global cave", "point": "first step in plane (entry face)"})


def tag_subjects(rows: Iterable[Mapping[str, Any]], state_definition_id: str) -> list[TagSubject]:
    """``TagSubject`` for each eligible, RESOLVED row. ``subject_type`` is a candidate, not THE SourceState (contract OPEN-01)."""
    return [TagSubject(f"p2:{r['stage2_sha'][:12]}:{r['entry']}:{r['track_id']}:{r['crossing_index']}", "p2_crossing_state", state_definition_id)
            for r in rows if r["status"] == "RESOLVED" and r["eligible"]]
