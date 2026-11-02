"""Post-shield crossings from FairShip 26.10 ``--MuonBack`` output, and the geometry that places them.

Facts only: no weight policy, no utility, no independence claim. Evidence for every constant is in
``notes/p2_extraction_v0.md``. ROOT is imported lazily; run under the FairShip pixi environment.

* P2 is ``UpstreamTaggerScoringPlane2``: point branch ``UpstreamTaggerScoringPoint``, ``detID == 2`` (the node copy
  number, ``Detector/ScoringPlane.h``), muons only, placed ``ShieldGap`` after ``muShield.z + muShield.length``.
* The file stores one point per entering track, at the end of its first step in the plane.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

P2_BRANCH = "UpstreamTaggerScoringPoint"
P2_DETID = 2
P2_NODE = "UpstreamTaggerScoringPlane2"


def _root():
    import ROOT  # type: ignore

    return ROOT


def read_crossings(sim_root: Path | str, branch: str = P2_BRANCH, det_id: int | None = P2_DETID) -> list[dict[str, Any]]:
    """One row per stored point of ``branch`` (restricted to ``det_id`` unless None), in file order.

    ``entry`` is the position in this file and ``mc_event_id`` the stored ``MCEventHeader`` id: neither is a
    stable source id after ``--MuonBack`` event filtering. ``crossing_index`` orders repeated crossings of one
    track in one entry by time, so ``(entry, track_id, crossing_index)`` is unique. ``weight`` is the stored
    ``MCTrack`` weight of the crossing track (None when the track was not kept, ``track_id < 0``); split fields are
    the stored ones, which ``--MuonBack`` output does not preserve (see the note).
    """
    f = _root().TFile.Open(str(sim_root))  # keep the TFile alive: a chained Open().Get() segfaults
    tree = f.Get("cbmsim")
    rows: list[dict[str, Any]] = []
    for entry in range(tree.GetEntries()):
        tree.GetEntry(entry)
        tracks, seen = tree.MCTrack, {}
        for i, p in enumerate(getattr(tree, branch)):
            if det_id is not None and p.GetDetectorID() != det_id:
                continue
            tid = p.GetTrackID()
            k = tracks[tid] if 0 <= tid < tracks.size() else None
            seen[tid] = seen.get(tid, -1) + 1
            px, py, pz = p.GetPx(), p.GetPy(), p.GetPz()
            rows.append({
                "entry": entry, "mc_event_id": tree.MCEventHeader.GetEventID(), "point_index": i, "crossing_index": seen[tid],
                "det_id": p.GetDetectorID(), "track_id": tid, "pdg": p.PdgCode(), "x": p.GetX(), "y": p.GetY(), "z": p.GetZ(),
                "px": px, "py": py, "pz": pz, "p": math.sqrt(px * px + py * py + pz * pz), "t": p.GetTime(),
                "weight": None if k is None else k.GetWeight(), "mother_id": None if k is None else k.GetMotherId(),
                "split_set": None if k is None else k.GetSplitSet(),
            })
    f.Close()
    return rows


def pair_crossings(a: Iterable[Mapping[str, Any]], b: Iterable[Mapping[str, Any]]):
    """Match crossings of two planes by ``(entry, track_id, crossing_index)``: ``(pairs, only_a, only_b)``."""
    key = lambda r: (r["entry"], r["track_id"], r["crossing_index"])  # noqa: E731
    ka, kb = {key(r): r for r in a}, {key(r): r for r in b}
    return ([(ka[k], kb[k]) for k in ka if k in kb], [ka[k] for k in ka if k not in kb], [kb[k] for k in kb if k not in ka])


def geometry_manifest(geofile: Path | str, extra_nodes: Iterable[str] = ()) -> dict[str, Any]:
    """Placement read from the geometry FairShip wrote (``FAIRGeom`` + ``ShipGeo``), never typed in.

    Nodes: both UBT scoring planes, every ``Magn*`` shield block and ``extra_nodes`` (e.g. a local test plane),
    searched in the cave and in ``MuonShieldArea``. Positions are global (parent matrices composed), in cm.
    """
    ROOT = _root()
    path = Path(geofile)
    f = ROOT.TFile.Open(str(path))
    geom, ship_geo = f.Get("FAIRGeom"), json.loads(str(f.Get("ShipGeo")))
    wanted = ("UpstreamTaggerScoringPlane", "Magn", *extra_nodes)  # prefixes; "SHiPMagnet" is not a shield block
    nodes: dict[str, Any] = {}

    def visit(node, parent, depth):
        m = ROOT.TGeoHMatrix(parent)
        m.Multiply(node.GetMatrix())
        vol, name = node.GetVolume(), node.GetName()
        if name.startswith(wanted):
            sh, t = vol.GetShape(), m.GetTranslation()
            nodes[name] = {"center": [t[0], t[1], t[2]], "half": [sh.GetDX(), sh.GetDY(), sh.GetDZ()],
                           "material": vol.GetMaterial().GetName(), "shape": sh.ClassName()}
        if depth == 0 or name.startswith("MuonShieldArea"):
            for j in range(vol.GetNdaughters()):
                visit(vol.GetNode(j), m, depth + 1)

    visit(geom.GetTopNode(), ROOT.TGeoIdentity(), 0)
    f.Close()
    magnets = {n: v for n, v in nodes.items() if n.startswith("Magn")}
    shield_end = max(v["center"][2] + v["half"][2] for v in magnets.values())
    sp = ship_geo["UpstreamTaggerScoringPlane"]
    return {
        "geofile": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "fairship": ship_geo.get("FairShip"), "fairship_library": ship_geo.get("FairShipLibrary"), "shield_name": ship_geo["shieldName"],
        "muShield": {k: ship_geo["muShield"][k] for k in ("z", "length")},
        "scoring_planes_config": {"thickness": sp["Z"], "P1": sp["DecayVessel"], "P2": sp["MuonShield"]},
        "shield_end_from_nodes": shield_end, "nodes": nodes,
        "p2_center_minus_shield_end": nodes[f"{P2_NODE}_{P2_DETID}"]["center"][2] - shield_end,
    }


def compare_with_standard(unfiltered: Iterable[Mapping[str, Any]], standard: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Crossings of the unfiltered file against the standard (``--MuonBack``-filtered) file of the same run.

    Key: ``(mc_event_id, track_id, crossing_index)``; ``mc_event_id`` survives the filter, ``entry`` does not.
    ``extra_in_standard`` must be empty (the filter only removes events); ``lost`` are the crossings it removed and
    ``lost_events`` their event ids (events whose P2 crossings were all removed with them).
    """
    key = lambda r: (r["mc_event_id"], r["track_id"], r["crossing_index"])  # noqa: E731
    raw, std = {key(r): r for r in unfiltered}, {key(r) for r in standard}
    lost = [raw[k] for k in raw if k not in std]
    return {"generated": len(raw), "kept": len(raw) - len(lost), "lost": lost, "extra_in_standard": sorted(std - raw.keys()),
            "lost_events": sorted({r["mc_event_id"] for r in lost} - {r["mc_event_id"] for r in raw.values() if key(r) in std})}
