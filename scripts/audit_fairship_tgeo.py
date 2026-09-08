"""Audit fixed injected states with FairShip TGeo and observed veto points.

The script reads existing ROOT outputs; it never redraws or changes candidates.
ROOT/FairShip are intentionally optional at import time so the CLI remains
inspectable in environments without the FairShip runtime.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from array import array
from pathlib import Path
from typing import Any


DECISION = "SBT_USABLE_FOR_FIRST_EMPIRICAL_ENDPOINT"
SBT_SOURCE = "FairShip/veto/veto.cxx:GeoSideObj/GeoCornerLiSc* sens=true"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def configure_fairship(fairship: Path) -> None:
    """Load the same dictionaries used by FairShip's run scripts."""
    os.environ.update(
        FAIRSHIP=str(fairship), FAIRSHIP_ROOT=str(fairship), VMCWORKDIR=str(fairship),
        GEOMPATH=str(fairship / "geometry"), CONFIG_DIR=str(fairship / "gconfig"),
        GENFIT_ROOT=str(fairship / ".pixi" / "envs" / "default"),
    )
    sys.path.insert(0, str(fairship / "build" / "lib"))
    sys.path.insert(0, str(fairship / "python"))
    import shipRoot_conf  # type: ignore

    shipRoot_conf.configure()


def candidate_state(root_dir: Path) -> dict[str, Any]:
    payload = load_json(root_dir / "candidate.json")
    if len(payload.get("records", [])) != 1 or len(payload.get("transformed_states", [])) != 1:
        raise ValueError(f"expected one candidate in {root_dir / 'candidate.json'}")
    state = dict(payload["transformed_states"][0])
    state["candidate_id"] = payload["records"][0]["candidate_id"]
    state["source_provenance"] = payload["records"][0].get("source_provenance", {})
    state["physical_source_weight"] = payload["records"][0].get("physical_source_weight")
    state["coordinate_transform_id"] = payload["records"][0].get("coordinate_transform_id")
    state["coordinate_transform_status"] = payload["records"][0].get("coordinate_transform_status", "PROVISIONAL")
    return state


def _point(view: Any) -> tuple[float, float, float]:
    return float(view[0]), float(view[1]), float(view[2])


def trajectory(geometry: Any, state: dict[str, Any], *, max_boundaries: int = 2000) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Walk TGeo boundaries from the recorded injected state."""
    momentum = math.sqrt(sum(float(state[k]) ** 2 for k in ("px", "py", "pz")))
    if not math.isfinite(momentum) or momentum <= 0:
        raise ValueError("candidate momentum must be positive and finite")
    direction = array("d", [float(state[k]) / momentum for k in ("px", "py", "pz")])
    # Candidate JSON and TTree transport declare positions in metres; TGeo
    # geometry coordinates are centimetres (the connector applies this same
    # conversion before FairShip receives the row).
    start = array("d", [100.0 * float(state[k]) for k in ("x", "y", "z")])
    initial = geometry.InitTrack(start, direction)
    if not initial:
        raise RuntimeError("TGeo could not locate the injection state")
    rows: list[dict[str, Any]] = []
    sensitive: list[str] = []
    for boundary_index in range(max_boundaries):
        node = geometry.GetCurrentNode()
        point = geometry.GetCurrentPoint()
        if not node:
            break
        volume = node.GetVolume().GetName()
        material = node.GetVolume().GetMaterial().GetName() if node.GetVolume().GetMaterial() else None
        path = geometry.GetPath()
        x, y, z = _point(point)
        rows.append({"boundary_index": boundary_index, "volume": volume, "path": path,
                     "material": material, "x_cm": x, "y_cm": y, "z_cm": z,
                     "step_cm": float(geometry.GetStep())})
        if volume.startswith("LiSc") and volume not in sensitive:
            sensitive.append(volume)
        if not geometry.FindNextBoundaryAndStep():
            break
    else:
        raise RuntimeError(f"TGeo boundary walk exceeded {max_boundaries} steps")
    if not rows:
        raise RuntimeError("TGeo returned no trajectory rows")
    return {"initial_volume": rows[0]["volume"], "initial_path": rows[0]["path"],
            "initial_material": rows[0]["material"], "sensitive_volumes": sensitive,
            "boundary_count": len(rows) - 1}, rows


def _entries(collection: Any) -> list[Any]:
    return [collection[i] for i in range(len(collection))]


def root_observations(root_path: Path, candidate_id: str, *, sbt_threshold_gev: float = 3.0) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import ROOT  # type: ignore

    root_file = ROOT.TFile.Open(str(root_path), "READ")
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"unreadable ROOT output: {root_path}")
    tree = root_file.Get("cbmsim")
    if not tree or tree.GetEntries() < 1 or tree.GetEntry(0) <= 0:
        raise RuntimeError(f"missing readable cbmsim event: {root_path}")
    observations: list[dict[str, Any]] = []
    qualifying: list[dict[str, Any]] = []
    for index, hit in enumerate(_entries(tree.vetoPoint)):
        px, py, pz = float(hit.GetPx()), float(hit.GetPy()), float(hit.GetPz())
        momentum = math.sqrt(px * px + py * py + pz * pz)
        detector_id, pdg_id, track_id = int(hit.GetDetectorID()), int(hit.PdgCode()), int(hit.GetTrackID())
        is_sbt = 1000 < detector_id < 999999 and abs(pdg_id) == 13 and sbt_threshold_gev < momentum
        row = {"candidate_id": candidate_id, "event": 0, "hit_index": index,
               "detector_id": detector_id, "pdg_id": pdg_id, "track_id": track_id,
               "momentum_gev": momentum, "x_cm": float(hit.GetX()), "y_cm": float(hit.GetY()),
               "z_cm": float(hit.GetZ()), "energy_loss": float(hit.GetEnergyLoss()),
               "time_ns": float(hit.GetTime()), "sbt_qualifying": is_sbt}
        observations.append(row)
        if is_sbt:
            qualifying.append(row)
    summary = {
        "veto_point_count": len(observations),
        "sbt_qualifying_hit_count": len(qualifying),
        "sbt_qualifying_track_ids": sorted({row["track_id"] for row in qualifying}),
        "upstream_tagger_point_count": len(_entries(tree.UpstreamTaggerPoint)),
        "strawtubes_point_count": len(_entries(tree.strawtubesPoint)),
        "mc_track_count": len(_entries(tree.MCTrack)),
        "current_preprocessing_decision": "SBT_SELECTED" if qualifying else "NOT_SBT_SELECTED",
        "selection_rule": "1000 < detector_id < 999999 AND abs(pdg_id) == 13 AND momentum_gev > 3.0",
    }
    root_file.Close()
    return summary, observations


def run_config(commands_path: Path) -> dict[str, Any]:
    text = commands_path.read_text(encoding="utf-8")
    values: dict[str, str | bool] = {}
    for option in ("shieldName", "strawDesign"):
        match = re.search(rf"--{option}\s+(\S+)", text)
        values[option] = match.group(1) if match else "UNDECLARED"
    for option in ("noSND", "reproducible", "validation"):
        values[option] = f"--{option}" in text
    return values


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row}) or ["candidate_id"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def audit(*, pilot_dir: Path, controls_dir: Path, fairship: Path, output: Path) -> dict[str, Any]:
    configure_fairship(fairship)
    import ROOT  # type: ignore

    pilot_candidates = list(csv.DictReader((pilot_dir / "candidates.csv").open(newline="", encoding="utf-8")))
    if len(pilot_candidates) != 12 or {int(row["pdg_id"]) for row in pilot_candidates} != {13, -13}:
        raise ValueError("fixed genuine NF cohort must contain exactly 12 rows and both charges")
    entries: list[tuple[str, Path, str]] = []
    source_metadata: dict[str, dict[str, Any]] = {}
    for row in pilot_candidates:
        candidate_id = row["candidate_id"]
        source_metadata[candidate_id] = {
            key: row[key] for key in (
                "tilt_id", "delta", "alpha", "nf_run_id", "checkpoint_path", "checkpoint_sha256",
                "model_config_hash", "training_dataset_hash", "utility_mode", "sampling_seed",
                "sample_index", "proposal_log_prob", "coordinate_transform_id", "coordinate_transform_status",
            ) if key in row
        }
        root_dir = pilot_dir / "executions" / candidate_id / "ttree"
        entries.append((candidate_id, root_dir, "genuine_nf_pilot"))
    for root_dir in sorted(controls_dir.glob("p3_row_*")):
        if (root_dir / "candidate.json").is_file():
            state = candidate_state(root_dir)
            entries.append((state["candidate_id"], root_dir, "empirical_sbt_positive_control"))
    if not any(kind == "empirical_sbt_positive_control" for _, _, kind in entries):
        raise ValueError("at least one empirical control is required")

    anchor = next(root_dir for _, root_dir, kind in entries if kind == "empirical_sbt_positive_control")
    geometry_path = anchor / "geo_connector.root"
    geometry_file = ROOT.TFile.Open(str(geometry_path), "READ")
    geometry = geometry_file.Get("FAIRGeom")
    if not geometry:
        raise RuntimeError(f"missing FAIRGeom in {geometry_path}")
    anchor_environment = load_json(anchor / "environment.json")
    anchor_config = run_config(anchor / "commands.txt")
    anchor_commit = anchor_environment.get("fairship_commit")
    candidate_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    observation_rows: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    configs = []
    for candidate_id, root_dir, kind in entries:
        state = candidate_state(root_dir)
        if state["candidate_id"] != candidate_id:
            raise ValueError(f"candidate ID mismatch for {root_dir}")
        sim_root, geo_root = root_dir / "sim_connector.root", root_dir / "geo_connector.root"
        if not sim_root.is_file() or not geo_root.is_file():
            raise FileNotFoundError(f"missing ROOT evidence for {candidate_id}")
        environment = load_json(root_dir / "environment.json")
        if environment.get("fairship_commit") != anchor_commit:
            raise RuntimeError(f"FairShip commit differs for {candidate_id}")
        geo_summary, rows = trajectory(geometry, state)
        sim_summary, detector_rows = root_observations(sim_root, candidate_id)
        source_hashes[candidate_id] = sha256(sim_root)
        configs.append(run_config(root_dir / "commands.txt"))
        candidate_rows.append({"candidate_id": candidate_id, "cohort": kind, **state,
                               **source_metadata.get(candidate_id, {}),
                               "injection_x_cm": 100.0 * state["x"],
                               "injection_y_cm": 100.0 * state["y"],
                               "injection_z_cm": 100.0 * state["z"], **geo_summary,
                               **sim_summary, "sim_root_sha256": source_hashes[candidate_id],
                               "geometry_root_sha256": sha256(geo_root),
                               "sim_root": str(sim_root), "geometry_anchor": str(geometry_path),
                               "coordinate_status": "PROVISIONAL"})
        trajectory_rows.extend({"candidate_id": candidate_id, "cohort": kind, **row} for row in rows)
        observation_rows.extend({"cohort": kind, **row} for row in detector_rows)
    geometry_file.Close()
    controls = [row for row in candidate_rows if row["cohort"] == "empirical_sbt_positive_control"]
    if not all(row["sbt_qualifying_hit_count"] > 0 for row in controls):
        raise RuntimeError("no empirical SBT-positive control was reproduced")
    if any(config != anchor_config for config in configs):
        raise RuntimeError("FairShip run configuration differs across audit inputs")

    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "candidate_audit.csv", candidate_rows)
    write_csv(output / "trajectory_boundaries.csv", trajectory_rows)
    write_csv(output / "detector_observations.csv", observation_rows)
    manifest = {
        "schema_version": "fairship_tgeo_audit_v0",
        "decision": DECISION,
        "decision_scope": "audit usability only; no endpoint B, rate, or causal claim",
        "cohort": {"genuine_nf_candidates": 12, "empirical_positive_controls": len(controls)},
        "inputs": {"pilot_dir": str(pilot_dir), "controls_dir": str(controls_dir)},
        "fairship": {"commit": anchor_commit, "root_version": ROOT.gROOT.GetVersion(),
                     "geometry_anchor": str(geometry_path), "geometry_anchor_sha256": sha256(geometry_path),
                     "declared_config": anchor_config, "sensitive_volume_authority": SBT_SOURCE},
        "coordinate_transform": {"status": "PROVISIONAL", "mapping_question": "not resolved by this audit",
                                 "rule": "existing connector transform recorded in candidate.json"},
        "sbt_selection": {"source": "FairShip/muonDIS/make_nTuple_SBT.py", "threshold_gev": 3.0,
                          "rule": "1000 < detector_id < 999999 AND abs(pdg_id) == 13 AND momentum_gev > 3.0"},
        "artifacts": {"candidate_audit": "candidate_audit.csv", "trajectory_boundaries": "trajectory_boundaries.csv",
                       "detector_observations": "detector_observations.csv"},
        "source_root_sha256": source_hashes,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    readme = f"""# FairShip matched TGeo audit v0

Decision: `{DECISION}`.

The fixed 12-candidate genuine NF cohort and {len(controls)} empirical SBT-positive controls were read from existing ROOT outputs. Every row is matched to the same FairShip commit/configuration and one geometry anchor. `candidate_audit.csv` keeps injection state, TGeo initial volume/material, sensitive-volume encounters, detector counts, and the current preprocessing decision together; `trajectory_boundaries.csv` and `detector_observations.csv` provide the inspectable detail.

`LiSc*` is not a hand mask: it is the FairShip veto implementation's volume family created with `sens=true` (`{SBT_SOURCE}`). The coordinate transform remains PROVISIONAL; this is recorded uncertainty, not a #27 activation or a causal interpretation. No candidate was redrawn or tuned and no endpoint/rate claim is made.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--controls-dir", type=Path, required=True)
    parser.add_argument("--fairship-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(pilot_dir=args.pilot_dir, controls_dir=args.controls_dir, fairship=args.fairship_dir, output=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
