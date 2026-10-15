"""Fixed 12-state utility-tilted NF -> FairShip pilot; no training occurs here."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

from ship_muon_bg.adapters.fairship import CandidateInjectionRecord, CoordinateTransformConfig, FairShipTTreeConnector
from ship_muon_bg.adapters.current_main_mudis import CurrentMainMuonDISRunner
from ship_muon_bg.entities.subject import TagSubject
from ship_muon_bg.simulation.evaluation import EvaluationRequest


TILT_ID = "UA_d0p1_a04"
DELTA, ALPHA, UTILITY_MODE = 0.1, 4.0, "UA"
N_PER_PDG = 6
Z_METADATA_M = 28.905
TRANSFORM = CoordinateTransformConfig(
    name="afterms_nominal_plane_to_current_shield_exit_plus_1cm_f73a305_v0",
    status="PROVISIONAL", z_rule="affine_declared_z", z_offset_m=2.5916,
)
FIELDS = [
    "candidate_id", "pdg_id", "tilt_id", "delta", "alpha", "nf_run_id", "checkpoint_path",
    "checkpoint_sha256", "model_config_hash", "training_dataset_hash", "utility_mode", "sampling_seed",
    "sample_index", "px", "py", "pz", "x", "y", "proposal_log_prob", "physical_source_weight",
    "fairship_transport_weight", "coordinate_transform_id", "coordinate_transform_status",
    "mechanical_injection_verified", "n_mctracks", "n_sbt_hits", "n_ubt_hits", "n_tracker_points",
    "current_mudis_preprocessing_eligible", "n_dis_realizations", "geant4_completed", "shipreco_completed",
    "technical_status",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _checkpoint_dirs(pi_root: Path) -> dict[int, Path]:
    root = pi_root / "artifacts" / "density_lab" / "pi_demo_v0"
    return {
        13: root / "D9_utility_tilt_UA_d0p1_a04_p13_identity_cartesian_v0_pi_demo_cpu_affine_seed11_7d07ed8aa04d",
        -13: root / "D9_utility_tilt_UA_d0p1_a04_m13_identity_cartesian_v0_pi_demo_cpu_affine_seed11_b06f8578d91b",
    }


def _sample_rows(pi_root: Path, output: Path) -> list[dict[str, Any]]:
    """Load the recorded Torch checkpoints and persist exactly six draws per charge."""
    import numpy as np
    from Nflow.torch_models.affine_coupling import AffineCouplingFlow
    from ship_muon_bg.data_contracts.feature_views import FeatureView
    from ship_muon_bg.density_lab.feature_pipeline import FittedFeaturePipeline

    rows: list[dict[str, Any]] = []
    for pdg_id, run_dir in _checkpoint_dirs(pi_root).items():
        model_manifest = json.loads((run_dir / "model_manifest.json").read_text())
        pipeline_manifest = json.loads((run_dir / "feature_pipeline_manifest.json").read_text())
        dataset_manifest = json.loads((run_dir / "dataset_manifest.json").read_text())
        view = FeatureView(pipeline_manifest["feature_view_id"])
        std = pipeline_manifest["standardization"]
        pipeline = FittedFeaturePipeline(feature_view=view, mean=np.asarray(std["mean"]),
                                         std=np.asarray(std["std"]), n_train_rows=std["n_train_rows"],
                                         zero_variance_policy=std["zero_variance_policy"])
        model = AffineCouplingFlow.load(run_dir, device="cpu")
        seed = 731013 if pdg_id == 13 else 731113
        normalized = model.sample(N_PER_PDG, seed=seed)
        physical = pipeline.inverse_to_physical(normalized)
        log_prob = pipeline.normalized_to_physical_log_prob(model.log_prob(normalized), physical)
        checkpoint = run_dir / "checkpoint" / "state_dict.pt"
        config = run_dir / "checkpoint" / "model_config.json"
        for index, (state, lp) in enumerate(zip(physical, log_prob)):
            rows.append({
                "candidate_id": f"utility-pilot-{TILT_ID}-pdg{pdg_id}-{index:02d}", "pdg_id": pdg_id,
                "tilt_id": TILT_ID, "delta": DELTA, "alpha": ALPHA, "utility_mode": UTILITY_MODE,
                "nf_run_id": run_dir.name, "checkpoint_path": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint),
                "model_config_hash": _sha256(config), "training_dataset_hash": dataset_manifest["partitions"]["train"]["raw_dataset_hash"],
                "sampling_seed": seed, "sample_index": index, "px": float(state[0]), "py": float(state[1]),
                "pz": float(state[2]), "x": float(state[3]), "y": float(state[4]),
                "proposal_log_prob": float(lp), "proposal_log_prob_space": "physical_5d", "z_metadata_m": Z_METADATA_M,
                "physical_source_weight": None, "fairship_transport_weight": 1.0,
                "coordinate_transform_id": TRANSFORM.name, "coordinate_transform_status": TRANSFORM.status,
                "model_checkpoint_hash": model_manifest["checkpoint_hash"],
            })
    _json(output / "candidates_generated.json", {"fixed_budget": {"per_pdg": N_PER_PDG, "total": 2 * N_PER_PDG}, "records": rows})
    return rows


def _root_counts(path: Path) -> dict[str, int]:
    import ROOT  # type: ignore

    root_file = ROOT.TFile.Open(str(path), "READ")
    if not root_file or root_file.IsZombie():
        raise RuntimeError("unreadable cbmsim")
    tree = root_file.Get("cbmsim")
    if not tree or tree.GetEntry(0) <= 0:
        raise RuntimeError("missing readable cbmsim entry")
    sbt = 0
    for hit in tree.vetoPoint:
        momentum = math.sqrt(hit.GetPx() ** 2 + hit.GetPy() ** 2 + hit.GetPz() ** 2)
        if 1000 < hit.GetDetectorID() < 999999 and abs(hit.PdgCode()) == 13 and momentum > 3.0:
            sbt += 1
    return {"n_mctracks": len(tree.MCTrack), "n_sbt_hits": sbt, "n_ubt_hits": len(tree.UpstreamTaggerPoint),
            "n_tracker_points": len(tree.strawtubesPoint)}


def _shipreco(fairship: Path, g4_dir: Path, *, timeout: float | None) -> tuple[bool, str]:
    command = [sys.executable, str(fairship / "macro" / "ShipReco.py"), "-f", str(g4_dir / "sim_current-main-mudis.root"),
               "-g", str(g4_dir / "geo_current-main-mudis.root"), "-n", "2", "--validation"]
    completed = subprocess.run(command, cwd=g4_dir, text=True, capture_output=True, check=False, timeout=timeout)
    (g4_dir / "shipreco.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (g4_dir / "shipreco.stderr.log").write_text(completed.stderr, encoding="utf-8")
    return completed.returncode == 0 and (g4_dir / "sim_current-main-mudis_rec.root").is_file(), str(completed.returncode)


def _write_tables(output: Path, rows: list[dict[str, Any]], observations: list[dict[str, Any]], executions: list[dict[str, Any]]) -> None:
    with (output / "candidates.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    for name, values in (("observations.csv", observations), ("executions.csv", executions)):
        keys = sorted({key for value in values for key in value}) or ["candidate_id"]
        with (output / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(values)


def _readme(output: Path, rows: list[dict[str, Any]]) -> None:
    lines = ["# Utility-guided FairShip pilot v0", "", "This fixed pilot sampled six states per charge from existing `UA_d0p1_a04` NF checkpoints; it did not retrain or success-select candidates.", "", "| PDG | generated | mechanical | SBT-hit | MuDIS eligible | DIS realizations | Geant4 | ShipReco | technical failures |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for pdg in (13, -13):
        group = [row for row in rows if row["pdg_id"] == pdg]
        count = lambda key: sum(bool(row.get(key)) for row in group)
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(pdg, len(group), count("mechanical_injection_verified"), sum((row.get("n_sbt_hits") or 0) > 0 for row in group), count("current_mudis_preprocessing_eligible"), sum(row.get("n_dis_realizations") or 0 for row in group), count("geant4_completed"), count("shipreco_completed"), sum(str(row.get("technical_status", "")).startswith("TECHNICAL") for row in group)))
    lines += ["", "All coordinates use the existing PROVISIONAL transform; this is utility-guided generation into real simulation, not physical utility enrichment or a rate/efficiency estimate."]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run(output: Path, fairship: Path, rows: list[dict[str, Any]], *, timeout: float | None) -> None:
    observations, executions = [], []
    for row in rows:
        candidate = CandidateInjectionRecord.generated(row["candidate_id"], px=row["px"], py=row["py"], pz=row["pz"],
                                                       x=row["x"], y=row["y"], z=row["z_metadata_m"], pdg_id=row["pdg_id"])
        artifact = output / "executions" / row["candidate_id"]
        connector = FairShipTTreeConnector(fairship, python_executable=sys.executable, timeout=timeout)
        result = connector.run([candidate], artifact / "ttree", transform=TRANSFORM, seed=740000 + len(executions))
        row.update({"mechanical_injection_verified": result.verification.get("mechanical_injection_verified"),
                    "technical_status": "OK" if result.ok else "TECHNICAL_TTREE_FAILURE", "n_dis_realizations": None,
                    "geant4_completed": None, "shipreco_completed": None})
        execution = {"candidate_id": row["candidate_id"], "ttree_artifact": str(artifact / "ttree"), "ttree_ok": result.ok,
                     "ttree_returncode": result.simulation_returncode, "mudis_artifact": "NA"}
        if result.ok:
            counts = _root_counts(result.output_root)
            row.update(counts)
            eligible = counts["n_sbt_hits"] > 0
            row["current_mudis_preprocessing_eligible"] = eligible
            observations.extend({"candidate_id": row["candidate_id"], "name": key, "value": value,
                                 "status": "computed", "evidence": str(result.output_root)} for key, value in counts.items())
            observations.append({"candidate_id": row["candidate_id"], "name": "current_mudis_preprocessing_eligible", "value": eligible,
                                 "status": "computed", "evidence": "SBT qualifying hits: {}".format(counts["n_sbt_hits"])})
            if eligible:
                request = EvaluationRequest("utility-pilot-" + row["candidate_id"], (TagSubject(row["candidate_id"], "nf_generated_post_shield_muon", "afterms_5d_nf_v0"),),
                                            "fairship-f73a305-try2026-straw10-v0", 750000 + len(executions), options={"nf_run_id": row["nf_run_id"], "checkpoint_sha256": row["checkpoint_sha256"], "tilt_id": TILT_ID, "coordinate_transform_status": TRANSFORM.status})
                downstream = CurrentMainMuonDISRunner(fairship, python_executable=sys.executable, timeout=timeout).run(request, input_cbmsim=result.output_root, output_directory=artifact / "mudis", dis_realizations_per_muon=2)
                execution["mudis_artifact"] = str(artifact / "mudis")
                if downstream.ok:
                    row.update({"n_dis_realizations": len(downstream.bundle.realizations), "geant4_completed": True})
                    reco_ok, reco_code = _shipreco(fairship, artifact / "mudis" / "geant4", timeout=timeout)
                    row["shipreco_completed"] = reco_ok
                    if not reco_ok:
                        row["technical_status"] = "TECHNICAL_SHIPRECO_FAILURE_" + reco_code
                else:
                    row["technical_status"] = "TECHNICAL_MUDIS_FAILURE"
        else:
            row.update({"n_mctracks": None, "n_sbt_hits": None, "n_ubt_hits": None, "n_tracker_points": None, "current_mudis_preprocessing_eligible": None})
        executions.append(execution)
    _write_tables(output, rows, observations, executions)
    _readme(output, rows)
    _json(output / "manifest.json", {"tilt_id": TILT_ID, "utility_mode": UTILITY_MODE, "delta": DELTA, "alpha": ALPHA,
                                       "fixed_budget": {"pdg_13": N_PER_PDG, "pdg_minus_13": N_PER_PDG}, "transform": TRANSFORM.manifest(),
                                       "fairship_commit": subprocess.check_output(["git", "-C", str(fairship), "rev-parse", "HEAD"], text=True).strip(),
                                       "candidates_file": "candidates_generated.json"})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("sample", "run"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pi-demo-root", type=Path)
    parser.add_argument("--fairship-dir", type=Path)
    parser.add_argument("--timeout", type=float, default=None)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "sample":
        if not args.pi_demo_root:
            parser.error("sample requires --pi-demo-root")
        _sample_rows(args.pi_demo_root, args.output)
    else:
        if not args.fairship_dir:
            parser.error("run requires --fairship-dir")
        rows = json.loads((args.output / "candidates_generated.json").read_text())["records"]
        if len(rows) != 12 or {row["pdg_id"] for row in rows} != {13, -13}:
            raise ValueError("fixed pilot requires exactly six candidates for each charge")
        _run(args.output, args.fairship_dir, rows, timeout=args.timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
