"""Current-main FairShip MuonDIS adapter (path A only).

The scientific core sees canonical records; this module alone knows the
current FairShip scripts and ROOT trees.  It deliberately supports one source
state per invocation because current ``make_nTuple_SBT.py`` does not preserve a
source-state id through a mixed input file.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from ship_muon_bg.entities.identifiers import definition_id
from ship_muon_bg.entities.lineage import (
    OPTIONS_DIGEST_PROVENANCE_KEY,
    ExecutionStatus,
    FSSimExecution,
    InteractionRealization,
)
from ship_muon_bg.entities.observation import (
    ObservationEnvelope,
    ObservationEvaluationStatus,
    ScalarObservationPayload,
)
from ship_muon_bg.simulation.evaluation import (
    EvaluationBundle,
    EvaluationRequest,
    verify_evaluation_bundle,
)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def _root_inventory(path: Path, tree_name: str, *, require_cross_section: bool = False) -> dict[str, Any]:
    """Return only scalar ROOT facts; no ROOT object escapes this adapter."""
    import ROOT  # type: ignore

    root_file = ROOT.TFile.Open(str(path), "READ")
    if not root_file or root_file.IsZombie() or root_file.TestBit(ROOT.TFile.kRecovered):
        raise RuntimeError(f"unreadable ROOT output: {path}")
    tree = root_file.Get(tree_name)
    if not tree:
        raise RuntimeError(f"missing {tree_name} tree in {path}")
    entries = int(tree.GetEntries())
    if entries < 1:
        raise RuntimeError(f"{tree_name} has no entries in {path}")
    branches = [branch.GetName() for branch in tree.GetListOfBranches()]
    values: list[float] = []
    muon_rows: list[list[float]] = []
    if tree_name == "MuonAndSoftInteractions":
        if tree.GetEntry(0) <= 0 or not hasattr(tree, "imuondata") or len(tree.imuondata) < 10:
            raise RuntimeError(f"missing readable imuondata in {path}")
        muon_rows.append([float(tree.imuondata[index]) for index in range(10)])
    elif tree_name == "DIS":
        for index in range(entries):
            if tree.GetEntry(index) <= 0 or not hasattr(tree, "InMuon") or len(tree.InMuon) < 1:
                raise RuntimeError(f"missing readable DIS InMuon entry {index} in {path}")
            muon_rows.append([float(tree.InMuon[0][column]) for column in range(11)])
    elif tree_name == "cbmsim":
        for index in range(entries):
            if tree.GetEntry(index) <= 0 or not hasattr(tree, "MCTrack") or len(tree.MCTrack) < 1:
                raise RuntimeError(f"missing readable MCTrack entry {index} in {path}")
            track = tree.MCTrack[0]
            muon_rows.append([float(track.GetPdgCode()), float(track.GetPx()), float(track.GetPy()), float(track.GetPz())])
    if require_cross_section:
        if "CrossSection" not in branches:
            raise RuntimeError(f"missing CrossSection branch in {path}")
        for index in range(entries):
            if tree.GetEntry(index) <= 0:
                raise RuntimeError(f"unreadable {tree_name} entry {index} in {path}")
            values.append(float(getattr(tree, "CrossSection")))
    root_file.Close()
    return {"path": str(path), "tree": tree_name, "entries": entries, "branches": branches,
            "cross_sections": values, "muon_rows": muon_rows}


@dataclass(frozen=True)
class CurrentMainMuonDISResult:
    output_directory: Path
    bundle: EvaluationBundle
    technical_failure: Optional[str]

    @property
    def ok(self) -> bool:
        return self.technical_failure is None


class CurrentMainMuonDISRunner:
    """Run current-main SBT -> Pythia6 MuonDIS -> Geant4 for one source state."""

    name = "fairship_current_main_mudis_v0"

    def __init__(self, fairship_dir: Path | str, *, python_executable: str = sys.executable,
                 timeout: Optional[float] = None) -> None:
        self.fairship_dir = Path(fairship_dir).resolve()
        self.python_executable = python_executable
        self.timeout = timeout

    def run(self, request: EvaluationRequest, *, input_cbmsim: Path | str,
            output_directory: Path | str, dis_realizations_per_muon: int = 2) -> CurrentMainMuonDISResult:
        if len(request.subjects) != 1 or request.replications_per_subject != 1:
            raise ValueError("current MuonDIS runner accepts one source state and one replication")
        if dis_realizations_per_muon <= 0 or dis_realizations_per_muon % 2:
            raise ValueError("dis_realizations_per_muon must be positive and even")
        source = Path(input_cbmsim).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        out = Path(output_directory).resolve()
        out.mkdir(parents=True, exist_ok=True)
        subject = request.subjects[0]
        execution_id = f"exec:{request.request_id}:{subject.subject_id}:0"
        fairship_commit = _git_head(self.fairship_dir)
        stage = out / "staging" / subject.subject_id
        stage.mkdir(parents=True, exist_ok=True)
        staged = stage / "ship.conical.MuonBack-TGeant4.root"
        if staged.exists() or staged.is_symlink():
            staged.unlink()
        try:
            staged.symlink_to(source)
        except OSError:
            shutil.copy2(source, staged)

        preprocessed = out / "muonsProduction_wsoft_SBT.root"
        dis_file = out / "muonDis.root"
        g4_dir = out / "geant4"
        g4_file = g4_dir / "sim_current-main-mudis.root"
        commands = [
            [self.python_executable, str(self.fairship_dir / "muonDIS" / "make_nTuple_SBT.py"),
             "-p", str(out / "staging"), "-o", str(preprocessed)],
            [self.python_executable, str(self.fairship_dir / "muonDIS" / "makeMuonDIS.py"),
             "-f", str(preprocessed), "-i", "0", "-n", "1", "-nDISPerMuon", str(dis_realizations_per_muon)],
            [self.python_executable, str(self.fairship_dir / "macro" / "run_simScript.py"),
             "--MuDIS", "-f", str(dis_file), "-i", "0", "-n", str(dis_realizations_per_muon),
             "-s", str(request.seed), "-r", str(request.seed), "-o", str(g4_dir),
             "--tag", "current-main-mudis", "--reproducible", "--validation", "--shieldName", "TRY_2026",
             "--strawDesign", "10", "--noSND"],
        ]
        _write_json(out / "request.json", {
            "request_id": request.request_id, "subject_id": subject.subject_id,
            "state_definition_id": subject.state_definition_id,
            "fs_sim_configuration_id": request.fs_sim_configuration_id, "seed": request.seed,
            "options": dict(request.options), "options_digest": request.options_digest, "input_cbmsim": str(source),
            "dis_realizations_per_muon": dis_realizations_per_muon,
        })
        _write_json(out / "environment.json", {
            "fairship_dir": str(self.fairship_dir), "fairship_commit": fairship_commit,
            "python_executable": self.python_executable,
            "pythia6_seed": "uncontrolled_wall_clock_in_current_main_makeMuonDIS.py",
        })
        (out / "commands.txt").write_text("\n".join(shlex.join(c) for c in commands) + "\n", encoding="utf-8")

        stdout: list[str] = []
        stderr: list[str] = []
        returns: list[Optional[int]] = []
        failure: Optional[str] = None
        for index, command in enumerate(commands):
            try:
                completed = subprocess.run(command, cwd=out, capture_output=True, text=True, shell=False,
                                           timeout=self.timeout, check=False)
                returns.append(completed.returncode)
                stdout.append(f"[{index}]\n{completed.stdout}")
                stderr.append(f"[{index}]\n{completed.stderr}")
                if completed.returncode:
                    failure = f"current-main command {index} failed with return code {completed.returncode}"
                    break
            except (OSError, subprocess.TimeoutExpired) as exc:
                returns.append(None)
                stderr.append(f"[{index}]\n{exc}")
                failure = f"current-main command {index} technical failure: {exc}"
                break
        (out / "stdout.log").write_text("\n".join(stdout), encoding="utf-8")
        (out / "stderr.log").write_text("\n".join(stderr), encoding="utf-8")

        inventories: dict[str, Any] = {}
        if failure is None:
            try:
                inventories = {
                    "preprocessing": _root_inventory(preprocessed, "MuonAndSoftInteractions"),
                    "dis": _root_inventory(dis_file, "DIS"),
                    "geant4": _root_inventory(g4_file, "cbmsim", require_cross_section=True),
                }
                if inventories["dis"]["entries"] != inventories["geant4"]["entries"]:
                    raise RuntimeError("DIS and Geant4 entry counts differ")
                self._verify_content_continuity(inventories)
            except (OSError, RuntimeError, ValueError) as exc:
                failure = str(exc)
        _write_json(out / "root_inventory.json", inventories if failure is None else {"technical_failure": failure, **inventories})
        hashes = {"input_cbmsim": _sha256(source)}
        for path in (preprocessed, dis_file, g4_file, g4_dir / "geo_current-main-mudis.root", g4_dir / "params_current-main-mudis.root"):
            if path.is_file():
                hashes[path.name] = _sha256(path)
        _write_json(out / "input_hashes.json", {"input_cbmsim": hashes["input_cbmsim"]})
        _write_json(out / "output_hashes.json", {key: value for key, value in hashes.items() if key != "input_cbmsim"})

        provenance = {
            OPTIONS_DIGEST_PROVENANCE_KEY: request.options_digest,
            "runner": self.name, "fairship_commit": fairship_commit, "input_cbmsim_sha256": hashes["input_cbmsim"],
            "artifact_directory": str(out), "pythia6_seed": "uncontrolled_wall_clock",
            "geant4_seed": str(request.seed), "dis_realizations_per_muon": str(dis_realizations_per_muon),
        }
        if failure is not None:
            execution = FSSimExecution(execution_id, subject.subject_id, request.fs_sim_configuration_id,
                                       ExecutionStatus.TECHNICAL_FAILURE, request.seed, failure, provenance)
            bundle = EvaluationBundle(request.request_id, self.name, True, executions=(execution,))
        else:
            execution = FSSimExecution(execution_id, subject.subject_id, request.fs_sim_configuration_id,
                                       ExecutionStatus.SUCCEEDED, request.seed, provenance=provenance)
            definition = definition_id("current_main_mudis_v0", {
                "fairship_commit": fairship_commit, "preprocessing_tree": "MuonAndSoftInteractions",
                "dis_tree": "DIS", "geant4_tree": "cbmsim", "cross_section_branch": "CrossSection",
            })
            realizations = tuple(InteractionRealization(
                f"real:{execution_id}:{index}", execution_id, "muon_dis", definition
            ) for index in range(inventories["geant4"]["entries"]))
            observations: list[ObservationEnvelope] = []
            for name, inventory in inventories.items():
                observations.append(self._observation(
                    f"obs:{name}:{execution_id}", execution_id, f"{self.name}:{name}_entries_v0",
                    "entries", float(inventory["entries"]), f"{inventory['path']}#{inventory['tree']}", request,
                ))
            for realization, value in zip(realizations, inventories["geant4"]["cross_sections"]):
                observations.append(self._observation(
                    f"obs:cross_section:{realization.realization_id}", realization.realization_id,
                    f"{self.name}:cross_section_native_v0", "FairShip-native", value,
                    f"{g4_file}#cbmsim:CrossSection", request,
                ))
            bundle = EvaluationBundle(request.request_id, self.name, True, executions=(execution,),
                                      realizations=realizations, observations=tuple(observations))
        verify_evaluation_bundle(bundle, request)
        _write_json(out / "verification.json", {
            "status": "verified" if failure is None else "technical_failure",
            "technical_failure": failure, "return_codes": returns,
            "mudis_preprocessing_completed": failure is None,
            "geant4_execution_completed": failure is None,
            "realization_count": len(bundle.realizations),
        })
        _write_json(out / "bundle.json", self._bundle_manifest(bundle))
        return CurrentMainMuonDISResult(out, bundle, failure)

    @staticmethod
    def _verify_content_continuity(inventories: Mapping[str, Mapping[str, Any]]) -> None:
        preprocessed = inventories["preprocessing"]["muon_rows"]
        dis_rows = inventories["dis"]["muon_rows"]
        geant4_rows = inventories["geant4"]["muon_rows"]
        cross_sections = inventories["geant4"]["cross_sections"]
        if len(preprocessed) != 1 or len(dis_rows) != len(geant4_rows) or len(dis_rows) != len(cross_sections):
            raise RuntimeError("missing muon lineage rows across current-main MuonDIS outputs")
        # makeMuonDIS copies the selected SBT muon into each DIS InMuon row;
        # run_simScript --MuDIS then uses it as the Geant4 primary.
        for index, dis in enumerate(dis_rows):
            for dis_column, pre_column in ((0, 0), (1, 1), (2, 2), (3, 3), (5, 4), (6, 5), (7, 6), (8, 7)):
                if not math.isclose(dis[dis_column], preprocessed[0][pre_column], rel_tol=1e-6, abs_tol=1e-6):
                    raise RuntimeError(f"preprocessing to DIS muon mismatch at realization {index}")
            g4 = geant4_rows[index]
            for left, right in zip(g4, dis[:4]):
                if not math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-6):
                    raise RuntimeError(f"DIS to Geant4 primary mismatch at realization {index}")
            if not math.isclose(cross_sections[index], dis[10], rel_tol=1e-5, abs_tol=1e-10):
                raise RuntimeError(f"DIS to Geant4 CrossSection mismatch at realization {index}")

    @staticmethod
    def _observation(observation_id: str, subject_ref: str, definition: str, units: str,
                     value: float, evidence: str, request: EvaluationRequest) -> ObservationEnvelope:
        return ObservationEnvelope(observation_id, subject_ref, definition, units,
                                   ObservationEvaluationStatus.COMPUTED, evidence,
                                   ScalarObservationPayload(value), {OPTIONS_DIGEST_PROVENANCE_KEY: request.options_digest})

    @staticmethod
    def _bundle_manifest(bundle: EvaluationBundle) -> Mapping[str, Any]:
        return {
            "request_id": bundle.request_id, "backend_name": bundle.backend_name, "is_physical": bundle.is_physical,
            "executions": [{"execution_id": item.execution_id, "subject_id": item.subject_id,
                            "status": item.execution_status.value, "provenance": dict(item.provenance)} for item in bundle.executions],
            "realizations": [{"realization_id": item.realization_id, "execution_id": item.execution_id,
                              "interaction_type": item.interaction_type, "interaction_definition_id": item.interaction_definition_id}
                             for item in bundle.realizations],
            "observations": [{"observation_id": item.observation_id, "subject_ref": item.subject_ref,
                              "definition": item.observation_definition_id, "units": item.units,
                              "value": getattr(item.payload, "value", None), "evidence_reference": item.evidence_reference}
                             for item in bundle.observations],
        }
