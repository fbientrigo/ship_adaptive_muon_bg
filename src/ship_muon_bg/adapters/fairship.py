"""Small candidate -> FairShip TTree connector.

Only this module knows the FairShip command line and ROOT output shape.  ROOT
is imported lazily, so the records and dry-run tests remain usable without the
FairShip environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np


class TechnicalFailure(RuntimeError):
    """A run that cannot provide a trustworthy physics answer."""


@dataclass(frozen=True, init=False)
class CandidateInjectionRecord:
    """Generated state plus metadata retained outside FairShip transport.

    ``px, py, pz, x, y`` are the generated physics state.  ``z`` is declared
    source metadata consumed by an explicit coordinate rule; it is not folded
    into the generated-state tuple.  ``w`` is never sent as a utility weight to
    FairShip: TTree rows always carry ``1.0``.
    """

    candidate_id: str
    px: float
    py: float
    pz: float
    x: float
    y: float
    z: float
    pdg_id: int
    physical_source_weight: Optional[float]
    fairship_transport_weight: float
    source_provenance: Mapping[str, Any]

    def __init__(
        self,
        candidate_id: str,
        px: float,
        py: float,
        pz: float,
        x: float,
        y: float,
        z: float,
        pdg_id: int = 13,
        physical_source_weight: Optional[float] = None,
        fairship_transport_weight: float = 1.0,
        source_provenance: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("candidate_id must be a non-empty string")
        values = (px, py, pz, x, y, z, fairship_transport_weight)
        if not all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in values):
            raise ValueError("candidate kinematics and transport weight must be finite numbers")
        if physical_source_weight is not None and not math.isfinite(float(physical_source_weight)):
            raise ValueError("physical_source_weight must be finite or None")
        if not isinstance(pdg_id, int) or isinstance(pdg_id, bool):
            raise TypeError("pdg_id must be an int")
        if float(fairship_transport_weight) <= 0:
            raise ValueError("fairship_transport_weight must be positive")
        for name, value in (("candidate_id", candidate_id), ("px", float(px)), ("py", float(py)),
                            ("pz", float(pz)), ("x", float(x)), ("y", float(y)), ("z", float(z)),
                            ("pdg_id", pdg_id), ("physical_source_weight", None if physical_source_weight is None else float(physical_source_weight)),
                            ("fairship_transport_weight", float(fairship_transport_weight))):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "source_provenance", dict(source_provenance or {}))

    @classmethod
    def generated(cls, candidate_id: str, *, px: float, py: float, pz: float,
                  x: float, y: float, z: float, pdg_id: int = 13) -> "CandidateInjectionRecord":
        """Construct a generated candidate with no empirical source weight."""
        return cls(candidate_id, px, py, pz, x, y, z, pdg_id, None, 1.0)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CandidateInjectionRecord":
        source_weight = value.get("physical_source_weight", value.get("weight"))
        return cls(
            value["candidate_id"], value["px"], value["py"], value["pz"],
            value["x"], value["y"], value.get("z", value.get("z_metadata")),
            value.get("pdg_id", value.get("pdg", 13)), source_weight,
            value.get("fairship_transport_weight", 1.0), value.get("source_provenance"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id, "px": self.px, "py": self.py,
            "pz": self.pz, "x": self.x, "y": self.y, "z": self.z,
            "pdg_id": self.pdg_id, "physical_source_weight": self.physical_source_weight,
            "fairship_transport_weight": self.fairship_transport_weight,
            "source_provenance": dict(self.source_provenance),
        }


@dataclass(frozen=True)
class CoordinateTransformConfig:
    """Named, intentionally unvalidated coordinate handshake.

    The default is identity in metres and uses each record's declared z.  No
    historical z offset is implicit.  ``status`` stays OPEN/PROVISIONAL until
    a geometry owner validates the convention.
    """

    name: str = "identity_declared_z_v0"
    status: str = "OPEN"
    x_unit: str = "m"
    y_unit: str = "m"
    x_to_m: float = 1.0
    y_to_m: float = 1.0
    z_rule: str = "declared_z"
    z_scale: float = 1.0
    z_offset_m: float = 0.0
    z_offset_cm: Optional[float] = None
    z_value_m: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.name or self.status not in {"OPEN", "PROVISIONAL"}:
            raise ValueError("transform requires a name and status OPEN or PROVISIONAL")
        if self.z_rule == "affine":
            object.__setattr__(self, "z_rule", "affine_declared_z")
        if self.z_rule not in {"declared_z", "constant_m", "affine_declared_z"}:
            raise ValueError("z_rule must be declared_z, affine_declared_z, or constant_m")
        if self.z_rule == "constant_m" and self.z_value_m is None:
            raise ValueError("constant_m requires z_value_m")
        if self.z_value_m is not None and not math.isfinite(float(self.z_value_m)):
            raise ValueError("z_value_m must be finite")
        if self.z_offset_cm is not None:
            object.__setattr__(self, "z_offset_m", float(self.z_offset_cm) / 100.0)
        if not all(math.isfinite(float(v)) for v in (self.x_to_m, self.y_to_m, self.z_scale, self.z_offset_m)):
            raise ValueError("coordinate conversion factors must be finite")

    def apply(self, candidate: CandidateInjectionRecord) -> tuple[float, float, float]:
        if self.z_rule == "declared_z":
            z = candidate.z
        elif self.z_rule == "constant_m":
            z = float(self.z_value_m)
        else:
            z = self.z_scale * candidate.z + self.z_offset_m
        return candidate.x * self.x_to_m, candidate.y * self.y_to_m, z

    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.name, "status": self.status, "x_unit": self.x_unit,
            "y_unit": self.y_unit, "x_to_m": self.x_to_m, "y_to_m": self.y_to_m,
            "z_rule": self.z_rule, "z_value_m": self.z_value_m,
            "z_scale": self.z_scale, "z_offset_m": self.z_offset_m,
        }


@dataclass
class TTreeConnectorResult:
    output_directory: Path
    converted_ntuple: Path
    output_root: Path
    importer_returncode: Optional[int]
    simulation_returncode: Optional[int]
    verification: dict[str, Any]
    technical_failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.technical_failures and self.verification.get("status") == "verified"

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok, "output_directory": str(self.output_directory),
            "converted_ntuple": str(self.converted_ntuple), "output_root": str(self.output_root),
            "importer_returncode": self.importer_returncode,
            "simulation_returncode": self.simulation_returncode,
            "verification": self.verification, "technical_failures": self.technical_failures,
        }


def _json_write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(path: Path) -> Optional[str]:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _ttree_config() -> dict[str, Any]:
    # Explicitly omits eminem_importer's historical z offset.
    return {
        "px": {"index": 0, "unit": "GeV", "type": "float64"},
        "py": {"index": 1, "unit": "GeV", "type": "float64"},
        "pz": {"index": 2, "unit": "GeV", "type": "float64"},
        "x": {"index": 3, "unit": "m", "type": "float64"},
        "y": {"index": 4, "unit": "m", "type": "float64"},
        "z": {"index": 5, "unit": "m", "type": "float64"},
        "id": {"index": 6, "type": "int32"},
        "w": {"index": 7, "type": "float64"},
    }


def inspect_root_output(root_path: Path | str, expected: CandidateInjectionRecord,
                        transform: CoordinateTransformConfig, *, atol: float = 1e-6,
                        fairship_dir: Path | str | None = None) -> dict[str, Any]:
    """Read cbmsim and falsify the first injected MCTrack if it differs."""
    try:
        import ROOT  # type: ignore
        import shipunit as u  # type: ignore
    except Exception as exc:
        if fairship_dir is None:
            raise TechnicalFailure(f"ROOT inspection unavailable: {exc}") from exc
        python_dir = str(Path(fairship_dir) / "python")
        sys.path.insert(0, python_dir)
        try:
            import ROOT  # type: ignore
            import shipunit as u  # type: ignore
        except Exception as retry_exc:
            raise TechnicalFailure(f"ROOT inspection unavailable: {retry_exc}") from retry_exc
        finally:
            if sys.path[0] == python_dir:
                sys.path.pop(0)
    file_handle = ROOT.TFile.Open(str(root_path), "READ")
    if not file_handle or file_handle.IsZombie():
        raise TechnicalFailure("ROOT output is missing or unreadable")
    if file_handle.TestBit(ROOT.TFile.kRecovered):
        raise TechnicalFailure("ROOT output was recovered after truncation")
    tree = file_handle.Get("cbmsim")
    if not tree or not tree.GetBranch("MCTrack"):
        raise TechnicalFailure("cbmsim or MCTrack branch is missing")
    if int(tree.GetEntries()) < 1 or tree.GetEntry(0) <= 0:
        raise TechnicalFailure("cbmsim has no readable first event")
    tracks = getattr(tree, "MCTrack", None)
    if tracks is None or len(tracks) < 1:
        raise TechnicalFailure("first cbmsim event has no MCTrack")
    track = tracks[0]
    x, y, z = transform.apply(expected)
    observed = {
        "px": float(track.GetPx()), "py": float(track.GetPy()), "pz": float(track.GetPz()),
        "x": float(track.GetStartX()), "y": float(track.GetStartY()), "z": float(track.GetStartZ()),
        "pdg_id": int(track.GetPdgCode()),
    }
    intended = {"px": expected.px * u.GeV, "py": expected.py * u.GeV, "pz": expected.pz * u.GeV,
                "x": x * u.m, "y": y * u.m, "z": z * u.m, "pdg_id": expected.pdg_id}
    mismatches = [key for key in ("px", "py", "pz", "x", "y", "z")
                  if not math.isclose(observed[key], float(intended[key]), rel_tol=0, abs_tol=atol * max(1.0, abs(float(intended[key]))))]
    if observed["pdg_id"] != intended["pdg_id"]:
        mismatches.append("pdg_id")
    if mismatches:
        raise TechnicalFailure(f"first MCTrack differs from injected state: {mismatches}")
    return {"status": "verified", "mechanical_injection_verified": True,
            "coordinate_physics_verified": False, "coordinate_transform_status": transform.status,
            "root_version": ROOT.gROOT.GetVersion(), "tree": "cbmsim",
            "branches": [branch.GetName() for branch in tree.GetListOfBranches()],
            "entries": int(tree.GetEntries()), "first_mctrack": observed, "intended": intended}


class FairShipTTreeConnector:
    def __init__(self, fairship_dir: Path | str, *, python_executable: str = sys.executable,
                 timeout: Optional[float] = None) -> None:
        self.fairship_dir = Path(fairship_dir).resolve()
        self.python_executable = python_executable
        self.timeout = timeout

    def run(self, candidates: Sequence[CandidateInjectionRecord], output_directory: Path | str,
            *, transform: Optional[CoordinateTransformConfig] = None, seed: int = 1,
            max_events: Optional[int] = None, inspect: bool = True) -> TTreeConnectorResult:
        transform = transform or CoordinateTransformConfig()
        candidates = tuple(candidates)
        if not candidates:
            raise ValueError("at least one candidate is required")
        if len({c.candidate_id for c in candidates}) != len(candidates):
            raise ValueError("candidate_id values must be unique")
        out = Path(output_directory).resolve()
        out.mkdir(parents=True, exist_ok=True)
        converted = out / "input_candidates.pkl"
        config_path = out / "ttree_column_config.json"
        sim_root = out / "sim_connector.root"
        transformed = [transform.apply(c) for c in candidates]
        rows = np.asarray([[c.px, c.py, c.pz, x, y, z, c.pdg_id, 1.0]
                           for c, (x, y, z) in zip(candidates, transformed)], dtype=np.float64)
        with converted.open("wb") as stream:
            pickle.dump(rows, stream, protocol=4)
        _json_write(config_path, _ttree_config())
        _json_write(out / "request.json", {"schema_version": "ttree_connector_v0", "seed": seed,
                    "max_events": max_events if max_events is not None else len(candidates),
                    "candidate_ids": [c.candidate_id for c in candidates], "transform": transform.manifest(),
                    "fairship_dir": str(self.fairship_dir)})
        _json_write(out / "candidate.json", {"records": [c.as_dict() for c in candidates],
                    "transformed_states": [{"px": c.px, "py": c.py, "pz": c.pz, "x": x, "y": y, "z": z, "pdg_id": c.pdg_id}
                                            for c, (x, y, z) in zip(candidates, transformed)]})
        _json_write(out / "bookkeeping.json", {"empirical_source_weights": [c.physical_source_weight for c in candidates],
                    "fairship_transport_weights": [c.fairship_transport_weight for c in candidates],
                    "ttree_w": [1.0] * len(candidates), "weight_rule": "physical source weight preserved separately; TTree w=1.0"})
        _json_write(out / "environment.json", {"python": sys.version, "platform": sys.platform,
                    "fairship_dir": str(self.fairship_dir), "fairship_commit": _git_head(self.fairship_dir)})
        importer = self.fairship_dir / "python" / "experimental" / "eminem_importer.py"
        macro = self.fairship_dir / "macro" / "run_simScript.py"
        n_events = max_events if max_events is not None else len(candidates)
        commands = [[self.python_executable, str(importer), str(converted), "--output", str(out / "converted_ntuple.root"),
                     "--format", "ttree", "--tree-name", "converted_ntuple", "--config", str(config_path)],
                    [self.python_executable, str(macro), "--ttree", "-f", str(out / "converted_ntuple.root"),
                     "-o", str(out), "-n", str(n_events), "-s", str(seed), "-r", str(seed), "--tag", "connector",
                     "--reproducible", "--validation", "--shieldName", "TRY_2026", "--strawDesign", "10", "--noSND"]]
        (out / "commands.txt").write_text("\n".join(shlex.join(c) for c in commands) + "\n", encoding="utf-8")
        returncodes: list[Optional[int]] = [None, None]
        stdout_logs: list[str] = []
        stderr_logs: list[str] = []
        failures: list[str] = []
        for index, command in enumerate(commands):
            if failures:
                break
            try:
                completed = subprocess.run(command, cwd=str(self.fairship_dir), capture_output=True, shell=False,
                                           text=True, timeout=self.timeout, check=False)
                returncodes[index] = completed.returncode
                stdout_logs.append(f"[{index}]\n{getattr(completed, 'stdout', '')}")
                stderr_logs.append(f"[{index}]\n{getattr(completed, 'stderr', '')}")
                if completed.returncode != 0:
                    failures.append(f"command {index} failed with return code {completed.returncode}")
            except (OSError, subprocess.TimeoutExpired) as exc:
                stderr_logs.append(f"[{index}]\n{exc}")
                failures.append(f"command {index} technical failure: {exc}")
        (out / "stdout.log").write_text("\n".join(stdout_logs), encoding="utf-8")
        (out / "stderr.log").write_text("\n".join(stderr_logs), encoding="utf-8")
        verification: dict[str, Any] = {"status": "technical_failure", "technical_failures": failures}
        if not failures and inspect:
            try:
                verification = inspect_root_output(sim_root, candidates[0], transform,
                                                   fairship_dir=self.fairship_dir)
            except (TechnicalFailure, OSError, ValueError) as exc:
                failures.append(str(exc))
                verification = {"status": "technical_failure", "technical_failures": failures}
        elif not failures:
            verification = {"status": "not_inspected"}
        _json_write(out / "verification.json", verification)
        input_files = (converted, config_path, out / "converted_ntuple.root")
        _json_write(out / "input_hashes.json", {p.name: _sha256(p) for p in input_files if p.exists()})
        _json_write(out / "output_hashes.json", {p.name: _sha256(p) for p in out.glob("*.root")
                    if p.name != "converted_ntuple.root"})
        _json_write(out / "root_inventory.json", {"status": verification.get("status"), "path": str(sim_root),
                    "exists": sim_root.exists(), "size": sim_root.stat().st_size if sim_root.exists() else 0,
                    "cbmsim": verification.get("tree") == "cbmsim",
                    "entries": verification.get("entries"),
                    "branches": verification.get("branches", []),
                    "first_mctrack_checked": "first_mctrack" in verification})
        result = TTreeConnectorResult(out, out / "converted_ntuple.root", sim_root, returncodes[0], returncodes[1], verification, failures)
        _json_write(out / "summary.json", result.as_dict())
        return result


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, help="JSON list or {records: [...]} input")
    parser.add_argument("--fairship-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--transform-name", default="identity_declared_z_v0")
    parser.add_argument("--transform-status", choices=("OPEN", "PROVISIONAL"), default="OPEN")
    parser.add_argument("--z-rule", choices=("declared_z", "affine_declared_z", "constant_m"), default="declared_z")
    parser.add_argument("--z-scale", type=float, default=1.0)
    parser.add_argument("--z-offset-m", type=float, default=0.0)
    parser.add_argument("--z-value-m", type=float)
    parser.add_argument("--no-inspect", action="store_true")
    args = parser.parse_args()
    payload = json.loads(Path(args.candidates).read_text(encoding="utf-8"))
    records = payload.get("records", payload) if isinstance(payload, dict) else payload
    candidates = [CandidateInjectionRecord.from_mapping(item) for item in records]
    result = FairShipTTreeConnector(args.fairship_dir).run(
        candidates, args.output_dir,
        transform=CoordinateTransformConfig(name=args.transform_name, status=args.transform_status,
                                            z_rule=args.z_rule, z_scale=args.z_scale,
                                            z_offset_m=args.z_offset_m, z_value_m=args.z_value_m),
        seed=args.seed, max_events=args.max_events, inspect=not args.no_inspect,
    )
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    return 0 if result.ok or (args.no_inspect and not result.technical_failures) else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
