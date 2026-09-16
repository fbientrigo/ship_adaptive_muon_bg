#!/usr/bin/env python
"""Audit a versioned after-MS → FairShip transform without acquiring labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same(left: Any, right: Any, *, tolerance: float = 1e-3) -> bool:
    return isinstance(left, (int, float)) and isinstance(right, (int, float)) and math.isclose(
        float(left), float(right), rel_tol=0, abs_tol=tolerance
    )


def _transform(record: dict[str, Any], transform: dict[str, Any]) -> dict[str, Any]:
    rule = transform["z_rule"]
    z = record["z"] if rule == "declared_z" else transform.get("z_value_m")
    if rule == "affine_declared_z":
        z = transform["z_scale"] * record["z"] + transform["z_offset_m"]
    if z is None:
        raise ValueError("transform has no z value")
    return {
        "px": record["px"], "py": record["py"], "pz": record["pz"],
        "x": record["x"] * transform["x_to_m"],
        "y": record["y"] * transform["y_to_m"], "z": z, "pdg_id": record["pdg_id"],
    }


def _artifact_path(root: Path, path: Path) -> str:
    """Keep published provenance stable without leaking a local worktree path."""
    return str(Path(root.parent.name) / root.name / path.name)


def validate(*, reference_dirs: list[Path], tgeo_audit_root: Path) -> dict[str, Any]:
    """Return a conservative replay audit; only common-frame evidence can pass it."""
    if len(reference_dirs) < 2:
        raise ValueError("at least two independently injected reference directories are required")
    audit_manifest = _json(tgeo_audit_root / "manifest.json")
    with (tgeo_audit_root / "candidate_audit.csv").open(newline="", encoding="utf-8") as stream:
        audit_rows = {row["candidate_id"]: row for row in csv.DictReader(stream)}

    references = []
    transforms = []
    commits = []
    geometry_ids = []
    source_rows = []
    for root in reference_dirs:
        root = root.resolve()
        candidate = _json(root / "candidate.json")
        request = _json(root / "request.json")
        verification = _json(root / "verification.json")
        environment = _json(root / "environment.json")
        records, transformed_states = candidate.get("records"), candidate.get("transformed_states")
        if not isinstance(records, list) or len(records) != 1 or not isinstance(transformed_states, list) or len(transformed_states) != 1:
            raise ValueError(f"reference must contain one injected state: {root}")
        record, transformed = records[0], transformed_states[0]
        candidate_id = record.get("candidate_id")
        if not isinstance(candidate_id, str) or candidate_id not in audit_rows:
            raise ValueError(f"reference is absent from TGeo audit: {root}")
        audit = audit_rows[candidate_id]
        transform = request.get("transform")
        if not isinstance(transform, dict):
            raise ValueError(f"reference lacks transform: {root}")
        expected = _transform(record, transform)
        observed = verification.get("first_mctrack", {})
        intended = verification.get("intended", {})
        replay_ok = (
            verification.get("status") == "verified"
            and verification.get("mechanical_injection_verified") is True
            and all(_same(transformed[key], expected[key]) for key in ("px", "py", "pz", "x", "y", "z"))
            and transformed.get("pdg_id") == expected["pdg_id"]
            and all(_same(intended.get(key), expected[key] * (100 if key in {"x", "y", "z"} else 1)) for key in ("px", "py", "pz", "x", "y", "z"))
            and all(_same(observed.get(key), intended.get(key)) for key in ("px", "py", "pz", "x", "y", "z"))
            and observed.get("pdg_id") == intended.get("pdg_id") == expected["pdg_id"]
        )
        sim_path, geometry_path = root / "sim_connector.root", root / "geo_connector.root"
        geometry_ok = (
            sim_path.is_file() and geometry_path.is_file()
            and _sha256(sim_path) == audit["sim_root_sha256"]
            and _sha256(geometry_path) == audit["geometry_root_sha256"]
            and bool(audit["geometry_id"])
        )
        references.append({
            "candidate_id": candidate_id, "reference_directory": str(Path(root.parent.name) / root.name),
            "sim_root": _artifact_path(root, sim_path), "sim_root_sha256": _sha256(sim_path),
            "geometry_anchor": _artifact_path(root, geometry_path), "geometry_anchor_sha256": _sha256(geometry_path),
            "geometry_id": audit["geometry_id"], "initial_volume": audit["initial_volume"],
            "initial_material": audit["initial_material"], "mechanical_replay_ok": replay_ok,
            "geometry_anchor_ok": geometry_ok,
            "coordinate_physics_verified": verification.get("coordinate_physics_verified") is True,
            "source_state": {key: record.get("source_provenance", {}).get(key) for key in (
                "source_kind", "source_dataset_hash", "sample_file_sha256", "sample_row_index"
            )},
        })
        transforms.append(transform)
        commits.append(environment.get("fairship_commit"))
        geometry_ids.append(audit["geometry_id"])
        source_rows.append(record.get("source_provenance", {}).get("sample_row_index"))

    mechanics_ok = all(row["mechanical_replay_ok"] and row["geometry_anchor_ok"] for row in references)
    identity_ok = len({json.dumps(value, sort_keys=True) for value in transforms}) == 1 and len(set(commits)) == 1
    identity_ok = identity_ok and commits[0] == audit_manifest.get("fairship", {}).get("commit")
    identity_ok = identity_ok and len(set(geometry_ids)) == 1 and geometry_ids[0] == audit_manifest.get("fairship", {}).get("geometry_id")
    independent = len({row["candidate_id"] for row in references}) == len(references) and len(set(source_rows)) == len(source_rows)
    common_frame_ok = all(row["coordinate_physics_verified"] for row in references)
    passed = mechanics_ok and identity_ok and independent and common_frame_ok
    return {
        "schema_version": "afterms_fairship_transform_validation_v0",
        "transform": transforms[0], "fairship_commit": commits[0], "geometry_id": geometry_ids[0],
        "tgeo_audit_root": str(Path("artifacts") / tgeo_audit_root.name),
        "tgeo_audit_manifest_sha256": _sha256(tgeo_audit_root / "manifest.json"),
        "fairship_configuration": audit_manifest.get("fairship", {}).get("declared_config"),
        "reproduction": {
            "script": "scripts/validate_afterms_fairship_transform.py",
            "reference_dirs": [row["reference_directory"] for row in references],
            "tgeo_audit_root": str(Path("artifacts") / tgeo_audit_root.name),
        },
        "references": references,
        "checks": {"mechanical_replay": mechanics_ok, "pinned_identity": identity_ok,
                   "independent_reference_states": independent, "common_frame_anchor": common_frame_ok},
        "regression_check": "tests/test_afterms_fairship_transform_validation.py",
        "decision": "PASSED" if passed else "BLOCKED",
        "blocking_reason": None if passed else (
            "The references replay the asserted transform and match their TGeo anchors, but do not independently "
            "locate the after-MS plane in the FairShip frame; coordinate_physics_verified remains false."
        ),
        "u0_label_acquisition_permitted": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", required=True, action="append", type=Path)
    parser.add_argument("--tgeo-audit-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = validate(reference_dirs=args.reference_dir, tgeo_audit_root=args.tgeo_audit_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "README.md").write_text(
        "# After-MS → FairShip transform validation\n\n"
        f"Decision: `{result['decision']}`. Mechanical replay used the two artifact-relative reference directories in `manifest.json`, pinned FairShip commit `{result['fairship_commit']}`, and ShipGeo identity `{result['geometry_id']}`. The manifest retains the source dataset hashes and ROOT hashes.\n\n"
        "Reproduce with `python scripts/validate_afterms_fairship_transform.py --reference-dir <fairship_connector_v0>/p3_row_25740 --reference-dir <fairship_connector_v0>/p3_row_32029 --tgeo-audit-root artifacts/fairship_tgeo_audit_v0 --output-dir artifacts/afterms_fairship_transform_validation_v0`.\n\n"
        f"{result['blocking_reason'] or 'The common-frame validation passed.'} U0 label acquisition remains disallowed while the decision is BLOCKED.\n",
        encoding="utf-8",
    )
    print(json.dumps({"decision": result["decision"], "output_dir": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
