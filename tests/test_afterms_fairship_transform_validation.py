from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path


def _load():
    path = Path(__file__).parents[1] / "scripts" / "validate_afterms_fairship_transform.py"
    spec = importlib.util.spec_from_file_location("transform_validation", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write(path: Path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_provisional_replay_is_blocked_until_a_common_frame_anchor_exists(tmp_path: Path) -> None:
    module = _load()
    audit = tmp_path / "audit"
    audit.mkdir()
    refs = []
    rows = []
    for index, pdg in enumerate((13, -13)):
        root = tmp_path / f"reference_{index}"
        root.mkdir()
        record = {"candidate_id": f"r{index}", "px": 1.0 + index, "py": 2.0, "pz": 3.0, "x": .1, "y": -.2, "z": 28.905, "pdg_id": pdg, "source_provenance": {"sample_row_index": index}}
        transform = {"name": "v0", "status": "PROVISIONAL", "x_to_m": 1.0, "y_to_m": 1.0, "z_rule": "affine_declared_z", "z_scale": 1.0, "z_offset_m": 2.5916, "z_value_m": None}
        transformed = {**record, "z": 31.4966}
        intended = {key: transformed[key] * (100 if key in {"x", "y", "z"} else 1) for key in ("px", "py", "pz", "x", "y", "z")}
        intended["pdg_id"] = pdg
        _write(root / "candidate.json", {"records": [record], "transformed_states": [transformed]})
        _write(root / "request.json", {"transform": transform})
        _write(root / "environment.json", {"fairship_commit": "abc"})
        _write(root / "verification.json", {"status": "verified", "mechanical_injection_verified": True, "coordinate_physics_verified": False, "intended": intended, "first_mctrack": intended})
        (root / "sim_connector.root").write_bytes(b"sim" + bytes([index]))
        (root / "geo_connector.root").write_bytes(b"geo" + bytes([index]))
        rows.append({"candidate_id": f"r{index}", "sim_root_sha256": hashlib.sha256((root / "sim_connector.root").read_bytes()).hexdigest(), "geometry_root_sha256": hashlib.sha256((root / "geo_connector.root").read_bytes()).hexdigest(), "geometry_id": "geometry", "initial_volume": "anchor", "initial_material": "iron"})
        refs.append(root)
    _write(audit / "manifest.json", {"fairship": {"commit": "abc", "geometry_id": "geometry"}})
    with (audit / "candidate_audit.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    result = module.validate(reference_dirs=refs, tgeo_audit_root=audit)
    assert result["checks"] == {"mechanical_replay": True, "pinned_identity": True, "independent_reference_states": True, "common_frame_anchor": False}
    assert result["decision"] == "BLOCKED"
    assert not result["u0_label_acquisition_permitted"]
    assert Path(result["references"][0]["reference_directory"]).name == "reference_0"
    assert result["tgeo_audit_root"] == "artifacts/audit"
