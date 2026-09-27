from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from ship_muon_bg.adapters.fairship import (
    CandidateInjectionRecord,
    CoordinateTransformConfig,
    FairShipTTreeConnector,
    TechnicalFailure,
)


def test_generated_and_empirical_weights_are_separate() -> None:
    generated = CandidateInjectionRecord.generated("g", px=1, py=2, pz=3, x=4, y=5, z=6)
    empirical = CandidateInjectionRecord("e", 1, 2, 3, 4, 5, 6, 13, 7.5, source_provenance={"row": 4})
    assert generated.physical_source_weight is None
    assert empirical.physical_source_weight == 7.5
    assert generated.fairship_transport_weight == empirical.fairship_transport_weight == 1.0
    assert empirical.source_provenance == {"row": 4}
    try:
        CandidateInjectionRecord("bad", 1, 2, 3, 4, 5, 6, 13, None, 2.0)
    except ValueError as error:
        assert "requires fairship_transport_weight" in str(error)
    else:
        raise AssertionError("non-unit transport weight must not be silently ignored")


def test_transform_is_explicit_and_does_not_use_historical_offset() -> None:
    record = CandidateInjectionRecord.generated("c", px=1, py=2, pz=3, x=2, y=3, z=4)
    transform = CoordinateTransformConfig(name="cm_to_m_handshake", status="PROVISIONAL",
                                           x_unit="cm", y_unit="cm", x_to_m=.01, y_to_m=.01)
    assert transform.apply(record) == (0.02, 0.03, 4.0)
    assert transform.manifest()["z_rule"] == "declared_z"
    assert -68.5 not in transform.manifest().values()


def test_dry_run_writes_rows_and_required_artifacts(tmp_path: Path, monkeypatch) -> None:
    records = [CandidateInjectionRecord("c", 1, 2, 3, 4, 5, 6, 13, 9.0)]
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return type("Completed", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr("ship_muon_bg.adapters.fairship.subprocess.run", fake_run)
    monkeypatch.setattr("ship_muon_bg.adapters.fairship.inspect_root_output",
                        lambda *args, **kwargs: {"status": "verified", "entries": 1})
    result = FairShipTTreeConnector(tmp_path / "FairShip").run(records, tmp_path / "artifacts")
    assert result.ok
    with (tmp_path / "artifacts" / "input_candidates.pkl").open("rb") as stream:
        rows = pickle.load(stream)
    np.testing.assert_allclose(rows, [[1, 2, 3, 4, 5, 6, 13, 1]])
    required = {"request.json", "candidate.json", "bookkeeping.json", "environment.json", "commands.txt",
                "stdout.log", "stderr.log", "input_hashes.json", "output_hashes.json", "root_inventory.json",
                "verification.json", "summary.json"}
    assert required <= {p.name for p in (tmp_path / "artifacts").iterdir()}
    assert len(calls) == 3 and all(call[1].get("shell") is not True for call in calls)
    assert json.loads((tmp_path / "artifacts" / "bookkeeping.json").read_text())["ttree_w"] == [1.0]


def test_multi_candidate_run_verifies_every_executed_event(tmp_path: Path, monkeypatch) -> None:
    records = [
        CandidateInjectionRecord.generated("c0", px=1, py=2, pz=3, x=4, y=5, z=6),
        CandidateInjectionRecord.generated("c1", px=7, py=8, pz=9, x=10, y=11, z=12),
    ]
    monkeypatch.setattr(
        "ship_muon_bg.adapters.fairship.subprocess.run",
        lambda *args, **kwargs: type(
            "Completed", (), {"returncode": 0, "stdout": "", "stderr": ""}
        )(),
    )
    seen = []

    def inspect(_root, expected, _transform, *, event_index=0, **_kwargs):
        seen.append((expected.candidate_id, event_index))
        return {
            "status": "verified",
            "mechanical_injection_verified": True,
            "entries": 2,
            "candidate_id": expected.candidate_id,
            "event_index": event_index,
        }

    monkeypatch.setattr("ship_muon_bg.adapters.fairship.inspect_root_output", inspect)
    result = FairShipTTreeConnector(tmp_path / "FairShip").run(
        records, tmp_path / "artifacts"
    )

    assert result.ok
    assert seen == [("c0", 0), ("c1", 1)]
    assert result.verification["verified_candidate_count"] == 2
    assert result.verification["expected_candidate_count"] == 2
    assert result.verification["mechanical_injection_verified"] is True
    assert [item["candidate_id"] for item in result.verification["candidate_verifications"]] == ["c0", "c1"]


def test_nonzero_importer_is_technical_failure_not_physics_negative(tmp_path: Path, monkeypatch) -> None:
    def fake_run(command, **kwargs):
        return type("Completed", (), {"returncode": 2, "stdout": "", "stderr": "bad ROOT"})()

    monkeypatch.setattr("ship_muon_bg.adapters.fairship.subprocess.run", fake_run)
    result = FairShipTTreeConnector(tmp_path / "FairShip").run(
        [CandidateInjectionRecord.generated("c", px=1, py=2, pz=3, x=4, y=5, z=6)], tmp_path / "artifacts")
    assert not result.ok
    assert result.verification["status"] == "technical_failure"
    assert "return code 2" in result.technical_failures[0]


def test_missing_root_is_technical_failure(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("ship_muon_bg.adapters.fairship.subprocess.run",
                        lambda *args, **kwargs: type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})())
    def fail(*args, **kwargs):
        raise TechnicalFailure("ROOT output is missing or unreadable")
    monkeypatch.setattr("ship_muon_bg.adapters.fairship.inspect_root_output", fail)
    result = FairShipTTreeConnector(tmp_path / "FairShip").run(
        [CandidateInjectionRecord.generated("c", px=1, py=2, pz=3, x=4, y=5, z=6)], tmp_path / "artifacts")
    assert result.verification["status"] == "technical_failure"
    assert result.technical_failures == ["ROOT output is missing or unreadable"]
