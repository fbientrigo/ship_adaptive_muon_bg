from pathlib import Path

from ship_muon_bg.adapters.current_main_mudis import CurrentMainMuonDISRunner
from ship_muon_bg.entities.subject import TagSubject
from ship_muon_bg.simulation.evaluation import EvaluationRequest, verify_evaluation_bundle


def _request() -> EvaluationRequest:
    return EvaluationRequest("req-1", (TagSubject("state-1", "post_shield_muon", "afterms_v0"),), "ship-v0", 42)


def _inventory(path: Path, tree: str, *, require_cross_section: bool = False):
    if tree == "MuonAndSoftInteractions":
        return {"path": str(path), "tree": tree, "entries": 1, "branches": [], "cross_sections": [], "muon_rows": [[13, 1, 2, 3, 4, 5, 6, 1, 0, 1]]}
    if tree == "DIS":
        return {"path": str(path), "tree": tree, "entries": 2, "branches": [], "cross_sections": [], "muon_rows": [[13, 1, 2, 3, 4, 4, 5, 6, 1, 1, .1], [13, 1, 2, 3, 4, 4, 5, 6, 1, 0, .2]]}
    return {"path": str(path), "tree": tree, "entries": 2, "branches": ["CrossSection"], "cross_sections": [.1, .2], "muon_rows": [[13, 1, 2, 3], [13, 1, 2, 3]]}


def test_current_main_chain_preserves_lineage_and_cross_sections(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "input.root"
    source.write_bytes(b"root")
    calls = []
    monkeypatch.setattr("ship_muon_bg.adapters.current_main_mudis._root_inventory", _inventory)
    monkeypatch.setattr("ship_muon_bg.adapters.current_main_mudis.subprocess.run", lambda command, **kwargs: calls.append(command) or type("R", (), {"returncode": 0, "stdout": "ok", "stderr": ""})())
    runner = CurrentMainMuonDISRunner(tmp_path / "FairShip")
    result = runner.run(_request(), input_cbmsim=source, output_directory=tmp_path / "out")
    assert result.ok and len(calls[1:]) == 3 and len(result.bundle.realizations) == 2
    assert {item.subject_ref for item in result.bundle.observations if item.units == "FairShip-native"} == {item.realization_id for item in result.bundle.realizations}
    assert result.bundle.executions[0].provenance["pythia6_seed"] == "uncontrolled_wall_clock"
    verify_evaluation_bundle(result.bundle, _request())


def test_command_failure_is_technical_failure_not_a_physics_negative(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "input.root"
    source.write_bytes(b"root")
    monkeypatch.setattr("ship_muon_bg.adapters.current_main_mudis.subprocess.run", lambda *args, **kwargs: type("R", (), {"returncode": 2, "stdout": "", "stderr": "bad"})())
    result = CurrentMainMuonDISRunner(tmp_path / "FairShip").run(_request(), input_cbmsim=source, output_directory=tmp_path / "out")
    assert not result.ok and result.bundle.executions[0].execution_status.value == "technical_failure"
    assert not result.bundle.realizations and not result.bundle.decisions
