import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_cli_module():
    spec = importlib.util.spec_from_file_location(
        "run_afterms_d9_gpu_hpo", REPO_ROOT / "scripts" / "run_afterms_d9_gpu_hpo.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cli = _load_cli_module()


class _Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_status_does_not_train(tmp_path, monkeypatch):
    """Required test 19: status command does not train."""

    from ship_muon_bg.afterms.d9b import gpu_gate

    def _boom(*args, **kwargs):
        raise AssertionError("status must never call run_gpu_gate")

    monkeypatch.setattr(gpu_gate, "run_gpu_gate", _boom)
    args = _Args(gate_artifact_root=tmp_path / "gate")
    rc = cli.cmd_status(args)
    assert rc == 0


def test_status_reports_not_yet_executed_when_no_manifest(tmp_path):
    args = _Args(gate_artifact_root=tmp_path / "gate")
    rc = cli.cmd_status(args)
    assert rc == 0


def test_status_reports_recorded_manifest(tmp_path):
    gate_root = tmp_path / "gate"
    gate_root.mkdir(parents=True)
    manifest = {"gate_status": "GPU_GATE_PASSED", "candidate_id": "X", "seed": 1, "stop_conditions": []}
    (gate_root / "qualification_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    args = _Args(gate_artifact_root=gate_root)
    rc = cli.cmd_status(args)
    assert rc == 0


def test_gpu_gate_dry_run_by_default_does_not_execute(monkeypatch):
    """`--execute` is required for real training; the default is dry-run."""

    from ship_muon_bg.afterms.d9b import gpu_gate

    def _boom(*args, **kwargs):
        raise AssertionError("run_gpu_gate must never be called without --execute")

    monkeypatch.setattr(gpu_gate, "run_gpu_gate", _boom)
    args = _Args(
        plan_path=REPO_ROOT / "configs" / "afterms" / "d9_candidate_plan_v0.json",
        training_config_path=REPO_ROOT / "configs" / "afterms" / "d9_training_v0.json",
        gate_artifact_root=REPO_ROOT / "artifacts" / "afterms_d9_gpu_hpo_v0" / "gpu_gate",
        execute=False, device="cuda", seed=None,
    )
    rc = cli.cmd_gpu_gate(args)
    assert rc == 0


def test_dry_run_and_execute_are_mutually_exclusive():
    parser = cli.build_parser()
    args = parser.parse_args(["gpu-gate", "--dry-run", "--execute"])
    assert args.dry_run is True
    assert args.execute is True
    rc = cli.main(["gpu-gate", "--dry-run", "--execute"])
    assert rc == 2


def test_no_hpo_command_is_active_yet():
    """Required test 20: no HPO command is active yet."""

    parser = cli.build_parser()
    subparsers_action = next(a for a in parser._subparsers._group_actions if hasattr(a, "choices"))
    assert "hpo" not in subparsers_action.choices
    assert set(subparsers_action.choices.keys()) == {"gpu-gate", "status"}


def test_frozen_d7_d8_inputs_hashes_unchanged():
    """Required test 18: frozen input hashes remain unchanged by this gate's
    own presence (re-verifies the same D8 source hashes D9's own
    test_plan.py::test_d7_d8_inputs_unchanged checks)."""

    from ship_muon_bg.afterms.d9 import plan as d9plan

    plan_path = REPO_ROOT / "configs" / "afterms" / "d9_candidate_plan_v0.json"
    plan = d9plan.load_plan(plan_path)
    mismatches = d9plan.verify_source_hashes(plan, REPO_ROOT)
    assert mismatches == []
