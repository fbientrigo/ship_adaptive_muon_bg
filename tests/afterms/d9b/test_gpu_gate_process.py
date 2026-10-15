import subprocess
import sys
from pathlib import Path

import pytest

from ship_muon_bg.afterms.d9b import gpu_gate

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_worker_script_exists_and_resolves():
    script = gpu_gate.resolve_worker_script(REPO_ROOT)
    assert script.name == "_d9b_gpu_gate_worker.py"
    assert script.exists()


@pytest.mark.local_env
def test_python_executable_resolves_into_venv():
    """Required test 5: child process command uses `.venv\\Scripts\\python.exe`."""

    exe = gpu_gate.resolve_python_executable(REPO_ROOT)
    assert ".venv" in exe.parts
    assert exe.name == "python.exe"


def test_build_train_command_uses_venv_python(tmp_path):
    python_exe = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    worker = REPO_ROOT / "scripts" / "_d9b_gpu_gate_worker.py"
    cmd = gpu_gate.build_train_command(
        python_exe, worker,
        candidate_config_json=tmp_path / "c.json", seed=1, shard_dir=tmp_path,
        artifact_root=tmp_path, device="cuda", telemetry_out=tmp_path / "t.json",
    )
    assert cmd[0] == str(python_exe)
    assert ".venv" in Path(cmd[0]).parts
    assert cmd[1] == str(worker)
    assert cmd[2] == "train"
    assert "--resume" not in cmd


def test_build_train_command_with_resume_and_interrupt(tmp_path):
    python_exe = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    worker = REPO_ROOT / "scripts" / "_d9b_gpu_gate_worker.py"
    cmd = gpu_gate.build_train_command(
        python_exe, worker,
        candidate_config_json=tmp_path / "c.json", seed=1, shard_dir=tmp_path,
        artifact_root=tmp_path, device="cuda", telemetry_out=tmp_path / "t.json",
        resume=True, interrupt_after_epochs=1,
    )
    assert "--resume" in cmd
    assert "--interrupt-after-epochs" in cmd
    assert cmd[cmd.index("--interrupt-after-epochs") + 1] == "1"


def test_build_reload_command_never_references_test_shard(tmp_path):
    python_exe = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    worker = REPO_ROOT / "scripts" / "_d9b_gpu_gate_worker.py"
    cmd = gpu_gate.build_reload_command(
        python_exe, worker, run_dir=tmp_path, device="cuda",
        sample_count=16, generation_seed=1, output_json=tmp_path / "o.json",
    )
    assert "reload" in cmd
    assert not any("test_shard" in part for part in cmd)


def _strip_docstrings_and_comments(source: str) -> str:
    """Strip all triple-quoted docstrings and `#` comments so code-level
    assertions below aren't tripped up by prose explaining what NOT to do
    (which necessarily names the forbidden symbols/flags to describe the
    safety invariant they violate)."""

    import re

    no_docstrings = re.sub(r'"""[\s\S]*?"""', "", source)
    no_docstrings = re.sub(r"'''[\s\S]*?'''", "", no_docstrings)
    return re.sub(r"#.*", "", no_docstrings)


def test_worker_module_never_imports_test_only_helpers():
    """Required test 4: test shards are structurally inaccessible to the worker."""

    worker_source = _strip_docstrings_and_comments(
        (REPO_ROOT / "scripts" / "_d9b_gpu_gate_worker.py").read_text(encoding="utf-8")
    )
    assert "evaluate_candidate_seed" not in worker_source
    assert "load_test_split" not in worker_source
    assert "test_shards" not in worker_source


def test_gpu_gate_module_never_issues_a_broad_kill():
    """Required test 8: no broad Python-process termination anywhere in this module."""

    full_source = (REPO_ROOT / "src" / "ship_muon_bg" / "afterms" / "d9b" / "gpu_gate.py").read_text(encoding="utf-8")
    source = _strip_docstrings_and_comments(full_source)
    assert "/IM" not in source
    assert "Stop-Process -Name" not in source
    assert "taskkill" in full_source  # present, but only in its PID-specific form
    assert "/PID" in full_source


@pytest.mark.local_env
def test_terminate_pid_tree_is_pid_specific(tmp_path):
    """Required test 7/8: PID-specific ownership; a harmless dummy subprocess
    is terminated by PID, not by name, and no unrelated process is touched."""

    dummy = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        result = gpu_gate.terminate_pid_tree(dummy.pid)
        assert result["pid"] == dummy.pid
        dummy.wait(timeout=10)
        assert dummy.returncode is not None
    finally:
        if dummy.poll() is None:
            dummy.kill()
            dummy.wait(timeout=10)


@pytest.mark.local_env
def test_enumerate_descendant_pids_finds_a_real_child(tmp_path):
    """Required test 7: PID-specific descendant enumeration via WMI, not by name."""

    parent = subprocess.Popen(
        [sys.executable, "-c", "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time; time.sleep(15)']); time.sleep(15)"]
    )
    try:
        descendants = []
        import time as _time
        for _ in range(20):
            descendants = gpu_gate.enumerate_descendant_pids(parent.pid)
            if descendants:
                break
            _time.sleep(0.5)
        assert descendants, "expected at least one descendant PID of the dummy parent process"
    finally:
        gpu_gate.terminate_pid_tree(parent.pid)
        parent.wait(timeout=10)


def test_process_supervisor_refuses_second_concurrent_child():
    """Required test 6: one active CUDA child maximum."""

    supervisor = gpu_gate.ProcessSupervisor()
    supervisor._active_pid = 999999  # simulate a still-active tracked child
    with pytest.raises(gpu_gate.ChildProcessAlreadyActiveError):
        supervisor.run([sys.executable, "-c", "pass"], label="should_not_launch")


def test_process_supervisor_runs_dummy_and_records_ledger():
    supervisor = gpu_gate.ProcessSupervisor()
    result = supervisor.run([sys.executable, "-c", "print('ok')"], label="dummy_ok", timeout=30)
    assert result["returncode"] == 0
    assert "ok" in result["stdout"]
    assert len(supervisor.ledger) == 1
    entry = supervisor.ledger[0]
    assert entry["label"] == "dummy_ok"
    assert entry["exit_status"] == 0
    assert supervisor.active_pid is None


def test_query_nvidia_smi_gpu_never_raises_on_missing_binary(monkeypatch):
    def _boom(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi not found")

    monkeypatch.setattr(subprocess, "run", _boom)
    result = gpu_gate.query_nvidia_smi_gpu()
    assert result["available"] is False


def test_cuda_telemetry_schema_from_mocked_snapshot():
    """Required test 9: CUDA telemetry schema."""

    telemetry = {
        "device": "cuda",
        "initial": {"allocated_bytes": 100, "reserved_bytes": 200},
        "peak": {"allocated_bytes": 500, "reserved_bytes": 600},
        "final": {"allocated_bytes": 50, "reserved_bytes": 200},
        "result": {"status": "completed"},
    }
    ledger_entry = {
        "nvidia_smi_before": {"memory_used_mib": 0},
        "nvidia_smi_after": {"memory_used_mib": 10},
    }
    row = gpu_gate._memory_observation_row("clean_run_1", telemetry, ledger_entry)
    for key in (
        "initial_allocated_bytes", "initial_reserved_bytes",
        "peak_allocated_bytes", "peak_reserved_bytes",
        "final_allocated_bytes", "final_reserved_bytes",
        "nvidia_smi_used_mib_before", "nvidia_smi_used_mib_after",
    ):
        assert key in row


def test_check_history_all_finite_flags_non_finite_loss():
    """Required test 11: non-finite loss fails the gate."""

    history = [
        {"epoch": 1, "finite_loss": True},
        {"epoch": 2, "finite_loss": False},
    ]
    violations = gpu_gate.check_history_all_finite(history, "clean_run_1")
    assert len(violations) == 1
    assert "epoch 2" in violations[0]


def test_check_history_all_finite_empty_for_clean_history():
    history = [{"epoch": 1, "finite_loss": True}, {"epoch": 2, "finite_loss": True}]
    assert gpu_gate.check_history_all_finite(history, "clean_run_1") == []


def test_compare_clean_runs_matching_hashes():
    """Required test 14 (comparison logic only -- real same-seed GPU runs are
    not required to be bit-identical, see gpu_gate.compare_clean_runs)."""

    h1 = [{"epoch": 1, "train_feature_nll": 1.0, "validation_feature_nll": 2.0},
          {"epoch": 2, "train_feature_nll": 0.9, "validation_feature_nll": 1.9}]
    h2 = [{"epoch": 1, "train_feature_nll": 1.0, "validation_feature_nll": 2.0},
          {"epoch": 2, "train_feature_nll": 0.9, "validation_feature_nll": 1.9}]
    comparison = gpu_gate.compare_clean_runs(h1, h2, "abc123", "abc123")
    assert comparison["history_length_match"] is True
    assert comparison["sample_hash_match"] is True
    assert comparison["final_train_nll_abs_delta"] == pytest.approx(0.0)


def test_compare_clean_runs_mismatched_hashes_is_not_an_error():
    h1 = [{"epoch": 1, "train_feature_nll": 1.0, "validation_feature_nll": 2.0}]
    h2 = [{"epoch": 1, "train_feature_nll": 1.0001, "validation_feature_nll": 2.0001}]
    comparison = gpu_gate.compare_clean_runs(h1, h2, "abc123", "def456")
    assert comparison["sample_hash_match"] is False
    assert comparison["history_length_match"] is True


def test_oom_error_propagates_and_is_recorded_not_silently_retried(tmp_path):
    """Required test 10: OOM does not silently change batch size.

    check_train_stage_result must surface a raised training exception (e.g. a
    CUDA OOM) as a stop-condition violation rather than the orchestrator
    silently shrinking batch_size and retrying.
    """

    telemetry_path = tmp_path / "telemetry.json"
    telemetry_path.write_text(
        '{"device": "cuda", "initial": {}, "peak": {}, "final": {}, '
        '"error": "RuntimeError: CUDA out of memory."}',
        encoding="utf-8",
    )
    run_result = {"returncode": 1, "stdout": "", "stderr": "", "ledger_entry": {}}
    violations = gpu_gate.check_train_stage_result("clean_run_1", run_result, telemetry_path)
    assert any("CUDA out of memory" in v for v in violations)


def test_gpu_gate_cli_has_only_gpu_gate_and_status_commands():
    """Required test 20: no HPO command is active yet."""

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_afterms_d9_gpu_hpo", REPO_ROOT / "scripts" / "run_afterms_d9_gpu_hpo.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parser = module.build_parser()
    subparsers_action = next(
        a for a in parser._subparsers._group_actions if hasattr(a, "choices")
    )
    assert set(subparsers_action.choices.keys()) == {"gpu-gate", "status"}
