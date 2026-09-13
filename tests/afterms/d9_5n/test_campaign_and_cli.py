"""Full synthetic multi-night campaign drain, read-only CLI commands
(status/tail never write), PID-specific stop-after-epoch/abort (never a
by-name kill), and genuine OS-level detached-process launch/survival.

Subprocess-based CLI tests always pass --artifact-root pointing at a tmp_path
canary directory, never the real production artifact root, and never pass
--execute against real training. The one real production artifact read here
(the scout report) is a small, already-committed, read-only JSON file, not
test data and not a GPU run.
"""

from __future__ import annotations

import subprocess
import sys
import time as time_module
from pathlib import Path

import pytest

from ship_muon_bg.afterms.d9_5 import model_adapter as ma
from ship_muon_bg.afterms.d9_5 import nightly_runner as nr

from . import _fake_subprocess as fsp

REPO_ROOT = Path(__file__).resolve().parents[3]
NIGHTLY_SCRIPT = REPO_ROOT / "scripts" / "run_afterms_d9_5_nightly.py"
FIT_SCRIPT = REPO_ROOT / "scripts" / "run_afterms_d9_5_model_family_arena.py"
PYTHON_EXE = REPO_ROOT / ".venv" / "Scripts" / "python.exe"

PDG_BY_TRACK = {"TRK_PDG13_UW_ID": 13, "TRK_PDGM13_UW_ID": -13}


def _run_nightly_cli(*args, artifact_root):
    cli_args = ["--artifact-root", str(artifact_root), *args]
    return subprocess.run(
        [sys.executable, str(NIGHTLY_SCRIPT), *cli_args], cwd=str(REPO_ROOT), capture_output=True, text=True,
    )


def _config(artifact_root, popen_fn, now_fn, **overrides):
    return nr.SupervisorConfig(
        artifact_root=artifact_root, repo_root=REPO_ROOT, python_exe=PYTHON_EXE, script_path=FIT_SCRIPT,
        device="cpu", duration_hours=8.0, soft_stop_hours=7.5, popen_fn=popen_fn, now_fn=now_fn,
        retry_delay_seconds=0.0, epoch_seconds_default=0.001, **overrides,
    )


def test_full_synthetic_multi_night_campaign_drains_queue_in_order_no_duplicate_epochs(
    tmp_path, tiny_scout_report, tiny_shard_dir, tiny_nf_execution_policy,
    tiny_nf_optimizer_settings, tiny_evaluation_policy,
):
    nr.init_nightly_runner(tmp_path, scout_report_path=tiny_scout_report)
    execution_policy = dict(tiny_nf_execution_policy, maximum_epochs=1, minimum_epochs=1, early_stopping_patience=5)

    clock = fsp.ManualClock(time_module.time())
    base_popen = fsp.make_real_tiny_fit_popen(
        artifact_root=tmp_path, execution_policy=execution_policy, optimizer_settings=tiny_nf_optimizer_settings,
        evaluation_policy=tiny_evaluation_policy, pdg_value_by_track=PDG_BY_TRACK, shard_dir=tiny_shard_dir,
        scout_report_path=tiny_scout_report, now_fn=clock.now,
    )
    # Each simulated "night" (block) advances the clock far past its own
    # hard deadline right after its one run completes, so calling
    # run_supervisor_block repeatedly genuinely simulates several short
    # deadline-bounded blocks back-to-back rather than one block draining
    # the whole queue.
    popen_fn = fsp.advance_clock_after(base_popen, clock, 999999.0)

    seen_order = []
    for _night in range(len(nr.FROZEN_QUEUE) + 1):  # +1 headroom: must finish within the frozen queue length
        items = nr.reconcile_queue_and_persist(tmp_path)
        current = nr.determine_current_run(items, tmp_path)
        if current is None:
            break
        seen_order.append((current["track_id"], current["seed"]))
        config = _config(tmp_path, popen_fn, clock.now)
        block = nr.run_supervisor_block(config)
        assert block["state"] in (nr.BLOCK_COMPLETED, nr.BLOCK_PLANNED, nr.BLOCK_RUNNING) or block["items"]

    assert seen_order == list(nr.FROZEN_QUEUE)

    items = nr.reconcile_queue_and_persist(tmp_path)
    assert nr.determine_current_run(items, tmp_path) is None
    assert all(it["run_state"] == nr.RUN_COMPLETED for it in items)

    for item in items:
        history_path = tmp_path / item["run_dir"] / "histories" / "training_history.json"
        history = ma.read_json(history_path)
        epochs = [rec["epoch"] for rec in history]
        assert epochs == sorted(set(epochs))  # no duplicate epochs anywhere
        assert epochs == list(range(1, epochs[-1] + 1))

    # Every block acquired and released its own lock in turn -- no leftover.
    assert not (nr.nightly_root(tmp_path) / "locks" / "supervisor.lock").exists()


@pytest.mark.local_env
def test_status_and_tail_never_write_any_file(tmp_path):
    init_result = _run_nightly_cli("init", artifact_root=tmp_path)
    assert init_result.returncode == 0, init_result.stderr

    root = nr.nightly_root(tmp_path)
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "block_dummy_run.log"
    log_path.write_text("hello\nworld\n", encoding="utf-8")

    before_files = sorted(p for p in root.rglob("*") if p.is_file())
    before_mtimes = {p: p.stat().st_mtime for p in before_files}

    status_result = _run_nightly_cli("status", artifact_root=tmp_path)
    assert status_result.returncode == 0, status_result.stderr
    tail_result = _run_nightly_cli("tail", "--lines", "5", artifact_root=tmp_path)
    assert tail_result.returncode == 0, tail_result.stderr
    assert "world" in tail_result.stdout

    after_files = sorted(p for p in root.rglob("*") if p.is_file())
    after_mtimes = {p: p.stat().st_mtime for p in after_files}
    assert before_files == after_files
    assert before_mtimes == after_mtimes


@pytest.mark.local_env
def test_init_via_cli_is_idempotent(tmp_path):
    first = _run_nightly_cli("init", artifact_root=tmp_path)
    assert first.returncode == 0, first.stderr
    second = _run_nightly_cli("init", artifact_root=tmp_path)
    assert second.returncode == 0, second.stderr
    assert "noop_already_initialized" in second.stdout


def test_stop_after_epoch_is_graceful_and_pid_specific(tmp_path):
    _run_nightly_cli("init", artifact_root=tmp_path)
    lock_path = nr._lock_path(tmp_path)
    lock_data = nr.acquire_supervisor_lock(lock_path, block_id="block_x", repo_path=tmp_path)

    result = _run_nightly_cli("stop-after-epoch", artifact_root=tmp_path)
    assert result.returncode == 0, result.stderr

    flag_path = nr._stop_flag_path(tmp_path)
    assert flag_path.exists()
    flag_data = ma.read_json(flag_path)
    assert flag_data["target_pid"] == lock_data["pid"]  # PID-specific, not a broad kill
    nr.release_supervisor_lock(lock_path)


def test_stop_after_epoch_requires_a_live_lock(tmp_path):
    _run_nightly_cli("init", artifact_root=tmp_path)
    result = _run_nightly_cli("stop-after-epoch", artifact_root=tmp_path)
    assert result.returncode != 0


def test_no_broad_process_kill_mechanism_anywhere_in_source():
    nightly_module_src = (REPO_ROOT / "src" / "ship_muon_bg" / "afterms" / "d9_5" / "nightly_runner.py").read_text(encoding="utf-8")
    cli_src = NIGHTLY_SCRIPT.read_text(encoding="utf-8")
    for forbidden in ("taskkill", "Stop-Process", "/IM ", "-Name "):
        assert forbidden not in nightly_module_src
        assert forbidden not in cli_src


@pytest.mark.local_env
def test_abort_terminates_only_the_exact_recorded_pids(tmp_path):
    _run_nightly_cli("init", artifact_root=tmp_path)

    supervisor_proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    child_proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    bystander_proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        run_id = "TRK_PDG13_UW_ID/FAKE_MODEL/seed_1"
        lock_path = nr._lock_path(tmp_path)
        nr.acquire_supervisor_lock(lock_path, block_id="block_y", repo_path=tmp_path, active_run_id=run_id, pid=supervisor_proc.pid)
        nr._append_csv_row(nr.nightly_root(tmp_path) / "process_ledger.csv", nr.PROCESS_LEDGER_HEADER, {
            "timestamp": time_module.time(), "block_id": "block_y", "run_id": run_id, "event": "started",
            "pid": child_proc.pid, "command": "fake", "exit_code": "", "retry_count": 0,
        })

        assert nr.is_pid_alive(bystander_proc.pid)
        result = _run_nightly_cli("abort", artifact_root=tmp_path)
        assert result.returncode == 0, result.stderr

        time_module.sleep(0.5)
        assert not nr.is_pid_alive(supervisor_proc.pid)
        assert not nr.is_pid_alive(child_proc.pid)
        assert nr.is_pid_alive(bystander_proc.pid)  # never touched -- PID-specific only
        assert not lock_path.exists()
    finally:
        for proc in (supervisor_proc, child_proc, bystander_proc):
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                pass


def test_abort_requires_a_live_lock(tmp_path):
    _run_nightly_cli("init", artifact_root=tmp_path)
    result = _run_nightly_cli("abort", artifact_root=tmp_path)
    assert result.returncode != 0


@pytest.mark.local_env
def test_detached_process_survives_conceptual_parent_exit_and_is_pid_terminable(tmp_path):
    """Proves the exact OS-level mechanism `start` uses (DETACHED_PROCESS |
    CREATE_NEW_PROCESS_GROUP, no wait()) genuinely detaches on this Windows
    machine -- a trivial dummy sleep command, not the real fit CLI, to keep
    this fast and free of any GPU/production dependency."""

    creationflags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    log_path = tmp_path / "detached_dummy.log"
    with open(log_path, "ab") as log_handle:
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(15)"],
            cwd=str(REPO_ROOT), creationflags=creationflags, close_fds=True,
            stdin=subprocess.DEVNULL, stdout=log_handle, stderr=subprocess.STDOUT,
        )
    try:
        # Do NOT wait() -- simulate the launching ("parent") process moving
        # on / conceptually exiting immediately, as `start` does.
        time_module.sleep(1.0)
        assert nr.is_pid_alive(proc.pid)  # still alive with no parent waiting on it
        start_time = nr.get_process_start_time(proc.pid)
        assert start_time is not None
    finally:
        nr.terminate_pid(proc.pid)
        time_module.sleep(0.5)
        assert not nr.is_pid_alive(proc.pid)


@pytest.mark.local_env
def test_canary_artifact_root_never_touches_production_artifact_root(tmp_path):
    canary_root = tmp_path / "afterms_d9_5_nightly_canary_v0"
    production_root = REPO_ROOT / "artifacts" / "afterms_d9_5_model_family_arena_v0"
    before = production_root.exists() and sorted(production_root.rglob("*"))

    result = _run_nightly_cli("init", artifact_root=canary_root)
    assert result.returncode == 0, result.stderr

    assert (nr.nightly_root(canary_root) / "run_queue.json").exists()
    if before:
        assert sorted(production_root.rglob("*")) == before
