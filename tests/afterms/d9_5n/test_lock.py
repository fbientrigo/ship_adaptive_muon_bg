"""Supervisor lock: stale-lock detection after simulated reboot, live-lock
preservation (refuse a second supervisor), PID creation-time verification
(reused pid is not falsely treated as live), and atomic writes.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time

import pytest

from ship_muon_bg.afterms.d9_5 import model_adapter as ma
from ship_muon_bg.afterms.d9_5 import nightly_runner as nr


def _unused_pid() -> int:
    """A PID that is (overwhelmingly likely to be) not alive."""

    candidate = 999_999
    while nr.is_pid_alive(candidate) and candidate > 2:
        candidate -= 1
    return candidate


def test_lock_write_is_atomic_temp_file_replace(tmp_path):
    lock_path = tmp_path / "locks" / "supervisor.lock"
    nr.acquire_supervisor_lock(lock_path, block_id="block_a", repo_path=tmp_path)
    # No leftover temp file from ma.atomic_write_json's mkstemp+replace pattern.
    leftovers = list(lock_path.parent.glob(".tmp_*"))
    assert leftovers == []
    assert lock_path.exists()
    data = ma.read_json(lock_path)
    assert data["pid"] == __import__("os").getpid()


def test_stale_lock_is_detected_after_simulated_reboot(tmp_path):
    lock_path = tmp_path / "locks" / "supervisor.lock"
    dead_pid = _unused_pid()
    ma.atomic_write_json(lock_path, {
        "pid": dead_pid, "process_start_time": 12345.0, "block_id": "block_old",
        "hostname": "old-host", "repo_path": str(tmp_path), "active_run_id": None,
    })
    assert nr.is_lock_live(ma.read_json(lock_path)) is False

    result = nr.reconcile_lock(lock_path)
    assert result["action"] == "removed_stale_lock"
    assert not lock_path.exists()


def test_live_lock_is_preserved_and_refuses_second_start(tmp_path):
    lock_path = tmp_path / "locks" / "supervisor.lock"
    nr.acquire_supervisor_lock(lock_path, block_id="block_a", repo_path=tmp_path)

    # reconcile must never remove a verified-live lock
    result = nr.reconcile_lock(lock_path)
    assert result["action"] == "none"
    assert lock_path.exists()

    with pytest.raises(nr.SupervisorAlreadyRunningError):
        nr.acquire_supervisor_lock(lock_path, block_id="block_b", repo_path=tmp_path)


def test_live_lock_with_real_child_process_refuses_second_start(tmp_path):
    """One supervisor maximum, verified against a genuinely running OS
    process (not just our own pid)."""

    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(20)"], stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        lock_path = tmp_path / "locks" / "supervisor.lock"
        nr.acquire_supervisor_lock(lock_path, block_id="block_a", repo_path=tmp_path, pid=proc.pid)
        with pytest.raises(nr.SupervisorAlreadyRunningError):
            nr.acquire_supervisor_lock(lock_path, block_id="block_b", repo_path=tmp_path)
    finally:
        proc.terminate()
        proc.wait(timeout=5)


@pytest.mark.local_env
def test_pid_reuse_with_mismatched_start_time_is_not_treated_as_live(tmp_path):
    """A stale lock's pid may have been reused by an unrelated process after
    a reboot; process_start_time mismatch must catch this, not just pid
    liveness."""

    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(20)"], stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        real_start_time = nr.get_process_start_time(proc.pid)
        lock_data = {
            "pid": proc.pid,
            # Deliberately wrong process_start_time, as if this pid used to
            # belong to a different (already-exited) supervisor process.
            "process_start_time": (real_start_time or 0.0) + 999999.0,
            "block_id": "block_old", "hostname": "old-host", "repo_path": str(tmp_path), "active_run_id": None,
        }
        assert nr.is_lock_live(lock_data) is False
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_release_lock_only_removes_own_pid(tmp_path):
    lock_path = tmp_path / "locks" / "supervisor.lock"
    ma.atomic_write_json(lock_path, {
        "pid": _unused_pid() - 1 if _unused_pid() > 3 else 3, "process_start_time": 1.0,
        "block_id": "block_x", "hostname": "h", "repo_path": str(tmp_path), "active_run_id": None,
    })
    # release_supervisor_lock must not remove a lock belonging to a
    # different pid than the current process.
    nr.release_supervisor_lock(lock_path)
    assert lock_path.exists()
