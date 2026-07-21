"""Focused regression tests for the after-MS shard-build progress/heartbeat,
subprocess UTF-8 hardening, Ctrl+C/KeyboardInterrupt cleanup, and atomic
shard-artifact writes (see docs/reviews/afterms_manual_execution_runbook_v0.md
for the incident these guard against: a silent-looking but genuinely-
progressing job 01, interrupted by the user, that left a stale
queue_state.json "active" entry).

Uses only a tiny synthetic PKL fixture -- never the real
muonsFullMC_afterMS.pkl -- and never runs neural training.
"""

import gzip
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import scripts.build_afterms_shards as build_shards_mod
import scripts.run_afterms_nightly_queue as queue_mod


def _write_tiny_afterms_pkl(path, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    px = rng.normal(size=n)
    py = rng.normal(size=n)
    pz = rng.uniform(0.1, 5.0, size=n)
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    z = rng.normal(size=n)
    ids = rng.choice([13.0, -13.0], size=n)
    w = rng.uniform(0.1, 1.0, size=n)
    arr = np.column_stack((px, py, pz, x, y, z, ids, w)).astype(np.float64)
    with gzip.open(path, "wb") as f:
        pickle.dump(arr, f)
    return arr


# --- 6. Shard builder emits flushed progress on a tiny fixture ---


def test_build_afterms_shards_tiny_fixture_emits_progress_and_completes(tmp_path):
    raw_file = tmp_path / "tiny_afterms.pkl.gz"
    _write_tiny_afterms_pkl(raw_file, n=3000)

    shard_dir = tmp_path / "shards"
    artifact_dir = tmp_path / "artifacts"

    proc = subprocess.run(
        [
            sys.executable, "-u",
            str(REPO_ROOT / "scripts" / "build_afterms_shards.py"),
            "--raw-file", str(raw_file),
            "--shard-dir", str(shard_dir),
            "--artifact-dir", str(artifact_dir),
            "--target-rows", "1000",
            "--job-name", "01_build_afterms_shards",
        ],
        capture_output=True, text=True, timeout=120, cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Loaded dataset" in proc.stdout
    assert "Shard construction completed successfully" in proc.stdout

    progress_path = artifact_dir / "jobs" / "01_build_afterms_shards" / "progress.json"
    assert progress_path.exists()
    progress = json.loads(progress_path.read_text())
    assert progress["phase"] == "shard_construction_completed"
    assert "heartbeat" in progress
    assert progress["total_elapsed_seconds"] >= 0.0

    assert (shard_dir / "shard_manifest.json").exists()
    # Atomic writes must never leave a stray .tmp file behind on success.
    assert list(shard_dir.glob("*.tmp")) == []


# --- 7. Parent reads UTF-8 child output correctly ---


def test_subprocess_pipe_reads_non_ascii_child_output_without_crashing():
    child_env = dict(os.environ)
    child_env["PYTHONUTF8"] = "1"
    child_env["PYTHONIOENCODING"] = "utf-8"

    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", "print('caf\\u00e9 \\u00f1 \\u65e5\\u672c\\u8a9e')"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=child_env,
    )
    line = proc.stdout.readline()
    proc.wait(timeout=10)
    assert proc.returncode == 0
    assert "café" in line
    assert "日本語" in line


# --- 9/10. Interrupted child is terminated; no child process remains alive ---


def test_terminate_process_tree_kills_a_real_child_process():
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )
    assert proc.poll() is None

    queue_mod._terminate_process_tree(proc, grace_seconds=10.0)

    assert proc.poll() is not None


def test_terminate_process_tree_is_idempotent_on_already_exited_process():
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait(timeout=10)
    assert proc.poll() is not None

    # Must not raise/hang when the child already exited.
    queue_mod._terminate_process_tree(proc, grace_seconds=2.0)


# --- 8/11. Ctrl+C writes "interrupted" (not "failed") and clears queue_state ---


class _FakeStdout:
    def __init__(self):
        self.closed = False
        self._raised = False

    def readline(self):
        if not self._raised:
            self._raised = True
            raise KeyboardInterrupt
        return ""

    def close(self):
        self.closed = True


class _FakeInterruptedProc:
    def __init__(self, pid=4242424):
        self.pid = pid
        self.stdout = _FakeStdout()
        self._alive = True

    def poll(self):
        return None if self._alive else 0

    def wait(self, timeout=None):
        self._alive = False
        return 0

    def send_signal(self, sig):
        pass

    def communicate(self):
        return ("", None)


class _Args:
    def __init__(self, artifact_dir, shard_dir, device="cpu", dry_run=False, resume=False, force_job=None):
        self.artifact_dir = str(artifact_dir)
        self.shard_dir = str(shard_dir)
        self.device = device
        self.dry_run = dry_run
        self.resume = resume
        self.force_job = force_job


def test_run_job_outer_keyboard_interrupt_writes_interrupted_status_and_clears_queue_state(tmp_path, monkeypatch):
    artifact_dir = tmp_path / "artifacts"
    shard_dir = tmp_path / "shards"
    args = _Args(artifact_dir, shard_dir)

    real_popen = queue_mod.subprocess.Popen

    def _fake_popen(cmd, *a, **kw):
        # Only fake the actual job-dispatch Popen call; let incidental
        # subprocess use inside pre/post metrics collection (nvidia-smi
        # probes) run for real, since those already fail closed via their
        # own try/except.
        if isinstance(cmd, list) and "--run-job" in cmd:
            return _FakeInterruptedProc()
        return real_popen(cmd, *a, **kw)

    monkeypatch.setattr(queue_mod.subprocess, "Popen", _fake_popen)

    with pytest.raises(KeyboardInterrupt):
        queue_mod.run_job_outer("00_environment_and_dataset_smoke", args, {"dataset_hash": "abc"}, "deadbeef")

    status_path = artifact_dir / "jobs" / "00_environment_and_dataset_smoke" / "status.json"
    status = json.loads(status_path.read_text())
    assert status["status"] == "interrupted"
    assert status["status"] != "failed"

    queue_state = json.loads((artifact_dir / "queue_state.json").read_text())
    assert queue_state["active_job"] is None
    assert queue_state["pid"] is None


# --- 12. Exit code is 130 on KeyboardInterrupt ---


def test_main_exits_with_130_on_keyboard_interrupt(tmp_path, monkeypatch):
    monkeypatch.setattr(
        queue_mod, "run_job_outer",
        lambda *a, **kw: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    argv = [
        "run_afterms_nightly_queue.py",
        "--jobs", "00_environment_and_dataset_smoke",
        "--artifact-dir", str(tmp_path / "artifacts"),
        "--shard-dir", str(tmp_path / "shards"),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc_info:
        queue_mod.main()
    assert exc_info.value.code == 130


# --- 14. Incomplete temporary shard artifacts are not treated as completed ---


def test_atomic_save_npy_never_leaves_a_partial_target_file(tmp_path, monkeypatch):
    def bad_save(handle, array, **kwargs):
        handle.write(b"PARTIAL")
        raise RuntimeError("simulated interruption mid-write")

    monkeypatch.setattr(build_shards_mod.np, "save", bad_save)
    target = tmp_path / "shard.npy"
    with pytest.raises(RuntimeError):
        build_shards_mod._atomic_save_npy(str(target), np.zeros(3))
    assert not target.exists()


def test_atomic_write_json_never_leaves_a_partial_target_file(tmp_path, monkeypatch):
    def bad_dump(obj, handle, **kwargs):
        handle.write("{PARTIAL")
        raise RuntimeError("simulated interruption mid-write")

    monkeypatch.setattr(build_shards_mod.json, "dump", bad_dump)
    target = tmp_path / "manifest.json"
    with pytest.raises(RuntimeError):
        build_shards_mod._atomic_write_json(str(target), {"a": 1})
    assert not target.exists()


# --- 15. Resume reruns an interrupted job (only "completed" is skip-worthy) ---


def test_resume_does_not_skip_a_previously_interrupted_job(tmp_path, monkeypatch):
    artifact_dir = tmp_path / "artifacts"
    shard_dir = tmp_path / "shards"
    job_name = "00_environment_and_dataset_smoke"
    job_dir = artifact_dir / "jobs" / job_name
    job_dir.mkdir(parents=True)

    config_payload = {"job_name": job_name, "device": "cpu", "seed": 20260720}
    import hashlib
    job_config_hash = hashlib.sha256(json.dumps(config_payload, sort_keys=True).encode("utf-8")).hexdigest()

    (job_dir / "status.json").write_text(json.dumps({
        "status": "interrupted",
        "job_name": job_name,
        "job_config_hash": job_config_hash,
        "dataset_hash": "abc",
        "git_commit": "deadbeef",
    }))

    args = _Args(artifact_dir, shard_dir, resume=True)
    job_dispatch_calls = []
    real_popen = queue_mod.subprocess.Popen

    def _fake_popen(cmd, *a, **kw):
        # get_gpu_memory_info()/get_active_cuda_processes() also invoke
        # subprocess internally (via nvidia-smi); only the actual job
        # dispatch call (running this same script with --run-job) matters
        # for the resume-skip assertion below.
        if isinstance(cmd, list) and "--run-job" in cmd:
            job_dispatch_calls.append(cmd)
            return _FakeInterruptedProc()
        return real_popen(cmd, *a, **kw)

    monkeypatch.setattr(queue_mod.subprocess, "Popen", _fake_popen)

    with pytest.raises(KeyboardInterrupt):
        queue_mod.run_job_outer(job_name, args, {"dataset_hash": "abc"}, "deadbeef")

    # An interrupted status must not satisfy the resume-skip check: the
    # subprocess must actually have been (re)launched, not silently skipped.
    assert len(job_dispatch_calls) == 1


# --- Job 13 (nightly report) queue-loop dispatch: it runs in-process (no
# subprocess), so it must create its own job_dir and clear queue_state.json
# exactly like every other job. This guards the "Job 13 failed: [Errno 2] No
# such file or directory: .../13_build_nightly_report/status.json" incident,
# where jobs/13_build_nightly_report/ was never created before the queue
# loop tried to write status.json into it. ---


def _write_smoke_jobs_completed(artifact_dir):
    for name in queue_mod.NIGHTLY_JOB_NAMES:
        if name == "13_build_nightly_report":
            continue
        job_dir = os.path.join(artifact_dir, "jobs", name)
        os.makedirs(job_dir, exist_ok=True)
        with open(os.path.join(job_dir, "status.json"), "w") as f:
            json.dump({"status": "completed"}, f)


def test_queue_loop_job13_success_clears_active_job_and_pid(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    shard_dir = tmp_path / "shards"
    _write_smoke_jobs_completed(str(artifact_dir))

    argv = [
        "run_afterms_nightly_queue.py",
        "--jobs", "13_build_nightly_report",
        "--artifact-dir", str(artifact_dir),
        "--shard-dir", str(shard_dir),
    ]
    import sys as sys_mod
    old_argv = sys_mod.argv
    sys_mod.argv = argv
    try:
        assert queue_mod.main() == 0
    finally:
        sys_mod.argv = old_argv

    status = json.loads((artifact_dir / "jobs" / "13_build_nightly_report" / "status.json").read_text())
    assert status["status"] == "completed"

    queue_state = json.loads((artifact_dir / "queue_state.json").read_text())
    assert queue_state["active_job"] is None
    assert queue_state["pid"] is None
    assert "13_build_nightly_report" in queue_state["completed_jobs"]


def test_queue_loop_job13_failure_clears_active_job_and_pid_and_marks_failed(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    shard_dir = tmp_path / "shards"
    # A metrics.json missing a required numeric key makes the markdown
    # renderer's `:.4f` format spec raise on a real value (None) -- a
    # genuine report-generation failure, not a mocked one.
    bad_job_dir = artifact_dir / "jobs" / "05_affine_preprocessing_ab_pdg13"
    bad_job_dir.mkdir(parents=True)
    (bad_job_dir / "status.json").write_text(json.dumps({"status": "completed"}))
    (bad_job_dir / "metrics.json").write_text(json.dumps({
        "identity_standardized_v0_affine_small_unweighted": {"metrics": {
            "physical_space_nll": 0.5, "wall_time_seconds": 1.0, "parameter_count": 10,
        }}
    }))

    argv = [
        "run_afterms_nightly_queue.py",
        "--jobs", "13_build_nightly_report",
        "--artifact-dir", str(artifact_dir),
        "--shard-dir", str(shard_dir),
    ]
    import sys as sys_mod
    old_argv = sys_mod.argv
    sys_mod.argv = argv
    try:
        with pytest.raises(SystemExit) as exc_info:
            queue_mod.main()
        assert exc_info.value.code == 1
    finally:
        sys_mod.argv = old_argv

    status = json.loads((artifact_dir / "jobs" / "13_build_nightly_report" / "status.json").read_text())
    assert status["status"] == "failed"

    queue_state = json.loads((artifact_dir / "queue_state.json").read_text())
    assert queue_state["active_job"] is None
    assert queue_state["pid"] is None
    assert "13_build_nightly_report" in queue_state["failed_jobs"]
