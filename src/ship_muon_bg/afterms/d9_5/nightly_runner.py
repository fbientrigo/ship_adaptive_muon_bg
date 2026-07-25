"""D9-5 Gate D nightly execution supervisor (pure orchestration).

This module is plumbing wrapped AROUND the frozen D9-5 NF_AC training code
(``ship_muon_bg.afterms.d9.runner.train_candidate_seed`` via
``scripts/run_afterms_d9_5_model_family_arena.py fit``). It never touches
scientific hyperparameters, never opens test-split data, and never trains
anything itself -- it only launches/monitors/retries the existing frozen
``fit`` CLI as a subprocess, once per epoch-boundary-safe block, so a human
can run one command per night and let the machine resume automatically.

Schema id: ``d9_5_nightly_runner_state_v0``.

Runtime state root (created by ``init_nightly_runner``)::

    artifacts/afterms_d9_5_model_family_arena_v0/nightly_runner/
        incidents/
        logs/
        nightly_blocks/
        locks/
        campaign_state.json
        run_queue.json
        process_ledger.csv
        resource_ledger.csv

The source of truth for "what state is run N in" is always the on-disk
``status.json`` (+ checkpoint files) for that run directory, re-derived every
time the supervisor starts -- ``campaign_state.json``/``run_queue.json`` are
caches, never trusted blindly.
"""

from __future__ import annotations

import csv
import ctypes
import os
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ship_muon_bg.afterms.d9 import runner as d9runner

from . import model_adapter as ma
from . import nf_ac_adapter

SCHEMA_ID = "d9_5_nightly_runner_state_v0"
MODEL_FAMILY = "NF_AC"

# Frozen production queue (exact order, never reordered).
FROZEN_QUEUE = (
    ("TRK_PDG13_UW_ID", 20260720),
    ("TRK_PDG13_UW_ID", 20260721),
    ("TRK_PDG13_UW_ID", 20260722),
    ("TRK_PDGM13_UW_ID", 20260720),
    ("TRK_PDGM13_UW_ID", 20260721),
    ("TRK_PDGM13_UW_ID", 20260722),
)

# Measured epoch wall time on the real production scope (Gate D runbook).
DEFAULT_EPOCH_SECONDS_ESTIMATE = 1011.8

DEADLINE_MARKER_FILENAME = ".nightly_runner_deadline_interrupt.json"

CPU_THREAD_ENV_VARS = {
    "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
    "NUMEXPR_NUM_THREADS": "2",
}

# --- run states -------------------------------------------------------

RUN_PENDING = "PENDING"
RUN_STARTING = "STARTING"
RUN_RUNNING = "RUNNING"
RUN_INTERRUPTED_AT_BLOCK_DEADLINE = "INTERRUPTED_AT_BLOCK_DEADLINE"
RUN_COMPLETED = "COMPLETED"
RUN_FAILED_RETRYABLE = "FAILED_RETRYABLE"
RUN_FAILED_BLOCKED = "FAILED_BLOCKED"

# --- block states -------------------------------------------------------

BLOCK_PLANNED = "PLANNED"
BLOCK_RUNNING = "RUNNING"
BLOCK_COMPLETED = "COMPLETED"
BLOCK_COMPLETED_WITH_RETRY = "COMPLETED_WITH_RETRY"
BLOCK_BLOCKED = "BLOCKED"
BLOCK_ABORTED_BY_USER = "ABORTED_BY_USER"

GATE_D_COMPLETE_MARKER = "GATE_D_COMPLETE_READY_FOR_VALIDATION_REVIEW"

PROCESS_LEDGER_HEADER = ["timestamp", "block_id", "run_id", "event", "pid", "command", "exit_code", "retry_count"]
RESOURCE_LEDGER_HEADER = ["timestamp", "block_id", "event", "detail"]

BLOCKING_MARKERS = (
    "cuda out of memory",
    "outofmemoryerror",
    "checkpointcompatibilityerror",
    "incompatible with the expected training identity",
    "hash mismatch",
    "filenotfounderror",
    "no such file or directory",
    "data/shards",
    "data\\shards",
)


class SupervisorAlreadyRunningError(RuntimeError):
    """A live supervisor lock already exists; refuse to start a second one."""


class IncompatibleQueueError(RuntimeError):
    """An on-disk run_queue.json does not match the frozen production queue."""


# --- small filesystem helpers -------------------------------------------


def nightly_root(artifact_root: Path) -> Path:
    return Path(artifact_root) / "nightly_runner"


def _lock_path(artifact_root: Path) -> Path:
    return nightly_root(artifact_root) / "locks" / "supervisor.lock"


def _stop_flag_path(artifact_root: Path) -> Path:
    return nightly_root(artifact_root) / "locks" / "stop_requested.flag"


def _read_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return None
    return ma.read_json(path)


def _append_csv_row(path: Path, header: List[str], row: Dict[str, Any]) -> None:
    """Atomic (temp-file + os.replace) rewrite-with-append. Ledgers stay small
    (one row per subprocess start/exit/retry across a handful of nightly
    blocks), so rewriting the whole file per append is simple and safe."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    rows.append({k: row.get(k, "") for k in header})
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".tmp_{path.name}_", suffix=".csv")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with tmp_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# --- pid / process-identity helpers (psutil if available, else ctypes) --


def _psutil():
    try:
        import psutil

        return psutil
    except ImportError:
        return None


def _posix_pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False
    except Exception:
        return False


def _configured_kernel32():
    """``kernel32`` with OpenProcess's return type declared as HANDLE.

    Without an explicit ``restype``, ctypes assumes a 32-bit ``c_int``
    return value; on 64-bit Windows a real HANDLE is 64 bits, so the pointer
    gets silently truncated -- the handle can look "truthy" for existence
    checks by luck, but is invalid for TerminateProcess/GetProcessTimes.
    """

    from ctypes import wintypes

    kernel32 = ctypes.windll.kernel32
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel32.GetProcessTimes.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME),
    ]
    kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    return kernel32


_STILL_ACTIVE = 259


def _win_process_start_time(pid: int) -> Optional[float]:
    if not hasattr(ctypes, "windll"):
        return None
    from ctypes import wintypes

    process_query_limited_information = 0x1000
    kernel32 = _configured_kernel32()
    handle = kernel32.OpenProcess(process_query_limited_information, False, int(pid))
    if not handle:
        return None
    try:
        creation = wintypes.FILETIME()
        exit_time = wintypes.FILETIME()
        kernel_time = wintypes.FILETIME()
        user_time = wintypes.FILETIME()
        ok = kernel32.GetProcessTimes(
            handle, ctypes.byref(creation), ctypes.byref(exit_time),
            ctypes.byref(kernel_time), ctypes.byref(user_time),
        )
        if not ok:
            return None
        value = (creation.dwHighDateTime << 32) | creation.dwLowDateTime
        if value == 0:
            return None
        windows_epoch_offset_seconds = 11644473600
        return value / 1e7 - windows_epoch_offset_seconds
    finally:
        kernel32.CloseHandle(handle)


def is_pid_alive(pid: int) -> bool:
    psutil = _psutil()
    if psutil is not None:
        try:
            return bool(psutil.pid_exists(int(pid)))
        except Exception:
            pass
    if hasattr(ctypes, "windll"):
        from ctypes import wintypes

        process_query_limited_information = 0x1000
        kernel32 = _configured_kernel32()
        handle = kernel32.OpenProcess(process_query_limited_information, False, int(pid))
        if not handle:
            return False
        try:
            # A PID can remain openable after the process has already
            # exited (e.g. its creator still holds a handle, keeping the
            # kernel object alive) -- OpenProcess succeeding is not itself
            # proof of liveness; GetExitCodeProcess == STILL_ACTIVE is.
            exit_code = wintypes.DWORD()
            ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            if not ok:
                return False
            return exit_code.value == _STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    return _posix_pid_alive(pid)


def get_process_start_time(pid: int) -> Optional[float]:
    psutil = _psutil()
    if psutil is not None:
        try:
            return float(psutil.Process(int(pid)).create_time())
        except Exception:
            return None
    if hasattr(ctypes, "windll"):
        return _win_process_start_time(pid)
    return None


def terminate_pid(pid: int, *, timeout: float = 5.0) -> None:
    """PID-specific termination -- never a broad by-name kill."""

    psutil = _psutil()
    if psutil is not None:
        try:
            proc = psutil.Process(int(pid))
            proc.terminate()
            try:
                proc.wait(timeout=timeout)
            except Exception:
                proc.kill()
            return
        except Exception:
            pass
    if hasattr(ctypes, "windll"):
        process_terminate = 0x0001
        kernel32 = _configured_kernel32()
        handle = kernel32.OpenProcess(process_terminate, False, int(pid))
        if handle:
            kernel32.TerminateProcess(handle, 1)
            kernel32.CloseHandle(handle)


# --- keep-awake -----------------------------------------------------------

_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


def set_keep_awake() -> bool:
    try:
        if not hasattr(ctypes, "windll"):
            return False
        result = ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED)
        return bool(result)
    except Exception:
        return False


def clear_keep_awake() -> bool:
    try:
        if not hasattr(ctypes, "windll"):
            return False
        result = ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
        return bool(result)
    except Exception:
        return False


# --- lock management --------------------------------------------------


def is_lock_live(lock_data: Optional[Dict[str, Any]]) -> bool:
    if not lock_data:
        return False
    pid = lock_data.get("pid")
    if pid is None or not is_pid_alive(pid):
        return False
    recorded_start = lock_data.get("process_start_time")
    current_start = get_process_start_time(pid)
    if recorded_start is None or current_start is None:
        # Cannot verify identity precisely on this platform; the pid-alive
        # check alone is the best available signal.
        return True
    return abs(float(current_start) - float(recorded_start)) < 2.0


def acquire_supervisor_lock(
    lock_path: Path, *, block_id: str, repo_path: Path, active_run_id: Optional[str] = None,
    pid: Optional[int] = None,
) -> Dict[str, Any]:
    lock_path = Path(lock_path)
    existing = _read_json_if_exists(lock_path)
    if existing is not None and is_lock_live(existing):
        raise SupervisorAlreadyRunningError(
            f"a supervisor is already running: pid={existing.get('pid')} block_id={existing.get('block_id')}"
        )
    pid = pid if pid is not None else os.getpid()
    data = {
        "pid": pid,
        "process_start_time": get_process_start_time(pid),
        "block_id": block_id,
        "hostname": socket.gethostname(),
        "repo_path": str(repo_path),
        "active_run_id": active_run_id,
        "acquired_at": time.time(),
    }
    ma.atomic_write_json(lock_path, data)
    return data


def release_supervisor_lock(lock_path: Path) -> None:
    """Self-release: only removes the lock if the CURRENT process holds it."""

    lock_path = Path(lock_path)
    existing = _read_json_if_exists(lock_path)
    if existing is not None and existing.get("pid") == os.getpid():
        lock_path.unlink(missing_ok=True)


def force_release_lock_after_abort(lock_path: Path) -> None:
    """Used only by ``abort``, after it has already verified the lock is
    live and terminated the exact recorded pid (never a third party
    self-releasing someone else's still-running lock)."""

    Path(lock_path).unlink(missing_ok=True)


def reconcile_lock(lock_path: Path) -> Dict[str, Any]:
    """Force resync of the lock file against reality. Never removes a
    verified-live lock."""

    existing = _read_json_if_exists(lock_path)
    if existing is None:
        return {"action": "none", "reason": "no lock file"}
    if is_lock_live(existing):
        return {"action": "none", "reason": "lock is live", "lock": existing}
    Path(lock_path).unlink(missing_ok=True)
    return {"action": "removed_stale_lock", "previous": existing}


# --- queue resolution / reconciliation ----------------------------------


def resolve_run_queue_items(artifact_root: Path, *, scout_report_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    kwargs = {} if scout_report_path is None else {"scout_report_path": scout_report_path}
    items = []
    for run_index, (track_id, seed) in enumerate(FROZEN_QUEUE):
        scout = nf_ac_adapter.select_scout_promoted_nf_config(track_id, **kwargs)
        model_config_id = scout["model_config_id"]
        run_dir = d9runner.run_directory(artifact_root, f"{track_id}/{model_config_id}", seed)
        run_id = f"{track_id}/{model_config_id}/seed_{seed}"
        items.append({
            "run_index": run_index,
            "run_id": run_id,
            "track_id": track_id,
            "model_family": MODEL_FAMILY,
            "model_config_id": model_config_id,
            "seed": seed,
            "run_dir": str(run_dir.relative_to(artifact_root)).replace("\\", "/"),
            "semantic_training_hash": None,
            "execution_policy_hash": None,
            "sampling_contract_version": None,
        })
    return items


def _queue_signature(items: List[Dict[str, Any]]) -> List[Tuple[str, int, str]]:
    return [(it["track_id"], it["seed"], it["model_config_id"]) for it in items]


def init_nightly_runner(
    artifact_root: Path, *, repo_root: Optional[Path] = None, scout_report_path: Optional[Path] = None,
) -> Dict[str, Any]:
    root = nightly_root(artifact_root)
    for sub in ("incidents", "logs", "nightly_blocks", "locks"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    queue_path = root / "run_queue.json"
    fresh_items = resolve_run_queue_items(artifact_root, scout_report_path=scout_report_path)
    fresh_signature = _queue_signature(fresh_items)

    if queue_path.exists():
        cached = ma.read_json(queue_path)
        cached_signature = _queue_signature(cached.get("items", []))
        if cached_signature != fresh_signature:
            raise IncompatibleQueueError(
                f"existing run_queue.json at {queue_path} does not match the frozen production "
                f"queue (cached={cached_signature!r} vs resolved={fresh_signature!r}); refusing to "
                "overwrite. Investigate before proceeding -- this is a blocking, hash-mismatch-style "
                "condition, not something to silently re-resolve."
            )
        return {"action": "noop_already_initialized", "queue_path": str(queue_path)}

    ma.atomic_write_json(queue_path, {"schema_id": SCHEMA_ID, "items": fresh_items})
    campaign_state_path = root / "campaign_state.json"
    if not campaign_state_path.exists():
        ma.atomic_write_json(campaign_state_path, {
            "schema_id": SCHEMA_ID, "blocks_completed": 0, "last_block_id": None, "last_block_state": None,
        })
    return {"action": "initialized", "queue_path": str(queue_path)}


def derive_run_state(run_dir: Path) -> Tuple[str, Optional[Dict[str, Any]]]:
    status_path = Path(run_dir) / "status.json"
    if not status_path.exists():
        return RUN_PENDING, None
    status = ma.read_json(status_path)
    raw_status = status.get("status")
    if raw_status == d9runner.STATUS_COMPLETED:
        return RUN_COMPLETED, status
    if raw_status == d9runner.STATUS_INTERRUPTED:
        marker_path = Path(run_dir) / DEADLINE_MARKER_FILENAME
        if marker_path.exists():
            return RUN_INTERRUPTED_AT_BLOCK_DEADLINE, status
        # "interrupted" without our marker means something other than the
        # supervisor's own deadline stopped it (manual Ctrl-C, crash before
        # the marker could be written) -- treat as retryable via --resume,
        # never as evidence to change hyperparameters.
        return RUN_FAILED_RETRYABLE, status
    if raw_status == d9runner.STATUS_RUNNING:
        # A "running" status with no live lock means a previous process died
        # mid-epoch; the last epoch's checkpoint is still valid.
        return RUN_FAILED_RETRYABLE, status
    if raw_status == d9runner.STATUS_EXCLUDED_CONTRACT_MISMATCH:
        return RUN_FAILED_BLOCKED, status
    return RUN_FAILED_RETRYABLE, status


def reconcile_queue_and_persist(artifact_root: Path) -> List[Dict[str, Any]]:
    root = nightly_root(artifact_root)
    queue_path = root / "run_queue.json"
    payload = ma.read_json(queue_path)
    changed = False
    for item in payload["items"]:
        run_dir = Path(artifact_root) / item["run_dir"]
        state, status = derive_run_state(run_dir)
        item["run_state"] = state
        if status is not None:
            for key in ("semantic_training_hash", "execution_policy_hash", "sampling_contract_version"):
                value = status.get(key)
                if value is not None and item.get(key) != value:
                    item[key] = value
                    changed = True
    if changed:
        ma.atomic_write_json(queue_path, payload)
    return payload["items"]


def determine_current_run(queue_items: List[Dict[str, Any]], artifact_root: Path) -> Optional[Dict[str, Any]]:
    """Walk the queue in order; return the first item not COMPLETED, or None
    if all items are COMPLETED (campaign done)."""

    for item in queue_items:
        run_dir = Path(artifact_root) / item["run_dir"]
        state, status = derive_run_state(run_dir)
        if state != RUN_COMPLETED:
            result = dict(item)
            result["run_state"] = state
            result["status"] = status
            return result
    return None


# --- epoch-time estimation ----------------------------------------------


def estimate_epoch_seconds(run_dir: Path, *, default: float = DEFAULT_EPOCH_SECONDS_ESTIMATE) -> float:
    history_path = Path(run_dir) / "histories" / "training_history.json"
    if not history_path.exists():
        return default
    try:
        history = ma.read_json(history_path)
    except Exception:
        return default
    if not isinstance(history, list) or not history:
        return default
    times = [rec.get("wall_time_seconds") for rec in history[-3:] if rec.get("wall_time_seconds")]
    if not times:
        return default
    return max(float(t) for t in times)


# --- subprocess launch ---------------------------------------------------


def build_fit_command(
    python_exe: Path, script_path: Path, *, artifact_root: Path, track_id: str, seed: int, device: str,
    resume: bool, deadline_timestamp: Optional[float],
) -> List[str]:
    cmd = [
        str(python_exe), str(script_path), "--artifact-root", str(artifact_root), "fit",
        "--track-id", track_id, "--model-family", MODEL_FAMILY, "--seed", str(seed),
        "--device", device, "--execute",
    ]
    if resume:
        cmd.append("--resume")
    if deadline_timestamp is not None:
        cmd += ["--deadline-timestamp", str(float(deadline_timestamp))]
    return cmd


def launch_fit_subprocess(cmd: List[str], *, log_path: Path, cwd: Path, popen_fn: Callable = subprocess.Popen):
    env = dict(os.environ)
    env.update(CPU_THREAD_ENV_VARS)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(log_path, "ab")
    process = popen_fn(cmd, cwd=str(cwd), env=env, stdout=log_handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    return process, log_handle


def classify_failure(log_text: str) -> str:
    lowered = (log_text or "").lower()
    for marker in BLOCKING_MARKERS:
        if marker in lowered:
            return "blocking"
    if "traceback" in lowered and ("nan" in lowered or "inf" in lowered):
        return "blocking"
    return "retryable"


def find_active_child_pid(artifact_root: Path, active_run_id: Optional[str]) -> Optional[int]:
    if not active_run_id:
        return None
    rows = read_csv_rows(nightly_root(artifact_root) / "process_ledger.csv")
    pid: Optional[str] = None
    for row in rows:
        if row.get("run_id") != active_run_id:
            continue
        if row.get("event") == "started":
            pid = row.get("pid")
        elif row.get("event") == "exited":
            pid = None
    return int(pid) if pid else None


def _write_deadline_marker(run_dir: Path, block_id: str) -> None:
    ma.atomic_write_json(
        Path(run_dir) / DEADLINE_MARKER_FILENAME,
        {"block_id": block_id, "reason": "soft_deadline_reached"},
    )


def write_incident(
    incidents_dir: Path, *, block_id: str, run_id: str, track_id: str, seed: int, exit_code, log_path: Path,
    run_dir: Path, retry_count: int, classification: str, python_exe: Path, script_path: Path, device: str,
    artifact_root: Path,
) -> str:
    incidents_dir = Path(incidents_dir)
    incidents_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"{ts}_{run_id.replace('/', '_')}"

    status_path = Path(run_dir) / "status.json"
    status = ma.read_json(status_path) if status_path.exists() else {}
    log_tail = ""
    if Path(log_path).exists():
        log_tail = Path(log_path).read_text(encoding="utf-8", errors="replace")[-4000:]

    recovery_command = " ".join(build_fit_command(
        python_exe, script_path, artifact_root=artifact_root, track_id=track_id, seed=seed, device=device,
        resume=True, deadline_timestamp=None,
    ))
    payload = {
        "schema_id": SCHEMA_ID, "block_id": block_id, "run_id": run_id, "track_id": track_id, "seed": seed,
        "epoch": status.get("final_epoch", status.get("last_completed_epoch")),
        "exception_text": log_tail, "exit_code": exit_code,
        "last_checkpoint_path": str(Path(run_dir) / "checkpoints" / "last_resumable_checkpoint.pt"),
        "semantic_training_hash": status.get("semantic_training_hash"),
        "execution_policy_hash": status.get("execution_policy_hash"),
        "evaluation_policy_hash": status.get("evaluation_policy_hash"),
        "retry_count": retry_count, "classification": classification,
        "automatic_action_taken": "no_retry_blocking" if classification == "blocking" else "retries_exhausted_blocked",
        "manual_recovery_command": recovery_command,
    }
    ma.atomic_write_json(incidents_dir / f"{stem}.json", payload)

    md_lines = [
        f"# Incident: {run_id}", "",
        f"- block_id: {block_id}", f"- classification: {classification}", f"- exit_code: {exit_code}",
        f"- retry_count: {retry_count}", f"- last_checkpoint_path: {payload['last_checkpoint_path']}",
        f"- semantic_training_hash: {payload['semantic_training_hash']}",
        f"- execution_policy_hash: {payload['execution_policy_hash']}", "",
        "## Manual recovery command", "", "```", recovery_command, "```", "",
        "## Log tail", "", "```", log_tail, "```",
    ]
    md_path = incidents_dir / f"{stem}.md"
    tmp = incidents_dir / f".tmp_{stem}.md"
    tmp.write_text("\n".join(md_lines), encoding="utf-8")
    os.replace(tmp, md_path)
    return stem


# --- campaign state --------------------------------------------------


def _update_campaign_state(root: Path, block_id: str, block_record: Dict[str, Any]) -> None:
    path = root / "campaign_state.json"
    state = ma.read_json(path) if path.exists() else {
        "schema_id": SCHEMA_ID, "blocks_completed": 0, "last_block_id": None, "last_block_state": None,
    }
    state["last_block_id"] = block_id
    state["last_block_state"] = block_record["state"]
    state["blocks_completed"] = state.get("blocks_completed", 0) + 1
    ma.atomic_write_json(path, state)


# --- supervisor loop -----------------------------------------------------


@dataclass
class SupervisorConfig:
    artifact_root: Path
    repo_root: Path
    python_exe: Path
    script_path: Path
    device: str = "cuda"
    duration_hours: float = 8.0
    soft_stop_hours: float = 7.5
    shutdown_margin_hours: float = 0.5
    max_retries: int = 2
    retry_delay_seconds: float = 3.0
    safety_margin: float = 1.15
    epoch_seconds_default: float = DEFAULT_EPOCH_SECONDS_ESTIMATE
    popen_fn: Callable = subprocess.Popen
    now_fn: Callable[[], float] = time.time
    sleep_fn: Callable[[float], None] = time.sleep


def run_supervisor_block(config: SupervisorConfig) -> Dict[str, Any]:
    """Run exactly one nightly block: acquire the lock, keep-awake, drive the
    queue forward (launch/retry/advance the frozen NF_AC fit subprocess) until
    the campaign completes, a blocking failure occurs, the soft/hard deadline
    is reached, or a stop-after-epoch request is honored -- then release
    everything and persist a block record. Never launches a second concurrent
    ``fit`` subprocess."""

    artifact_root = Path(config.artifact_root)
    root = nightly_root(artifact_root)
    lock_path = _lock_path(artifact_root)
    stop_flag_path = _stop_flag_path(artifact_root)

    block_start = config.now_fn()
    soft_deadline = block_start + config.soft_stop_hours * 3600.0
    hard_deadline = block_start + config.duration_hours * 3600.0
    block_id = "block_" + datetime.fromtimestamp(block_start, tz=timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

    reconcile_lock(lock_path)
    if stop_flag_path.exists():
        stop_flag_path.unlink(missing_ok=True)

    lock_data = acquire_supervisor_lock(lock_path, block_id=block_id, repo_path=config.repo_root)

    process_ledger_path = root / "process_ledger.csv"
    resource_ledger_path = root / "resource_ledger.csv"
    keep_awake_ok = set_keep_awake()
    _append_csv_row(resource_ledger_path, RESOURCE_LEDGER_HEADER, {
        "timestamp": config.now_fn(), "block_id": block_id, "event": "keep_awake_set", "detail": str(keep_awake_ok),
    })

    block_record: Dict[str, Any] = {
        "schema_id": SCHEMA_ID, "block_id": block_id, "state": BLOCK_RUNNING,
        "start_ts": block_start, "soft_deadline": soft_deadline, "hard_deadline": hard_deadline,
        "items": [],
    }
    retry_counts: Dict[str, int] = {}

    try:
        while True:
            queue_items = reconcile_queue_and_persist(artifact_root)
            current = determine_current_run(queue_items, artifact_root)
            if current is None:
                block_record["state"] = BLOCK_COMPLETED
                print(GATE_D_COMPLETE_MARKER)
                break

            if stop_flag_path.exists():
                block_record["state"] = BLOCK_COMPLETED_WITH_RETRY if retry_counts else BLOCK_PLANNED
                break

            now = config.now_fn()
            if now >= hard_deadline:
                break

            run_dir = artifact_root / current["run_dir"]
            estimated_epoch_seconds = estimate_epoch_seconds(run_dir, default=config.epoch_seconds_default)
            remaining = soft_deadline - now
            if remaining < estimated_epoch_seconds * config.safety_margin:
                # No conservative time for one more epoch before the soft
                # deadline -- leave the run in its current state and exit.
                break

            resume = current["run_state"] != RUN_PENDING
            log_path = root / "logs" / f"{block_id}_{current['run_id'].replace('/', '_')}.log"
            cmd = build_fit_command(
                config.python_exe, config.script_path, artifact_root=artifact_root, track_id=current["track_id"],
                seed=current["seed"], device=config.device, resume=resume, deadline_timestamp=soft_deadline,
            )

            lock_data["active_run_id"] = current["run_id"]
            ma.atomic_write_json(lock_path, lock_data)

            process, log_handle = launch_fit_subprocess(cmd, log_path=log_path, cwd=config.repo_root, popen_fn=config.popen_fn)
            _append_csv_row(process_ledger_path, PROCESS_LEDGER_HEADER, {
                "timestamp": config.now_fn(), "block_id": block_id, "run_id": current["run_id"], "event": "started",
                "pid": process.pid, "command": " ".join(cmd), "exit_code": "",
                "retry_count": retry_counts.get(current["run_id"], 0),
            })
            returncode = process.wait()
            try:
                log_handle.close()
            except Exception:
                pass
            _append_csv_row(process_ledger_path, PROCESS_LEDGER_HEADER, {
                "timestamp": config.now_fn(), "block_id": block_id, "run_id": current["run_id"], "event": "exited",
                "pid": process.pid, "command": " ".join(cmd), "exit_code": returncode,
                "retry_count": retry_counts.get(current["run_id"], 0),
            })

            new_state, status = derive_run_state(run_dir)

            if new_state == RUN_COMPLETED:
                block_record["items"].append({"run_id": current["run_id"], "outcome": "completed"})
                if config.now_fn() >= hard_deadline:
                    break
                continue

            if returncode == 0 and status is not None and status.get("status") == d9runner.STATUS_INTERRUPTED:
                # A clean exit that isn't "completed" and carries status
                # "interrupted" here is the fit subprocess's own
                # --deadline-timestamp mechanism doing exactly what it was
                # asked: it declined to start a new epoch past our soft
                # deadline. Mark it ours and end this block cleanly.
                _write_deadline_marker(run_dir, block_id)
                block_record["items"].append({"run_id": current["run_id"], "outcome": "interrupted_at_deadline"})
                break

            if new_state == RUN_FAILED_BLOCKED:
                classification = "blocking"
            else:
                log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
                classification = classify_failure(log_text)

            if classification == "blocking":
                incident_id = write_incident(
                    root / "incidents", block_id=block_id, run_id=current["run_id"], track_id=current["track_id"],
                    seed=current["seed"], exit_code=returncode, log_path=log_path, run_dir=run_dir,
                    retry_count=retry_counts.get(current["run_id"], 0), classification="blocking",
                    python_exe=config.python_exe, script_path=config.script_path, device=config.device,
                    artifact_root=artifact_root,
                )
                block_record["state"] = BLOCK_BLOCKED
                block_record["items"].append({"run_id": current["run_id"], "outcome": "blocked", "incident_id": incident_id})
                break

            retry_counts[current["run_id"]] = retry_counts.get(current["run_id"], 0) + 1
            _append_csv_row(process_ledger_path, PROCESS_LEDGER_HEADER, {
                "timestamp": config.now_fn(), "block_id": block_id, "run_id": current["run_id"], "event": "retry",
                "pid": "", "command": " ".join(cmd), "exit_code": returncode,
                "retry_count": retry_counts[current["run_id"]],
            })

            if retry_counts[current["run_id"]] > config.max_retries:
                incident_id = write_incident(
                    root / "incidents", block_id=block_id, run_id=current["run_id"], track_id=current["track_id"],
                    seed=current["seed"], exit_code=returncode, log_path=log_path, run_dir=run_dir,
                    retry_count=retry_counts[current["run_id"]], classification="retryable_exhausted",
                    python_exe=config.python_exe, script_path=config.script_path, device=config.device,
                    artifact_root=artifact_root,
                )
                block_record["state"] = BLOCK_BLOCKED
                block_record["items"].append({"run_id": current["run_id"], "outcome": "retries_exhausted", "incident_id": incident_id})
                break

            if config.now_fn() >= hard_deadline:
                break
            if config.retry_delay_seconds:
                config.sleep_fn(config.retry_delay_seconds)
    finally:
        clear_ok = clear_keep_awake()
        _append_csv_row(resource_ledger_path, RESOURCE_LEDGER_HEADER, {
            "timestamp": config.now_fn(), "block_id": block_id, "event": "keep_awake_cleared", "detail": str(clear_ok),
        })
        release_supervisor_lock(lock_path)
        if stop_flag_path.exists():
            stop_flag_path.unlink(missing_ok=True)

    if block_record["state"] == BLOCK_RUNNING:
        block_record["state"] = BLOCK_PLANNED
    block_record["end_ts"] = config.now_fn()
    ma.atomic_write_json(root / "nightly_blocks" / f"{block_id}.json", block_record)
    _update_campaign_state(root, block_id, block_record)
    return block_record
