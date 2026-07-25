#!/usr/bin/env python3
"""run_afterms_d9_5_nightly.py: nightly execution supervisor for Gate D of the
D9-5 model-family arena (6 NF_AC production training runs).

Pure execution/orchestration plumbing wrapped AROUND the frozen scientific
training code (``run_afterms_d9_5_model_family_arena.py fit``). Never touches
scientific hyperparameters, never opens test-split data.

Subcommands:
    init              Create the nightly_runner/ state tree. Idempotent;
                       refuses to overwrite an incompatible frozen queue.
    doctor            Read-only environment checks.
    start             Acquire the lock and launch a DETACHED run-foreground
                       supervisor loop that survives this process exiting.
    run-foreground    The actual supervisor loop (state machine + deadlines +
                       subprocess management + keep-awake).
    status            Read-only campaign/lock/queue report.
    tail              Read-only tail of the current block's log file.
    stop-after-epoch  Request a graceful stop at the next epoch boundary.
    abort             PID-specific termination of the live supervisor (+ its
                       one active fit child), never a by-name kill.
    reconcile         Force a state resync against disk without launching
                       anything.
    history           Read-only listing of past nightly blocks.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from ship_muon_bg.afterms.d9_5 import model_adapter as ma  # noqa: E402
from ship_muon_bg.afterms.d9_5 import nightly_runner as nr  # noqa: E402

DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "afterms_d9_5_model_family_arena_v0"
FIT_SCRIPT_PATH = REPO_ROOT / "scripts" / "run_afterms_d9_5_model_family_arena.py"
DEFAULT_PYTHON_EXE = REPO_ROOT / ".venv" / "Scripts" / "python.exe"


def _config_from_args(args) -> nr.SupervisorConfig:
    return nr.SupervisorConfig(
        artifact_root=args.artifact_root, repo_root=REPO_ROOT, python_exe=args.python_exe,
        script_path=FIT_SCRIPT_PATH, device=args.device, duration_hours=args.duration_hours,
        soft_stop_hours=args.soft_stop_hours,
    )


def cmd_init(args) -> int:
    try:
        result = nr.init_nightly_runner(args.artifact_root, repo_root=REPO_ROOT)
    except nr.IncompatibleQueueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, default=str))
    return 0


def cmd_doctor(args) -> int:
    report: dict = {"python_exe": str(args.python_exe), "python_exe_exists": Path(args.python_exe).exists()}
    try:
        import torch

        report["cuda_available"] = bool(torch.cuda.is_available())
        report["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as exc:
        report["cuda_available"] = None
        report["torch_import_error"] = str(exc)
    try:
        import shutil

        drive = Path(args.artifact_root).anchor or "C:\\"
        usage = shutil.disk_usage(drive)
        report["free_disk_gb"] = round(usage.free / (1024 ** 3), 2)
    except Exception:
        report["free_disk_gb"] = None
    report["nightly_runner_initialized"] = (nr.nightly_root(args.artifact_root) / "run_queue.json").exists()
    report["psutil_available"] = nr._psutil() is not None
    # nightly_runner never names a test shard anywhere in its own code/config
    # (it only invokes the frozen `fit` CLI, which itself never reads test
    # data outside `evaluate-test`); this flag documents that structural
    # property for a human doctor-run, not a runtime scan.
    report["no_test_shard_reference_in_nightly_code"] = True
    print(json.dumps(report, indent=2, default=str))
    return 0


def cmd_start(args) -> int:
    try:
        nr.init_nightly_runner(args.artifact_root, repo_root=REPO_ROOT)
    except nr.IncompatibleQueueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    lock_path = nr._lock_path(args.artifact_root)
    existing = nr._read_json_if_exists(lock_path)
    if existing is not None and nr.is_lock_live(existing):
        print(f"ERROR: a supervisor is already running (pid={existing.get('pid')}, "
              f"block_id={existing.get('block_id')})", file=sys.stderr)
        return 3

    log_dir = nr.nightly_root(args.artifact_root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    supervisor_log = log_dir / f"supervisor_{ts}.log"

    cmd = [
        str(args.python_exe), str(Path(__file__).resolve()),
        "--artifact-root", str(args.artifact_root), "--python-exe", str(args.python_exe),
        "run-foreground", "--device", args.device,
        "--duration-hours", str(args.duration_hours), "--soft-stop-hours", str(args.soft_stop_hours),
    ]
    creationflags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    log_handle = open(supervisor_log, "ab")
    process = subprocess.Popen(
        cmd, cwd=str(REPO_ROOT), creationflags=creationflags, close_fds=True,
        stdin=subprocess.DEVNULL, stdout=log_handle, stderr=subprocess.STDOUT,
    )

    queue_items = nr.reconcile_queue_and_persist(args.artifact_root)
    current = nr.determine_current_run(queue_items, args.artifact_root)
    now = time.time()
    soft_deadline = now + args.soft_stop_hours * 3600.0
    hard_deadline = now + args.duration_hours * 3600.0
    report = {
        "detached_supervisor_pid": process.pid,
        "current_run": current["run_id"] if current else None,
        "current_run_completed_epoch": ((current or {}).get("status") or {}).get("final_epoch") if current else None,
        "soft_deadline_ts": soft_deadline,
        "soft_deadline_human": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(soft_deadline)),
        "hard_deadline_ts": hard_deadline,
        "hard_deadline_human": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(hard_deadline)),
        "log_path": str(supervisor_log),
        "next_command": f"{sys.executable} {Path(__file__).name} status --artifact-root {args.artifact_root}",
    }
    print(json.dumps(report, indent=2, default=str))
    return 0


def cmd_run_foreground(args) -> int:
    config = _config_from_args(args)
    try:
        block_record = nr.run_supervisor_block(config)
    except nr.SupervisorAlreadyRunningError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3
    print(json.dumps(block_record, indent=2, default=str))
    return 0


def cmd_status(args) -> int:
    root = nr.nightly_root(args.artifact_root)
    queue_path = root / "run_queue.json"
    if not queue_path.exists():
        print(json.dumps({"initialized": False}, indent=2))
        return 0

    lock_data = nr._read_json_if_exists(nr._lock_path(args.artifact_root))
    lock_live = nr.is_lock_live(lock_data)

    items = ma.read_json(queue_path)["items"]
    annotated = []
    for item in items:
        run_dir = args.artifact_root / item["run_dir"]
        state, status = nr.derive_run_state(run_dir)
        annotated.append({
            "run_id": item["run_id"], "state": state,
            "epoch": (status or {}).get("final_epoch", (status or {}).get("last_completed_epoch")),
        })
    current = next((it for it in annotated if it["state"] != nr.RUN_COMPLETED), None)
    campaign_state_path = root / "campaign_state.json"
    campaign_state = ma.read_json(campaign_state_path) if campaign_state_path.exists() else {}
    ledger_tail = nr.read_csv_rows(root / "process_ledger.csv")[-5:]

    if current is None:
        print(nr.GATE_D_COMPLETE_MARKER)

    report = {
        "lock": {"present": lock_data is not None, "live": lock_live, "data": lock_data},
        "queue": annotated, "current_run": current, "campaign_state": campaign_state,
        "recent_process_ledger": ledger_tail,
    }
    print(json.dumps(report, indent=2, default=str))
    return 0


def cmd_tail(args) -> int:
    logs_dir = nr.nightly_root(args.artifact_root) / "logs"
    candidates = sorted(logs_dir.glob("block_*"), key=lambda p: p.stat().st_mtime) if logs_dir.exists() else []
    if not candidates:
        print("no block log files yet", file=sys.stderr)
        return 1
    log_path = candidates[-1]

    def _print_new(from_offset: int) -> int:
        with log_path.open("r", encoding="utf-8", errors="replace") as fh:
            fh.seek(from_offset)
            data = fh.read()
        if data:
            sys.stdout.write(data)
        return log_path.stat().st_size

    text = log_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    for line in lines[-args.lines:]:
        print(line)
    offset = log_path.stat().st_size

    if args.follow:
        try:
            while True:
                time.sleep(1.0)
                offset = _print_new(offset)
        except KeyboardInterrupt:
            pass
    return 0


def cmd_stop_after_epoch(args) -> int:
    lock_data = nr._read_json_if_exists(nr._lock_path(args.artifact_root))
    if lock_data is None or not nr.is_lock_live(lock_data):
        print("ERROR: no live supervisor lock found", file=sys.stderr)
        return 3
    flag_path = nr._stop_flag_path(args.artifact_root)
    ma.atomic_write_json(flag_path, {
        "requested_pid": os.getpid(), "requested_at": time.time(), "target_pid": lock_data["pid"],
    })
    print(json.dumps({"action": "stop_requested", "flag_path": str(flag_path)}, indent=2))
    return 0


def cmd_abort(args) -> int:
    lock_data = nr._read_json_if_exists(nr._lock_path(args.artifact_root))
    if lock_data is None or not nr.is_lock_live(lock_data):
        print("ERROR: no live supervisor lock found; nothing to abort", file=sys.stderr)
        return 3

    supervisor_pid = lock_data["pid"]
    active_run_id = lock_data.get("active_run_id")
    child_pid = nr.find_active_child_pid(args.artifact_root, active_run_id)
    terminated = []
    if child_pid is not None and nr.is_pid_alive(child_pid):
        nr.terminate_pid(child_pid)
        terminated.append(child_pid)
    if nr.is_pid_alive(supervisor_pid):
        nr.terminate_pid(supervisor_pid)
        terminated.append(supervisor_pid)
    nr.force_release_lock_after_abort(nr._lock_path(args.artifact_root))

    print(json.dumps({"action": "aborted", "terminated_pids": terminated, "block_id": lock_data.get("block_id")}, indent=2))
    return 0


def cmd_reconcile(args) -> int:
    lock_result = nr.reconcile_lock(nr._lock_path(args.artifact_root))
    queue_items = nr.reconcile_queue_and_persist(args.artifact_root)
    print(json.dumps({"lock": lock_result, "queue": queue_items}, indent=2, default=str))
    return 0


def cmd_history(args) -> int:
    blocks_dir = nr.nightly_root(args.artifact_root) / "nightly_blocks"
    records = [ma.read_json(p) for p in sorted(blocks_dir.glob("*.json"))] if blocks_dir.exists() else []
    print(json.dumps(records, indent=2, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--python-exe", type=Path, default=DEFAULT_PYTHON_EXE)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="Create the nightly_runner/ state tree. Idempotent.").set_defaults(func=cmd_init)
    sub.add_parser("doctor", help="Read-only environment checks.").set_defaults(func=cmd_doctor)

    for name, func, help_text in (
        ("start", cmd_start, "Acquire the lock and launch a detached run-foreground supervisor."),
        ("run-foreground", cmd_run_foreground, "The actual supervisor loop (state machine + deadlines)."),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--device", default="cuda")
        p.add_argument("--duration-hours", type=float, default=8.0, dest="duration_hours")
        p.add_argument("--soft-stop-hours", type=float, default=7.5, dest="soft_stop_hours")
        p.add_argument("--shutdown-margin-hours", type=float, default=0.5, dest="shutdown_margin_hours")
        p.set_defaults(func=func)

    sub.add_parser("status", help="Read-only campaign/lock/queue report.").set_defaults(func=cmd_status)

    p_tail = sub.add_parser("tail", help="Read-only tail of the current block's log file.")
    p_tail.add_argument("--lines", type=int, default=40)
    p_tail.add_argument("--follow", action="store_true")
    p_tail.set_defaults(func=cmd_tail)

    sub.add_parser("stop-after-epoch", help="Request a graceful stop at the next epoch boundary.").set_defaults(func=cmd_stop_after_epoch)
    sub.add_parser("abort", help="PID-specific termination of the live supervisor + its active fit child.").set_defaults(func=cmd_abort)
    sub.add_parser("reconcile", help="Force a state resync against disk without launching anything.").set_defaults(func=cmd_reconcile)
    sub.add_parser("history", help="Read-only listing of past nightly blocks.").set_defaults(func=cmd_history)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
