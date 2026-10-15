#!/usr/bin/env python3
"""watch_afterms_nightly_queue.py: Watcher for Phase D queue execution and subprocess health checks.

Usage:
    python scripts/watch_afterms_nightly_queue.py [--loop] [--interval SECONDS]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

def check_process_alive(pid: int) -> bool:
    if pid is None:
        return False
    try:
        # On Windows, tasklist is standard
        import ctypes
        PROCESS_QUERY_INFORMATION = 0x0400
        PROCESS_VM_READ = 0x0010
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid)
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    except Exception:
        # Fallback using tasklist
        try:
            out = os.popen(f"tasklist /FI \"PID eq {pid}\"").read()
            return str(pid) in out
        except Exception:
            return False


def parse_watch_packet(artifact_dir):
    queue_state_path = os.path.join(artifact_dir, "queue_state.json")
    
    # Defaults
    packet = {
        "status": "idle",
        "active_job": None,
        "pid": None,
        "start_time": None,
        "elapsed_time": None,
        "latest_epoch": None,
        "train_loss": None,
        "validation_loss": None,
        "cpu_rss_bytes": None,
        "gpu_allocated_bytes": None,
        "gpu_reserved_bytes": None,
        "last_heartbeat": None,
        "last_20_log_lines": [],
        "completed_jobs": [],
        "failed_jobs": [],
        "pending_jobs": [],
        "full_log_path": None,
        "warnings": []
    }
    
    if not os.path.exists(queue_state_path):
        packet["warnings"].append("No queue_state.json found. Queue has not started.")
        return packet
        
    try:
        with open(queue_state_path, "r") as f:
            q_state = json.load(f)
    except Exception as e:
        packet["warnings"].append(f"Failed to read queue_state.json: {e}")
        return packet

    packet["active_job"] = q_state.get("active_job")
    packet["pid"] = q_state.get("pid")
    packet["completed_jobs"] = q_state.get("completed_jobs", [])
    packet["failed_jobs"] = q_state.get("failed_jobs", [])
    packet["pending_jobs"] = q_state.get("pending_jobs", [])
    packet["last_heartbeat"] = q_state.get("heartbeat")

    # Check process status
    is_alive = False
    if packet["pid"]:
        is_alive = check_process_alive(packet["pid"])
        
    if packet["active_job"]:
        if is_alive:
            packet["status"] = "running"
        else:
            packet["status"] = "stale_or_exited"
            packet["warnings"].append(f"Active job subprocess (PID {packet['pid']}) is not running.")
            
        # Try to read active job's progress.json
        progress_path = os.path.join(artifact_dir, "jobs", packet["active_job"], "progress.json")
        if os.path.exists(progress_path):
            try:
                with open(progress_path, "r") as f:
                    prog = json.load(f)
                packet["latest_epoch"] = prog.get("epoch")
                packet["train_loss"] = prog.get("train_loss")
                packet["validation_loss"] = prog.get("validation_loss")
                packet["cpu_rss_bytes"] = prog.get("cpu_rss_bytes")
                packet["gpu_allocated_bytes"] = prog.get("gpu_allocated_bytes")
                packet["gpu_reserved_bytes"] = prog.get("gpu_reserved_bytes")
                
                # Heartbeat check
                prog_heartbeat = prog.get("heartbeat", 0)
                if time.time() - prog_heartbeat > 120.0 and is_alive:
                    packet["warnings"].append("Job heartbeat is stale (>120s old) but process is alive. Possible hang.")
            except Exception as e:
                packet["warnings"].append(f"Failed to read progress.json: {e}")
                
        # Read log file
        log_path = os.path.join(artifact_dir, "jobs", packet["active_job"], "run.log")
        packet["full_log_path"] = os.path.abspath(log_path)
        if os.path.exists(log_path):
            try:
                with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
                packet["last_20_log_lines"] = [l.strip() for l in lines[-20:]]
                
                # Check for critical errors in logs
                log_text = "".join(lines).lower()
                if "out of memory" in log_text or "oom" in log_text:
                    packet["warnings"].append("CUDA OOM error detected in active job logs.")
                if "nan" in log_text or "inf" in log_text:
                    # check for non-finite loss
                    if "non-finite loss" in log_text or "loss = nan" in log_text:
                        packet["warnings"].append("Non-finite loss detected in active job logs.")
                if "cpu fallback" in log_text or "unexpected cpu fallback" in log_text:
                    packet["warnings"].append("Unexpected CPU fallback detected in logs.")
            except Exception as e:
                packet["warnings"].append(f"Failed to read run.log: {e}")
    else:
        packet["status"] = "idle"

    # System-level warnings (disk-space exhaustion)
    try:
        usage = shutil.disk_usage(".")
        free_gb = usage.free / (1024**3)
        if free_gb < 2.0: # Less than 2 GB free
            packet["warnings"].append(f"Disk space exhaustion warning: only {free_gb:.2f} GB free.")
    except Exception:
        pass

    return packet


def write_watch_packet(packet, artifact_dir):
    os.makedirs(artifact_dir, exist_ok=True)
    json_path = os.path.join(artifact_dir, "watch_packet.json")
    md_path = os.path.join(artifact_dir, "watch_packet.md")
    
    # Write JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(packet, f, indent=2, sort_keys=True)
        f.write("\n")
        
    # Write MD
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# WATCHER HEARTBEAT PACKET\n\n")
        f.write(f"- **Status**: `{packet['status'].upper()}`\n")
        f.write(f"- **Active Job**: `{packet['active_job'] or 'None'}`\n")
        f.write(f"- **Active Job PID**: `{packet['pid'] or 'None'}`\n")
        f.write(f"- **Last Heartbeat**: `{packet['last_heartbeat'] or 'None'}`\n\n")
        
        if packet["warnings"]:
            f.write("## ⚠️ WARNINGS / HEALTH ALERTS\n\n")
            for w in packet["warnings"]:
                f.write(f"- {w}\n")
            f.write("\n")
            
        f.write("## Active Job Metrics\n\n")
        f.write(f"- **Epoch**: `{packet['latest_epoch'] or 'N/A'}`\n")
        f.write(f"- **Train Loss**: `{packet['train_loss'] or 'N/A'}`\n")
        f.write(f"- **Val Loss**: `{packet['validation_loss'] or 'N/A'}`\n")
        if packet["cpu_rss_bytes"]:
            cpu_mb = packet["cpu_rss_bytes"] / (1024**2)
            f.write(f"- **CPU RSS**: `{cpu_mb:.2f} MB`\n")
        if packet["gpu_allocated_bytes"]:
            gpu_mb = packet["gpu_allocated_bytes"] / (1024**2)
            f.write(f"- **GPU Allocated**: `{gpu_mb:.2f} MB`\n")
        f.write("\n")
        
        f.write("## Queue Details\n\n")
        f.write(f"- **Completed Jobs**: {len(packet['completed_jobs'])} completed\n")
        f.write(f"- **Failed Jobs**: {len(packet['failed_jobs'])} failed\n")
        f.write(f"- **Pending Jobs**: {len(packet['pending_jobs'])} pending\n\n")
        
        if packet["last_20_log_lines"]:
            f.write("## Latest Active Job Logs\n\n")
            f.write("```text\n")
            for line in packet["last_20_log_lines"]:
                f.write(f"{line}\n")
            f.write("```\n")
            
    return json_path, md_path


def main():
    parser = argparse.ArgumentParser(description="Deterministic campaign watcher")
    parser.add_argument("--loop", action="store_true", help="Run in a loop")
    parser.add_argument("--interval", type=int, default=10, help="Loop interval in seconds")
    parser.add_argument("--artifact-dir", type=str, default="artifacts/afterms_nightly_v0", help="Artifacts folder")
    args = parser.parse_args()

    if args.loop:
        print(f"Watcher started in loop mode. Interval: {args.interval}s")
        try:
            while True:
                packet = parse_watch_packet(args.artifact_dir)
                write_watch_packet(packet, args.artifact_dir)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("Watcher stopped.")
    else:
        packet = parse_watch_packet(args.artifact_dir)
        json_path, md_path = write_watch_packet(packet, args.artifact_dir)
        print(f"Watcher packet written to:\n  JSON: {json_path}\n  MD: {md_path}")
        if packet["warnings"]:
            print("\nWarnings:")
            for w in packet["warnings"]:
                print(f"  - {w}")


if __name__ == "__main__":
    main()
