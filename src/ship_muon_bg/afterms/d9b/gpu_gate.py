"""D9B RTX 2060 real-CUDA qualification gate orchestration (Gate B).

Supervises the existing D9 training/reload machinery in isolated CUDA child
processes -- one at a time, always via ``.venv\\Scripts\\python.exe`` -- and
records process ownership and GPU telemetry for each. This module never
reimplements ``ship_muon_bg.afterms.d9.runner.train_candidate_seed``; it only
spawns and supervises ``scripts/_d9b_gpu_gate_worker.py``, which itself calls
that runner unmodified.

Process ownership is always PID-specific: descendants are enumerated via WMI
(``Get-CimInstance Win32_Process -Filter "ParentProcessId=<pid>"``) and, if a
child fails to exit on its own, terminated with ``taskkill /PID <pid> /T``.
Neither this module nor any helper here ever issues a name-based kill
(``taskkill /IM ...`` / ``Stop-Process -Name ...``).
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

GATE_PASSED = "GPU_GATE_PASSED"
GATE_FAILED = "GPU_GATE_FAILED"
GATE_PARTIAL = "GPU_GATE_PARTIAL"

WORKER_RELATIVE_PATH = Path("scripts") / "_d9b_gpu_gate_worker.py"


class GpuGateError(RuntimeError):
    """A GPU-gate stop condition was hit (section 5 of the mission spec)."""


class ChildProcessAlreadyActiveError(GpuGateError):
    """A second CUDA child was requested while one is still tracked active."""


def resolve_python_executable(repo_root: Path) -> Path:
    exe = Path(repo_root, ".venv", "Scripts", "python.exe")
    if not exe.exists():
        raise GpuGateError(f".venv python executable not found at {exe}")
    return exe


def resolve_worker_script(repo_root: Path) -> Path:
    script = Path(repo_root, WORKER_RELATIVE_PATH)
    if not script.exists():
        raise GpuGateError(f"gpu-gate worker script not found at {script}")
    return script


# ----------------------------------------------------------------------
# nvidia-smi / process ownership helpers
# ----------------------------------------------------------------------


def query_nvidia_smi_gpu() -> Dict[str, Any]:
    """One-shot GPU name/memory/driver query. Never raises -- returns
    ``{"available": False, ...}`` if ``nvidia-smi`` cannot be run."""

    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=15, check=True,
        )
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    line = out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 5:
        return {"available": False, "error": f"unexpected nvidia-smi output: {line!r}"}
    name, total, used, free, driver = parts
    try:
        return {
            "available": True, "gpu_name": name,
            "memory_total_mib": int(total), "memory_used_mib": int(used),
            "memory_free_mib": int(free), "driver_version": driver,
        }
    except ValueError:
        return {"available": False, "error": f"non-numeric nvidia-smi memory field: {line!r}"}


def query_nvidia_smi_compute_apps() -> List[Dict[str, Any]]:
    """Processes currently holding GPU compute memory, per ``nvidia-smi``.
    Never raises -- returns ``[]`` on any failure."""

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True,
        )
    except Exception:
        return []
    rows = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            try:
                rows.append({"pid": int(parts[0]), "process_name": parts[1], "used_memory_mib": int(parts[2])})
            except ValueError:
                continue
    return rows


def enumerate_descendant_pids(pid: int) -> List[int]:
    """PID-specific descendant enumeration via WMI. Never lists processes by
    name. Returns ``[]`` if there are none or the query fails."""

    try:
        out = subprocess.run(
            [
                "powershell", "-NoProfile", "-Command",
                f"(Get-CimInstance Win32_Process -Filter \"ParentProcessId={int(pid)}\" | "
                "Select-Object -ExpandProperty ProcessId) -join ','",
            ],
            capture_output=True, text=True, timeout=15, check=True,
        )
    except Exception:
        return []
    text = out.stdout.strip()
    if not text:
        return []
    children = [int(p) for p in text.split(",") if p.strip().isdigit()]
    descendants = list(children)
    for child in children:
        descendants.extend(enumerate_descendant_pids(child))
    return descendants


def terminate_pid_tree(pid: int) -> Dict[str, Any]:
    """PID-specific termination: ``taskkill /PID <pid> /T`` only. Never
    ``/IM`` (image-name kill) and never ``Stop-Process -Name``. Only ever
    called on a PID this module itself launched and is still tracking."""

    try:
        result = subprocess.run(
            ["taskkill", "/PID", str(int(pid)), "/T", "/F"],
            capture_output=True, text=True, timeout=15,
        )
        return {"pid": int(pid), "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except Exception as exc:
        return {"pid": int(pid), "error": str(exc)}


# ----------------------------------------------------------------------
# Child-process supervision + ledger
# ----------------------------------------------------------------------


class ProcessSupervisor:
    """Runs at most one tracked CUDA child process at a time and builds a
    process-ownership ledger of every child it launches."""

    def __init__(self) -> None:
        self._active_pid: Optional[int] = None
        self.ledger: List[Dict[str, Any]] = []

    @property
    def active_pid(self) -> Optional[int]:
        return self._active_pid

    def run(
        self, command: List[str], *, label: str, cwd: Optional[Path] = None, timeout: Optional[float] = 1800.0,
    ) -> Dict[str, Any]:
        if self._active_pid is not None:
            raise ChildProcessAlreadyActiveError(
                f"cannot launch {label!r}: PID {self._active_pid} is still tracked active "
                "(one CUDA child process at a time)"
            )

        nvidia_before = query_nvidia_smi_gpu()
        start_time = time.time()
        proc = subprocess.Popen(
            command, cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        self._active_pid = proc.pid
        descendants_during = enumerate_descendant_pids(proc.pid)

        timed_out = False
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            terminate_pid_tree(proc.pid)
            stdout, stderr = proc.communicate()
        finally:
            end_time = time.time()
            self._active_pid = None

        descendants_after = enumerate_descendant_pids(proc.pid)
        orphan_process_result = (
            "no_descendants_detected" if not descendants_after else f"descendants_still_present:{descendants_after}"
        )
        if descendants_after:
            for descendant_pid in descendants_after:
                terminate_pid_tree(descendant_pid)

        entry = {
            "label": label,
            "command": command,
            "child_pid": proc.pid,
            "descendant_pids_during": descendants_during,
            "descendant_pids_after": descendants_after,
            "start_time": start_time,
            "end_time": end_time,
            "duration_seconds": end_time - start_time,
            "exit_status": proc.returncode,
            "timed_out": timed_out,
            "nvidia_smi_before": nvidia_before,
            "nvidia_smi_after": query_nvidia_smi_gpu(),
            "orphan_process_result": orphan_process_result,
        }
        self.ledger.append(entry)
        return {"returncode": proc.returncode, "stdout": stdout, "stderr": stderr, "ledger_entry": entry}


# ----------------------------------------------------------------------
# Child command builders
# ----------------------------------------------------------------------


def build_train_command(
    python_exe: Path, worker_script: Path, *,
    candidate_config_json: Path, seed: int, shard_dir: Path, artifact_root: Path,
    device: str, telemetry_out: Path, resume: bool = False,
    interrupt_after_epochs: Optional[int] = None,
) -> List[str]:
    cmd = [
        str(python_exe), str(worker_script), "train",
        "--candidate-config-json", str(candidate_config_json),
        "--seed", str(seed),
        "--shard-dir", str(shard_dir),
        "--artifact-root", str(artifact_root),
        "--device", device,
        "--telemetry-out", str(telemetry_out),
    ]
    if resume:
        cmd.append("--resume")
    if interrupt_after_epochs is not None:
        cmd += ["--interrupt-after-epochs", str(interrupt_after_epochs)]
    return cmd


def build_reload_command(
    python_exe: Path, worker_script: Path, *,
    run_dir: Path, device: str, sample_count: int, generation_seed: int, output_json: Path,
) -> List[str]:
    return [
        str(python_exe), str(worker_script), "reload",
        "--run-dir", str(run_dir),
        "--device", device,
        "--sample-count", str(sample_count),
        "--generation-seed", str(generation_seed),
        "--output-json", str(output_json),
    ]


# ----------------------------------------------------------------------
# Stop-condition checks (section 5)
# ----------------------------------------------------------------------


def check_train_stage_result(stage_label: str, run_result: Dict[str, Any], telemetry_path: Path) -> List[str]:
    """Return a list of stop-condition violations for a completed `train`
    child (empty if none). Never raises -- the caller decides how to react."""

    violations: List[str] = []
    if run_result["returncode"] not in (0, 1):
        violations.append(f"{stage_label}: worker exited with unexpected code {run_result['returncode']}")
    telemetry = None
    if telemetry_path.exists():
        telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
        if telemetry.get("error"):
            violations.append(f"{stage_label}: worker raised {telemetry['error']}")
    else:
        violations.append(f"{stage_label}: telemetry file was not written")

    if telemetry is not None and telemetry.get("result"):
        result = telemetry["result"]
        if result.get("status") not in ("completed", "interrupted"):
            violations.append(f"{stage_label}: training status {result.get('status')!r} is not completed/interrupted")
    return violations


def check_history_all_finite(history: List[Dict[str, Any]], stage_label: str) -> List[str]:
    violations = []
    for record in history:
        if not record.get("finite_loss", True):
            violations.append(f"{stage_label}: non-finite loss at epoch {record.get('epoch')}")
    return violations


# ----------------------------------------------------------------------
# Clean-run comparison (pure, real-run-independent logic -- section 4D)
# ----------------------------------------------------------------------


def compare_clean_runs(
    history_1: List[Dict[str, Any]], history_2: List[Dict[str, Any]],
    sample_hash_1: Optional[str], sample_hash_2: Optional[str],
) -> Dict[str, Any]:
    """Compare two clean-run training histories and deterministic sample
    hashes. A hash mismatch is reported as an observation, never a failure:
    GPU float non-associativity (cuBLAS reduction order) can make two
    same-seed CUDA runs diverge slightly in trained weights even though both
    are legitimate, deterministic-per-run executions of the same training
    path -- see the qualification report's determinism-result section."""

    same_length = len(history_1) == len(history_2)
    final_train_delta = None
    final_validation_delta = None
    if history_1 and history_2:
        final_train_delta = abs(history_1[-1]["train_feature_nll"] - history_2[-1]["train_feature_nll"])
        final_validation_delta = abs(
            history_1[-1]["validation_feature_nll"] - history_2[-1]["validation_feature_nll"]
        )
    return {
        "history_length_match": same_length,
        "history_length_1": len(history_1),
        "history_length_2": len(history_2),
        "final_train_nll_abs_delta": final_train_delta,
        "final_validation_nll_abs_delta": final_validation_delta,
        "sample_hash_1": sample_hash_1,
        "sample_hash_2": sample_hash_2,
        "sample_hash_match": (
            sample_hash_1 is not None and sample_hash_2 is not None and sample_hash_1 == sample_hash_2
        ),
    }


# ----------------------------------------------------------------------
# Manifest / report / CSV output (section 6)
# ----------------------------------------------------------------------


def write_process_ledger_csv(path: Path, ledger: List[Dict[str, Any]]) -> None:
    import csv

    fields = [
        "label", "child_pid", "descendant_pids_during", "descendant_pids_after",
        "start_time", "end_time", "duration_seconds", "exit_status", "timed_out",
        "orphan_process_result",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for entry in ledger:
            writer.writerow({k: entry.get(k) for k in fields})


def write_memory_observations_csv(path: Path, observations: List[Dict[str, Any]]) -> None:
    import csv

    fields = [
        "stage", "device",
        "initial_allocated_bytes", "initial_reserved_bytes",
        "peak_allocated_bytes", "peak_reserved_bytes",
        "final_allocated_bytes", "final_reserved_bytes",
        "nvidia_smi_used_mib_before", "nvidia_smi_used_mib_after",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for entry in observations:
            writer.writerow({k: entry.get(k) for k in fields})


def _memory_observation_row(stage: str, telemetry: Dict[str, Any], ledger_entry: Dict[str, Any]) -> Dict[str, Any]:
    initial = telemetry.get("initial", {}) or {}
    peak = telemetry.get("peak", {}) or {}
    final = telemetry.get("final", {}) or {}
    before = (ledger_entry.get("nvidia_smi_before") or {})
    after = (ledger_entry.get("nvidia_smi_after") or {})
    return {
        "stage": stage,
        "device": telemetry.get("device"),
        "initial_allocated_bytes": initial.get("allocated_bytes"),
        "initial_reserved_bytes": initial.get("reserved_bytes"),
        "peak_allocated_bytes": peak.get("allocated_bytes"),
        "peak_reserved_bytes": peak.get("reserved_bytes"),
        "final_allocated_bytes": final.get("allocated_bytes"),
        "final_reserved_bytes": final.get("reserved_bytes"),
        "nvidia_smi_used_mib_before": before.get("memory_used_mib"),
        "nvidia_smi_used_mib_after": after.get("memory_used_mib"),
    }


# ----------------------------------------------------------------------
# Top-level orchestration
# ----------------------------------------------------------------------


def dry_run_plan(candidate_config: Dict[str, Any], selection_report: Dict[str, Any], seed: int, device: str) -> Dict[str, Any]:
    """Describe the intended GPU-gate run without training anything or
    touching any shard file."""

    return {
        "dry_run": True,
        "selection_report": selection_report,
        "candidate_id": candidate_config["candidate_id"],
        "seed": seed,
        "device": device,
        "planned_stages": [
            "A. baseline_inspection", "B. clean_run_1", "C. checkpoint_reload",
            "D. clean_run_2", "E. interruption_and_resume",
        ],
        "planned_epochs_per_clean_run": 2,
        "note": "No training was executed by this dry-run; no shard file was read.",
    }


def run_gpu_gate(
    *, repo_root: Path, gate_artifact_root: Path, plan: Dict[str, Any], training_config: Dict[str, Any],
    device: str = "cuda", seed: Optional[int] = None, max_epochs_override: int = 2,
    interrupt_after_epochs: int = 1, sample_count: int = 64, generation_seed: int = 20260720,
) -> Dict[str, Any]:
    """Execute the full real-CUDA qualification sequence (section 4) and
    write the manifest/report/CSV outputs under ``gate_artifact_root``.

    Never called for a dry-run -- the caller (CLI) only invokes this when
    ``--execute`` was passed explicitly.
    """

    from ship_muon_bg.afterms.d9b import candidate_selection as cs
    from ship_muon_bg.afterms.d9 import runner as d9runner

    candidate_config, selection_report = cs.select_gpu_gate_candidate(plan, training_config)
    candidate_config = dict(candidate_config, max_epochs=max_epochs_override)
    if seed is None:
        seed = candidate_config["seed_set"][0]

    python_exe = resolve_python_executable(repo_root)
    worker_script = resolve_worker_script(repo_root)
    shard_dir = Path(training_config["shard_dir"])

    supervisor = ProcessSupervisor()
    memory_observations: List[Dict[str, Any]] = []
    stop_conditions: List[str] = []
    stages: Dict[str, Any] = {}

    gate_artifact_root.mkdir(parents=True, exist_ok=True)

    # --- A. Baseline inspection ---------------------------------------
    baseline_gpu = query_nvidia_smi_gpu()
    baseline_compute_apps = query_nvidia_smi_compute_apps()
    import torch as _torch  # the orchestrator process itself runs under .venv's CUDA-enabled torch

    stages["baseline_inspection"] = {
        "nvidia_smi_gpu": baseline_gpu,
        "compute_apps_before_gate": baseline_compute_apps,
        "torch_version": str(_torch.__version__),
        "cuda_version": _torch.version.cuda,
        "cuda_available": bool(_torch.cuda.is_available()),
        "gpu_name": _torch.cuda.get_device_name(0) if _torch.cuda.is_available() else None,
    }
    if device == "cuda" and not _torch.cuda.is_available():
        stop_conditions.append("CUDA is unavailable in this environment; refusing --device cuda execution")

    def _candidate_config_path(stage_dir: Path) -> Path:
        stage_dir.mkdir(parents=True, exist_ok=True)
        path = stage_dir / "candidate_config.json"
        path.write_text(json.dumps(candidate_config, indent=2, default=str), encoding="utf-8")
        return path

    def _run_train(stage_label: str, stage_dir: Path, *, resume: bool, interrupt_after: Optional[int]) -> Dict[str, Any]:
        stage_dir.mkdir(parents=True, exist_ok=True)
        config_path = _candidate_config_path(stage_dir)
        telemetry_path = stage_dir / "telemetry.json"
        cmd = build_train_command(
            python_exe, worker_script,
            candidate_config_json=config_path, seed=seed, shard_dir=shard_dir,
            artifact_root=stage_dir, device=device, telemetry_out=telemetry_path,
            resume=resume, interrupt_after_epochs=interrupt_after,
        )
        run_result = supervisor.run(cmd, label=stage_label, cwd=repo_root)
        violations = check_train_stage_result(stage_label, run_result, telemetry_path)
        telemetry = json.loads(telemetry_path.read_text(encoding="utf-8")) if telemetry_path.exists() else {}
        run_dir = d9runner.run_directory(stage_dir, candidate_config["candidate_id"], seed)
        history_path = run_dir / "histories" / "training_history.json"
        history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
        violations += check_history_all_finite(history, stage_label)
        memory_observations.append(_memory_observation_row(stage_label, telemetry, run_result["ledger_entry"]))
        return {
            "run_result": run_result, "telemetry": telemetry, "run_dir": run_dir,
            "history": history, "violations": violations,
            "status_path": run_dir / "status.json",
        }

    if not stop_conditions:
        # --- B. Clean run 1 ---------------------------------------------
        clean_1_dir = gate_artifact_root / "clean_run_1"
        clean_1 = _run_train("clean_run_1", clean_1_dir, resume=False, interrupt_after=None)
        stages["clean_run_1"] = clean_1
        stop_conditions += clean_1["violations"]

    if not stop_conditions:
        # --- C. Checkpoint reload ----------------------------------------
        reload_dir = gate_artifact_root / "checkpoint_reload"
        reload_dir.mkdir(parents=True, exist_ok=True)
        reload_output = reload_dir / "reload_result.json"
        cmd = build_reload_command(
            python_exe, worker_script, run_dir=stages["clean_run_1"]["run_dir"],
            device=device, sample_count=sample_count, generation_seed=generation_seed,
            output_json=reload_output,
        )
        reload_result = supervisor.run(cmd, label="checkpoint_reload", cwd=repo_root)
        reload_payload = json.loads(reload_output.read_text(encoding="utf-8")) if reload_output.exists() else {}
        stages["checkpoint_reload"] = {"run_result": reload_result, "payload": reload_payload}
        if reload_result["returncode"] != 0:
            stop_conditions.append("checkpoint_reload: worker exited non-zero (reload failed or non-finite sample)")
        if reload_payload.get("checkpoint_reload", {}).get("compatible") is False:
            stop_conditions.append("checkpoint_reload: best/final checkpoints reported incompatible")

    if not stop_conditions:
        # --- D. Clean run 2 -----------------------------------------------
        clean_2_dir = gate_artifact_root / "clean_run_2"
        clean_2 = _run_train("clean_run_2", clean_2_dir, resume=False, interrupt_after=None)
        stages["clean_run_2"] = clean_2
        stop_conditions += clean_2["violations"]

        if not stop_conditions:
            reload_2_output = clean_2_dir / "reload_result.json"
            cmd = build_reload_command(
                python_exe, worker_script, run_dir=clean_2["run_dir"],
                device=device, sample_count=sample_count, generation_seed=generation_seed,
                output_json=reload_2_output,
            )
            reload_2_result = supervisor.run(cmd, label="clean_run_2_reload", cwd=repo_root)
            reload_2_payload = json.loads(reload_2_output.read_text(encoding="utf-8")) if reload_2_output.exists() else {}
            stages["clean_run_2_reload"] = {"run_result": reload_2_result, "payload": reload_2_payload}

            hash_1 = stages["checkpoint_reload"]["payload"].get("deterministic_sample", {}).get("sample_hash_sha256")
            hash_2 = reload_2_payload.get("deterministic_sample", {}).get("sample_hash_sha256")
            stages["clean_run_comparison"] = compare_clean_runs(
                stages["clean_run_1"]["history"], clean_2["history"], hash_1, hash_2,
            )

    if not stop_conditions:
        # --- E. Interruption and resume -----------------------------------
        interrupt_dir = gate_artifact_root / "interrupted_resume"
        interrupted = _run_train(
            "interrupted_resume__interrupt", interrupt_dir, resume=False, interrupt_after=interrupt_after_epochs,
        )
        stages["interrupted_resume__interrupt"] = interrupted
        interrupt_status = (
            json.loads(interrupted["status_path"].read_text(encoding="utf-8"))
            if interrupted["status_path"].exists() else {}
        )
        stages["interrupted_resume__interrupt"]["status"] = interrupt_status
        if interrupt_status.get("status") != "interrupted":
            stop_conditions.append(
                f"interrupted_resume: expected status 'interrupted', got {interrupt_status.get('status')!r}"
            )
        resumable_ckpt = interrupted["run_dir"] / "checkpoints" / "last_resumable_checkpoint.pt"
        if not resumable_ckpt.exists():
            stop_conditions.append("interrupted_resume: last_resumable_checkpoint.pt is missing after interrupt")

        if not stop_conditions:
            history_before_resume = list(interrupted["history"])
            resumed = _run_train(
                "interrupted_resume__resume", interrupt_dir, resume=True, interrupt_after=None,
            )
            stages["interrupted_resume__resume"] = resumed
            resume_status = (
                json.loads(resumed["status_path"].read_text(encoding="utf-8"))
                if resumed["status_path"].exists() else {}
            )
            stages["interrupted_resume__resume"]["status"] = resume_status
            if resume_status.get("status") != "completed":
                stop_conditions.append(
                    f"interrupted_resume: resume did not complete, got {resume_status.get('status')!r}"
                )
            epochs_after_resume = [r["epoch"] for r in resumed["history"]]
            if epochs_after_resume != sorted(set(epochs_after_resume)):
                stop_conditions.append("interrupted_resume: duplicate epochs found in post-resume history")
            if len(resumed["history"]) != max_epochs_override:
                stop_conditions.append(
                    f"interrupted_resume: expected {max_epochs_override} total epochs after resume, "
                    f"got {len(resumed['history'])}"
                )
            stages["interrupted_resume__comparison"] = {
                "history_before_resume_epochs": [r["epoch"] for r in history_before_resume],
                "history_after_resume_epochs": epochs_after_resume,
            }

    # --- Orphan / process-ownership final check --------------------------
    orphan_pids = []
    for entry in supervisor.ledger:
        if entry["descendant_pids_after"]:
            orphan_pids.extend(entry["descendant_pids_after"])
    if orphan_pids:
        stop_conditions.append(f"orphaned descendant PIDs detected: {orphan_pids}")

    gate_status = GATE_FAILED if stop_conditions else GATE_PASSED

    manifest = {
        "gate_status": gate_status,
        "stop_conditions": stop_conditions,
        "selection_report": selection_report,
        "candidate_id": candidate_config["candidate_id"],
        "seed": seed,
        "device": device,
        "max_epochs_override": max_epochs_override,
        "interrupt_after_epochs": interrupt_after_epochs,
        "shard_dir": str(shard_dir),
        "train_shards": candidate_config["train_shards"],
        "validation_shards": candidate_config["validation_shards"],
        "stages": stages,
        "process_ledger": supervisor.ledger,
        "memory_observations": memory_observations,
        "orphan_pid_check": "no_orphans_detected" if not orphan_pids else f"orphans_detected:{orphan_pids}",
        "allowed_conclusion": "real_cuda_training_path_validated" if gate_status == GATE_PASSED else None,
    }

    gate_artifact_root.mkdir(parents=True, exist_ok=True)
    (gate_artifact_root / "qualification_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8",
    )
    write_process_ledger_csv(gate_artifact_root / "process_ledger.csv", supervisor.ledger)
    write_memory_observations_csv(gate_artifact_root / "memory_observations.csv", memory_observations)
    (gate_artifact_root / "qualification_report.md").write_text(
        render_qualification_report(manifest), encoding="utf-8",
    )

    return manifest


def render_qualification_report(manifest: Dict[str, Any]) -> str:
    """Render the human-readable qualification_report.md from a manifest
    produced by ``run_gpu_gate``. Never emits a convergence/production-ready
    claim -- only the disclosed allowed conclusion and stage-by-stage facts."""

    lines: List[str] = []
    lines.append("# D9B GPU Gate -- RTX 2060 Real-CUDA Qualification Report")
    lines.append("")
    lines.append(f"**Gate status:** {manifest['gate_status']}")
    lines.append(f"**Candidate:** {manifest['candidate_id']}")
    lines.append(f"**Seed:** {manifest['seed']}")
    lines.append(f"**Device:** {manifest['device']}")
    lines.append(f"**Shard dir:** {manifest['shard_dir']}")
    lines.append(f"**Train shards:** {manifest['train_shards']}")
    lines.append(f"**Validation shards:** {manifest['validation_shards']}")
    lines.append("")

    sel = manifest["selection_report"]
    lines.append("## Candidate selection")
    lines.append(f"- Rule: {sel['selection_rule']}")
    lines.append(f"- Eligible (ranked): {sel['eligible_candidate_ids_ranked']}")
    lines.append(f"- Tie at minimum parameter_count: {sel['tied_at_minimum_candidate_ids']}")
    lines.append(f"- Selected: {sel['selected_candidate_id']}")
    lines.append("")

    baseline = manifest["stages"].get("baseline_inspection", {})
    lines.append("## A. Baseline inspection")
    lines.append(f"- GPU: {baseline.get('gpu_name')}")
    lines.append(f"- CUDA available: {baseline.get('cuda_available')}, CUDA version: {baseline.get('cuda_version')}")
    lines.append(f"- torch version: {baseline.get('torch_version')}")
    lines.append(f"- nvidia-smi: {baseline.get('nvidia_smi_gpu')}")
    lines.append(f"- Compute apps before gate: {baseline.get('compute_apps_before_gate')}")
    lines.append("")

    def _history_summary(history: List[Dict[str, Any]]) -> str:
        if not history:
            return "no epochs recorded"
        rows = [
            f"epoch {r['epoch']}: train={r['train_feature_nll']:.6f} val={r['validation_feature_nll']:.6f} "
            f"wall_s={r['wall_time_seconds']:.2f} finite={r['finite_loss']}"
            for r in history
        ]
        return "; ".join(rows)

    for stage_key, title in (
        ("clean_run_1", "B. Clean run 1"), ("clean_run_2", "D. Clean run 2"),
    ):
        stage = manifest["stages"].get(stage_key)
        lines.append(f"## {title}")
        if stage is None:
            lines.append("- not reached")
        else:
            lines.append(f"- history: {_history_summary(stage['history'])}")
            lines.append(f"- violations: {stage['violations'] or 'none'}")
        lines.append("")

    reload_stage = manifest["stages"].get("checkpoint_reload")
    lines.append("## C. Checkpoint reload")
    if reload_stage is None:
        lines.append("- not reached")
    else:
        payload = reload_stage["payload"]
        lines.append(f"- compatibility: {payload.get('checkpoint_reload')}")
        lines.append(f"- deterministic sample: {payload.get('deterministic_sample')}")
    lines.append("")

    comparison = manifest["stages"].get("clean_run_comparison")
    lines.append("## Clean-run 1 vs 2 comparison (determinism result)")
    if comparison is None:
        lines.append("- not reached")
    else:
        lines.append(f"- history_length_match: {comparison['history_length_match']}")
        lines.append(f"- final_train_nll_abs_delta: {comparison['final_train_nll_abs_delta']}")
        lines.append(f"- final_validation_nll_abs_delta: {comparison['final_validation_nll_abs_delta']}")
        lines.append(f"- sample_hash_1: {comparison['sample_hash_1']}")
        lines.append(f"- sample_hash_2: {comparison['sample_hash_2']}")
        lines.append(f"- sample_hash_match: {comparison['sample_hash_match']}")
        lines.append(
            "- Note: a hash mismatch does not by itself indicate a defect. GPU floating-point "
            "reduction order (cuBLAS) is not guaranteed bit-identical across separate process "
            "launches even with the same seed; this is a known, disclosed limitation, not a stop condition."
        )
    lines.append("")

    lines.append("## E. Interruption and resume")
    interrupt_stage = manifest["stages"].get("interrupted_resume__interrupt")
    resume_stage = manifest["stages"].get("interrupted_resume__resume")
    if interrupt_stage is None:
        lines.append("- not reached")
    else:
        lines.append(f"- interrupt status: {interrupt_stage.get('status', {}).get('status')}")
        lines.append(f"- history at interrupt: {_history_summary(interrupt_stage['history'])}")
        if resume_stage is not None:
            lines.append(f"- resume status: {resume_stage.get('status', {}).get('status')}")
            lines.append(f"- history after resume: {_history_summary(resume_stage['history'])}")
            comp = manifest["stages"].get("interrupted_resume__comparison", {})
            lines.append(f"- epochs before resume: {comp.get('history_before_resume_epochs')}")
            lines.append(f"- epochs after resume: {comp.get('history_after_resume_epochs')}")
    lines.append("")

    lines.append("## Peak VRAM / runtime")
    for obs in manifest["memory_observations"]:
        lines.append(
            f"- {obs['stage']}: peak_allocated={obs['peak_allocated_bytes']} bytes, "
            f"peak_reserved={obs['peak_reserved_bytes']} bytes, "
            f"nvidia_smi_used_mib before/after={obs['nvidia_smi_used_mib_before']}/{obs['nvidia_smi_used_mib_after']}"
        )
    lines.append("")

    lines.append("## Process cleanup / orphan check")
    lines.append(f"- {manifest['orphan_pid_check']}")
    for entry in manifest["process_ledger"]:
        lines.append(
            f"- {entry['label']}: pid={entry['child_pid']} exit_status={entry['exit_status']} "
            f"duration_s={entry['duration_seconds']:.2f} orphan_result={entry['orphan_process_result']}"
        )
    lines.append("")

    lines.append("## Stop conditions")
    lines.append(f"{manifest['stop_conditions'] or 'none'}")
    lines.append("")

    lines.append("## Limitations and non-claims")
    lines.append(
        "- This gate validates that the real CUDA training/checkpoint/resume path executes "
        "correctly on the RTX 2060 for a small number of epochs. It does NOT claim convergence, "
        "a best model, production readiness, or that a GPU memory leak is impossible."
    )
    lines.append(
        "- Two epochs is far short of this candidate's declared max_epochs/minimum_epochs "
        "(qualification only; see configs/afterms/d9_training_v0.json for the full campaign policy)."
    )
    lines.append("")

    lines.append("## Conclusion")
    if manifest["gate_status"] == GATE_PASSED:
        lines.append(f"**{manifest['allowed_conclusion']}**")
    else:
        lines.append("Gate did not pass; see stop conditions above.")
    lines.append("")
    lines.append(f"**{manifest['gate_status']}**")

    return "\n".join(lines)
