"""Environment / provenance capture for reproducible runs.

Records git commit + dirty state, interpreter and library versions, requested
vs actual device, GPU name, hostname and timestamps. No secrets and no
absolute user-specific paths are recorded. Heavy libraries are probed via
lazy imports; absence is recorded as ``None`` rather than raising.
"""

from __future__ import annotations

import datetime as _dt
import platform
import socket
import subprocess
from typing import Any, Dict, Optional


def _run_git(args) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def _optional_version(module_name: str) -> Optional[str]:
    try:
        module = __import__(module_name)
    except Exception:
        return None
    return getattr(module, "__version__", None)


def utc_timestamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def resolve_actual_device(requested: str) -> Dict[str, Any]:
    """Resolve the requested device to the actual device + GPU name (lazy torch)."""

    info: Dict[str, Any] = {
        "requested_device": requested,
        "actual_device": "cpu",
        "gpu_name": None,
        "cuda_available": False,
        "cuda_runtime": None,
    }
    try:
        import torch
    except Exception:
        return info
    info["cuda_available"] = bool(torch.cuda.is_available())
    info["cuda_runtime"] = getattr(torch.version, "cuda", None)
    if requested == "cuda" or (requested == "auto" and info["cuda_available"]):
        if info["cuda_available"]:
            info["actual_device"] = "cuda"
            try:
                info["gpu_name"] = torch.cuda.get_device_name(0)
            except Exception:
                info["gpu_name"] = None
    return info


def capture_environment(*, requested_device: str = "cpu") -> Dict[str, Any]:
    """Return a JSON-serializable environment/provenance record."""

    device_info = resolve_actual_device(requested_device)
    return {
        "git_commit": _run_git(["rev-parse", "HEAD"]),
        "dirty_worktree": bool(_run_git(["status", "--porcelain"])),
        "python_version": platform.python_version(),
        "numpy_version": _optional_version("numpy"),
        "sklearn_version": _optional_version("sklearn"),
        "torch_version": _optional_version("torch"),
        "mlflow_version": _optional_version("mlflow"),
        "matplotlib_version": _optional_version("matplotlib"),
        "cuda_runtime": device_info["cuda_runtime"],
        "cuda_available": device_info["cuda_available"],
        "requested_device": device_info["requested_device"],
        "actual_device": device_info["actual_device"],
        "gpu_name": device_info["gpu_name"],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "captured_at": utc_timestamp(),
    }
