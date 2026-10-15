"""Fake subprocess.Popen-compatible helpers for nightly_runner tests.

Instead of actually spawning ``python scripts/run_afterms_d9_5_model_family_arena.py
fit ...``, these either (a) run the real (tiny, CPU) ``NfAcAdapter.fit(...)``
in-process and return a fake completed-process object, so the supervisor
state machine is exercised against a real, tiny NF_AC training run without
subprocess/GPU overhead, or (b) simulate a forced failure by writing text to
the log handle and returning a nonzero exit code. Not a fixture module --
plain helper functions/classes, imported directly by test modules.
"""

from __future__ import annotations

import os
import time as time_module
from typing import Callable, Dict, List

from ship_muon_bg.afterms.d9_5 import data_scope as _data_scope
from ship_muon_bg.afterms.d9_5 import nf_ac_adapter as _nf_ac_adapter


class FakeCompletedProcess:
    def __init__(self, pid: int, returncode: int) -> None:
        self.pid = pid
        self.returncode = returncode

    def wait(self) -> int:
        return self.returncode


def parse_fit_cmd(cmd: List[str]) -> Dict[str, object]:
    parsed: Dict[str, object] = {"resume": False, "deadline_timestamp": None}
    it = iter(cmd)
    for token in it:
        if token == "--track-id":
            parsed["track_id"] = next(it)
        elif token == "--seed":
            parsed["seed"] = int(next(it))
        elif token == "--device":
            parsed["device"] = next(it)
        elif token == "--resume":
            parsed["resume"] = True
        elif token == "--deadline-timestamp":
            parsed["deadline_timestamp"] = float(next(it))
        elif token == "--model-family":
            parsed["model_family"] = next(it)
    return parsed


def make_real_tiny_fit_popen(
    *, artifact_root, execution_policy, optimizer_settings, evaluation_policy, pdg_value_by_track: Dict[str, int],
    shard_dir, scout_report_path, now_fn: Callable[[], float] = time_module.time,
):
    """A popen_fn compatible with nightly_runner.launch_fit_subprocess that
    actually runs a real tiny CPU NF_AC fit synchronously (never a mock of
    the training loop itself). Resolves the architecture the same way
    ``nightly_runner.resolve_run_queue_items`` does (via
    ``select_scout_promoted_nf_config``), so the run_dir this fit actually
    writes to matches the run_dir the queue/state machine expects. Loads and
    PDG-filters train/validation rows per-track (from ``shard_dir``), exactly
    once per call, so multi-track queues (both TRK_PDG13_UW_ID and
    TRK_PDGM13_UW_ID) never see the wrong track's rows."""

    manifest = _data_scope.load_shard_manifest(shard_dir)

    def _popen(cmd, *, cwd, env, stdout, stderr, stdin):
        parsed = parse_fit_cmd(cmd)
        track_id = parsed["track_id"]
        pdg_value = pdg_value_by_track[track_id]
        train_raw = _data_scope.load_filtered_split(shard_dir, manifest, "train", pdg_value)
        validation_raw = _data_scope.load_filtered_split(shard_dir, manifest, "validation", pdg_value)

        scout = _nf_ac_adapter.select_scout_promoted_nf_config(track_id, scout_report_path=scout_report_path)
        adapter = _nf_ac_adapter.NfAcAdapter(
            track_id=track_id, pdg_value=pdg_value, architecture=scout["architecture"],
            execution_policy=execution_policy, optimizer_settings=optimizer_settings,
            evaluation_policy=evaluation_policy, artifact_root=artifact_root, device="cpu",
        )
        interrupt_flag = None
        if parsed["deadline_timestamp"] is not None:
            deadline_ts = parsed["deadline_timestamp"]
            interrupt_flag = lambda: now_fn() >= deadline_ts
        result = adapter.fit(
            train_raw, validation_raw, seed=parsed["seed"], resume=parsed["resume"], interrupt_flag=interrupt_flag,
        )
        try:
            stdout.write(f"fit result: {result}\n".encode("utf-8"))
            stdout.flush()
        except Exception:
            pass
        return FakeCompletedProcess(pid=os.getpid(), returncode=0)

    return _popen


def make_forced_failure_popen(*, text: str, returncode: int = 1):
    def _popen(cmd, *, cwd, env, stdout, stderr, stdin):
        try:
            stdout.write(text.encode("utf-8"))
            stdout.flush()
        except Exception:
            pass
        return FakeCompletedProcess(pid=os.getpid(), returncode=returncode)

    return _popen


def advance_clock_after(popen_fn: Callable, clock: "ManualClock", delta_seconds: float) -> Callable:
    """Wraps a popen_fn so that, after it returns, ``clock`` jumps forward by
    ``delta_seconds`` -- used to deterministically force exactly one queue
    item's worth of work per simulated nightly block in tests, without
    depending on real wall-clock timing of the (fast but nonzero) tiny fit."""

    def _wrapped(cmd, **kwargs):
        result = popen_fn(cmd, **kwargs)
        clock.advance(delta_seconds)
        return result

    return _wrapped


def make_sequenced_popen(popen_fns: List[Callable]):
    """Calls the next function in ``popen_fns`` each invocation, staying on
    the last one once exhausted."""

    state = {"i": 0}

    def _popen(cmd, **kwargs):
        idx = min(state["i"], len(popen_fns) - 1)
        state["i"] += 1
        return popen_fns[idx](cmd, **kwargs)

    return _popen


class ManualClock:
    """A fully test-controlled clock for ``SupervisorConfig.now_fn``."""

    def __init__(self, start: float) -> None:
        self.t = float(start)

    def now(self) -> float:
        return self.t

    def advance(self, delta_seconds: float) -> None:
        self.t += float(delta_seconds)
