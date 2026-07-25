"""Supervisor block state machine: retry policy, blocking-failure
classification, CPU thread env vars, keep-awake bookkeeping, queue
advancement, and the no-time-for-another-epoch soft-deadline skip. All
against tiny synthetic CPU fixtures via a fake (in-process) popen -- never a
real subprocess/GPU.
"""

from __future__ import annotations

import time as time_module
from pathlib import Path

from ship_muon_bg.afterms.d9_5 import model_adapter as ma
from ship_muon_bg.afterms.d9_5 import nightly_runner as nr

from . import _fake_subprocess as fsp

REPO_ROOT = Path(__file__).resolve().parents[3]
FIT_SCRIPT = REPO_ROOT / "scripts" / "run_afterms_d9_5_model_family_arena.py"
PYTHON_EXE = REPO_ROOT / ".venv" / "Scripts" / "python.exe"

PDG_BY_TRACK = {"TRK_PDG13_UW_ID": 13, "TRK_PDGM13_UW_ID": -13}


def _config(artifact_root, popen_fn, *, duration_hours=8.0, soft_stop_hours=7.5, **overrides):
    return nr.SupervisorConfig(
        artifact_root=artifact_root, repo_root=REPO_ROOT, python_exe=PYTHON_EXE, script_path=FIT_SCRIPT,
        device="cpu", duration_hours=duration_hours, soft_stop_hours=soft_stop_hours,
        popen_fn=popen_fn, retry_delay_seconds=0.0, **overrides,
    )


def _real_popen(tmp_path, execution_policy, tiny_nf_optimizer_settings, tiny_evaluation_policy, tiny_scout_report,
                 tiny_shard_dir, now_fn=time_module.time):
    return fsp.make_real_tiny_fit_popen(
        artifact_root=tmp_path, execution_policy=execution_policy, optimizer_settings=tiny_nf_optimizer_settings,
        evaluation_policy=tiny_evaluation_policy, pdg_value_by_track=PDG_BY_TRACK, shard_dir=tiny_shard_dir,
        scout_report_path=tiny_scout_report, now_fn=now_fn,
    )


def test_completed_run_advances_queue_to_next_item(
    tmp_path, tiny_scout_report, tiny_shard_dir, tiny_nf_execution_policy,
    tiny_nf_optimizer_settings, tiny_evaluation_policy,
):
    nr.init_nightly_runner(tmp_path, scout_report_path=tiny_scout_report)
    execution_policy = dict(tiny_nf_execution_policy, maximum_epochs=1, minimum_epochs=1, early_stopping_patience=5)

    clock = fsp.ManualClock(time_module.time())
    real_popen = _real_popen(
        tmp_path, execution_policy, tiny_nf_optimizer_settings, tiny_evaluation_policy, tiny_scout_report,
        tiny_shard_dir,
    )
    # Deterministically process exactly one queue item per simulated block:
    # jump the clock far past the hard deadline right after this one item
    # completes, instead of depending on the real (fast but nonzero) wall
    # time of a tiny fit.
    popen_fn = fsp.advance_clock_after(real_popen, clock, 999999.0)
    config = _config(tmp_path, popen_fn, epoch_seconds_default=0.001, now_fn=clock.now)
    block = nr.run_supervisor_block(config)

    items = nr.reconcile_queue_and_persist(tmp_path)
    current = nr.determine_current_run(items, tmp_path)
    assert current is not None
    assert current["seed"] == 20260721  # advanced past the first (20260720) item
    assert block["items"][0]["outcome"] == "completed"


def test_interrupted_run_remains_current_item(
    tmp_path, tiny_scout_report, tiny_shard_dir, tiny_nf_execution_policy,
    tiny_nf_optimizer_settings, tiny_evaluation_policy,
):
    nr.init_nightly_runner(tmp_path, scout_report_path=tiny_scout_report)
    execution_policy = dict(tiny_nf_execution_policy, maximum_epochs=3, minimum_epochs=1, early_stopping_patience=5)

    # Soft deadline already in the past -> the fit subprocess's own
    # --deadline-timestamp mechanism refuses to start any epoch.
    now = time_module.time()
    popen_fn = _real_popen(
        tmp_path, execution_policy, tiny_nf_optimizer_settings, tiny_evaluation_policy, tiny_scout_report,
        tiny_shard_dir, now_fn=lambda: now + 999,
    )
    config = _config(tmp_path, popen_fn, soft_stop_hours=0.0001, duration_hours=8.0, epoch_seconds_default=0.001)
    block = nr.run_supervisor_block(config)

    items = nr.reconcile_queue_and_persist(tmp_path)
    current = nr.determine_current_run(items, tmp_path)
    assert current["seed"] == 20260720  # still the first item, not advanced
    assert block["items"][0]["outcome"] == "interrupted_at_deadline"

    run_dir = tmp_path / current["run_dir"]
    assert (run_dir / nr.DEADLINE_MARKER_FILENAME).exists()


def test_no_new_epoch_starts_after_soft_deadline_skips_launch(tmp_path, tiny_scout_report):
    nr.init_nightly_runner(tmp_path, scout_report_path=tiny_scout_report)

    calls = []

    def never_call_popen(cmd, **kwargs):  # pragma: no cover - must never run
        calls.append(cmd)
        raise AssertionError("subprocess must not be launched when there is no time for one more epoch")

    # Estimated epoch time (huge default) exceeds the remaining time to the
    # soft deadline -> the block must end without ever launching.
    config = _config(tmp_path, never_call_popen, soft_stop_hours=0.0000001, epoch_seconds_default=999999.0)
    block = nr.run_supervisor_block(config)

    assert calls == []
    assert block["items"] == []
    ledger_rows = nr.read_csv_rows(nr.nightly_root(tmp_path) / "process_ledger.csv")
    assert ledger_rows == []


def test_retryable_failure_retries_same_command_up_to_max_then_succeeds(
    tmp_path, tiny_scout_report, tiny_shard_dir, tiny_nf_execution_policy,
    tiny_nf_optimizer_settings, tiny_evaluation_policy,
):
    nr.init_nightly_runner(tmp_path, scout_report_path=tiny_scout_report)
    execution_policy = dict(tiny_nf_execution_policy, maximum_epochs=1, minimum_epochs=1, early_stopping_patience=5)

    clock = fsp.ManualClock(time_module.time())
    real_popen = _real_popen(
        tmp_path, execution_policy, tiny_nf_optimizer_settings, tiny_evaluation_policy, tiny_scout_report,
        tiny_shard_dir,
    )
    success_popen = fsp.advance_clock_after(real_popen, clock, 999999.0)
    flaky_popen = fsp.make_sequenced_popen([
        fsp.make_forced_failure_popen(text="Traceback (most recent call last):\nRuntimeError: transient glitch\n", returncode=1),
        fsp.make_forced_failure_popen(text="Traceback (most recent call last):\nRuntimeError: transient glitch\n", returncode=1),
        success_popen,
    ])
    config = _config(tmp_path, flaky_popen, epoch_seconds_default=0.001, max_retries=2, now_fn=clock.now)
    block = nr.run_supervisor_block(config)

    assert block["items"][-1]["outcome"] == "completed"
    ledger_rows = nr.read_csv_rows(nr.nightly_root(tmp_path) / "process_ledger.csv")
    retry_events = [r for r in ledger_rows if r["event"] == "retry"]
    assert len(retry_events) == 2
    commands = {r["command"] for r in ledger_rows if r["event"] == "started"}
    assert len(commands) == 1  # exact same command retried


def test_blocking_failure_sets_failed_blocked_and_does_not_advance_or_retry(tmp_path, tiny_scout_report):
    nr.init_nightly_runner(tmp_path, scout_report_path=tiny_scout_report)

    blocking_popen = fsp.make_forced_failure_popen(
        text="RuntimeError: CUDA out of memory. Tried to allocate ...\n", returncode=1,
    )
    config = _config(tmp_path, blocking_popen, epoch_seconds_default=0.001)
    block = nr.run_supervisor_block(config)

    assert block["state"] == nr.BLOCK_BLOCKED
    assert block["items"][-1]["outcome"] == "blocked"

    items = nr.reconcile_queue_and_persist(tmp_path)
    current = nr.determine_current_run(items, tmp_path)
    assert current["seed"] == 20260720  # never advanced
    assert current["run_state"] == nr.RUN_PENDING  # status.json was never written by our fake subprocess

    ledger_rows = nr.read_csv_rows(nr.nightly_root(tmp_path) / "process_ledger.csv")
    assert not any(r["event"] == "retry" for r in ledger_rows)  # blocking failures are never retried

    incidents_dir = nr.nightly_root(tmp_path) / "incidents"
    incident_files = list(incidents_dir.glob("*.json"))
    assert len(incident_files) == 1
    incident = ma.read_json(incident_files[0])
    assert incident["classification"] == "blocking"
    assert "fit" in incident["manual_recovery_command"]
    assert "--resume" in incident["manual_recovery_command"]


def test_cpu_thread_env_vars_are_set_on_child(
    tmp_path, tiny_scout_report, tiny_shard_dir, tiny_nf_execution_policy,
    tiny_nf_optimizer_settings, tiny_evaluation_policy,
):
    nr.init_nightly_runner(tmp_path, scout_report_path=tiny_scout_report)
    execution_policy = dict(tiny_nf_execution_policy, maximum_epochs=1, minimum_epochs=1, early_stopping_patience=5)

    clock = fsp.ManualClock(time_module.time())
    real_popen = _real_popen(
        tmp_path, execution_policy, tiny_nf_optimizer_settings, tiny_evaluation_policy, tiny_scout_report,
        tiny_shard_dir,
    )
    real_popen = fsp.advance_clock_after(real_popen, clock, 999999.0)
    captured_envs = []

    def capturing_popen(cmd, **kwargs):
        captured_envs.append(kwargs["env"])
        return real_popen(cmd, **kwargs)

    config = _config(tmp_path, capturing_popen, epoch_seconds_default=0.001, now_fn=clock.now)
    nr.run_supervisor_block(config)

    assert len(captured_envs) == 1
    for key, value in nr.CPU_THREAD_ENV_VARS.items():
        assert captured_envs[0][key] == value


def test_keep_awake_is_attempted_recorded_and_undone(
    tmp_path, tiny_scout_report, tiny_shard_dir, tiny_nf_execution_policy,
    tiny_nf_optimizer_settings, tiny_evaluation_policy,
):
    nr.init_nightly_runner(tmp_path, scout_report_path=tiny_scout_report)
    execution_policy = dict(tiny_nf_execution_policy, maximum_epochs=1, minimum_epochs=1, early_stopping_patience=5)
    clock = fsp.ManualClock(time_module.time())
    real_popen = _real_popen(
        tmp_path, execution_policy, tiny_nf_optimizer_settings, tiny_evaluation_policy, tiny_scout_report,
        tiny_shard_dir,
    )
    popen_fn = fsp.advance_clock_after(real_popen, clock, 999999.0)
    config = _config(tmp_path, popen_fn, epoch_seconds_default=0.001, now_fn=clock.now)
    nr.run_supervisor_block(config)

    rows = nr.read_csv_rows(nr.nightly_root(tmp_path) / "resource_ledger.csv")
    events = [r["event"] for r in rows]
    assert "keep_awake_set" in events
    assert "keep_awake_cleared" in events
    # This is a real Windows machine: ctypes.windll is available, so both
    # calls should report success.
    set_row = next(r for r in rows if r["event"] == "keep_awake_set")
    cleared_row = next(r for r in rows if r["event"] == "keep_awake_cleared")
    assert set_row["detail"] == "True"
    assert cleared_row["detail"] == "True"


def test_lock_is_released_after_block_completes(tmp_path, tiny_scout_report):
    nr.init_nightly_runner(tmp_path, scout_report_path=tiny_scout_report)
    blocking_popen = fsp.make_forced_failure_popen(text="RuntimeError: CUDA out of memory\n", returncode=1)
    config = _config(tmp_path, blocking_popen, epoch_seconds_default=0.001)
    nr.run_supervisor_block(config)
    assert not (nr.nightly_root(tmp_path) / "locks" / "supervisor.lock").exists()
