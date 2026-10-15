"""``run_campaign``: chaining multiple nightly blocks back-to-back, unattended,
until the frozen queue is fully drained or a blocking failure needs a human.
Against tiny synthetic CPU fixtures via a fake (in-process) popen -- never a
real subprocess/GPU.
"""

from __future__ import annotations

import time as time_module
from pathlib import Path

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


def test_campaign_chains_blocks_until_queue_fully_drained(
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
    # Each simulated block completes exactly one queue item, then the clock
    # jumps far past that block's hard deadline so run_supervisor_block exits
    # and run_campaign starts the next block for the next queue item.
    popen_fn = fsp.advance_clock_after(real_popen, clock, 999999.0)
    config = _config(tmp_path, popen_fn, epoch_seconds_default=0.001, now_fn=clock.now)

    campaign_record = nr.run_campaign(config)

    assert campaign_record["final_state"] == nr.BLOCK_COMPLETED
    # One deadline-bounded block per queue item (six), plus one final
    # zero-launch block that discovers the drained queue and stops the chain.
    assert campaign_record["block_count"] == 7
    assert all(b["state"] in (nr.BLOCK_PLANNED, nr.BLOCK_COMPLETED) for b in campaign_record["blocks_run"])
    assert campaign_record["blocks_run"][-1]["state"] == nr.BLOCK_COMPLETED
    assert all(b["state"] == nr.BLOCK_PLANNED for b in campaign_record["blocks_run"][:-1])

    items = nr.reconcile_queue_and_persist(tmp_path)
    assert nr.determine_current_run(items, tmp_path) is None  # queue fully drained
    assert not nr._campaign_lock_path(tmp_path).exists()  # lock released


def test_campaign_stops_at_first_blocking_failure(tmp_path, tiny_scout_report):
    nr.init_nightly_runner(tmp_path, scout_report_path=tiny_scout_report)

    blocking_popen = fsp.make_forced_failure_popen(
        text="RuntimeError: CUDA out of memory. Tried to allocate ...\n", returncode=1,
    )
    config = _config(tmp_path, blocking_popen, epoch_seconds_default=0.001)

    campaign_record = nr.run_campaign(config)

    assert campaign_record["block_count"] == 1
    assert campaign_record["final_state"] == nr.BLOCK_BLOCKED
    assert not nr._campaign_lock_path(tmp_path).exists()  # lock released even on early stop


def test_campaign_refuses_to_start_when_another_campaign_lock_is_live(tmp_path, tiny_scout_report):
    nr.init_nightly_runner(tmp_path, scout_report_path=tiny_scout_report)
    campaign_lock_path = nr._campaign_lock_path(tmp_path)
    nr.acquire_supervisor_lock(campaign_lock_path, block_id="campaign", repo_path=tmp_path, pid=nr.os.getpid())

    def never_call_popen(cmd, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("must not launch a fit subprocess when a campaign is already running")

    config = _config(tmp_path, never_call_popen, epoch_seconds_default=0.001)
    try:
        raised = False
        try:
            nr.run_campaign(config)
        except nr.SupervisorAlreadyRunningError:
            raised = True
        assert raised
    finally:
        nr.release_supervisor_lock(campaign_lock_path)
