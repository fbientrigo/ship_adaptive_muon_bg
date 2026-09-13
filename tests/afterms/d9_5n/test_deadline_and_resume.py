"""Deadline-timestamp interrupt semantics (Step 1 wiring) and exact resume
with no duplicate epochs. Reuses the interrupt_flag-based resume test
pattern from tests/afterms/d9_5/test_nf_adapter.py. Tiny synthetic CPU
fixtures only.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

from ship_muon_bg.afterms.d9 import runner as d9runner

REPO_ROOT = Path(__file__).resolve().parents[3]
FIT_SCRIPT = REPO_ROOT / "scripts" / "run_afterms_d9_5_model_family_arena.py"


def _run_cli(*args, artifact_root=None):
    cli_args = []
    if artifact_root is not None:
        cli_args += ["--artifact-root", str(artifact_root)]
    cli_args += list(args)
    return subprocess.run(
        [sys.executable, str(FIT_SCRIPT), *cli_args], cwd=str(REPO_ROOT), capture_output=True, text=True,
    )


def test_fit_help_documents_deadline_timestamp_flag():
    result = _run_cli("fit", "--help")
    assert "--deadline-timestamp" in result.stdout


@pytest.mark.local_env
def test_fit_dry_run_unchanged_when_deadline_timestamp_omitted(tmp_path):
    result = _run_cli(
        "fit", "--track-id", "TRK_PDG13_UW_ID", "--model-family", "NF_AC", "--seed", "20260720",
        artifact_root=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert '"executed": false' in result.stdout


@pytest.mark.local_env
def test_fit_dry_run_unchanged_when_deadline_timestamp_present(tmp_path):
    """Regression guard: passing --deadline-timestamp without --execute must
    behave identically to omitting it (no scientific-behavior change from
    this purely-additive flag)."""

    without = _run_cli(
        "fit", "--track-id", "TRK_PDG13_UW_ID", "--model-family", "NF_AC", "--seed", "20260720",
        artifact_root=tmp_path,
    )
    with_flag = _run_cli(
        "fit", "--track-id", "TRK_PDG13_UW_ID", "--model-family", "NF_AC", "--seed", "20260720",
        "--deadline-timestamp", "99999999999.0", artifact_root=tmp_path,
    )
    assert without.returncode == with_flag.returncode == 0
    import json

    without_json = json.loads(without.stdout)
    with_json = json.loads(with_flag.stdout)
    without_json.pop("run_dir", None)
    with_json.pop("run_dir", None)
    assert without_json == with_json


def test_deadline_in_the_past_interrupts_before_first_epoch(make_nf_adapter, tiny_train_validation, tmp_path, tiny_nf_execution_policy):
    train_raw, validation_raw = tiny_train_validation
    execution_policy = dict(tiny_nf_execution_policy, maximum_epochs=3, minimum_epochs=1, early_stopping_patience=5)
    adapter = make_nf_adapter("TRK_PDG13_UW_ID", 13, tmp_path, execution_policy)

    deadline_timestamp = time.time() - 1.0  # already expired
    interrupt_flag = lambda: time.time() >= deadline_timestamp
    result = adapter.fit(train_raw, validation_raw, seed=20260720, interrupt_flag=interrupt_flag)
    assert result["status"] == d9runner.STATUS_INTERRUPTED


def test_deadline_interrupt_then_resume_completes_with_no_duplicate_epochs(
    make_nf_adapter, tiny_train_validation, tmp_path, tiny_nf_execution_policy,
):
    train_raw, validation_raw = tiny_train_validation
    execution_policy = dict(tiny_nf_execution_policy, maximum_epochs=3, minimum_epochs=1, early_stopping_patience=5)

    epoch_polls = {"count": 0}

    def interrupt_after_epoch_one():
        epoch_polls["count"] += 1
        return epoch_polls["count"] > 1  # only the epoch-boundary poll, never mid-epoch

    adapter1 = make_nf_adapter("TRK_PDG13_UW_ID", 13, tmp_path, execution_policy)
    result1 = adapter1.fit(train_raw, validation_raw, seed=20260720, interrupt_flag=interrupt_after_epoch_one)
    assert result1["status"] == d9runner.STATUS_INTERRUPTED
    # Exactly one epoch boundary was crossed before the interrupt fired --
    # never a mid-epoch stop.
    assert epoch_polls["count"] == 2

    adapter2 = make_nf_adapter("TRK_PDG13_UW_ID", 13, tmp_path, execution_policy)
    result2 = adapter2.fit(train_raw, validation_raw, seed=20260720, resume=True)
    assert result2["status"] == d9runner.STATUS_COMPLETED

    history = (adapter2._run_dir / "histories" / "training_history.json").read_text(encoding="utf-8")
    import json as _json

    epochs = [rec["epoch"] for rec in _json.loads(history)]
    assert epochs == sorted(set(epochs))  # strictly increasing, no duplicates
    assert epochs == list(range(1, epochs[-1] + 1))
