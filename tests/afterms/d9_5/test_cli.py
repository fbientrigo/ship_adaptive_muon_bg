"""D9-5 CLI safety tests: dry-run performs no fit, no train-all default.

Covers required tests 35, 36. Runs against the real repo configs (fast:
every path exercised here is read-only/no-fit).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "run_afterms_d9_5_model_family_arena.py"


def _run(*args, artifact_root=None):
    cli_args = []
    if artifact_root is not None:
        cli_args += ["--artifact-root", str(artifact_root)]
    cli_args += list(args)
    return subprocess.run(
        [sys.executable, str(SCRIPT), *cli_args], cwd=str(REPO_ROOT), capture_output=True, text=True,
    )


def test_audit_dry_run_performs_no_scan_and_writes_nothing(tmp_path):
    result = _run("audit", "--dry-run", artifact_root=tmp_path)
    assert result.returncode == 0, result.stderr
    assert '"executed": false' in result.stdout
    assert list(tmp_path.rglob("*")) == []


def test_plan_dry_run_shows_workload_table_and_fits_nothing(tmp_path):
    result = _run("plan", "--dry-run", artifact_root=tmp_path)
    assert result.returncode == 0, result.stderr
    assert "TRK_PDG13_UW_ID" in result.stdout
    assert "TRK_PDGM13_UW_ID" in result.stdout
    assert "No fitting was executed by this dry-run." in result.stdout
    assert list(tmp_path.rglob("*")) == []


def test_fit_requires_track_id_and_model_family():
    result = _run("fit")
    assert result.returncode != 0
    assert "--track-id" in result.stderr
    assert "--model-family" in result.stderr


def test_fit_stochastic_family_requires_seed(tmp_path):
    result = _run("fit", "--track-id", "TRK_PDG13_UW_ID", "--model-family", "NF_AC", artifact_root=tmp_path)
    assert result.returncode != 0
    assert "--seed" in result.stderr


def test_fit_deterministic_family_rejects_seed(tmp_path):
    result = _run(
        "fit", "--track-id", "TRK_PDG13_UW_ID", "--model-family", "GAUSS_DIAG", "--seed", "20260720",
        artifact_root=tmp_path,
    )
    assert result.returncode != 0
    assert "deterministic" in result.stderr


def test_fit_without_execute_performs_no_fit(tmp_path):
    result = _run(
        "fit", "--track-id", "TRK_PDG13_UW_ID", "--model-family", "GAUSS_DIAG", artifact_root=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert '"executed": false' in result.stdout
    assert list(tmp_path.rglob("*")) == []


def test_status_never_writes_artifacts(tmp_path):
    before = list(tmp_path.rglob("*"))
    result = _run("status", artifact_root=tmp_path)
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.rglob("*")) == before


def test_no_subcommand_is_an_error():
    result = subprocess.run([sys.executable, str(SCRIPT)], cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert result.returncode != 0


def test_no_campaign_flag_that_fits_everything():
    # There is no flag on the parser that would fit more than one
    # (track, family[, seed]) in a single invocation.
    help_text = _run("fit", "--help").stdout
    assert "--campaign" not in help_text
