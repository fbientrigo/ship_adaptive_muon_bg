import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "run_afterms_d9_campaign.py"


def _run(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
    )


def test_plan_dry_run_shows_workload_table_and_trains_nothing():
    """Required test 36: CLI dry-run and workload estimate."""

    result = _run("plan", "--dry-run")
    assert result.returncode == 0
    assert "candidate" in result.stdout
    assert "est_runtime" in result.stdout
    assert "No training was executed by this dry-run." in result.stdout
    assert "Total estimated serial GPU time" in result.stdout


def test_train_requires_candidate_id_and_seed():
    """Required tests 30/31: one-candidate/one-seed CLI requirement; no
    accidental train-all default."""

    result = _run("train")
    assert result.returncode != 0
    assert "--candidate-id" in result.stderr
    assert "--seed" in result.stderr


def test_train_without_seed_fails():
    result = _run("train", "--candidate-id", "A1_capacity_medium_identity_pdg13_unweighted")
    assert result.returncode != 0


def test_status_never_writes_artifacts(tmp_path):
    """Required: status never writes scientific artifacts."""

    before = list(tmp_path.rglob("*"))
    result = _run("--artifact-root", str(tmp_path), "status")
    assert result.returncode == 0
    after = list(tmp_path.rglob("*"))
    assert before == after


def test_no_subcommand_is_an_error():
    result = _run()
    assert result.returncode != 0
