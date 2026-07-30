"""CLI tests for ``scripts/build_dataset_report.py`` (D9/D7 dataset validation
command). Exercises the repository afterMS fixture directly through the same
CLI a full local dataset would use -- only the ``--dataset`` path differs.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO_ROOT, "scripts", "build_dataset_report.py")
FIXTURE = os.path.join(
    REPO_ROOT, "data", "samples", "muonsFullMC_afterMS_sample.npz"
)
TINY_PKL = os.path.join(REPO_ROOT, "tests", "fixtures", "muon_sample_tiny.pkl.gz")


def _run(args):
    return subprocess.run(
        [sys.executable, SCRIPT] + args,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def test_validate_only_on_repository_fixture_succeeds_and_reports_pdg_counts():
    result = _run(
        ["--dataset", FIXTURE, "--validate-only", "--allow-zero-weight", "--seed", "1234"]
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["validate_only"] is True
    assert report["n_rows"] == 40000
    assert report["expected_pdg_counts"] == {"13": 19788, "-13": 20212}
    assert all(c["passed"] for c in report["validation"])
    assert report["duplicate_rows"]["checked"] is True
    assert report["duplicate_rows"]["duplicate_row_count"] == 0
    assert report["duplicate_source_identifiers"]["status"] == (
        "not_applicable_no_source_event_identifier_column"
    )
    assert report["weight_column_status"]["present"] is True
    assert report["memory_footprint_estimate"]["raw_array_bytes"] == 40000 * 8 * 8
    assert report["split_feasibility"]["feasible"] is True


def test_validate_only_writes_output_file(tmp_path):
    out = tmp_path / "report.json"
    result = _run(
        [
            "--dataset", FIXTURE, "--validate-only", "--allow-zero-weight",
            "--seed", "1234", "--output", str(out),
        ]
    )
    assert result.returncode == 0, result.stderr
    assert out.exists()
    report = json.loads(out.read_text())
    assert report["n_rows"] == 40000


def test_validate_only_respects_max_rows_deterministically(tmp_path):
    out1 = tmp_path / "r1.json"
    out2 = tmp_path / "r2.json"
    common = [
        "--dataset", FIXTURE, "--validate-only", "--allow-zero-weight",
        "--seed", "99", "--max-rows", "1000",
    ]
    r1 = _run(common + ["--output", str(out1)])
    r2 = _run(common + ["--output", str(out2)])
    assert r1.returncode == 0 and r2.returncode == 0
    report1 = json.loads(out1.read_text())
    report2 = json.loads(out2.read_text())
    assert report1["row_budget"] == {
        "requested_max_rows": 1000, "available_rows_before_cap": 40000, "rows_used": 1000,
    }
    # Deterministic: same seed/max-rows -> identical dataset_hash.
    assert report1["dataset_hash"] == report2["dataset_hash"]


def test_max_rows_without_seed_is_rejected():
    result = _run(["--dataset", FIXTURE, "--validate-only", "--max-rows", "100"])
    assert result.returncode != 0
    assert "--max-rows requires --seed" in result.stderr


def test_campaign_config_compatibility_check(tmp_path):
    cfg = tmp_path / "campaign.json"
    cfg.write_text(json.dumps({
        "dataset": {"n_train": 100, "n_validation": 50, "n_test": 50},
        "pdg_ids": [13, -13],
    }))
    result = _run(
        [
            "--dataset", FIXTURE, "--validate-only", "--allow-zero-weight",
            "--seed", "1", "--campaign-config", str(cfg),
        ]
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    compat = report["campaign_config_compatibility"]
    assert compat["checked"] is True
    assert compat["all_requested_pdg_tracks_sufficient"] is True
    assert compat["required_rows_per_pdg"] == 200


def test_default_mode_matches_legacy_three_artifact_layout(tmp_path):
    out_dir = tmp_path / "artifacts"
    result = _run(
        ["--dataset", TINY_PKL, "--output-dir", str(out_dir), "--seed", "1234"]
    )
    assert result.returncode == 0, result.stderr
    for name in ("dataset_report.json", "split_manifest.json", "normalization.json"):
        assert (out_dir / name).exists()


def test_legacy_input_flag_still_accepted(tmp_path):
    out_dir = tmp_path / "artifacts"
    result = _run(
        ["--input", TINY_PKL, "--output-dir", str(out_dir), "--seed", "1234"]
    )
    assert result.returncode == 0, result.stderr
    assert (out_dir / "dataset_report.json").exists()


def test_default_mode_requires_output_dir_or_validate_only():
    result = _run(["--dataset", TINY_PKL, "--seed", "1234"])
    assert result.returncode != 0
    assert "--output-dir is required" in result.stderr


def test_default_mode_requires_seed():
    result = _run(["--dataset", TINY_PKL, "--output-dir", "/tmp/whatever"])
    assert result.returncode != 0
    assert "--seed is required" in result.stderr


def test_invalid_dataset_exits_nonzero_with_finite_check_failing():
    """An invalid dataset must fail loudly via a non-zero exit under
    --validate-only, not silently report success."""

    import gzip
    import pickle

    import numpy as np

    bad_path = "/tmp/build_dataset_report_cli_invalid.pkl.gz"
    array = np.zeros((10, 8), dtype=np.float64)
    array[0, 0] = np.nan
    array[:, 6] = 13.0
    array[:, 7] = 1.0
    with gzip.open(bad_path, "wb") as handle:
        pickle.dump(array, handle)
    try:
        result = _run(["--dataset", bad_path, "--validate-only", "--seed", "1"])
        assert result.returncode == 1, result.stdout
        report = json.loads(result.stdout)
        assert any(not c["passed"] for c in report["validation"])
    finally:
        os.remove(bad_path)
