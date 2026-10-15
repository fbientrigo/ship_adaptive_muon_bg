"""CLI --dry-run: reports plan, touches no filesystem, never invokes D7 queue/training."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _run_cli(input_artifact_dir, shard_dir, output_dir, extra_args=()):
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "build_afterms_smoke_arena.py"),
        "--input-artifact-dir", str(input_artifact_dir),
        "--shard-dir", str(shard_dir),
        "--output-dir", str(output_dir),
        *extra_args,
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


def test_dry_run_reports_plan_without_writing_outputs(input_artifact_dir, shard_dir, tmp_path):
    output_dir = tmp_path / "d8_out"
    result = _run_cli(input_artifact_dir, shard_dir, output_dir, extra_args=["--dry-run"])
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "DRY_RUN_OK"
    assert "discovered_runs" in payload
    assert "arenas" in payload
    assert "evaluation_budgets" in payload
    assert not output_dir.exists()


def test_dry_run_never_calls_d7_queue_module():
    source = (REPO_ROOT / "scripts" / "build_afterms_smoke_arena.py").read_text()
    assert "run_afterms_nightly_queue" not in source
    assert "build_afterms_shards" not in source
    assert ".fit(" not in source


def test_dry_run_blocks_on_inconsistent_input(input_artifact_dir, shard_dir, raw_pkl, tmp_path):
    raw_pkl.write_bytes(b"mutated, breaks the hash relation")
    output_dir = tmp_path / "d8_out"
    result = _run_cli(input_artifact_dir, shard_dir, output_dir, extra_args=["--dry-run"])
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "AFTERMS_SMOKE_ARENA_BLOCKED_BY_INPUTS"


def test_full_run_gate1_outputs_and_refuses_overwrite_without_force(input_artifact_dir, shard_dir, tmp_path):
    output_dir = tmp_path / "d8_out"
    result = _run_cli(input_artifact_dir, shard_dir, output_dir)
    assert result.returncode == 0, result.stderr
    assert (output_dir / "registry" / "run_registry.json").exists()
    assert (output_dir / "arenas" / "model_arena.json").exists()
    assert (output_dir / "audit" / "immutable_input_manifest.json").exists()

    result2 = _run_cli(input_artifact_dir, shard_dir, output_dir)
    assert result2.returncode == 2
