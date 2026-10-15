"""CLI full run (Gate 3): tiny bounded budgets, checks every phase's outputs land
and a second run is byte-identical (deterministic samples) and does not
disturb the frozen inputs."""

from __future__ import annotations

import filecmp
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

TINY_BUDGET_ARGS = [
    "--sample-size", "40",
    "--energy-sample-size", "20",
    "--permutations", "9",
    "--bootstrap-repetitions", "5",
    "--c2st-sample-size", "30",
]


def _run_cli(input_artifact_dir, shard_dir, output_dir, extra_args=()):
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "build_afterms_smoke_arena.py"),
        "--input-artifact-dir", str(input_artifact_dir),
        "--shard-dir", str(shard_dir),
        "--output-dir", str(output_dir),
        *TINY_BUDGET_ARGS,
        *extra_args,
    ]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=300)


def test_full_run_writes_every_phase_output(input_artifact_dir, shard_dir, tmp_path):
    output_dir = tmp_path / "d8_out"
    result = _run_cli(input_artifact_dir, shard_dir, output_dir)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "AFTERMS_SMOKE_ARENA_COMPLETE"

    for rel in (
        "audit/immutable_input_manifest.json",
        "registry/run_registry.json",
        "arenas/model_arena.json",
        "generated_samples",
        "reference_samples",
        "statistics/one_dimensional_tests.json",
        "statistics/two_dimensional_tests.json",
        "statistics/ndimensional_c2st.json",
        "sample_matrices",
        "pz_diagnostics",
        "report/afterms_smoke_arena.md",
        "report/afterms_smoke_arena.json",
        "report/afterms_smoke_arena.csv",
    ):
        assert (output_dir / rel).exists(), f"missing {rel}"


def test_full_run_is_deterministic_across_two_invocations(input_artifact_dir, shard_dir, tmp_path, raw_pkl):
    before = raw_pkl.read_bytes()

    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    r1 = _run_cli(input_artifact_dir, shard_dir, out1)
    r2 = _run_cli(input_artifact_dir, shard_dir, out2)
    assert r1.returncode == 0, r1.stderr
    assert r2.returncode == 0, r2.stderr

    cmp = filecmp.dircmp(out1 / "generated_samples", out2 / "generated_samples")
    assert not cmp.diff_files, f"non-deterministic files: {cmp.diff_files}"
    assert not cmp.left_only and not cmp.right_only

    assert raw_pkl.read_bytes() == before  # D7 input untouched


def test_full_run_never_writes_physical_nll_for_quantile_run(input_artifact_dir, shard_dir, tmp_path):
    output_dir = tmp_path / "d8_out"
    _run_cli(input_artifact_dir, shard_dir, output_dir)
    registry_payload = json.loads((output_dir / "registry" / "run_registry.json").read_text())
    for row in registry_payload:
        if row["preprocessing_name"] == "quantile_normal_v0" or row["model_family"] == "legacy_normalizing_flow":
            assert row["physical_space_nll"] is None
