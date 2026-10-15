import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "run_afterms_d9_model_arena.py"
ARENA_ROOT = REPO_ROOT / "artifacts" / "afterms_d9_model_arena_v0"
NAMED_DIR = ARENA_ROOT / "named"

pytestmark = pytest.mark.skipif(
    not (ARENA_ROOT / "arena_manifest.json").exists(),
    reason="requires the existing afterms_d9_model_arena_v0 CUDA scout artifacts",
)


def _run(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=120,
    )


def _hash_run_evidence_files():
    digests = {}
    for path in sorted((ARENA_ROOT / "runs").rglob("*.json")):
        digests[str(path.relative_to(ARENA_ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    digests["arena_manifest.json"] = hashlib.sha256((ARENA_ROOT / "arena_manifest.json").read_bytes()).hexdigest()
    return digests


def test_relabel_existing_exits_zero_and_performs_no_training():
    result = _run("--relabel-existing")
    assert result.returncode == 0, result.stderr
    assert "Named alias reports written" in result.stdout


def test_relabel_existing_does_not_modify_original_run_evidence():
    before = _hash_run_evidence_files()
    result = _run("--relabel-existing")
    assert result.returncode == 0, result.stderr
    after = _hash_run_evidence_files()
    assert before == after


def test_relabel_existing_is_deterministic():
    _run("--relabel-existing")
    first = {p.name: p.read_bytes() for p in NAMED_DIR.glob("*")}
    _run("--relabel-existing")
    second = {p.name: p.read_bytes() for p in NAMED_DIR.glob("*")}
    assert first == second


def test_named_outputs_exist():
    _run("--relabel-existing")
    expected = {
        "alias_snapshot.json", "run_inventory_named.json", "run_inventory_named.csv",
        "per_track_model_comparison.md", "per_track_model_comparison.csv",
        "affine_capacity_scout_summary.md",
    }
    actual = {p.name for p in NAMED_DIR.glob("*")}
    assert expected <= actual


def test_run_inventory_retains_internal_candidate_ids():
    _run("--relabel-existing")
    rows = json.loads((NAMED_DIR / "run_inventory_named.json").read_text(encoding="utf-8"))
    assert len(rows) == 30
    for row in rows:
        assert row["internal_candidate_id"].startswith(row["legacy_base_candidate_id"])
        assert "__arena_cap_" in row["internal_candidate_id"]


def test_run_inventory_pairs_internal_id_with_human_readable_alias():
    _run("--relabel-existing")
    rows = json.loads((NAMED_DIR / "run_inventory_named.json").read_text(encoding="utf-8"))
    for row in rows:
        assert row["model_config_id"].startswith("NF_AC_")
        assert row["track_id"].startswith("TRK_")
        assert row["model_config_id"] in row["display_name"]
        assert row["track_id"] in row["display_name"]


def test_alias_snapshot_covers_all_six_tracks():
    _run("--relabel-existing")
    snapshot = json.loads((NAMED_DIR / "alias_snapshot.json").read_text(encoding="utf-8"))
    track_ids = {record["track_id"] for record in snapshot["base_candidates"].values()}
    assert track_ids == {
        "TRK_PDG13_UW_ID", "TRK_PDG13_UW_LOGPZ", "TRK_PDGM13_UW_LOGPZ",
        "TRK_PDGM13_UW_ID", "TRK_PDG13_W_ID", "TRK_PDGM13_W_ID",
    }


def test_no_global_cross_track_ranking_field_present():
    _run("--relabel-existing")
    rows = json.loads((NAMED_DIR / "run_inventory_named.json").read_text(encoding="utf-8"))
    for row in rows:
        assert "global_rank" not in row
        assert "cross_track_rank" not in row
    summary = (NAMED_DIR / "affine_capacity_scout_summary.md").read_text(encoding="utf-8")
    assert "No global cross-track ranking is produced" in summary


def test_per_track_report_ranks_within_track_only():
    _run("--relabel-existing")
    md = (NAMED_DIR / "per_track_model_comparison.md").read_text(encoding="utf-8")
    # Each track gets its own "## Track TRK_..." section with its own rank-1..5 table.
    assert md.count("## Track TRK_") == 6


def test_variant_model_config_ids_come_from_recorded_architecture():
    """Required test 13: cross-check one variant's alias against the actual
    per-run training_config.json snapshot on disk (not a scale-recomputed guess)."""

    _run("--relabel-existing")
    variant_dir = ARENA_ROOT / "runs" / "A1_capacity_medium_identity_pdg13_unweighted__arena_cap_0.75x" / "seed_20260720"
    recorded = json.loads((variant_dir / "training_config.json").read_text(encoding="utf-8"))
    arch = recorded["architecture"]
    expected_config_id = f"NF_AC_b{arch['number_of_blocks']:02d}_w{arch['hidden_width']:03d}_d{arch['hidden_depth']:02d}"

    rows = json.loads((NAMED_DIR / "run_inventory_named.json").read_text(encoding="utf-8"))
    row = next(r for r in rows if r["internal_candidate_id"] == recorded["candidate_id"])
    assert row["model_config_id"] == expected_config_id
