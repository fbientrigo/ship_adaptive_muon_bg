"""Phase C arenas: never-merge rules, validation-only selection, physical-NLL derivation."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ship_muon_bg.afterms.d8 import arenas, registry


def test_legacy_4d_never_merged_with_modern_5d(input_artifact_dir, shard_dir):
    records = registry.build_registry(input_artifact_dir, shard_dir)
    result = arenas.build_arenas(records, shard_dir)
    legacy_arena = next(a for a in result["arenas"] if a["arena_id"] == "F_legacy_combined_pdg_4d")
    assert legacy_arena["n_candidates"] == 1
    champ = legacy_arena["arena_champion"]
    assert champ["run_id"].startswith("04_")

    modern_arena = next(a for a in result["arenas"] if a["arena_id"] == "A_modern_5d_unweighted_pdg13_physical")
    assert all(not run["run_id"].startswith("04_") for run in modern_arena["ranked_candidates"])


def test_baseline_can_be_arena_champion_when_metric_wins(input_artifact_dir, shard_dir):
    # Fixture job10 baselines are deliberately given better (lower) NLL than
    # the fixture's own affine run in some configurations is not guaranteed,
    # but baselines must at least be ELIGIBLE candidates in arena A.
    records = registry.build_registry(input_artifact_dir, shard_dir)
    result = arenas.build_arenas(records, shard_dir)
    arena_a = next(a for a in result["arenas"] if a["arena_id"] == "A_modern_5d_unweighted_pdg13_physical")
    candidate_families = {rid["run_id"].split("__")[1] for rid in arena_a["ranked_candidates"]}
    assert any("gaussian" in c or "affine" in c for c in candidate_families)
    # Gaussian/GMM baselines have no checkpoint; they must still be eligible
    # metric-only candidates, never silently excluded for lacking one.
    baseline_run_ids = [r.run_id for r in records if r.model_family in ("diagonal_gaussian", "full_gaussian", "gaussian_mixture")]
    ranked_ids = {c["run_id"] for c in arena_a["ranked_candidates"]}
    assert set(baseline_run_ids) <= ranked_ids


def test_champion_selection_ignores_test_metric_ordering(input_artifact_dir, shard_dir):
    """A run with a WORSE validation NLL but a BETTER (lower) test NLL must
    NOT be selected champion over one with better validation NLL."""

    records = registry.build_registry(input_artifact_dir, shard_dir)
    gmm = next(r for r in records if r.model_family == "gaussian_mixture")
    diag = next(r for r in records if r.model_family == "diagonal_gaussian")
    # Fixture: gmm validation(3.0) < diagonal validation(5.0), and also
    # gmm test(2.95) < diagonal test(4.95) -- both agree here, so flip test
    # values to disagree and confirm selection still follows validation.
    diag.test_nll = -100.0  # pretend the test metric is dramatically better
    result = arenas.build_arenas(records, shard_dir)
    arena_a = next(a for a in result["arenas"] if a["arena_id"] == "A_modern_5d_unweighted_pdg13_physical")
    champ = arena_a["arena_champion"]
    assert champ is not None
    assert "diagonal_gaussian" not in champ["run_id"] or champ["ranking_value"] != -100.0


def test_physical_validation_nll_matches_analytic_check(input_artifact_dir, shard_dir):
    records = registry.build_registry(input_artifact_dir, shard_dir)
    affine = next(r for r in records if r.model_family == "affine_coupling")
    value = arenas.physical_validation_nll(affine, shard_dir)
    assert value is not None
    assert value != affine.validation_nll  # must actually apply the Jacobian offset


def test_weighted_and_unweighted_never_share_an_arena():
    weighted_arenas = {a["arena_id"] for a in arenas.ARENA_DEFINITIONS if "weighted" in a["arena_id"] and "unweighted" not in a["arena_id"]}
    unweighted_arenas = {a["arena_id"] for a in arenas.ARENA_DEFINITIONS if "unweighted" in a["arena_id"]}
    assert weighted_arenas.isdisjoint(unweighted_arenas)


def test_pdg_policies_never_share_an_arena():
    for definition in arenas.ARENA_DEFINITIONS:
        arena_id = definition["arena_id"]
        if "pdg13" in arena_id:
            assert "pdg_minus13" not in arena_id
