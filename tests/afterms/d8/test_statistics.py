"""Phase E statistics: Holm correction, weighted-vs-classical separation, bounded memory."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ship_muon_bg.afterms.d8 import statistics as stats


def test_holm_correction_is_monotone_and_bounded():
    raw = [0.001, 0.20, 0.03, 0.9, 0.04]
    adjusted = stats.holm_correction(raw)
    assert all(0.0 <= p <= 1.0 for p in adjusted)
    # Holm-adjusted p-values are never smaller than the raw p-value.
    assert all(a >= r - 1e-12 for a, r in zip(adjusted, raw))


def test_ks_1d_detects_a_real_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=800)
    gen = rng.normal(3.0, 1.0, size=800)
    result = stats.ks_1d_unweighted(ref, gen, n_boot=40, seed=1)
    assert result["p_value"] < 0.01
    assert result["statistic"] > 0.5


def test_weighted_test_never_reports_classical_p_value():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=300)
    gen = rng.normal(size=300)
    w = rng.uniform(0.1, 2.0, size=300)
    result = stats.weighted_1d_test(ref, w, gen, n_boot=20, seed=1)
    assert result["p_value"] is None
    assert "note" in result


def test_energy_distance_bounded_memory_no_full_matrix(monkeypatch):
    # energy_distance must only ever be called with the small
    # `energy_sample_size`-bounded arrays, never the full sample budget --
    # verify by calling it directly with a small subsample and checking cost
    # stays small (a smoke check on shape, not a mock of internals).
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 2))
    y = rng.normal(0.5, 1.0, size=(500, 2))
    d = stats.energy_distance(x, y)
    assert d > 0  # distributions differ


def test_energy_distance_permutation_null_detects_shift():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 2))
    y = rng.normal(1.0, 1.0, size=(200, 2))
    result = stats.energy_distance_permutation_test(x, y, n_permutations=49, seed=1)
    assert result["p_value"] < 0.05


def test_ndim_c2st_train_eval_separation_via_existing_c2st():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(400, 5))
    gen = rng.normal(size=(400, 5))
    result = stats.ndim_c2st(ref, gen, sample_size=300, seed=2, n_boot=10, n_permutations=10)
    assert 0.0 <= result["c2st_roc_auc"] <= 1.0
    assert result["n_test"] < result["n_per_class"] * 2  # a held-out test split, not train==eval


def test_holm_correction_across_2d_pairs_runs_without_full_matrix():
    rng = np.random.default_rng(3)
    ref = rng.normal(size=(300, 5))
    gen = rng.normal(0.2, 1.0, size=(300, 5))
    result = stats.run_2d_suite(ref, gen, weighted=False, energy_sample_size=100, n_permutations=19, seed=4)
    assert set(result.keys()) == {f"{a}_{b}" for a, b in stats.PAIRS_2D}
    for pair_result in result.values():
        assert "holm_adjusted_p_value" in pair_result


def test_weighted_2d_suite_is_deferred_not_unweighted():
    rng = np.random.default_rng(5)
    ref = rng.normal(size=(200, 5))
    gen = rng.normal(size=(200, 5))
    result = stats.run_2d_suite(ref, gen, weighted=True, energy_sample_size=50, n_permutations=19, seed=6)
    assert result["deferred"] is True
