"""Guard-rail tests for the committed distilled datasets and the distiller.

Needs no FairShip, no ROOT, no GPU, and no network — only the committed
``data/distilled/`` files and NumPy. Protects three things:

1. the committed distilled ``.npz`` files stay loadable and contract-valid;
2. the distiller stays deterministic and range-preserving;
3. the distribution-comparison tools actually detect fidelity (KS ~ 0 for a
   uniform subsample, and a large KS when the distribution genuinely differs),
   and bimodality is preserved on a synthetic bimodal fixture.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from ship_muon_bg.data_contracts import (
    compare_distributions,
    ks_2samp,
    load_muon_npz,
    moment_summary,
    representative_subset,
    schema,
    validate_muon_array,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DISTILLED_DIR = os.path.join(REPO_ROOT, "data", "distilled")

DISTILLED_STEMS = ("muons_FullMC_distilled", "muonsFullMC_afterMS_distilled")

# Each distilled file targets ~40 MB; keep a generous ceiling so a re-distill with
# a slightly different row count doesn't spuriously fail.
PER_FILE_SIZE_CEILING = 45 * 1024 * 1024


def _path(stem, ext):
    return os.path.join(DISTILLED_DIR, f"{stem}{ext}")


@pytest.mark.parametrize("stem", DISTILLED_STEMS)
def test_distilled_loads_and_validates(stem):
    array = load_muon_npz(_path(stem, ".npz"))
    assert array.ndim == 2 and array.shape[1] == schema.N_COLUMNS
    # The one w == 0 row is a range anchor; allow it per the README.
    validate_muon_array(array, allow_zero_weight=True)


@pytest.mark.parametrize("stem", DISTILLED_STEMS)
def test_distilled_size_within_budget(stem):
    size = os.path.getsize(_path(stem, ".npz"))
    assert size <= PER_FILE_SIZE_CEILING, f"{stem}.npz is {size} bytes (> ceiling)"


@pytest.mark.parametrize("stem", DISTILLED_STEMS)
def test_manifest_and_report_match_committed_file(stem):
    array = load_muon_npz(_path(stem, ".npz"))
    with open(_path(stem, "_manifest.json"), encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["distilled_n_rows"] == array.shape[0]
    assert manifest["columns"] == list(schema.COLUMNS)

    with open(_path(stem, "_distribution_report.json"), encoding="utf-8") as handle:
        report = json.load(handle)
    assert report["distilled_n_rows"] == array.shape[0]
    # Committed evidence: every kinematic column is statistically indistinguishable.
    for col, stats in report["per_column"].items():
        assert stats["ks_statistic"] < 0.01, f"{stem} {col} KS too large"
        assert stats["range_preserved"] is True


# ----- distiller behaviour on a synthetic bimodal fixture -----


def _bimodal_full(n=200_000, seed=3):
    """A synthetic (N, 8) dataset with a clearly bimodal x column."""
    rng = np.random.default_rng(seed)
    px = rng.normal(0, 2, n)
    py = rng.normal(0, 0.3, n)
    pz = np.abs(rng.normal(25, 25, n))
    # Two well-separated lobes -> bimodal x.
    lobe = rng.integers(0, 2, n)
    x = np.where(lobe == 0, rng.normal(-3, 0.6, n), rng.normal(3, 0.6, n))
    y = rng.normal(0, 0.6, n)
    z = np.full(n, 28.9)
    ids = rng.choice([13.0, -13.0], size=n)
    w = rng.uniform(0.1, 5.0, n)
    return np.column_stack([px, py, pz, x, y, z, ids, w]).astype(np.float64)


def test_distiller_deterministic_and_range_preserving():
    full = _bimodal_full()
    a, idx_a = representative_subset(full, 20_000, seed=42)
    b, idx_b = representative_subset(full, 20_000, seed=42)
    assert np.array_equal(idx_a, idx_b) and np.array_equal(a, b)
    for col in range(full.shape[1]):
        assert a[:, col].min() == full[:, col].min()
        assert a[:, col].max() == full[:, col].max()


def test_distill_preserves_distribution_and_bimodality():
    full = _bimodal_full()
    distilled, idx = representative_subset(full, 20_000, seed=7)
    report = compare_distributions(full, distilled, distilled_indices=idx)

    x = report["per_column"]["x"]
    # x is bimodal in the full data and must stay bimodal in the distill.
    assert x["full"]["bimodality_coeff"] > 0.555
    assert x["distilled"]["bimodality_coeff"] > 0.555
    # Uniform subsample => tiny KS distance and matching moments on every column.
    for col, stats in report["per_column"].items():
        assert stats["ks_statistic"] < 0.05, f"{col} KS unexpectedly large"
        if stats["full"]["std"] == 0.0:
            continue  # constant column: relative shifts are undefined (nan)
        assert stats["rel_mean_shift_in_std"] < 0.1
        assert stats["rel_std_shift"] < 0.1


def test_ks_2samp_detects_a_real_difference():
    rng = np.random.default_rng(0)
    same_a = rng.normal(0, 1, 5000)
    same_b = rng.normal(0, 1, 5000)
    shifted = rng.normal(1.5, 1, 5000)
    d_same, p_same = ks_2samp(same_a, same_b)
    d_diff, p_diff = ks_2samp(same_a, shifted)
    assert d_same < 0.1 and p_same > 0.01  # same law: small D, non-significant
    assert d_diff > 0.4 and p_diff < 1e-6  # shifted: large D, significant


def test_moment_summary_flags_bimodality():
    rng = np.random.default_rng(1)
    unimodal = rng.normal(0, 1, 100_000)
    bimodal = np.concatenate([rng.normal(-3, 0.5, 50_000), rng.normal(3, 0.5, 50_000)])
    assert moment_summary(unimodal)["bimodality_coeff"] < 0.555
    assert moment_summary(bimodal)["bimodality_coeff"] > 0.555
