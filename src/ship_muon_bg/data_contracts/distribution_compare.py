"""Compare a distilled subset against its full dataset (pure NumPy).

The distilled datasets under ``data/distilled/`` are large uniform random subsets
of the full Muon NTuples v1.0 releases. "Distilled" means *the same distribution
with fewer rows*; this module produces the evidence for that claim — per-column
moments, a Sarle bimodality coefficient, and a two-sample Kolmogorov–Smirnov test
— so it can be recorded as an artifact and asserted in tests.

Everything here is NumPy + stdlib ``math`` only (no SciPy), keeping the core's
numpy-only guarantee. The KS statistic and its asymptotic p-value are implemented
directly.

Hypothesis-testing note: a distilled set is a *subset* of its full set, so a KS
test of distilled-vs-full is not two independent samples. The rigorous comparison
is distilled vs the **full complement** (``full \\ distilled``): two independent
uniform samples of the same underlying law, for which the KS p-value is meaningful.
:func:`compare_distributions` reports that complement test, and also the plain
full-vs-distilled moment table for readability. Comparisons are always within one
dataset — after-shield with after-shield, scoring-plane with scoring-plane — never
across the two.
"""

from __future__ import annotations

import math

import numpy as np

from . import schema

COMPARE_SCHEMA_VERSION = "0"

# Columns whose distribution fidelity we report. ``id`` (discrete PDG code) and
# ``w`` (weight) are excluded from the continuous-distribution comparison.
COMPARE_COLUMNS = schema.FEATURE_COLUMNS  # (px, py, pz, x, y, z)


def ks_2samp(a, b):
    """Two-sample Kolmogorov–Smirnov statistic and asymptotic p-value.

    Mirrors ``scipy.stats.ks_2samp`` (asymptotic mode) but in pure NumPy: the
    statistic is the maximum absolute difference between the two empirical CDFs,
    evaluated on the pooled sample via ``searchsorted``.

    Parameters
    ----------
    a, b : array-like
        1-D samples (need not be equal length).

    Returns
    -------
    (float, float)
        ``(D, p_value)``. ``D`` in ``[0, 1]``; larger means more different.
    """
    a = np.sort(np.asarray(a, dtype=np.float64).ravel())
    b = np.sort(np.asarray(b, dtype=np.float64).ravel())
    n, m = a.size, b.size
    if n == 0 or m == 0:
        raise ValueError("both samples must be non-empty")

    pooled = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, pooled, side="right") / n
    cdf_b = np.searchsorted(b, pooled, side="right") / m
    d = float(np.max(np.abs(cdf_a - cdf_b)))

    # Asymptotic Kolmogorov distribution p-value.
    en = math.sqrt(n * m / (n + m))
    lam = (en + 0.12 + 0.11 / en) * d
    p = _kolmogorov_sf(lam)
    return d, float(min(max(p, 0.0), 1.0))


def _kolmogorov_sf(lam):
    """Survival function of the Kolmogorov distribution, ``Q(lam)``.

    ``Q(lam) = 2 * sum_{j>=1} (-1)^{j-1} exp(-2 j^2 lam^2)``. Returns 1.0 for
    ``lam <= 0``. The series converges fast for the ``lam`` values seen here.
    """
    if lam <= 0.0:
        return 1.0
    total = 0.0
    for j in range(1, 101):
        term = 2.0 * ((-1) ** (j - 1)) * math.exp(-2.0 * j * j * lam * lam)
        total += term
        if abs(term) < 1e-12:
            break
    return total


def moment_summary(x):
    """Mean, std, skewness, excess kurtosis, and Sarle bimodality coefficient.

    The bimodality coefficient is ``BC = (skew^2 + 1) / kurtosis`` using the
    non-excess kurtosis; ``BC > 5/9 ≈ 0.555`` is the usual heuristic threshold
    for a bimodal (or otherwise non-unimodal) distribution.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    mean = float(np.mean(x))
    std = float(np.std(x))
    if std == 0.0:
        return {
            "mean": mean,
            "std": 0.0,
            "skew": 0.0,
            "excess_kurtosis": 0.0,
            "bimodality_coeff": float("nan"),
        }
    z = (x - mean) / std
    skew = float(np.mean(z**3))
    kurt = float(np.mean(z**4))  # non-excess
    bc = (skew**2 + 1.0) / kurt
    return {
        "mean": mean,
        "std": std,
        "skew": skew,
        "excess_kurtosis": kurt - 3.0,
        "bimodality_coeff": float(bc),
    }


def _complement(full, distilled_indices):
    """Rows of ``full`` not selected into the distilled subset."""
    mask = np.ones(full.shape[0], dtype=bool)
    mask[np.asarray(distilled_indices, dtype=int)] = False
    return full[mask]


def compare_distributions(full, distilled, *, distilled_indices=None, columns=COMPARE_COLUMNS):
    """Per-column fidelity of ``distilled`` against ``full``.

    For each column: the moment summary of full and of distilled (so mean/std/
    skew/kurtosis/bimodality can be eyeballed side by side), the absolute and
    std-relative deltas, and a KS test. When ``distilled_indices`` is given the KS
    test is the rigorous distilled-vs-complement test; otherwise it falls back to
    distilled-vs-full (documented as the weaker, dependent comparison).

    Returns a JSON-serializable dict.
    """
    full = np.asarray(full, dtype=np.float64)
    distilled = np.asarray(distilled, dtype=np.float64)

    if distilled_indices is not None:
        reference = _complement(full, distilled_indices)
        ks_reference = "complement"
    else:
        reference = full
        ks_reference = "full"

    per_column = {}
    for name in columns:
        idx = schema.COLUMN_INDEX[name]
        full_m = moment_summary(full[:, idx])
        dist_m = moment_summary(distilled[:, idx])
        d, p = ks_2samp(distilled[:, idx], reference[:, idx])
        std = full_m["std"] or float("nan")
        per_column[name] = {
            "full": full_m,
            "distilled": dist_m,
            "delta_mean": dist_m["mean"] - full_m["mean"],
            "delta_std": dist_m["std"] - full_m["std"],
            "rel_mean_shift_in_std": abs(dist_m["mean"] - full_m["mean"]) / std,
            "rel_std_shift": abs(dist_m["std"] - full_m["std"]) / std,
            "ks_statistic": d,
            "ks_pvalue": p,
            "full_min": float(np.min(full[:, idx])),
            "full_max": float(np.max(full[:, idx])),
            "distilled_min": float(np.min(distilled[:, idx])),
            "distilled_max": float(np.max(distilled[:, idx])),
            "range_preserved": bool(
                np.min(full[:, idx]) == np.min(distilled[:, idx])
                and np.max(full[:, idx]) == np.max(distilled[:, idx])
            ),
        }

    return {
        "schema_version": COMPARE_SCHEMA_VERSION,
        "ks_reference": ks_reference,
        "columns": list(columns),
        "full_n_rows": int(full.shape[0]),
        "distilled_n_rows": int(distilled.shape[0]),
        "per_column": per_column,
    }
