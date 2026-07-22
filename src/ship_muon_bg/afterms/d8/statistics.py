"""Phase E (D8 spec §12): 1D/2D/N-D distribution diagnostics at fixed, bounded budgets.

Never allocates a full `sample_size x sample_size` distance matrix (§12.2):
the 2D energy-distance test runs on the smaller `--energy-sample-size`
subsample. Weighted arenas never discard weights to get a convenient
classical p-value (§12.1/§12.2): the 1D test becomes a weighted-reference-
ECDF-vs-generated-ECDF supremum distance with its own bootstrap interval (no
attached KS p-value), and the 2D energy-distance test is marked deferred
rather than silently unweighted.

A non-rejection never means "equal" (§12.1); every real-vs-real baseline
exists to give the model-vs-reference statistic something to be judged
against, not to be a pass/fail gate on its own.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import ks_2samp

VARIABLES_1D = ("x", "y", "px", "py", "pz")
FEATURE_INDEX = {"px": 0, "py": 1, "pz": 2, "x": 3, "y": 4}
PAIRS_2D = (("x", "y"), ("x", "px"), ("x", "py"), ("y", "px"), ("y", "py"), ("px", "py"))


def holm_correction(p_values: Sequence[float]) -> List[float]:
    """Holm-Bonferroni step-down adjustment. Monotone non-decreasing output."""

    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=np.float64)
    running_max = 0.0
    for rank, idx in enumerate(order):
        raw = p_values[idx] * (m - rank)
        running_max = max(running_max, raw)
        adjusted[idx] = min(1.0, running_max)
    return adjusted.tolist()


def _bootstrap_statistic(values_a: np.ndarray, values_b: np.ndarray, statistic_fn, n_boot: int, seed: int) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n_a, n_b = values_a.shape[0], values_b.shape[0]
    boot_stats = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx_a = rng.integers(0, n_a, size=n_a)
        idx_b = rng.integers(0, n_b, size=n_b)
        boot_stats[i] = statistic_fn(values_a[idx_a], values_b[idx_b])
    return float(np.quantile(boot_stats, 0.025)), float(np.quantile(boot_stats, 0.975))


def ks_1d_unweighted(reference: np.ndarray, generated: np.ndarray, *, n_boot: int, seed: int) -> Dict[str, Any]:
    stat, p_value = ks_2samp(reference, generated)
    ci_lo, ci_hi = _bootstrap_statistic(
        reference, generated, lambda a, b: float(ks_2samp(a, b).statistic), n_boot, seed,
    )
    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "bootstrap_ci_95": [ci_lo, ci_hi],
        "n_reference": int(reference.shape[0]),
        "n_generated": int(generated.shape[0]),
    }


def weighted_ecdf_sup_distance(reference: np.ndarray, weights: np.ndarray, generated: np.ndarray) -> float:
    """Weighted-reference-ECDF vs unweighted-generated-ECDF supremum distance.

    Generated samples carry no natural importance weight (they are draws
    from the model, not re-weighted real events), so only the reference side
    is weighted -- this is deliberately NOT a classical two-sample KS
    statistic and must never be reported with a classical KS p-value.
    """

    order = np.argsort(reference)
    ref_sorted = reference[order]
    w_sorted = np.asarray(weights, dtype=np.float64)[order]
    w_cum = np.cumsum(w_sorted)
    w_total = w_cum[-1]
    ref_ecdf_at = w_cum / w_total

    gen_sorted = np.sort(generated)
    grid = np.union1d(ref_sorted, gen_sorted)

    ref_ecdf = np.searchsorted(ref_sorted, grid, side="right")
    ref_ecdf = np.where(ref_ecdf > 0, w_cum[np.clip(ref_ecdf - 1, 0, len(w_cum) - 1)] / w_total, 0.0)
    gen_ecdf = np.searchsorted(gen_sorted, grid, side="right") / gen_sorted.shape[0]
    return float(np.max(np.abs(ref_ecdf - gen_ecdf)))


def weighted_1d_test(reference: np.ndarray, weights: np.ndarray, generated: np.ndarray, *, n_boot: int, seed: int) -> Dict[str, Any]:
    stat = weighted_ecdf_sup_distance(reference, weights, generated)

    rng = np.random.default_rng(seed)
    n_ref, n_gen = reference.shape[0], generated.shape[0]
    boot = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx_ref = rng.integers(0, n_ref, size=n_ref)
        idx_gen = rng.integers(0, n_gen, size=n_gen)
        boot[i] = weighted_ecdf_sup_distance(reference[idx_ref], weights[idx_ref], generated[idx_gen])

    return {
        "statistic": stat,
        "p_value": None,  # deliberately absent: not a classical KS statistic
        "bootstrap_ci_95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "n_reference": int(n_ref),
        "n_generated": int(n_gen),
        "note": "weighted-reference-ECDF vs unweighted-generated-ECDF supremum distance; no classical KS p-value attached",
    }


def energy_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Bounded-memory two-sample energy distance (Szekely-Rizzo).

    Caller is responsible for bounding `x`/`y` to `--energy-sample-size`
    before calling this -- never pass the full `--sample-size` arrays.
    """

    def _pairwise_mean(a: np.ndarray, b: np.ndarray) -> float:
        dists = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
        return float(np.mean(dists))

    a_a = _pairwise_mean(x, x)
    b_b = _pairwise_mean(y, y)
    a_b = _pairwise_mean(x, y)
    return float(2.0 * a_b - a_a - b_b)


def energy_distance_permutation_test(x: np.ndarray, y: np.ndarray, *, n_permutations: int, seed: int) -> Dict[str, Any]:
    observed = energy_distance(x, y)
    pooled = np.concatenate([x, y], axis=0)
    n_x = x.shape[0]
    rng = np.random.default_rng(seed)
    perm_stats = np.empty(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        perm = rng.permutation(pooled.shape[0])
        perm_stats[i] = energy_distance(pooled[perm[:n_x]], pooled[perm[n_x:]])
    p_value = float((np.count_nonzero(perm_stats >= observed) + 1) / (n_permutations + 1))
    return {
        "statistic": observed,
        "p_value": p_value,
        "n_permutations": int(n_permutations),
        "n_x": int(x.shape[0]),
        "n_y": int(y.shape[0]),
    }


def ndim_c2st(reference: np.ndarray, generated: np.ndarray, *, sample_size: int, seed: int, n_boot: int, n_permutations: int) -> Dict[str, Any]:
    from ship_muon_bg.density_lab.metrics import c2st

    n = min(sample_size, reference.shape[0], generated.shape[0])
    ref = reference[:n]
    gen = generated[:n]
    result = c2st(ref, gen, seed=seed)

    rng = np.random.default_rng(seed)
    boot_auc = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx_ref = rng.integers(0, n, size=n)
        idx_gen = rng.integers(0, n, size=n)
        boot_auc[i] = c2st(ref[idx_ref], gen[idx_gen], seed=seed)["c2st_roc_auc"]

    pooled = np.concatenate([ref, gen], axis=0)
    perm_auc = np.empty(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        perm = rng.permutation(pooled.shape[0])
        perm_auc[i] = c2st(pooled[perm[:n]], pooled[perm[n:2 * n]], seed=seed)["c2st_roc_auc"]
    p_value = float((np.count_nonzero(np.abs(perm_auc - 0.5) >= abs(result["c2st_roc_auc"] - 0.5)) + 1) / (n_permutations + 1))

    return {
        **result,
        "bootstrap_ci_95": [float(np.quantile(boot_auc, 0.025)), float(np.quantile(boot_auc, 0.975))],
        "permutation_p_value": p_value,
        "n_permutations": int(n_permutations),
    }


def real_vs_real_baseline_1d(subset_a: np.ndarray, subset_b: np.ndarray, *, n_boot: int, seed: int) -> Dict[str, Any]:
    return ks_1d_unweighted(subset_a, subset_b, n_boot=n_boot, seed=seed)


def real_vs_real_baseline_ndim_c2st(subset_a: np.ndarray, subset_b: np.ndarray, *, sample_size: int, seed: int, n_boot: int, n_permutations: int) -> Dict[str, Any]:
    return ndim_c2st(subset_a, subset_b, sample_size=sample_size, seed=seed, n_boot=n_boot, n_permutations=n_permutations)


def within_real_vs_real_band(value: float, baseline_ci: Sequence[float]) -> bool:
    return baseline_ci[0] <= value <= baseline_ci[1]


def run_1d_suite(
    reference: np.ndarray, generated: np.ndarray, *, weighted: bool, weights: Optional[np.ndarray],
    n_boot: int, seed: int,
) -> Dict[str, Any]:
    results = {}
    p_values = []
    var_order = []
    for var in VARIABLES_1D:
        idx = FEATURE_INDEX[var]
        ref_col = reference[:, idx]
        gen_col = generated[:, idx]
        if weighted:
            results[var] = weighted_1d_test(ref_col, weights, gen_col, n_boot=n_boot, seed=seed)
        else:
            results[var] = ks_1d_unweighted(ref_col, gen_col, n_boot=n_boot, seed=seed)
            p_values.append(results[var]["p_value"])
            var_order.append(var)
    if p_values:
        adjusted = holm_correction(p_values)
        for var, adj in zip(var_order, adjusted):
            results[var]["holm_adjusted_p_value"] = adj
    return results


def run_2d_suite(
    reference: np.ndarray, generated: np.ndarray, *, weighted: bool, energy_sample_size: int,
    n_permutations: int, seed: int,
) -> Dict[str, Any]:
    if weighted:
        return {
            "deferred": True,
            "reason": (
                "no verified weighted two-sample energy-distance estimator is implemented; discarding "
                "weights to get a convenient p-value is explicitly disallowed (D8 spec §12.2)"
            ),
        }

    results = {}
    p_values = []
    pair_order = []
    for a, b in PAIRS_2D:
        idx_a, idx_b = FEATURE_INDEX[a], FEATURE_INDEX[b]
        n = min(energy_sample_size, reference.shape[0], generated.shape[0])
        ref_pair = reference[:n][:, [idx_a, idx_b]]
        gen_pair = generated[:n][:, [idx_a, idx_b]]
        key = f"{a}_{b}"
        results[key] = energy_distance_permutation_test(ref_pair, gen_pair, n_permutations=n_permutations, seed=seed)
        p_values.append(results[key]["p_value"])
        pair_order.append(key)
    adjusted = holm_correction(p_values)
    for key, adj in zip(pair_order, adjusted):
        results[key]["holm_adjusted_p_value"] = adj
    return results
