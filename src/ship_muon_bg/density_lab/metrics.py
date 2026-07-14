"""Exact-target evaluation metrics (physical space, JSON-safe results).

Conventions: ``p`` is the exact controlled target, ``q`` is the fitted model.
All cross-model metrics are computed in physical coordinates. Metric functions
take plain NumPy arrays (or small callables) and return JSON-serializable
dicts. Optional scikit-learn is imported lazily inside :func:`c2st` only.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np


class MetricError(ValueError):
    """A metric precondition was violated (e.g. non-finite importance weights)."""


def _finite(name: str, array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise MetricError("{} contains non-finite values".format(name))
    return array


# --- 1. held-out NLL & 2. forward KL ---------------------------------------


def held_out_nll(q_log_prob_on_test: np.ndarray) -> Dict[str, Any]:
    """Physical-space held-out negative log-likelihood, ``mean(-log q(x))``."""

    values = np.asarray(q_log_prob_on_test, dtype=np.float64)
    finite = np.isfinite(values)
    return {
        "held_out_nll": float(-np.mean(values[finite])) if finite.any() else float("nan"),
        "n_test": int(values.size),
        "non_finite_count": int((~finite).sum()),
    }


def forward_kl(
    p_log_prob_on_test: np.ndarray, q_log_prob_on_test: np.ndarray
) -> Dict[str, Any]:
    """Monte-Carlo ``KL(p||q) = E_p[log p(x) - log q(x)]`` on target samples."""

    p = np.asarray(p_log_prob_on_test, dtype=np.float64)
    q = np.asarray(q_log_prob_on_test, dtype=np.float64)
    diff = p - q
    finite = np.isfinite(diff)
    return {
        "forward_kl": float(np.mean(diff[finite])) if finite.any() else float("nan"),
        "forward_kl_stderr": (
            float(np.std(diff[finite]) / math.sqrt(finite.sum()))
            if finite.sum() > 1
            else float("nan")
        ),
        "n_effective": int(finite.sum()),
        "non_finite_count": int((~finite).sum()),
    }


# --- 3. importance ESS/N ----------------------------------------------------


def importance_ess(
    p_log_prob_on_q_samples: np.ndarray,
    q_log_prob_on_q_samples: np.ndarray,
    *,
    catastrophic_threshold: float = 0.01,
) -> Dict[str, Any]:
    """Stable importance ESS/N for proposal ``q`` targeting ``p``.

    ``x ~ q``; ``log_w = log p(x) - log q(x)``. ESS/N is invariant to a constant
    log-weight shift, so weights are max-shifted for stability. Non-finite log
    weights are a hard failure.
    """

    log_p = _finite("p_log_prob_on_q_samples", p_log_prob_on_q_samples)
    log_q = _finite("q_log_prob_on_q_samples", q_log_prob_on_q_samples)
    log_w = log_p - log_q
    if not np.isfinite(log_w).all():
        raise MetricError("importance log weights are non-finite")
    n = log_w.size
    shifted = log_w - np.max(log_w)
    w = np.exp(shifted)
    sum_w = float(w.sum())
    sum_w2 = float((w * w).sum())
    ess_over_n = (sum_w * sum_w) / (n * sum_w2) if sum_w2 > 0 else 0.0
    normalized = w / sum_w if sum_w > 0 else w
    return {
        "ess_over_n": float(ess_over_n),
        "n_proposal": int(n),
        "max_normalized_weight": float(normalized.max()),
        "log_weight_min": float(log_w.min()),
        "log_weight_max": float(log_w.max()),
        "log_weight_range": float(log_w.max() - log_w.min()),
        "catastrophic": bool(ess_over_n < catastrophic_threshold),
    }


# --- 4. C2ST ----------------------------------------------------------------


def c2st(
    p_samples: np.ndarray,
    q_samples: np.ndarray,
    *,
    seed: int = 0,
    test_fraction: float = 0.5,
) -> Dict[str, Any]:
    """Classifier two-sample test (held-out accuracy + ROC AUC).

    Balanced classes, deterministic split and classifier random state, no
    train/eval reuse. Uses scikit-learn (optional ``lab`` dependency).
    """

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    p = np.asarray(p_samples, dtype=np.float64)
    q = np.asarray(q_samples, dtype=np.float64)
    n = min(p.shape[0], q.shape[0])
    p = p[:n]
    q = q[:n]
    x = np.vstack([p, q])
    y = np.concatenate([np.ones(n), np.zeros(n)])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(2 * n)
    x, y = x[perm], y[perm]
    n_test = int(round(2 * n * test_fraction))
    x_test, y_test = x[:n_test], y[:n_test]
    x_train, y_train = x[n_test:], y[n_test:]
    scaler = StandardScaler().fit(x_train)
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(scaler.transform(x_train), y_train)
    proba = clf.predict_proba(scaler.transform(x_test))[:, 1]
    accuracy = float(np.mean((proba >= 0.5) == (y_test == 1)))
    try:
        auc = float(roc_auc_score(y_test, proba))
    except ValueError:  # pragma: no cover - single-class test fold
        auc = float("nan")
    return {
        "c2st_accuracy": accuracy,
        "c2st_roc_auc": auc,
        "n_per_class": int(n),
        "n_test": int(n_test),
        "classifier": "logistic_regression_standardized",
        "seed": int(seed),
    }


# --- 5. component posterior mass diagnostics --------------------------------


def component_posterior_mass(
    target, physical: np.ndarray, *, pdg_id: int
) -> Dict[str, Any]:
    """Mean exact-target component posterior mass on a set of physical rows."""

    posterior = target.component_posterior(physical, pdg_id=pdg_id)
    return {
        "mean_component_posterior": posterior.mean(axis=0).tolist(),
        "n_components": int(posterior.shape[1]),
    }


# --- 8. tail quantile errors, 9. exceedance probability errors --------------


def tail_quantile_errors(
    p_samples: np.ndarray,
    q_samples: np.ndarray,
    *,
    quantiles: Sequence[float],
    column_names: Sequence[str],
) -> Dict[str, Any]:
    """Per-column quantile errors ``|Q_q - Q_p|`` at the requested quantiles."""

    p = np.asarray(p_samples, dtype=np.float64)
    q = np.asarray(q_samples, dtype=np.float64)
    out: Dict[str, Any] = {}
    for j, name in enumerate(column_names):
        col_errors = {}
        for quant in quantiles:
            qp = float(np.quantile(p[:, j], quant))
            qq = float(np.quantile(q[:, j], quant))
            col_errors[str(quant)] = {
                "target": qp,
                "model": qq,
                "abs_error": abs(qq - qp),
            }
        out[name] = col_errors
    return out


def exceedance_probability_errors(
    p_samples: np.ndarray,
    q_samples: np.ndarray,
    *,
    thresholds: Sequence[float],
    column_index: int = 2,
) -> Dict[str, Any]:
    """Errors in ``P(column > threshold)`` (default column: pz)."""

    p = np.asarray(p_samples, dtype=np.float64)[:, column_index]
    q = np.asarray(q_samples, dtype=np.float64)[:, column_index]
    out: Dict[str, Any] = {}
    for threshold in thresholds:
        pp = float(np.mean(p > threshold))
        pq = float(np.mean(q > threshold))
        out[str(threshold)] = {
            "target": pp,
            "model": pq,
            "abs_error": abs(pq - pp),
        }
    return out


# --- 10-12. sanity / hygiene diagnostics ------------------------------------


def non_finite_density_rate(q_log_prob: np.ndarray) -> Dict[str, Any]:
    values = np.asarray(q_log_prob, dtype=np.float64)
    non_finite = int((~np.isfinite(values)).sum())
    return {
        "non_finite_density_rate": float(non_finite / values.size),
        "non_finite_count": non_finite,
        "n": int(values.size),
    }


def support_violation_rate(
    q_samples: np.ndarray, *, pz_index: int = 2
) -> Dict[str, Any]:
    q = np.asarray(q_samples, dtype=np.float64)
    finite = np.isfinite(q).all(axis=1)
    pz_violation = int(np.sum(q[:, pz_index] <= 0.0))
    return {
        "non_finite_row_rate": float((~finite).mean()),
        "pz_nonpositive_rate": float(pz_violation / q.shape[0]),
        "pz_nonpositive_count": pz_violation,
        "n": int(q.shape[0]),
    }


def duplicate_diagnostics(
    q_samples: np.ndarray, *, atol: float = 1e-6
) -> Dict[str, Any]:
    """Exact and near-duplicate row diagnostics (sorted-neighbour heuristic)."""

    q = np.asarray(q_samples, dtype=np.float64)
    n = q.shape[0]
    # exact duplicates
    _, unique_counts = np.unique(q, axis=0, return_counts=True)
    exact_dup = int(n - unique_counts.size)
    # near-duplicates: nearest-neighbour distance under a KD-free scan on a
    # subsample (bounded cost, O(m^2) memory); count fraction below atol.
    m = min(n, 1000)
    sub = q[:m]
    dists = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    near = int(np.sum(dists.min(axis=1) < atol))
    return {
        "exact_duplicate_count": exact_dup,
        "exact_duplicate_rate": float(exact_dup / n),
        "near_duplicate_count_subsample": near,
        "near_duplicate_subsample_size": int(m),
        "atol": float(atol),
    }


# --- 6, 7.3 rare-mode diagnostics -------------------------------------------


def rare_mode_diagnostics(
    target,
    *,
    pdg_id: int,
    region_id: str,
    q_samples_physical: np.ndarray,
    q_log_prob_on_target_samples: np.ndarray,
    p_log_prob_on_target_samples: np.ndarray,
    target_rare_labels_mask: np.ndarray,
    n_train: int,
) -> Dict[str, Any]:
    """D5 rare-region diagnostics that do not use model-internal labels.

    - ``q_rare_region_mass``: fraction of model samples inside the exact rare
      region (mapped back through the exact target inverse);
    - ``soft_target_rare_posterior_mean_on_q_samples``: mean exact-target rare
      posterior on model samples;
    - rare-labelled target-row mean log q and forward-KL contribution;
    - probability of zero model rare samples given the target mass and n.
    """

    manifest = target.manifest()
    target_rare_mass = float(manifest["rare_mass"])
    rare_component_id = int(target.rare_component_id(pdg_id=pdg_id))

    q_samples = np.asarray(q_samples_physical, dtype=np.float64)
    finite_rows = np.isfinite(q_samples).all(axis=1)
    q_in_region = np.zeros(q_samples.shape[0], dtype=bool)
    if finite_rows.any():
        q_in_region[finite_rows] = target.region_mask(
            q_samples[finite_rows], pdg_id=pdg_id, region_id=region_id
        )
    observed_q_rare = int(q_in_region.sum())
    q_rare_region_mass = float(observed_q_rare / q_samples.shape[0])

    posterior = target.component_posterior(
        q_samples[finite_rows], pdg_id=pdg_id
    ) if finite_rows.any() else np.zeros((0, 1))
    soft_rare = (
        float(posterior[:, rare_component_id].mean())
        if posterior.shape[0] and rare_component_id < posterior.shape[1]
        else 0.0
    )

    mask = np.asarray(target_rare_labels_mask, dtype=bool)
    q_log = np.asarray(q_log_prob_on_target_samples, dtype=np.float64)
    p_log = np.asarray(p_log_prob_on_target_samples, dtype=np.float64)
    if mask.any():
        rare_mean_log_q = float(np.mean(q_log[mask][np.isfinite(q_log[mask])]))
        diff = (p_log[mask] - q_log[mask])
        rare_kl = float(np.mean(diff[np.isfinite(diff)]))
    else:
        rare_mean_log_q = float("nan")
        rare_kl = float("nan")

    # P(zero model rare samples | target mass, n): a model matching the target
    # mass would draw ~Binomial(n, target_rare_mass); the exact zero-count prob.
    n_q = q_samples.shape[0]
    prob_zero = float((1.0 - target_rare_mass) ** n_q)

    return {
        "region_id": region_id,
        "target_rare_mass": target_rare_mass,
        "q_rare_region_mass": q_rare_region_mass,
        "rare_region_mass_ratio": (
            q_rare_region_mass / target_rare_mass if target_rare_mass > 0 else float("nan")
        ),
        "observed_q_rare_sample_count": observed_q_rare,
        "soft_target_rare_posterior_mean_on_q_samples": soft_rare,
        "rare_target_rows_mean_log_q": rare_mean_log_q,
        "rare_target_rows_forward_kl_contribution": rare_kl,
        "n_rare_target_rows": int(mask.sum()),
        "probability_of_zero_rare_samples_given_target_mass_and_n": prob_zero,
        "zero_rare_samples_flag": bool(observed_q_rare == 0),
    }
