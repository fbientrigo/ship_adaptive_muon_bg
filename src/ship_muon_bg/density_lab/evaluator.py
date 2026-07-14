"""Model-agnostic evaluator (physical-space metrics).

Given an exact target, physical test data, a fitted feature pipeline, a fitted
model, an evaluation config and a seed, the evaluator produces the metric
bundle. It knows nothing about specific model classes: it only calls the
``DensityEstimator`` boundary (``log_prob`` / ``sample``) and the pipeline's
density accounting. All cross-model metrics are physical-space; feature-space
NLL is retained only as a debugging field.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..benchmarks import embed_physical_to_raw
from ..benchmarks.controlled_targets import TransformedControlledTarget
from ..data_contracts.feature_views import PHYSICAL_STATE_COLUMNS
from . import metrics as M


def _derived_seed(base_seed: int, salt: int) -> int:
    seq = np.random.SeedSequence([int(base_seed), int(salt)])
    return int(seq.generate_state(1)[0])


def _physical_q_log_prob(pipeline, model, physical_q, pdg_id) -> np.ndarray:
    raw = embed_physical_to_raw(physical_q, pdg_id=pdg_id, plane_z=0.0)
    normalized = pipeline.transform_raw(raw)
    normalized_lp = np.asarray(model.log_prob(normalized), dtype=np.float64)
    return pipeline.normalized_to_physical_log_prob(normalized_lp, raw)


def evaluate_run(
    *,
    target,
    pdg_id: int,
    test_physical: np.ndarray,
    test_rare_mask: Optional[np.ndarray],
    pipeline,
    model,
    evaluation,
    seed: int,
    n_train: int,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Return (metrics dict, model sample array in physical space)."""

    results: Dict[str, Any] = {"physical_space": True}

    # --- draw model samples once (physical space) ---
    n_samples = int(max(evaluation.ess_sample_count, evaluation.c2st_sample_count))
    sample_seed = _derived_seed(seed, 1)
    t0 = time.perf_counter()
    normalized_samples = model.sample(n_samples, seed=sample_seed)
    sample_seconds = time.perf_counter() - t0
    physical_q = pipeline.inverse_to_physical(normalized_samples)

    finite_rows = np.isfinite(physical_q).all(axis=1)
    physical_q_finite = physical_q[finite_rows]

    # --- q log-prob on model samples (for ESS) ---
    # model.log_prob on the normalized samples we drew is the normalized log q;
    # convert to physical using the samples' own pz.
    t1 = time.perf_counter()
    normalized_lp_samples = np.asarray(
        model.log_prob(normalized_samples), dtype=np.float64
    )
    logprob_seconds = time.perf_counter() - t1
    q_log_on_samples = pipeline.normalized_to_physical_log_prob(
        normalized_lp_samples, physical_q
    )

    # --- test-set (x ~ p) physical densities ---
    q_log_on_test = _physical_q_log_prob(pipeline, model, test_physical, pdg_id)
    p_log_on_test = target.log_prob(test_physical, pdg_id=pdg_id)

    results["held_out"] = M.held_out_nll(q_log_on_test)
    results["forward_kl"] = M.forward_kl(p_log_on_test, q_log_on_test)

    # --- importance ESS on model samples (finite subset) ---
    p_log_on_samples = np.full(physical_q.shape[0], -np.inf)
    if finite_rows.any():
        p_log_on_samples[finite_rows] = target.log_prob(
            physical_q_finite, pdg_id=pdg_id
        )
    ess_valid = finite_rows & np.isfinite(q_log_on_samples) & np.isfinite(p_log_on_samples)
    try:
        results["importance_ess"] = M.importance_ess(
            p_log_on_samples[ess_valid],
            q_log_on_samples[ess_valid],
            catastrophic_threshold=evaluation.catastrophic_ess_threshold,
        )
        results["importance_ess"]["n_excluded_non_finite"] = int((~ess_valid).sum())
    except M.MetricError as exc:
        results["importance_ess"] = {"error": str(exc), "catastrophic": True}

    # --- C2ST (physical space) ---
    n_c2st = min(evaluation.c2st_sample_count, test_physical.shape[0], physical_q_finite.shape[0])
    if n_c2st >= 10:
        results["c2st"] = M.c2st(
            test_physical[:n_c2st],
            physical_q_finite[:n_c2st],
            seed=_derived_seed(seed, 2),
        )

    # --- distributional shape metrics ---
    results["tail_quantile_errors"] = M.tail_quantile_errors(
        test_physical,
        physical_q_finite,
        quantiles=evaluation.tail_quantiles,
        column_names=PHYSICAL_STATE_COLUMNS,
    )
    results["exceedance_probability_errors"] = M.exceedance_probability_errors(
        test_physical,
        physical_q_finite,
        thresholds=evaluation.exceedance_pz_thresholds,
    )
    results["component_posterior_mass_on_test"] = M.component_posterior_mass(
        target, test_physical, pdg_id=pdg_id
    )

    # --- hygiene / support ---
    results["non_finite_density"] = M.non_finite_density_rate(q_log_on_test)
    results["support_violation"] = M.support_violation_rate(physical_q)
    results["duplicates"] = M.duplicate_diagnostics(
        physical_q_finite, atol=evaluation.near_duplicate_atol
    )

    # --- rare-mode diagnostics (D5) ---
    if (
        isinstance(target, TransformedControlledTarget)
        and target.declared_regions()
        and test_rare_mask is not None
    ):
        region_id = target.declared_regions()[0]
        results["rare_mode"] = M.rare_mode_diagnostics(
            target,
            pdg_id=pdg_id,
            region_id=region_id,
            q_samples_physical=physical_q,
            q_log_prob_on_target_samples=q_log_on_test,
            p_log_prob_on_target_samples=p_log_on_test,
            target_rare_labels_mask=test_rare_mask,
            n_train=n_train,
        )

    # --- throughput / capacity ---
    results["throughput"] = {
        "sample_rows": int(n_samples),
        "sample_seconds": float(sample_seconds),
        "sample_rows_per_second": float(n_samples / sample_seconds) if sample_seconds > 0 else None,
        "log_prob_rows": int(n_samples),
        "log_prob_seconds": float(logprob_seconds),
        "log_prob_rows_per_second": float(n_samples / logprob_seconds) if logprob_seconds > 0 else None,
    }
    results["parameter_count"] = int(model.parameter_count())

    # --- debugging-only feature-space NLL ---
    normalized_test = pipeline.transform_raw(
        embed_physical_to_raw(test_physical, pdg_id=pdg_id, plane_z=0.0)
    )
    feature_lp = np.asarray(model.log_prob(normalized_test), dtype=np.float64)
    results["debug_feature_space_nll"] = (
        float(-np.mean(feature_lp[np.isfinite(feature_lp)]))
        if np.isfinite(feature_lp).any()
        else float("nan")
    )

    return results, physical_q
