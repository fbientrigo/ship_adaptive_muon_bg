"""Deterministic component-label sampling for controlled D5 experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

IID_TARGET = "iid_target"
STRATIFIED_DIAGNOSTIC = "stratified_unweighted_diagnostic"
STRATIFIED_SELF_NORMALIZED_PROVISIONAL = "stratified_self_normalized_provisional"
SAMPLING_REGIMES = (IID_TARGET, STRATIFIED_DIAGNOSTIC, STRATIFIED_SELF_NORMALIZED_PROVISIONAL)


@dataclass(frozen=True)
class SamplingResult:
    """Sampled physical rows, exact component labels, weights, and provenance."""

    physical: np.ndarray
    component_id: np.ndarray
    sample_weight: np.ndarray
    manifest: Dict[str, Any]


def validate_sample_weight(weight: np.ndarray, n_rows: int) -> np.ndarray:
    """Return validated finite, nonnegative float64 weights with positive total.

    Parameters
    ----------
    weight:
        Candidate one-dimensional row weights.
    n_rows:
        Required number of rows.
    """

    value = np.asarray(weight, dtype=np.float64)
    if value.shape != (int(n_rows),):
        raise ValueError("sample_weight must have shape ({},)".format(n_rows))
    if not np.isfinite(value).all() or np.any(value < 0.0):
        raise ValueError("sample_weight must be finite and nonnegative")
    if float(value.sum()) <= 0.0:
        raise ValueError("sample_weight total must be positive")
    return np.ascontiguousarray(value)


def _stratified_rows(target, *, pdg_id: int, n: int, seed: int, rare_id: int,
                     rare_fraction: float) -> Tuple[np.ndarray, np.ndarray]:
    n_rare = int(round(n * rare_fraction))
    n_main = n - n_rare
    base = getattr(target, "_base", None)
    if base is None or not hasattr(base, "_components_for"):
        raise ValueError("stratification requires a controlled mixture target")
    components = base._components_for(pdg_id)
    rng = np.random.default_rng(int(seed))
    main_ids = np.asarray([i for i in range(len(components)) if i != rare_id])
    main_prob = np.asarray([components[i].weight for i in main_ids], dtype=np.float64)
    main_prob /= main_prob.sum()
    component = np.concatenate((
        rng.choice(main_ids, size=n_main, p=main_prob),
        np.full(n_rare, rare_id, dtype=np.int64),
    ))
    rng.shuffle(component)
    eps = rng.standard_normal((n, 5))
    physical = np.empty((n, 5), dtype=np.float64)
    for index, mixture_component in enumerate(components):
        mask = component == index
        physical[mask] = (
            mixture_component.mean + eps[mask] @ mixture_component._cholesky.T
        )
    transform = getattr(target, "_transform", None)
    if transform is not None:
        physical = transform.forward(physical)
    return np.ascontiguousarray(physical), np.ascontiguousarray(component)


def sample_controlled(
    target,
    *,
    pdg_id: int,
    n: int,
    seed: int,
    regime: str = IID_TARGET,
    sampling_rare_fraction: Optional[float] = None,
) -> SamplingResult:
    """Sample a controlled target under an explicit component-label regime.

    Stratification is supported only for targets exposing ``rare_mass`` and
    ``rare_component_id``. Provisional self-normalized weights are exact
    component-mass ratios and are not rescaled after assignment. The trainer
    still normalizes each minibatch by its own weight sum, so unbiased target-
    risk estimation is not established by this regime.
    """

    if regime not in SAMPLING_REGIMES:
        raise ValueError("unknown sampling regime {!r}".format(regime))
    target_mass = getattr(target, "rare_mass", None)
    if regime == IID_TARGET:
        batch = target.sample(n, pdg_id=pdg_id, seed=seed)
        physical, component = batch.physical, batch.component_id
        weight = np.ones(n, dtype=np.float64)
        sampling_mass = target_mass
        rare_id = (
            target.rare_component_id(pdg_id=pdg_id) if target_mass is not None else None
        )
    else:
        if target_mass is None:
            raise ValueError("stratified regimes require a labelled rare component")
        if sampling_rare_fraction is None:
            raise ValueError("sampling_rare_fraction is required for stratification")
        sampling_mass = float(sampling_rare_fraction)
        if not 0.0 < sampling_mass < 1.0:
            raise ValueError("sampling_rare_fraction must be strictly between 0 and 1")
        rare_id = target.rare_component_id(pdg_id=pdg_id)
        physical, component = _stratified_rows(
            target, pdg_id=pdg_id, n=n, seed=seed, rare_id=rare_id,
            rare_fraction=sampling_mass,
        )
        if regime == STRATIFIED_SELF_NORMALIZED_PROVISIONAL:
            weight = np.where(
                component == rare_id,
                float(target_mass) / sampling_mass,
                (1.0 - float(target_mass)) / (1.0 - sampling_mass),
            )
        else:
            weight = np.ones(n, dtype=np.float64)
    weight = validate_sample_weight(weight, n)
    total = float(weight.sum())
    ess = total * total / float(np.square(weight).sum())
    rare_count = None if rare_id is None else int(np.count_nonzero(component == rare_id))
    import hashlib

    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(physical, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(component, dtype=np.int64).tobytes())
    manifest = {
        "regime": regime,
            "diagnostic_only": regime == STRATIFIED_DIAGNOSTIC,
            "estimator_family": (
                "self_normalized_importance_weighted_minibatch"
                if regime == STRATIFIED_SELF_NORMALIZED_PROVISIONAL
                else ("unweighted_iid_target" if regime == IID_TARGET else "unweighted_stratified_diagnostic")
            ),
            "unbiasedness_status": (
                "not_established"
                if regime == STRATIFIED_SELF_NORMALIZED_PROVISIONAL
                else "not_applicable"
            ),
            "scientific_scope": (
                "provisional_target_estimator"
                if regime == STRATIFIED_SELF_NORMALIZED_PROVISIONAL
                else ("target_density" if regime == IID_TARGET else "diagnostic_capacity_only")
            ),
        "target_stratum_masses": (
            None if target_mass is None else {"main": 1.0 - float(target_mass), "rare": float(target_mass)}
        ),
        "sampling_stratum_masses": (
            None if sampling_mass is None else {"main": 1.0 - float(sampling_mass), "rare": float(sampling_mass)}
        ),
        "target_rare_mass": target_mass,
        "sampling_rare_fraction": sampling_mass,
        "stratum_weights": {
            "main": None if target_mass is None or sampling_mass is None else (
                1.0 if regime != STRATIFIED_SELF_NORMALIZED_PROVISIONAL else (1.0 - float(target_mass)) / (1.0 - sampling_mass)
            ),
            "rare": None if target_mass is None or sampling_mass is None else (
                1.0 if regime != STRATIFIED_SELF_NORMALIZED_PROVISIONAL else float(target_mass) / sampling_mass
            ),
        },
        "rare_weight": (
            None if target_mass is None or sampling_mass is None else
            (1.0 if regime != STRATIFIED_SELF_NORMALIZED_PROVISIONAL else float(target_mass) / sampling_mass)
        ),
        "main_weight": (
            None if target_mass is None or sampling_mass is None else
            (1.0 if regime != STRATIFIED_SELF_NORMALIZED_PROVISIONAL else
             (1.0 - float(target_mass)) / (1.0 - sampling_mass))
        ),
        "weight_normalization": "sum_weights",
        "weight_total": total,
        "effective_sample_size": ess,
        "ess_over_n": ess / n,
        "rare_count": rare_count,
        "component_labels_used_for_loss": True,
        "seed": int(seed),
        "dataset_hash": digest.hexdigest(),
    }
    return SamplingResult(
        np.ascontiguousarray(physical, dtype=np.float64),
        np.ascontiguousarray(component, dtype=np.int64),
        weight,
        manifest,
    )
