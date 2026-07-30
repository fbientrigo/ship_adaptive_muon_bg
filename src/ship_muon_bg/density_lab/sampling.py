"""Deterministic component-label sampling for controlled D5 experiments."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

IID_TARGET = "iid_target"
STRATIFIED_DIAGNOSTIC = "stratified_unweighted_diagnostic"
STRATIFIED_SELF_NORMALIZED_PROVISIONAL = "stratified_self_normalized_provisional"
# Fixed-composition Horvitz-Thompson estimator (Issue #17, arm C). Appended
# after the three legacy regimes; existing order and semantics are untouched.
STRATIFIED_HT_FIXED_COMPOSITION = "stratified_horvitz_thompson_fixed_composition"
SAMPLING_REGIMES = (
    IID_TARGET,
    STRATIFIED_DIAGNOSTIC,
    STRATIFIED_SELF_NORMALIZED_PROVISIONAL,
    STRATIFIED_HT_FIXED_COMPOSITION,
)

# -- regime resolution: (pool_law, estimator) -------------------------------
#
# ``pool_law`` describes how the *training partition* is drawn (uniformly
# from the target, or with an exactly fixed stratum composition).
# ``estimator`` describes the loss/weight law applied to minibatches sliced
# from that partition. The two axes are independent: a fixed-composition
# pool can be trained with an unweighted mean (diagnostic), a self-normalized
# ratio (legacy provisional), or -- only for arm C, which additionally
# fixes composition at the *minibatch* level via a ``MinibatchPlan`` -- a
# fixed-denominator Horvitz-Thompson estimator.

POOL_IID = "iid_target"
POOL_FIXED_COMPOSITION = "fixed_composition"
POOL_LAWS = (POOL_IID, POOL_FIXED_COMPOSITION)

ESTIMATOR_MINIBATCH_MEAN = "minibatch_mean"
ESTIMATOR_SELF_NORMALIZED = "self_normalized"
ESTIMATOR_HORVITZ_THOMPSON_FIXED = "horvitz_thompson_fixed"
ESTIMATORS = (
    ESTIMATOR_MINIBATCH_MEAN,
    ESTIMATOR_SELF_NORMALIZED,
    ESTIMATOR_HORVITZ_THOMPSON_FIXED,
)

_REGIME_RESOLUTION: Dict[str, Tuple[str, str]] = {
    IID_TARGET: (POOL_IID, ESTIMATOR_MINIBATCH_MEAN),
    STRATIFIED_DIAGNOSTIC: (POOL_FIXED_COMPOSITION, ESTIMATOR_MINIBATCH_MEAN),
    STRATIFIED_SELF_NORMALIZED_PROVISIONAL: (
        POOL_FIXED_COMPOSITION,
        ESTIMATOR_SELF_NORMALIZED,
    ),
    STRATIFIED_HT_FIXED_COMPOSITION: (
        POOL_FIXED_COMPOSITION,
        ESTIMATOR_HORVITZ_THOMPSON_FIXED,
    ),
}


def resolve_regime(regime: str) -> Tuple[str, str]:
    """Return ``(pool_law, estimator)`` for a named sampling regime.

    Pure lookup; raises :class:`ValueError` for an unknown regime string.
    """

    try:
        return _REGIME_RESOLUTION[regime]
    except KeyError as exc:
        raise ValueError("unknown sampling regime {!r}".format(regime)) from exc


# -- fixed-composition minibatch planner (arm C) -----------------------------
#
# Pure NumPy. Must never import torch, sklearn, matplotlib, or mlflow: this
# module is on the import path of ``ship_muon_bg.density_lab`` (see the
# package docstring's NumPy-only guarantee, enforced by
# ``tests/test_rare_aware_estimators.py::test_density_lab_import_stays_numpy_only``).

DENOMINATOR_FIXED_BATCH_SIZE = "fixed_batch_size"

_REPLACEMENT_MODES = ("without_replacement_within_epoch", "with_replacement")
_STEPS_PER_EPOCH_RULES = ("min_stratum_pass", "recycle_scarce_stratum")
_KNOWN_STRATA = ("main", "rare")


@dataclass(frozen=True)
class MinibatchPlan:
    """An explicit, pre-computed fixed-composition minibatch schedule.

    Every step's per-stratum row indices into the training partition are
    materialized up front from realized integer counts -- the trainer never
    re-derives composition, and the estimator never normalizes by a random
    weight sum (A4). ``incomplete_batch_rule`` is always ``"drop_last"`` in
    this implementation: ``steps_per_epoch`` is chosen so every step is a
    full, exactly-composed batch (no short trailing batch is ever planned).
    """

    batch_size: int
    counts_by_stratum: Dict[str, int]
    weights_by_stratum: Dict[str, float]
    steps_per_epoch: int
    replacement: str
    incomplete_batch_rule: str
    denominator_mode: str
    step_indices_by_stratum: Tuple[Dict[str, np.ndarray], ...]
    stratum_pool_sizes: Dict[str, int]
    stratum_recycled: Dict[str, bool]
    recycle_count_by_stratum: Dict[str, int]
    plan_hash: str

    def step_indices(self, step: int, *, shuffle_seed: Optional[int] = None) -> np.ndarray:
        """Concatenated row indices for one step.

        ``shuffle_seed``, if given, reorders rows within the batch. This is
        cosmetic only -- the fixed-denominator loss sums over rows regardless
        of order, so it never changes the estimator's value. Use the same
        ``shuffle_seed`` with :meth:`step_weights` to keep weights aligned.
        """

        per_stratum = self.step_indices_by_stratum[step]
        indices = np.concatenate([per_stratum[h] for h in self.counts_by_stratum])
        if shuffle_seed is not None:
            indices = np.random.default_rng(int(shuffle_seed)).permutation(indices)
        return indices

    def step_weights(self, step: int, *, shuffle_seed: Optional[int] = None) -> np.ndarray:
        """Per-row Horvitz-Thompson weights aligned with :meth:`step_indices`.

        With the same ``shuffle_seed``, a fresh ``default_rng`` reconstructs
        the identical permutation used by :meth:`step_indices` (the Fisher-
        Yates swap sequence depends only on the generator's state and the
        array length, never on array content), so weights and indices stay
        row-aligned without storing the permutation twice.
        """

        pieces = [
            np.full(
                self.step_indices_by_stratum[step][h].shape[0],
                self.weights_by_stratum[h],
                dtype=np.float64,
            )
            for h in self.counts_by_stratum
        ]
        weights = np.concatenate(pieces)
        if shuffle_seed is not None:
            order = np.random.default_rng(int(shuffle_seed)).permutation(weights.shape[0])
            weights = weights[order]
        return weights

    def to_manifest(self) -> Dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "counts_by_stratum": dict(self.counts_by_stratum),
            "weights_by_stratum": dict(self.weights_by_stratum),
            "steps_per_epoch": self.steps_per_epoch,
            "replacement": self.replacement,
            "incomplete_batch_rule": self.incomplete_batch_rule,
            "denominator_mode": self.denominator_mode,
            "stratum_pool_sizes": dict(self.stratum_pool_sizes),
            "stratum_recycled": dict(self.stratum_recycled),
            "recycle_count_by_stratum": dict(self.recycle_count_by_stratum),
            "plan_hash": self.plan_hash,
        }


def plan_fixed_composition_batches(
    *,
    component_id: np.ndarray,
    rare_id: int,
    target_stratum_masses: Mapping[str, float],
    batch_size: int,
    counts: Mapping[str, int],
    replacement: str = "without_replacement_within_epoch",
    steps_per_epoch_rule: str = "min_stratum_pass",
    seed: int,
    physical_weight: Optional[np.ndarray] = None,
) -> MinibatchPlan:
    """Plan an exact fixed-composition minibatch schedule (arm C).

    Builds, up front, explicit per-step per-stratum row indices into the
    training partition indexed by ``component_id`` (values ``rare_id`` vs.
    anything else -- the "main"/"rare" split used throughout D5), together
    with Horvitz-Thompson correction weights ``w_h = p_h * batch_size / m_h``
    computed from the *realized* integer ``counts`` (never a configured
    fraction; A3/A4). Raises rather than silently degrading on every
    scarcity or malformed-input condition in the rare-aware estimator
    contract; never recycles a scarce stratum unless
    ``steps_per_epoch_rule="recycle_scarce_stratum"`` is explicitly chosen.
    """

    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if replacement not in _REPLACEMENT_MODES:
        raise ValueError(
            "replacement must be one of {}, got {!r}".format(_REPLACEMENT_MODES, replacement)
        )
    if steps_per_epoch_rule not in _STEPS_PER_EPOCH_RULES:
        raise ValueError(
            "steps_per_epoch_rule must be one of {}, got {!r}".format(
                _STEPS_PER_EPOCH_RULES, steps_per_epoch_rule
            )
        )
    if physical_weight is not None:
        physical_weight_array = np.asarray(physical_weight, dtype=np.float64)
        if not np.allclose(physical_weight_array, 1.0, atol=1e-12, rtol=0.0):
            raise ValueError(
                "arm C (fixed-composition Horvitz-Thompson) requires unit physical "
                "row weights (assumption A6); physical MC-weight semantics are not "
                "implemented in this stage"
            )

    strata = tuple(target_stratum_masses.keys())
    unknown_strata = set(strata) - set(_KNOWN_STRATA)
    if unknown_strata:
        raise ValueError(
            "plan_fixed_composition_batches only supports strata {}; got unknown "
            "stratum names {}".format(_KNOWN_STRATA, sorted(unknown_strata))
        )
    if set(counts.keys()) != set(strata):
        raise ValueError(
            "counts strata {} must exactly match target_stratum_masses strata {}".format(
                sorted(counts.keys()), sorted(strata)
            )
        )
    mass_total = 0.0
    for stratum, mass in target_stratum_masses.items():
        mass = float(mass)
        if not np.isfinite(mass) or mass < 0.0 or mass > 1.0:
            raise ValueError("invalid stratum mass for {!r}: {}".format(stratum, mass))
        mass_total += mass
    if abs(mass_total - 1.0) > 1e-9:
        raise ValueError("target_stratum_masses must sum to 1.0, got {}".format(mass_total))

    counts_int: Dict[str, int] = {}
    for stratum in strata:
        m_h = counts[stratum]
        if not isinstance(m_h, (int, np.integer)) or isinstance(m_h, bool):
            raise ValueError("counts[{!r}] must be an int, got {!r}".format(stratum, m_h))
        m_h = int(m_h)
        if m_h < 0:
            raise ValueError("counts[{!r}] must be >= 0, got {}".format(stratum, m_h))
        if m_h == 0 and float(target_stratum_masses[stratum]) > 0.0:
            raise ValueError(
                "stratum {!r} has positive target mass {} but zero allocated count; "
                "every positive-mass stratum must be represented in a fixed-composition "
                "minibatch".format(stratum, target_stratum_masses[stratum])
            )
        counts_int[stratum] = m_h
    if sum(counts_int.values()) != batch_size:
        raise ValueError(
            "sum of counts {} != batch_size {}".format(sum(counts_int.values()), batch_size)
        )

    component_id = np.asarray(component_id, dtype=np.int64)
    rare_mask = component_id == int(rare_id)
    pool_indices = {"rare": np.flatnonzero(rare_mask), "main": np.flatnonzero(~rare_mask)}

    stratum_pool_sizes = {h: int(pool_indices[h].shape[0]) for h in strata}
    for stratum in strata:
        if stratum_pool_sizes[stratum] < counts_int[stratum]:
            raise ValueError(
                "stratum {!r} has {} available rows, fewer than the requested fixed "
                "count {}".format(stratum, stratum_pool_sizes[stratum], counts_int[stratum])
            )

    capacity = {
        h: (stratum_pool_sizes[h] // counts_int[h] if counts_int[h] > 0 else 0)
        for h in strata
    }
    if steps_per_epoch_rule == "min_stratum_pass":
        steps_per_epoch = min(capacity.values())
    else:  # "recycle_scarce_stratum"
        steps_per_epoch = max(capacity.values())
    if steps_per_epoch <= 0:
        raise ValueError(
            "steps_per_epoch would be 0 under rule {!r} with pool sizes {} and counts "
            "{}".format(steps_per_epoch_rule, stratum_pool_sizes, counts_int)
        )

    rng_by_stratum = {
        stratum: np.random.default_rng(int(child.generate_state(1)[0]))
        for stratum, child in zip(strata, np.random.SeedSequence(int(seed)).spawn(len(strata)))
    }

    step_indices_by_stratum: List[Dict[str, np.ndarray]] = [
        {} for _ in range(steps_per_epoch)
    ]
    stratum_recycled = {h: False for h in strata}
    recycle_count_by_stratum = {h: 0 for h in strata}

    for h in strata:
        m_h = counts_int[h]
        pool = pool_indices[h]
        rng = rng_by_stratum[h]
        if replacement == "with_replacement":
            for step in range(steps_per_epoch):
                step_indices_by_stratum[step][h] = rng.choice(pool, size=m_h, replace=True)
            continue
        # without_replacement_within_epoch: consume successive slices of a
        # fresh permutation of the stratum's pool; re-permute only when the
        # current permutation cannot supply the next full slice (i.e. only
        # under recycle_scarce_stratum, since min_stratum_pass never asks for
        # more steps than every stratum's own capacity).
        permutation = rng.permutation(pool)
        cursor = 0
        for step in range(steps_per_epoch):
            if cursor + m_h > permutation.shape[0]:
                permutation = rng.permutation(pool)
                cursor = 0
                stratum_recycled[h] = True
                recycle_count_by_stratum[h] += 1
            step_indices_by_stratum[step][h] = permutation[cursor:cursor + m_h]
            cursor += m_h

    weights_by_stratum: Dict[str, float] = {}
    for h in strata:
        a_h = counts_int[h] / float(batch_size)
        w_h = float(target_stratum_masses[h]) / a_h if a_h > 0 else 0.0
        if not np.isfinite(w_h) or w_h < 0.0:
            raise ValueError(
                "computed non-finite or negative weight for stratum {!r}: {}".format(h, w_h)
            )
        weights_by_stratum[h] = w_h

    frozen_steps = tuple(
        {h: np.ascontiguousarray(step[h], dtype=np.int64) for h in strata}
        for step in step_indices_by_stratum
    )

    digest = hashlib.sha256()
    digest.update(str(batch_size).encode())
    for h in strata:
        digest.update(h.encode())
        digest.update(str(counts_int[h]).encode())
        digest.update(repr(weights_by_stratum[h]).encode())
    digest.update(str(steps_per_epoch).encode())
    digest.update(replacement.encode())
    digest.update(steps_per_epoch_rule.encode())
    for step in frozen_steps:
        for h in strata:
            digest.update(step[h].tobytes())
    plan_hash = digest.hexdigest()

    return MinibatchPlan(
        batch_size=batch_size,
        counts_by_stratum=counts_int,
        weights_by_stratum=weights_by_stratum,
        steps_per_epoch=int(steps_per_epoch),
        replacement=replacement,
        incomplete_batch_rule="drop_last",
        denominator_mode=DENOMINATOR_FIXED_BATCH_SIZE,
        step_indices_by_stratum=frozen_steps,
        stratum_pool_sizes=stratum_pool_sizes,
        stratum_recycled=stratum_recycled,
        recycle_count_by_stratum=recycle_count_by_stratum,
        plan_hash=plan_hash,
    )


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


def _stratified_rows(
    target, *, pdg_id: int, n: int, seed: int, rare_fraction: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw a fixed-composition pool of ``n`` rows via the public stratum API.

    Uses ``target.sample_stratum`` (never private ``_base._components_for`` /
    ``_cholesky`` reach-through) so both the exact D5 transformed target and
    the pre-D4 base-stage target work identically.
    """

    n_rare = int(round(n * rare_fraction))
    n_main = n - n_rare
    if not hasattr(target, "sample_stratum"):
        raise ValueError("stratification requires a controlled mixture target")
    main_seed, rare_seed, shuffle_seed = (
        int(child.generate_state(1)[0])
        for child in np.random.SeedSequence(int(seed)).spawn(3)
    )
    main_batch = target.sample_stratum(
        n_main, pdg_id=pdg_id, stratum="main", seed=main_seed
    )
    rare_batch = target.sample_stratum(
        n_rare, pdg_id=pdg_id, stratum="rare", seed=rare_seed
    )
    physical = np.concatenate((main_batch.physical, rare_batch.physical), axis=0)
    component = np.concatenate(
        (main_batch.component_id, rare_batch.component_id), axis=0
    )
    order = np.random.default_rng(shuffle_seed).permutation(n)
    physical = physical[order]
    component = component[order]
    return np.ascontiguousarray(physical), np.ascontiguousarray(component)


SAMPLING_SCHEMA_VERSION = "1"

# Assumption identifiers for the arm-C (fixed-composition Horvitz-Thompson)
# unbiasedness claim; see docs/contracts/rare_aware_minibatch_estimators_v0.md.
# Recorded verbatim in every sampling manifest so a reader never has to infer
# which assumptions a run's claim depends on.
UNBIASEDNESS_ASSUMPTIONS_HT = (
    "A1_exact_exhaustive_mutually_exclusive_strata",
    "A2_unmodified_target_conditional_sampling",
    "A3_nonempty_fixed_allocation_before_draw",
    "A4_fixed_known_denominator_never_random_weight_sum",
    "A5_integrability_and_gradient_interchange",
    "A6_unit_physical_weights",
)

# Legacy/non-HT regimes carry no unbiasedness claim; this is an explicit,
# non-empty marker so "no assumptions recorded" is distinguishable from
# "arm C's assumption list happens to be empty".
UNBIASEDNESS_ASSUMPTIONS_NOT_APPLICABLE = ("not_applicable_to_this_estimator",)

_ESTIMATOR_FAMILY_BY_REGIME = {
    IID_TARGET: "unweighted_iid_target",
    STRATIFIED_DIAGNOSTIC: "unweighted_stratified_diagnostic",
    STRATIFIED_SELF_NORMALIZED_PROVISIONAL: "self_normalized_importance_weighted_minibatch",
    STRATIFIED_HT_FIXED_COMPOSITION: "fixed_composition_horvitz_thompson_minibatch",
}

_UNBIASEDNESS_STATUS_BY_REGIME = {
    IID_TARGET: "not_applicable",
    STRATIFIED_DIAGNOSTIC: "not_applicable",
    STRATIFIED_SELF_NORMALIZED_PROVISIONAL: "not_established",
    STRATIFIED_HT_FIXED_COMPOSITION: "established_under_stated_assumptions",
}

_SCIENTIFIC_SCOPE_BY_REGIME = {
    IID_TARGET: "target_density",
    STRATIFIED_DIAGNOSTIC: "diagnostic_capacity_only",
    STRATIFIED_SELF_NORMALIZED_PROVISIONAL: "provisional_target_estimator",
    STRATIFIED_HT_FIXED_COMPOSITION: "unbiased_target_risk_estimator_under_stated_assumptions",
}


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
    ``rare_component_id``. This function fixes the *training-partition pool*
    composition only (see ``resolve_regime``'s ``pool_law``); it never
    fixes per-minibatch composition and never applies a Horvitz-Thompson
    correction -- both are the trainer's ``MinibatchPlan`` concern
    (``plan_fixed_composition_batches``). Provisional self-normalized weights
    are exact component-mass ratios and are not rescaled after assignment.
    The legacy trainer path still normalizes each minibatch by its own
    weight sum, so unbiased target-risk estimation is not established by
    ``STRATIFIED_SELF_NORMALIZED_PROVISIONAL``. Component labels can
    influence that loss indirectly through deterministic weight assignment;
    the loss consumes the resulting weights, never the labels themselves.
    """

    if regime not in SAMPLING_REGIMES:
        raise ValueError("unknown sampling regime {!r}".format(regime))
    pool_law, estimator = resolve_regime(regime)
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
            target, pdg_id=pdg_id, n=n, seed=seed, rare_fraction=sampling_mass,
        )
        if regime == STRATIFIED_SELF_NORMALIZED_PROVISIONAL:
            weight = np.where(
                component == rare_id,
                float(target_mass) / sampling_mass,
                (1.0 - float(target_mass)) / (1.0 - sampling_mass),
            )
        else:
            # STRATIFIED_DIAGNOSTIC and STRATIFIED_HT_FIXED_COMPOSITION both
            # draw an unweighted fixed-composition pool at this layer. Arm C's
            # actual Horvitz-Thompson correction weights are computed from
            # *realized minibatch* counts by ``plan_fixed_composition_batches``,
            # never baked into this partition-level pool.
            weight = np.ones(n, dtype=np.float64)
    weight = validate_sample_weight(weight, n)
    total = float(weight.sum())
    ess = total * total / float(np.square(weight).sum())
    rare_count = None if rare_id is None else int(np.count_nonzero(component == rare_id))
    stratum_pool_counts = (
        None
        if rare_id is None
        else {
            "main": int(n - rare_count),
            "rare": int(rare_count),
        }
    )

    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(physical, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(component, dtype=np.int64).tobytes())
    is_self_normalized = regime == STRATIFIED_SELF_NORMALIZED_PROVISIONAL
    manifest = {
        "sampling_schema_version": SAMPLING_SCHEMA_VERSION,
        "regime": regime,
        "pool_law": pool_law,
        "estimator": estimator,
        "diagnostic_only": regime == STRATIFIED_DIAGNOSTIC,
        "estimator_family": _ESTIMATOR_FAMILY_BY_REGIME[regime],
        "unbiasedness_status": _UNBIASEDNESS_STATUS_BY_REGIME[regime],
        "unbiasedness_assumptions": (
            list(UNBIASEDNESS_ASSUMPTIONS_HT)
            if regime == STRATIFIED_HT_FIXED_COMPOSITION
            else list(UNBIASEDNESS_ASSUMPTIONS_NOT_APPLICABLE)
        ),
        "scientific_scope": _SCIENTIFIC_SCOPE_BY_REGIME[regime],
        "target_stratum_masses": (
            None if target_mass is None else {"main": 1.0 - float(target_mass), "rare": float(target_mass)}
        ),
        "sampling_stratum_masses": (
            None if sampling_mass is None else {"main": 1.0 - float(sampling_mass), "rare": float(sampling_mass)}
        ),
        "target_rare_mass": target_mass,
        "sampling_rare_fraction": sampling_mass,
        "stratum_pool_counts": stratum_pool_counts,
        "stratum_weights": {
            "main": None if target_mass is None or sampling_mass is None else (
                1.0 if not is_self_normalized else (1.0 - float(target_mass)) / (1.0 - sampling_mass)
            ),
            "rare": None if target_mass is None or sampling_mass is None else (
                1.0 if not is_self_normalized else float(target_mass) / sampling_mass
            ),
        },
        "rare_weight": (
            None if target_mass is None or sampling_mass is None else
            (1.0 if not is_self_normalized else float(target_mass) / sampling_mass)
        ),
        "main_weight": (
            None if target_mass is None or sampling_mass is None else
            (1.0 if not is_self_normalized else
             (1.0 - float(target_mass)) / (1.0 - sampling_mass))
        ),
        "weight_normalization": "sum_weights",
        "weight_total": total,
        "effective_sample_size": ess,
        "ess_over_n": ess / n,
        # Explicit namespace for training-weight ESS (never the model-vs-
        # target metrics.importance_ess.ess_over_n gate input).
        "training_weight_ess": ess,
        "training_weight_ess_over_n": ess / n,
        "rare_count": rare_count,
        "component_labels_used_for_sampling": regime != IID_TARGET,
        "component_labels_used_for_weight_assignment": is_self_normalized,
        "component_labels_used_for_diagnostic_slices": target_mass is not None,
        "component_labels_directly_consumed_by_loss": False,
        # Legacy field records indirect label influence through deterministic
        # weights. The loss itself consumes weights, not component labels.
        "component_labels_used_for_loss": is_self_normalized,
        "physical_weights_unit": True,
        "physical_weight_semantics": "not_implemented",
        "seed": int(seed),
        "dataset_hash": digest.hexdigest(),
    }
    return SamplingResult(
        np.ascontiguousarray(physical, dtype=np.float64),
        np.ascontiguousarray(component, dtype=np.int64),
        weight,
        manifest,
    )
