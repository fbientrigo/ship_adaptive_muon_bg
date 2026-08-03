"""D9 -- utility-guided conditional per-epoch direct sampling (v0 fixture arena).

This module extends the verified conditional-charge nominal sampling line
(``density_lab.conditional_charge``, v1) with a *within-charge* synthetic
utility tilt. It changes exactly one thing relative to v1: the per-epoch
categorical law each charge's training rows are drawn from. Everything else
in the v1 contract is preserved unchanged and re-used, not re-implemented:

- the physical charge convention ``PDG 13 (mu-) -> -1``, ``PDG -13 (mu+) -> +1``
  (:data:`~.conditional_charge.MUON_ELECTRIC_CHARGE_SIGN_BY_PDG`);
- one shared conditional affine-coupling flow with ``condition_dim=1``;
- exact train-only macro-weighted preprocessing
  (:meth:`~.feature_pipeline.FittedFeaturePipeline.fit_macro_weighted`), fit
  once per (split, model seed) from the *nominal* measure and shared by every
  variant;
- deterministic per-epoch direct sampling with equal draw counts per charge;
- ordinary unweighted NLL after direct sampling (no ``w_i`` and no ``r_i``
  ever reaches the loss);
- exact weighted validation without resampling;
- a closed test payload (never materialized, never used for thresholds);
- train-only, charge-specific ``B_toy`` thresholds;
- no hard charge symmetry.

Mathematics
-----------

For each charge ``c`` the nominal empirical measure is unchanged::

    pi_i^(0,c) = w_i / sum_{j: C_j = c} w_j

The synthetic utility multiplier is the repository's single canonical
definition (:func:`~.utility_tilt.h_alpha_delta`), never re-derived here::

    r_i(delta, alpha) = [delta + (1 - delta) * U_i] ** alpha

with the binary synthetic utility ``U_A = 1[x_i in B_toy,c]``
(:func:`~.utility_tilt.compute_utility_a`) and the train-fitted, charge
specific region ``B_toy,c = 1[pT > t_pT,c and R_xy < t_R,c]``. The tilted
within-charge law is::

    pi_i^(U,c) = w_i * r_i / sum_{j: C_j = c} w_j * r_j

The macro charge prior stays balanced at ``P(C = 13) = P(C = -13) = 1/2``:
every epoch draws exactly ``draws_per_epoch_per_charge`` rows for each
charge, so the total tilted mass of one charge can never shift the 50/50
charge allocation. The tilt acts strictly *within* a charge.

Scope
-----

``B_toy`` is a project-defined synthetic diagnostic region, never a validated
FairShip danger metric; ``U_A`` is a synthetic utility score, never a
calibrated physical acceptance probability. This arena produces no physical
rate estimate and makes no FairShip acceptance claim -- FairShip/GEANT4
remains the final downstream oracle.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import statistics
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..data_contracts import schema
from ..data_contracts.hashing import dataset_hash
from ..data_contracts.feature_views import FeatureView, IDENTITY_CARTESIAN_VIEW_ID
from .config import canonical_hash
from .conditional_charge import (
    CONDITION_FIELD_NAME,
    MUON_ELECTRIC_CHARGE_SIGN_BY_PDG,
    muon_electric_charge_sign,
)
from .empirical import (
    EmpiricalDatasetSpec,
    EmpiricalTrainValidationDataset,
    build_empirical_train_validation_dataset,
)
from .feature_pipeline import FittedFeaturePipeline
from .utility_tilt import (
    NOMINAL_PHYSICAL_VARIANT_ID,
    NOMINAL_SAMPLING_REGIME,
    TILT_CONFIG_BY_ID,
    TILT_SAMPLING_REGIME,
    AliasSampler,
    TiltConfig,
    ToyThresholds,
    UtilityTiltError,
    compact_distribution_summary,
    compute_b_toy,
    compute_utility_a,
    empirical_draw_diagnostics,
    fit_toy_thresholds,
    make_tilt_id,
    nominal_prevalence,
    pi_nominal,
    pi_tilt,
    theoretical_concentration_diagnostics,
    theoretical_tilted_prevalence,
    validate_arena_variant_ids,
)

CONDITIONAL_UTILITY_SCHEMA_VERSION = "1"

# The v0 conditional arena: the un-tilted nominal arm plus exactly three
# binary-utility (U-A) tilt configurations. Variant ids are the repository's
# canonical ids (``utility_tilt.make_tilt_id``), never new strings -- the
# tilt grid itself is untouched. No probabilistic U-P variant enters this
# gate and the historical 20-variant grid is never materialized here.
CONDITIONAL_UTILITY_ARENA_TILT_MODES_DELTAS_ALPHAS: Tuple[Tuple[str, float, float], ...] = (
    ("UA", 0.9, 4),
    ("UA", 0.9, 8),
    ("UA", 0.1, 1),
)
CONDITIONAL_UTILITY_ARENA_TILT_IDS: Tuple[str, ...] = tuple(
    make_tilt_id(mode, delta, alpha)
    for mode, delta, alpha in CONDITIONAL_UTILITY_ARENA_TILT_MODES_DELTAS_ALPHAS
)
CONDITIONAL_UTILITY_ARENA_VARIANT_IDS: Tuple[str, ...] = (
    NOMINAL_PHYSICAL_VARIANT_ID,
) + CONDITIONAL_UTILITY_ARENA_TILT_IDS

CONDITIONAL_UTILITY_MODEL_SEEDS: Tuple[int, ...] = (11, 12, 13)

# Task-facing short label -> canonical repository variant id. The task names
# the un-tilted arm "NOMINAL"; the repository's canonical id for exactly that
# arm is NOMINAL_PHYSICAL, and that canonical id is what every artifact,
# hash, and seed derivation uses.
VARIANT_ID_ALIASES: Dict[str, str] = {"NOMINAL": NOMINAL_PHYSICAL_VARIANT_ID}

CONDITIONAL_UTILITY_SAMPLING_REGIME_BY_KIND: Dict[str, str] = {
    "nominal": "conditional_macro_balanced_physical_nominal_direct_per_epoch",
    "tilt": "conditional_macro_balanced_utility_tilted_direct_per_epoch",
}

# Wilson / normal two-sided 95% coverage constant.
WILSON_Z_95 = 1.959963984540054

_W_COLUMN = schema.COLUMN_INDEX["w"]
_PHYSICAL_FEATURE_NAMES: Tuple[str, ...] = ("px", "py", "pz", "x", "y")

_ALLOWED_CONFIG_KEYS = {
    "schema_version",
    "experiment_id",
    "description",
    "fixture_only",
    "dataset_path",
    "pdg_ids",
    "seed",
    "model_seeds",
    "variants",
    "max_rows_per_charge",
    "draws_per_epoch_per_charge",
    "validation_fraction",
    "test_fraction",
    "generated_sample_count_per_charge",
    "feature_view",
    "model",
}


class ConditionalUtilityError(ValueError):
    """An invalid or unsafe conditional utility-tilt arena configuration."""


# --- variant resolution -------------------------------------------------------


def resolve_variant_id(variant_id: str) -> str:
    """Map a task-facing label onto the canonical repository variant id."""

    return VARIANT_ID_ALIASES.get(str(variant_id), str(variant_id))


def validate_conditional_utility_variant_ids(
    variant_ids: Sequence[str],
) -> Tuple[str, ...]:
    """Resolve, then reject empty / duplicate / unknown / out-of-gate variants.

    Duplicate and unknown-id rejection is delegated to the repository's
    existing :func:`~.utility_tilt.validate_arena_variant_ids`; this wrapper
    additionally refuses any variant outside
    :data:`CONDITIONAL_UTILITY_ARENA_VARIANT_IDS` (in particular every
    probabilistic ``U_P`` configuration, which is deliberately out of scope
    for this gate).
    """

    resolved = [resolve_variant_id(v) for v in variant_ids]
    try:
        validated = validate_arena_variant_ids(resolved)
    except UtilityTiltError as exc:
        raise ConditionalUtilityError(str(exc)) from exc
    out_of_gate = sorted(
        set(validated) - set(CONDITIONAL_UTILITY_ARENA_VARIANT_IDS)
    )
    if out_of_gate:
        raise ConditionalUtilityError(
            "variant(s) outside the v0 conditional utility arena: {} "
            "(allowed: {})".format(out_of_gate, list(CONDITIONAL_UTILITY_ARENA_VARIANT_IDS))
        )
    return validated


def variant_tilt_config(variant_id: str) -> Optional[TiltConfig]:
    """Canonical :class:`~.utility_tilt.TiltConfig`, or ``None`` for the nominal arm."""

    variant_id = resolve_variant_id(variant_id)
    if variant_id == NOMINAL_PHYSICAL_VARIANT_ID:
        return None
    try:
        return TILT_CONFIG_BY_ID[variant_id]
    except KeyError as exc:
        raise ConditionalUtilityError(
            "unknown variant_id {!r}".format(variant_id)
        ) from exc


def variant_multiplier(variant_id: str, utility: np.ndarray) -> np.ndarray:
    """``r_i`` for one variant: ones for NOMINAL, else the canonical ``h_{alpha,delta}``.

    The formula itself lives in :func:`~.utility_tilt.h_alpha_delta` and is
    never restated here.
    """

    utility = np.asarray(utility, dtype=np.float64)
    config = variant_tilt_config(variant_id)
    if config is None:
        return np.ones_like(utility)
    return config.r_utility_for(utility)


# --- deterministic seed derivation ---------------------------------------------


def variant_seed_component(variant_id: str) -> int:
    """Stable 32-bit integer derived from the canonical variant id.

    Included in every per-epoch seed so two variants can never share a draw
    sequence even at identical (model seed, epoch, charge).
    """

    digest = hashlib.sha256(resolve_variant_id(variant_id).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def derived_epoch_seed(
    *, global_seed: int, epoch: int, pdg_id: int, variant_id: str, salt: int = 0
) -> int:
    """Deterministic seed from ``(global model seed, epoch, pdg_id, variant_id)``."""

    return int(
        np.random.SeedSequence(
            [
                int(global_seed),
                int(epoch),
                abs(int(pdg_id)),
                int(int(pdg_id) < 0),
                variant_seed_component(variant_id),
                int(salt),
            ]
        ).generate_state(1)[0]
    )


def derived_generation_seed(*, global_seed: int, pdg_id: int, variant_id: str) -> int:
    """Deterministic generation seed for the trained model's diagnostic samples."""

    return derived_epoch_seed(
        global_seed=global_seed, epoch=0, pdg_id=pdg_id, variant_id=variant_id, salt=7
    ) % (2 ** 31 - 1)


# --- small numeric helpers ------------------------------------------------------


def _array_hash(array: np.ndarray) -> str:
    hasher = hashlib.sha256()
    hasher.update(np.ascontiguousarray(np.asarray(array, dtype=np.float64)).tobytes())
    return hasher.hexdigest()


def weighted_nll(log_prob: np.ndarray, weights: np.ndarray) -> float:
    """``-sum(weights * log q) / sum(weights)`` over finite rows."""

    log_prob = np.asarray(log_prob, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if log_prob.shape != weights.shape:
        raise ConditionalUtilityError("log_prob and weights must have the same shape")
    finite = np.isfinite(log_prob) & np.isfinite(weights)
    total = float(weights[finite].sum())
    if not finite.any() or not (total > 0.0):
        raise ConditionalUtilityError("no finite positively-weighted validation rows")
    return float(-np.sum(weights[finite] * log_prob[finite]) / total)


def wilson_interval(successes: float, trials: int, *, z: float = WILSON_Z_95):
    """Wilson score interval for a binomial proportion (``None`` when ``trials == 0``)."""

    trials = int(trials)
    if trials <= 0:
        return (None, None)
    p = float(successes) / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / denominator
    half = (
        z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials))
    ) / denominator
    return (float(centre - half), float(centre + half))


def binomial_standard_error(successes: float, trials: int) -> Optional[float]:
    trials = int(trials)
    if trials <= 0:
        return None
    p = float(successes) / trials
    return float(math.sqrt(max(p * (1.0 - p), 0.0) / trials))


def _safe_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or not (denominator > 0.0):
        return None
    return float(numerator) / float(denominator)


# --- per-charge sampling law ----------------------------------------------------


class ChargeSamplingLaw:
    """One charge's nominal and variant sampling law, built from TRAIN rows only.

    Holds every quantity the arena has to hash or report for a single
    ``(variant, charge)`` pair: the nominal probability table, the synthetic
    utility vector, the multiplier vector, the tilted probability table, the
    alias table, the train-fitted ``B_toy`` thresholds, and the declared
    nominal/tilted ``B_toy`` target probabilities.
    """

    def __init__(
        self,
        *,
        pdg_id: int,
        raw: np.ndarray,
        physical: np.ndarray,
        variant_id: str,
    ) -> None:
        self.pdg_id = int(pdg_id)
        self.variant_id = resolve_variant_id(variant_id)
        self.tilt_config = variant_tilt_config(self.variant_id)
        self.is_nominal = self.tilt_config is None

        self.raw = raw
        self.weights = np.ascontiguousarray(raw[:, _W_COLUMN], dtype=np.float64)
        if not np.isfinite(self.weights).all() or np.any(self.weights < 0.0):
            raise ConditionalUtilityError("weights must be finite and nonnegative")

        self.source_table_hash = dataset_hash(raw)
        self.pi_nominal = pi_nominal(self.weights)

        self.thresholds: ToyThresholds = fit_toy_thresholds(
            physical, self.weights, pdg_id=self.pdg_id
        )
        self.b_toy = compute_b_toy(physical, self.thresholds)
        self.utility = compute_utility_a(self.b_toy)
        self.r_utility = variant_multiplier(self.variant_id, self.utility)
        self.pi_variant = (
            self.pi_nominal if self.is_nominal else pi_tilt(self.weights, self.r_utility)
        )

        self.nominal_b_toy_probability = nominal_prevalence(self.b_toy, self.weights)
        if self.is_nominal:
            self.tilted_b_toy_probability = self.nominal_b_toy_probability
        else:
            self.tilted_b_toy_probability = theoretical_tilted_prevalence(
                self.nominal_b_toy_probability,
                self.tilt_config.r_positive(),
                self.tilt_config.r_negative(),
            )
        # Independent cross-check: direct summation over the tilted table.
        self.tilted_b_toy_probability_direct = float(
            np.sum(self.pi_variant * self.b_toy)
        )
        self.target_b_toy_probability = (
            self.nominal_b_toy_probability if self.is_nominal else self.tilted_b_toy_probability
        )

        self.nominal_probability_table_hash = _array_hash(self.pi_nominal)
        self.utility_vector_hash = _array_hash(self.utility)
        self.multiplier_vector_hash = _array_hash(self.r_utility)
        self.tilted_probability_table_hash = _array_hash(self.pi_variant)
        self.sampler = AliasSampler.from_probabilities(
            self.pi_variant, source_hash=self.source_table_hash
        )
        self.alias_table_hash = self.sampler.table_hash()

    @property
    def n_train(self) -> int:
        return int(self.raw.shape[0])

    def draw(self, n_draws: int, *, seed: int) -> np.ndarray:
        return self.sampler.draw(int(n_draws), seed=int(seed))

    def hashes(self) -> Dict[str, str]:
        return {
            "source_table_hash": self.source_table_hash,
            "nominal_probability_table_hash": self.nominal_probability_table_hash,
            "utility_vector_hash": self.utility_vector_hash,
            "multiplier_vector_hash": self.multiplier_vector_hash,
            "tilted_probability_table_hash": self.tilted_probability_table_hash,
            "alias_table_hash": self.alias_table_hash,
        }

    def concentration(self, *, draw_budget: int) -> Dict[str, Any]:
        """Concentration already in ``w_mc`` versus the extra concentration from ``r_utility``."""

        nominal = theoretical_concentration_diagnostics(
            self.pi_nominal, draw_budget=int(draw_budget)
        )
        variant = theoretical_concentration_diagnostics(
            self.pi_variant, draw_budget=int(draw_budget)
        )
        n_train = float(self.n_train)
        for block in (nominal, variant):
            block["n_eff_over_n_train"] = float(block["n_eff"]) / n_train
        return {
            "pdg_id": self.pdg_id,
            "variant_id": self.variant_id,
            "n_train": self.n_train,
            "draw_budget": int(draw_budget),
            "source_weight_concentration_w_mc": nominal,
            "variant_sampling_concentration": variant,
            "additional_concentration_from_r_utility": {
                "n_eff_ratio_variant_over_nominal": _safe_ratio(
                    variant["n_eff"], nominal["n_eff"]
                ),
                "top_10_mass_delta": float(
                    variant["top_10_mass"] - nominal["top_10_mass"]
                ),
                "top_100_mass_delta": float(
                    variant["top_100_mass"] - nominal["top_100_mass"]
                ),
                "expected_n_unique_delta": float(
                    variant["expected_n_unique"] - nominal["expected_n_unique"]
                ),
            },
            "nominal_b_toy_probability": self.nominal_b_toy_probability,
            "tilted_b_toy_probability": self.tilted_b_toy_probability,
            "tilted_b_toy_probability_direct_summation": self.tilted_b_toy_probability_direct,
            "amplification_factor_pU_over_p0": _safe_ratio(
                self.tilted_b_toy_probability, self.nominal_b_toy_probability
            ),
            "thresholds": self.thresholds.to_dict(),
            "thresholds_fitted_on": "train_partition_only",
        }


# --- per-epoch conditional utility sampler ---------------------------------------


class ConditionalUtilityEpochSampler:
    """Per-epoch, per-charge direct sampling from ``pi^(variant, c)``.

    Structurally identical to the verified v1 nominal sampler
    (:class:`~.conditional_charge._EpochDirectSampler`) except that each
    charge's categorical law is the variant's law rather than always the
    nominal one, and the derived per-epoch seed additionally depends on the
    variant id. For :data:`~.utility_tilt.NOMINAL_PHYSICAL_VARIANT_ID` the
    law *is* the v1 nominal law, bit for bit.

    Every epoch draws exactly ``draws_per_epoch_per_charge`` rows for PDG 13
    and the same number for PDG -13 (with replacement), so the macro charge
    prior stays 1/2 -- 1/2 whatever the tilt does to the within-charge mass.
    No sample weight and no utility multiplier is ever attached to the drawn
    rows: both ``sample_weight_applied_to_loss`` and
    ``utility_weight_applied_to_loss`` are always ``False`` and the loss on
    the drawn pool is the ordinary unweighted NLL.
    """

    def __init__(
        self,
        *,
        datasets: Mapping[int, EmpiricalTrainValidationDataset],
        pipeline: FittedFeaturePipeline,
        variant_id: str,
        draws_per_epoch_per_charge: int,
        seed: int,
    ) -> None:
        if int(draws_per_epoch_per_charge) < 1:
            raise ConditionalUtilityError("draws_per_epoch_per_charge must be positive")
        self.variant_id = resolve_variant_id(variant_id)
        validate_conditional_utility_variant_ids([self.variant_id])
        self.tilt_config = variant_tilt_config(self.variant_id)
        self.is_nominal = self.tilt_config is None
        self._pipeline = pipeline
        self.draws_per_epoch_per_charge = int(draws_per_epoch_per_charge)
        self._seed = int(seed)
        self.laws: Dict[int, ChargeSamplingLaw] = {}
        self._cumulative_unique: Dict[int, set] = {13: set(), -13: set()}
        for pdg_id in (13, -13):
            partition = datasets[pdg_id].train
            self.laws[pdg_id] = ChargeSamplingLaw(
                pdg_id=pdg_id,
                raw=partition.raw,
                physical=partition.physical,
                variant_id=self.variant_id,
            )
        self.epoch_records: List[Dict[str, Any]] = []

    # -- convenience accessors ------------------------------------------------

    @property
    def thresholds(self) -> Dict[int, ToyThresholds]:
        return {pdg_id: law.thresholds for pdg_id, law in self.laws.items()}

    def sampling_regime(self) -> str:
        return CONDITIONAL_UTILITY_SAMPLING_REGIME_BY_KIND[
            "nominal" if self.is_nominal else "tilt"
        ]

    # -- the epoch draw --------------------------------------------------------

    def draw(self, epoch: int) -> Dict[str, Any]:
        epoch = int(epoch)
        per_charge_raw: Dict[int, np.ndarray] = {}
        per_charge_meta: Dict[str, Any] = {}
        for pdg_id in (13, -13):
            law = self.laws[pdg_id]
            derived_seed = derived_epoch_seed(
                global_seed=self._seed,
                epoch=epoch,
                pdg_id=pdg_id,
                variant_id=self.variant_id,
            )
            indices = law.draw(self.draws_per_epoch_per_charge, seed=derived_seed)
            per_charge_raw[pdg_id] = law.raw[indices]
            self._cumulative_unique[pdg_id].update(int(i) for i in np.unique(indices))
            diagnostics = empirical_draw_diagnostics(indices, b_toy=law.b_toy)
            per_charge_meta[str(pdg_id)] = {
                "variant_id": self.variant_id,
                "pdg_id": int(pdg_id),
                "condition": muon_electric_charge_sign(pdg_id),
                "derived_seed": derived_seed,
                "draw_count": self.draws_per_epoch_per_charge,
                "draw_hash": canonical_hash(indices.tolist()),
                "empirical_b_toy_occupancy": diagnostics["empirical_b_toy_fraction"],
                "declared_target_b_toy_probability": law.target_b_toy_probability,
                "nominal_b_toy_probability": law.nominal_b_toy_probability,
                "tilted_b_toy_probability": law.tilted_b_toy_probability,
                "unique_source_rows": diagnostics["unique_rows_drawn"],
                "unique_rows_fraction": diagnostics["unique_rows_fraction"],
                "max_reuse_count": diagnostics["max_reuse_count"],
                "cumulative_unique_source_rows": len(self._cumulative_unique[pdg_id]),
                "sample_weight_applied_to_loss": False,
                "utility_weight_applied_to_loss": False,
                "sampling_with_replacement": True,
                **law.hashes(),
            }
        raw_combined = np.concatenate((per_charge_raw[13], per_charge_raw[-13]))
        condition = np.concatenate(
            (
                np.full((self.draws_per_epoch_per_charge, 1), MUON_ELECTRIC_CHARGE_SIGN_BY_PDG[13]),
                np.full((self.draws_per_epoch_per_charge, 1), MUON_ELECTRIC_CHARGE_SIGN_BY_PDG[-13]),
            )
        )
        shuffle_seed = derived_epoch_seed(
            global_seed=self._seed,
            epoch=epoch,
            pdg_id=0,
            variant_id=self.variant_id,
            salt=1,
        )
        permutation = np.random.default_rng(shuffle_seed).permutation(raw_combined.shape[0])
        raw_combined = raw_combined[permutation]
        condition = condition[permutation]
        x = self._pipeline.transform_raw(raw_combined)

        record = {
            "epoch": epoch,
            "variant_id": self.variant_id,
            "charge_counts": {
                "13": self.draws_per_epoch_per_charge,
                "-13": self.draws_per_epoch_per_charge,
            },
            "equal_draw_count_per_charge": True,
            "sample_weight_applied_to_loss": False,
            "utility_weight_applied_to_loss": False,
            "sampling_with_replacement": True,
            "per_charge": per_charge_meta,
        }
        self.epoch_records.append(record)
        return {"x": x, "condition": condition, "metadata": record}

    # -- manifests -------------------------------------------------------------

    def concentration_diagnostics(self) -> Dict[str, Any]:
        return {
            "schema_version": CONDITIONAL_UTILITY_SCHEMA_VERSION,
            "variant_id": self.variant_id,
            "draws_per_epoch_per_charge": self.draws_per_epoch_per_charge,
            "n_epochs_recorded": len(self.epoch_records),
            "diagnostic_only_no_threshold": True,
            "charges": {
                str(pdg_id): dict(
                    self.laws[pdg_id].concentration(
                        draw_budget=self.draws_per_epoch_per_charge
                    ),
                    condition=muon_electric_charge_sign(pdg_id),
                    empirical_unique_rows_per_epoch=[
                        record["per_charge"][str(pdg_id)]["unique_source_rows"]
                        for record in self.epoch_records
                    ],
                    max_row_reuse_per_epoch=[
                        record["per_charge"][str(pdg_id)]["max_reuse_count"]
                        for record in self.epoch_records
                    ],
                    cumulative_unique_source_rows=len(self._cumulative_unique[pdg_id]),
                    cumulative_unique_source_rows_fraction=(
                        float(len(self._cumulative_unique[pdg_id]))
                        / self.laws[pdg_id].n_train
                    ),
                    max_row_reuse_over_training=max(
                        [
                            record["per_charge"][str(pdg_id)]["max_reuse_count"]
                            for record in self.epoch_records
                        ]
                        or [0]
                    ),
                )
                for pdg_id in (13, -13)
            },
        }

    def manifest(self) -> Dict[str, Any]:
        return {
            "schema_version": CONDITIONAL_UTILITY_SCHEMA_VERSION,
            "variant_id": self.variant_id,
            "sampling_regime": self.sampling_regime(),
            "legacy_sampling_regime_id": (
                NOMINAL_SAMPLING_REGIME if self.is_nominal else TILT_SAMPLING_REGIME
            ),
            "tilt_config": (
                None if self.is_nominal else self.tilt_config.to_dict()
            ),
            "utility_definition": {
                "utility_id": "U_A",
                "formula": "U_A(x) = 1[x in B_toy,c], binary synthetic utility",
                "multiplier_formula": "r_i = [delta + (1 - delta) * U_i] ** alpha",
                "multiplier_source": "ship_muon_bg.density_lab.utility_tilt.h_alpha_delta",
                "nominal_multiplier": "r_i = 1 for the NOMINAL_PHYSICAL arm",
                "synthetic_not_calibrated_acceptance": True,
            },
            "draws_per_epoch_per_charge": self.draws_per_epoch_per_charge,
            "draws_per_epoch_total": 2 * self.draws_per_epoch_per_charge,
            "global_seed": self._seed,
            "seed_derivation": "SeedSequence[global_seed, epoch, |pdg|, pdg<0, sha256(variant_id)[:8], salt]",
            "macro_charge_prior": {"13": 0.5, "-13": 0.5},
            "charge_prior_unaffected_by_tilt": True,
            "sample_weight_applied_to_loss": False,
            "utility_weight_applied_to_loss": False,
            "sampling_with_replacement": True,
            "gradient_estimator_target": (
                "E_epoch_draws[gradient] = (1/2) * grad L^(variant)_13 "
                "+ (1/2) * grad L^(variant)_-13"
            ),
            "unbiasedness_scope": (
                "per_step_stochastic_gradient_only; the final trained parameters "
                "are not claimed to be an unbiased estimator of any macro target"
            ),
            "charges": {
                str(pdg_id): {
                    "pdg_id": int(pdg_id),
                    "condition": muon_electric_charge_sign(pdg_id),
                    "n_train": self.laws[pdg_id].n_train,
                    "nominal_b_toy_probability": self.laws[pdg_id].nominal_b_toy_probability,
                    "tilted_b_toy_probability": self.laws[pdg_id].tilted_b_toy_probability,
                    "declared_target_b_toy_probability": self.laws[pdg_id].target_b_toy_probability,
                    "thresholds": self.laws[pdg_id].thresholds.to_dict(),
                    "cumulative_unique_source_rows": len(self._cumulative_unique[pdg_id]),
                    **self.laws[pdg_id].hashes(),
                }
                for pdg_id in (13, -13)
            },
            "epochs": list(self.epoch_records),
            "n_epochs_recorded": len(self.epoch_records),
        }


# --- configuration ----------------------------------------------------------------


def _validate_config(config: Mapping[str, Any], repo_root: Path) -> Dict[str, Any]:
    unknown = sorted(set(config) - _ALLOWED_CONFIG_KEYS)
    if unknown:
        raise ConditionalUtilityError(
            "undocumented config fields are refused: {}".format(unknown)
        )
    if config.get("schema_version") != CONDITIONAL_UTILITY_SCHEMA_VERSION:
        raise ConditionalUtilityError(
            "schema_version must be {!r}".format(CONDITIONAL_UTILITY_SCHEMA_VERSION)
        )
    if config.get("fixture_only") is not True:
        raise ConditionalUtilityError("fixture_only must be true")
    dataset_path = (repo_root / str(config["dataset_path"])).resolve()
    fixture_root = (repo_root / "data" / "samples").resolve()
    if fixture_root not in dataset_path.parents or "_sample." not in dataset_path.name:
        raise ConditionalUtilityError(
            "the v0 conditional utility arena accepts only a repository "
            "data/samples/*_sample.* fixture"
        )
    if tuple(int(value) for value in config.get("pdg_ids", ())) != (13, -13):
        raise ConditionalUtilityError("pdg_ids must be exactly [13, -13]")
    if config.get("feature_view", {}).get("view_id") != IDENTITY_CARTESIAN_VIEW_ID:
        raise ConditionalUtilityError(
            "only the no-transform identity_cartesian_v0 baseline is documented"
        )
    model = config.get("model", {})
    if model.get("family") != "affine_coupling":
        raise ConditionalUtilityError("model.family must be 'affine_coupling'")
    if model.get("conditional") is not True:
        raise ConditionalUtilityError("model.conditional must be true")
    if model.get("condition_name") != CONDITION_FIELD_NAME:
        raise ConditionalUtilityError(
            "model.condition_name must be {!r}".format(CONDITION_FIELD_NAME)
        )
    if list(model.get("condition_values", ())) != [-1, 1]:
        raise ConditionalUtilityError("model.condition_values must be [-1, 1]")
    params = dict(model.get("params", {}))
    if params.get("condition_dim", 1) != 1:
        raise ConditionalUtilityError("conditional charge requires model condition_dim=1")
    for name in (
        "max_rows_per_charge",
        "draws_per_epoch_per_charge",
        "generated_sample_count_per_charge",
    ):
        if not isinstance(config.get(name), int) or config[name] < 2:
            raise ConditionalUtilityError("{} must be an integer >= 2".format(name))
    variants = validate_conditional_utility_variant_ids(
        config.get("variants", CONDITIONAL_UTILITY_ARENA_VARIANT_IDS)
    )
    model_seeds = [int(s) for s in config.get("model_seeds", CONDITIONAL_UTILITY_MODEL_SEEDS)]
    if not model_seeds or len(set(model_seeds)) != len(model_seeds):
        raise ConditionalUtilityError("model_seeds must be a non-empty list of distinct ints")

    resolved = dict(config)
    resolved["dataset_path"] = str(dataset_path)
    resolved["variants"] = list(variants)
    resolved["model_seeds"] = model_seeds
    resolved["model"] = dict(model)
    resolved["model"]["params"] = params
    resolved["model"]["params"]["condition_dim"] = 1
    return resolved


def _current_git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[3]),
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _fixture_manifest(dataset_path: Path) -> Dict[str, Any]:
    manifest_path = dataset_path.parent / (
        dataset_path.name.split(".npz")[0] + "_manifest.json"
    )
    if not manifest_path.is_file():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        "fixture_manifest_file": manifest_path.name,
        "subset_dataset_hash": payload.get("subset_dataset_hash"),
        "parent_full_dataset_hash": payload.get("source_dataset_hash"),
        "parent_full_dataset_sha256": payload.get("source_sha256"),
        "parent_full_dataset_n_rows": payload.get("source_n_rows"),
        "subset_n_rows": payload.get("subset_n_rows"),
    }


def _split_hash(dataset: EmpiricalTrainValidationDataset) -> str:
    return canonical_hash(
        {
            "train_source_row_indices": dataset.train.source_row_indices.tolist(),
            "validation_source_row_indices": dataset.validation.source_row_indices.tolist(),
            "test_split_hash": dataset.test_split_hash,
        }
    )


def build_split_and_pipeline(
    resolved: Mapping[str, Any], *, seed: int
) -> Tuple[Dict[int, EmpiricalTrainValidationDataset], FittedFeaturePipeline]:
    """Load train+validation partitions and fit the shared nominal preprocessing.

    The preprocessing transform is the *fixed nominal macro-weighted
    train-only* transform (section 4 of the gate): it is fit exactly once per
    (data split, model seed) from ``pi^(0,c)``, never from resampled rows,
    never from tilted weights, never per variant, and never from validation
    or test rows. It is therefore identical across all variants sharing a
    seed and split, which the arena verifies by hash.
    """

    datasets: Dict[int, EmpiricalTrainValidationDataset] = {}
    for pdg_id in (13, -13):
        datasets[pdg_id] = build_empirical_train_validation_dataset(
            EmpiricalDatasetSpec(
                dataset_path=resolved["dataset_path"],
                pdg_id=pdg_id,
                seed=int(seed),
                val_fraction=float(resolved.get("validation_fraction", 0.2)),
                test_fraction=float(resolved.get("test_fraction", 0.2)),
                max_rows=int(resolved["max_rows_per_charge"]),
            )
        )
    pipeline = FittedFeaturePipeline.fit_macro_weighted(
        raw_by_charge={pdg_id: datasets[pdg_id].train.raw for pdg_id in (13, -13)},
        weights_by_charge={
            pdg_id: datasets[pdg_id].train.raw[:, _W_COLUMN] for pdg_id in (13, -13)
        },
        feature_view=FeatureView(IDENTITY_CARTESIAN_VIEW_ID),
    )
    return datasets, pipeline


def _test_provenance(
    datasets: Mapping[int, EmpiricalTrainValidationDataset]
) -> Dict[str, Any]:
    return {
        str(pdg_id): {
            "pdg_id": int(pdg_id),
            "test_row_count": datasets[pdg_id].test_row_count,
            "test_split_hash": datasets[pdg_id].test_split_hash,
            "test_payload_loaded": False,
            "test_used_for_training": False,
            "test_used_for_preprocessing": False,
            "test_used_for_utility_thresholds": False,
            "test_used_for_model_selection": False,
            "test_used_for_evaluation": False,
        }
        for pdg_id in (13, -13)
    }


# --- one arena run (one variant, one model seed) ------------------------------------


def run_conditional_utility_run(
    config: Mapping[str, Any],
    *,
    variant_id: str,
    seed: int,
    output_dir: Path,
    repo_root: Optional[Path] = None,
    datasets: Optional[Mapping[int, EmpiricalTrainValidationDataset]] = None,
    pipeline: Optional[FittedFeaturePipeline] = None,
) -> Dict[str, Any]:
    """Train and evaluate one ``(variant, model seed)`` cell of the fixture arena."""

    repo_root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[3]
    )
    resolved = _validate_config(config, repo_root)
    variant_id = validate_conditional_utility_variant_ids([variant_id])[0]
    seed = int(seed)

    if datasets is None or pipeline is None:
        datasets, pipeline = build_split_and_pipeline(resolved, seed=seed)

    validation_raw = np.concatenate(
        (datasets[13].validation.raw, datasets[-13].validation.raw)
    )
    validation_condition = np.concatenate(
        (
            np.full((datasets[13].validation.n_rows, 1), MUON_ELECTRIC_CHARGE_SIGN_BY_PDG[13]),
            np.full((datasets[-13].validation.n_rows, 1), MUON_ELECTRIC_CHARGE_SIGN_BY_PDG[-13]),
        )
    )
    # Exact weighted validation, never resampled: each charge's physical
    # weights are normalized within the charge so the two charges enter the
    # monitored validation objective with equal macro mass.
    validation_weights = np.concatenate(
        tuple(
            datasets[pdg_id].validation.raw[:, _W_COLUMN]
            / datasets[pdg_id].validation.raw[:, _W_COLUMN].sum()
            for pdg_id in (13, -13)
        )
    )

    from Nflow.torch_models.affine_coupling import AffineCouplingFlow

    model = AffineCouplingFlow(
        dimension=pipeline.dimension,
        device=resolved["model"].get("device", "cpu"),
        **resolved["model"]["params"],
    )

    sampler = ConditionalUtilityEpochSampler(
        datasets=datasets,
        pipeline=pipeline,
        variant_id=variant_id,
        draws_per_epoch_per_charge=int(resolved["draws_per_epoch_per_charge"]),
        seed=seed,
    )

    fit = model.fit(
        None,
        x_validation=pipeline.transform_raw(validation_raw),
        seed=seed,
        validation_sample_weight=validation_weights,
        validation_condition=validation_condition,
        epoch_sampler=sampler.draw,
    )
    if fit.status != "ok":
        raise ConditionalUtilityError(
            "conditional utility flow fit failed: {}".format(fit.warnings)
        )

    generated_count = int(resolved["generated_sample_count_per_charge"])
    nominal_validation: List[Dict[str, Any]] = []
    tilted_validation: List[Dict[str, Any]] = []
    generated_summary: List[Dict[str, Any]] = []

    for pdg_id in (13, -13):
        law = sampler.laws[pdg_id]
        partition = datasets[pdg_id].validation
        condition = np.full((partition.n_rows, 1), MUON_ELECTRIC_CHARGE_SIGN_BY_PDG[pdg_id])
        normalized = pipeline.transform_raw(partition.raw)
        physical_lp = pipeline.normalized_to_physical_log_prob(
            model.log_prob(normalized, condition=condition), partition.raw
        )
        weights = np.ascontiguousarray(partition.raw[:, _W_COLUMN], dtype=np.float64)

        # Validation utility uses the TRAIN-fitted thresholds (never refit on
        # validation rows) and the same canonical multiplier as training.
        validation_b_toy = compute_b_toy(partition.physical, law.thresholds)
        validation_utility = compute_utility_a(validation_b_toy)
        validation_r = variant_multiplier(variant_id, validation_utility)

        # Cross-measure validation: the same trained model scored under every
        # arena variant's measure. Each variant's own tilted NLL is a
        # *different functional*, so comparing "tilted validation NLL" across
        # variant columns compares different quantities; this block is what
        # makes the improve/degrade trade-off comparable at fixed measure.
        cross_measure = {
            measure_id: weighted_nll(
                physical_lp,
                weights * variant_multiplier(measure_id, validation_utility),
            )
            for measure_id in CONDITIONAL_UTILITY_ARENA_VARIANT_IDS
        }

        nominal_validation.append(
            {
                "variant_id": variant_id,
                "model_seed": seed,
                "pdg_id": int(pdg_id),
                "condition": muon_electric_charge_sign(pdg_id),
                "validation_rows": partition.n_rows,
                "validation_weight_total": float(weights.sum()),
                "nominal_validation_nll": weighted_nll(physical_lp, weights),
                "validation_log_prob_finite_fraction": float(np.isfinite(physical_lp).mean()),
                "resampled": False,
                "formula": "-sum_i w_i log q(x_i|c) / sum_i w_i",
            }
        )
        tilted_validation.append(
            {
                "variant_id": variant_id,
                "model_seed": seed,
                "pdg_id": int(pdg_id),
                "condition": muon_electric_charge_sign(pdg_id),
                "validation_rows": partition.n_rows,
                "validation_weight_r_total": float(np.sum(weights * validation_r)),
                "tilted_validation_nll": weighted_nll(physical_lp, weights * validation_r),
                "cross_measure_validation_nll": cross_measure,
                "cross_measure_note": (
                    "the same trained model scored under every arena variant's "
                    "r-measure; compare variants column-wise at fixed measure"
                ),
                "validation_b_toy_weighted_prevalence": nominal_prevalence(
                    validation_b_toy, weights
                ),
                "thresholds_source": "train_partition_only",
                "thresholds": law.thresholds.to_dict(),
                "resampled": False,
                "formula": "-sum_i w_i r_i log q(x_i|c) / sum_i w_i r_i",
            }
        )

        generation_seed = derived_generation_seed(
            global_seed=seed, pdg_id=pdg_id, variant_id=variant_id
        )
        generated_condition = np.full(
            (generated_count, 1), MUON_ELECTRIC_CHARGE_SIGN_BY_PDG[pdg_id]
        )
        samples_normalized = model.sample(
            generated_count, seed=generation_seed, condition=generated_condition
        )
        samples_physical = pipeline.inverse_to_physical(samples_normalized)
        generated_log_prob = model.log_prob(
            samples_normalized, condition=generated_condition
        )
        finite_mask = np.isfinite(samples_physical).all(axis=1)
        finite_physical = samples_physical[finite_mask]
        n_finite = int(finite_physical.shape[0])
        if n_finite:
            generated_b_toy = compute_b_toy(finite_physical, law.thresholds)
            successes = float(generated_b_toy.sum())
            occupancy = float(successes / n_finite)
            low, high = wilson_interval(successes, n_finite)
            standard_error = binomial_standard_error(successes, n_finite)
        else:
            occupancy = None
            low = high = standard_error = None

        empirical_summary = compact_distribution_summary(
            partition.physical, weights=weights
        )
        generated_dist_summary = compact_distribution_summary(samples_physical)
        generated_summary.append(
            {
                "variant_id": variant_id,
                "model_seed": seed,
                "pdg_id": int(pdg_id),
                "condition": muon_electric_charge_sign(pdg_id),
                "generation_seed": generation_seed,
                "generated_sample_count": generated_count,
                "generated_sample_finite_fraction": float(finite_mask.mean()),
                "generated_log_prob_finite_fraction": float(
                    np.isfinite(generated_log_prob).mean()
                ),
                "generated_b_toy_occupancy": occupancy,
                "nominal_target_b_toy_probability": law.nominal_b_toy_probability,
                "tilted_target_b_toy_probability": law.tilted_b_toy_probability,
                "training_target_b_toy_probability": law.target_b_toy_probability,
                "training_target_measure": (
                    "nominal" if law.is_nominal else "tilted"
                ),
                "generated_over_nominal_b_toy_ratio": _safe_ratio(
                    occupancy, law.nominal_b_toy_probability
                ),
                "generated_over_tilted_b_toy_ratio": _safe_ratio(
                    occupancy, law.tilted_b_toy_probability
                ),
                "generated_over_training_target_b_toy_ratio": _safe_ratio(
                    occupancy, law.target_b_toy_probability
                ),
                "generated_b_toy_binomial_standard_error": standard_error,
                "generated_b_toy_wilson_95_low": low,
                "generated_b_toy_wilson_95_high": high,
                "weighted_empirical_summary": empirical_summary,
                "generated_summary": generated_dist_summary,
                "weighted_empirical_correlation_matrix": empirical_summary.get(
                    "correlation_matrix"
                ),
                "generated_correlation_matrix": generated_dist_summary.get(
                    "correlation_matrix"
                ),
                "correlation_feature_order": list(_PHYSICAL_FEATURE_NAMES),
            }
        )

    sampling_manifest = sampler.manifest()
    concentration = sampler.concentration_diagnostics()
    training_metrics = {
        "variant_id": variant_id,
        "model_seed": seed,
        "fit": fit.to_dict(),
        "training_history": fit.train_history,
        "loss_contract": {
            "sample_weight_applied_to_loss": False,
            "utility_weight_applied_to_loss": False,
            "ordinary_unweighted_nll_after_direct_sampling": True,
            "weight_normalization": "epoch_direct_sampling_unweighted",
        },
    }

    nominal_values = [row["nominal_validation_nll"] for row in nominal_validation]
    tilted_values = [row["tilted_validation_nll"] for row in tilted_validation]
    dataset_path = Path(resolved["dataset_path"])
    summary: Dict[str, Any] = {
        "schema_version": CONDITIONAL_UTILITY_SCHEMA_VERSION,
        "experiment_id": resolved["experiment_id"],
        "variant_id": variant_id,
        "model_seed": seed,
        "status": "completed_fixture_run",
        "scope": "fixture_only_not_physics_acceptance",
        "config_hash": canonical_hash(config),
        "sampling_regime": sampler.sampling_regime(),
        "condition_field_name": CONDITION_FIELD_NAME,
        "condition_definition": {
            "pdg_13": MUON_ELECTRIC_CHARGE_SIGN_BY_PDG[13],
            "pdg_-13": MUON_ELECTRIC_CHARGE_SIGN_BY_PDG[-13],
        },
        "draws_per_epoch_per_charge": int(resolved["draws_per_epoch_per_charge"]),
        "generated_sample_count_per_charge": generated_count,
        "test_provenance": _test_provenance(datasets),
        "lineage": {
            "code_commit": _current_git_commit(),
            "fixture_path": dataset_path.name,
            "fixture_dataset_hash": datasets[13].source_file_dataset_hash,
            "fixture_manifest": _fixture_manifest(dataset_path),
            "charges": {
                str(pdg_id): {
                    "split_hash": _split_hash(datasets[pdg_id]),
                    "train_rows": datasets[pdg_id].train.n_rows,
                    "validation_rows": datasets[pdg_id].validation.n_rows,
                    "test_row_count": datasets[pdg_id].test_row_count,
                    "test_split_hash": datasets[pdg_id].test_split_hash,
                    "test_payload_loaded": False,
                    **sampler.laws[pdg_id].hashes(),
                }
                for pdg_id in (13, -13)
            },
            "preprocessing_hash": pipeline.config_hash(),
            "preprocessing_weighting": pipeline.weighting,
            "preprocessing_fit_measure": "nominal_macro_weighted_train_only",
            "model_config_hash": canonical_hash(model.config()),
            "checkpoint_hash": model.checkpoint_hash(),
            "generation_seeds": {
                str(pdg_id): derived_generation_seed(
                    global_seed=seed, pdg_id=pdg_id, variant_id=variant_id
                )
                for pdg_id in (13, -13)
            },
        },
        "pipeline_manifest": pipeline.manifest(),
        "model_manifest": model.manifest(),
        "fit": training_metrics["fit"],
        "validation": {
            "nominal": {
                "per_charge": nominal_values,
                "macro_mean": float(np.mean(nominal_values)),
                "worst_charge": float(np.max(nominal_values)),
            },
            "tilted": {
                "per_charge": tilted_values,
                "macro_mean": float(np.mean(tilted_values)),
                "worst_charge": float(np.max(tilted_values)),
            },
            "selection_policy": (
                "diagnostic_trade_off_reported; no model is selected by nominal "
                "NLL alone and none by tilted NLL alone"
            ),
        },
        "per_charge_nominal_validation": nominal_validation,
        "per_charge_tilted_validation": tilted_validation,
        "per_charge_generated_summary": generated_summary,
        "sampling_manifest": sampling_manifest,
        "concentration_diagnostics": concentration,
    }

    _write_run_artifacts(
        Path(output_dir),
        summary,
        sampling_manifest=sampling_manifest,
        training_metrics=training_metrics,
        nominal_validation=nominal_validation,
        tilted_validation=tilted_validation,
        generated_summary=generated_summary,
        concentration=concentration,
    )
    return summary


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _write_run_artifacts(
    output_dir: Path,
    summary: Mapping[str, Any],
    *,
    sampling_manifest: Mapping[str, Any],
    training_metrics: Mapping[str, Any],
    nominal_validation: Sequence[Mapping[str, Any]],
    tilted_validation: Sequence[Mapping[str, Any]],
    generated_summary: Sequence[Mapping[str, Any]],
    concentration: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "conditional_utility_sampling_manifest.json": sampling_manifest,
        "conditional_utility_training_metrics.json": training_metrics,
        "per_charge_nominal_validation.json": list(nominal_validation),
        "per_charge_tilted_validation.json": list(tilted_validation),
        "per_charge_generated_summary.json": list(generated_summary),
        "concentration_diagnostics.json": concentration,
        "summary.json": summary,
    }
    for name, payload in files.items():
        _write_json(output_dir / name, payload)


# --- arena orchestration ------------------------------------------------------------


ARENA_SCIENTIFIC_SCOPE = (
    "This fixture arena tests whether one shared charge-conditional flow "
    "trained from a utility-tilted empirical proposal reproduces the declared "
    "tilted target for both charges while retaining measurable nominal "
    "coverage. It is not a physical-rate estimate, does not treat B_toy as a "
    "physical endpoint, does not treat U_A as a calibrated acceptance "
    "probability, does not claim FairShip acceptance, and declares no "
    "universal winner. FairShip/GEANT4 remains the final physical oracle."
)

ARENA_ROW_COLUMNS: Tuple[str, ...] = (
    "variant_id",
    "model_seed",
    "pdg_id",
    "condition",
    "technical_status",
    "nominal_validation_nll",
    "tilted_validation_nll",
    "nominal_target_b_toy",
    "tilted_target_b_toy",
    "training_target_b_toy",
    "generated_b_toy",
    "generated_over_nominal_ratio",
    "generated_over_tilted_ratio",
    "generated_b_toy_standard_error",
    "generated_b_toy_wilson_95_low",
    "generated_b_toy_wilson_95_high",
    "generated_sample_finite_fraction",
    "generated_log_prob_finite_fraction",
    "n_eff",
    "n_eff_over_n_train",
    "nominal_n_eff",
    "top_10_mass",
    "top_100_mass",
    "zero_probability_fraction",
    "expected_unique_rows_per_epoch",
    "empirical_unique_rows_per_epoch_mean",
    "cumulative_unique_source_rows",
    "max_row_reuse",
    "amplification_factor",
    "preprocessing_hash",
    "alias_table_hash",
    "tilted_probability_table_hash",
) + tuple(
    "val_nll_under_{}".format(measure_id)
    for measure_id in CONDITIONAL_UTILITY_ARENA_VARIANT_IDS
)


def _arena_rows_from_summary(summary: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    nominal_by_pdg = {row["pdg_id"]: row for row in summary["per_charge_nominal_validation"]}
    tilted_by_pdg = {row["pdg_id"]: row for row in summary["per_charge_tilted_validation"]}
    generated_by_pdg = {row["pdg_id"]: row for row in summary["per_charge_generated_summary"]}
    concentration = summary["concentration_diagnostics"]["charges"]
    charges = summary["sampling_manifest"]["charges"]
    for pdg_id in (13, -13):
        key = str(pdg_id)
        conc = concentration[key]
        variant_conc = conc["variant_sampling_concentration"]
        nominal_conc = conc["source_weight_concentration_w_mc"]
        unique_per_epoch = conc["empirical_unique_rows_per_epoch"]
        generated = generated_by_pdg[pdg_id]
        rows.append(
            {
                "variant_id": summary["variant_id"],
                "model_seed": summary["model_seed"],
                "pdg_id": pdg_id,
                "condition": muon_electric_charge_sign(pdg_id),
                "technical_status": summary["status"],
                "nominal_validation_nll": nominal_by_pdg[pdg_id]["nominal_validation_nll"],
                "tilted_validation_nll": tilted_by_pdg[pdg_id]["tilted_validation_nll"],
                "nominal_target_b_toy": generated["nominal_target_b_toy_probability"],
                "tilted_target_b_toy": generated["tilted_target_b_toy_probability"],
                "training_target_b_toy": generated["training_target_b_toy_probability"],
                "generated_b_toy": generated["generated_b_toy_occupancy"],
                "generated_over_nominal_ratio": generated["generated_over_nominal_b_toy_ratio"],
                "generated_over_tilted_ratio": generated["generated_over_tilted_b_toy_ratio"],
                "generated_b_toy_standard_error": generated[
                    "generated_b_toy_binomial_standard_error"
                ],
                "generated_b_toy_wilson_95_low": generated["generated_b_toy_wilson_95_low"],
                "generated_b_toy_wilson_95_high": generated["generated_b_toy_wilson_95_high"],
                "generated_sample_finite_fraction": generated["generated_sample_finite_fraction"],
                "generated_log_prob_finite_fraction": generated[
                    "generated_log_prob_finite_fraction"
                ],
                "n_eff": variant_conc["n_eff"],
                "n_eff_over_n_train": variant_conc["n_eff_over_n_train"],
                "nominal_n_eff": nominal_conc["n_eff"],
                "top_10_mass": variant_conc["top_10_mass"],
                "top_100_mass": variant_conc["top_100_mass"],
                "zero_probability_fraction": variant_conc["fraction_zero_probability_rows"],
                "expected_unique_rows_per_epoch": variant_conc["expected_n_unique"],
                "empirical_unique_rows_per_epoch_mean": (
                    float(np.mean(unique_per_epoch)) if unique_per_epoch else None
                ),
                "cumulative_unique_source_rows": conc["cumulative_unique_source_rows"],
                "max_row_reuse": conc["max_row_reuse_over_training"],
                "amplification_factor": conc["amplification_factor_pU_over_p0"],
                "preprocessing_hash": summary["lineage"]["preprocessing_hash"],
                "alias_table_hash": charges[key]["alias_table_hash"],
                "tilted_probability_table_hash": charges[key][
                    "tilted_probability_table_hash"
                ],
                **{
                    "val_nll_under_{}".format(measure_id): value
                    for measure_id, value in tilted_by_pdg[pdg_id][
                        "cross_measure_validation_nll"
                    ].items()
                },
            }
        )
    return rows


def _variant_sort_index(variant_id: str) -> int:
    try:
        return CONDITIONAL_UTILITY_ARENA_VARIANT_IDS.index(variant_id)
    except ValueError:
        return len(CONDITIONAL_UTILITY_ARENA_VARIANT_IDS)


def _row_sort_key(row: Mapping[str, Any]) -> Tuple[int, str, int, int]:
    return (
        _variant_sort_index(str(row["variant_id"])),
        str(row["variant_id"]),
        int(row["model_seed"]),
        # PDG 13 (mu-) first, then PDG -13 (mu+), deterministically.
        0 if int(row["pdg_id"]) == 13 else 1,
    )


def _aggregate(values: Sequence[Optional[float]]) -> Dict[str, Any]:
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not finite:
        return {"n": 0, "median": None, "min": None, "max": None, "values": list(values)}
    return {
        "n": len(finite),
        "median": statistics.median(finite),
        "min": min(finite),
        "max": max(finite),
        "values": list(values),
    }


def build_arena_aggregate(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Per-(variant, charge) aggregation across model seeds. No winner is declared."""

    aggregate: Dict[str, Any] = {}
    metrics = (
        "nominal_validation_nll",
        "tilted_validation_nll",
        "generated_b_toy",
        "generated_over_nominal_ratio",
        "generated_over_tilted_ratio",
        "n_eff",
        "cumulative_unique_source_rows",
        "max_row_reuse",
    ) + tuple(
        "val_nll_under_{}".format(measure_id)
        for measure_id in CONDITIONAL_UTILITY_ARENA_VARIANT_IDS
    )
    for variant_id in CONDITIONAL_UTILITY_ARENA_VARIANT_IDS:
        variant_rows = [r for r in rows if r["variant_id"] == variant_id]
        if not variant_rows:
            continue
        per_charge: Dict[str, Any] = {}
        for pdg_id in (13, -13):
            charge_rows = sorted(
                (r for r in variant_rows if int(r["pdg_id"]) == pdg_id),
                key=lambda r: int(r["model_seed"]),
            )
            per_charge[str(pdg_id)] = {
                "condition": muon_electric_charge_sign(pdg_id),
                "model_seeds": [int(r["model_seed"]) for r in charge_rows],
                "nominal_target_b_toy": _aggregate(
                    [r["nominal_target_b_toy"] for r in charge_rows]
                ),
                "tilted_target_b_toy": _aggregate(
                    [r["tilted_target_b_toy"] for r in charge_rows]
                ),
                **{
                    metric: _aggregate([r[metric] for r in charge_rows])
                    for metric in metrics
                },
            }
        aggregate[variant_id] = {
            "variant_id": variant_id,
            "per_charge": per_charge,
            "macro_nominal_validation_nll": _aggregate(
                [r["nominal_validation_nll"] for r in variant_rows]
            ),
            "macro_tilted_validation_nll": _aggregate(
                [r["tilted_validation_nll"] for r in variant_rows]
            ),
        }
    return {
        "schema_version": CONDITIONAL_UTILITY_SCHEMA_VERSION,
        "scientific_scope": ARENA_SCIENTIFIC_SCOPE,
        "no_composite_score": True,
        "no_winner_declared": True,
        "variants": aggregate,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return "{:.6g}".format(float(value))
    return str(value)


_MARKDOWN_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("variant_id", "variant"),
    ("model_seed", "seed"),
    ("pdg_id", "PDG"),
    ("nominal_validation_nll", "nominal val NLL"),
    ("tilted_validation_nll", "tilted val NLL"),
    ("nominal_target_b_toy", "nominal p0(B)"),
    ("tilted_target_b_toy", "tilted pU(B)"),
    ("generated_b_toy", "generated B"),
    ("generated_over_nominal_ratio", "gen/nom"),
    ("generated_over_tilted_ratio", "gen/tilt"),
    ("generated_b_toy_standard_error", "gen SE"),
    ("n_eff", "N_eff"),
    ("cumulative_unique_source_rows", "cum unique"),
    ("max_row_reuse", "max reuse"),
)


def write_arena_reports(
    rows: Sequence[Mapping[str, Any]],
    *,
    out_dir: Path,
    experiment_id: str,
    completed: Sequence[Mapping[str, Any]],
    requested: Sequence[Mapping[str, Any]],
    preprocessing_hashes: Mapping[str, Any],
    lineage: Mapping[str, Any],
) -> Dict[str, Any]:
    """Deterministic arena artifacts: run summary JSON, CSV, Markdown, aggregate JSON."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=_row_sort_key)

    run_summary = {
        "schema_version": CONDITIONAL_UTILITY_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "scientific_scope": ARENA_SCIENTIFIC_SCOPE,
        "declared_variant_ids": list(CONDITIONAL_UTILITY_ARENA_VARIANT_IDS),
        "requested_matrix": list(requested),
        "completed_matrix": list(completed),
        "completion_status": (
            "CONDITIONAL_UTILITY_ARENA_VERIFIED"
            if len(completed) == len(requested)
            else "CONDITIONAL_UTILITY_ARENA_PARTIAL"
        ),
        "preprocessing_hash_by_model_seed": dict(preprocessing_hashes),
        "lineage": dict(lineage),
        "rows": rows,
        "no_composite_score": True,
        "no_winner_declared": True,
    }
    _write_json(out_dir / "arena_run_summary.json", run_summary)

    aggregate = build_arena_aggregate(rows)
    _write_json(out_dir / "arena_aggregate.json", aggregate)

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=list(ARENA_ROW_COLUMNS), lineterminator="\n", extrasaction="ignore"
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column) for column in ARENA_ROW_COLUMNS})
    (out_dir / "arena_summary.csv").write_text(stream.getvalue(), encoding="utf-8")

    lines = [
        "# D9 conditional utility-tilt fixture arena (v0)",
        "",
        "**Scientific scope.** {}".format(ARENA_SCIENTIFIC_SCOPE),
        "",
        "- Completion: `{}` ({}/{} runs)".format(
            run_summary["completion_status"], len(completed), len(requested)
        ),
        "- Charge convention: PDG 13 (mu-) -> -1, PDG -13 (mu+) -> +1",
        "- Loss: ordinary unweighted NLL after direct sampling "
        "(`sample_weight_applied_to_loss = false`, `utility_weight_applied_to_loss = false`)",
        "- Preprocessing: one fixed nominal macro-weighted train-only transform per "
        "(split, model seed), shared by every variant",
        "- Test payload: never materialized; never used for thresholds",
        "",
        "| " + " | ".join(label for _, label in _MARKDOWN_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in _MARKDOWN_COLUMNS) + " |",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(_fmt(row.get(key)) for key, _ in _MARKDOWN_COLUMNS) + " |"
        )
    lines.append("")
    lines.append(
        "Concentration diagnostics (N_eff, top-k mass, uniqueness, reuse) are "
        "diagnostic only: no ESS threshold and no automatic rejection criterion "
        "is defined. Generated occupancy must be read together with its binomial "
        "standard error and across seeds."
    )
    (out_dir / "arena_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return run_summary


def run_conditional_utility_arena(
    config: Mapping[str, Any],
    *,
    output_dir: Path,
    repo_root: Optional[Path] = None,
    variants: Optional[Sequence[str]] = None,
    model_seeds: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Run the ``variants x model seeds`` fixture arena and write the reports.

    Seeds are iterated in the outer loop and variants in the inner loop, so a
    truncated budget completes all four variants for the first seed before
    starting the next one (section 9 of the gate). Preprocessing is fit once
    per seed from the nominal measure and reused by every variant of that
    seed; each run additionally records its own preprocessing hash so hash
    equality across variants is verifiable from the artifacts alone.
    """

    repo_root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[3]
    )
    resolved = _validate_config(config, repo_root)
    variant_ids = validate_conditional_utility_variant_ids(
        variants if variants is not None else resolved["variants"]
    )
    seeds = [int(s) for s in (model_seeds if model_seeds is not None else resolved["model_seeds"])]
    output_dir = Path(output_dir)

    requested = [
        {"variant_id": variant_id, "model_seed": seed}
        for seed in seeds
        for variant_id in variant_ids
    ]
    completed: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    preprocessing_hashes: Dict[str, Any] = {}
    lineage: Dict[str, Any] = {}

    for seed in seeds:
        datasets, pipeline = build_split_and_pipeline(resolved, seed=seed)
        seed_hashes: List[str] = []
        for variant_id in variant_ids:
            run_dir = output_dir / "seed_{}".format(seed) / variant_id
            summary = run_conditional_utility_run(
                config,
                variant_id=variant_id,
                seed=seed,
                output_dir=run_dir,
                repo_root=repo_root,
                datasets=datasets,
                pipeline=pipeline,
            )
            rows.extend(_arena_rows_from_summary(summary))
            seed_hashes.append(summary["lineage"]["preprocessing_hash"])
            completed.append(
                {
                    "variant_id": variant_id,
                    "model_seed": seed,
                    "output_dir": str(run_dir.relative_to(output_dir)),
                    "status": summary["status"],
                }
            )
            if not lineage:
                lineage = {
                    key: summary["lineage"][key]
                    for key in ("fixture_dataset_hash", "fixture_manifest", "fixture_path")
                }
                lineage["code_commit"] = summary["lineage"]["code_commit"]
        unique_hashes = sorted(set(seed_hashes))
        if len(unique_hashes) != 1:
            raise ConditionalUtilityError(
                "preprocessing hash differs across variants for model seed {}: {}".format(
                    seed, unique_hashes
                )
            )
        preprocessing_hashes[str(seed)] = {
            "preprocessing_hash": unique_hashes[0],
            "identical_across_variants": True,
            "variants": list(variant_ids),
            "split_hash": {
                str(pdg_id): _split_hash(datasets[pdg_id]) for pdg_id in (13, -13)
            },
        }

    return write_arena_reports(
        rows,
        out_dir=output_dir,
        experiment_id=str(resolved["experiment_id"]),
        completed=completed,
        requested=requested,
        preprocessing_hashes=preprocessing_hashes,
        lineage=lineage,
    )
