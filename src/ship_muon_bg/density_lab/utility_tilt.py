"""D9 -- direct-sampling utility-tilt experiment on the D7 empirical fixture.

This module implements the mathematical contract of
``docs/d9_direct_utility_tilt_sampling_v0.md``: given physical MC weights
``w_i`` on a declared training pool, it builds a nominal direct-sampling law
``pi_nominal = w_i / sum(w_j)`` and, for 20 utility-tilt configurations, a
tilted law ``pi_tilt = w_i * r_i / sum(w_j * r_j)`` where ``r_i`` is a
*synthetic* utility multiplier -- never itself called a physical weight.
Training draws indices directly (with replacement) from ``pi_tilt`` via an
O(1) Walker-alias sampler and trains with **ordinary unweighted NLL**: the
tilt's effect is already encoded in which rows get drawn, so the loss must
never be multiplied by ``w_i``, ``r_i``, or ``w_i * r_i`` again.

Deliberately reused, unmodified:

- ``ship_muon_bg.data_contracts`` (schema, weighted quantiles, PDG filtering);
- ``density_lab.empirical.build_empirical_dataset`` / ``EmpiricalDataset`` for
  the load -> validate -> PDG-filter -> split pipeline (identical for the
  repository fixture and a future full local dataset -- only
  ``dataset_path`` differs);
- ``density_lab.feature_pipeline.FittedFeaturePipeline`` (fit once on the
  *un-resampled* train partition so preprocessing is identical across every
  tilt run; only the row-sampling law changes);
- ``Nflow.registry.create_density_estimator`` and its unmodified ``.fit()``
  legacy IID path (``sample_weight=None``, ``batch_plan=None``) -- the
  resampled draw array is simply a new ``x_train`` array, so this is exactly
  arm A applied to a resampled row set, not a new estimator arm;
- ``density_lab.artifacts.ArtifactStore`` for run identity and resume/skip.

``B_toy`` is a project-defined synthetic diagnostic region (thresholds on
``pT`` and ``R_xy``), never a validated FairShip danger metric; ``U_A``/
``U_P`` are synthetic utility scores, never calibrated probabilities.
Concentration diagnostics (``N_eff``, expected-unique, top-k mass, ...) are
computed for every tilt but are diagnostic-only: this module introduces no
ESS threshold and no automatic rejection criterion.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..data_contracts import schema, weighted_prevalence, weighted_quantile
from .artifacts import STATUS_COMPLETED, STATUS_FAILED, ArtifactStore, derive_run_id
from .config import (
    DatasetSpec,
    EvaluationSpec,
    FeatureViewSpec,
    ModelSpec,
    RunSpec,
    TargetSpec,
    canonical_hash,
)
from .empirical import EmpiricalDataset, EmpiricalDatasetSpec, build_empirical_dataset
from .environment import capture_environment, utc_timestamp
from .feature_pipeline import FittedFeaturePipeline

SCHEMA_VERSION = "0"
DIMENSION = 5
UTILITY_TILT_TARGET_ID = "D9_utility_tilt"
PHYSICAL_FEATURE_NAMES: Tuple[str, ...] = ("px", "py", "pz", "x", "y")
DISTRIBUTION_SUMMARY_QUANTILES: Tuple[float, ...] = (0.01, 0.05, 0.5, 0.95, 0.99)

UTILITY_MODES: Tuple[str, ...] = ("UA", "UP")
TILT_DELTAS: Tuple[float, ...] = (0.1, 0.9)
TILT_ALPHAS: Tuple[float, ...] = (1, 2, 4, 8, 16)

# The explicit un-tilted arm (task section 5A): sampling from Table A's
# pi_nominal directly, never a disguised utility configuration. Reserved and
# never a member of TILT_CONFIG_BY_ID / ALL_TILT_CONFIGS -- callers branch on
# this sentinel explicitly rather than looking up a fabricated TiltConfig.
NOMINAL_PHYSICAL_VARIANT_ID = "NOMINAL_PHYSICAL"
NOMINAL_SAMPLING_REGIME = "direct_nominal_physical"
TILT_SAMPLING_REGIME = "direct_utility_tilt"

PT_QUANTILE = 0.95
RXY_QUANTILE = 0.05
THRESHOLD_DEFINITION_ID = "d9_b_toy_v0_pt_q095_rxy_q005"

U_A_POSITIVE = 1.0
U_A_NEGATIVE = 0.0
U_P_POSITIVE = 0.99
U_P_NEGATIVE = 0.01

_W_COLUMN = schema.COLUMN_INDEX["w"]

# The exact four tilt configurations trained during the bounded cloud proof
# (task section 6): UA vs UP, mild vs strong tilt, never alpha in {8, 16}.
FOUR_CLOUD_TILT_MODES_DELTAS_ALPHAS: Tuple[Tuple[str, float, float], ...] = (
    ("UA", 0.9, 1),
    ("UA", 0.1, 4),
    ("UP", 0.9, 1),
    ("UP", 0.1, 4),
)


class UtilityTiltError(ValueError):
    """An invalid utility-tilt configuration, table, or sampler precondition."""


# --- kinematics: pT, R_xy, B_toy ---------------------------------------------


def compute_pt(physical: np.ndarray) -> np.ndarray:
    """``sqrt(px**2 + py**2)`` from a ``(N, 5)`` ``[px, py, pz, x, y]`` array."""

    physical = np.asarray(physical, dtype=np.float64)
    return np.sqrt(physical[:, 0] ** 2 + physical[:, 1] ** 2)


def compute_rxy(physical: np.ndarray) -> np.ndarray:
    """``sqrt(x**2 + y**2)``, the project-defined synthetic diagnostic coordinate."""

    physical = np.asarray(physical, dtype=np.float64)
    return np.sqrt(physical[:, 3] ** 2 + physical[:, 4] ** 2)


@dataclasses.dataclass(frozen=True)
class ToyThresholds:
    """``t_pT``/``t_R``, fit from TRAIN-SPLIT rows only, weighted by ``w_i``."""

    pdg_id: int
    t_pT: float
    t_R: float
    pt_quantile: float = PT_QUANTILE
    rxy_quantile: float = RXY_QUANTILE
    definition_id: str = THRESHOLD_DEFINITION_ID

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pdg_id": int(self.pdg_id),
            "t_pT": float(self.t_pT),
            "t_R": float(self.t_R),
            "pt_quantile": float(self.pt_quantile),
            "rxy_quantile": float(self.rxy_quantile),
            "definition_id": self.definition_id,
        }


def fit_toy_thresholds(train_physical: np.ndarray, train_weights: np.ndarray, *, pdg_id: int) -> ToyThresholds:
    """Fit ``t_pT = Q^(w)_0.95(pT)``, ``t_R = Q^(w)_0.05(R_xy)`` from train rows only.

    Callers must pass only the TRAIN partition's physical rows/weights for one
    PDG track -- never validation or test rows (see the module docstring's
    "no threshold leakage" contract).
    """

    pt = compute_pt(train_physical)
    rxy = compute_rxy(train_physical)
    t_pt = weighted_quantile(pt, train_weights, PT_QUANTILE)
    t_r = weighted_quantile(rxy, train_weights, RXY_QUANTILE)
    return ToyThresholds(pdg_id=int(pdg_id), t_pT=t_pt, t_R=t_r)


def compute_b_toy(physical: np.ndarray, thresholds: ToyThresholds) -> np.ndarray:
    """``1[pT > t_pT and R_xy < t_R]`` as a float64 0/1 array.

    The joint prevalence must be *measured*, never assumed to be the product
    of the two marginal quantile levels (0.95 and 0.05) -- ``pT`` and
    ``R_xy`` are not independent.
    """

    pt = compute_pt(physical)
    rxy = compute_rxy(physical)
    return ((pt > thresholds.t_pT) & (rxy < thresholds.t_R)).astype(np.float64)


def compute_utility_a(b_toy: np.ndarray) -> np.ndarray:
    """U-A: hard 0/1 synthetic utility."""

    b_toy = np.asarray(b_toy, dtype=np.float64)
    return np.where(b_toy >= 0.5, U_A_POSITIVE, U_A_NEGATIVE)


def compute_utility_p(b_toy: np.ndarray) -> np.ndarray:
    """U-P: soft 0.99/0.01 synthetic utility."""

    b_toy = np.asarray(b_toy, dtype=np.float64)
    return np.where(b_toy >= 0.5, U_P_POSITIVE, U_P_NEGATIVE)


def nominal_prevalence(b_toy: np.ndarray, weights: np.ndarray) -> float:
    """``p_0(B_toy) = sum(w_i * B_i) / sum(w_i)``."""

    return weighted_prevalence(b_toy, weights)


# --- utility transform h_{alpha,delta} and the 20-config tilt grid ----------


def h_alpha_delta(u, *, alpha: float, delta: float) -> np.ndarray:
    """``r = [delta + (1 - delta) * U] ** alpha``. Reserve ``r`` as ``r_utility``, never a physical weight."""

    u = np.asarray(u, dtype=np.float64)
    if not (0.0 <= float(delta) <= 1.0):
        raise UtilityTiltError("delta must be in [0, 1], got {!r}".format(delta))
    if not (float(alpha) > 0.0):
        raise UtilityTiltError("alpha must be positive, got {!r}".format(alpha))
    if np.any(u < 0.0) or np.any(u > 1.0) or not np.isfinite(u).all():
        raise UtilityTiltError("utility scores U must be finite and lie in [0, 1]")
    base = float(delta) + (1.0 - float(delta)) * u
    return base ** float(alpha)


def _format_delta(delta: float) -> str:
    text = "{:g}".format(float(delta)).replace(".", "p")
    return "d{}".format(text)


def _format_alpha(alpha: float) -> str:
    return "a{:02d}".format(int(alpha))


def make_tilt_id(mode: str, delta: float, alpha: float) -> str:
    if mode not in UTILITY_MODES:
        raise UtilityTiltError("mode must be one of {}, got {!r}".format(UTILITY_MODES, mode))
    return "{}_{}_{}".format(mode, _format_delta(delta), _format_alpha(alpha))


@dataclasses.dataclass(frozen=True)
class TiltConfig:
    """One point of the 20-config utility-tilt grid: (mode, delta, alpha)."""

    tilt_id: str
    mode: str
    delta: float
    alpha: float

    def u_positive(self) -> float:
        return U_A_POSITIVE if self.mode == "UA" else U_P_POSITIVE

    def u_negative(self) -> float:
        return U_A_NEGATIVE if self.mode == "UA" else U_P_NEGATIVE

    def r_positive(self) -> float:
        return float(h_alpha_delta(np.array([self.u_positive()]), alpha=self.alpha, delta=self.delta)[0])

    def r_negative(self) -> float:
        return float(h_alpha_delta(np.array([self.u_negative()]), alpha=self.alpha, delta=self.delta)[0])

    def rho(self) -> float:
        return self.r_positive() / self.r_negative()

    def utility_for(self, b_toy: np.ndarray) -> np.ndarray:
        return compute_utility_a(b_toy) if self.mode == "UA" else compute_utility_p(b_toy)

    def r_utility_for(self, utility: np.ndarray) -> np.ndarray:
        return h_alpha_delta(utility, alpha=self.alpha, delta=self.delta)

    def r_utility_for_b_toy(self, b_toy: np.ndarray) -> np.ndarray:
        return self.r_utility_for(self.utility_for(b_toy))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tilt_id": self.tilt_id,
            "mode": self.mode,
            "delta": float(self.delta),
            "alpha": float(self.alpha),
        }


def theoretical_tilted_prevalence(p0: float, r_positive: float, r_negative: float) -> float:
    """Binary closed-form ``p_U(B_toy)``; must agree with direct summation over ``pi_tilt``."""

    p0 = float(p0)
    numerator = p0 * float(r_positive)
    denominator = numerator + (1.0 - p0) * float(r_negative)
    if not (denominator > 0.0):
        raise UtilityTiltError("theoretical tilted prevalence denominator is non-positive")
    return numerator / denominator


def build_tilt_grid() -> Tuple[TiltConfig, ...]:
    """All 20 ``(mode, delta, alpha)`` combinations, each exactly once, sorted by ``tilt_id``."""

    configs = [
        TiltConfig(make_tilt_id(mode, delta, alpha), mode, delta, alpha)
        for mode in UTILITY_MODES
        for delta in TILT_DELTAS
        for alpha in TILT_ALPHAS
    ]
    return tuple(sorted(configs, key=lambda c: c.tilt_id))


ALL_TILT_CONFIGS: Tuple[TiltConfig, ...] = build_tilt_grid()
TILT_CONFIG_BY_ID: Dict[str, TiltConfig] = {c.tilt_id: c for c in ALL_TILT_CONFIGS}
FOUR_CLOUD_TILT_IDS: Tuple[str, ...] = tuple(
    make_tilt_id(mode, delta, alpha) for mode, delta, alpha in FOUR_CLOUD_TILT_MODES_DELTAS_ALPHAS
)
for _tid in FOUR_CLOUD_TILT_IDS:
    assert _tid in TILT_CONFIG_BY_ID, "cloud tilt id {} missing from the 20-config grid".format(_tid)

# The bounded D9 utility-tilt arena (task section 5B): NOMINAL_PHYSICAL plus
# exactly eight tilt configurations -- mild delta=0.9 alpha in {4, 8, 16} and
# strong delta=0.1 alpha=1, for both U-A and U-P. No new alpha/delta value is
# introduced beyond the existing 20-config grid.
ARENA_TILT_MODES_DELTAS_ALPHAS: Tuple[Tuple[str, float, float], ...] = (
    ("UA", 0.9, 4), ("UA", 0.9, 8), ("UA", 0.9, 16), ("UA", 0.1, 1),
    ("UP", 0.9, 4), ("UP", 0.9, 8), ("UP", 0.9, 16), ("UP", 0.1, 1),
)
ARENA_TILT_IDS: Tuple[str, ...] = tuple(
    make_tilt_id(mode, delta, alpha) for mode, delta, alpha in ARENA_TILT_MODES_DELTAS_ALPHAS
)
for _tid in ARENA_TILT_IDS:
    assert _tid in TILT_CONFIG_BY_ID, "arena tilt id {} missing from the 20-config grid".format(_tid)
ARENA_VARIANT_IDS: Tuple[str, ...] = (NOMINAL_PHYSICAL_VARIANT_ID,) + ARENA_TILT_IDS


def validate_arena_variant_ids(variant_ids: Sequence[str]) -> Tuple[str, ...]:
    """Reject an empty, duplicated, or unknown arena variant list.

    A variant is either :data:`NOMINAL_PHYSICAL_VARIANT_ID` or a key of
    :data:`TILT_CONFIG_BY_ID`; nothing else is a valid arena arm.
    """

    variant_ids = list(variant_ids)
    if not variant_ids:
        raise UtilityTiltError("arena variant_ids must be non-empty")
    seen: Dict[str, int] = {}
    for vid in variant_ids:
        seen[vid] = seen.get(vid, 0) + 1
    duplicates = sorted(vid for vid, count in seen.items() if count > 1)
    if duplicates:
        raise UtilityTiltError("arena variant_ids contains duplicates: {}".format(duplicates))
    unknown = sorted(
        vid for vid in variant_ids if vid != NOMINAL_PHYSICAL_VARIANT_ID and vid not in TILT_CONFIG_BY_ID
    )
    if unknown:
        raise UtilityTiltError("arena variant_ids contains unknown variant(s): {}".format(unknown))
    return tuple(variant_ids)


# --- pi_nominal / pi_tilt -----------------------------------------------------


def pi_nominal(weights: np.ndarray) -> np.ndarray:
    """``pi_i^(0) = w_i / sum(w_j)`` over the declared training pool."""

    weights = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise UtilityTiltError("weights must be finite and non-negative")
    total = float(weights.sum())
    if not (total > 0.0):
        raise UtilityTiltError("sum of weights must be positive")
    return weights / total


def pi_tilt(weights: np.ndarray, r_utility: np.ndarray) -> np.ndarray:
    """``pi_i^(U) = w_i * r_i / sum(w_j * r_j)``."""

    weights = np.asarray(weights, dtype=np.float64)
    r_utility = np.asarray(r_utility, dtype=np.float64)
    if weights.shape != r_utility.shape:
        raise UtilityTiltError("weights and r_utility must have the same shape")
    if not np.isfinite(r_utility).all() or np.any(r_utility <= 0.0):
        raise UtilityTiltError("r_utility must be finite and strictly positive")
    w_mc_times_r = weights * r_utility
    total = float(w_mc_times_r.sum())
    if not (total > 0.0):
        raise UtilityTiltError("sum of w_mc_times_r must be positive")
    return w_mc_times_r / total


# --- O(1) Walker-alias categorical sampler -----------------------------------


class AliasSampler:
    """O(1)-per-draw categorical sampler (Walker's alias method).

    O(N) one-time preprocessing (:meth:`from_probabilities`); each draw after
    that is one uniform-int lookup plus one Bernoulli coin flip, independent
    of N. Never silently falls back to uniform sampling: a probability
    vector that is negative, non-finite, or does not sum to 1 (outside
    floating-point tolerance) raises :class:`UtilityTiltError` at
    construction time rather than degrading quietly.
    """

    def __init__(self, probability_table: np.ndarray, alias_table: np.ndarray, *, n: int, source_hash: Optional[str] = None):
        self.probability_table = np.asarray(probability_table, dtype=np.float64)
        self.alias_table = np.asarray(alias_table, dtype=np.int64)
        self.n = int(n)
        self._source_hash = source_hash

    @classmethod
    def from_probabilities(cls, probabilities: Sequence[float], *, source_hash: Optional[str] = None) -> "AliasSampler":
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if probabilities.ndim != 1 or probabilities.shape[0] == 0:
            raise UtilityTiltError("probabilities must be a non-empty 1-D array")
        if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0):
            raise UtilityTiltError("probabilities must be finite and non-negative")
        total = float(probabilities.sum())
        if not np.isclose(total, 1.0, atol=1e-6):
            raise UtilityTiltError("probabilities must sum to 1.0 (got {})".format(total))

        n = int(probabilities.shape[0])
        scaled = probabilities.astype(np.float64) * n
        alias = np.zeros(n, dtype=np.int64)
        prob_table = np.ones(n, dtype=np.float64)

        small = [i for i in range(n) if scaled[i] < 1.0]
        large = [i for i in range(n) if scaled[i] >= 1.0]
        while small and large:
            s = small.pop()
            l = large.pop()
            prob_table[s] = scaled[s]
            alias[s] = l
            scaled[l] = scaled[l] - (1.0 - scaled[s])
            if scaled[l] < 1.0:
                small.append(l)
            else:
                large.append(l)
        for l in large:
            prob_table[l] = 1.0
        for s in small:
            prob_table[s] = 1.0
        return cls(prob_table, alias, n=n, source_hash=source_hash)

    def table_hash(self) -> str:
        hasher = hashlib.sha256()
        hasher.update(self.probability_table.tobytes())
        hasher.update(self.alias_table.tobytes())
        if self._source_hash:
            hasher.update(self._source_hash.encode("utf-8"))
        return hasher.hexdigest()

    def draw(self, n_draws: int, *, seed: int) -> np.ndarray:
        """Deterministic-seed batch draw with replacement; O(n_draws) total, O(1) per draw."""

        if n_draws < 0:
            raise UtilityTiltError("n_draws must be >= 0")
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, self.n, size=int(n_draws))
        coin = rng.random(int(n_draws))
        return np.where(coin < self.probability_table[idx], idx, self.alias_table[idx]).astype(np.int64)


# --- concentration diagnostics (diagnostic only; never gating) --------------


def theoretical_concentration_diagnostics(
    pi: np.ndarray, *, draw_budget: int, top_k: Sequence[int] = (10, 100)
) -> Dict[str, Any]:
    """N_eff, expected-unique-draws, top-k mass, etc. Diagnostic only -- no ESS threshold."""

    pi = np.asarray(pi, dtype=np.float64)
    t = int(draw_budget)
    sorted_pi = np.sort(pi)[::-1]
    result: Dict[str, Any] = {
        "n_rows": int(pi.shape[0]),
        "draw_budget": t,
        "n_eff": float(1.0 / np.sum(pi ** 2)),
        "expected_n_unique": float(np.sum(1.0 - (1.0 - pi) ** t)),
        "max_row_probability": float(np.max(pi)),
        "expected_repetitions_of_max_row": float(t * np.max(pi)),
        "fraction_zero_probability_rows": float(np.mean(pi <= 0.0)),
    }
    for k in top_k:
        k = int(k)
        mass = float(sorted_pi[: min(k, sorted_pi.shape[0])].sum())
        result["top_{}_mass".format(k)] = mass
    return result


def empirical_draw_diagnostics(drawn_indices: np.ndarray, *, b_toy: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Realized draw statistics: unique rows, max reuse, empirical B_toy occupancy."""

    drawn_indices = np.asarray(drawn_indices, dtype=np.int64)
    unique, counts = np.unique(drawn_indices, return_counts=True)
    total_draws = int(drawn_indices.shape[0])
    result: Dict[str, Any] = {
        "total_draws": total_draws,
        "unique_rows_drawn": int(unique.shape[0]),
        "unique_rows_fraction": (float(unique.shape[0]) / total_draws) if total_draws else None,
        "max_reuse_count": int(counts.max()) if counts.size else 0,
    }
    if b_toy is not None:
        b_toy = np.asarray(b_toy, dtype=np.float64)
        result["empirical_b_toy_fraction"] = float(np.mean(b_toy[drawn_indices])) if drawn_indices.size else None
    return result


# --- compact distribution diagnostics (descriptive only) ---------------------


def _weighted_mean_std(values: np.ndarray, weights: np.ndarray) -> Tuple[float, float]:
    mean = float(np.average(values, weights=weights))
    variance = float(np.average((values - mean) ** 2, weights=weights))
    return mean, float(np.sqrt(max(variance, 0.0)))


def _weighted_correlation_matrix(physical: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted Pearson correlation over ``physical`` columns, weights need not sum to 1."""

    weights = np.asarray(weights, dtype=np.float64)
    total = float(weights.sum())
    mean = np.average(physical, axis=0, weights=weights)
    centered = physical - mean
    cov = (centered * weights[:, None]).T @ centered / total
    std = np.sqrt(np.diag(cov))
    denom = np.outer(std, std)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.where(denom > 0.0, cov / denom, np.nan)
    return corr


def compact_distribution_summary(
    physical: np.ndarray,
    *,
    weights: Optional[np.ndarray] = None,
    feature_names: Sequence[str] = PHYSICAL_FEATURE_NAMES,
    quantile_levels: Sequence[float] = DISTRIBUTION_SUMMARY_QUANTILES,
) -> Dict[str, Any]:
    """Mean/std/quantiles per feature plus a Pearson correlation matrix.

    Descriptive only -- no pass/fail threshold. ``weights`` (optional) makes
    this the *declared empirical target* summary (weighted by ``pi``); no
    ``weights`` makes this a plain *generated-sample* summary. Non-finite rows
    are dropped before computing anything and reported via
    ``finite_fraction``; if zero rows remain, ``available`` is ``False`` and
    no statistic is fabricated.
    """

    physical = np.asarray(physical, dtype=np.float64)
    n_total = int(physical.shape[0])
    finite_mask = np.isfinite(physical).all(axis=1) if n_total else np.zeros(0, dtype=bool)
    n_finite = int(finite_mask.sum())
    result: Dict[str, Any] = {
        "n_rows": n_total,
        "n_finite_rows": n_finite,
        "finite_fraction": (float(n_finite) / n_total) if n_total else None,
        "available": n_finite > 0,
    }
    if n_finite == 0:
        return result

    physical_f = physical[finite_mask]
    w = None if weights is None else np.asarray(weights, dtype=np.float64)[finite_mask]
    if w is not None and float(w.sum()) <= 0.0:
        result["available"] = False
        result["reason"] = "non_positive_weight_sum_after_finite_filter"
        return result

    per_feature: Dict[str, Any] = {}
    for i, name in enumerate(feature_names):
        column = physical_f[:, i]
        if w is not None:
            mean, std = _weighted_mean_std(column, w)
            quantiles = {
                "q{:02d}".format(int(round(q * 100))): weighted_quantile(column, w, q)
                for q in quantile_levels
            }
        else:
            mean, std = float(np.mean(column)), float(np.std(column))
            quantiles = {
                "q{:02d}".format(int(round(q * 100))): float(np.quantile(column, q))
                for q in quantile_levels
            }
        per_feature[name] = {"mean": mean, "std": std, **quantiles}
    result["per_feature"] = per_feature

    if w is not None:
        corr = _weighted_correlation_matrix(physical_f, w)
    else:
        corr = np.corrcoef(physical_f, rowvar=False)
    result["correlation_matrix"] = corr.tolist()
    result["correlation_feature_order"] = list(feature_names)
    result["weighted"] = w is not None
    return result


# --- reference tables ---------------------------------------------------------


def _current_git_commit() -> Optional[str]:
    try:
        repo_root = Path(__file__).resolve().parents[3]
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root), capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


@dataclasses.dataclass(frozen=True)
class NominalTable:
    """Table A: one record per training row, one PDG track (task section 3)."""

    pdg_id: int
    source_dataset_hash: str
    split_id: str
    split_hash: str
    thresholds: ToyThresholds
    row_ref: np.ndarray
    source_row_index: np.ndarray
    w_mc: np.ndarray
    pi_nominal: np.ndarray
    pT: np.ndarray
    R_xy: np.ndarray
    B_toy: np.ndarray
    U_A: np.ndarray
    U_P: np.ndarray
    zero_weight: np.ndarray

    @property
    def n_rows(self) -> int:
        return int(self.row_ref.shape[0])

    def table_hash(self) -> str:
        hasher = hashlib.sha256()
        for arr in (
            self.row_ref, self.source_row_index, self.w_mc, self.pi_nominal,
            self.pT, self.R_xy, self.B_toy, self.U_A, self.U_P, self.zero_weight,
        ):
            hasher.update(np.ascontiguousarray(arr).tobytes())
        hasher.update(self.source_dataset_hash.encode("utf-8"))
        hasher.update(self.split_hash.encode("utf-8"))
        return hasher.hexdigest()

    def manifest(self, *, seed: int) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "table_kind": "nominal_sampling_table",
            "source_dataset_hash": self.source_dataset_hash,
            "split_id": self.split_id,
            "split_hash": self.split_hash,
            "pdg_id": int(self.pdg_id),
            "n_rows": self.n_rows,
            "feature_and_coordinate_definitions": {
                "pT": "sqrt(px**2 + py**2)",
                "R_xy": "sqrt(x**2 + y**2)",
            },
            "weighted_quantile_method": "inverted_cdf_step_function_tie_break_by_original_row_index",
            "thresholds": self.thresholds.to_dict(),
            "utility_definitions": {
                "U_A": {"positive": U_A_POSITIVE, "negative": U_A_NEGATIVE},
                "U_P": {"positive": U_P_POSITIVE, "negative": U_P_NEGATIVE},
            },
            "normalization_sum_before": float(self.w_mc.sum()),
            "normalization_sum_after": float(self.pi_nominal.sum()),
            "code_commit": _current_git_commit(),
            "generation_seed": int(seed),
            "table_hash": self.table_hash(),
        }

    def save(self, directory: Path, *, seed: int) -> Dict[str, Any]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            directory / "table_a_nominal.npz",
            row_ref=self.row_ref, source_row_index=self.source_row_index, w_mc=self.w_mc,
            pi_nominal=self.pi_nominal, pT=self.pT, R_xy=self.R_xy, B_toy=self.B_toy,
            U_A=self.U_A, U_P=self.U_P, zero_weight=self.zero_weight,
        )
        manifest = self.manifest(seed=seed)
        (directory / "table_a_nominal.manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
        return manifest

    @classmethod
    def load(cls, directory: Path) -> "NominalTable":
        directory = Path(directory)
        manifest = json.loads((directory / "table_a_nominal.manifest.json").read_text())
        data = np.load(directory / "table_a_nominal.npz")
        thresholds = ToyThresholds(**manifest["thresholds"])
        table = cls(
            pdg_id=int(manifest["pdg_id"]), source_dataset_hash=manifest["source_dataset_hash"],
            split_id=manifest["split_id"], split_hash=manifest["split_hash"], thresholds=thresholds,
            row_ref=data["row_ref"], source_row_index=data["source_row_index"], w_mc=data["w_mc"],
            pi_nominal=data["pi_nominal"], pT=data["pT"], R_xy=data["R_xy"], B_toy=data["B_toy"],
            U_A=data["U_A"], U_P=data["U_P"], zero_weight=data["zero_weight"],
        )
        if table.table_hash() != manifest["table_hash"]:
            raise UtilityTiltError("loaded nominal table content does not match its manifest table_hash")
        return table


def build_nominal_table(dataset: EmpiricalDataset, *, thresholds: Optional[ToyThresholds] = None) -> NominalTable:
    """Table A from ``dataset.train`` only. Never reads validation/test rows."""

    train = dataset.train
    physical = train.physical
    weights = np.ascontiguousarray(train.raw[:, _W_COLUMN], dtype=np.float64)
    pdg_id = int(dataset.dataset_spec.pdg_id)
    if thresholds is None:
        thresholds = fit_toy_thresholds(physical, weights, pdg_id=pdg_id)
    elif thresholds.pdg_id != pdg_id:
        raise UtilityTiltError(
            "thresholds.pdg_id={} does not match dataset pdg_id={}".format(thresholds.pdg_id, pdg_id)
        )
    b_toy = compute_b_toy(physical, thresholds)
    return NominalTable(
        pdg_id=pdg_id,
        source_dataset_hash=dataset.source_file_dataset_hash,
        split_id="train",
        split_hash=train.manifest()["raw_dataset_hash"],
        thresholds=thresholds,
        row_ref=np.arange(train.n_rows, dtype=np.int64),
        source_row_index=np.asarray(train.source_row_indices, dtype=np.int64),
        w_mc=weights,
        pi_nominal=pi_nominal(weights),
        pT=compute_pt(physical),
        R_xy=compute_rxy(physical),
        B_toy=b_toy,
        U_A=compute_utility_a(b_toy),
        U_P=compute_utility_p(b_toy),
        zero_weight=(weights <= 0.0),
    )


def tilt_pi_vector(table_a: NominalTable, tilt_config: TiltConfig) -> np.ndarray:
    """``pi_tilt`` for one configuration, computed directly from Table A (single source of truth)."""

    r = tilt_config.r_utility_for_b_toy(table_a.B_toy)
    return pi_tilt(table_a.w_mc, r)


@dataclasses.dataclass(frozen=True)
class TiltTable:
    """Table B: long-form utility-tilt sampling table, referencing Table A rows (task section 3)."""

    pdg_id: int
    table_a_hash: str
    tilt_ids: Tuple[str, ...]
    row_ref: np.ndarray
    tilt_id_index: np.ndarray
    mode_code: np.ndarray  # int8: 0=UA, 1=UP
    delta: np.ndarray
    alpha: np.ndarray
    U: np.ndarray
    r_utility: np.ndarray
    w_mc_times_r: np.ndarray
    pi_tilt: np.ndarray

    @property
    def n_rows_per_tilt(self) -> int:
        return int(self.row_ref.shape[0]) // len(self.tilt_ids) if self.tilt_ids else 0

    def block(self, tilt_id: str) -> np.ndarray:
        """Boolean mask selecting this tilt's long-form rows."""

        idx = self.tilt_ids.index(tilt_id)
        return self.tilt_id_index == idx

    def table_hash(self) -> str:
        hasher = hashlib.sha256()
        for arr in (
            self.row_ref, self.tilt_id_index, self.mode_code, self.delta,
            self.alpha, self.U, self.r_utility, self.w_mc_times_r, self.pi_tilt,
        ):
            hasher.update(np.ascontiguousarray(arr).tobytes())
        hasher.update(self.table_a_hash.encode("utf-8"))
        hasher.update("|".join(self.tilt_ids).encode("utf-8"))
        return hasher.hexdigest()

    def manifest(self, tilt_configs: Sequence[TiltConfig], *, source_dataset_hash: str, split_hash: str, seed: int) -> Dict[str, Any]:
        n_per = self.n_rows_per_tilt
        per_tilt = []
        pi_sums: Dict[str, float] = {}
        for i, config in enumerate(tilt_configs):
            block = self.pi_tilt[i * n_per: (i + 1) * n_per]
            pi_sums[config.tilt_id] = float(block.sum())
            per_tilt.append({
                **config.to_dict(),
                "r_positive": config.r_positive(),
                "r_negative": config.r_negative(),
                "rho": config.rho(),
            })
        return {
            "schema_version": SCHEMA_VERSION,
            "table_kind": "utility_tilt_sampling_table",
            "source_dataset_hash": source_dataset_hash,
            "split_hash": split_hash,
            "pdg_id": int(self.pdg_id),
            "table_a_hash": self.table_a_hash,
            "tilt_ids": list(self.tilt_ids),
            "mode_code_legend": {"0": "UA", "1": "UP"},
            "tilt_grid": per_tilt,
            "pi_tilt_sum_per_tilt": pi_sums,
            "code_commit": _current_git_commit(),
            "generation_seed": int(seed),
            "table_hash": self.table_hash(),
        }

    def save(self, directory: Path, tilt_configs: Sequence[TiltConfig], *, source_dataset_hash: str, split_hash: str, seed: int) -> Dict[str, Any]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            directory / "table_b_tilt.npz",
            row_ref=self.row_ref, tilt_id_index=self.tilt_id_index, mode_code=self.mode_code,
            delta=self.delta, alpha=self.alpha, U=self.U, r_utility=self.r_utility,
            w_mc_times_r=self.w_mc_times_r, pi_tilt=self.pi_tilt,
        )
        manifest = self.manifest(tilt_configs, source_dataset_hash=source_dataset_hash, split_hash=split_hash, seed=seed)
        (directory / "table_b_tilt.manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
        return manifest

    @classmethod
    def load(cls, directory: Path) -> "TiltTable":
        directory = Path(directory)
        manifest = json.loads((directory / "table_b_tilt.manifest.json").read_text())
        data = np.load(directory / "table_b_tilt.npz")
        table = cls(
            pdg_id=int(manifest["pdg_id"]), table_a_hash=manifest["table_a_hash"],
            tilt_ids=tuple(manifest["tilt_ids"]), row_ref=data["row_ref"],
            tilt_id_index=data["tilt_id_index"], mode_code=data["mode_code"], delta=data["delta"],
            alpha=data["alpha"], U=data["U"], r_utility=data["r_utility"],
            w_mc_times_r=data["w_mc_times_r"], pi_tilt=data["pi_tilt"],
        )
        if table.table_hash() != manifest["table_hash"]:
            raise UtilityTiltError("loaded tilt table content does not match its manifest table_hash")
        return table


def build_tilt_table(table_a: NominalTable, tilt_configs: Sequence[TiltConfig]) -> TiltTable:
    """Table B (long-form) for the given tilt configurations, referencing Table A rows only."""

    if not tilt_configs:
        raise UtilityTiltError("tilt_configs must be non-empty")
    tilt_ids = tuple(c.tilt_id for c in tilt_configs)
    if len(set(tilt_ids)) != len(tilt_ids):
        raise UtilityTiltError("tilt_configs must have unique tilt_id values")

    n = table_a.n_rows
    n_tilts = len(tilt_configs)
    row_ref = np.tile(table_a.row_ref, n_tilts)
    tilt_id_index = np.repeat(np.arange(n_tilts, dtype=np.int64), n)
    mode_code = np.repeat(
        np.array([0 if c.mode == "UA" else 1 for c in tilt_configs], dtype=np.int8), n
    )
    delta = np.repeat(np.array([c.delta for c in tilt_configs], dtype=np.float64), n)
    alpha = np.repeat(np.array([c.alpha for c in tilt_configs], dtype=np.float64), n)

    u_blocks, r_blocks, wr_blocks, pi_blocks = [], [], [], []
    for config in tilt_configs:
        utility = config.utility_for(table_a.B_toy)
        r = config.r_utility_for(utility)
        u_blocks.append(utility)
        r_blocks.append(r)
        wr_blocks.append(table_a.w_mc * r)
        pi_blocks.append(pi_tilt(table_a.w_mc, r))

    return TiltTable(
        pdg_id=table_a.pdg_id,
        table_a_hash=table_a.table_hash(),
        tilt_ids=tilt_ids,
        row_ref=row_ref,
        tilt_id_index=tilt_id_index,
        mode_code=mode_code,
        delta=delta,
        alpha=alpha,
        U=np.concatenate(u_blocks),
        r_utility=np.concatenate(r_blocks),
        w_mc_times_r=np.concatenate(wr_blocks),
        pi_tilt=np.concatenate(pi_blocks),
    )


# --- direct-sampling training path -------------------------------------------


def _derived_seed(base_seed: int, salt: int) -> int:
    seq = np.random.SeedSequence([int(base_seed), int(salt)])
    return int(seq.generate_state(1)[0])


def _make_tilt_run_spec(
    *,
    experiment_id: str,
    dataset: EmpiricalDataset,
    variant: str,
    feature_view: FeatureViewSpec,
    model: ModelSpec,
    evaluation: EvaluationSpec,
    device: str,
) -> RunSpec:
    from .gates import ScientificGateSpec

    target = TargetSpec(target_id=UTILITY_TILT_TARGET_ID, variant=variant, stage="empirical")
    ds = DatasetSpec(
        n_train=dataset.train.n_rows, n_validation=dataset.validation.n_rows, n_test=dataset.test.n_rows,
    )
    return RunSpec(
        experiment_id=experiment_id, target=target, pdg_id=int(dataset.dataset_spec.pdg_id),
        feature_view=feature_view, model=model, seed=int(dataset.dataset_spec.seed), dataset=ds,
        evaluation=evaluation, device=device, scientific_gates=ScientificGateSpec(),
    )


def _weighted_nll_from_log_prob(log_prob: np.ndarray, weights: np.ndarray) -> Optional[float]:
    log_prob = np.asarray(log_prob, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    finite = np.isfinite(log_prob) & np.isfinite(weights)
    total_weight = float(weights[finite].sum())
    if not finite.any() or total_weight <= 0.0:
        return None
    return float(-np.sum(weights[finite] * log_prob[finite]) / total_weight)


def run_direct_sampling_training(
    dataset_spec: EmpiricalDatasetSpec,
    store: ArtifactStore,
    *,
    experiment_id: str,
    tilt_id: str,
    feature_view: FeatureViewSpec,
    model: ModelSpec,
    sampler_seed: int,
    n_draws: Optional[int] = None,
    evaluation: Optional[EvaluationSpec] = None,
    device: str = "cpu",
    force: bool = False,
) -> Dict[str, Any]:
    """Build one PDG track's D9 dataset, draw training rows directly, train, evaluate.

    ``tilt_id`` is either :data:`NOMINAL_PHYSICAL_VARIANT_ID` (draw from
    ``pi_nominal``, the untilted physical law) or a key of
    :data:`TILT_CONFIG_BY_ID` (draw from ``pi_tilt``); both branches share
    every other step (dataset build, preprocessing, model init, optimizer,
    draw budget, evaluation path) so the nominal and tilted arms can never
    diverge except in which sampling table feeds the alias sampler. Mirrors
    ``empirical.run_empirical_single``'s resume/skip and technical status
    contract. The training loss receives no second ``w``/``r``/``w*r``
    factor: ``estimator.fit`` is called with ``sample_weight=None`` on the
    *resampled* array, which is the established unweighted-IID legacy path
    (arm A) applied to rows already drawn from the intended target law.
    """

    from Nflow.registry import create_density_estimator

    is_nominal = tilt_id == NOMINAL_PHYSICAL_VARIANT_ID
    if not is_nominal and tilt_id not in TILT_CONFIG_BY_ID:
        raise UtilityTiltError(
            "unknown tilt_id {!r}; expected {!r} or a key of TILT_CONFIG_BY_ID".format(
                tilt_id, NOMINAL_PHYSICAL_VARIANT_ID
            )
        )
    tilt_config = None if is_nominal else TILT_CONFIG_BY_ID[tilt_id]
    evaluation = evaluation or EvaluationSpec()
    model.validate()

    from .empirical import EmpiricalDataError

    try:
        dataset = build_empirical_dataset(dataset_spec)
    except EmpiricalDataError as exc:
        return {
            "run_id": None, "status": STATUS_FAILED, "reason": "dataset_build_failed",
            "error": str(exc), "tilt_id": tilt_id,
        }

    run_spec = _make_tilt_run_spec(
        experiment_id=experiment_id, dataset=dataset, variant=tilt_id, feature_view=feature_view,
        model=model, evaluation=evaluation, device=device,
    )
    run_spec.dataset.validate()
    run_spec.evaluation.validate()

    run_id = derive_run_id(run_spec)
    if not force and store.is_complete(run_spec):
        status = store.read_run_status(run_spec) or {}
        return {"run_id": run_id, "status": "skipped_completed", "tilt_id": tilt_id, "technical_status": status.get("technical_status", STATUS_COMPLETED)}

    started_at = utc_timestamp()
    try:
        table_a = build_nominal_table(dataset)
        pi = table_a.pi_nominal if is_nominal else tilt_pi_vector(table_a, tilt_config)
        alias_sampler = AliasSampler.from_probabilities(pi, source_hash=table_a.table_hash())

        draw_budget = int(n_draws) if n_draws is not None else int(dataset.train.n_rows)
        drawn_indices = alias_sampler.draw(draw_budget, seed=sampler_seed)

        view = _build_feature_view(feature_view)
        pipeline = FittedFeaturePipeline.fit(dataset.train.raw, view)
        resampled_raw = dataset.train.raw[drawn_indices]
        normalized_train = pipeline.transform_raw(resampled_raw)
        normalized_val = pipeline.transform_raw(dataset.validation.raw)

        estimator = create_density_estimator(model, dimension=DIMENSION, device=device)
        fit_result = estimator.fit(normalized_train, x_validation=normalized_val, seed=run_spec.seed)
        environment = capture_environment(requested_device=device)

        if fit_result.status != "ok":
            store.write_run(
                run_spec, environment=environment, dataset_manifest=dataset.manifest(),
                feature_pipeline_manifest=pipeline.manifest(), model_manifest=estimator.manifest(),
                fit_result=fit_result.to_dict(), metrics={"status": "fit_failed", "tilt_id": tilt_id},
                training_history=fit_result.train_history, samples={}, status=STATUS_FAILED,
                run_id=run_id, error="fit returned status={}".format(fit_result.status),
            )
            return {"run_id": run_id, "status": STATUS_FAILED, "reason": "fit_failed", "tilt_id": tilt_id}

        val_physical = dataset.validation.physical
        val_weights = np.ascontiguousarray(dataset.validation.raw[:, _W_COLUMN], dtype=np.float64)
        val_b_toy = compute_b_toy(val_physical, table_a.thresholds)
        val_r = np.ones_like(val_b_toy) if is_nominal else tilt_config.r_utility_for_b_toy(val_b_toy)
        normalized_lp_val = np.asarray(estimator.log_prob(normalized_val), dtype=np.float64)
        physical_log_q_val = pipeline.normalized_to_physical_log_prob(normalized_lp_val, dataset.validation.raw)

        theoretical_diag = theoretical_concentration_diagnostics(pi, draw_budget=draw_budget)
        empirical_diag = empirical_draw_diagnostics(drawn_indices, b_toy=table_a.B_toy)
        p0_b_toy = nominal_prevalence(table_a.B_toy, table_a.w_mc)
        theoretical_prevalence = (
            p0_b_toy if is_nominal
            else theoretical_tilted_prevalence(p0_b_toy, tilt_config.r_positive(), tilt_config.r_negative())
        )

        # Generated-distribution diagnostics: draw the same evaluation-sample
        # budget from the trained model for every variant, descriptive only
        # (no pass/fail threshold; see compact_distribution_summary).
        generated_sample_count = int(max(evaluation.ess_sample_count, evaluation.c2st_sample_count))
        generated_sample_seed = _derived_seed(run_spec.seed, 101)
        normalized_generated = estimator.sample(generated_sample_count, seed=generated_sample_seed)
        physical_generated = pipeline.inverse_to_physical(normalized_generated)
        generated_log_prob = np.asarray(estimator.log_prob(normalized_generated), dtype=np.float64)
        generated_finite_mask = np.isfinite(physical_generated).all(axis=1)
        physical_generated_finite = physical_generated[generated_finite_mask]
        generated_b_toy_occupancy = (
            float(np.mean(compute_b_toy(physical_generated_finite, table_a.thresholds)))
            if physical_generated_finite.shape[0] else None
        )
        generated_distribution_summary = compact_distribution_summary(physical_generated)
        declared_target_distribution_summary = compact_distribution_summary(dataset.train.physical, weights=pi)

        metrics: Dict[str, Any] = {
            "tilt_id": tilt_id,
            "variant_id": tilt_id,
            "tilt_config": (
                {"tilt_id": NOMINAL_PHYSICAL_VARIANT_ID, "mode": None, "delta": None, "alpha": None}
                if is_nominal else tilt_config.to_dict()
            ),
            "sampler_table_hash": alias_sampler.table_hash(),
            "source_table_hash": table_a.table_hash(),
            "init_seed": run_spec.seed,
            "sampler_seed": int(sampler_seed),
            "draw_budget": draw_budget,
            "theoretical_concentration": theoretical_diag,
            "empirical_draw_diagnostics": empirical_diag,
            "nominal_b_toy_prevalence": p0_b_toy,
            "theoretical_target_b_toy_prevalence": theoretical_prevalence,
            "theoretical_tilted_b_toy_prevalence": theoretical_prevalence,
            "sample_weight_applied_to_loss": False,
            "sampling_regime": NOMINAL_SAMPLING_REGIME if is_nominal else TILT_SAMPLING_REGIME,
            "nominal_validation_nll_weighted_by_w": _weighted_nll_from_log_prob(physical_log_q_val, val_weights),
            "tilted_validation_nll_weighted_by_w_times_r": _weighted_nll_from_log_prob(physical_log_q_val, val_weights * val_r),
            "fit_wall_time_seconds": fit_result.wall_time_seconds,
            "training_final": fit_result.train_history[-1] if fit_result.train_history else {},
            "estimator_family": (
                "unweighted_iid_direct_sample_from_pi_nominal" if is_nominal
                else "unweighted_iid_direct_sample_from_pi_tilt"
            ),
            "scientific_scope": "pipeline_verification_only_not_model_ranking",
            "diagnostic_only": True,
            "generated_sample_count": generated_sample_count,
            "generated_sample_seed": generated_sample_seed,
            "generated_b_toy_occupancy": generated_b_toy_occupancy,
            "generated_log_prob_finite_fraction": (
                float(np.mean(np.isfinite(generated_log_prob))) if generated_log_prob.size else None
            ),
            "generated_distribution_summary": generated_distribution_summary,
            "declared_target_distribution_summary": declared_target_distribution_summary,
            "ended_at": utc_timestamp(),
        }
        final_record = metrics["training_final"] if isinstance(metrics["training_final"], dict) else {}
        final_loss = final_record.get("feature_space_train_nll")
        metrics["final_train_nll"] = final_loss
        metrics["finite_train_loss"] = bool(final_loss is not None and np.isfinite(final_loss))

        paths = store.run_paths(run_spec)
        paths.run_dir.mkdir(parents=True, exist_ok=True)
        save_manifest = estimator.save(paths.run_dir)

        hashes = {
            "run_config_hash": run_spec.config_hash(),
            "dataset_manifest_hash": dataset.config_hash(),
            "feature_pipeline_hash": pipeline.config_hash(),
            "model_config_hash": run_spec.model.config_hash(),
            "sampler_table_hash": alias_sampler.table_hash(),
            "source_table_hash": table_a.table_hash(),
            "checkpoint_hash": save_manifest.get("checkpoint_hash"),
            "source_file_dataset_hash": dataset.source_file_dataset_hash,
        }
        store.write_run(
            run_spec, environment=environment, dataset_manifest=dataset.manifest(),
            feature_pipeline_manifest=pipeline.manifest(), model_manifest=estimator.manifest(),
            fit_result=fit_result.to_dict(), metrics=metrics, training_history=fit_result.train_history,
            samples={}, status=STATUS_COMPLETED, run_id=run_id, save_manifest=save_manifest,
            hashes=hashes, scientific_status="not_applicable", decision_scope="diagnostic_pipeline_verification_only",
            scientific_failure_reasons=[],
        )
        return {
            "run_id": run_id, "status": STATUS_COMPLETED, "technical_status": STATUS_COMPLETED,
            "tilt_id": tilt_id, "started_at": started_at, "metrics": metrics,
        }
    except Exception as exc:  # isolate: record and continue, matching empirical.run_empirical_single
        import traceback

        tb = traceback.format_exc()
        try:
            paths = store.run_paths(run_spec)
            paths.run_dir.mkdir(parents=True, exist_ok=True)
            store.write_run(
                run_spec, environment=capture_environment(requested_device=device),
                dataset_manifest={"status": "unavailable"}, feature_pipeline_manifest={"status": "unavailable"},
                model_manifest={"status": "unavailable"}, fit_result={"status": "failed"},
                metrics={"status": "error", "tilt_id": tilt_id}, training_history=[], samples={},
                status=STATUS_FAILED, run_id=run_id, error=tb,
            )
        except Exception:  # pragma: no cover - best-effort failure record
            pass
        return {"run_id": run_id, "status": STATUS_FAILED, "error": str(exc), "tilt_id": tilt_id}


def _build_feature_view(spec: FeatureViewSpec):
    """Local copy of ``empirical._build_feature_view`` (kept private, not re-exported)."""

    from ..data_contracts.feature_views import FeatureView

    if spec.pz_unit_gev is None:
        return FeatureView(spec.view_id)
    return FeatureView(spec.view_id, pz_unit_gev=spec.pz_unit_gev)


# --- campaign orchestration ----------------------------------------------------


def build_and_validate_pdg_tables(
    dataset_path: str, pdg_id: int, *, seed: int, val_fraction: float = 0.2, test_fraction: float = 0.2,
    max_rows: Optional[int] = None, allow_zero_weight: bool = True, tilt_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Build Table A + Table B (selected or all 20 tilts) for one PDG track and validate them.

    Validation checks (all diagnostic assertions, never a training gate):
    ``sum(pi_nominal) == 1``; ``sum(pi_tilt) == 1`` per tilt; alias-sampler
    empirical frequencies agree with the target table within Monte Carlo
    tolerance; direct summation of ``pi_tilt`` over ``B_toy`` agrees with the
    binary closed-form ``p_U(B_toy)``.
    """

    configs = [TILT_CONFIG_BY_ID[t] for t in tilt_ids] if tilt_ids else list(ALL_TILT_CONFIGS)
    dataset_spec = EmpiricalDatasetSpec(
        dataset_path=dataset_path, pdg_id=int(pdg_id), seed=int(seed), val_fraction=val_fraction,
        test_fraction=test_fraction, max_rows=max_rows, allow_zero_weight=allow_zero_weight,
    )
    dataset = build_empirical_dataset(dataset_spec)
    table_a = build_nominal_table(dataset)
    table_b = build_tilt_table(table_a, configs)

    checks: Dict[str, Any] = {
        "pi_nominal_sum": float(table_a.pi_nominal.sum()),
        "pi_nominal_sums_to_one": bool(np.isclose(table_a.pi_nominal.sum(), 1.0, atol=1e-9)),
        "per_tilt": {},
    }
    p0 = nominal_prevalence(table_a.B_toy, table_a.w_mc)
    n_per = table_b.n_rows_per_tilt
    for i, config in enumerate(configs):
        block_pi = table_b.pi_tilt[i * n_per: (i + 1) * n_per]
        direct = float(np.sum(block_pi * table_a.B_toy))
        closed_form = theoretical_tilted_prevalence(p0, config.r_positive(), config.r_negative())
        checks["per_tilt"][config.tilt_id] = {
            "pi_tilt_sum": float(block_pi.sum()),
            "pi_tilt_sums_to_one": bool(np.isclose(block_pi.sum(), 1.0, atol=1e-9)),
            "direct_summation_prevalence": direct,
            "closed_form_prevalence": closed_form,
            "agrees_within_tolerance": bool(np.isclose(direct, closed_form, atol=1e-9)),
        }
    return {"dataset": dataset, "table_a": table_a, "table_b": table_b, "checks": checks, "p0_b_toy": p0}


def validate_alias_sampler_against_table(pi: np.ndarray, *, n_draws: int, seed: int, atol: float = 0.02) -> Dict[str, Any]:
    """Draw from an alias sampler built on ``pi`` and compare empirical vs. target mass, aggregated by decile bucket."""

    sampler = AliasSampler.from_probabilities(pi)
    drawn = sampler.draw(n_draws, seed=seed)
    empirical_counts = np.bincount(drawn, minlength=pi.shape[0]).astype(np.float64)
    empirical_freq = empirical_counts / float(n_draws)
    order = np.argsort(pi)[::-1]
    n_top = min(10, pi.shape[0])
    top_idx = order[:n_top]
    target_top_mass = float(pi[top_idx].sum())
    empirical_top_mass = float(empirical_freq[top_idx].sum())
    return {
        "n_draws": int(n_draws),
        "target_top_mass": target_top_mass,
        "empirical_top_mass": empirical_top_mass,
        "abs_diff": abs(target_top_mass - empirical_top_mass),
        "within_tolerance": abs(target_top_mass - empirical_top_mass) <= atol,
        "sampler_table_hash": sampler.table_hash(),
    }


# --- D9 arena aggregate report (task section 5D) -----------------------------

ARENA_SCIENTIFIC_SCOPE = (
    "This arena compares physical-nominal versus moderately utility-tilted "
    "direct-sampling distributions on a bounded fixture pilot. It does not "
    "declare an optimal tilt, does not claim FairShip background enrichment, "
    "does not estimate a final background rate, does not treat B_toy as a "
    "physical endpoint, and does not treat U-A/U-P as calibrated "
    "probabilities. No composite score is computed and no variant is ranked "
    "a winner; FairShip/GEANT4 remains the final physical oracle."
)


def _maybe_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _load_run_artifacts(store: ArtifactStore, run_id: Optional[str]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Best-effort read of a run's persisted ``run_status.json`` / ``metrics.json`` by ``run_id``.

    Used only as a fallback when a caller's in-memory record (e.g. a
    ``skipped_completed`` result from :func:`run_direct_sampling_training`)
    does not itself carry a ``metrics`` payload.
    """

    if not run_id:
        return None, None
    run_dir = store.experiment_dir / run_id
    return _maybe_json(run_dir / "run_status.json"), _maybe_json(run_dir / "metrics.json")


def _arena_row_from_record(record: Dict[str, Any], store: ArtifactStore, *, pdg_id: int, epochs: Optional[int]) -> Dict[str, Any]:
    """Flatten one variant's training record (plus, if needed, its on-disk artifacts) into one report row."""

    variant_id = record.get("tilt_id") or record.get("variant_id")
    run_id = record.get("run_id")
    metrics = record.get("metrics")
    status_payload = None
    if metrics is None:
        status_payload, metrics = _load_run_artifacts(store, run_id)
    technical_status = (
        record.get("technical_status") or record.get("status")
        or (status_payload or {}).get("technical_status") or "unknown"
    )
    metrics = metrics or {}
    theoretical_concentration = metrics.get("theoretical_concentration") or {}
    empirical_draw = metrics.get("empirical_draw_diagnostics") or {}
    generated_summary = metrics.get("generated_distribution_summary") or {}

    return {
        "variant_id": variant_id,
        "run_id": run_id,
        "pdg_id": int(pdg_id),
        "epochs": epochs,
        "technical_status": technical_status,
        "scientific_status": "not_applicable",
        "decision_scope": "diagnostic_pipeline_verification_only",
        "sampling_regime": metrics.get("sampling_regime"),
        "init_seed": metrics.get("init_seed"),
        "sampler_seed": metrics.get("sampler_seed"),
        "draw_budget": metrics.get("draw_budget"),
        "source_table_hash": metrics.get("source_table_hash"),
        "sampler_table_hash": metrics.get("sampler_table_hash"),
        "nominal_b_toy_prevalence": metrics.get("nominal_b_toy_prevalence"),
        "theoretical_target_b_toy_prevalence": metrics.get("theoretical_target_b_toy_prevalence"),
        "n_eff": theoretical_concentration.get("n_eff"),
        "top_10_mass": theoretical_concentration.get("top_10_mass"),
        "top_100_mass": theoretical_concentration.get("top_100_mass"),
        "expected_n_unique": theoretical_concentration.get("expected_n_unique"),
        "empirical_unique_rows_drawn": empirical_draw.get("unique_rows_drawn"),
        "empirical_unique_rows_fraction": empirical_draw.get("unique_rows_fraction"),
        "empirical_max_reuse_count": empirical_draw.get("max_reuse_count"),
        "empirical_sampled_b_toy_fraction": empirical_draw.get("empirical_b_toy_fraction"),
        "final_train_nll": metrics.get("final_train_nll"),
        "finite_train_loss": metrics.get("finite_train_loss"),
        "nominal_validation_nll_weighted_by_w": metrics.get("nominal_validation_nll_weighted_by_w"),
        "tilted_validation_nll_weighted_by_w_times_r": metrics.get("tilted_validation_nll_weighted_by_w_times_r"),
        "generated_sample_count": metrics.get("generated_sample_count"),
        "generated_b_toy_occupancy": metrics.get("generated_b_toy_occupancy"),
        "generated_log_prob_finite_fraction": metrics.get("generated_log_prob_finite_fraction"),
        "generated_distribution_available": generated_summary.get("available"),
        "generated_distribution_finite_fraction": generated_summary.get("finite_fraction"),
        "sample_weight_applied_to_loss": metrics.get("sample_weight_applied_to_loss"),
        "estimator_family": metrics.get("estimator_family"),
        "scientific_scope": metrics.get("scientific_scope"),
    }


_ARENA_ROW_SORT_INDEX: Dict[str, int] = {vid: i for i, vid in enumerate(ARENA_VARIANT_IDS)}


def _arena_row_sort_key(row: Dict[str, Any]) -> Tuple[int, str]:
    variant_id = row.get("variant_id") or ""
    return (_ARENA_ROW_SORT_INDEX.get(variant_id, len(_ARENA_ROW_SORT_INDEX)), variant_id)


ARENA_REPORT_COLUMNS: Tuple[str, ...] = (
    "variant_id", "run_id", "pdg_id", "epochs", "technical_status", "scientific_status",
    "decision_scope", "sampling_regime", "init_seed", "sampler_seed", "draw_budget",
    "source_table_hash", "sampler_table_hash", "nominal_b_toy_prevalence",
    "theoretical_target_b_toy_prevalence", "n_eff", "top_10_mass", "top_100_mass",
    "expected_n_unique", "empirical_unique_rows_drawn", "empirical_unique_rows_fraction",
    "empirical_max_reuse_count", "empirical_sampled_b_toy_fraction", "final_train_nll",
    "finite_train_loss", "nominal_validation_nll_weighted_by_w",
    "tilted_validation_nll_weighted_by_w_times_r", "generated_sample_count",
    "generated_b_toy_occupancy", "generated_log_prob_finite_fraction",
    "generated_distribution_available", "generated_distribution_finite_fraction",
    "sample_weight_applied_to_loss", "estimator_family", "scientific_scope",
)


def build_arena_report(
    records: Sequence[Dict[str, Any]],
    store: ArtifactStore,
    *,
    out_dir: Path,
    pdg_id: int,
    experiment_id: str,
    epochs: Optional[int] = None,
) -> Dict[str, Any]:
    """Build the deterministic D9 arena summary (JSON authoritative, CSV, Markdown).

    One row per variant (never a composite score, never a declared winner),
    sorted deterministically by :data:`ARENA_VARIANT_IDS` order (any variant
    id outside that list, e.g. from a future extension, sorts after it,
    alphabetically). ``records`` are the dicts returned by
    :func:`run_direct_sampling_training` -- a ``skipped_completed`` record
    (no in-memory ``metrics``) is transparently backfilled from the run's
    persisted ``metrics.json`` via ``store``.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        _arena_row_from_record(record, store, pdg_id=pdg_id, epochs=epochs) for record in records
    ]
    rows.sort(key=_arena_row_sort_key)

    generated_variant_ids = tuple(r["variant_id"] for r in rows)
    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "pdg_id": int(pdg_id),
        "arena_variant_ids_declared": list(ARENA_VARIANT_IDS),
        "arena_variant_ids_present": list(generated_variant_ids),
        "epochs": epochs,
        "scientific_scope": ARENA_SCIENTIFIC_SCOPE,
        "no_composite_score": True,
        "no_winner_declared": True,
        "rows": rows,
    }
    (out_dir / "arena_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))

    with (out_dir / "arena_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ARENA_REPORT_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c) for c in ARENA_REPORT_COLUMNS})

    lines = [
        "# D9 Utility-Tilt Arena — Summary (v1)",
        "",
        "**Scientific scope.** {}".format(ARENA_SCIENTIFIC_SCOPE),
        "",
        "No composite score is computed; no variant is ranked or declared a winner. "
        "Rows are sorted deterministically by the declared arena variant order.",
        "",
        "| variant | technical | draw budget | nominal p(B_toy) | target p(B_toy) | "
        "empirical B_toy frac | unique rows frac | max reuse | final train NLL | "
        "nominal val NLL(w) | tilted val NLL(w*r) | generated B_toy occ |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {variant_id} | {technical_status} | {draw_budget} | {p0} | {ptarget} | "
            "{ebtoy} | {ufrac} | {reuse} | {trainnll} | {nomnll} | {tiltnll} | {genbtoy} |".format(
                variant_id=row["variant_id"],
                technical_status=row["technical_status"],
                draw_budget=_fmt_arena(row["draw_budget"]),
                p0=_fmt_arena(row["nominal_b_toy_prevalence"]),
                ptarget=_fmt_arena(row["theoretical_target_b_toy_prevalence"]),
                ebtoy=_fmt_arena(row["empirical_sampled_b_toy_fraction"]),
                ufrac=_fmt_arena(row["empirical_unique_rows_fraction"]),
                reuse=_fmt_arena(row["empirical_max_reuse_count"]),
                trainnll=_fmt_arena(row["final_train_nll"]),
                nomnll=_fmt_arena(row["nominal_validation_nll_weighted_by_w"]),
                tiltnll=_fmt_arena(row["tilted_validation_nll_weighted_by_w_times_r"]),
                genbtoy=_fmt_arena(row["generated_b_toy_occupancy"]),
            )
        )
    lines.append("")
    lines.append(
        "Diagnostics only: ESS/N_eff, top-k mass, uniqueness, and reuse never gate a "
        "run's technical or scientific status (task section 3)."
    )
    (out_dir / "arena_summary.md").write_text("\n".join(lines) + "\n")

    return payload


def _fmt_arena(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return "{:.4g}".format(value)
    return str(value)
