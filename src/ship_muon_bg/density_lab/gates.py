"""Model-independent scientific gate layer for the controlled density lab.

This module answers a single, deliberately narrow question: given a metric
bundle produced by :func:`ship_muon_bg.density_lab.evaluator.evaluate_run`, what
is the *scientific* status of the run — separate from whether the run executed
without a technical error?

Design constraints (enforced by the tests):

* **Pure and light.** ``gates.py`` imports only the standard library. It never
  imports numpy, torch, scikit-learn, matplotlib, mlflow, or any density model.
  Finiteness is decided with :func:`math.isfinite`, not numpy.
* **No recomputation.** Gates *consume* the existing metric bundle. They never
  re-estimate ESS, KL, C2ST, or rare-mode mass. Statistical estimation lives in
  ``metrics.py``; classification lives here.
* **Threshold classes are explicit.** Every gate declares which of four classes
  it belongs to, so a reader can never mistake a provisional engineering
  reference for a preregistered physics criterion:

  - ``mathematical_invariant`` — a quantity that must be finite for the estimate
    to mean anything at all (non-finite density/loss);
  - ``catastrophic_guard`` — a hard, model-independent failure floor
    (ESS/N below the configured catastrophic threshold; D5 producing zero
    generated rare-region samples);
  - ``provisional_engineering_gate`` — a working engineering reference that is
    reported but does **not** decide scientific pass/fail by default;
  - ``preregistered_scientific_gate`` — where a preregistered physics criterion
    would live. None are active by default: forward KL and C2ST are reported
    only and are never used as a sole pass/fail selector.

Scientific statuses (``pass``, ``fail``, ``catastrophic``, ``inconclusive``) are
aggregated from the per-gate outcomes by a documented severity precedence:

    catastrophic  >  fail  >  inconclusive  >  pass

``catastrophic`` dominates: a definitive hard failure is reported even when an
unrelated metric is missing. ``inconclusive`` means a mandatory metric was
absent or malformed, so a responsible ``pass`` or ``fail`` cannot be assigned.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import numbers
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

GATE_SCHEMA_VERSION = "0"

# What a ``scientific_status`` of "pass" is allowed to mean. Recorded verbatim on
# every result (and in run_status.json) so a reader can never over-read it:
#
#     scientific_status = "pass"  <=>  every *currently active* gate passed.
#
# It does **not** assert sufficient rare-mode fidelity, a validated minimum
# capacity, or final scientific acceptance. The rare-region mass ratio is
# report-only under this scope, so a run whose rare mass is far below the target
# still reads "pass"; that is a statement about the active gate set, not about
# the physics. Bump the suffix when the active gate set changes.
DECISION_SCOPE = "active_gates_v0"

# -- threshold classes -------------------------------------------------------
THRESHOLD_MATHEMATICAL_INVARIANT = "mathematical_invariant"
THRESHOLD_CATASTROPHIC_GUARD = "catastrophic_guard"
THRESHOLD_PROVISIONAL_ENGINEERING = "provisional_engineering_gate"
THRESHOLD_PREREGISTERED_SCIENTIFIC = "preregistered_scientific_gate"

# -- scientific statuses -----------------------------------------------------
STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_CATASTROPHIC = "catastrophic"
STATUS_INCONCLUSIVE = "inconclusive"

# Single source of truth for what a gate-produced scientific_status can be. Used
# by campaign.py to recognize a persisted run_status.json value that is not a
# real gate verdict (missing, corrupted, or from a schema this code doesn't
# know) so it can be reported as "unavailable" instead of silently trusted.
SCIENTIFIC_STATUSES = frozenset(
    {STATUS_PASS, STATUS_FAIL, STATUS_CATASTROPHIC, STATUS_INCONCLUSIVE}
)
STATUS_UNAVAILABLE = "unavailable"

# -- per-gate outcomes (a gate with outcome "report" never changes status) ---
OUTCOME_PASS = "pass"
OUTCOME_FAIL = "fail"
OUTCOME_CATASTROPHIC = "catastrophic"
OUTCOME_INCONCLUSIVE = "inconclusive"
OUTCOME_REPORT = "report"

# Severity precedence used to aggregate gate outcomes into a run status. A
# "report" outcome has severity 0 (never elevates the status).
_SEVERITY = {
    OUTCOME_REPORT: 0,
    OUTCOME_PASS: 0,
    OUTCOME_INCONCLUSIVE: 1,
    OUTCOME_FAIL: 2,
    OUTCOME_CATASTROPHIC: 3,
}
_STATUS_BY_SEVERITY = {
    0: STATUS_PASS,
    1: STATUS_INCONCLUSIVE,
    2: STATUS_FAIL,
    3: STATUS_CATASTROPHIC,
}


class GateConfigError(ValueError):
    """An invalid scientific gate specification."""


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _canonical_hash(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ScientificGateSpec:
    """Frozen, canonically-serializable scientific gate configuration.

    ``catastrophic_ess_threshold`` defaults to ``None`` meaning "inherit from
    :class:`~ship_muon_bg.density_lab.config.EvaluationSpec`". This keeps a
    single source of truth for the ESS catastrophic floor: the concrete value is
    resolved from the evaluation config at gate-evaluation time (see
    :meth:`resolve`). Setting it explicitly to a value that contradicts the
    evaluation config is rejected by :meth:`validate`.
    """

    gate_schema_version: str = GATE_SCHEMA_VERSION
    # None => inherit EvaluationSpec.catastrophic_ess_threshold (single source of
    # truth). A concrete value here must match the evaluation config.
    catastrophic_ess_threshold: Optional[float] = None
    # If True, a D5 run missing rare-mode metrics is inconclusive rather than
    # silently treated as a non-rare target.
    require_d5_rare_metrics: bool = True
    # Provisional engineering reference for the rare-region mass ratio. Reported
    # only; never a final physics criterion and never changes scientific status.
    # None keeps the ratio purely reported with no reference line.
    rare_region_mass_ratio_provisional_floor: Optional[float] = None

    def validate(self, evaluation: Any = None) -> None:
        if not isinstance(self.gate_schema_version, str) or not self.gate_schema_version:
            raise GateConfigError("gate_schema_version must be a non-empty string")
        if not isinstance(self.require_d5_rare_metrics, bool):
            raise GateConfigError("require_d5_rare_metrics must be a bool")
        ess = self.catastrophic_ess_threshold
        if ess is not None:
            if not isinstance(ess, (int, float)) or isinstance(ess, bool):
                raise GateConfigError("catastrophic_ess_threshold must be a number or None")
            if not (0.0 < float(ess) < 1.0):
                raise GateConfigError("catastrophic_ess_threshold must lie in (0, 1)")
            if evaluation is not None:
                eval_ess = getattr(evaluation, "catastrophic_ess_threshold", None)
                if eval_ess is not None and float(ess) != float(eval_ess):
                    raise GateConfigError(
                        "ScientificGateSpec.catastrophic_ess_threshold ({}) contradicts "
                        "EvaluationSpec.catastrophic_ess_threshold ({}); leave it None to "
                        "inherit the single source of truth".format(ess, eval_ess)
                    )
        floor = self.rare_region_mass_ratio_provisional_floor
        if floor is not None and (not isinstance(floor, (int, float)) or isinstance(floor, bool)):
            raise GateConfigError("rare_region_mass_ratio_provisional_floor must be a number or None")

    def resolve(self, evaluation: Any) -> "ScientificGateSpec":
        """Return a spec with the ESS threshold resolved to a concrete value.

        When ``catastrophic_ess_threshold`` is ``None`` the value is taken from
        ``evaluation.catastrophic_ess_threshold`` (the single source of truth).
        """

        ess = self.catastrophic_ess_threshold
        if ess is None:
            ess = getattr(evaluation, "catastrophic_ess_threshold", 0.01)
        return dataclasses.replace(self, catastrophic_ess_threshold=float(ess))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_schema_version": self.gate_schema_version,
            "catastrophic_ess_threshold": self.catastrophic_ess_threshold,
            "require_d5_rare_metrics": self.require_d5_rare_metrics,
            "rare_region_mass_ratio_provisional_floor": (
                self.rare_region_mass_ratio_provisional_floor
            ),
        }

    def config_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass
class ScientificGateResult:
    """JSON-safe result of scientific gate evaluation for one run."""

    gate_schema_version: str
    scientific_status: str
    scientific_failure_reasons: List[Dict[str, Any]]
    gate_results: List[Dict[str, Any]]
    gate_config_hash: str
    # Bounds what scientific_status is allowed to claim; see DECISION_SCOPE.
    decision_scope: str = DECISION_SCOPE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_schema_version": self.gate_schema_version,
            "scientific_status": self.scientific_status,
            "decision_scope": self.decision_scope,
            "scientific_failure_reasons": [dict(r) for r in self.scientific_failure_reasons],
            "gate_results": [dict(g) for g in self.gate_results],
            "gate_config_hash": self.gate_config_hash,
        }


# --- metric access helpers (pure) -------------------------------------------

_MISSING = object()


def _fetch(metrics: Mapping[str, Any], path: Tuple[str, ...]) -> Any:
    node: Any = metrics
    for key in path:
        if not isinstance(node, Mapping) or key not in node:
            return _MISSING
        node = node[key]
    return node


def _classify_number(value: Any) -> str:
    """Classify a metric value: ``finite`` / ``nonfinite`` / ``malformed`` / ``missing``."""

    if value is _MISSING or value is None:
        return "missing"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "malformed"
    return "finite" if math.isfinite(float(value)) else "nonfinite"


# Invariants for the non-finite *evidence* fields consumed by the density
# finiteness gate. Both classifiers are total and return one of
# ``positive`` / ``zero`` / ``impossible`` / ``missing``.
#
# Documented invariant: a counter is a non-boolean integer >= 0, and a rate is a
# non-boolean finite real in [0, 1]. A value outside its domain (negative count,
# rate above 1 or non-finite, a bool used as a number, a string) is *impossible*,
# not merely unknown: it means the metric writer is broken and the bundle can no
# longer certify finiteness. An impossible value is therefore treated exactly
# like missing evidence -- it can never yield ``pass``, and it degrades the gate
# to ``inconclusive`` unless some other field independently proves a catastrophe
# (catastrophic evidence always takes precedence; see _density_finiteness_gate).
#
# numbers.Integral/Real are used instead of int/float so NumPy scalars (np.int64,
# np.float64) validate identically without importing NumPy here.


def _classify_count(value: Any) -> str:
    """Classify a non-finite-row counter (non-bool integer >= 0)."""

    if value is _MISSING or value is None:
        return "missing"
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        return "impossible"
    count = int(value)
    if count < 0:
        return "impossible"
    return "positive" if count > 0 else "zero"


def _classify_rate(value: Any) -> str:
    """Classify a non-finite-density rate (non-bool finite real in [0, 1])."""

    if value is _MISSING or value is None:
        return "missing"
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return "impossible"
    rate = float(value)
    if not math.isfinite(rate) or not (0.0 <= rate <= 1.0):
        return "impossible"
    return "positive" if rate > 0.0 else "zero"


def _gate(
    gate_id: str,
    threshold_class: str,
    outcome: str,
    *,
    active: bool,
    value: Any = None,
    threshold: Any = None,
    message: str = "",
) -> Dict[str, Any]:
    return {
        "gate_id": gate_id,
        "threshold_class": threshold_class,
        "active": bool(active),
        "outcome": outcome,
        "value": value,
        "threshold": threshold,
        "message": message,
    }


# --- individual gates -------------------------------------------------------


# Aggregate scalars. metrics.held_out_nll / metrics.forward_kl deliberately
# average over the *finite subset* of rows, so a finite scalar here proves
# nothing on its own about the discarded rows -- the counters below are what
# make this gate sound.
_FINITENESS_SCALARS = (
    ("held_out.held_out_nll", ("held_out", "held_out_nll")),
    ("forward_kl.forward_kl", ("forward_kl", "forward_kl")),
)
# Per-row non-finite evidence recorded alongside those aggregates. The evaluator
# writes all three unconditionally, so a bundle lacking one cannot be certified.
_FINITENESS_COUNTS = (
    ("held_out.non_finite_count", ("held_out", "non_finite_count")),
    ("forward_kl.non_finite_count", ("forward_kl", "non_finite_count")),
    ("non_finite_density.non_finite_count", ("non_finite_density", "non_finite_count")),
)
_FINITENESS_RATES = (
    (
        "non_finite_density.non_finite_density_rate",
        ("non_finite_density", "non_finite_density_rate"),
    ),
)


def _density_finiteness_gate(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    """Non-finite density/loss is a mathematical-invariant catastrophe.

    The gate consumes *all* recorded finiteness evidence, not just the aggregate
    scalars: ``metrics.held_out_nll`` and ``metrics.forward_kl`` compute their
    scalar over the finite rows while separately recording ``non_finite_count``,
    and the evaluator also writes a ``non_finite_density`` block. Checking only
    the aggregates would let a model that assigns non-finite density to some
    held-out rows report ``pass``, which is exactly what this mathematical
    invariant exists to forbid.

    Decision order (catastrophic evidence always wins over missing evidence):

    1. any non-finite aggregate, any positive counter, or a positive rate
       -> ``catastrophic``;
    2. otherwise any missing/malformed/impossible evidence -> ``inconclusive``;
    3. otherwise (finite aggregates, all counters and the rate zero) -> ``pass``.

    Counter/rate domains are defined by :func:`_classify_count` and
    :func:`_classify_rate`. No density or statistic is recomputed here.
    """

    values: Dict[str, Any] = {}
    catastrophic: List[str] = []
    unusable: List[str] = []

    def record(name: str, raw: Any) -> Any:
        values[name] = None if raw is _MISSING else raw
        return raw

    for name, path in _FINITENESS_SCALARS:
        raw = record(name, _fetch(metrics, path))
        cls = _classify_number(raw)
        if cls == "nonfinite":
            catastrophic.append("{} is non-finite".format(name))
        elif cls != "finite":
            unusable.append("{} ({})".format(name, cls))

    for name, path in _FINITENESS_COUNTS:
        raw = record(name, _fetch(metrics, path))
        cls = _classify_count(raw)
        if cls == "positive":
            catastrophic.append("{} = {}".format(name, int(raw)))
        elif cls != "zero":
            unusable.append("{} ({})".format(name, cls))

    for name, path in _FINITENESS_RATES:
        raw = record(name, _fetch(metrics, path))
        cls = _classify_rate(raw)
        if cls == "positive":
            catastrophic.append("{} = {:.4g}".format(name, float(raw)))
        elif cls != "zero":
            unusable.append("{} ({})".format(name, cls))

    if catastrophic:
        return _gate(
            "density_finiteness",
            THRESHOLD_MATHEMATICAL_INVARIANT,
            OUTCOME_CATASTROPHIC,
            active=True,
            value=values,
            message="non-finite density/loss (mathematical invariant violated): "
            + ", ".join(sorted(catastrophic)),
        )
    if unusable:
        return _gate(
            "density_finiteness",
            THRESHOLD_MATHEMATICAL_INVARIANT,
            OUTCOME_INCONCLUSIVE,
            active=True,
            value=values,
            message="mandatory finiteness evidence missing, malformed or impossible: "
            + ", ".join(sorted(unusable)),
        )
    return _gate(
        "density_finiteness",
        THRESHOLD_MATHEMATICAL_INVARIANT,
        OUTCOME_PASS,
        active=True,
        value=values,
        message="held-out NLL and forward KL are finite; no non-finite density rows recorded",
    )


def _ess_catastrophic_gate(
    metrics: Mapping[str, Any], threshold: float
) -> Dict[str, Any]:
    """ESS/N below the configured catastrophic floor is a catastrophic guard.

    Boundary semantics: the guard fires iff ``ess_over_n < threshold`` (strictly
    less-than), matching ``metrics.importance_ess``. A value exactly at the
    threshold is *not* catastrophic.
    """

    ess_node = _fetch(metrics, ("importance_ess",))
    if ess_node is _MISSING or not isinstance(ess_node, Mapping):
        return _gate(
            "importance_ess_catastrophic",
            THRESHOLD_CATASTROPHIC_GUARD,
            OUTCOME_INCONCLUSIVE,
            active=True,
            threshold=threshold,
            message="importance_ess metric missing",
        )
    if "ess_over_n" in ess_node:
        value = ess_node["ess_over_n"]
        cls = _classify_number(value)
        if cls in ("missing", "malformed"):
            return _gate(
                "importance_ess_catastrophic",
                THRESHOLD_CATASTROPHIC_GUARD,
                OUTCOME_INCONCLUSIVE,
                active=True,
                value=None if value is _MISSING else value,
                threshold=threshold,
                message="importance_ess.ess_over_n malformed",
            )
        if cls == "nonfinite":
            return _gate(
                "importance_ess_catastrophic",
                THRESHOLD_CATASTROPHIC_GUARD,
                OUTCOME_CATASTROPHIC,
                active=True,
                value=value,
                threshold=threshold,
                message="importance_ess.ess_over_n is non-finite",
            )
        value = float(value)
        if value < threshold:
            return _gate(
                "importance_ess_catastrophic",
                THRESHOLD_CATASTROPHIC_GUARD,
                OUTCOME_CATASTROPHIC,
                active=True,
                value=value,
                threshold=threshold,
                message="ESS/N {:.4g} below catastrophic threshold {:.4g}".format(
                    value, threshold
                ),
            )
        return _gate(
            "importance_ess_catastrophic",
            THRESHOLD_CATASTROPHIC_GUARD,
            OUTCOME_PASS,
            active=True,
            value=value,
            threshold=threshold,
            message="ESS/N {:.4g} at or above catastrophic threshold {:.4g}".format(
                value, threshold
            ),
        )
    # No ess_over_n: the evaluator records {"error": ..., "catastrophic": True}
    # when importance weights are non-finite. Treat that as a catastrophe.
    if ess_node.get("error") or ess_node.get("catastrophic"):
        return _gate(
            "importance_ess_catastrophic",
            THRESHOLD_CATASTROPHIC_GUARD,
            OUTCOME_CATASTROPHIC,
            active=True,
            value=None,
            threshold=threshold,
            message="importance_ess reported error/catastrophic (non-finite weights): {}".format(
                ess_node.get("error", "flagged catastrophic")
            ),
        )
    return _gate(
        "importance_ess_catastrophic",
        THRESHOLD_CATASTROPHIC_GUARD,
        OUTCOME_INCONCLUSIVE,
        active=True,
        threshold=threshold,
        message="importance_ess present but ess_over_n absent",
    )


def _d5_zero_rare_gate(
    metrics: Mapping[str, Any], *, require_metrics: bool
) -> Dict[str, Any]:
    """D5 producing zero generated rare-region samples is a catastrophic guard.

    ``observed_q_rare_sample_count`` is a count and must satisfy the same
    documented invariant as the finiteness counters: a non-boolean integer >= 0
    (see :func:`_classify_count`). The value is never coerced with ``int()``
    before validation — a malformed count (negative, fractional, a bool, a
    string) must never be silently treated as a real sample count.

    ``require_metrics`` controls the waiver for a ``rare_mode`` block that is
    *entirely absent* only (a target/run that legitimately has no rare-mode
    metrics). It does not waive a *malformed* count, and it does not waive a
    block that is present but omits the mandatory count key: a rare_mode block
    that is present but corrupt or partial is always ``inconclusive`` and
    active, whether or not the metric is required, because "rare metrics are
    optional here" is not "garbage is acceptable when supplied".
    """

    block_absent = _fetch(metrics, ("rare_mode",)) is _MISSING
    value = _fetch(metrics, ("rare_mode", "observed_q_rare_sample_count"))
    cls = _classify_count(value)

    if cls == "missing":
        if not block_absent:
            return _gate(
                "d5_zero_rare_samples",
                THRESHOLD_CATASTROPHIC_GUARD,
                OUTCOME_INCONCLUSIVE,
                active=True,
                value=None,
                threshold=0,
                message="D5 rare_mode block present but missing "
                "observed_q_rare_sample_count",
            )
        outcome = OUTCOME_INCONCLUSIVE if require_metrics else OUTCOME_REPORT
        return _gate(
            "d5_zero_rare_samples",
            THRESHOLD_CATASTROPHIC_GUARD,
            outcome,
            active=require_metrics,
            value=None,
            threshold=0,
            message=(
                "D5 rare-mode metric missing"
                + ("" if require_metrics else "; not required for this run")
            ),
        )
    if cls == "impossible":
        return _gate(
            "d5_zero_rare_samples",
            THRESHOLD_CATASTROPHIC_GUARD,
            OUTCOME_INCONCLUSIVE,
            active=True,
            value=value,
            threshold=0,
            message="observed_q_rare_sample_count malformed (must be a non-negative "
            "integer): {!r}".format(value),
        )
    count = int(value)
    if cls == "zero":
        return _gate(
            "d5_zero_rare_samples",
            THRESHOLD_CATASTROPHIC_GUARD,
            OUTCOME_CATASTROPHIC,
            active=True,
            value=count,
            threshold=0,
            message="D5 generated zero rare-region samples (mode collapse)",
        )
    return _gate(
        "d5_zero_rare_samples",
        THRESHOLD_CATASTROPHIC_GUARD,
        OUTCOME_PASS,
        active=True,
        value=count,
        threshold=0,
        message="D5 generated {} rare-region sample(s)".format(count),
    )


def _rare_region_mass_ratio_gate(
    metrics: Mapping[str, Any], *, provisional_floor: Optional[float]
) -> Dict[str, Any]:
    """Report the rare-region mass ratio (provisional engineering reference).

    This gate never changes the scientific status. A provisional floor, if set,
    is a working engineering reference only — explicitly not a physics criterion.
    """

    value = _fetch(metrics, ("rare_mode", "rare_region_mass_ratio"))
    cls = _classify_number(value)
    reported = None if cls in ("missing", "malformed") else float(value) if cls == "finite" else value
    message = "rare-region mass ratio reported only (provisional engineering reference; not a physics criterion)"
    if provisional_floor is not None and cls == "finite" and float(value) < float(provisional_floor):
        message = (
            "rare-region mass ratio {:.4g} below provisional engineering reference {:.4g} "
            "(reported only; not a physics criterion)".format(float(value), float(provisional_floor))
        )
    return _gate(
        "rare_region_mass_ratio",
        THRESHOLD_PROVISIONAL_ENGINEERING,
        OUTCOME_REPORT,
        active=False,
        value=reported,
        threshold=provisional_floor,
        message=message,
    )


def _reported_scientific_reference(
    gate_id: str, metrics: Mapping[str, Any], path: Tuple[str, ...], label: str
) -> Dict[str, Any]:
    """A preregistered-scientific-gate slot that is reported only (inactive).

    Forward KL and C2ST live here: reported, never used as a sole pass/fail
    selector, and never a preregistered physics threshold by default.
    """

    value = _fetch(metrics, path)
    cls = _classify_number(value)
    reported = None if cls in ("missing", "malformed") else float(value) if cls == "finite" else value
    return _gate(
        gate_id,
        THRESHOLD_PREREGISTERED_SCIENTIFIC,
        OUTCOME_REPORT,
        active=False,
        value=reported,
        threshold=None,
        message="{} reported only; no preregistered scientific threshold; "
        "not a sole pass/fail selector".format(label),
    )


# --- public entry point -----------------------------------------------------


def evaluate_scientific_gates(
    metrics: Mapping[str, Any],
    *,
    target_id: str,
    gate_spec: ScientificGateSpec,
) -> ScientificGateResult:
    """Classify the scientific status of a run from its metric bundle.

    ``gate_spec`` must already be resolved (concrete ESS threshold); callers
    typically pass ``spec.resolve(evaluation)``. If ``catastrophic_ess_threshold``
    is ``None`` it falls back to ``0.01`` so the function is total.

    ``gate_spec`` is validated here (not only by ``ExperimentConfig.from_dict``)
    so a hand-built spec passed directly to this function -- e.g. through a
    manually constructed ``RunSpec`` -- cannot smuggle a non-bool
    ``require_d5_rare_metrics`` (or any other malformed field) past the D5 gate
    activation check (Codex P2).
    """

    if not isinstance(metrics, Mapping):
        raise GateConfigError("metrics must be a mapping")

    gate_spec.validate()

    threshold = gate_spec.catastrophic_ess_threshold
    if threshold is None:
        threshold = 0.01
    threshold = float(threshold)

    gates: List[Dict[str, Any]] = []
    gates.append(_density_finiteness_gate(metrics))
    gates.append(_ess_catastrophic_gate(metrics, threshold))

    is_d5 = str(target_id) == "D5"
    if is_d5:
        gates.append(
            _d5_zero_rare_gate(metrics, require_metrics=gate_spec.require_d5_rare_metrics)
        )
        gates.append(
            _rare_region_mass_ratio_gate(
                metrics,
                provisional_floor=gate_spec.rare_region_mass_ratio_provisional_floor,
            )
        )

    # Reported-only scientific references (never a sole pass/fail selector).
    gates.append(
        _reported_scientific_reference(
            "forward_kl_reference", metrics, ("forward_kl", "forward_kl"), "forward KL"
        )
    )
    gates.append(
        _reported_scientific_reference(
            "c2st_reference", metrics, ("c2st", "c2st_accuracy"), "C2ST accuracy"
        )
    )

    # Aggregate by documented severity precedence.
    worst = 0
    for g in gates:
        worst = max(worst, _SEVERITY.get(g["outcome"], 0))
    scientific_status = _STATUS_BY_SEVERITY[worst]

    reasons: List[Dict[str, Any]] = []
    for g in gates:
        if g["outcome"] in (OUTCOME_CATASTROPHIC, OUTCOME_FAIL, OUTCOME_INCONCLUSIVE):
            reasons.append(
                {
                    "gate_id": g["gate_id"],
                    "threshold_class": g["threshold_class"],
                    "outcome": g["outcome"],
                    "message": g["message"],
                }
            )

    return ScientificGateResult(
        gate_schema_version=gate_spec.gate_schema_version,
        scientific_status=scientific_status,
        scientific_failure_reasons=reasons,
        gate_results=gates,
        gate_config_hash=gate_spec.config_hash(),
        decision_scope=DECISION_SCOPE,
    )
