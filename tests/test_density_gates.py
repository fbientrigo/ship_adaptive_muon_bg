"""Behavioral tests for the model-independent scientific gate layer.

These tests define the contract of ``density_lab.gates``: what makes a run a
scientific ``pass`` / ``fail`` / ``catastrophic`` / ``inconclusive``, that the
gate layer never recomputes statistics, that it imports no heavy dependency, and
that threshold classes are reported explicitly. They are pure and fast.
"""

from __future__ import annotations

import sys

import pytest

from ship_muon_bg.density_lab.gates import (
    DECISION_SCOPE,
    GATE_SCHEMA_VERSION,
    STATUS_CATASTROPHIC,
    STATUS_INCONCLUSIVE,
    STATUS_PASS,
    THRESHOLD_CATASTROPHIC_GUARD,
    THRESHOLD_MATHEMATICAL_INVARIANT,
    THRESHOLD_PREREGISTERED_SCIENTIFIC,
    THRESHOLD_PROVISIONAL_ENGINEERING,
    GateConfigError,
    ScientificGateSpec,
    evaluate_scientific_gates,
)


def _healthy_metrics(ess=0.8, *, d5=False, rare_count=25, rare_ratio=1.0):
    """A metric bundle that passes every active gate.

    Evaluator-shaped: ``held_out``/``forward_kl`` carry the ``non_finite_count``
    they record alongside their finite-subset aggregate, and the ``non_finite_density``
    block the evaluator always writes is present. The finiteness gate requires
    this evidence, so a healthy bundle must supply it as zeros.
    """

    metrics = {
        "held_out": {"held_out_nll": 1.23, "n_test": 2000, "non_finite_count": 0},
        "forward_kl": {"forward_kl": 0.05, "n_effective": 2000, "non_finite_count": 0},
        "non_finite_density": {
            "non_finite_density_rate": 0.0,
            "non_finite_count": 0,
            "n": 2000,
        },
        "importance_ess": {"ess_over_n": ess, "catastrophic": ess < 0.01},
        "c2st": {"c2st_accuracy": 0.55},
    }
    if d5:
        metrics["rare_mode"] = {
            "observed_q_rare_sample_count": rare_count,
            "rare_region_mass_ratio": rare_ratio,
            "target_rare_mass": 1e-3,
            "q_rare_region_mass": 1e-3 * rare_ratio,
        }
    return metrics


def _spec(**kw):
    return ScientificGateSpec(**kw).resolve(_FakeEval(kw.pop("_eval_ess", 0.01)))


class _FakeEval:
    def __init__(self, ess):
        self.catastrophic_ess_threshold = ess


# --- happy path -------------------------------------------------------------


def test_healthy_run_passes():
    result = evaluate_scientific_gates(
        _healthy_metrics(ess=0.8), target_id="D3", gate_spec=_spec()
    )
    assert result.scientific_status == STATUS_PASS
    assert result.scientific_failure_reasons == []
    assert result.gate_schema_version == GATE_SCHEMA_VERSION
    assert result.gate_config_hash  # non-empty


# --- ESS catastrophic guard + boundary semantics ----------------------------


def test_ess_below_threshold_is_catastrophic():
    result = evaluate_scientific_gates(
        _healthy_metrics(ess=0.005), target_id="D3", gate_spec=_spec()
    )
    assert result.scientific_status == STATUS_CATASTROPHIC
    codes = {r["threshold_class"] for r in result.scientific_failure_reasons}
    assert THRESHOLD_CATASTROPHIC_GUARD in codes


def test_ess_exactly_at_boundary_is_not_catastrophic():
    # Boundary semantics: the guard fires iff ess_over_n < threshold (strict).
    # A value exactly equal to the threshold passes the guard.
    result = evaluate_scientific_gates(
        _healthy_metrics(ess=0.01), target_id="D3", gate_spec=_spec()
    )
    assert result.scientific_status == STATUS_PASS
    ess_gate = next(
        g for g in result.gate_results if g["gate_id"] == "importance_ess_catastrophic"
    )
    assert ess_gate["outcome"] == "pass"
    assert ess_gate["threshold"] == pytest.approx(0.01)


def test_ess_just_below_boundary_is_catastrophic():
    result = evaluate_scientific_gates(
        _healthy_metrics(ess=0.0099), target_id="D3", gate_spec=_spec()
    )
    assert result.scientific_status == STATUS_CATASTROPHIC


def test_ess_error_dict_is_catastrophic():
    metrics = _healthy_metrics()
    metrics["importance_ess"] = {"error": "non-finite weights", "catastrophic": True}
    result = evaluate_scientific_gates(metrics, target_id="D3", gate_spec=_spec())
    assert result.scientific_status == STATUS_CATASTROPHIC


# --- D5 zero-rare catastrophe -----------------------------------------------


def test_d5_zero_rare_samples_is_catastrophic():
    metrics = _healthy_metrics(ess=0.5, d5=True, rare_count=0, rare_ratio=0.0)
    result = evaluate_scientific_gates(metrics, target_id="D5", gate_spec=_spec())
    assert result.scientific_status == STATUS_CATASTROPHIC
    reasons = [r["gate_id"] for r in result.scientific_failure_reasons]
    assert "d5_zero_rare_samples" in reasons


def test_d5_with_rare_samples_and_good_ess_passes():
    metrics = _healthy_metrics(ess=0.5, d5=True, rare_count=30, rare_ratio=0.9)
    result = evaluate_scientific_gates(metrics, target_id="D5", gate_spec=_spec())
    assert result.scientific_status == STATUS_PASS


def test_non_d5_without_rare_metrics_is_not_failed():
    # A non-D5 target that has no rare-mode metrics must not be penalized.
    metrics = _healthy_metrics(ess=0.7)
    assert "rare_mode" not in metrics
    result = evaluate_scientific_gates(metrics, target_id="D3", gate_spec=_spec())
    assert result.scientific_status == STATUS_PASS
    assert all(g["gate_id"] != "d5_zero_rare_samples" for g in result.gate_results)


def test_d5_missing_rare_metrics_is_inconclusive_when_required():
    metrics = _healthy_metrics(ess=0.7)  # no rare_mode block
    result = evaluate_scientific_gates(
        metrics, target_id="D5", gate_spec=_spec(require_d5_rare_metrics=True)
    )
    assert result.scientific_status == STATUS_INCONCLUSIVE


def test_d5_missing_rare_metrics_can_be_waived():
    metrics = _healthy_metrics(ess=0.7)
    result = evaluate_scientific_gates(
        metrics, target_id="D5", gate_spec=_spec(require_d5_rare_metrics=False)
    )
    assert result.scientific_status == STATUS_PASS


# --- regression: malformed D5 rare counts must never reach pass (Codex P2) ---
#
# _d5_zero_rare_gate used to classify observed_q_rare_sample_count with
# _classify_number and then coerce with int(), so a negative or fractional
# count was silently truncated to a "positive" integer and reported pass. This
# pins the fix: the field is validated like any other counter (non-boolean
# integer >= 0) and a malformed value is always inconclusive, never coerced.


def _d5_gate(result):
    return next(g for g in result.gate_results if g["gate_id"] == "d5_zero_rare_samples")


@pytest.mark.parametrize(
    "rare_count,expected_outcome,expected_status",
    [
        (0, "catastrophic", STATUS_CATASTROPHIC),
        (1, "pass", STATUS_PASS),
        (-1, "inconclusive", STATUS_INCONCLUSIVE),
        (1.5, "inconclusive", STATUS_INCONCLUSIVE),
        (True, "inconclusive", STATUS_INCONCLUSIVE),
        (float("nan"), "inconclusive", STATUS_INCONCLUSIVE),
        ("1", "inconclusive", STATUS_INCONCLUSIVE),
        (None, "inconclusive", STATUS_INCONCLUSIVE),  # missing, required
    ],
)
def test_d5_rare_count_validated_like_a_counter(
    rare_count, expected_outcome, expected_status
):
    metrics = _healthy_metrics(ess=0.5, d5=True, rare_count=1, rare_ratio=1.0)
    if rare_count is None:
        del metrics["rare_mode"]["observed_q_rare_sample_count"]
    else:
        metrics["rare_mode"]["observed_q_rare_sample_count"] = rare_count
    result = evaluate_scientific_gates(metrics, target_id="D5", gate_spec=_spec())

    gate = _d5_gate(result)
    assert gate["gate_id"] == "d5_zero_rare_samples"
    assert gate["threshold_class"] == THRESHOLD_CATASTROPHIC_GUARD
    assert gate["outcome"] == expected_outcome
    assert result.scientific_status == expected_status
    if expected_outcome != "pass":
        reason = next(
            r for r in result.scientific_failure_reasons
            if r["gate_id"] == "d5_zero_rare_samples"
        )
        assert reason["outcome"] == expected_outcome
        assert reason["threshold_class"] == THRESHOLD_CATASTROPHIC_GUARD


def test_d5_malformed_rare_count_never_coerced_with_int():
    # -1 and 1.5 used to pass int() coercion and read as "positive count".
    for bad_value in (-1, 1.5):
        metrics = _healthy_metrics(ess=0.5, d5=True, rare_count=1, rare_ratio=1.0)
        metrics["rare_mode"]["observed_q_rare_sample_count"] = bad_value
        result = evaluate_scientific_gates(metrics, target_id="D5", gate_spec=_spec())
        assert result.scientific_status != STATUS_PASS, (
            "malformed rare count {!r} must never reach pass".format(bad_value)
        )
        assert result.scientific_status == STATUS_INCONCLUSIVE


def test_d5_missing_rare_metrics_waiver_is_unaffected():
    # The missing-block waiver (require_d5_rare_metrics=False) is unchanged by
    # the counter-validation fix: still report-only, still passes.
    metrics = _healthy_metrics(ess=0.7)  # no rare_mode block at all
    result = evaluate_scientific_gates(
        metrics, target_id="D5", gate_spec=_spec(require_d5_rare_metrics=False)
    )
    gate = _d5_gate(result)
    assert gate["outcome"] == "report"
    assert gate["active"] is False
    assert result.scientific_status == STATUS_PASS


def test_d5_malformed_rare_count_not_waived_even_when_metric_not_required():
    # The waiver covers a MISSING rare_mode block, not a present-but-corrupt one:
    # "you needn't supply rare metrics" is not "garbage is fine when you do".
    metrics = _healthy_metrics(ess=0.5, d5=True, rare_count=1, rare_ratio=1.0)
    metrics["rare_mode"]["observed_q_rare_sample_count"] = -1
    result = evaluate_scientific_gates(
        metrics, target_id="D5", gate_spec=_spec(require_d5_rare_metrics=False)
    )
    gate = _d5_gate(result)
    assert gate["outcome"] == "inconclusive"
    assert gate["active"] is True
    assert result.scientific_status == STATUS_INCONCLUSIVE


def test_d5_partial_rare_mode_block_not_waived_when_metric_not_required():
    # Codex P2: the waiver covers a rare_mode block that is entirely ABSENT,
    # not one that is present but omits the mandatory count key.
    metrics = _healthy_metrics(ess=0.5, d5=True, rare_count=1, rare_ratio=1.0)
    del metrics["rare_mode"]["observed_q_rare_sample_count"]
    result = evaluate_scientific_gates(
        metrics, target_id="D5", gate_spec=_spec(require_d5_rare_metrics=False)
    )
    gate = _d5_gate(result)
    assert gate["outcome"] == "inconclusive"
    assert gate["active"] is True
    assert result.scientific_status == STATUS_INCONCLUSIVE


def test_evaluate_scientific_gates_validates_spec_before_d5_activation():
    # Codex P2: a hand-built gate_spec (e.g. a manually constructed RunSpec)
    # bypasses ExperimentConfig.from_dict's bool check. evaluate_scientific_gates
    # must validate the spec itself before require_d5_rare_metrics can gate D5
    # activation.
    bad_spec = ScientificGateSpec(require_d5_rare_metrics=None).resolve(_FakeEval(0.01))
    metrics = _healthy_metrics(ess=0.7)  # no rare_mode block
    with pytest.raises(GateConfigError, match="require_d5_rare_metrics must be a bool"):
        evaluate_scientific_gates(metrics, target_id="D5", gate_spec=bad_spec)


# --- non-finite density / loss ----------------------------------------------


def test_non_finite_nll_is_catastrophic():
    metrics = _healthy_metrics()
    metrics["held_out"]["held_out_nll"] = float("nan")
    result = evaluate_scientific_gates(metrics, target_id="D3", gate_spec=_spec())
    assert result.scientific_status == STATUS_CATASTROPHIC
    reasons = [r["threshold_class"] for r in result.scientific_failure_reasons]
    assert THRESHOLD_MATHEMATICAL_INVARIANT in reasons


def test_non_finite_forward_kl_is_catastrophic():
    metrics = _healthy_metrics()
    metrics["forward_kl"]["forward_kl"] = float("inf")
    result = evaluate_scientific_gates(metrics, target_id="D3", gate_spec=_spec())
    assert result.scientific_status == STATUS_CATASTROPHIC


# --- regression: non-finite *counters* must be honored (Codex P1) ------------
#
# metrics.held_out_nll / metrics.forward_kl average over the finite subset of
# rows and record the discarded rows in non_finite_count; the evaluator also
# writes a non_finite_density block. A gate reading only the aggregate scalars
# would call these runs "pass" despite a violated mathematical invariant.


def _finiteness_gate(result):
    return next(
        g for g in result.gate_results if g["gate_id"] == "density_finiteness"
    )


def _assert_finiteness_catastrophe(result, *, evidence):
    """The density_finiteness gate itself must be the catastrophic one."""

    assert result.scientific_status == STATUS_CATASTROPHIC
    gate = _finiteness_gate(result)
    assert gate["gate_id"] == "density_finiteness"
    assert gate["threshold_class"] == THRESHOLD_MATHEMATICAL_INVARIANT
    assert gate["outcome"] == "catastrophic"
    assert gate["active"] is True
    assert evidence in gate["message"]
    reason = next(
        r for r in result.scientific_failure_reasons if r["gate_id"] == "density_finiteness"
    )
    assert reason["threshold_class"] == THRESHOLD_MATHEMATICAL_INVARIANT
    assert reason["outcome"] == "catastrophic"
    assert evidence in reason["message"]


def test_held_out_non_finite_count_is_catastrophic_despite_finite_aggregates():
    # 1. Finite aggregate NLL/KL but one held-out row had non-finite density.
    metrics = _healthy_metrics(ess=0.8)
    metrics["held_out"]["non_finite_count"] = 1
    result = evaluate_scientific_gates(metrics, target_id="D3", gate_spec=_spec())
    _assert_finiteness_catastrophe(result, evidence="held_out.non_finite_count = 1")


def test_forward_kl_non_finite_count_is_catastrophic_despite_finite_aggregates():
    metrics = _healthy_metrics(ess=0.8)
    metrics["forward_kl"]["non_finite_count"] = 3
    result = evaluate_scientific_gates(metrics, target_id="D3", gate_spec=_spec())
    _assert_finiteness_catastrophe(result, evidence="forward_kl.non_finite_count = 3")


def test_non_finite_density_count_is_catastrophic_despite_finite_aggregates():
    # 2. The evaluator's dedicated non_finite_density block must be honored.
    metrics = _healthy_metrics(ess=0.8)
    metrics["non_finite_density"]["non_finite_count"] = 1
    result = evaluate_scientific_gates(metrics, target_id="D3", gate_spec=_spec())
    _assert_finiteness_catastrophe(
        result, evidence="non_finite_density.non_finite_count = 1"
    )


def test_non_finite_density_rate_is_catastrophic_despite_finite_aggregates():
    metrics = _healthy_metrics(ess=0.8)
    metrics["non_finite_density"]["non_finite_density_rate"] = 0.0005
    result = evaluate_scientific_gates(metrics, target_id="D3", gate_spec=_spec())
    _assert_finiteness_catastrophe(
        result, evidence="non_finite_density.non_finite_density_rate = 0.0005"
    )


def test_all_zero_counters_and_finite_aggregates_pass_the_finiteness_gate():
    # 3. The full evidence set present and clean is the only way to pass.
    result = evaluate_scientific_gates(
        _healthy_metrics(ess=0.8), target_id="D3", gate_spec=_spec()
    )
    gate = _finiteness_gate(result)
    assert gate["outcome"] == "pass"
    assert gate["threshold_class"] == THRESHOLD_MATHEMATICAL_INVARIANT
    assert gate["active"] is True
    assert result.scientific_status == STATUS_PASS
    # every consumed field is reported back, none of them None
    for field in (
        "held_out.held_out_nll",
        "forward_kl.forward_kl",
        "held_out.non_finite_count",
        "forward_kl.non_finite_count",
        "non_finite_density.non_finite_count",
        "non_finite_density.non_finite_density_rate",
    ):
        assert field in gate["value"]
        assert gate["value"][field] is not None


def test_catastrophic_counter_beats_unrelated_missing_field():
    # 4. Catastrophic evidence wins over missing evidence -> not inconclusive.
    metrics = _healthy_metrics(ess=0.8)
    metrics["held_out"]["non_finite_count"] = 1  # catastrophic evidence
    del metrics["non_finite_density"]  # unrelated missing evidence
    result = evaluate_scientific_gates(metrics, target_id="D3", gate_spec=_spec())
    _assert_finiteness_catastrophe(result, evidence="held_out.non_finite_count = 1")
    assert result.scientific_status != STATUS_INCONCLUSIVE


def test_catastrophic_counter_beats_missing_ess_metric():
    metrics = _healthy_metrics(ess=0.8)
    metrics["non_finite_density"]["non_finite_count"] = 2
    del metrics["importance_ess"]  # would otherwise be inconclusive
    result = evaluate_scientific_gates(metrics, target_id="D3", gate_spec=_spec())
    assert result.scientific_status == STATUS_CATASTROPHIC


@pytest.mark.parametrize(
    "block,field,value",
    [
        # 5. Malformed / impossible counters and rates can never yield pass.
        ("held_out", "non_finite_count", -1),  # negative count
        ("held_out", "non_finite_count", True),  # bool used as a number
        ("held_out", "non_finite_count", "1"),  # string
        ("held_out", "non_finite_count", 1.5),  # non-integral count
        ("forward_kl", "non_finite_count", -5),
        ("non_finite_density", "non_finite_count", -1),
        ("non_finite_density", "non_finite_density_rate", 1.5),  # rate > 1
        ("non_finite_density", "non_finite_density_rate", -0.1),  # rate < 0
        ("non_finite_density", "non_finite_density_rate", float("nan")),
        ("non_finite_density", "non_finite_density_rate", float("inf")),
        ("non_finite_density", "non_finite_density_rate", True),
        ("non_finite_density", "non_finite_density_rate", "0.0"),
    ],
)
def test_impossible_counters_never_pass(block, field, value):
    metrics = _healthy_metrics(ess=0.8)
    metrics[block][field] = value
    result = evaluate_scientific_gates(metrics, target_id="D3", gate_spec=_spec())
    gate = _finiteness_gate(result)
    assert gate["outcome"] != "pass"
    assert result.scientific_status != STATUS_PASS
    # documented invariant: impossible evidence is unusable, not proof of failure
    assert gate["outcome"] == "inconclusive"
    assert gate["threshold_class"] == THRESHOLD_MATHEMATICAL_INVARIANT
    assert result.scientific_status == STATUS_INCONCLUSIVE
    assert "impossible" in gate["message"] or "malformed" in gate["message"]


@pytest.mark.parametrize(
    "block,field",
    [
        ("held_out", "non_finite_count"),
        ("forward_kl", "non_finite_count"),
        ("non_finite_density", "non_finite_count"),
        ("non_finite_density", "non_finite_density_rate"),
    ],
)
def test_missing_finiteness_evidence_is_inconclusive(block, field):
    metrics = _healthy_metrics(ess=0.8)
    del metrics[block][field]
    result = evaluate_scientific_gates(metrics, target_id="D3", gate_spec=_spec())
    gate = _finiteness_gate(result)
    assert gate["outcome"] == "inconclusive"
    assert result.scientific_status == STATUS_INCONCLUSIVE
    assert field in gate["message"]


def test_zero_counters_expressed_as_numpy_integers_still_pass():
    # The gate uses numbers.Integral/Real, so NumPy scalars validate identically
    # without gates.py importing NumPy.
    np = pytest.importorskip("numpy")
    metrics = _healthy_metrics(ess=0.8)
    metrics["held_out"]["non_finite_count"] = np.int64(0)
    metrics["non_finite_density"]["non_finite_density_rate"] = np.float64(0.0)
    result = evaluate_scientific_gates(metrics, target_id="D3", gate_spec=_spec())
    assert _finiteness_gate(result)["outcome"] == "pass"


def test_numpy_positive_counter_is_catastrophic():
    np = pytest.importorskip("numpy")
    metrics = _healthy_metrics(ess=0.8)
    metrics["held_out"]["non_finite_count"] = np.int64(4)
    result = evaluate_scientific_gates(metrics, target_id="D3", gate_spec=_spec())
    assert result.scientific_status == STATUS_CATASTROPHIC


def test_complete_evaluator_shaped_bundle_is_supported():
    # 6. The bundle shape evaluate_run actually produces stays a clean pass.
    metrics = {
        "held_out": {"held_out_nll": 1.23, "n_test": 2000, "non_finite_count": 0},
        "forward_kl": {
            "forward_kl": 0.05,
            "forward_kl_stderr": 0.001,
            "n_effective": 2000,
            "non_finite_count": 0,
        },
        "importance_ess": {
            "ess_over_n": 0.42,
            "n_proposal": 2000,
            "max_normalized_weight": 0.01,
            "log_weight_min": -3.0,
            "log_weight_max": 2.0,
            "log_weight_range": 5.0,
            "catastrophic": False,
            "n_excluded_non_finite": 0,
        },
        "c2st": {"c2st_accuracy": 0.55, "c2st_roc_auc": 0.57, "n_per_class": 1000},
        "non_finite_density": {
            "non_finite_density_rate": 0.0,
            "non_finite_count": 0,
            "n": 2000,
        },
        "support_violation": {
            "non_finite_row_rate": 0.0,
            "pz_nonpositive_rate": 0.0,
            "pz_nonpositive_count": 0,
            "n": 2000,
        },
        "duplicates": {"exact_duplicate_count": 0, "exact_duplicate_rate": 0.0},
        "rare_mode": {
            "region_id": "rare_1e-3",
            "target_rare_mass": 1e-3,
            "q_rare_region_mass": 5e-5,
            "rare_region_mass_ratio": 0.05,
            "observed_q_rare_sample_count": 1,
            "zero_rare_samples_flag": False,
        },
    }
    result = evaluate_scientific_gates(metrics, target_id="D5", gate_spec=_spec())
    assert _finiteness_gate(result)["outcome"] == "pass"
    assert result.scientific_status == STATUS_PASS
    assert result.scientific_failure_reasons == []


# --- inconclusive: missing / malformed mandatory metrics --------------------


def test_missing_importance_ess_is_inconclusive():
    metrics = _healthy_metrics()
    del metrics["importance_ess"]
    result = evaluate_scientific_gates(metrics, target_id="D3", gate_spec=_spec())
    assert result.scientific_status == STATUS_INCONCLUSIVE


def test_missing_density_metric_is_inconclusive():
    metrics = _healthy_metrics()
    del metrics["held_out"]
    result = evaluate_scientific_gates(metrics, target_id="D3", gate_spec=_spec())
    assert result.scientific_status == STATUS_INCONCLUSIVE


def test_malformed_ess_is_inconclusive_not_catastrophic():
    metrics = _healthy_metrics()
    metrics["importance_ess"] = {"ess_over_n": "not-a-number"}
    result = evaluate_scientific_gates(metrics, target_id="D3", gate_spec=_spec())
    assert result.scientific_status == STATUS_INCONCLUSIVE


# --- precedence: catastrophic dominates inconclusive ------------------------


def test_catastrophic_dominates_missing_metric():
    metrics = _healthy_metrics(ess=0.001)  # catastrophic ESS
    del metrics["forward_kl"]  # also an inconclusive-inducing gap
    result = evaluate_scientific_gates(metrics, target_id="D3", gate_spec=_spec())
    assert result.scientific_status == STATUS_CATASTROPHIC


# --- threshold-class visibility ---------------------------------------------


def test_all_four_threshold_classes_present_for_d5():
    metrics = _healthy_metrics(ess=0.5, d5=True, rare_count=10, rare_ratio=0.5)
    result = evaluate_scientific_gates(metrics, target_id="D5", gate_spec=_spec())
    classes = {g["threshold_class"] for g in result.gate_results}
    assert THRESHOLD_MATHEMATICAL_INVARIANT in classes
    assert THRESHOLD_CATASTROPHIC_GUARD in classes
    assert THRESHOLD_PROVISIONAL_ENGINEERING in classes
    assert THRESHOLD_PREREGISTERED_SCIENTIFIC in classes


def test_forward_kl_and_c2st_are_report_only():
    metrics = _healthy_metrics(ess=0.5)
    result = evaluate_scientific_gates(metrics, target_id="D3", gate_spec=_spec())
    for gid in ("forward_kl_reference", "c2st_reference"):
        g = next(x for x in result.gate_results if x["gate_id"] == gid)
        assert g["outcome"] == "report"
        assert g["active"] is False
        assert g["threshold_class"] == THRESHOLD_PREREGISTERED_SCIENTIFIC


def test_result_is_json_safe():
    import json

    metrics = _healthy_metrics(ess=0.5, d5=True)
    result = evaluate_scientific_gates(metrics, target_id="D5", gate_spec=_spec())
    payload = result.to_dict()
    # round-trips through JSON without a custom encoder
    assert json.loads(json.dumps(payload)) == payload


# --- decision scope: what a "pass" is allowed to claim -----------------------


def test_decision_scope_is_reported_on_every_result():
    # "pass" means only "all active gates passed" -- the scope states that
    # explicitly so the status cannot be read as scientific acceptance.
    for metrics, target in (
        (_healthy_metrics(ess=0.8), "D3"),
        (_healthy_metrics(ess=0.001), "D3"),  # catastrophic
        (_healthy_metrics(ess=0.8, d5=True), "D5"),
    ):
        result = evaluate_scientific_gates(metrics, target_id=target, gate_spec=_spec())
        assert result.decision_scope == DECISION_SCOPE
        assert result.to_dict()["decision_scope"] == "active_gates_v0"


def test_pass_with_low_rare_mass_ratio_is_scoped_not_accepted():
    # The pilot's affine_medium shape: 1 rare sample in 20k, ratio ~= 0.05. The
    # rare-mass ratio stays report-only, so this passes the active gates -- and
    # the decision scope is what stops that reading as rare-mode fidelity.
    metrics = _healthy_metrics(ess=0.5, d5=True, rare_count=1, rare_ratio=0.05)
    result = evaluate_scientific_gates(metrics, target_id="D5", gate_spec=_spec())
    assert result.scientific_status == STATUS_PASS
    assert result.decision_scope == "active_gates_v0"
    ratio_gate = next(
        g for g in result.gate_results if g["gate_id"] == "rare_region_mass_ratio"
    )
    assert ratio_gate["outcome"] == "report"
    assert ratio_gate["active"] is False


# --- config: single source of truth for the ESS threshold -------------------


def test_spec_inherits_ess_threshold_from_evaluation():
    spec = ScientificGateSpec()  # catastrophic_ess_threshold defaults to None
    resolved = spec.resolve(_FakeEval(0.02))
    assert resolved.catastrophic_ess_threshold == pytest.approx(0.02)


def test_contradictory_ess_threshold_is_rejected():
    spec = ScientificGateSpec(catastrophic_ess_threshold=0.05)
    with pytest.raises(GateConfigError):
        spec.validate(_FakeEval(0.01))


def test_matching_explicit_ess_threshold_validates():
    spec = ScientificGateSpec(catastrophic_ess_threshold=0.01)
    spec.validate(_FakeEval(0.01))  # no error


def test_gate_config_hash_changes_with_config():
    a = ScientificGateSpec().config_hash()
    b = ScientificGateSpec(require_d5_rare_metrics=False).config_hash()
    assert a != b


# --- config: require_d5_rare_metrics must be an actual bool (Codex P2) -------
#
# bool is a subclass of int in Python, so a naive isinstance(x, bool) check (or
# worse, truthiness) would silently accept 0/1 and coerce them to False/True.
# This is a scientific gate: a malformed config must never silently become
# "rare metrics not required".

_VALID_REQUIRE_D5_RARE_METRICS = [True, False]
_INVALID_REQUIRE_D5_RARE_METRICS = [None, 0, 1, 0.0, "true", "false", [], {}]


@pytest.mark.parametrize("value", _VALID_REQUIRE_D5_RARE_METRICS)
def test_require_d5_rare_metrics_accepts_valid_bool(value):
    spec = ScientificGateSpec(require_d5_rare_metrics=value)
    spec.validate(_FakeEval(0.01))  # no error


@pytest.mark.parametrize("value", _INVALID_REQUIRE_D5_RARE_METRICS)
def test_require_d5_rare_metrics_rejects_non_bool(value):
    spec = ScientificGateSpec(require_d5_rare_metrics=value)
    with pytest.raises(GateConfigError, match="require_d5_rare_metrics must be a bool"):
        spec.validate(_FakeEval(0.01))


# The True-keeps-mandatory / False-waives-missing-block semantics are already
# pinned by test_d5_missing_rare_metrics_is_inconclusive_when_required and
# test_d5_missing_rare_metrics_can_be_waived above; the hash-changes-with-value
# semantics by test_gate_config_hash_changes_with_config. Together with the
# bool-only validation above, no malformed value can reach gate activation
# (validate() runs before resolve()/evaluate_scientific_gates() ever sees the
# spec), and no malformed value can silently coerce to False: it is rejected
# outright instead.


# --- no heavy dependency imported by gates.py -------------------------------


def test_gates_module_imports_no_heavy_dependency():
    import ship_muon_bg.density_lab.gates as gates_mod

    # Importing the gate module must not drag in numpy/torch/sklearn/etc. We
    # check the module's own references rather than the whole interpreter.
    forbidden = {"numpy", "torch", "sklearn", "matplotlib", "mlflow", "scipy"}
    referenced = {
        v.__name__.split(".")[0]
        for v in vars(gates_mod).values()
        if getattr(v, "__name__", None) and hasattr(v, "__name__")
    }
    assert forbidden.isdisjoint(referenced)
    # And gates.py itself must not appear to depend on them at module load.
    assert "numpy" not in gates_mod.__dict__


def test_gates_source_has_no_heavy_imports():
    import ship_muon_bg.density_lab.gates as gates_mod

    source = open(gates_mod.__file__).read()
    for lib in ("import numpy", "import torch", "import sklearn", "import matplotlib"):
        assert lib not in source
