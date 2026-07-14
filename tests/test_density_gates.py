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
    """A metric bundle that passes every active gate."""

    metrics = {
        "held_out": {"held_out_nll": 1.23},
        "forward_kl": {"forward_kl": 0.05},
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
