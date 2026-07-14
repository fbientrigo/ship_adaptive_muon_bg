"""Tests for the exact-target metrics (KL, ESS, rare-mode, hygiene).

Pure NumPy for the formula tests; the C2ST test is marked ``lab``.
"""

from __future__ import annotations

import numpy as np
import pytest

from ship_muon_bg.density_lab import metrics as M


def test_forward_kl_is_zero_when_p_equals_q():
    rng = np.random.default_rng(0)
    lp = rng.standard_normal(1000)
    out = M.forward_kl(lp, lp.copy())
    assert out["forward_kl"] == pytest.approx(0.0, abs=1e-12)


def test_forward_kl_matches_manual_mean():
    p = np.array([-1.0, -2.0, -3.0])
    q = np.array([-1.5, -2.5, -2.0])
    out = M.forward_kl(p, q)
    assert out["forward_kl"] == pytest.approx(np.mean(p - q))


def test_ess_is_one_when_weights_equal():
    n = 500
    log_p = np.zeros(n)
    log_q = np.zeros(n)
    out = M.importance_ess(log_p, log_q)
    assert out["ess_over_n"] == pytest.approx(1.0)
    assert out["catastrophic"] is False
    assert out["max_normalized_weight"] == pytest.approx(1.0 / n)


def test_ess_is_shift_invariant():
    rng = np.random.default_rng(1)
    log_p = rng.standard_normal(400)
    log_q = rng.standard_normal(400)
    a = M.importance_ess(log_p, log_q)["ess_over_n"]
    b = M.importance_ess(log_p + 10.0, log_q)["ess_over_n"]
    assert a == pytest.approx(b, rel=1e-9)


def test_ess_catastrophic_flag():
    # one dominant weight -> tiny ESS
    n = 1000
    log_p = np.zeros(n)
    log_q = np.zeros(n)
    log_p[0] = 50.0  # single dominating importance weight
    out = M.importance_ess(log_p, log_q, catastrophic_threshold=0.01)
    assert out["ess_over_n"] < 0.01
    assert out["catastrophic"] is True


def test_ess_rejects_non_finite():
    log_p = np.array([0.0, np.inf, 0.0])
    log_q = np.zeros(3)
    with pytest.raises(M.MetricError):
        M.importance_ess(log_p, log_q)


def test_ess_reference_two_point_case():
    # weights [1, e] -> ESS/N = (1+e)^2 / (2*(1+e^2))
    log_p = np.array([0.0, 1.0])
    log_q = np.zeros(2)
    e = np.e
    expected = (1 + e) ** 2 / (2 * (1 + e**2))
    assert M.importance_ess(log_p, log_q)["ess_over_n"] == pytest.approx(expected)


def test_held_out_nll():
    lp = np.array([-1.0, -2.0, -3.0])
    assert M.held_out_nll(lp)["held_out_nll"] == pytest.approx(2.0)


def test_support_violation_rate_counts_nonpositive_pz():
    q = np.array([[0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0, 0.0]])
    out = M.support_violation_rate(q)
    assert out["pz_nonpositive_count"] == 1
    assert out["pz_nonpositive_rate"] == pytest.approx(0.5)


def test_non_finite_density_rate():
    q = np.array([0.0, np.nan, -np.inf, 1.0])
    out = M.non_finite_density_rate(q)
    assert out["non_finite_count"] == 2
    assert out["non_finite_density_rate"] == pytest.approx(0.5)


def test_duplicate_diagnostics_detects_exact_duplicates():
    q = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]] * 3 + [[0.0, 0.0, 0.0, 0.0, 0.0]])
    out = M.duplicate_diagnostics(q)
    assert out["exact_duplicate_count"] == 2


def test_tail_quantile_errors_zero_for_identical():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((2000, 5))
    out = M.tail_quantile_errors(
        x, x.copy(), quantiles=[0.9, 0.99], column_names=["a", "b", "c", "d", "e"]
    )
    assert out["a"]["0.9"]["abs_error"] == pytest.approx(0.0)


def test_exceedance_probability_errors():
    p = np.zeros((100, 5))
    p[:, 2] = 50.0
    p[:60, 2] = 70.0
    q = np.zeros((100, 5))
    q[:, 2] = 50.0
    q[:40, 2] = 70.0
    out = M.exceedance_probability_errors(p, q, thresholds=[60.0])
    assert out["60.0"]["target"] == pytest.approx(0.6)
    assert out["60.0"]["model"] == pytest.approx(0.4)
    assert out["60.0"]["abs_error"] == pytest.approx(0.2)


def test_rare_mode_diagnostics_flags_zero_count():
    from ship_muon_bg.benchmarks import make_controlled_target

    target = make_controlled_target("D5", variant="rare_1e-2")
    # model samples entirely in the main mode (no rare rows)
    batch = target.sample(2000, pdg_id=13, seed=1)
    main_only = batch.physical[batch.component_id != target.rare_component_id(pdg_id=13)]
    q_samples = main_only[:1000]
    target_samples = target.sample(1000, pdg_id=13, seed=2)
    p_log = target.log_prob(target_samples.physical, pdg_id=13)
    q_log = p_log + 0.1  # dummy but finite
    mask = target.region_mask(target_samples.physical, pdg_id=13, region_id="rare_tail")
    out = M.rare_mode_diagnostics(
        target,
        pdg_id=13,
        region_id="rare_tail",
        q_samples_physical=q_samples,
        q_log_prob_on_target_samples=q_log,
        p_log_prob_on_target_samples=p_log,
        target_rare_labels_mask=mask,
    )
    assert out["observed_q_rare_sample_count"] == 0
    assert out["zero_rare_samples_flag"] is True
    assert out["target_rare_mass"] == pytest.approx(1e-2)


@pytest.mark.lab
def test_c2st_near_half_for_identical_distributions():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((3000, 5))
    y = rng.standard_normal((3000, 5))
    out = M.c2st(x, y, seed=0)
    # indistinguishable samples -> accuracy near 0.5
    assert 0.4 < out["c2st_accuracy"] < 0.6


@pytest.mark.lab
def test_c2st_high_for_shifted_distributions():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((3000, 5))
    y = rng.standard_normal((3000, 5)) + 4.0
    out = M.c2st(x, y, seed=0)
    assert out["c2st_accuracy"] > 0.9
