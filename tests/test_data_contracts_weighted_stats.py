"""Unit tests for ``ship_muon_bg.data_contracts.weighted_stats`` (D9 direct
utility-tilt sampling contract). Hand-computed cases only -- no repository
fixture data is loaded here; the fixture-driven threshold fitting is
exercised separately by the sampling-table tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from ship_muon_bg.data_contracts import (
    WeightedQuantileError,
    weighted_prevalence,
    weighted_quantile,
)


# --- weighted_quantile: hand-computed cases -----------------------------------


def test_weighted_quantile_uniform_weights_matches_unweighted_median():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    weights = np.ones(4)
    # cumulative fractions: 0.25, 0.5, 0.75, 1.0 -> q=0.5 hits index 1 (value 2.0)
    assert weighted_quantile(values, weights, 0.5) == 2.0


def test_weighted_quantile_q_zero_is_always_the_minimum():
    values = np.array([5.0, 1.0, 9.0, 3.0])
    weights = np.array([0.0, 0.0, 100.0, 0.0])
    # min value (1.0) carries zero weight -- q=0 must still select it.
    assert weighted_quantile(values, weights, 0.0) == 1.0


def test_weighted_quantile_q_one_is_always_the_maximum_with_positive_weight():
    values = np.array([1.0, 2.0, 3.0])
    weights = np.array([1.0, 1.0, 0.0])
    # max value (3.0) has zero weight; the largest weighted mass sits at 2.0.
    assert weighted_quantile(values, weights, 1.0) == 2.0


def test_weighted_quantile_hand_computed_skewed_weights():
    # values sorted: 10 (w=1), 20 (w=1), 30 (w=8); total=10
    # cumulative fractions: 0.1, 0.2, 1.0
    values = np.array([30.0, 10.0, 20.0])
    weights = np.array([8.0, 1.0, 1.0])
    assert weighted_quantile(values, weights, 0.05) == 10.0
    assert weighted_quantile(values, weights, 0.15) == 20.0
    assert weighted_quantile(values, weights, 0.5) == 30.0


def test_weighted_quantile_step_function_result_is_always_an_input_value():
    rng = np.random.default_rng(0)
    values = rng.normal(size=50)
    weights = rng.uniform(0.0, 3.0, size=50)
    for q in (0.0, 0.1, 0.37, 0.5, 0.9, 1.0):
        result = weighted_quantile(values, weights, q)
        assert result in values.tolist()


def test_weighted_quantile_deterministic_tie_breaking_by_original_index():
    # Duplicate values at different original indices; order must not affect result.
    values = np.array([2.0, 2.0, 2.0])
    weights = np.array([1.0, 1.0, 1.0])
    assert weighted_quantile(values, weights, 0.5) == 2.0
    # Shuffled but same multiset -> same result.
    values2 = np.array([2.0, 2.0, 2.0])
    weights2 = np.array([1.0, 1.0, 1.0])
    assert weighted_quantile(values, weights, 0.5) == weighted_quantile(values2, weights2, 0.5)


def test_weighted_quantile_is_invariant_to_input_order():
    values = np.array([40.0, 10.0, 30.0, 20.0])
    weights = np.array([1.0, 4.0, 1.0, 2.0])
    perm = np.array([2, 0, 3, 1])
    a = weighted_quantile(values, weights, 0.6)
    b = weighted_quantile(values[perm], weights[perm], 0.6)
    assert a == b


# --- weighted_quantile: precondition errors -----------------------------------


def test_weighted_quantile_rejects_shape_mismatch():
    with pytest.raises(WeightedQuantileError):
        weighted_quantile(np.array([1.0, 2.0]), np.array([1.0]), 0.5)


def test_weighted_quantile_rejects_empty_input():
    with pytest.raises(WeightedQuantileError):
        weighted_quantile(np.array([]), np.array([]), 0.5)


def test_weighted_quantile_rejects_non_finite_values():
    with pytest.raises(WeightedQuantileError):
        weighted_quantile(np.array([1.0, np.nan]), np.array([1.0, 1.0]), 0.5)
    with pytest.raises(WeightedQuantileError):
        weighted_quantile(np.array([1.0, np.inf]), np.array([1.0, 1.0]), 0.5)


def test_weighted_quantile_rejects_negative_weights():
    with pytest.raises(WeightedQuantileError):
        weighted_quantile(np.array([1.0, 2.0]), np.array([1.0, -1.0]), 0.5)


def test_weighted_quantile_rejects_non_finite_weights():
    with pytest.raises(WeightedQuantileError):
        weighted_quantile(np.array([1.0, 2.0]), np.array([1.0, np.nan]), 0.5)
    with pytest.raises(WeightedQuantileError):
        weighted_quantile(np.array([1.0, 2.0]), np.array([1.0, np.inf]), 0.5)


def test_weighted_quantile_accepts_zero_weights_when_some_weight_is_positive():
    values = np.array([1.0, 2.0, 3.0])
    weights = np.array([0.0, 0.0, 5.0])
    assert weighted_quantile(values, weights, 0.5) == 3.0


def test_weighted_quantile_rejects_all_zero_total_weight():
    with pytest.raises(WeightedQuantileError):
        weighted_quantile(np.array([1.0, 2.0]), np.array([0.0, 0.0]), 0.5)


def test_weighted_quantile_rejects_q_outside_unit_interval():
    values = np.array([1.0, 2.0])
    weights = np.array([1.0, 1.0])
    with pytest.raises(WeightedQuantileError):
        weighted_quantile(values, weights, -0.01)
    with pytest.raises(WeightedQuantileError):
        weighted_quantile(values, weights, 1.01)


def test_weighted_quantile_rejects_non_1d_input():
    with pytest.raises(WeightedQuantileError):
        weighted_quantile(np.array([[1.0, 2.0]]), np.array([[1.0, 1.0]]), 0.5)


# --- weighted_prevalence -------------------------------------------------------


def test_weighted_prevalence_hand_computed():
    indicator = np.array([1.0, 0.0, 1.0, 0.0])
    weights = np.array([1.0, 1.0, 3.0, 5.0])
    # weighted mass of indicator==1 rows: 1 + 3 = 4; total = 10
    assert weighted_prevalence(indicator, weights) == pytest.approx(0.4)


def test_weighted_prevalence_all_zero_indicator_is_zero():
    indicator = np.zeros(5)
    weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert weighted_prevalence(indicator, weights) == 0.0


def test_weighted_prevalence_all_one_indicator_is_one():
    indicator = np.ones(5)
    weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert weighted_prevalence(indicator, weights) == 1.0


def test_weighted_prevalence_accepts_boolean_indicator():
    indicator = np.array([True, False, True])
    weights = np.array([1.0, 1.0, 1.0])
    assert weighted_prevalence(indicator, weights) == pytest.approx(2.0 / 3.0)


def test_weighted_prevalence_rejects_non_binary_indicator():
    with pytest.raises(WeightedQuantileError):
        weighted_prevalence(np.array([0.0, 0.5, 1.0]), np.array([1.0, 1.0, 1.0]))


def test_weighted_prevalence_rejects_shape_mismatch():
    with pytest.raises(WeightedQuantileError):
        weighted_prevalence(np.array([1.0, 0.0]), np.array([1.0]))


def test_weighted_prevalence_rejects_negative_or_non_finite_weights():
    with pytest.raises(WeightedQuantileError):
        weighted_prevalence(np.array([1.0, 0.0]), np.array([-1.0, 1.0]))
    with pytest.raises(WeightedQuantileError):
        weighted_prevalence(np.array([1.0, 0.0]), np.array([np.nan, 1.0]))


def test_weighted_prevalence_rejects_zero_total_weight():
    with pytest.raises(WeightedQuantileError):
        weighted_prevalence(np.array([1.0, 0.0]), np.array([0.0, 0.0]))
