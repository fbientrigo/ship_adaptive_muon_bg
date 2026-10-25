"""Eligibility gate, tilt and draw bookkeeping of the minimal loop (no ROOT)."""

import numpy as np
import pytest

from ship_muon_bg import minimal_loop as ml


def toy():
    a = np.ones((6, 8))
    a[4, :3] = 0.0  # p == 0 -> invalid
    u0 = np.array([1.0, 0.0, 1.0, 0.0, np.nan, np.nan])
    volume = np.array(["cave", "Cavern", "Magn7_MagRetR", "Magn7_MiddleMagL", "", ""])
    return a, u0, volume


def test_eligibility_never_turns_exclusions_into_rows():
    a, u0, volume = toy()
    assert ml.eligibility(a, u0, volume).tolist() == ["ELIGIBLE", "ELIGIBLE", "INELIGIBLE_IN_SHIELD_MAGNET",
                                                       "INELIGIBLE_IN_SHIELD_MAGNET", "INVALID_ROW", "UNSCORED"]


@pytest.mark.parametrize("arm", ml.ARMS)
def test_arms_put_no_mass_outside_the_gate(arm):
    a, u0, volume = toy()
    status = ml.eligibility(a, u0, volume)
    pi = ml.arm_probabilities(arm, status, u0)
    assert pi.sum() == pytest.approx(1) and (pi[status != "ELIGIBLE"] == 0).all()


def test_tilt_is_h_over_eligible_rows():
    a, u0, volume = toy()
    pi = ml.arm_probabilities("tilt_u0", ml.eligibility(a, u0, volume), u0)
    assert pi[0] / pi[1] == pytest.approx(1 / ml.DELTA ** ml.ALPHA)  # h(1)/h(0)
    with pytest.raises(ValueError):
        ml.arm_probabilities("nominal_physical", ml.eligibility(a, u0, volume), u0)


def test_draws_have_unique_seeds_and_ids_and_are_reproducible():
    a, u0, volume = toy()
    status = ml.eligibility(a, u0, volume)
    rows = ml.draw(status, u0, 25, seed=7)
    assert len({r["sim_seed"] for r in rows}) == len({r["draw_id"] for r in rows}) == 50
    assert all(1 <= r["sim_seed"] <= 899_999_999 and status[r["row_index"]] == "ELIGIBLE" for r in rows)
    assert len({r["row_index"] for r in rows}) < 25  # repeats of a state are kept, not deduplicated away
    assert rows == ml.draw(status, u0, 25, seed=7)


def test_concentration_reports_mass_and_effective_rows():
    a, u0, volume = toy()
    c = ml.concentration(ml.eligibility(a, u0, volume), u0)
    assert c["reference_uniform_eligible"] == {"mass_u0_1": 0.5, "effective_rows": pytest.approx(2.0), "rows_for_90pct_mass": 2}
    assert c["tilt_u0"]["mass_u0_1"] == pytest.approx(1 / (1 + ml.DELTA ** ml.ALPHA))
