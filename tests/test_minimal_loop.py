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


def test_draws_csv_round_trip_keeps_numeric_types(tmp_path):
    import csv, importlib.util, pathlib
    spec = importlib.util.spec_from_file_location("rml", pathlib.Path(__file__).parents[1] / "scripts" / "run_minimal_loop.py")
    rml = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rml)
    a, u0, volume = toy()
    rows = [{**r, "status": "ELIGIBLE", "initial_volume": "cave", "p_gev": 1.5} for r in ml.draw(ml.eligibility(a, u0, volume), u0, 3, 1)]
    with (tmp_path / "d.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    assert rml.read_draws(tmp_path / "d.csv") == rows  # a string "0.0" would silently empty the U0 strata
    assert rml.upper95(0) is None and abs(rml.upper95(20) - 0.1391) < 1e-4
