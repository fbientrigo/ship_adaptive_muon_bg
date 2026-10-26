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


def test_fit_u_counts_states_not_executions():
    u = ml.fit_u({"a": 0, "b": 0, "c": 3}, {"a": False, "b": False, "c": True})
    assert u["G0_none"] == {"valid_states": 2, "positive_states": 0, "u": 0.25} and u["G3_tracker"]["u"] == 2 / 3
    assert u["G1_upstream"]["u"] == 0.5  # no evidence -> prior, never zero


def test_allocate_floor_then_proportional_capped_by_availability():
    u = {"G0_none": 0.05, "G1_upstream": 0.2, "G2_sbt": 0.3, "G3_tracker": 0.75}
    alloc = ml.allocate(u, {"G0_none": 9999, "G1_upstream": 9999, "G2_sbt": 0, "G3_tracker": 17}, 30, 5)
    assert sum(alloc.values()) == 30 and alloc["G2_sbt"] == 0 and alloc["G3_tracker"] <= 17
    assert min(alloc["G0_none"], alloc["G1_upstream"]) >= 5 and alloc["G3_tracker"] > alloc["G1_upstream"] > alloc["G0_none"]
    assert sum(ml.allocate(u, {"G0_none": 2, "G1_upstream": 1, "G2_sbt": 0, "G3_tracker": 0}, 30, 5).values()) == 3


def test_draw_strata_without_replacement_and_excluding_known_states():
    cls = np.array([0, 0, 0, 1, 1, 3, 3])
    rows = ml.draw_strata(cls, np.ones(7, bool), {"G0_none": 2, "G1_upstream": 2, "G3_tracker": 1}, exclude={5}, seed=3)
    idx = [r["row_index"] for r in rows]
    assert len(idx) == len(set(idx)) == 5 and 5 not in idx and len({r["sim_seed"] for r in rows}) == 5
    assert [r for r in rows if r["arm"] == "G3_tracker"][0] == {**rows[-1], "row_index": 6, "proposal_prob": 1.0}
