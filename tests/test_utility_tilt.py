"""Tests for the D9 direct-sampling utility-tilt pipeline
(``ship_muon_bg.density_lab.utility_tilt``): the mathematical contract
(``pi_nominal``/``pi_tilt``, the 20-config tilt grid), weighted-threshold
fitting, the two reference tables, the O(1) Walker-alias sampler, and the
direct-sampling training path (ordinary unweighted NLL, no double
weighting).

Table/math/alias tests need only NumPy. Training tests use the
``affine_coupling`` family and are marked ``flow`` (auto-skipped without the
optional torch stack, see ``tests/conftest.py``).
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys

import numpy as np
import pytest

from ship_muon_bg.density_lab import utility_tilt as ut

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(REPO_ROOT, "data", "samples", "muonsFullMC_afterMS_sample.npz")


# --- Compatibility: import hygiene ---------------------------------------------


def test_density_lab_import_with_utility_tilt_stays_numpy_only():
    code = (
        "import sys\n"
        "import ship_muon_bg.density_lab\n"
        "banned = [m for m in ('torch', 'ROOT', 'sklearn', 'matplotlib', 'mlflow') if m in sys.modules]\n"
        "assert not banned, 'heavy modules imported: {}'.format(banned)\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [REPO_ROOT, os.path.join(REPO_ROOT, "src"), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


# --- 1. Mathematics ------------------------------------------------------------


def test_u_a_mapping_is_exactly_zero_or_one():
    b_toy = np.array([1.0, 0.0, 1.0, 0.0])
    u = ut.compute_utility_a(b_toy)
    assert np.array_equal(u, np.array([1.0, 0.0, 1.0, 0.0]))
    assert set(np.unique(u).tolist()) <= {0.0, 1.0}


def test_u_p_mapping_is_exactly_099_001():
    b_toy = np.array([1.0, 0.0, 1.0, 0.0])
    u = ut.compute_utility_p(b_toy)
    assert np.array_equal(u, np.array([0.99, 0.01, 0.99, 0.01]))


def test_all_twenty_tilt_ids_generated_exactly_once():
    ids = [c.tilt_id for c in ut.ALL_TILT_CONFIGS]
    assert len(ids) == 20
    assert len(set(ids)) == 20
    expected = {
        "{}_{}_{}".format(mode, "d0p1" if delta == 0.1 else "d0p9", "a{:02d}".format(int(alpha)))
        for mode in ("UA", "UP")
        for delta in (0.1, 0.9)
        for alpha in (1, 2, 4, 8, 16)
    }
    assert set(ids) == expected


def test_h_alpha_delta_matches_hand_calculation():
    # h(U=1, alpha=2, delta=0.5) = (0.5 + 0.5*1)**2 = 1.0
    assert ut.h_alpha_delta(np.array([1.0]), alpha=2, delta=0.5)[0] == pytest.approx(1.0)
    # h(U=0, alpha=2, delta=0.5) = (0.5 + 0.5*0)**2 = 0.25
    assert ut.h_alpha_delta(np.array([0.0]), alpha=2, delta=0.5)[0] == pytest.approx(0.25)
    # h(U=0.99, alpha=1, delta=0.9) = 0.9 + 0.1*0.99 = 0.999
    assert ut.h_alpha_delta(np.array([0.99]), alpha=1, delta=0.9)[0] == pytest.approx(0.999)
    # h(U=0, alpha=4, delta=0.1) = 0.1**4 = 0.0001
    assert ut.h_alpha_delta(np.array([0.0]), alpha=4, delta=0.1)[0] == pytest.approx(0.0001)


def test_r_utility_always_positive_across_full_grid():
    for config in ut.ALL_TILT_CONFIGS:
        assert config.r_positive() > 0.0
        assert config.r_negative() > 0.0


def test_rho_is_r_positive_over_r_negative():
    config = ut.TILT_CONFIG_BY_ID["UA_d0p1_a04"]
    assert config.rho() == pytest.approx(config.r_positive() / config.r_negative())
    # hand check: UA -> U_pos=1, U_neg=0; delta=0.1, alpha=4
    # r_pos = (0.1 + 0.9*1)**4 = 1.0; r_neg = (0.1 + 0.9*0)**4 = 0.1**4 = 1e-4
    assert config.r_positive() == pytest.approx(1.0)
    assert config.r_negative() == pytest.approx(1e-4)
    assert config.rho() == pytest.approx(1e4)


def test_direct_and_closed_form_tilted_prevalence_agree_synthetic():
    rng = np.random.default_rng(0)
    n = 5000
    weights = rng.uniform(0.1, 5.0, n)
    b_toy = (rng.uniform(size=n) < 0.07).astype(np.float64)
    p0 = ut.nominal_prevalence(b_toy, weights)
    for config in ut.ALL_TILT_CONFIGS[:6]:
        r = config.r_utility_for_b_toy(b_toy)
        pi = ut.pi_tilt(weights, r)
        direct = float(np.sum(pi * b_toy))
        closed_form = ut.theoretical_tilted_prevalence(p0, config.r_positive(), config.r_negative())
        assert direct == pytest.approx(closed_form, abs=1e-9)


def test_pi_nominal_sums_to_one():
    rng = np.random.default_rng(1)
    weights = rng.uniform(0.0, 3.0, 500)
    pi = ut.pi_nominal(weights)
    assert np.sum(pi) == pytest.approx(1.0)


def test_every_pi_tilt_sums_to_one_across_full_grid():
    rng = np.random.default_rng(2)
    weights = rng.uniform(0.0, 3.0, 300)
    b_toy = (rng.uniform(size=300) < 0.1).astype(np.float64)
    for config in ut.ALL_TILT_CONFIGS:
        r = config.r_utility_for_b_toy(b_toy)
        pi = ut.pi_tilt(weights, r)
        assert np.sum(pi) == pytest.approx(1.0)


def test_w_mc_times_r_is_not_a_normalized_probability():
    rng = np.random.default_rng(3)
    weights = rng.uniform(0.1, 3.0, 50)
    b_toy = (rng.uniform(size=50) < 0.2).astype(np.float64)
    config = ut.TILT_CONFIG_BY_ID["UA_d0p9_a02"]
    r = config.r_utility_for_b_toy(b_toy)
    w_mc_times_r = weights * r
    pi = ut.pi_tilt(weights, r)
    assert np.sum(w_mc_times_r) != pytest.approx(1.0)
    assert np.sum(pi) == pytest.approx(1.0)


def test_pi_tilt_rejects_non_positive_r_utility():
    weights = np.array([1.0, 1.0])
    with pytest.raises(ut.UtilityTiltError):
        ut.pi_tilt(weights, np.array([1.0, 0.0]))


# --- 2. Weighted thresholds (in the tilt context) ------------------------------


def test_fit_toy_thresholds_hand_computed():
    # 4 rows: px,py,pz,x,y (only px,py,x,y matter)
    physical = np.array([
        [3.0, 4.0, 0.0, 0.1, 0.1],   # pT=5
        [0.0, 0.0, 0.0, 0.1, 0.1],   # pT=0
        [1.0, 0.0, 0.0, 5.0, 0.0],   # pT=1, R_xy=5
        [0.0, 1.0, 0.0, 0.0, 0.0],   # pT=1, R_xy=0
    ])
    weights = np.array([1.0, 1.0, 1.0, 1.0])
    thresholds = ut.fit_toy_thresholds(physical, weights, pdg_id=13)
    # weighted_quantile(pT=[5,0,1,1], q=0.95) -> sorted [0,1,1,5], cum=[.25,.5,.75,1.0] -> q=0.95 selects 5
    assert thresholds.t_pT == pytest.approx(5.0)
    # weighted_quantile(R_xy=[sqrt(.02),sqrt(.02),5,0], q=0.05) -> sorted [0, .1414,.1414,5] cum=[.25,.5,.75,1] -> q=0.05 selects 0
    assert thresholds.t_R == pytest.approx(0.0)


def test_thresholds_are_isolated_per_pdg_on_real_fixture():
    from ship_muon_bg.density_lab.empirical import EmpiricalDatasetSpec, build_empirical_dataset

    spec_pos = EmpiricalDatasetSpec(dataset_path=FIXTURE, pdg_id=13, seed=11, max_rows=1000)
    spec_neg = EmpiricalDatasetSpec(dataset_path=FIXTURE, pdg_id=-13, seed=11, max_rows=1000)
    table_pos = ut.build_nominal_table(build_empirical_dataset(spec_pos))
    table_neg = ut.build_nominal_table(build_empirical_dataset(spec_neg))
    assert table_pos.thresholds.pdg_id == 13
    assert table_neg.thresholds.pdg_id == -13
    # Independently-fit tracks need not share thresholds.
    assert (table_pos.thresholds.t_pT, table_pos.thresholds.t_R) != (
        table_neg.thresholds.t_pT, table_neg.thresholds.t_R,
    )


def test_thresholds_fit_from_train_split_only_not_validation_or_test():
    from ship_muon_bg.density_lab.empirical import EmpiricalDatasetSpec, build_empirical_dataset

    dataset = build_empirical_dataset(EmpiricalDatasetSpec(dataset_path=FIXTURE, pdg_id=13, seed=11, max_rows=1000))
    thresholds_from_train = ut.fit_toy_thresholds(
        dataset.train.physical, dataset.train.raw[:, ut._W_COLUMN], pdg_id=13,
    )
    # Refitting on validation rows must generally disagree (different pool) --
    # this demonstrates fit_toy_thresholds only ever consumes what is passed
    # to it, and build_nominal_table only ever passes dataset.train.
    thresholds_from_val = ut.fit_toy_thresholds(
        dataset.validation.physical, dataset.validation.raw[:, ut._W_COLUMN], pdg_id=13,
    )
    table = ut.build_nominal_table(dataset)
    assert table.thresholds.t_pT == pytest.approx(thresholds_from_train.t_pT)
    assert table.thresholds.t_R == pytest.approx(thresholds_from_train.t_R)
    assert table.split_id == "train"
    assert table.split_hash == dataset.train.manifest()["raw_dataset_hash"]
    # Not asserting inequality with thresholds_from_val (small-sample tracks
    # can coincide by chance); the identity check above is the real contract.
    del thresholds_from_val


# --- 3. Tables -------------------------------------------------------------------


def _real_dataset(pdg_id=13, max_rows=600, seed=11):
    from ship_muon_bg.density_lab.empirical import EmpiricalDatasetSpec, build_empirical_dataset

    return build_empirical_dataset(
        EmpiricalDatasetSpec(dataset_path=FIXTURE, pdg_id=pdg_id, seed=seed, max_rows=max_rows)
    )


def test_table_a_row_refs_are_stable_and_source_hashes_preserved():
    dataset = _real_dataset()
    table = ut.build_nominal_table(dataset)
    assert np.array_equal(table.row_ref, np.arange(table.n_rows))
    assert table.source_dataset_hash == dataset.source_file_dataset_hash
    assert table.split_hash == dataset.train.manifest()["raw_dataset_hash"]


def test_table_hashes_are_deterministic_given_identical_inputs():
    dataset = _real_dataset()
    table_a1 = ut.build_nominal_table(dataset)
    table_a2 = ut.build_nominal_table(dataset)
    assert table_a1.table_hash() == table_a2.table_hash()

    configs = [ut.TILT_CONFIG_BY_ID["UA_d0p9_a01"]]
    table_b1 = ut.build_tilt_table(table_a1, configs)
    table_b2 = ut.build_tilt_table(table_a2, configs)
    assert table_b1.table_hash() == table_b2.table_hash()


def test_zero_weight_rows_have_zero_nominal_and_tilted_probability():
    weights = np.array([0.0, 1.0, 2.0, 0.0])
    b_toy = np.array([1.0, 0.0, 1.0, 1.0])
    pi_nom = ut.pi_nominal(weights)
    assert pi_nom[0] == 0.0
    assert pi_nom[3] == 0.0
    config = ut.TILT_CONFIG_BY_ID["UA_d0p9_a04"]
    r = config.r_utility_for_b_toy(b_toy)
    pi = ut.pi_tilt(weights, r)
    assert pi[0] == 0.0
    assert pi[3] == 0.0


def test_tilt_table_does_not_duplicate_kinematic_variables():
    dataset = _real_dataset()
    table_a = ut.build_nominal_table(dataset)
    table_b = ut.build_tilt_table(table_a, [ut.TILT_CONFIG_BY_ID["UA_d0p9_a01"]])
    field_names = {f.name for f in __import__("dataclasses").fields(table_b)}
    assert "pT" not in field_names
    assert "R_xy" not in field_names
    assert "row_ref" in field_names  # references Table A instead


def test_selected_tilt_materialization_builds_only_requested_configs():
    dataset = _real_dataset()
    table_a = ut.build_nominal_table(dataset)
    subset = [ut.TILT_CONFIG_BY_ID[t] for t in ("UA_d0p9_a01", "UP_d0p1_a04")]
    table_b = ut.build_tilt_table(table_a, subset)
    assert table_b.tilt_ids == ("UA_d0p9_a01", "UP_d0p1_a04")
    assert table_b.row_ref.shape[0] == 2 * table_a.n_rows


def test_nominal_and_tilt_table_save_load_round_trip(tmp_path):
    dataset = _real_dataset(max_rows=300)
    table_a = ut.build_nominal_table(dataset)
    table_a.save(tmp_path, seed=11)
    reloaded_a = ut.NominalTable.load(tmp_path)
    assert reloaded_a.table_hash() == table_a.table_hash()
    assert np.array_equal(reloaded_a.pi_nominal, table_a.pi_nominal)

    configs = [ut.TILT_CONFIG_BY_ID["UA_d0p9_a01"]]
    table_b = ut.build_tilt_table(table_a, configs)
    table_b.save(tmp_path, configs, source_dataset_hash=table_a.source_dataset_hash, split_hash=table_a.split_hash, seed=11)
    reloaded_b = ut.TiltTable.load(tmp_path)
    assert reloaded_b.table_hash() == table_b.table_hash()


def test_build_and_validate_pdg_tables_all_twenty_configs_agree():
    result = ut.build_and_validate_pdg_tables(FIXTURE, 13, seed=11, max_rows=600)
    assert result["checks"]["pi_nominal_sums_to_one"]
    assert len(result["checks"]["per_tilt"]) == 20
    for tilt_id, check in result["checks"]["per_tilt"].items():
        assert check["pi_tilt_sums_to_one"], tilt_id
        assert check["agrees_within_tolerance"], tilt_id


# --- 4. Alias sampler --------------------------------------------------------------


def test_alias_sampler_deterministic_seed_reproduces_same_draws():
    pi = np.array([0.5, 0.3, 0.2])
    sampler = ut.AliasSampler.from_probabilities(pi)
    a = sampler.draw(1000, seed=42)
    b = sampler.draw(1000, seed=42)
    assert np.array_equal(a, b)
    c = sampler.draw(1000, seed=43)
    assert not np.array_equal(a, c)


def test_alias_sampler_draws_are_within_valid_index_range():
    pi = np.array([0.1, 0.2, 0.3, 0.4])
    sampler = ut.AliasSampler.from_probabilities(pi)
    draws = sampler.draw(5000, seed=0)
    assert draws.min() >= 0
    assert draws.max() < 4


def test_alias_sampler_o1_table_structure_exists():
    pi = np.array([0.25, 0.25, 0.25, 0.25])
    sampler = ut.AliasSampler.from_probabilities(pi)
    assert sampler.probability_table.shape == (4,)
    assert sampler.alias_table.shape == (4,)
    assert sampler.n == 4
    assert isinstance(sampler.table_hash(), str) and len(sampler.table_hash()) == 64


def test_alias_sampler_rejects_probabilities_not_summing_to_one():
    with pytest.raises(ut.UtilityTiltError):
        ut.AliasSampler.from_probabilities(np.array([0.5, 0.6]))


def test_alias_sampler_empirical_frequencies_agree_with_known_categorical_law():
    rng = np.random.default_rng(7)
    raw = rng.uniform(0.1, 5.0, size=12)
    pi = raw / raw.sum()
    sampler = ut.AliasSampler.from_probabilities(pi)
    drawn = sampler.draw(300000, seed=99)
    empirical = np.bincount(drawn, minlength=12) / 300000.0
    assert np.max(np.abs(empirical - pi)) < 0.01


def test_alias_sampler_empirical_b_toy_occupancy_agrees_with_theoretical():
    result = ut.build_and_validate_pdg_tables(FIXTURE, 13, seed=11, max_rows=800, tilt_ids=["UA_d0p9_a01"])
    table_a = result["table_a"]
    config = ut.TILT_CONFIG_BY_ID["UA_d0p9_a01"]
    pi = ut.tilt_pi_vector(table_a, config)
    theoretical = ut.theoretical_tilted_prevalence(result["p0_b_toy"], config.r_positive(), config.r_negative())
    validation = ut.validate_alias_sampler_against_table(pi, n_draws=200000, seed=5)
    assert validation["sampler_table_hash"]
    sampler = ut.AliasSampler.from_probabilities(pi)
    drawn = sampler.draw(200000, seed=5)
    empirical = ut.empirical_draw_diagnostics(drawn, b_toy=table_a.B_toy)
    assert abs(empirical["empirical_b_toy_fraction"] - theoretical) < 0.01


# --- Concentration diagnostics ------------------------------------------------


def test_theoretical_concentration_diagnostics_uniform_law():
    pi = np.full(10, 0.1)
    diag = ut.theoretical_concentration_diagnostics(pi, draw_budget=100)
    assert diag["n_eff"] == pytest.approx(10.0)
    assert diag["max_row_probability"] == pytest.approx(0.1)
    assert diag["fraction_zero_probability_rows"] == 0.0
    assert diag["top_10_mass"] == pytest.approx(1.0)


def test_empirical_draw_diagnostics_counts_unique_and_max_reuse():
    drawn = np.array([0, 0, 0, 1, 2])
    diag = ut.empirical_draw_diagnostics(drawn)
    assert diag["total_draws"] == 5
    assert diag["unique_rows_drawn"] == 3
    assert diag["max_reuse_count"] == 3


# --- 5. Training ---------------------------------------------------------------


def _smoke_model_spec(max_epochs=1, batch_size=16):
    from ship_muon_bg.density_lab.config import ModelSpec

    return ModelSpec(
        name="d9_test_tiny_affine", family="affine_coupling",
        params={
            "number_of_blocks": 1, "hidden_width": 8, "hidden_depth": 1,
            "max_epochs": max_epochs, "batch_size": batch_size,
            "early_stopping": False, "checkpoint_interval": 1,
        },
    )


@pytest.mark.flow
def test_direct_sampling_training_completes_cpu_one_epoch_smoke(tmp_path):
    from ship_muon_bg.density_lab.artifacts import ArtifactStore
    from ship_muon_bg.density_lab.config import EvaluationSpec, FeatureViewSpec
    from ship_muon_bg.density_lab.empirical import EmpiricalDatasetSpec

    store = ArtifactStore("d9_test_smoke", root=tmp_path)
    dataset_spec = EmpiricalDatasetSpec(dataset_path=FIXTURE, pdg_id=13, seed=11, max_rows=150)
    record = ut.run_direct_sampling_training(
        dataset_spec, store, experiment_id="d9_test_smoke", tilt_id="UA_d0p9_a01",
        feature_view=FeatureViewSpec("identity_cartesian_v0"), model=_smoke_model_spec(),
        sampler_seed=7, device="cpu", evaluation=EvaluationSpec(ess_sample_count=20, c2st_sample_count=20),
    )
    assert record["status"] == "completed"
    assert record["metrics"]["finite_train_loss"]
    assert record["metrics"]["sample_weight_applied_to_loss"] is False
    assert record["metrics"]["nominal_validation_nll_weighted_by_w"] is not None
    assert np.isfinite(record["metrics"]["nominal_validation_nll_weighted_by_w"])
    assert np.isfinite(record["metrics"]["tilted_validation_nll_weighted_by_w_times_r"])

    run_dirs = list((tmp_path / "d9_test_smoke").iterdir())
    assert len(run_dirs) == 1
    written = {p.name for p in run_dirs[0].iterdir()}
    assert "metrics.json" in written
    assert "run_status.json" in written
    assert "checkpoint" in written
    assert "training_history.jsonl" in written


@pytest.mark.flow
def test_direct_sampling_training_resume_and_force(tmp_path):
    from ship_muon_bg.density_lab.artifacts import ArtifactStore
    from ship_muon_bg.density_lab.config import FeatureViewSpec
    from ship_muon_bg.density_lab.empirical import EmpiricalDatasetSpec

    store = ArtifactStore("d9_test_resume", root=tmp_path)
    dataset_spec = EmpiricalDatasetSpec(dataset_path=FIXTURE, pdg_id=13, seed=11, max_rows=150)
    kwargs = dict(
        store=store, experiment_id="d9_test_resume", tilt_id="UA_d0p9_a01",
        feature_view=FeatureViewSpec("identity_cartesian_v0"), model=_smoke_model_spec(), sampler_seed=7, device="cpu",
    )
    first = ut.run_direct_sampling_training(dataset_spec, **kwargs)
    assert first["status"] == "completed"
    second = ut.run_direct_sampling_training(dataset_spec, **kwargs)
    assert second["status"] == "skipped_completed"
    third = ut.run_direct_sampling_training(dataset_spec, force=True, **kwargs)
    assert third["status"] == "completed"


@pytest.mark.flow
def test_direct_sampling_never_opens_the_test_split(tmp_path, monkeypatch):
    from ship_muon_bg.density_lab.artifacts import ArtifactStore
    from ship_muon_bg.density_lab.config import FeatureViewSpec
    from ship_muon_bg.density_lab.empirical import EmpiricalDatasetSpec, EmpiricalDatasetPartition

    store = ArtifactStore("d9_test_no_test_split", root=tmp_path)
    dataset_spec = EmpiricalDatasetSpec(dataset_path=FIXTURE, pdg_id=13, seed=11, max_rows=150)

    original_physical = EmpiricalDatasetPartition.physical

    def _guarded_physical(self):
        if self.partition == "test":
            raise AssertionError("test split .physical was accessed during direct-sampling training")
        return original_physical.fget(self)

    monkeypatch.setattr(EmpiricalDatasetPartition, "physical", property(_guarded_physical))

    record = ut.run_direct_sampling_training(
        dataset_spec, store, experiment_id="d9_test_no_test_split", tilt_id="UA_d0p9_a01",
        feature_view=FeatureViewSpec("identity_cartesian_v0"), model=_smoke_model_spec(), sampler_seed=7, device="cpu",
    )
    assert record["status"] == "completed"


@pytest.mark.flow
def test_exactly_four_cloud_configurations_train_with_identical_config_except_sampling_table(tmp_path):
    from ship_muon_bg.density_lab.artifacts import ArtifactStore
    from ship_muon_bg.density_lab.config import FeatureViewSpec
    from ship_muon_bg.density_lab.empirical import EmpiricalDatasetSpec

    assert len(ut.FOUR_CLOUD_TILT_IDS) == 4
    store = ArtifactStore("d9_test_four_cloud", root=tmp_path)
    dataset_spec = EmpiricalDatasetSpec(dataset_path=FIXTURE, pdg_id=13, seed=11, max_rows=150)
    model = _smoke_model_spec()
    fv = FeatureViewSpec("identity_cartesian_v0")

    records = []
    for tilt_id in ut.FOUR_CLOUD_TILT_IDS:
        record = ut.run_direct_sampling_training(
            dataset_spec, store, experiment_id="d9_test_four_cloud", tilt_id=tilt_id,
            feature_view=fv, model=model, sampler_seed=7, device="cpu",
        )
        records.append(record)

    assert len(records) == 4
    assert {r["tilt_id"] for r in records} == set(ut.FOUR_CLOUD_TILT_IDS)
    assert all(r["status"] == "completed" for r in records)
    assert all(r["metrics"]["finite_train_loss"] for r in records)
    # Every run used the identical model/feature-view/device/seed -- only the
    # tilt-specific sampling table (and hence draws) differs.
    run_dirs = sorted((tmp_path / "d9_test_four_cloud").iterdir())
    assert len(run_dirs) == 4
    import json as _json

    configs = [
        _json.loads((d / "experiment_config.json").read_text()) for d in run_dirs
    ]
    for config in configs:
        assert config["model"] == configs[0]["model"]
        assert config["feature_view"] == configs[0]["feature_view"]
        assert config["device"] == configs[0]["device"]
        assert config["seed"] == configs[0]["seed"]
        assert config["pdg_id"] == configs[0]["pdg_id"]
        # The only intentional difference: which tilt the run's target variant names.
    variants = {c["target"]["variant"] for c in configs}
    assert variants == set(ut.FOUR_CLOUD_TILT_IDS)


# --- 6. NOMINAL_PHYSICAL direct-sampling arm (D9 arena, task section 5A) -------


def test_nominal_physical_variant_id_is_not_a_tilt_config():
    assert ut.NOMINAL_PHYSICAL_VARIANT_ID == "NOMINAL_PHYSICAL"
    assert ut.NOMINAL_PHYSICAL_VARIANT_ID not in ut.TILT_CONFIG_BY_ID


def test_alias_sampler_on_pi_nominal_directly_reproduces_pi_nominal():
    """Requirement 1: nominal direct sampling reproduces ``pi_nominal`` (math-only, no torch)."""

    dataset = _real_dataset(max_rows=800)
    table_a = ut.build_nominal_table(dataset)
    validation = ut.validate_alias_sampler_against_table(table_a.pi_nominal, n_draws=200000, seed=3)
    assert validation["within_tolerance"]
    sampler = ut.AliasSampler.from_probabilities(table_a.pi_nominal, source_hash=table_a.table_hash())
    drawn = sampler.draw(200000, seed=3)
    empirical_freq = np.bincount(drawn, minlength=table_a.pi_nominal.shape[0]) / 200000.0
    assert np.max(np.abs(empirical_freq - table_a.pi_nominal)) < 0.01


@pytest.mark.flow
def test_run_direct_sampling_training_nominal_variant_samples_pi_nominal_and_records_regime(tmp_path):
    from ship_muon_bg.density_lab.artifacts import ArtifactStore
    from ship_muon_bg.density_lab.config import FeatureViewSpec
    from ship_muon_bg.density_lab.empirical import EmpiricalDatasetSpec

    store = ArtifactStore("d9_test_nominal", root=tmp_path)
    dataset_spec = EmpiricalDatasetSpec(dataset_path=FIXTURE, pdg_id=13, seed=11, max_rows=150)
    record = ut.run_direct_sampling_training(
        dataset_spec, store, experiment_id="d9_test_nominal", tilt_id=ut.NOMINAL_PHYSICAL_VARIANT_ID,
        feature_view=FeatureViewSpec("identity_cartesian_v0"), model=_smoke_model_spec(),
        sampler_seed=7, device="cpu",
    )
    assert record["status"] == "completed"
    metrics = record["metrics"]
    # Requirement 1 (training path): not a disguised utility configuration --
    # the un-tilted law is sampled directly, so nominal and "theoretical
    # target" B_toy prevalence must agree exactly (r=1 for every row).
    assert metrics["sampling_regime"] == "direct_nominal_physical"
    assert metrics["tilt_config"]["tilt_id"] == ut.NOMINAL_PHYSICAL_VARIANT_ID
    assert metrics["tilt_config"]["mode"] is None
    assert metrics["nominal_b_toy_prevalence"] == pytest.approx(metrics["theoretical_target_b_toy_prevalence"])
    assert metrics["estimator_family"] == "unweighted_iid_direct_sample_from_pi_nominal"
    # Requirement 2: nominal training never applies sample weights to the loss.
    assert metrics["sample_weight_applied_to_loss"] is False
    assert np.isfinite(metrics["nominal_validation_nll_weighted_by_w"])
    assert np.isfinite(metrics["tilted_validation_nll_weighted_by_w_times_r"])
    # An identity tilt (r=1 everywhere): nominal and "tilted" validation NLL
    # (weighted by w*1) must be numerically identical.
    assert metrics["tilted_validation_nll_weighted_by_w_times_r"] == pytest.approx(
        metrics["nominal_validation_nll_weighted_by_w"]
    )


@pytest.mark.flow
def test_nominal_and_tilted_arms_share_preprocessing_and_budget_contracts(tmp_path):
    """Requirement 3: NOMINAL_PHYSICAL and a tilted arm use identical preprocessing/budget."""

    from ship_muon_bg.density_lab.artifacts import ArtifactStore
    from ship_muon_bg.density_lab.config import FeatureViewSpec
    from ship_muon_bg.density_lab.empirical import EmpiricalDatasetSpec

    store = ArtifactStore("d9_test_nominal_vs_tilt", root=tmp_path)
    dataset_spec = EmpiricalDatasetSpec(dataset_path=FIXTURE, pdg_id=13, seed=11, max_rows=150)
    model = _smoke_model_spec()
    fv = FeatureViewSpec("identity_cartesian_v0")

    nominal_record = ut.run_direct_sampling_training(
        dataset_spec, store, experiment_id="d9_test_nominal_vs_tilt", tilt_id=ut.NOMINAL_PHYSICAL_VARIANT_ID,
        feature_view=fv, model=model, sampler_seed=7, device="cpu",
    )
    tilt_record = ut.run_direct_sampling_training(
        dataset_spec, store, experiment_id="d9_test_nominal_vs_tilt", tilt_id="UA_d0p9_a04",
        feature_view=fv, model=model, sampler_seed=7, device="cpu",
    )
    assert nominal_record["status"] == "completed"
    assert tilt_record["status"] == "completed"

    import json as _json

    nominal_dir = tmp_path / "d9_test_nominal_vs_tilt" / nominal_record["run_id"]
    tilt_dir = tmp_path / "d9_test_nominal_vs_tilt" / tilt_record["run_id"]
    nominal_config = _json.loads((nominal_dir / "experiment_config.json").read_text())
    tilt_config = _json.loads((tilt_dir / "experiment_config.json").read_text())
    nominal_pipeline = _json.loads((nominal_dir / "feature_pipeline_manifest.json").read_text())
    tilt_pipeline = _json.loads((tilt_dir / "feature_pipeline_manifest.json").read_text())

    for key in ("model", "feature_view", "device", "seed", "pdg_id", "dataset"):
        assert nominal_config[key] == tilt_config[key], key
    assert nominal_pipeline == tilt_pipeline
    assert nominal_record["metrics"]["draw_budget"] == tilt_record["metrics"]["draw_budget"]
    assert nominal_record["metrics"]["init_seed"] == tilt_record["metrics"]["init_seed"]
    assert nominal_record["metrics"]["generated_sample_count"] == tilt_record["metrics"]["generated_sample_count"]
    # The only intentional difference: which sampling table fed the alias draw.
    assert nominal_record["metrics"]["sampler_table_hash"] != tilt_record["metrics"]["sampler_table_hash"]


# --- 7. D9 arena: variant validation and aggregate report (task section 5B/5D) -


def test_arena_variant_ids_match_declared_nine_variant_grid():
    assert len(ut.ARENA_VARIANT_IDS) == 9
    assert ut.ARENA_VARIANT_IDS[0] == ut.NOMINAL_PHYSICAL_VARIANT_ID
    expected_tilts = {
        "UA_d0p9_a04", "UA_d0p9_a08", "UA_d0p9_a16", "UA_d0p1_a01",
        "UP_d0p9_a04", "UP_d0p9_a08", "UP_d0p9_a16", "UP_d0p1_a01",
    }
    assert set(ut.ARENA_VARIANT_IDS[1:]) == expected_tilts
    assert len(set(ut.ARENA_VARIANT_IDS)) == 9  # no duplicates


def test_validate_arena_variant_ids_accepts_the_declared_grid():
    assert ut.validate_arena_variant_ids(ut.ARENA_VARIANT_IDS) == tuple(ut.ARENA_VARIANT_IDS)


def test_validate_arena_variant_ids_rejects_duplicates():
    with pytest.raises(ut.UtilityTiltError):
        ut.validate_arena_variant_ids([ut.NOMINAL_PHYSICAL_VARIANT_ID, ut.NOMINAL_PHYSICAL_VARIANT_ID])
    with pytest.raises(ut.UtilityTiltError):
        ut.validate_arena_variant_ids(["UA_d0p9_a04", "UA_d0p9_a04"])


def test_validate_arena_variant_ids_rejects_unknown_ids():
    with pytest.raises(ut.UtilityTiltError):
        ut.validate_arena_variant_ids(["NOT_A_REAL_VARIANT"])
    with pytest.raises(ut.UtilityTiltError):
        ut.validate_arena_variant_ids([])


def _fake_arena_record(variant_id, *, run_id=None, status="completed", extra_metrics=None):
    metrics = {
        "sampling_regime": "direct_nominal_physical" if variant_id == ut.NOMINAL_PHYSICAL_VARIANT_ID else "direct_utility_tilt",
        "init_seed": 11, "sampler_seed": 7, "draw_budget": 1000,
        "source_table_hash": "abc123", "sampler_table_hash": "def456" + variant_id,
        "nominal_b_toy_prevalence": 0.02, "theoretical_target_b_toy_prevalence": 0.05,
        "theoretical_concentration": {"n_eff": 900.0, "top_10_mass": 0.01, "top_100_mass": 0.05, "expected_n_unique": 950.0},
        "empirical_draw_diagnostics": {
            "unique_rows_drawn": 800, "unique_rows_fraction": 0.8, "max_reuse_count": 3,
            "empirical_b_toy_fraction": 0.049,
        },
        "final_train_nll": 1.23, "finite_train_loss": True,
        "nominal_validation_nll_weighted_by_w": 1.30, "tilted_validation_nll_weighted_by_w_times_r": 1.10,
        "generated_sample_count": 500, "generated_b_toy_occupancy": 0.04,
        "generated_log_prob_finite_fraction": 1.0,
        "generated_distribution_summary": {"available": True, "finite_fraction": 1.0},
        "sample_weight_applied_to_loss": False,
        "estimator_family": "unweighted_iid_direct_sample_from_pi_tilt",
        "scientific_scope": "pipeline_verification_only_not_model_ranking",
    }
    if extra_metrics:
        metrics.update(extra_metrics)
    return {
        "run_id": run_id or "run_{}".format(variant_id), "status": status,
        "technical_status": status, "tilt_id": variant_id,
        "metrics": metrics if status == "completed" else None,
    }


def test_arena_report_ordering_and_schema_are_deterministic(tmp_path):
    """Requirement 5: aggregate report ordering/schema are deterministic regardless of input order."""

    from ship_muon_bg.density_lab.artifacts import ArtifactStore

    store = ArtifactStore("d9_test_arena_report", root=tmp_path)
    variant_ids = list(ut.ARENA_VARIANT_IDS)
    records_in_order = [_fake_arena_record(v) for v in variant_ids]
    records_shuffled = [records_in_order[i] for i in (3, 0, 7, 1, 8, 2, 5, 4, 6)]

    report_a = ut.build_arena_report(
        records_in_order, store, out_dir=tmp_path / "report_a", pdg_id=13,
        experiment_id="d9_test_arena_report", epochs=2,
    )
    report_b = ut.build_arena_report(
        records_shuffled, store, out_dir=tmp_path / "report_b", pdg_id=13,
        experiment_id="d9_test_arena_report", epochs=2,
    )
    order_a = [row["variant_id"] for row in report_a["rows"]]
    order_b = [row["variant_id"] for row in report_b["rows"]]
    assert order_a == order_b == variant_ids  # declared ARENA_VARIANT_IDS order, not input order
    assert report_a["rows"] == report_b["rows"]

    for out_dir in ("report_a", "report_b"):
        assert (tmp_path / out_dir / "arena_summary.json").exists()
        assert (tmp_path / out_dir / "arena_summary.csv").exists()
        assert (tmp_path / out_dir / "arena_summary.md").exists()

    # No composite score, no declared winner (task section 5D).
    assert report_a["no_composite_score"] is True
    assert report_a["no_winner_declared"] is True
    for row in report_a["rows"]:
        assert "composite_score" not in row
        assert "rank" not in row
        assert "winner" not in row


def test_arena_report_keeps_technical_failure_separate_from_scientific_interpretation(tmp_path):
    """Requirement 6: a technical failure stays visible and distinct from scientific status."""

    from ship_muon_bg.density_lab.artifacts import ArtifactStore

    store = ArtifactStore("d9_test_arena_failure", root=tmp_path)
    ok_record = _fake_arena_record(ut.NOMINAL_PHYSICAL_VARIANT_ID)
    failed_record = {
        "run_id": "run_failed", "status": "failed", "technical_status": "failed",
        "tilt_id": "UA_d0p9_a04", "reason": "fit_failed",
    }
    report = ut.build_arena_report(
        [ok_record, failed_record], store, out_dir=tmp_path / "report", pdg_id=13,
        experiment_id="d9_test_arena_failure", epochs=2,
    )
    rows_by_variant = {row["variant_id"]: row for row in report["rows"]}
    assert rows_by_variant[ut.NOMINAL_PHYSICAL_VARIANT_ID]["technical_status"] == "completed"
    assert rows_by_variant["UA_d0p9_a04"]["technical_status"] == "failed"
    # A technical failure is never relabeled a scientific negative: it still
    # carries the same "not_applicable" scientific-status placeholder as a
    # completed diagnostic-only run, never something implying a bad result.
    assert rows_by_variant["UA_d0p9_a04"]["scientific_status"] == "not_applicable"
    assert rows_by_variant[ut.NOMINAL_PHYSICAL_VARIANT_ID]["scientific_status"] == "not_applicable"
    # The failed row is still present (never silently dropped) with no
    # fabricated metrics.
    assert rows_by_variant["UA_d0p9_a04"]["final_train_nll"] is None


# --- 8. Train-only thresholds are unaffected by validation/test contents ------


def test_thresholds_unchanged_by_mutated_validation_or_test_contents():
    """Requirement 7: train-only thresholds stay fixed regardless of validation/test contents."""

    dataset = _real_dataset(max_rows=600)
    table_before = ut.build_nominal_table(dataset)

    mutated_validation = dataclasses.replace(
        dataset.validation, raw=np.full_like(dataset.validation.raw, 999.0)
    )
    mutated_test = dataclasses.replace(
        dataset.test, raw=np.full_like(dataset.test.raw, -999.0)
    )
    mutated_dataset = dataclasses.replace(dataset, validation=mutated_validation, test=mutated_test)

    table_after = ut.build_nominal_table(mutated_dataset)
    assert table_after.thresholds.t_pT == pytest.approx(table_before.thresholds.t_pT)
    assert table_after.thresholds.t_R == pytest.approx(table_before.thresholds.t_R)
    assert table_after.table_hash() == table_before.table_hash()


# --- 9. Compact generated-distribution diagnostics (finite-or-unavailable) ----


def test_compact_distribution_summary_finite_data_reports_available_and_finite_stats():
    rng = np.random.default_rng(21)
    physical = rng.normal(size=(500, 5))
    result = ut.compact_distribution_summary(physical)
    assert result["available"] is True
    assert result["finite_fraction"] == pytest.approx(1.0)
    for name in ut.PHYSICAL_FEATURE_NAMES:
        stats = result["per_feature"][name]
        assert all(np.isfinite(v) for v in stats.values())
    corr = np.array(result["correlation_matrix"])
    assert np.isfinite(corr).all()
    assert corr.shape == (5, 5)


def test_compact_distribution_summary_all_non_finite_rows_marked_unavailable_not_fabricated():
    """Requirement 8: generated diagnostics are finite or explicitly marked unavailable, never fabricated."""

    physical = np.full((10, 5), np.nan)
    result = ut.compact_distribution_summary(physical)
    assert result["available"] is False
    assert result["finite_fraction"] == pytest.approx(0.0)
    assert "per_feature" not in result
    assert "correlation_matrix" not in result


def test_compact_distribution_summary_weighted_matches_hand_computation():
    physical = np.array([[1.0, 0, 0, 0, 0], [3.0, 0, 0, 0, 0], [5.0, 0, 0, 0, 0]])
    weights = np.array([1.0, 1.0, 2.0])  # weighted mean of px: (1+3+10)/4 = 3.5
    result = ut.compact_distribution_summary(physical, weights=weights)
    assert result["available"] is True
    assert result["weighted"] is True
    assert result["per_feature"]["px"]["mean"] == pytest.approx(3.5)


@pytest.mark.flow
def test_generated_distribution_diagnostics_present_and_well_typed(tmp_path):
    """Requirement 8 (training path): generated diagnostics never crash and are typed consistently."""

    from ship_muon_bg.density_lab.artifacts import ArtifactStore
    from ship_muon_bg.density_lab.config import FeatureViewSpec
    from ship_muon_bg.density_lab.empirical import EmpiricalDatasetSpec

    store = ArtifactStore("d9_test_generated_diag", root=tmp_path)
    dataset_spec = EmpiricalDatasetSpec(dataset_path=FIXTURE, pdg_id=13, seed=11, max_rows=150)
    record = ut.run_direct_sampling_training(
        dataset_spec, store, experiment_id="d9_test_generated_diag", tilt_id="UA_d0p9_a04",
        feature_view=FeatureViewSpec("identity_cartesian_v0"), model=_smoke_model_spec(),
        sampler_seed=7, device="cpu",
    )
    assert record["status"] == "completed"
    metrics = record["metrics"]
    summary = metrics["generated_distribution_summary"]
    assert isinstance(summary["available"], bool)
    if summary["available"]:
        assert set(summary["per_feature"]) == set(ut.PHYSICAL_FEATURE_NAMES)
    occupancy = metrics["generated_b_toy_occupancy"]
    assert occupancy is None or (isinstance(occupancy, float) and 0.0 <= occupancy <= 1.0)
    finite_frac = metrics["generated_log_prob_finite_fraction"]
    assert finite_frac is None or (isinstance(finite_frac, float) and 0.0 <= finite_frac <= 1.0)
    target_summary = metrics["declared_target_distribution_summary"]
    assert isinstance(target_summary["available"], bool)
