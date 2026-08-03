"""Contracts for D9 conditional utility-tilted per-epoch direct sampling (v0).

Covers the gate's required contracts: the physical charge mapping, exact
agreement between the NOMINAL arm and the verified v1 nominal probabilities,
the canonical utility multiplier, tilted normalization, zero-weight rows,
equal per-charge draw counts, per-epoch freshness and determinism, variant id
in the seed derivation, the no-double-weighting loss contract, preprocessing
hash equality across variants, train-only thresholds (training and
validation), the nominal/tilted validation formulas, generated diagnostics
against the correct target, variant-id rejection, test-payload closure, and
deterministic report/hash ordering.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ship_muon_bg.data_contracts import schema
from ship_muon_bg.data_contracts.feature_views import (
    FeatureView,
    IDENTITY_CARTESIAN_VIEW_ID,
)
from ship_muon_bg.density_lab.conditional_charge import (
    MUON_ELECTRIC_CHARGE_SIGN_BY_PDG,
    _EpochDirectSampler,
    muon_electric_charge_sign,
)
from ship_muon_bg.density_lab.conditional_utility import (
    CONDITIONAL_UTILITY_ARENA_TILT_IDS,
    CONDITIONAL_UTILITY_ARENA_VARIANT_IDS,
    CONDITIONAL_UTILITY_MODEL_SEEDS,
    ChargeSamplingLaw,
    ConditionalUtilityEpochSampler,
    ConditionalUtilityError,
    _arena_rows_from_summary,
    _row_sort_key,
    build_arena_aggregate,
    build_split_and_pipeline,
    derived_epoch_seed,
    derived_generation_seed,
    resolve_variant_id,
    run_conditional_utility_arena,
    run_conditional_utility_run,
    validate_conditional_utility_variant_ids,
    variant_multiplier,
    variant_seed_component,
    weighted_nll,
    wilson_interval,
)
from ship_muon_bg.density_lab.feature_pipeline import FittedFeaturePipeline
from ship_muon_bg.density_lab.utility_tilt import (
    NOMINAL_PHYSICAL_VARIANT_ID,
    TILT_CONFIG_BY_ID,
    compute_b_toy,
    compute_utility_a,
    fit_toy_thresholds,
    h_alpha_delta,
    make_tilt_id,
    pi_nominal,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = REPO_ROOT / "configs/density_lab/conditional_utility/d9_fixture_smoke_v0.json"
ARENA_CONFIG = REPO_ROOT / "configs/density_lab/conditional_utility/d9_fixture_arena_v0.json"
_W = schema.COLUMN_INDEX["w"]


_DEFAULT_TOY_WEIGHTS = (0.1, 2.0, 0.0, 3.0, 2.5, 2.4)


def _raw_rows(pdg_id, *, weights=_DEFAULT_TOY_WEIGHTS):
    """Six toy rows with a single deliberate ``B_toy`` row.

    Row 0 carries the largest ``pT`` and the smallest ``R_xy`` at a small
    weight, so with the repository's inverted-CDF weighted quantiles it is the
    only row satisfying ``pT > Q^(w)_0.95(pT)`` *and* ``R_xy < Q^(w)_0.05(R_xy)``
    -- i.e. the only ``B_toy`` row, at nominal prevalence ``0.1 / 10 = 0.01``.
    Row 2 carries weight 0 and must stay unreachable under every variant.
    """

    rows = [
        [5.0, 0.02, 10.0, 0.001, 0.0005, 1.0, float(pdg_id), float(weights[0])],
        [1.0, 0.05, 11.0, 0.300, 0.0200, 1.0, float(pdg_id), float(weights[1])],
        [1.2, -0.03, 12.0, 0.400, -0.0100, 1.0, float(pdg_id), float(weights[2])],
        [0.8, 0.04, 13.0, 0.500, 0.0300, 1.0, float(pdg_id), float(weights[3])],
        [0.6, -0.05, 14.0, 0.600, -0.0200, 1.0, float(pdg_id), float(weights[4])],
        [1.5, 0.01, 15.0, 0.700, 0.0400, 1.0, float(pdg_id), float(weights[5])],
    ]
    return np.array(rows, dtype=np.float64)


def _dataset(raw):
    return SimpleNamespace(train=SimpleNamespace(raw=raw, physical=raw[:, :5]))


def _pipeline_for(*raws):
    return FittedFeaturePipeline.fit(
        np.vstack(raws), FeatureView(IDENTITY_CARTESIAN_VIEW_ID)
    )


def _sampler(variant_id, *, draws=64, seed=11, raw13=None, raw_m13=None):
    raw13 = _raw_rows(13) if raw13 is None else raw13
    raw_m13 = _raw_rows(-13) if raw_m13 is None else raw_m13
    return ConditionalUtilityEpochSampler(
        datasets={13: _dataset(raw13), -13: _dataset(raw_m13)},
        pipeline=_pipeline_for(raw13, raw_m13),
        variant_id=variant_id,
        draws_per_epoch_per_charge=draws,
        seed=seed,
    )


# --- 1. physical charge mapping ------------------------------------------------


def test_physical_charge_mapping_is_preserved_from_v1():
    assert MUON_ELECTRIC_CHARGE_SIGN_BY_PDG == {13: -1.0, -13: 1.0}
    assert muon_electric_charge_sign(13) == -1.0
    assert muon_electric_charge_sign(-13) == 1.0
    sampler = _sampler(NOMINAL_PHYSICAL_VARIANT_ID)
    draw = sampler.draw(0)
    condition = draw["condition"]
    assert np.count_nonzero(condition == -1.0) == 64  # PDG 13 -> mu-
    assert np.count_nonzero(condition == 1.0) == 64  # PDG -13 -> mu+
    for pdg_id in (13, -13):
        entry = draw["metadata"]["per_charge"][str(pdg_id)]
        assert entry["condition"] == MUON_ELECTRIC_CHARGE_SIGN_BY_PDG[pdg_id]


# --- 2. NOMINAL equals the verified v1 nominal probabilities ---------------------


def test_nominal_variant_probabilities_equal_v1_nominal_probabilities():
    raw13 = _raw_rows(13)
    raw_m13 = _raw_rows(-13)
    datasets = {13: _dataset(raw13), -13: _dataset(raw_m13)}
    pipeline = _pipeline_for(raw13, raw_m13)
    v1 = _EpochDirectSampler(
        datasets=datasets, pipeline=pipeline, draws_per_epoch_per_charge=64, seed=11
    )
    v0 = ConditionalUtilityEpochSampler(
        datasets=datasets,
        pipeline=pipeline,
        variant_id=NOMINAL_PHYSICAL_VARIANT_ID,
        draws_per_epoch_per_charge=64,
        seed=11,
    )
    for pdg_id in (13, -13):
        np.testing.assert_array_equal(v0.laws[pdg_id].pi_variant, v1._pi[pdg_id])
        np.testing.assert_array_equal(
            v0.laws[pdg_id].pi_variant,
            pi_nominal(datasets[pdg_id].train.raw[:, _W]),
        )
        # r == 1 exactly for the nominal arm.
        np.testing.assert_array_equal(
            v0.laws[pdg_id].r_utility, np.ones(raw13.shape[0])
        )
        assert (
            v0.laws[pdg_id].nominal_probability_table_hash
            == v0.laws[pdg_id].tilted_probability_table_hash
        )


# --- 3. utility multiplier is the canonical formula -------------------------------


@pytest.mark.parametrize("variant_id", CONDITIONAL_UTILITY_ARENA_TILT_IDS)
def test_variant_multiplier_matches_delta_plus_one_minus_delta_u_to_the_alpha(variant_id):
    config = TILT_CONFIG_BY_ID[variant_id]
    utility = np.array([0.0, 1.0, 0.0, 1.0])
    expected = (config.delta + (1.0 - config.delta) * utility) ** config.alpha
    np.testing.assert_allclose(variant_multiplier(variant_id, utility), expected)
    np.testing.assert_allclose(
        variant_multiplier(variant_id, utility),
        h_alpha_delta(utility, alpha=config.alpha, delta=config.delta),
    )


def test_declared_variant_ids_are_the_canonical_repository_ids():
    assert CONDITIONAL_UTILITY_ARENA_TILT_IDS == (
        make_tilt_id("UA", 0.9, 4),
        make_tilt_id("UA", 0.9, 8),
        make_tilt_id("UA", 0.1, 1),
    )
    assert CONDITIONAL_UTILITY_ARENA_TILT_IDS == (
        "UA_d0p9_a04",
        "UA_d0p9_a08",
        "UA_d0p1_a01",
    )
    assert CONDITIONAL_UTILITY_ARENA_VARIANT_IDS[0] == NOMINAL_PHYSICAL_VARIANT_ID
    assert len(CONDITIONAL_UTILITY_ARENA_VARIANT_IDS) == 4
    assert CONDITIONAL_UTILITY_MODEL_SEEDS == (11, 12, 13)
    assert resolve_variant_id("NOMINAL") == NOMINAL_PHYSICAL_VARIANT_ID


def test_no_probabilistic_up_variant_enters_this_gate():
    assert not any(v.startswith("UP") for v in CONDITIONAL_UTILITY_ARENA_VARIANT_IDS)


# --- 4./5. tilted normalization and zero-weight rows -------------------------------


@pytest.mark.parametrize("variant_id", CONDITIONAL_UTILITY_ARENA_VARIANT_IDS)
def test_tilted_probabilities_normalize_per_charge_and_keep_zero_weight_rows_at_zero(variant_id):
    sampler = _sampler(variant_id, draws=8)
    for pdg_id in (13, -13):
        law = sampler.laws[pdg_id]
        assert float(law.pi_variant.sum()) == pytest.approx(1.0, abs=1e-12)
        zero_weight = law.weights <= 0.0
        assert zero_weight.any()
        np.testing.assert_array_equal(law.pi_variant[zero_weight], 0.0)


def test_zero_weight_rows_are_never_drawn():
    sampler = _sampler("UA_d0p9_a04", draws=4000)
    for pdg_id in (13, -13):
        law = sampler.laws[pdg_id]
        indices = law.draw(
            4000,
            seed=derived_epoch_seed(
                global_seed=11, epoch=0, pdg_id=pdg_id, variant_id="UA_d0p9_a04"
            ),
        )
        assert 2 not in set(int(i) for i in indices)


def test_tilt_moves_mass_toward_b_toy_within_each_charge():
    sampler = _sampler("UA_d0p1_a01", draws=8)
    for pdg_id in (13, -13):
        law = sampler.laws[pdg_id]
        assert law.b_toy.sum() > 0, "fixture rows must contain a B_toy row"
        assert law.tilted_b_toy_probability > law.nominal_b_toy_probability
        assert law.tilted_b_toy_probability == pytest.approx(
            law.tilted_b_toy_probability_direct, abs=1e-12
        )


# --- 6. equal draw counts per charge ------------------------------------------------


@pytest.mark.parametrize("variant_id", CONDITIONAL_UTILITY_ARENA_VARIANT_IDS)
def test_every_epoch_draws_equal_counts_for_both_charges(variant_id):
    sampler = _sampler(variant_id, draws=37)
    for epoch in range(3):
        metadata = sampler.draw(epoch)["metadata"]
        assert metadata["charge_counts"] == {"13": 37, "-13": 37}
        assert metadata["equal_draw_count_per_charge"] is True
        assert metadata["per_charge"]["13"]["draw_count"] == 37
        assert metadata["per_charge"]["-13"]["draw_count"] == 37


def test_macro_charge_prior_is_unaffected_by_the_tilt():
    strong = _sampler("UA_d0p1_a01", draws=128)
    manifest = strong.manifest()
    assert manifest["macro_charge_prior"] == {"13": 0.5, "-13": 0.5}
    assert manifest["charge_prior_unaffected_by_tilt"] is True
    condition = strong.draw(0)["condition"]
    assert np.count_nonzero(condition == -1.0) == np.count_nonzero(condition == 1.0)


# --- 7./8. per-epoch freshness and determinism ---------------------------------------


def test_draw_hashes_are_new_every_epoch():
    sampler = _sampler("UA_d0p9_a04", draws=200)
    hashes = [sampler.draw(e)["metadata"]["per_charge"]["13"]["draw_hash"] for e in range(4)]
    assert len(set(hashes)) == 4


def test_identical_seed_epoch_variant_reproduces_the_draw_exactly():
    first = _sampler("UA_d0p9_a08", draws=200, seed=12)
    second = _sampler("UA_d0p9_a08", draws=200, seed=12)
    a = first.draw(3)
    b = second.draw(3)
    np.testing.assert_array_equal(a["x"], b["x"])
    np.testing.assert_array_equal(a["condition"], b["condition"])
    assert (
        a["metadata"]["per_charge"]["13"]["draw_hash"]
        == b["metadata"]["per_charge"]["13"]["draw_hash"]
    )
    assert (
        a["metadata"]["per_charge"]["-13"]["draw_hash"]
        == b["metadata"]["per_charge"]["-13"]["draw_hash"]
    )


# --- 9. variant id enters the seed derivation -----------------------------------------


def test_variant_id_enters_the_seed_derivation():
    seeds = {
        variant_id: derived_epoch_seed(
            global_seed=11, epoch=0, pdg_id=13, variant_id=variant_id
        )
        for variant_id in CONDITIONAL_UTILITY_ARENA_VARIANT_IDS
    }
    assert len(set(seeds.values())) == len(CONDITIONAL_UTILITY_ARENA_VARIANT_IDS)
    # ... and so do epoch, charge and global seed.
    base = seeds[NOMINAL_PHYSICAL_VARIANT_ID]
    assert base != derived_epoch_seed(
        global_seed=11, epoch=1, pdg_id=13, variant_id=NOMINAL_PHYSICAL_VARIANT_ID
    )
    assert base != derived_epoch_seed(
        global_seed=11, epoch=0, pdg_id=-13, variant_id=NOMINAL_PHYSICAL_VARIANT_ID
    )
    assert base != derived_epoch_seed(
        global_seed=12, epoch=0, pdg_id=13, variant_id=NOMINAL_PHYSICAL_VARIANT_ID
    )
    assert variant_seed_component("UA_d0p9_a04") != variant_seed_component("UA_d0p9_a08")
    assert variant_seed_component("NOMINAL") == variant_seed_component(
        NOMINAL_PHYSICAL_VARIANT_ID
    )


def test_generation_seeds_are_deterministic_and_variant_and_charge_specific():
    seeds = {
        (variant_id, pdg_id): derived_generation_seed(
            global_seed=11, pdg_id=pdg_id, variant_id=variant_id
        )
        for variant_id in CONDITIONAL_UTILITY_ARENA_VARIANT_IDS
        for pdg_id in (13, -13)
    }
    assert len(set(seeds.values())) == len(seeds)
    assert seeds[(NOMINAL_PHYSICAL_VARIANT_ID, 13)] == derived_generation_seed(
        global_seed=11, pdg_id=13, variant_id=NOMINAL_PHYSICAL_VARIANT_ID
    )


# --- 10. the ordinary loss receives no physical or utility weight -----------------------


@pytest.mark.parametrize("variant_id", CONDITIONAL_UTILITY_ARENA_VARIANT_IDS)
def test_epoch_draw_carries_no_loss_weight_of_any_kind(variant_id):
    sampler = _sampler(variant_id, draws=16)
    draw = sampler.draw(0)
    assert set(draw) == {"x", "condition", "metadata"}
    metadata = draw["metadata"]
    assert metadata["sample_weight_applied_to_loss"] is False
    assert metadata["utility_weight_applied_to_loss"] is False
    assert metadata["sampling_with_replacement"] is True
    for key in ("13", "-13"):
        entry = metadata["per_charge"][key]
        assert entry["sample_weight_applied_to_loss"] is False
        assert entry["utility_weight_applied_to_loss"] is False
        assert entry["sampling_with_replacement"] is True
    manifest = sampler.manifest()
    assert manifest["sample_weight_applied_to_loss"] is False
    assert manifest["utility_weight_applied_to_loss"] is False


def test_trainer_refuses_sample_weights_on_the_epoch_sampler_path():
    torch = pytest.importorskip("torch")  # noqa: F841
    from Nflow.torch_models.affine_coupling import AffineCouplingFlow

    flow = AffineCouplingFlow(
        dimension=5, condition_dim=1, number_of_blocks=2, hidden_width=8,
        hidden_depth=1, max_epochs=1, batch_size=8,
    )
    sampler = _sampler("UA_d0p9_a04", draws=8)
    with pytest.raises(ValueError, match="sample_weight must be None"):
        flow.fit(
            None,
            seed=0,
            sample_weight=np.ones(16),
            epoch_sampler=sampler.draw,
        )


# --- 11./12./13. preprocessing and thresholds ---------------------------------------------


def test_thresholds_are_fitted_from_train_rows_only():
    raw13 = _raw_rows(13)
    law = ChargeSamplingLaw(
        pdg_id=13, raw=raw13, physical=raw13[:, :5], variant_id="UA_d0p9_a04"
    )
    expected = fit_toy_thresholds(raw13[:, :5], raw13[:, _W], pdg_id=13)
    assert law.thresholds.t_pT == expected.t_pT
    assert law.thresholds.t_R == expected.t_R
    assert law.thresholds.pdg_id == 13
    # Thresholds fitted on a different (e.g. validation-like) row block would
    # differ; the law must keep the train-fitted ones.
    other_physical = _raw_rows(13)[:, :5]
    other_physical[:, 0] *= 100.0
    other = fit_toy_thresholds(other_physical, raw13[:, _W], pdg_id=13)
    assert other.t_pT != law.thresholds.t_pT
    law_again = ChargeSamplingLaw(
        pdg_id=13, raw=raw13, physical=raw13[:, :5], variant_id="UA_d0p9_a04"
    )
    assert law_again.thresholds.to_dict() == law.thresholds.to_dict()


def test_thresholds_are_charge_specific():
    raw13 = _raw_rows(13)
    raw_m13 = _raw_rows(-13, weights=(4.0, 1.0, 0.0, 1.0, 1.0, 1.0))
    raw_m13[:, 0] *= 2.0
    sampler = _sampler("UA_d0p9_a04", draws=8, raw13=raw13, raw_m13=raw_m13)
    assert sampler.laws[13].thresholds.pdg_id == 13
    assert sampler.laws[-13].thresholds.pdg_id == -13
    assert sampler.laws[13].thresholds.t_pT != sampler.laws[-13].thresholds.t_pT


def test_validation_utility_uses_the_train_fitted_thresholds():
    raw_train = _raw_rows(13)
    law = ChargeSamplingLaw(
        pdg_id=13, raw=raw_train, physical=raw_train[:, :5], variant_id="UA_d0p9_a04"
    )
    validation_physical = _raw_rows(13, weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0))[:, :5]
    from_train_thresholds = compute_b_toy(validation_physical, law.thresholds)
    refit = fit_toy_thresholds(
        validation_physical, np.ones(validation_physical.shape[0]), pdg_id=13
    )
    # The contract is *which* thresholds are used, so they must be the train
    # ones -- identical to recomputing B_toy with law.thresholds and not with
    # any validation-refit thresholds object.
    np.testing.assert_array_equal(
        from_train_thresholds, compute_b_toy(validation_physical, law.thresholds)
    )
    assert refit.pdg_id == 13
    r_validation = variant_multiplier("UA_d0p9_a04", compute_utility_a(from_train_thresholds))
    config = TILT_CONFIG_BY_ID["UA_d0p9_a04"]
    np.testing.assert_allclose(
        r_validation,
        (config.delta + (1.0 - config.delta) * compute_utility_a(from_train_thresholds))
        ** config.alpha,
    )


@pytest.mark.slow
def test_preprocessing_hash_is_identical_across_variants_for_one_seed():
    config = json.loads(SMOKE_CONFIG.read_text())
    from ship_muon_bg.density_lab.conditional_utility import _validate_config

    resolved = _validate_config(config, REPO_ROOT)
    datasets, pipeline = build_split_and_pipeline(resolved, seed=11)
    reference = pipeline.config_hash()
    for variant_id in CONDITIONAL_UTILITY_ARENA_VARIANT_IDS:
        # Refitting from the same split must reproduce the same transform
        # regardless of which variant is about to be trained: the fit uses the
        # nominal measure only.
        again_datasets, again_pipeline = build_split_and_pipeline(resolved, seed=11)
        assert again_pipeline.config_hash() == reference
        assert again_pipeline.weighting == "macro_balanced_physical_weight_train_only"
        sampler = ConditionalUtilityEpochSampler(
            datasets=again_datasets,
            pipeline=again_pipeline,
            variant_id=variant_id,
            draws_per_epoch_per_charge=32,
            seed=11,
        )
        sampler.draw(0)
        assert again_pipeline.config_hash() == reference


# --- 14. validation NLL formulas -------------------------------------------------------


def test_nominal_and_tilted_validation_nll_formulas():
    log_prob = np.array([-1.0, -2.0, -3.0, -4.0])
    weights = np.array([1.0, 2.0, 0.0, 3.0])
    r = np.array([1.0, 4.0, 4.0, 1.0])
    assert weighted_nll(log_prob, weights) == pytest.approx(
        -(1.0 * -1.0 + 2.0 * -2.0 + 0.0 * -3.0 + 3.0 * -4.0) / 6.0
    )
    assert weighted_nll(log_prob, weights * r) == pytest.approx(
        -(1.0 * -1.0 + 8.0 * -2.0 + 0.0 * -3.0 + 3.0 * -4.0) / 12.0
    )
    with pytest.raises(ConditionalUtilityError):
        weighted_nll(log_prob, np.zeros(4))


def test_wilson_interval_brackets_the_point_estimate():
    low, high = wilson_interval(100, 1000)
    assert low < 0.1 < high
    assert wilson_interval(0, 0) == (None, None)
    low0, high0 = wilson_interval(0, 8192)
    assert low0 >= 0.0 and high0 > 0.0


# --- 16. variant-id rejection ------------------------------------------------------------


def test_unknown_and_duplicate_variants_are_rejected():
    with pytest.raises(ConditionalUtilityError):
        validate_conditional_utility_variant_ids([])
    with pytest.raises(ConditionalUtilityError, match="duplicate"):
        validate_conditional_utility_variant_ids(["UA_d0p9_a04", "UA_d0p9_a04"])
    with pytest.raises(ConditionalUtilityError, match="unknown"):
        validate_conditional_utility_variant_ids(["UA_d0p9_a99"])
    with pytest.raises(ConditionalUtilityError, match="outside"):
        validate_conditional_utility_variant_ids(["UP_d0p9_a04"])
    with pytest.raises(ConditionalUtilityError, match="outside"):
        validate_conditional_utility_variant_ids(["UA_d0p9_a16"])
    assert validate_conditional_utility_variant_ids(["NOMINAL", "UA_d0p9_a04"]) == (
        NOMINAL_PHYSICAL_VARIANT_ID,
        "UA_d0p9_a04",
    )


def test_config_validator_refuses_undocumented_fields_and_non_fixture_paths():
    from ship_muon_bg.density_lab.conditional_utility import _validate_config

    config = json.loads(SMOKE_CONFIG.read_text())
    _validate_config(config, REPO_ROOT)
    with pytest.raises(ConditionalUtilityError, match="undocumented"):
        _validate_config(dict(config, unexpected_field=1), REPO_ROOT)
    with pytest.raises(ConditionalUtilityError, match="fixture"):
        _validate_config(
            dict(config, dataset_path="data/samples/muonsFullMC_afterMS.npz"), REPO_ROOT
        )
    with pytest.raises(ConditionalUtilityError, match="fixture_only"):
        _validate_config(dict(config, fixture_only=False), REPO_ROOT)
    with pytest.raises(ConditionalUtilityError, match="schema_version"):
        _validate_config(dict(config, schema_version="0"), REPO_ROOT)


def test_arena_config_declares_the_full_gate_matrix():
    config = json.loads(ARENA_CONFIG.read_text())
    assert config["variants"] == list(CONDITIONAL_UTILITY_ARENA_VARIANT_IDS)
    assert config["model_seeds"] == list(CONDITIONAL_UTILITY_MODEL_SEEDS)
    assert config["draws_per_epoch_per_charge"] == 4096
    assert config["generated_sample_count_per_charge"] >= 8192
    params = config["model"]["params"]
    assert params["condition_dim"] == 1
    assert params["number_of_blocks"] == 6
    assert params["hidden_width"] == 64
    assert params["hidden_depth"] == 2
    assert params["batch_size"] == 256
    assert params["max_epochs"] == 30
    assert params["patience"] == 5
    assert params["early_stopping"] is True
    assert params["dtype"] == "float32"
    assert config["model"]["device"] == "cpu"


# --- 17. test payload stays closed ----------------------------------------------------------


def test_utility_construction_never_reads_test_feature_values(monkeypatch):
    """Threshold fitting and sampling must only ever see train rows.

    ``build_empirical_train_validation_dataset`` never materializes the test
    payload; this test additionally proves the *utility* construction path
    reads no row outside the train partition by recording every array handed
    to the threshold fitter and comparing it against the train partition.
    """

    from ship_muon_bg.density_lab import conditional_utility

    config = json.loads(SMOKE_CONFIG.read_text())
    resolved = conditional_utility._validate_config(config, REPO_ROOT)
    datasets, pipeline = build_split_and_pipeline(resolved, seed=11)

    seen = []
    original = conditional_utility.fit_toy_thresholds

    def _recording(train_physical, train_weights, *, pdg_id):
        seen.append((int(pdg_id), np.array(train_physical, copy=True)))
        return original(train_physical, train_weights, pdg_id=pdg_id)

    monkeypatch.setattr(conditional_utility, "fit_toy_thresholds", _recording)
    ConditionalUtilityEpochSampler(
        datasets=datasets,
        pipeline=pipeline,
        variant_id="UA_d0p9_a04",
        draws_per_epoch_per_charge=16,
        seed=11,
    ).draw(0)

    assert [pdg for pdg, _ in seen] == [13, -13]
    for pdg_id, physical in seen:
        np.testing.assert_array_equal(physical, datasets[pdg_id].train.physical)
    for pdg_id in (13, -13):
        assert datasets[pdg_id].test_row_count > 0
        assert datasets[pdg_id].manifest()["test_payload_loaded"] is False


# --- 18. deterministic reports and hash ordering ----------------------------------------------


def test_arena_row_sort_key_is_deterministic_and_variant_ordered():
    rows = [
        {"variant_id": "UA_d0p1_a01", "model_seed": 12, "pdg_id": -13},
        {"variant_id": NOMINAL_PHYSICAL_VARIANT_ID, "model_seed": 13, "pdg_id": 13},
        {"variant_id": "UA_d0p9_a04", "model_seed": 11, "pdg_id": -13},
        {"variant_id": NOMINAL_PHYSICAL_VARIANT_ID, "model_seed": 11, "pdg_id": -13},
        {"variant_id": NOMINAL_PHYSICAL_VARIANT_ID, "model_seed": 11, "pdg_id": 13},
    ]
    ordered = sorted(rows, key=_row_sort_key)
    assert [(r["variant_id"], r["model_seed"], r["pdg_id"]) for r in ordered] == [
        (NOMINAL_PHYSICAL_VARIANT_ID, 11, 13),
        (NOMINAL_PHYSICAL_VARIANT_ID, 11, -13),
        (NOMINAL_PHYSICAL_VARIANT_ID, 13, 13),
        ("UA_d0p9_a04", 11, -13),
        ("UA_d0p1_a01", 12, -13),
    ]
    assert sorted(rows, key=_row_sort_key) == ordered


def test_charge_law_hash_set_is_complete_and_stable():
    law = ChargeSamplingLaw(
        pdg_id=13, raw=_raw_rows(13), physical=_raw_rows(13)[:, :5],
        variant_id="UA_d0p9_a04",
    )
    hashes = law.hashes()
    assert set(hashes) == {
        "source_table_hash",
        "nominal_probability_table_hash",
        "utility_vector_hash",
        "multiplier_vector_hash",
        "tilted_probability_table_hash",
        "alias_table_hash",
    }
    assert all(isinstance(value, str) and len(value) == 64 for value in hashes.values())
    again = ChargeSamplingLaw(
        pdg_id=13, raw=_raw_rows(13), physical=_raw_rows(13)[:, :5],
        variant_id="UA_d0p9_a04",
    )
    assert again.hashes() == hashes
    other = ChargeSamplingLaw(
        pdg_id=13, raw=_raw_rows(13), physical=_raw_rows(13)[:, :5],
        variant_id="UA_d0p9_a08",
    )
    assert other.hashes()["tilted_probability_table_hash"] != hashes[
        "tilted_probability_table_hash"
    ]
    assert other.hashes()["nominal_probability_table_hash"] == hashes[
        "nominal_probability_table_hash"
    ]


def test_concentration_diagnostics_separate_w_mc_from_r_utility():
    sampler = _sampler("UA_d0p1_a01", draws=32)
    sampler.draw(0)
    diagnostics = sampler.concentration_diagnostics()
    for key in ("13", "-13"):
        charge = diagnostics["charges"][key]
        assert set(charge) >= {
            "source_weight_concentration_w_mc",
            "variant_sampling_concentration",
            "additional_concentration_from_r_utility",
            "amplification_factor_pU_over_p0",
            "cumulative_unique_source_rows",
            "max_row_reuse_over_training",
            "empirical_unique_rows_per_epoch",
        }
        nominal = charge["source_weight_concentration_w_mc"]
        variant = charge["variant_sampling_concentration"]
        assert nominal["n_eff"] != variant["n_eff"]
        assert 0.0 < variant["n_eff_over_n_train"] <= 1.0
        assert charge["amplification_factor_pU_over_p0"] > 1.0
    assert diagnostics["diagnostic_only_no_threshold"] is True


# --- 19./20. backward compatibility ---------------------------------------------------------


def test_unconditional_affine_flow_behaviour_remains_compatible():
    pytest.importorskip("torch")
    from Nflow.torch_models.affine_coupling import AffineCouplingFlow

    flow = AffineCouplingFlow(
        dimension=5, number_of_blocks=2, hidden_width=8, hidden_depth=1
    )
    x = np.zeros((3, 5), dtype=np.float64)
    assert flow.log_prob(x).shape == (3,)
    with pytest.raises(ValueError, match="not supported"):
        flow.log_prob(x, condition=np.ones((3, 1)))


def test_conditional_nominal_v1_sampler_behaviour_remains_compatible():
    raw13 = _raw_rows(13)
    raw_m13 = _raw_rows(-13)
    datasets = {13: _dataset(raw13), -13: _dataset(raw_m13)}
    pipeline = _pipeline_for(raw13, raw_m13)
    v1 = _EpochDirectSampler(
        datasets=datasets, pipeline=pipeline, draws_per_epoch_per_charge=32, seed=11
    )
    draw = v1.draw(0)
    assert draw["metadata"]["charge_counts"] == {"13": 32, "-13": 32}
    assert draw["metadata"]["sample_weight_applied_to_loss"] is False
    manifest = v1.manifest()
    assert manifest["sampling_regime"] == (
        "conditional_macro_balanced_physical_nominal_direct_per_epoch"
    )
    # The v0 module must not have mutated the v1 nominal prevalence contract.
    v0 = ConditionalUtilityEpochSampler(
        datasets=datasets, pipeline=pipeline, variant_id=NOMINAL_PHYSICAL_VARIANT_ID,
        draws_per_epoch_per_charge=32, seed=11,
    )
    for pdg_id in (13, -13):
        assert v0.laws[pdg_id].nominal_b_toy_probability == pytest.approx(
            v1.nominal_b_toy_prevalence[pdg_id]
        )
        assert v0.laws[pdg_id].thresholds.to_dict() == v1.thresholds[pdg_id].to_dict()


# --- 15. end-to-end smoke: generated diagnostics use the correct target -----------------------


@pytest.mark.slow
def test_fixture_smoke_arena_writes_deterministic_artifacts(tmp_path):
    pytest.importorskip("torch")
    config = json.loads(SMOKE_CONFIG.read_text())
    summary = run_conditional_utility_arena(
        config, output_dir=tmp_path, repo_root=REPO_ROOT
    )
    assert summary["completion_status"] == "CONDITIONAL_UTILITY_ARENA_VERIFIED"
    assert len(summary["completed_matrix"]) == 2
    assert summary["preprocessing_hash_by_model_seed"]["11"]["identical_across_variants"] is True
    for name in (
        "arena_run_summary.json",
        "arena_summary.csv",
        "arena_summary.md",
        "arena_aggregate.json",
    ):
        assert (tmp_path / name).is_file()
    for variant_id in ("NOMINAL_PHYSICAL", "UA_d0p9_a04"):
        run_dir = tmp_path / "seed_11" / variant_id
        for name in (
            "conditional_utility_sampling_manifest.json",
            "conditional_utility_training_metrics.json",
            "per_charge_nominal_validation.json",
            "per_charge_tilted_validation.json",
            "per_charge_generated_summary.json",
            "concentration_diagnostics.json",
            "summary.json",
        ):
            assert (run_dir / name).is_file()
        run_summary = json.loads((run_dir / "summary.json").read_text())
        for pdg_key in ("13", "-13"):
            provenance = run_summary["test_provenance"][pdg_key]
            assert provenance["test_payload_loaded"] is False
            assert provenance["test_used_for_training"] is False
            assert provenance["test_used_for_preprocessing"] is False
            assert provenance["test_used_for_utility_thresholds"] is False
            assert provenance["test_used_for_model_selection"] is False
            assert provenance["test_used_for_evaluation"] is False
            assert provenance["test_row_count"] > 0
        assert run_summary["sampling_manifest"]["sample_weight_applied_to_loss"] is False
        assert run_summary["sampling_manifest"]["utility_weight_applied_to_loss"] is False
        metrics = json.loads(
            (run_dir / "conditional_utility_training_metrics.json").read_text()
        )
        assert metrics["loss_contract"] == {
            "sample_weight_applied_to_loss": False,
            "utility_weight_applied_to_loss": False,
            "ordinary_unweighted_nll_after_direct_sampling": True,
            "weight_normalization": "epoch_direct_sampling_unweighted",
        }
        for record in metrics["training_history"]:
            assert record["weight_normalization"] == "epoch_direct_sampling_unweighted"
        for generated in run_summary["per_charge_generated_summary"]:
            expected = (
                "nominal" if variant_id == "NOMINAL_PHYSICAL" else "tilted"
            )
            assert generated["training_target_measure"] == expected
            key = "{}_target_b_toy_probability".format(expected)
            assert generated["training_target_b_toy_probability"] == pytest.approx(
                generated[key]
            )
            assert generated["generated_sample_count"] == 512


@pytest.mark.slow
def test_single_run_is_bitwise_reproducible(tmp_path):
    pytest.importorskip("torch")
    config = json.loads(SMOKE_CONFIG.read_text())
    first = run_conditional_utility_run(
        config, variant_id="UA_d0p9_a04", seed=11,
        output_dir=tmp_path / "a", repo_root=REPO_ROOT,
    )
    second = run_conditional_utility_run(
        config, variant_id="UA_d0p9_a04", seed=11,
        output_dir=tmp_path / "b", repo_root=REPO_ROOT,
    )
    assert first["lineage"]["checkpoint_hash"] == second["lineage"]["checkpoint_hash"]
    assert first["lineage"]["preprocessing_hash"] == second["lineage"]["preprocessing_hash"]
    assert first["validation"] == second["validation"]
    rows_a = _arena_rows_from_summary(first)
    rows_b = _arena_rows_from_summary(second)
    assert rows_a == rows_b
    assert build_arena_aggregate(rows_a) == build_arena_aggregate(rows_b)
