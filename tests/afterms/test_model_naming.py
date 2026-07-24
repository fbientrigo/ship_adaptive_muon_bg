from pathlib import Path

import pytest

from ship_muon_bg.afterms import model_naming as mn

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def registry():
    return mn.load_alias_registry()


def test_registry_loads_from_default_path(registry):
    assert registry["schema_version"] == "model_alias_registry_v0"


def test_all_six_required_tracks_resolve(registry):
    expected = {
        (13, "row_empirical_unweighted", "identity_standardized_v0"): "TRK_PDG13_UW_ID",
        (13, "row_empirical_unweighted", "cartesian_log1p_pz_v0"): "TRK_PDG13_UW_LOGPZ",
        (-13, "row_empirical_unweighted", "identity_standardized_v0"): "TRK_PDGM13_UW_ID",
        (-13, "row_empirical_unweighted", "cartesian_log1p_pz_v0"): "TRK_PDGM13_UW_LOGPZ",
        (13, "production_weighted", "identity_standardized_v0"): "TRK_PDG13_W_ID",
        (-13, "production_weighted", "identity_standardized_v0"): "TRK_PDGM13_W_ID",
    }
    for (pdg, weighting, preprocessing), expected_id in expected.items():
        result = mn.resolve_track(registry, pdg_value=pdg, weighting_policy=weighting, preprocessing_name=preprocessing)
        assert result["track_id"] == expected_id
        assert result["alias_resolution_status"] == "CURATED"


def test_track_derivation_ignores_legacy_candidate_id(registry):
    # Two calls with identical empirical fields but nothing that looks like
    # A1/B1/etc. resolve to the same track -- track_id is never derived from
    # the legacy candidate id.
    a = mn.resolve_track(registry, pdg_value=13, weighting_policy="row_empirical_unweighted", preprocessing_name="identity_standardized_v0")
    b = mn.resolve_track(registry, pdg_value=13, weighting_policy="row_empirical_unweighted", preprocessing_name="identity_standardized_v0")
    assert a["track_id"] == b["track_id"]


def test_track_resolution_is_deterministic(registry):
    calls = [
        mn.resolve_track(registry, pdg_value=13, weighting_policy="row_empirical_unweighted", preprocessing_name="cartesian_log1p_pz_v0")
        for _ in range(5)
    ]
    assert len({c["track_id"] for c in calls}) == 1


def test_unrecognized_preprocessing_falls_back_generically(registry):
    result = mn.resolve_track(registry, pdg_value=13, weighting_policy="row_empirical_unweighted", preprocessing_name="quantile_normal_v0")
    assert result["track_id"] == "TRK_PDG13_UW_QUANTILE_NORMAL_V0"
    assert result["alias_resolution_status"] == "GENERIC_FALLBACK"


def test_nf_ac_model_config_uses_actual_architecture_values(registry):
    config = mn.nf_ac_model_config({"number_of_blocks": 8, "hidden_width": 128, "hidden_depth": 2, "capacity_label": "medium"})
    assert config["model_config_id"] == "NF_AC_b08_w128_d02"
    assert config["model_family_id"] == "NF_AC"
    assert config["architecture_parameters"] == {"number_of_blocks": 8, "hidden_width": 128, "hidden_depth": 2}


def test_nf_ac_model_config_id_round_trips(registry):
    for blocks, width, depth in [(4, 64, 2), (8, 128, 2), (12, 192, 2), (16, 256, 3)]:
        config_id = mn.nf_ac_model_config_id(number_of_blocks=blocks, hidden_width=width, hidden_depth=depth)
        parsed = mn.parse_nf_ac_model_config_id(config_id)
        assert parsed == {"number_of_blocks": blocks, "hidden_width": width, "hidden_depth": depth}


def test_same_architecture_yields_same_model_config_id_across_tracks():
    a = mn.nf_ac_model_config({"number_of_blocks": 4, "hidden_width": 64, "hidden_depth": 2})
    b = mn.nf_ac_model_config({"number_of_blocks": 4, "hidden_width": 64, "hidden_depth": 2})
    assert a["model_config_id"] == b["model_config_id"] == "NF_AC_b04_w064_d02"


def test_different_architecture_yields_different_model_config_id():
    a = mn.nf_ac_model_config({"number_of_blocks": 4, "hidden_width": 64, "hidden_depth": 2})
    b = mn.nf_ac_model_config({"number_of_blocks": 8, "hidden_width": 128, "hidden_depth": 2})
    assert a["model_config_id"] != b["model_config_id"]


def test_collision_check_passes_for_same_architecture_different_tracks():
    records = [
        {"model_config_id": "NF_AC_b04_w064_d02", "architecture_parameters": {"number_of_blocks": 4, "hidden_width": 64, "hidden_depth": 2}},
        {"model_config_id": "NF_AC_b04_w064_d02", "architecture_parameters": {"number_of_blocks": 4, "hidden_width": 64, "hidden_depth": 2}},
    ]
    mn.assert_no_model_config_collisions(records)  # must not raise


def test_collision_check_rejects_two_architectures_sharing_one_id():
    records = [
        {"model_config_id": "NF_AC_b04_w064_d02", "architecture_parameters": {"number_of_blocks": 4, "hidden_width": 64, "hidden_depth": 2}},
        {"model_config_id": "NF_AC_b04_w064_d02", "architecture_parameters": {"number_of_blocks": 4, "hidden_width": 64, "hidden_depth": 3}},
    ]
    with pytest.raises(mn.ModelConfigCollisionError):
        mn.assert_no_model_config_collisions(records)


def test_diagonal_gaussian_maps_to_gauss_diag():
    config = mn.gauss_model_config(family="diagonal_gaussian", dimension=5)
    assert config["model_config_id"] == "GAUSS_DIAG_d05"
    assert config["model_family_id"] == "GAUSS_DIAG"


def test_full_gaussian_maps_to_gauss_full():
    config = mn.gauss_model_config(family="full_gaussian", dimension=5)
    assert config["model_config_id"] == "GAUSS_FULL_d05"
    assert config["model_family_id"] == "GAUSS_FULL"


def test_gmm_alias_includes_only_verified_fields():
    config = mn.gmm_model_config(n_components=4, covariance_type="full", dimension=5)
    assert config["model_config_id"] == "GMM_k04_covFULL_d05"
    assert config["architecture_parameters"] == {"n_components": 4, "covariance_type": "full", "dimension": 5}
    assert config["alias_resolution_status"] == "CURATED"


def test_gmm_unresolved_metadata_stays_explicit():
    config = mn.gmm_model_config(n_components=4, covariance_type=None, dimension=5)
    assert config["model_config_id"] == "GMM_LEGACY_CONFIG_UNRESOLVED"
    assert config["alias_resolution_status"] == "UNRESOLVED"
    assert "covariance_type" in config["unresolved_missing_fields"]


def test_gmm_never_invents_a_value_when_all_fields_missing():
    config = mn.gmm_model_config(n_components=None, covariance_type=None, dimension=None)
    assert config["model_config_id"] == "GMM_LEGACY_CONFIG_UNRESOLVED"
    assert set(config["unresolved_missing_fields"]) == {"n_components", "covariance_type", "dimension"}


def test_no_flow_matching_alias_exists(registry):
    assert "FM" not in registry["model_families"]
    assert "CNF" not in registry["model_families"]
    assert "diffusion" not in registry["model_families"]
    assert "FM" in registry.get("not_yet_registered_model_families", [])


def test_scout_variant_alias_derives_from_actual_architecture_not_scale(registry):
    # A variant whose recorded architecture does NOT match what naive
    # scale-multiplication of the base (8, 128, 2) architecture would give at
    # 0.5x (which would be (4, 64, 2)) -- the resolver must report what the
    # run actually recorded, never a value re-derived from capacity_scale.
    variant_training_config = {
        "candidate_id": "A1_capacity_medium_identity_pdg13_unweighted__arena_cap_0.50x",
        "architecture": {"number_of_blocks": 5, "hidden_width": 70, "hidden_depth": 2, "capacity_label": "arena_scale_0.50x"},
        "pdg_value": 13,
        "weighting_policy": "row_empirical_unweighted",
        "preprocessing_name": "identity_standardized_v0",
        "modeled_features": ["px", "py", "pz", "x", "y"],
    }
    record = mn.resolve_scout_variant_alias(
        registry=registry,
        variant_training_config=variant_training_config,
        base_candidate_id="A1_capacity_medium_identity_pdg13_unweighted",
        capacity_scale=0.5,
        evidence_source="test",
    )
    assert record["model_config_id"] == "NF_AC_b05_w070_d02"
    assert record["model_config_id"] != "NF_AC_b04_w064_d02"


def test_display_name_matches_recommended_form(registry):
    track = mn.resolve_track(registry, pdg_value=13, weighting_policy="row_empirical_unweighted", preprocessing_name="identity_standardized_v0")
    model_config = mn.nf_ac_model_config({"number_of_blocks": 8, "hidden_width": 128, "hidden_depth": 2})
    record = mn.build_alias_record(
        internal_candidate_id="A1_capacity_medium_identity_pdg13_unweighted",
        legacy_run_id="09_affine_capacity_smoke_pdg13",
        track=track,
        model_config=model_config,
        preprocessing_name="identity_standardized_v0",
        weighting_policy="row_empirical_unweighted",
        pdg_value=13,
        modeled_features=["px", "py", "pz", "x", "y"],
        modeled_dimension=5,
        alias_registry_version=registry["schema_version"],
        alias_evidence_source="test",
    )
    assert record["display_name"] == "NF_AC_b08_w128_d02 on TRK_PDG13_UW_ID"
    assert record["internal_candidate_id"] == "A1_capacity_medium_identity_pdg13_unweighted"
