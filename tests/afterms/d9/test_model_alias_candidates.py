import copy
import json
from pathlib import Path

import pytest

from ship_muon_bg.afterms import model_naming as mn
from ship_muon_bg.afterms.d9 import contract as d9contract
from ship_muon_bg.afterms.d9 import plan as d9plan
from ship_muon_bg.afterms.d9 import runner as d9runner
from ship_muon_bg.afterms.d9 import training_config as d9tc

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINING_CONFIG_PATH = REPO_ROOT / "configs" / "afterms" / "d9_training_v0.json"
PLAN_PATH = REPO_ROOT / "configs" / "afterms" / "d9_candidate_plan_v0.json"

EXPECTED_TRACKS = {
    "A1_capacity_medium_identity_pdg13_unweighted": "TRK_PDG13_UW_ID",
    "A2_capacity_small_cartesian_pdg13_unweighted": "TRK_PDG13_UW_LOGPZ",
    "B1_capacity_small_cartesian_pdg_minus13_unweighted": "TRK_PDGM13_UW_LOGPZ",
    "B2_capacity_small_identity_pdg_minus13_unweighted": "TRK_PDGM13_UW_ID",
    "C1_weighted_small_identity_pdg13": "TRK_PDG13_W_ID",
    "D1_weighted_small_identity_pdg_minus13": "TRK_PDGM13_W_ID",
}

EXPECTED_MODEL_CONFIGS = {
    "A1_capacity_medium_identity_pdg13_unweighted": "NF_AC_b08_w128_d02",
    "A2_capacity_small_cartesian_pdg13_unweighted": "NF_AC_b04_w064_d02",
    "B1_capacity_small_cartesian_pdg_minus13_unweighted": "NF_AC_b04_w064_d02",
    "B2_capacity_small_identity_pdg_minus13_unweighted": "NF_AC_b04_w064_d02",
    "C1_weighted_small_identity_pdg13": "NF_AC_b04_w064_d02",
    "D1_weighted_small_identity_pdg_minus13": "NF_AC_b04_w064_d02",
}


@pytest.fixture
def registry():
    return mn.load_alias_registry()


@pytest.fixture
def plan():
    return d9plan.load_plan(PLAN_PATH)


@pytest.fixture
def training_config():
    return d9tc.load_training_config(TRAINING_CONFIG_PATH)


@pytest.fixture
def plan_by_id(plan):
    return {c["candidate_id"]: c for c in plan["candidates"]}


@pytest.fixture
def training_by_id(training_config):
    return {c["candidate_id"]: c for c in training_config["candidates"]}


def _resolve(registry, plan_by_id, training_by_id, candidate_id):
    return mn.resolve_d9_training_candidate_alias(
        registry=registry,
        plan_entry=plan_by_id[candidate_id],
        training_entry=training_by_id[candidate_id],
        evidence_source="configs/afterms/d9_training_v0.json",
    )


def test_all_six_candidates_present(training_by_id):
    assert set(training_by_id.keys()) == set(EXPECTED_TRACKS.keys())


@pytest.mark.parametrize("candidate_id", list(EXPECTED_TRACKS.keys()))
def test_internal_candidate_id_is_retained(registry, plan_by_id, training_by_id, candidate_id):
    record = _resolve(registry, plan_by_id, training_by_id, candidate_id)
    assert record["internal_candidate_id"] == candidate_id


@pytest.mark.parametrize("candidate_id", list(EXPECTED_TRACKS.keys()))
def test_candidate_receives_expected_track_id(registry, plan_by_id, training_by_id, candidate_id):
    record = _resolve(registry, plan_by_id, training_by_id, candidate_id)
    assert record["track_id"] == EXPECTED_TRACKS[candidate_id]


@pytest.mark.parametrize("candidate_id", list(EXPECTED_TRACKS.keys()))
def test_candidate_receives_nf_ac_model_config(registry, plan_by_id, training_by_id, candidate_id):
    record = _resolve(registry, plan_by_id, training_by_id, candidate_id)
    assert record["model_family_id"] == "NF_AC"
    assert record["model_config_id"] == EXPECTED_MODEL_CONFIGS[candidate_id]


def test_aliases_use_actual_architecture_not_capacity_label(registry, plan_by_id, training_by_id):
    record = _resolve(registry, plan_by_id, training_by_id, "A1_capacity_medium_identity_pdg13_unweighted")
    arch = training_by_id["A1_capacity_medium_identity_pdg13_unweighted"]["architecture"]
    assert record["architecture_parameters"] == {
        "number_of_blocks": arch["number_of_blocks"],
        "hidden_width": arch["hidden_width"],
        "hidden_depth": arch["hidden_depth"],
    }


def test_same_architecture_across_tracks_shares_model_config_id(registry, plan_by_id, training_by_id):
    ids = ["A2_capacity_small_cartesian_pdg13_unweighted", "B1_capacity_small_cartesian_pdg_minus13_unweighted",
           "C1_weighted_small_identity_pdg13", "D1_weighted_small_identity_pdg_minus13"]
    configs = {_resolve(registry, plan_by_id, training_by_id, cid)["model_config_id"] for cid in ids}
    assert configs == {"NF_AC_b04_w064_d02"}


def test_different_tracks_remain_different_with_same_architecture(registry, plan_by_id, training_by_id):
    ids = ["A2_capacity_small_cartesian_pdg13_unweighted", "B1_capacity_small_cartesian_pdg_minus13_unweighted",
           "C1_weighted_small_identity_pdg13", "D1_weighted_small_identity_pdg_minus13"]
    tracks = [_resolve(registry, plan_by_id, training_by_id, cid)["track_id"] for cid in ids]
    assert len(set(tracks)) == len(tracks)


def test_alias_resolution_is_deterministic(registry, plan_by_id, training_by_id):
    first = [_resolve(registry, plan_by_id, training_by_id, cid) for cid in EXPECTED_TRACKS]
    second = [_resolve(registry, plan_by_id, training_by_id, cid) for cid in EXPECTED_TRACKS]
    assert first == second


def test_aliases_unique_where_configurations_differ(registry, plan_by_id, training_by_id):
    display_names = {_resolve(registry, plan_by_id, training_by_id, cid)["display_name"] for cid in EXPECTED_TRACKS}
    assert len(display_names) == len(EXPECTED_TRACKS)


def test_no_model_config_collisions_across_all_six(registry, plan_by_id, training_by_id):
    records = [_resolve(registry, plan_by_id, training_by_id, cid) for cid in EXPECTED_TRACKS]
    mn.assert_no_model_config_collisions(records)  # must not raise


def test_alias_resolution_does_not_mutate_training_entry(registry, plan_by_id, training_by_id):
    candidate_id = "A1_capacity_medium_identity_pdg13_unweighted"
    entry = training_by_id[candidate_id]
    before = copy.deepcopy(entry)
    _resolve(registry, plan_by_id, training_by_id, candidate_id)
    assert entry == before


def test_alias_resolution_does_not_change_semantic_training_hash_inputs(training_config, plan_by_id, training_by_id, registry):
    candidate_id = "A1_capacity_medium_identity_pdg13_unweighted"
    entry = training_by_id[candidate_id]
    identity_relevant_before = d9runner._identity_relevant_config(copy.deepcopy(entry))
    _resolve(registry, plan_by_id, training_by_id, candidate_id)
    identity_relevant_after = d9runner._identity_relevant_config(entry)
    assert identity_relevant_before == identity_relevant_after


def test_alias_resolution_does_not_change_execution_policy_hash_inputs(training_by_id, plan_by_id, registry):
    candidate_id = "A1_capacity_medium_identity_pdg13_unweighted"
    entry = training_by_id[candidate_id]
    hash_before = d9contract.execution_policy_hash(
        minimum_epochs=entry["minimum_epochs"],
        early_stopping_patience=entry["early_stopping_patience"],
        checkpoint_policy=entry["checkpoint_policy"],
    )
    _resolve(registry, plan_by_id, training_by_id, candidate_id)
    hash_after = d9contract.execution_policy_hash(
        minimum_epochs=entry["minimum_epochs"],
        early_stopping_patience=entry["early_stopping_patience"],
        checkpoint_policy=entry["checkpoint_policy"],
    )
    assert hash_before == hash_after


def test_alias_resolution_never_writes_the_training_config_file():
    before = TRAINING_CONFIG_PATH.read_bytes()
    training_config = d9tc.load_training_config(TRAINING_CONFIG_PATH)
    plan = d9plan.load_plan(PLAN_PATH)
    registry = mn.load_alias_registry()
    plan_by_id = {c["candidate_id"]: c for c in plan["candidates"]}
    training_by_id = {c["candidate_id"]: c for c in training_config["candidates"]}
    for candidate_id in EXPECTED_TRACKS:
        _resolve(registry, plan_by_id, training_by_id, candidate_id)
    after = TRAINING_CONFIG_PATH.read_bytes()
    assert before == after


def test_frozen_d7_d8_source_hashes_unchanged(plan):
    mismatches = d9plan.verify_source_hashes(plan, REPO_ROOT)
    assert mismatches == []


def test_no_flow_matching_alias_in_any_resolved_candidate(registry, plan_by_id, training_by_id):
    for candidate_id in EXPECTED_TRACKS:
        record = _resolve(registry, plan_by_id, training_by_id, candidate_id)
        assert record["model_family_id"] != "FM"
        assert not record["model_config_id"].startswith("FM_")
