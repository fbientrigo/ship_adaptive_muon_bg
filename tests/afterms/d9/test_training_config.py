from pathlib import Path

import pytest

from ship_muon_bg.afterms.d9 import training_config as tc
from ship_muon_bg.afterms.d9 import plan as d9plan

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINING_CONFIG_PATH = REPO_ROOT / "configs" / "afterms" / "d9_training_v0.json"
PLAN_PATH = REPO_ROOT / "configs" / "afterms" / "d9_candidate_plan_v0.json"


@pytest.fixture(scope="module")
def training_config():
    return tc.load_training_config(TRAINING_CONFIG_PATH)


def test_every_candidate_config_is_valid(training_config):
    for candidate in training_config["candidates"]:
        violations = tc.validate_candidate_config(candidate)
        assert violations == [], (candidate["candidate_id"], violations)


def test_training_config_covers_exactly_the_plan_candidates(training_config):
    plan = d9plan.load_plan(PLAN_PATH)
    plan_ids = {c["candidate_id"] for c in plan["candidates"]}
    config_ids = set(tc.candidates_by_id(training_config).keys())
    assert plan_ids == config_ids


def test_default_seed_set_has_three_engineering_seeds():
    assert tc.DEFAULT_SEED_SET == [20260720, 20260721, 20260722]


def test_config_hash_is_stable_and_sensitive(training_config):
    c = training_config["candidates"][0]
    h1 = tc.config_hash(c)
    h2 = tc.config_hash(dict(c))
    assert h1 == h2
    mutated = dict(c, learning_rate=c["learning_rate"] * 2)
    assert tc.config_hash(mutated) != h1


def test_seeds_differ_across_candidates_use_same_default_set(training_config):
    """Required test 25 groundwork: distinct seeds are configured per candidate,
    not silently collapsed to one."""

    for c in training_config["candidates"]:
        assert len(set(c["seed_set"])) == len(c["seed_set"]) >= 3
