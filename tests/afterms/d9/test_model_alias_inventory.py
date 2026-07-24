import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
D8_REGISTRY_PATH = REPO_ROOT / "artifacts" / "afterms_d8_evaluation_v0" / "registry" / "run_registry.json"

pytestmark = pytest.mark.skipif(
    not D8_REGISTRY_PATH.exists(), reason="requires the frozen D8 evaluation registry artifact",
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "build_afterms_model_alias_inventory", REPO_ROOT / "scripts" / "build_afterms_model_alias_inventory.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


inventory_module = _load_module()


def test_diagonal_gaussian_resolves_to_gauss_diag():
    inventory = inventory_module.build_inventory()
    diag = [r for r in inventory["resolved"] if r["model_family_id"] == "GAUSS_DIAG"]
    assert len(diag) == 2  # pdg13 + pdg_minus13
    for r in diag:
        assert r["model_config_id"] == "GAUSS_DIAG_d05"


def test_full_gaussian_resolves_to_gauss_full():
    inventory = inventory_module.build_inventory()
    full = [r for r in inventory["resolved"] if r["model_family_id"] == "GAUSS_FULL"]
    assert len(full) == 2
    for r in full:
        assert r["model_config_id"] == "GAUSS_FULL_d05"


def test_gmm_resolves_with_verified_evidence_only():
    inventory = inventory_module.build_inventory()
    gmm = [r for r in inventory["resolved"] if r["model_family_id"] == "GMM"]
    assert len(gmm) == 2
    for r in gmm:
        assert r["model_config_id"] == "GMM_k04_covFULL_d05"
        assert r["architecture_parameters"]["n_components"] == 4
        assert r["architecture_parameters"]["covariance_type"] == "full"


def test_no_model_is_refit_reconstruction_status_preserved():
    inventory = inventory_module.build_inventory()
    for r in inventory["resolved"]:
        assert r["d8_reconstruction_status"] == "MISSING_HISTORICAL_CHECKPOINT"
        assert r["d8_checkpoint_path"] is None


def test_pdg13_and_pdg_minus13_map_to_distinct_tracks():
    inventory = inventory_module.build_inventory()
    tracks_by_pdg = {r["pdg_value"]: r["track_id"] for r in inventory["resolved"]}
    assert tracks_by_pdg[13] == "TRK_PDG13_UW_ID"
    assert tracks_by_pdg[-13] == "TRK_PDGM13_UW_ID"


def test_build_inventory_never_touches_frozen_d8_registry_file():
    before = hashlib.sha256(D8_REGISTRY_PATH.read_bytes()).hexdigest()
    inventory_module.build_inventory()
    after = hashlib.sha256(D8_REGISTRY_PATH.read_bytes()).hexdigest()
    assert before == after


def test_build_inventory_is_deterministic():
    first = inventory_module.build_inventory()
    second = inventory_module.build_inventory()
    assert first == second
