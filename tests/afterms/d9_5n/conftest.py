"""Reuses the tiny synthetic fixtures from tests/afterms/d9_5/conftest.py
instead of duplicating them. Never touches the real
``data/shards/afterms_nightly_v1`` scope or any real production artifact.
"""

from __future__ import annotations

import sys
from pathlib import Path

# tests/afterms has no __init__.py (package boundary), so it's the sys.path
# entry that makes the sibling `d9_5` package (which does have __init__.py)
# importable as a top-level module.
_TESTS_AFTERMS_DIR = Path(__file__).resolve().parents[1]
if str(_TESTS_AFTERMS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_AFTERMS_DIR))

from d9_5.conftest import (  # noqa: E402,F401
    TINY_EVALUATION_POLICY,
    TINY_NF_ARCHITECTURE,
    TINY_NF_EXECUTION_POLICY,
    TINY_NF_OPTIMIZER_SETTINGS,
    tiny_data_scope_config,
    tiny_evaluation_policy,
    tiny_nf_architecture,
    tiny_nf_execution_policy,
    tiny_nf_optimizer_settings,
    tiny_scout_report,
    tiny_shard_dir,
)

import pytest

from ship_muon_bg.afterms.d9_5 import data_scope, nf_ac_adapter


@pytest.fixture
def make_nf_adapter(tiny_nf_architecture, tiny_nf_optimizer_settings, tiny_evaluation_policy):
    def _make(track_id, pdg_value, artifact_root, execution_policy):
        return nf_ac_adapter.NfAcAdapter(
            track_id=track_id, pdg_value=pdg_value, architecture=tiny_nf_architecture,
            execution_policy=execution_policy, optimizer_settings=tiny_nf_optimizer_settings,
            evaluation_policy=tiny_evaluation_policy, artifact_root=artifact_root, device="cpu",
        )
    return _make


@pytest.fixture
def tiny_train_validation(tiny_shard_dir):
    manifest = data_scope.load_shard_manifest(tiny_shard_dir)
    train_raw = data_scope.load_filtered_split(tiny_shard_dir, manifest, "train", 13)
    validation_raw = data_scope.load_filtered_split(tiny_shard_dir, manifest, "validation", 13)
    return train_raw, validation_raw
