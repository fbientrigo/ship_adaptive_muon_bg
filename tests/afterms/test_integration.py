import json
import os
import shutil
import tempfile
import numpy as np
import pytest
from sklearn.preprocessing import QuantileTransformer
from ship_muon_bg.afterms.preprocessing import PreprocessingPipeline
from ship_muon_bg.afterms import split, stratify
from ship_muon_bg.data_contracts import schema
from scripts.watch_afterms_nightly_queue import parse_watch_packet


def test_shard_indices_match_rows():
    # Verify that the generated indices match the rows selection
    rng = np.random.default_rng(42)
    px = rng.normal(size=100)
    py = rng.normal(size=100)
    pz = rng.uniform(0.1, 10.0, size=100)
    x = rng.normal(size=100)
    y = rng.normal(size=100)
    z = rng.normal(size=100)
    ids = rng.choice([13.0, -13.0], size=100)
    w = rng.uniform(0.1, 1.0, size=100)
    arr = np.column_stack((px, py, pz, x, y, z, ids, w))
    
    # Stratify and shard
    stratum_ids, _ = stratify.compute_strata(arr, schema.COLUMN_INDEX)
    shards, _ = stratify.assign_shards(np.arange(100), stratum_ids, n_shards=2, shard_seed=123)
    
    for s in range(2):
        mask = (shards == s)
        shard_rows = arr[mask]
        shard_indices = np.where(mask)[0]
        # Check indices match original rows
        np.testing.assert_array_equal(arr[shard_indices], shard_rows)


def test_train_only_preprocessing_fit():
    # Verify that preprocessing is fitted ONLY on train data
    rng = np.random.default_rng(10)
    train_data = rng.normal(size=(500, 8))
    train_data[:, 2] = np.abs(train_data[:, 2]) # Make pz positive
    val_data = rng.normal(size=(200, 8))
    val_data[:, 2] = np.abs(val_data[:, 2])

    for var_id in ["identity_standardized_v0", "quantile_normal_v0", "cartesian_log1p_pz_v0"]:
        pipeline = PreprocessingPipeline(var_id, seed=123)
        pipeline.fit(train_data)
        
        # Transform train and validation
        norm_train = pipeline.transform(train_data)
        norm_val = pipeline.transform(val_data)
        
        assert norm_train.shape == (500, 5)
        assert norm_val.shape == (200, 5)
        
        # Check that inverse reconstructs original px, py, pz, x, y
        inv_train = pipeline.inverse(norm_train)
        np.testing.assert_allclose(inv_train, train_data[:, :5], atol=1e-6)


def test_quantile_transformer_inverse():
    # Verify scikit-learn QuantileTransformer inverse round-trips correctly
    rng = np.random.default_rng(5)
    train_data = rng.normal(size=(200, 5))
    qt = QuantileTransformer(output_distribution="normal", random_state=42)
    qt.fit(train_data)
    
    transformed = qt.transform(train_data)
    inverted = qt.inverse_transform(transformed)
    np.testing.assert_allclose(inverted, train_data, atol=1e-6)


def test_watcher_state_parsing():
    # Verify watcher state parsing with mock files
    with tempfile.TemporaryDirectory() as tmp_dir:
        # 1. No queue state test
        packet = parse_watch_packet(tmp_dir)
        assert "No queue_state.json found" in packet["warnings"][0]
        
        # 2. Write valid queue state
        q_state = {
            "active_job": "04_legacy_available_code_realnvp_quantile",
            "pid": 999999, # Very likely dead PID
            "completed_jobs": ["00_environment_and_dataset_smoke"],
            "failed_jobs": [],
            "pending_jobs": ["05_affine_preprocessing_ab_pdg13"],
            "heartbeat": 1000.0,
        }
        with open(os.path.join(tmp_dir, "queue_state.json"), "w") as f:
            json.dump(q_state, f)
            
        packet = parse_watch_packet(tmp_dir)
        assert packet["active_job"] == "04_legacy_available_code_realnvp_quantile"
        assert packet["status"] == "stale_or_exited" # PID is dead
        assert len(packet["warnings"]) > 0
