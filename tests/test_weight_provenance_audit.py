"""Structure-only checks for the weight audit. Asserts observed facts, never a meaning of ``w``."""

import importlib.util
import json
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE = os.path.join(ROOT, "data", "samples", "muonsFullMC_afterMS_sample.pkl.gz")
MANIFEST = os.path.join(ROOT, "data", "samples", "muonsFullMC_afterMS_sample_manifest.json")

spec = importlib.util.spec_from_file_location("audit_wp", os.path.join(ROOT, "scripts", "audit_weight_provenance.py"))
audit_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_mod)


def test_synthetic_classes_and_prediction_flag():
    w = np.array([768.75, 7.6875, 7.6875, 0.0, 12.5])
    arr = np.zeros((5, 8)); arr[:, 6] = 13; arr[:, 7] = w; arr[:, 2] = np.arange(1, 6)
    r = audit_mod.audit(arr, "syn")
    assert r["n_unique_w"] == 4 and r["n_zero_w"] == 1
    assert r["unpredicted_w"] == [12.5]
    assert r["sum_w"] == pytest.approx(w.sum())


@pytest.mark.skipif(not os.path.exists(SAMPLE), reason="committed sample absent")
def test_committed_sample_structure_matches_manifest():
    arr = audit_mod.loader.load_muon_pkl(SAMPLE)
    r = audit_mod.audit(arr, "sample")
    assert r["n_rows"] == json.load(open(MANIFEST))["subset_n_rows"]
    assert r["classes"] == audit_mod.audit(arr, "sample")["classes"]  # deterministic
    assert sum(c["n"] for c in r["classes"]) == r["n_rows"]
