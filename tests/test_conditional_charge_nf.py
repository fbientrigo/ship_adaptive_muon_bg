"""Focused contracts for the Phase 2 conditional-charge flow."""

from types import SimpleNamespace
import json
from pathlib import Path

import numpy as np
import pytest

from Nflow.torch_models.affine_coupling import AffineCouplingFlow
from ship_muon_bg.data_contracts.hashing import dataset_hash
from ship_muon_bg.density_lab.conditional_charge import (
    ConditionalChargeError,
    build_balanced_draw_pool,
    charge_condition,
    run_symmetry_audit,
    run_fixture_pilot,
)
from ship_muon_bg.density_lab.feature_pipeline import FittedFeaturePipeline
from ship_muon_bg.data_contracts.feature_views import FeatureView, IDENTITY_CARTESIAN_VIEW_ID


def _raw_rows():
    return np.array(
        [
            [1.0, 0.2, 10.0, 0.1, 0.2, 1.0, 13.0, 1.0],
            [1.2, 0.1, 11.0, 0.2, 0.3, 1.0, 13.0, 2.0],
            [1.4, 0.0, 12.0, 0.3, 0.4, 1.0, 13.0, 0.0],
            [1.6, -0.1, 13.0, 0.4, 0.5, 1.0, 13.0, 3.0],
        ],
        dtype=np.float64,
    )


def _dataset(raw):
    return SimpleNamespace(train=SimpleNamespace(raw=raw, physical=raw[:, :5]))


def test_unconditional_flow_api_remains_backward_compatible():
    flow = AffineCouplingFlow(
        dimension=5, number_of_blocks=2, hidden_width=8, hidden_depth=1
    )
    x = np.zeros((3, 5), dtype=np.float64)
    assert flow.log_prob(x).shape == (3,)
    with pytest.raises(ValueError, match="not supported"):
        flow.log_prob(x, condition=np.ones((3, 1)))


def test_conditional_log_prob_requires_valid_external_condition():
    flow = AffineCouplingFlow(
        dimension=5, condition_dim=1, number_of_blocks=2, hidden_width=8, hidden_depth=1
    )
    x = np.zeros((3, 5), dtype=np.float64)
    with pytest.raises(ValueError, match="condition is required"):
        flow.log_prob(x)
    with pytest.raises(ValueError, match="shape"):
        flow.log_prob(x, condition=np.ones((2, 1)))
    assert np.isfinite(flow.log_prob(x, condition=np.ones((3, 1)))).all()


def test_condition_is_external_and_not_a_jacobian_dimension():
    flow = AffineCouplingFlow(
        dimension=5, condition_dim=1, number_of_blocks=2, hidden_width=8, hidden_depth=1
    )
    x = np.zeros((4, 5), dtype=np.float64)
    condition = np.ones((4, 1), dtype=np.float64)
    z, log_det = flow._module.inverse(flow._to_tensor(x), flow._condition_to_tensor(condition, 4))
    assert z.shape == (4, 5)
    assert log_det.shape == (4,)
    assert flow.config()["dimension"] == 5


def test_charge_mapping_is_deterministic():
    assert charge_condition(13) == 1.0
    assert charge_condition(-13) == -1.0
    with pytest.raises(ConditionalChargeError):
        charge_condition(11)


def test_balanced_sampling_reproduces_within_charge_pi_and_zero_weight_rule():
    raw = _raw_rows()
    datasets = {13: _dataset(raw), -13: _dataset(raw.copy())}
    pool_a = build_balanced_draw_pool(datasets, draw_budget=20000, seed=11)
    pool_b = build_balanced_draw_pool(datasets, draw_budget=20000, seed=11)
    np.testing.assert_array_equal(pool_a["raw"], pool_b["raw"])
    np.testing.assert_array_equal(pool_a["condition"], pool_b["condition"])
    assert np.count_nonzero(pool_a["condition"] == 1.0) == 10000
    assert np.count_nonzero(pool_a["condition"] == -1.0) == 10000
    for draw in pool_a["draws"].values():
        assert 2 not in draw["indices"]
        frequencies = np.bincount(draw["indices"], minlength=4) / draw["indices"].size
        np.testing.assert_allclose(frequencies, [1 / 6, 2 / 6, 0.0, 3 / 6], atol=0.02)
        assert draw["sampling"]["sample_weight_applied_to_loss"] is False
        assert draw["sampling"]["source_table_hash"] == dataset_hash(raw)


def test_odd_draw_budget_is_rejected():
    datasets = {13: _dataset(_raw_rows()), -13: _dataset(_raw_rows())}
    with pytest.raises(ConditionalChargeError, match="even"):
        build_balanced_draw_pool(datasets, draw_budget=5, seed=11)


def test_pooled_preprocessing_records_only_pooled_train_draws():
    raw = np.vstack((_raw_rows(), _raw_rows()))
    pipeline = FittedFeaturePipeline.fit(
        raw,
        FeatureView(IDENTITY_CARTESIAN_VIEW_ID),
    )
    assert pipeline.manifest()["standardization"]["fit_on"] == "train"
    assert pipeline.n_train_rows == raw.shape[0]


def test_symmetry_audit_refuses_undocumented_transform():
    raw = _raw_rows()
    empirical = {
        13: {"physical": raw[:, :5], "weights": raw[:, 7]},
        -13: {"physical": raw[:, :5], "weights": raw[:, 7]},
    }
    audit = run_symmetry_audit(empirical=empirical)
    assert audit["transform"] == "none"
    assert audit["evidence_status"] == "OPEN"
    with pytest.raises(ConditionalChargeError, match="undocumented"):
        run_symmetry_audit(empirical=empirical, transform="reflect_y")


def test_fixture_smoke_writes_deterministic_contract_artifacts(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (repo_root / "configs/density_lab/conditional_charge/d9_fixture_smoke_v0.json").read_text()
    )
    summary = run_fixture_pilot(config, output_dir=tmp_path, repo_root=repo_root)
    assert summary["status"] == "completed_fixture_pilot"
    assert summary["validation"]["macro_nll"] == pytest.approx(
        sum(summary["validation"]["per_charge"]) / 2.0
    )
    assert summary["condition_definition"] == {"pdg_13": 1.0, "pdg_-13": -1.0}
    for name in (
        "conditional_sampling_manifest.json",
        "conditional_training_metrics.json",
        "per_charge_validation.json",
        "per_charge_generated_summary.json",
        "symmetry_audit.json",
        "conditional_fixture_summary.json",
        "conditional_fixture_summary.csv",
        "report.md",
    ):
        assert (tmp_path / name).is_file()
