"""Focused contracts for the Phase 2 conditional-charge flow (v1).

Covers: the physical muon-electric-charge-sign condition (not the PDG
numeric sign), the per-epoch direct sampler (arm D), and the end-to-end
fixture pilot with a closed test payload.
"""

from types import SimpleNamespace
import json
from pathlib import Path

import numpy as np
import pytest

from Nflow.torch_models.affine_coupling import AffineCouplingFlow
from ship_muon_bg.density_lab.conditional_charge import (
    CONDITION_FIELD_NAME,
    MUON_ELECTRIC_CHARGE_SIGN_BY_PDG,
    ConditionalChargeError,
    _EpochDirectSampler,
    muon_electric_charge_sign,
    run_symmetry_audit,
    run_fixture_pilot,
)
from ship_muon_bg.density_lab.feature_pipeline import FittedFeaturePipeline
from ship_muon_bg.data_contracts.feature_views import FeatureView, IDENTITY_CARTESIAN_VIEW_ID


def _raw_rows(pdg_id):
    return np.array(
        [
            [1.0, 0.2, 10.0, 0.1, 0.2, 1.0, float(pdg_id), 1.0],
            [1.2, 0.1, 11.0, 0.2, 0.3, 1.0, float(pdg_id), 2.0],
            [1.4, 0.0, 12.0, 0.3, 0.4, 1.0, float(pdg_id), 0.0],
            [1.6, -0.1, 13.0, 0.4, 0.5, 1.0, float(pdg_id), 3.0],
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


# --- physical charge-sign condition (v0 correction) --------------------------


def test_condition_field_name_is_muon_electric_charge_sign():
    assert CONDITION_FIELD_NAME == "muon_electric_charge_sign"


def test_charge_mapping_uses_physical_charge_not_pdg_numeric_sign():
    """PDG 13 names the mu- (physical charge -1); PDG -13 names the mu+ (physical charge +1).

    The PDG numeric sign is the *opposite* convention for this pair -- this
    is the exact defect the v1 correction fixes.
    """

    assert muon_electric_charge_sign(13) == -1.0
    assert muon_electric_charge_sign(-13) == 1.0
    assert MUON_ELECTRIC_CHARGE_SIGN_BY_PDG == {13: -1.0, -13: 1.0}
    with pytest.raises(ConditionalChargeError):
        muon_electric_charge_sign(11)


def test_symmetry_audit_refuses_undocumented_transform():
    raw13 = _raw_rows(13)
    raw_m13 = _raw_rows(-13)
    empirical = {
        13: {"physical": raw13[:, :5], "weights": raw13[:, 7]},
        -13: {"physical": raw_m13[:, :5], "weights": raw_m13[:, 7]},
    }
    audit = run_symmetry_audit(empirical=empirical)
    assert audit["transform"] == "none"
    assert audit["evidence_status"] == "OPEN"
    with pytest.raises(ConditionalChargeError, match="undocumented"):
        run_symmetry_audit(empirical=empirical, transform="reflect_y")


# --- per-epoch direct sampling (arm D) ----------------------------------------


def _pipeline_for(raw13, raw_m13):
    combined = np.vstack((raw13, raw_m13))
    return FittedFeaturePipeline.fit(combined, FeatureView(IDENTITY_CARTESIAN_VIEW_ID))


def test_epoch_direct_sampler_reproduces_within_charge_pi_and_excludes_zero_weight():
    raw13 = _raw_rows(13)
    raw_m13 = _raw_rows(-13)
    datasets = {13: _dataset(raw13), -13: _dataset(raw_m13)}
    sampler = _EpochDirectSampler(
        datasets=datasets, pipeline=_pipeline_for(raw13, raw_m13),
        draws_per_epoch_per_charge=20000, seed=11,
    )
    draw = sampler.draw(0)
    record = draw["metadata"]["per_charge"]["13"]
    assert record["draw_count"] == 20000
    assert draw["x"].shape == (40000, 5)
    assert draw["condition"].shape == (40000, 1)
    assert np.count_nonzero(draw["condition"] == 1.0) == 20000  # -13 -> +1
    assert np.count_nonzero(draw["condition"] == -1.0) == 20000  # 13 -> -1

    # Re-derive the draw for charge 13 directly to check pi reproduction and
    # the zero-weight-row exclusion (row index 2 has weight 0).
    indices = sampler._samplers[13].draw(20000, seed=record["derived_seed"])
    assert 2 not in indices
    frequencies = np.bincount(indices, minlength=4) / indices.size
    np.testing.assert_allclose(frequencies, [1 / 6, 2 / 6, 0.0, 3 / 6], atol=0.02)


def test_epoch_direct_sampler_draws_fresh_data_every_epoch():
    raw13 = _raw_rows(13)
    raw_m13 = _raw_rows(-13)
    datasets = {13: _dataset(raw13), -13: _dataset(raw_m13)}
    sampler = _EpochDirectSampler(
        datasets=datasets, pipeline=_pipeline_for(raw13, raw_m13),
        draws_per_epoch_per_charge=500, seed=11,
    )
    draw0 = sampler.draw(0)
    draw1 = sampler.draw(1)
    assert draw0["metadata"]["per_charge"]["13"]["draw_hash"] != draw1["metadata"]["per_charge"]["13"]["draw_hash"]
    assert not np.array_equal(draw0["x"], draw1["x"])

    # Rebuilding a fresh sampler with the same seed reproduces epoch 0 exactly.
    sampler_again = _EpochDirectSampler(
        datasets=datasets, pipeline=_pipeline_for(raw13, raw_m13),
        draws_per_epoch_per_charge=500, seed=11,
    )
    draw0_again = sampler_again.draw(0)
    assert draw0["metadata"]["per_charge"]["13"]["draw_hash"] == draw0_again["metadata"]["per_charge"]["13"]["draw_hash"]
    np.testing.assert_array_equal(draw0["x"], draw0_again["x"])


def test_epoch_direct_sampler_records_required_fields_and_no_sample_weight():
    raw13 = _raw_rows(13)
    raw_m13 = _raw_rows(-13)
    datasets = {13: _dataset(raw13), -13: _dataset(raw_m13)}
    sampler = _EpochDirectSampler(
        datasets=datasets, pipeline=_pipeline_for(raw13, raw_m13),
        draws_per_epoch_per_charge=50, seed=3,
    )
    draw = sampler.draw(0)
    metadata = draw["metadata"]
    assert metadata["sample_weight_applied_to_loss"] is False
    assert metadata["charge_counts"] == {"13": 50, "-13": 50}
    for pdg_key in ("13", "-13"):
        entry = metadata["per_charge"][pdg_key]
        for field in (
            "draw_hash", "source_probability_table_hash", "alias_table_hash",
            "unique_rows_drawn", "max_reuse_count", "cumulative_unique_source_rows",
        ):
            assert field in entry
    assert draw["condition"] is not None


def test_epoch_direct_sampler_cumulative_unique_rows_is_nondecreasing_and_bounded():
    raw13 = _raw_rows(13)
    raw_m13 = _raw_rows(-13)
    datasets = {13: _dataset(raw13), -13: _dataset(raw_m13)}
    sampler = _EpochDirectSampler(
        datasets=datasets, pipeline=_pipeline_for(raw13, raw_m13),
        draws_per_epoch_per_charge=10, seed=17,
    )
    previous = 0
    for epoch in range(5):
        draw = sampler.draw(epoch)
        cumulative = draw["metadata"]["per_charge"]["13"]["cumulative_unique_source_rows"]
        assert cumulative >= previous
        assert cumulative <= raw13.shape[0]
        previous = cumulative


def test_epoch_direct_sampler_rejects_nonpositive_draw_count():
    raw13 = _raw_rows(13)
    raw_m13 = _raw_rows(-13)
    datasets = {13: _dataset(raw13), -13: _dataset(raw_m13)}
    with pytest.raises(ConditionalChargeError):
        _EpochDirectSampler(
            datasets=datasets, pipeline=_pipeline_for(raw13, raw_m13),
            draws_per_epoch_per_charge=0, seed=11,
        )


# --- end-to-end fixture pilot --------------------------------------------------


def test_fixture_smoke_writes_deterministic_contract_artifacts(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (repo_root / "configs/density_lab/conditional_charge/d9_fixture_smoke_v1.json").read_text()
    )
    summary = run_fixture_pilot(config, output_dir=tmp_path, repo_root=repo_root)
    assert summary["status"] == "completed_fixture_pilot"
    assert summary["schema_version"] == "1"
    assert summary["validation"]["macro_nll"] == pytest.approx(
        sum(summary["validation"]["per_charge"]) / 2.0
    )
    assert summary["condition_field_name"] == "muon_electric_charge_sign"
    assert summary["condition_definition"] == {"pdg_13": -1.0, "pdg_-13": 1.0}
    for pdg_key in ("13", "-13"):
        provenance = summary["test_provenance"][pdg_key]
        assert provenance["test_payload_loaded"] is False
        assert provenance["test_used_for_training"] is False
        assert provenance["test_used_for_preprocessing"] is False
        assert provenance["test_used_for_model_selection"] is False
        assert provenance["test_used_for_evaluation"] is False
        assert provenance["test_row_count"] > 0
    assert summary["pipeline_manifest"]["standardization"]["weighting"] == (
        "macro_balanced_physical_weight_train_only"
    )
    assert summary["sampling_manifest"]["n_epochs_recorded"] == 2
    epoch_hashes = {
        epoch["per_charge"]["13"]["draw_hash"] for epoch in summary["sampling_manifest"]["epochs"]
    }
    assert len(epoch_hashes) == 2  # two distinct epochs, two distinct draws
    for name in (
        "conditional_sampling_manifest.json",
        "conditional_training_metrics.json",
        "per_charge_validation.json",
        "per_charge_generated_summary.json",
        "symmetry_audit.json",
        "conditional_fixture_summary.json",
        "conditional_fixture_summary.csv",
        "conditional_fixture_summary.md",
        "report.md",
    ):
        assert (tmp_path / name).is_file()


def test_fixture_pilot_rejects_legacy_v0_condition_field_name(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (repo_root / "configs/density_lab/conditional_charge/d9_fixture_smoke_v1.json").read_text()
    )
    config["model"]["condition_name"] = "charge_sign"
    with pytest.raises(ConditionalChargeError, match="muon_electric_charge_sign"):
        run_fixture_pilot(config, output_dir=tmp_path, repo_root=repo_root)


def test_fixture_pilot_rejects_schema_version_0(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (repo_root / "configs/density_lab/conditional_charge/d9_fixture_smoke_v1.json").read_text()
    )
    config["schema_version"] = "0"
    with pytest.raises(ConditionalChargeError, match="schema_version"):
        run_fixture_pilot(config, output_dir=tmp_path, repo_root=repo_root)
