"""Tiny synthetic end-to-end family arena: all four families fit, evaluated,
persisted, reloaded and sampled deterministically on the same tiny track.

Covers required tests 37, 38.
"""

from __future__ import annotations

import numpy as np

from ship_muon_bg.afterms.d9_5 import data_scope, evaluation, gaussian_adapter, gmm_adapter, nf_ac_adapter


def test_tiny_end_to_end_family_arena(
    tiny_shard_dir, tmp_path, tiny_nf_architecture, tiny_nf_execution_policy,
    tiny_nf_optimizer_settings, tiny_evaluation_policy,
):
    manifest = data_scope.load_shard_manifest(tiny_shard_dir)
    train_raw = data_scope.load_filtered_split(tiny_shard_dir, manifest, "train", 13)
    validation_raw = data_scope.load_filtered_split(tiny_shard_dir, manifest, "validation", 13)

    adapters = {
        "GAUSS_DIAG": gaussian_adapter.DiagonalGaussianAdapter(),
        "GAUSS_FULL": gaussian_adapter.FullGaussianAdapter(),
        "GMM": gmm_adapter.GmmAdapter(max_iter=15),
        "NF_AC": nf_ac_adapter.NfAcAdapter(
            track_id="TRK_PDG13_UW_ID", pdg_value=13, architecture=tiny_nf_architecture,
            execution_policy=tiny_nf_execution_policy, optimizer_settings=tiny_nf_optimizer_settings,
            evaluation_policy=tiny_evaluation_policy, artifact_root=tmp_path, device="cpu",
        ),
    }
    fit_seed_by_family = {"GAUSS_DIAG": 0, "GAUSS_FULL": 0, "GMM": 20260720, "NF_AC": 20260720}

    for family, adapter in adapters.items():
        result = adapter.fit(train_raw, validation_raw, seed=fit_seed_by_family[family])
        assert result is not None

    budget = tiny_evaluation_policy["quick_validation_budget"]
    results = []
    for family, adapter in adapters.items():
        result = evaluation.evaluate_adapter(
            adapter, validation_raw, split_name="validation", track_id="TRK_PDG13_UW_ID",
            budget=budget, budget_name="quick_validation_budget", generation_seed=1,
        )
        assert result["model_family_id"] == family
        assert result["physical_nll"] is not None or result["physical_nll_error"] is not None
        results.append(result)
    assert {r["model_family_id"] for r in results} == {"GAUSS_DIAG", "GAUSS_FULL", "GMM", "NF_AC"}

    # Full parameter/checkpoint reload and deterministic sampling for every family.
    bundle_dirs = {
        "GAUSS_DIAG": tmp_path / "bundles" / "gauss_diag",
        "GAUSS_FULL": tmp_path / "bundles" / "gauss_full",
        "GMM": tmp_path / "bundles" / "gmm",
    }
    for family, out_dir in bundle_dirs.items():
        adapters[family].save_bundle(out_dir)

    reloaded = {
        "GAUSS_DIAG": gaussian_adapter.DiagonalGaussianAdapter.load_bundle(bundle_dirs["GAUSS_DIAG"]),
        "GAUSS_FULL": gaussian_adapter.FullGaussianAdapter.load_bundle(bundle_dirs["GAUSS_FULL"]),
        "GMM": gmm_adapter.GmmAdapter.load_bundle(bundle_dirs["GMM"]),
        "NF_AC": nf_ac_adapter.NfAcAdapter.load_bundle(adapters["NF_AC"]._run_dir),
    }
    for family in adapters:
        original_samples = adapters[family].sample(25, seed=7)
        reloaded_samples = reloaded[family].sample(25, seed=7)
        np.testing.assert_allclose(original_samples, reloaded_samples)
