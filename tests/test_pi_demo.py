"""Tests for ``ship_muon_bg.density_lab.pi_demo`` (the analysis layer for the
PI-ready utility-guided sampling demonstration).

Pure NumPy -- no matplotlib, no torch, no sklearn. Not marked ``lab`` or
``flow``: these tests need only numpy, which is core, per
``tests/conftest.py``'s stated goal that ``python -m pytest -q`` passes with
only numpy installed. All synthetic tables here are tiny (well under 100
rows) so the suite runs in well under a second.
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from ship_muon_bg.density_lab import pi_demo
from ship_muon_bg.density_lab import utility_tilt as ut


def _tiny_nominal_table(rng: np.random.Generator, n: int = 30, pdg_id: int = 13) -> ut.NominalTable:
    """A small, fully in-memory Table A (never touches disk)."""

    w_mc = rng.uniform(0.1, 5.0, size=n)
    # Deterministic positive-B_toy pattern (every 5th row) so p0 is a known,
    # reproducible, strictly positive quantity regardless of the rng draw --
    # unlike a random threshold, this can never accidentally degenerate to
    # an all-zero B_toy for a small n.
    b_toy = (np.arange(n) % 5 == 0).astype(np.float64)
    thresholds = ut.ToyThresholds(pdg_id=pdg_id, t_pT=1.0, t_R=1.0)
    return ut.NominalTable(
        pdg_id=pdg_id,
        source_dataset_hash="test-source-hash",
        split_id="train",
        split_hash="test-split-hash",
        thresholds=thresholds,
        row_ref=np.arange(n, dtype=np.int64),
        source_row_index=np.arange(n, dtype=np.int64),
        w_mc=w_mc,
        pi_nominal=ut.pi_nominal(w_mc),
        pT=rng.uniform(size=n),
        R_xy=rng.uniform(size=n),
        B_toy=b_toy,
        U_A=ut.compute_utility_a(b_toy),
        U_P=ut.compute_utility_p(b_toy),
        zero_weight=(w_mc <= 0.0),
    )


def test_compute_summary_row_closed_form_matches_direct_across_configs():
    rng = np.random.default_rng(0)
    table_a = _tiny_nominal_table(rng, n=40)
    # Spans both modes (UA/UP) and both deltas (0.1/0.9), including the
    # extreme alpha=16 corner where r_negative can be ~1e-16.
    tilt_ids = ("UA_d0p9_a01", "UA_d0p1_a04", "UP_d0p9_a08", "UP_d0p1_a16")
    configs = [ut.TILT_CONFIG_BY_ID[tid] for tid in tilt_ids]
    table_b = ut.build_tilt_table(table_a, configs)

    for config in configs:
        row, abs_closed_vs_direct = pi_demo.compute_summary_row(
            table_a, table_b, config, draw_budget=200, sampler_seed=3,
        )
        assert abs_closed_vs_direct < 1e-9
        assert row["pU_closed_form"] == pytest.approx(row["pU_direct"], abs=1e-9)
        assert np.isfinite(row["pU_closed_form"])
        assert np.isfinite(row["pU_direct"])
        # Frozen schema: exact key set AND exact order.
        assert list(row.keys()) == list(pi_demo.CSV_FIELDNAMES)


def test_compute_summary_row_enrichment_guards_zero_prevalence():
    rng = np.random.default_rng(4)
    n = 20
    w_mc = rng.uniform(0.1, 5.0, size=n)
    b_toy = np.zeros(n, dtype=np.float64)  # p0 == 0 by construction
    thresholds = ut.ToyThresholds(pdg_id=13, t_pT=1.0, t_R=1.0)
    table_a = ut.NominalTable(
        pdg_id=13, source_dataset_hash="h", split_id="train", split_hash="s",
        thresholds=thresholds, row_ref=np.arange(n, dtype=np.int64),
        source_row_index=np.arange(n, dtype=np.int64), w_mc=w_mc,
        pi_nominal=ut.pi_nominal(w_mc), pT=rng.uniform(size=n), R_xy=rng.uniform(size=n),
        B_toy=b_toy, U_A=ut.compute_utility_a(b_toy), U_P=ut.compute_utility_p(b_toy),
        zero_weight=(w_mc <= 0.0),
    )
    config = ut.TILT_CONFIG_BY_ID["UA_d0p9_a02"]
    table_b = ut.build_tilt_table(table_a, [config])

    row, _ = pi_demo.compute_summary_row(table_a, table_b, config, draw_budget=50, sampler_seed=1)

    assert row["p0"] == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(row["enrichment"])


def test_write_summary_csv_round_trip_matches_frozen_schema(tmp_path):
    rng = np.random.default_rng(1)
    table_a = _tiny_nominal_table(rng, n=25)
    config = ut.TILT_CONFIG_BY_ID["UA_d0p9_a02"]
    table_b = ut.build_tilt_table(table_a, [config])
    row, _ = pi_demo.compute_summary_row(table_a, table_b, config, draw_budget=50, sampler_seed=5)

    out_path = tmp_path / "summary.csv"
    pi_demo.write_summary_csv([row], out_path)

    with out_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(pi_demo.CSV_FIELDNAMES)
        rows = list(reader)

    assert len(rows) == 1
    for column in (
        "nf_run_id", "nf_b_toy_fraction", "nf_minus_target",
        "nominal_validation_nll", "tilted_validation_nll",
    ):
        assert rows[0][column] == ""
    assert rows[0]["tilt_id"] == "UA_d0p9_a02"


def _write_fake_table_manifests(tables_root, pdg_id, *, n_rows):
    pdg_dir = tables_root / "pdg_{}".format(pdg_id)
    pdg_dir.mkdir(parents=True)
    (pdg_dir / "table_a_nominal.manifest.json").write_text(json.dumps({
        "n_rows": n_rows, "table_hash": "hash-a-{}".format(pdg_id), "split_hash": "hash-split-{}".format(pdg_id),
    }))
    (pdg_dir / "table_b_tilt.manifest.json").write_text(json.dumps({
        "table_hash": "hash-b-{}".format(pdg_id),
    }))
    return pdg_dir


def test_write_run_manifest_dataset_scope_and_terminology(tmp_path):
    tables_root = tmp_path / "tables"
    _write_fake_table_manifests(tables_root, 13, n_rows=10)

    out_path = tmp_path / "run_manifest.json"
    validation_gate = {
        "pi_nominal_sums_to_one": {13: True},
        "pi_tilt_sums_to_one": {13: {}},
        "max_abs_closed_vs_direct": 0.0,
        "all_pdg_kept_separate": True,
    }
    payload = pi_demo.write_run_manifest(
        out_path=out_path,
        dataset_path="data/samples/muonsFullMC_afterMS_sample.npz",
        dataset_hash="deadbeef",
        dataset_scope="sample",
        pdg_ids=[13],
        tables_root=tables_root,
        seed=11,
        sampler_seed=7,
        draw_budget_per_pdg={13: 10},
        validation_gate=validation_gate,
    )

    assert payload["dataset_scope"] == "sample"
    assert "never a physical weight" in payload["terminology"]["r_utility"]

    written = json.loads(out_path.read_text())
    assert written["dataset_scope"] == "sample"
    assert "never a physical weight" in written["terminology"]["r_utility"]


def test_write_run_manifest_rejects_non_sample_scope(tmp_path):
    tables_root = tmp_path / "tables"
    _write_fake_table_manifests(tables_root, 13, n_rows=10)

    with pytest.raises(ValueError):
        pi_demo.write_run_manifest(
            out_path=tmp_path / "run_manifest.json",
            dataset_path="data/samples/muonsFullMC_afterMS_sample.npz",
            dataset_hash="deadbeef",
            dataset_scope="full",
            pdg_ids=[13],
            tables_root=tables_root,
            seed=11,
            sampler_seed=7,
            draw_budget_per_pdg={13: 10},
            validation_gate={
                "pi_nominal_sums_to_one": {}, "pi_tilt_sums_to_one": {},
                "max_abs_closed_vs_direct": 0.0, "all_pdg_kept_separate": True,
            },
        )


def test_compute_summary_row_empirical_pu_is_deterministic():
    rng = np.random.default_rng(2)
    table_a = _tiny_nominal_table(rng, n=35)
    config = ut.TILT_CONFIG_BY_ID["UP_d0p9_a04"]
    table_b = ut.build_tilt_table(table_a, [config])

    row1, _ = pi_demo.compute_summary_row(table_a, table_b, config, draw_budget=500, sampler_seed=42)
    row2, _ = pi_demo.compute_summary_row(table_a, table_b, config, draw_budget=500, sampler_seed=42)

    assert row1["empirical_pU"] == row2["empirical_pU"]
    assert row1["observed_unique_rows"] == row2["observed_unique_rows"]
    assert row1["max_reuse_count"] == row2["max_reuse_count"]
