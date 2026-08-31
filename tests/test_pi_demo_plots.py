"""Tests for ``ship_muon_bg.density_lab.pi_demo_plots``.

All figure tests are marked ``lab`` and start with
``pytest.importorskip("matplotlib")``, per ``tests/test_density_reporting.py``'s
plot-test idiom. Tables are tiny, fully in-memory ``NominalTable``/
``TiltTable`` fixtures built the same way as ``tests/test_pi_demo.py``
(never touching the real 40k dataset or NF training), so the whole suite
runs in well under a few seconds.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from ship_muon_bg.density_lab import pi_demo
from ship_muon_bg.density_lab import utility_tilt as ut


def _tiny_nominal_table(rng: np.random.Generator, n: int = 30, pdg_id: int = 13) -> ut.NominalTable:
    """A small, fully in-memory Table A (never touches disk). Mirrors test_pi_demo.py."""

    w_mc = rng.uniform(0.1, 5.0, size=n)
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


def _write_tiny_tables(tmp_path: Path, pdg_id: int, tilt_ids, *, n: int = 30, seed: int = 0) -> Path:
    """Build and save tiny Table A/B under ``<tmp_path>/tables/pdg_<id>/``; return tables_root."""

    rng = np.random.default_rng(seed)
    table_a = _tiny_nominal_table(rng, n=n, pdg_id=pdg_id)
    configs = [ut.TILT_CONFIG_BY_ID[tid] for tid in tilt_ids]
    table_b = ut.build_tilt_table(table_a, configs)

    tables_root = tmp_path / "tables"
    pdg_dir = tables_root / "pdg_{}".format(pdg_id)
    table_a.save(pdg_dir, seed=seed)
    table_b.save(
        pdg_dir, configs, source_dataset_hash=table_a.source_dataset_hash,
        split_hash=table_a.split_hash, seed=seed,
    )
    return tables_root


def _build_full_csv_rows(tmp_path: Path, pdg_ids=(13, -13), n: int = 30) -> list:
    """One CSV row per (pdg_id, tilt_id) across the full 20-config grid, two PDGs."""

    all_rows = []
    for i, pdg_id in enumerate(pdg_ids):
        rng = np.random.default_rng(100 + i)
        table_a = _tiny_nominal_table(rng, n=n, pdg_id=pdg_id)
        configs = list(ut.ALL_TILT_CONFIGS)
        table_b = ut.build_tilt_table(table_a, configs)
        for config in configs:
            row, _ = pi_demo.compute_summary_row(
                table_a, table_b, config, draw_budget=200, sampler_seed=3,
            )
            all_rows.append(row)
    return all_rows


def _tiny_arena_reports(pdg_ids=(13, -13)):
    variants = ["NOMINAL_PHYSICAL", "UA_d0p9_a01", "UA_d0p1_a04", "UP_d0p9_a01", "UP_d0p1_a04"]
    reports = {}
    for pdg_id in pdg_ids:
        rows = []
        for i, vid in enumerate(variants):
            rows.append(
                {
                    "variant_id": vid,
                    "run_id": "fake-run-{}-{}".format(pdg_id, vid),
                    "nominal_b_toy_prevalence": 0.05,
                    "theoretical_target_b_toy_prevalence": 0.05 + 0.02 * i,
                    "generated_b_toy_occupancy": 0.04 + 0.02 * i,
                    "final_train_nll": 3.5 - 0.1 * i,
                    "finite_train_loss": True,
                }
            )
        reports[pdg_id] = {"rows": rows, "pdg_id": pdg_id}
    return reports


def _assert_two_paths_written(paths):
    assert len(paths) == 2
    suffixes = sorted(Path(p).suffix for p in paths)
    assert suffixes == [".pdf", ".png"]
    for p in paths:
        path = Path(p)
        assert path.exists()
        assert path.stat().st_size > 0


# --- load_summary_rows ---------------------------------------------------------


def test_load_summary_rows_round_trips_against_hand_written_csv(tmp_path):
    pytest.importorskip("matplotlib")
    from ship_muon_bg.density_lab import pi_demo_plots as plots

    row = {field: "" for field in pi_demo.CSV_FIELDNAMES}
    row.update(
        {
            "pdg_id": "13", "tilt_id": "UA_d0p9_a01", "mode": "UA", "delta": "0.9", "alpha": "1",
            "n_rows": "100", "p0": "0.05", "pU_closed_form": "0.06", "pU_direct": "0.061",
            "empirical_pU": "0.059", "enrichment": "1.22", "n_eff": "80.0", "n_eff_fraction": "0.8",
            "expected_n_unique": "90.0", "expected_unique_fraction": "0.9", "max_row_probability": "0.02",
            "top_10_mass": "0.15", "top_100_mass": "0.5", "draw_budget": "100",
            "observed_unique_rows": "88", "max_reuse_count": "3",
        }
    )
    csv_path = tmp_path / "summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pi_demo.CSV_FIELDNAMES), lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)

    rows = plots.load_summary_rows(csv_path)
    assert len(rows) == 1
    r = rows[0]
    assert r["pdg_id"] == 13 and isinstance(r["pdg_id"], int)
    assert r["tilt_id"] == "UA_d0p9_a01"
    assert r["mode"] == "UA"
    assert r["delta"] == pytest.approx(0.9)
    assert r["alpha"] == pytest.approx(1.0)
    assert r["p0"] == pytest.approx(0.05)
    assert r["nf_run_id"] is None
    assert r["nf_b_toy_fraction"] is None
    assert r["nominal_validation_nll"] is None


# --- figure 01 -------------------------------------------------------------


@pytest.mark.lab
def test_plot_01_writes_two_files(tmp_path):
    pytest.importorskip("matplotlib")
    from ship_muon_bg.density_lab import pi_demo_plots as plots

    csv_rows = _build_full_csv_rows(tmp_path, pdg_ids=(13, -13), n=20)
    out_dir = tmp_path / "figures"
    paths = plots.plot_01_tilt_enrichment_vs_alpha(csv_rows, tmp_path / "tables", out_dir)
    _assert_two_paths_written(paths)


# --- figure 02 -------------------------------------------------------------


@pytest.mark.lab
def test_plot_02_writes_two_files(tmp_path):
    pytest.importorskip("matplotlib")
    from ship_muon_bg.density_lab import pi_demo_plots as plots

    csv_rows = _build_full_csv_rows(tmp_path, pdg_ids=(13, -13), n=20)
    out_dir = tmp_path / "figures"
    paths = plots.plot_02_enrichment_vs_concentration(csv_rows, out_dir)
    _assert_two_paths_written(paths)


# --- figure 03 -------------------------------------------------------------


@pytest.mark.lab
def test_plot_03_writes_two_files(tmp_path):
    pytest.importorskip("matplotlib")
    from ship_muon_bg.density_lab import pi_demo_plots as plots

    tables_root = _write_tiny_tables(tmp_path, 13, ["UA_d0p1_a16"], n=25, seed=1)
    out_dir = tmp_path / "figures"
    paths = plots.plot_03_support_preservation(tables_root, out_dir)
    _assert_two_paths_written(paths)


# --- figure 04 -------------------------------------------------------------


@pytest.mark.lab
def test_plot_04_writes_two_files(tmp_path):
    pytest.importorskip("matplotlib")
    from ship_muon_bg.density_lab import pi_demo_plots as plots

    tables_root = _write_tiny_tables(tmp_path, 13, ["UA_d0p9_a01", "UA_d0p1_a04"], n=25, seed=2)
    out_dir = tmp_path / "figures"
    paths = plots.plot_04_direct_sampler_validation(tables_root, out_dir)
    _assert_two_paths_written(paths)


@pytest.mark.lab
def test_plot_04_deterministic_draws_across_repeated_calls(tmp_path):
    """Determinism check: same seed -> byte-identical drawn-sample arrays.

    Full PNG byte-identity is not asserted here: matplotlib's text-rendering
    float rounding is not guaranteed byte-stable across two independent
    render passes in this environment, so instead we verify the thing that
    actually matters for reproducibility -- the AliasSampler draw arrays fed
    into the plot are identical given the same fixed seed (_DIRECT_SAMPLER_SEED).
    """
    pytest.importorskip("matplotlib")
    from ship_muon_bg.density_lab import pi_demo_plots as plots

    tables_root = _write_tiny_tables(tmp_path, 13, ["UA_d0p9_a01", "UA_d0p1_a04"], n=25, seed=2)
    table_a = ut.NominalTable.load(tables_root / "pdg_13")
    table_b = ut.TiltTable.load(tables_root / "pdg_13")

    for tilt_id, _ in plots._TILT_PAIR_04:
        block_mask = table_b.block(tilt_id)
        pi_tilt_block = np.asarray(table_b.pi_tilt, dtype=np.float64)[block_mask]
        drawn_1 = ut.AliasSampler.from_probabilities(pi_tilt_block).draw(500, seed=plots._DIRECT_SAMPLER_SEED)
        drawn_2 = ut.AliasSampler.from_probabilities(pi_tilt_block).draw(500, seed=plots._DIRECT_SAMPLER_SEED)
        assert np.array_equal(drawn_1, drawn_2)

    # Also confirm the two calls to the plotting function themselves succeed
    # and both produce non-empty, real output (a weaker but still meaningful
    # determinism-adjacent smoke check on the actual public API).
    out_dir_1 = tmp_path / "figures_1"
    out_dir_2 = tmp_path / "figures_2"
    paths_1 = plots.plot_04_direct_sampler_validation(tables_root, out_dir_1)
    paths_2 = plots.plot_04_direct_sampler_validation(tables_root, out_dir_2)
    _assert_two_paths_written(paths_1)
    _assert_two_paths_written(paths_2)


# --- figure 05 -------------------------------------------------------------


@pytest.mark.lab
def test_plot_05_writes_two_files(tmp_path):
    pytest.importorskip("matplotlib")
    from ship_muon_bg.density_lab import pi_demo_plots as plots

    arena_reports = _tiny_arena_reports(pdg_ids=(13, -13))
    out_dir = tmp_path / "figures"
    paths = plots.plot_05_nf_preserves_tilt(arena_reports, out_dir, caveat="test caveat text")
    _assert_two_paths_written(paths)


@pytest.mark.lab
def test_plot_05_uses_default_caveat_when_not_provided(tmp_path):
    pytest.importorskip("matplotlib")
    from ship_muon_bg.density_lab import pi_demo_plots as plots

    arena_reports = _tiny_arena_reports(pdg_ids=(13,))
    out_dir = tmp_path / "figures"
    paths = plots.plot_05_nf_preserves_tilt(arena_reports, out_dir)
    _assert_two_paths_written(paths)


# --- review round 1 regression guards (findings I2, I6, I7, I8) -----------


@pytest.mark.lab
def test_plot_05_renders_missing_nf_value_without_crashing_or_faking_zero(tmp_path, monkeypatch):
    """A missing/untrained generated_b_toy_occupancy must not silently render
    as a zero-height bar (finding I6) -- it must not crash the figure either.

    Asserts an actual "N/A" text object is drawn on the axes -- not just
    that files were written (review round 1, finding NEW-3: the original
    version of this test only checked file existence, which would not have
    caught a regression back to a zero-height bar)."""

    pytest.importorskip("matplotlib")
    from ship_muon_bg.density_lab import pi_demo_plots as plots

    captured_figs = []
    original_save = plots._save_png_and_pdf

    def _capture_and_save(fig, out_dir, stem):
        captured_figs.append(fig)
        return original_save(fig, out_dir, stem)

    monkeypatch.setattr(plots, "_save_png_and_pdf", _capture_and_save)

    arena_reports = _tiny_arena_reports(pdg_ids=(13,))
    arena_reports[13]["rows"][1]["generated_b_toy_occupancy"] = None
    arena_reports[13]["rows"][1]["theoretical_target_b_toy_prevalence"] = None
    out_dir = tmp_path / "figures"
    paths = plots.plot_05_nf_preserves_tilt(arena_reports, out_dir, caveat="test caveat text")
    _assert_two_paths_written(paths)

    assert len(captured_figs) == 1
    fig = captured_figs[0]
    all_text = " ".join(t.get_text() for ax in fig.axes for t in ax.texts)
    assert "N/A" in all_text, "expected an 'N/A' text object for the missing NF value, found none"


@pytest.mark.lab
def test_plot_05_writes_two_files_with_generated_sample_count_present(tmp_path):
    """Same smoke check as test_plot_05_writes_two_files, but with
    generated_sample_count present (as real arena reports have it) so the
    binomial-SE error-bar code path (finding I2) actually executes, not just
    its None-safe fallback."""

    pytest.importorskip("matplotlib")
    from ship_muon_bg.density_lab import pi_demo_plots as plots

    arena_reports = _tiny_arena_reports(pdg_ids=(13,))
    for row in arena_reports[13]["rows"]:
        row["generated_sample_count"] = 4000
    out_dir = tmp_path / "figures"
    paths = plots.plot_05_nf_preserves_tilt(arena_reports, out_dir, caveat="test caveat text")
    _assert_two_paths_written(paths)


def test_plot_05_source_computes_binomial_error_bars():
    """Regression guard (finding I2): Q_theta bars must carry a binomial
    standard-error bar, computed from generated_sample_count."""

    import inspect

    from ship_muon_bg.density_lab import pi_demo_plots as module

    source = inspect.getsource(module.plot_05_nf_preserves_tilt)
    assert "generated_sample_count" in source
    assert "yerr" in source


def test_role_style_colors_disjoint_from_combo_and_pdg_colors():
    """The nominal/exact_target/empirical/nf role palette must never reuse a
    color that also encodes a tilt family or a PDG track elsewhere (finding
    I7) -- otherwise the same color means three different things across
    figures a reader may compare side by side."""

    from ship_muon_bg.density_lab import pi_demo_plots as plots

    role_colors = {v["color"].lower() for v in plots.ROLE_STYLE.values()}
    combo_colors = {c.lower() for c in plots._COMBO_COLORS.values()}
    # Figure 03's PDG-track colors, kept in sync with plot_03_support_preservation's
    # local ``pdg_colors`` dict by convention (tab:blue/tab:red).
    pdg_colors = {"tab:blue", "tab:red"}

    assert role_colors.isdisjoint(combo_colors), (role_colors, combo_colors)
    assert role_colors.isdisjoint(pdg_colors), (role_colors, pdg_colors)


def test_no_forbidden_physics_language_anywhere_in_plot_source():
    """Regression guard: no figure title/caption/caveat may ever describe the
    synthetic B_toy mechanism as a real FairShip/physics result, or imply an
    ESS pass/fail threshold (finding I8, mission-wide constraint)."""

    import inspect

    from ship_muon_bg.density_lab import pi_demo_plots as module

    source = inspect.getsource(module)
    lowered = " ".join(source.lower().split())  # collapse whitespace/newlines for substring checks

    # A plain substring blocklist is unusable here: the required disclaimers
    # legitimately CONTAIN phrases like "background rate" and "validated dis
    # label" (negated -- "never a background rate"), so any naive forbidden-
    # word scan false-positives on the very sentences that are supposed to be
    # present. Assert the required disclaimers positively instead (finding I8):
    assert "collapse" not in lowered  # never appears in this module, negated or not
    assert "not a universal law or an ess/ood threshold" in lowered
    assert "never a validated fairship danger metric" in lowered
    assert "never a background rate" in lowered
    assert "never a validated dis label" in lowered
    assert "never calibrated probabilities" in lowered


def test_support_preservation_caveat_is_actually_used():
    """The required verbatim support-preservation caveat sentence must be both
    defined and actually rendered by plot_03, not merely defined and orphaned."""

    import inspect

    from ship_muon_bg.density_lab import pi_demo_plots as module

    required_sentence = (
        "Support preservation is relative to the finite empirical nominal "
        "population."
    )
    assert required_sentence in module._SUPPORT_PRESERVATION_CAVEAT

    source = inspect.getsource(module.plot_03_support_preservation)
    assert "_SUPPORT_PRESERVATION_CAVEAT" in source
