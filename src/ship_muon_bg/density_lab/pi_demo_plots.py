"""Figures 01-05 for the PI-ready utility-guided sampling demonstration.

Reads the already-built ``utility_guided_summary.csv``, Table A / Table B
reference tables (``density_lab.utility_tilt``), and NF arena reports
(``arena_summary.json``). Contains no tilt mathematics of its own -- every
number plotted is either read straight from those artifacts or computed via
an already-tested function imported from ``utility_tilt``/``pi_demo``.

Matplotlib is imported ONLY lazily inside functions (never at module import
time), matching ``reporting.py``'s pattern and the density_lab import-hygiene
convention: ``ship_muon_bg.density_lab`` must remain importable with only
numpy present.

``B_toy`` is a project-defined SYNTHETIC diagnostic region (thresholds on
``pT``, ``R_xy``) -- never a validated FairShip danger metric, never a
background rate, never a validated DIS label. ``U_A``/``U_P`` are synthetic
utility scores, never calibrated probabilities. Every figure title/caption
below states this explicitly; nothing here should be read as a SHiP physics
result.

Visual identity (module-level, reused across every figure that shows these
roles -- see mission instructions). ROLE_STYLE deliberately avoids the
tab10 palette entirely (gray + purple + teal + brown) so it never collides
with _COMBO_COLORS (tilt-family encoding, Figures 01/02) or the PDG-track
colors (Figure 03), which both use tab10 colors -- review round 1 (finding
I7) found the original palette reused tab:blue/tab:green for two different
meanings across figures:

- "nominal"      -- the untilted reference law / p0 (gray).
- "exact_target" -- an exact, closed-form or direct-summation quantity
  (purple).
- "empirical"    -- an alias-sampler empirical draw statistic (teal).
- "nf"           -- a trained normalizing-flow's own generated samples
  (brown, hatched) -- used only in Figure 05, where it is the fourth,
  necessarily distinct role (an NF sample is neither an exact target nor a
  plain alias-sampler draw).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .pi_demo import CSV_FIELDNAMES
from .utility_tilt import (
    AliasSampler,
    NominalTable,
    TiltTable,
    empirical_draw_diagnostics,
    h_alpha_delta,
    theoretical_concentration_diagnostics,
)

# --- shared visual identity ---------------------------------------------------

ROLE_STYLE: Dict[str, Dict[str, Any]] = {
    "nominal": {"color": "#7f7f7f", "marker": "s", "linestyle": "--"},
    "exact_target": {"color": "#9467bd", "marker": "o", "linestyle": "-"},
    "empirical": {"color": "#17becf", "marker": "^", "linestyle": ":"},
    "nf": {"color": "#8c564b", "marker": "D", "hatch": "//"},
}

# Figure 01/02: (mode, delta) combination colors -- a genuinely separate
# encoding axis from ROLE_STYLE (that dict encodes *what kind of quantity*
# is plotted -- nominal/exact/empirical/NF; this one encodes *which tilt
# family*, all tab10 colors, none of which appear in ROLE_STYLE above).
# Kept in the requested display order.
_MODE_DELTA_COMBOS: Tuple[Tuple[str, float], ...] = (
    ("UA", 0.1), ("UA", 0.9), ("UP", 0.1), ("UP", 0.9),
)
_COMBO_COLORS: Dict[Tuple[str, float], str] = {
    ("UA", 0.1): "tab:blue",
    ("UA", 0.9): "tab:orange",
    ("UP", 0.1): "tab:green",
    ("UP", 0.9): "tab:red",
}

_ALPHA_SIZES: Dict[float, float] = {1: 30, 2: 55, 4: 85, 8: 125, 16: 180}

_SUPPORT_PRESERVATION_CAVEAT = (
    "Support preservation is relative to the finite empirical nominal "
    "population. It does not prove that the available dataset covers all "
    "physically relevant SHiP regions."
)

_DEFAULT_NF_CAVEAT = (
    "NF results here are diagnostic-only: a bounded CPU-scale flow, not a "
    "converged production model. An NLL comparison is not equivalent to "
    "direct-sampling agreement."
)


def _figure():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_png_and_pdf(fig, out_dir: Path, stem: str) -> List[str]:
    """Write ``<stem>.png`` (dpi=110, house style) and ``<stem>.pdf``; close ``fig``."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    for suffix in (".png", ".pdf"):
        path = out_dir / (stem + suffix)
        fig.savefig(path, dpi=110)
        written.append(str(path))
    import matplotlib.pyplot as plt

    plt.close(fig)
    return written


# --- CSV loading ---------------------------------------------------------------

# String-valued fields: never coerced to float/int. ``nf_run_id`` is an
# identifier string (empty -> None when a tilt was not NF-trained), never a
# numeric quantity.
_STRING_FIELDS = frozenset({"tilt_id", "mode", "nf_run_id"})
_INT_FIELDS = frozenset({"pdg_id", "n_rows", "draw_budget", "observed_unique_rows", "max_reuse_count"})


def load_summary_rows(csv_path: Path) -> List[Dict[str, Any]]:
    """Thin ``csv.DictReader`` wrapper: coerces numeric columns, empty NF fields -> None."""

    csv_path = Path(csv_path)
    rows: List[Dict[str, Any]] = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(CSV_FIELDNAMES):
            raise ValueError(
                "unexpected summary CSV header: got {!r}, expected {!r}".format(
                    reader.fieldnames, list(CSV_FIELDNAMES)
                )
            )
        for raw in reader:
            converted: Dict[str, Any] = {}
            for field in CSV_FIELDNAMES:
                value = raw.get(field, "")
                if field in _STRING_FIELDS:
                    converted[field] = value if value != "" else None
                elif value == "":
                    converted[field] = None
                elif field in _INT_FIELDS:
                    converted[field] = int(value)
                else:
                    converted[field] = float(value)
            rows.append(converted)
    return rows


def _pdg_ids_sorted(rows: Sequence[Dict[str, Any]]) -> List[int]:
    return sorted({int(r["pdg_id"]) for r in rows}, key=lambda v: (abs(v), v))


def _rows_for(rows: Sequence[Dict[str, Any]], pdg_id: int, mode: str, delta: float) -> List[Dict[str, Any]]:
    subset = [
        r for r in rows
        if int(r["pdg_id"]) == pdg_id and r["mode"] == mode and float(r["delta"]) == float(delta)
    ]
    return sorted(subset, key=lambda r: float(r["alpha"]))


# --- Figure 01 -------------------------------------------------------------


def plot_01_tilt_enrichment_vs_alpha(
    csv_rows: Sequence[Dict[str, Any]], tables_root: Path, out_dir: Path
) -> List[str]:
    """Closed-form vs direct-summation vs empirical tilted prevalence, by alpha.

    ``tables_root`` is accepted for interface symmetry with the other table-
    reading figures but is unused here: every quantity plotted (closed-form,
    direct, empirical prevalence, p0) is already a column of ``csv_rows``.
    """

    del tables_root  # unused: everything needed is already in csv_rows
    plt = _figure()

    pdg_ids = _pdg_ids_sorted(csv_rows)
    n = max(len(pdg_ids), 1)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 7), squeeze=False)
    axes = axes[0]

    max_abs_diff = 0.0
    for ax, pdg_id in zip(axes, pdg_ids):
        pdg_rows = [r for r in csv_rows if int(r["pdg_id"]) == pdg_id]
        p0 = float(pdg_rows[0]["p0"]) if pdg_rows else float("nan")

        for mode, delta in _MODE_DELTA_COMBOS:
            series = _rows_for(csv_rows, pdg_id, mode, delta)
            if not series:
                continue
            alphas = [float(r["alpha"]) for r in series]
            closed = [float(r["pU_closed_form"]) for r in series]
            direct = [float(r["pU_direct"]) for r in series]
            empirical = [r["empirical_pU"] for r in series]
            color = _COMBO_COLORS[(mode, delta)]
            label = "{}, delta={:g}".format(mode, delta)

            ax.plot(
                alphas, closed, color=color, marker=ROLE_STYLE["exact_target"]["marker"],
                linestyle="-", linewidth=1.3, label="{} closed-form".format(label),
            )
            ax.scatter(alphas, direct, color=color, marker="x", s=45, label="{} direct-sum".format(label))
            emp_alphas = [a for a, e in zip(alphas, empirical) if e is not None]
            emp_vals = [e for e in empirical if e is not None]
            if emp_vals:
                ax.scatter(
                    emp_alphas, emp_vals, facecolors="none", edgecolors=color,
                    marker=ROLE_STYLE["empirical"]["marker"], s=60, label="{} empirical (alias draw)".format(label),
                )

            for c, d in zip(closed, direct):
                max_abs_diff = max(max_abs_diff, abs(c - d))

        ax.axhline(
            p0, color=ROLE_STYLE["nominal"]["color"], linestyle=ROLE_STYLE["nominal"]["linestyle"],
            linewidth=1.2, label="nominal p0",
        )
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 4, 8, 16])
        ax.set_xticklabels(["1", "2", "4", "8", "16"])
        ax.set_xlabel("alpha (exploitation strength, log-spaced grid)")
        ax.set_ylabel("tilted B_toy prevalence (synthetic)")
        ax.set_title("PDG {}".format(pdg_id))
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)

    fig.suptitle(
        "Exact-vs-empirical agreement of the utility tilt on SYNTHETIC B_toy "
        "(not a FairShip physics result)\nmax |pU_closed_form - pU_direct| "
        "observed across all plotted rows = {:.3e}".format(max_abs_diff),
        fontsize=11, wrap=True,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return _save_png_and_pdf(fig, out_dir, "01_tilt_enrichment_vs_alpha")


# --- Figure 02 -------------------------------------------------------------


def plot_02_enrichment_vs_concentration(csv_rows: Sequence[Dict[str, Any]], out_dir: Path) -> List[str]:
    """Enrichment (pU_direct/p0) vs concentration (N_eff/N), all (pdg, tilt) points."""

    plt = _figure()

    pdg_ids = _pdg_ids_sorted(csv_rows)
    n = max(len(pdg_ids), 1)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 7), squeeze=False)
    axes = axes[0]

    mode_markers = {"UA": "o", "UP": "^"}
    mode_colors = {"UA": "tab:blue", "UP": "tab:orange"}

    # Shared axis limits across both panels (finding O4): without this, the
    # same N_eff/N value lands at a different horizontal position in each
    # panel, defeating the point of a side-by-side PDG comparison.
    all_x = [float(r["n_eff_fraction"]) for r in csv_rows]
    all_y = [float(r["enrichment"]) for r in csv_rows]
    x_margin = 0.05 * (max(all_x) - min(all_x)) if all_x else 0.0
    shared_xlim = (min(all_x) - x_margin, max(all_x) + x_margin) if all_x else None
    shared_ylim = (min(all_y) / 1.3, max(all_y) * 1.3) if all_y else None

    for ax, pdg_id in zip(axes, pdg_ids):
        pdg_rows = [r for r in csv_rows if int(r["pdg_id"]) == pdg_id]
        for r in pdg_rows:
            mode = r["mode"]
            delta = float(r["delta"])
            alpha = float(r["alpha"])
            x = float(r["n_eff_fraction"])
            y = float(r["enrichment"])
            color = mode_colors[mode]
            size = _ALPHA_SIZES.get(alpha, 60)
            filled = delta <= 0.5  # delta=0.1 -> filled, delta=0.9 -> open
            ax.scatter(
                [x], [y], marker=mode_markers[mode], s=size,
                facecolors=color if filled else "none", edgecolors=color, linewidths=1.3,
            )
            if alpha in (1.0, 16.0):
                # UA labels nudged up-right, UP labels down-right, so the
                # two families' annotations separate instead of overlapping
                # at tightly clustered points (finding O6).
                dy = 6 if mode == "UA" else -10
                ax.annotate(
                    "a={:g}".format(alpha), (x, y), fontsize=6, color=color,
                    xytext=(4, dy), textcoords="offset points",
                )
        ax.set_yscale("log")
        if shared_xlim is not None:
            ax.set_xlim(*shared_xlim)
        if shared_ylim is not None:
            ax.set_ylim(*shared_ylim)
        ax.set_xlabel("N_eff / N (concentration)")
        ax.set_ylabel("enrichment = pU_direct / p0 (log scale)")
        ax.set_title("PDG {}".format(pdg_id))

        # Manual legend: marker shape = mode, fill = delta; alpha is size-coded
        # and annotated at its two extremes (see above), not in the legend.
        handles = [
            ax.scatter([], [], marker="o", facecolors="tab:blue", edgecolors="tab:blue", label="UA, delta=0.1 (filled)"),
            ax.scatter([], [], marker="o", facecolors="none", edgecolors="tab:blue", label="UA, delta=0.9 (open)"),
            ax.scatter([], [], marker="^", facecolors="tab:orange", edgecolors="tab:orange", label="UP, delta=0.1 (filled)"),
            ax.scatter([], [], marker="^", facecolors="none", edgecolors="tab:orange", label="UP, delta=0.9 (open)"),
        ]
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)

    fig.suptitle(
        "Increasing exploitation (alpha) can increase utility enrichment while "
        "concentrating sampling probability -- an observation on SYNTHETIC "
        "B_toy, not a universal law or an ESS/OOD threshold",
        fontsize=11, wrap=True,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _save_png_and_pdf(fig, out_dir, "02_enrichment_vs_concentration")


# --- Figure 03 -------------------------------------------------------------


def plot_03_support_preservation(tables_root: Path, out_dir: Path) -> List[str]:
    """Tilted/nominal ratio (strictly positive, never zero) plus the h floor."""

    plt = _figure()
    tables_root = Path(tables_root)

    tilt_id = "UA_d0p1_a16"
    pdg_colors = {13: "tab:blue", -13: "tab:red"}

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 7))

    pdg_dirs = sorted(p for p in tables_root.iterdir() if p.is_dir() and p.name.startswith("pdg_"))
    plotted_any = False
    for pdg_dir in pdg_dirs:
        pdg_id = int(pdg_dir.name[len("pdg_"):])
        try:
            table_a = NominalTable.load(pdg_dir)
            table_b = TiltTable.load(pdg_dir)
        except FileNotFoundError:
            continue
        if tilt_id not in table_b.tilt_ids:
            continue
        block_mask = table_b.block(tilt_id)
        pi_tilt_block = np.asarray(table_b.pi_tilt, dtype=np.float64)[block_mask]
        pi_nom = np.asarray(table_a.pi_nominal, dtype=np.float64)
        positive = pi_nom > 0.0
        if not np.any(positive):
            continue
        ratio = pi_tilt_block[positive] / pi_nom[positive]
        ratio_sorted = np.sort(ratio)[::-1]
        color = pdg_colors.get(pdg_id, "tab:gray")
        ax_a.plot(
            np.arange(1, ratio_sorted.shape[0] + 1), ratio_sorted,
            color=color, linewidth=1.3, label="pdg={}".format(pdg_id),
        )
        plotted_any = True

    ax_a.set_yscale("log")
    ax_a.set_xlabel("rank (sorted rows, nominal-positive support)")
    ax_a.set_ylabel("pi_tilt / pi_nominal (log scale)")
    ax_a.set_title("Tilted/nominal ratio, {} (strongest available exploitation)".format(tilt_id))
    if plotted_any:
        ax_a.legend(fontsize=7)

    u = np.linspace(0.0, 1.0, 200)
    h_floor = h_alpha_delta(u, alpha=16, delta=0.1)
    h_zero = h_alpha_delta(u, alpha=16, delta=0.0)
    ax_b.plot(u, h_floor, color=ROLE_STYLE["exact_target"]["color"], linewidth=1.5, label="delta=0.1 (support-preserving floor)")
    ax_b.plot(u, h_zero, color=ROLE_STYLE["nominal"]["color"], linestyle="--", linewidth=1.5, label="delta=0.0 (reference, touches zero)")
    ax_b.set_yscale("log")
    ax_b.set_xlabel("u")
    ax_b.set_ylabel("h_alpha_delta(u), alpha=16 (log scale)")
    ax_b.set_title("Utility transform with vs without the delta floor")
    ax_b.legend(fontsize=7)

    fig.suptitle(
        "Why delta>0: strictly positive, finite tilted/nominal ratio (synthetic B_toy, not a FairShip result)",
        fontsize=11, wrap=True,
    )
    fig.text(0.5, 0.01, _SUPPORT_PRESERVATION_CAVEAT, ha="center", fontsize=8, wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    return _save_png_and_pdf(fig, out_dir, "03_support_preservation")


# --- Figure 04 -------------------------------------------------------------


_TILT_PAIR_04: Tuple[Tuple[str, str], ...] = (
    ("UA_d0p9_a01", "mild tilt"),
    ("UA_d0p1_a04", "strong tilt"),
)
_DIRECT_SAMPLER_SEED = 7


def plot_04_direct_sampler_validation(tables_root: Path, out_dir: Path) -> List[str]:
    """Rank-ordered cumulative declared vs empirical alias-draw mass, PDG 13 only."""

    plt = _figure()
    tables_root = Path(tables_root)
    pdg_dir = tables_root / "pdg_13"
    table_a = NominalTable.load(pdg_dir)
    table_b = TiltTable.load(pdg_dir)

    n_train = table_a.n_rows
    budgets = (n_train, 200000)
    # Both empirical-draw budgets share the "empirical" role color (distinct
    # linestyle differentiates T=n_train from T=200000); avoids reusing
    # ROLE_STYLE["exact_target"]'s purple, which is also drawn in this figure.
    budget_styles = {
        n_train: {"linestyle": "--", "color": ROLE_STYLE["empirical"]["color"]},
        200000: {"linestyle": ":", "color": "#bcbd22"},
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, (tilt_id, tag) in zip(axes, _TILT_PAIR_04):
        block_mask = table_b.block(tilt_id)
        pi_tilt_block = np.asarray(table_b.pi_tilt, dtype=np.float64)[block_mask]
        order = np.argsort(pi_tilt_block)[::-1]
        rank_of_row = np.empty_like(order)
        rank_of_row[order] = np.arange(order.shape[0])
        ranks = np.arange(1, pi_tilt_block.shape[0] + 1)

        declared_cum = np.cumsum(pi_tilt_block[order])
        ax.plot(
            ranks, declared_cum, color=ROLE_STYLE["exact_target"]["color"],
            linewidth=1.5, label="declared pi_tilt (exact, Table B)",
        )

        pU_direct = float(np.sum(pi_tilt_block * table_a.B_toy))
        report_lines = ["{} ({}): target B_toy prevalence (pU_direct) = {:.6f}".format(tilt_id, tag, pU_direct)]

        for budget in budgets:
            sampler = AliasSampler.from_probabilities(pi_tilt_block)
            drawn = sampler.draw(int(budget), seed=_DIRECT_SAMPLER_SEED)
            counts = np.bincount(rank_of_row[drawn], minlength=pi_tilt_block.shape[0])
            emp_cum = np.cumsum(counts) / float(budget)
            style = budget_styles[budget]
            ax.plot(
                ranks, emp_cum, linestyle=style["linestyle"], color=style["color"], linewidth=1.2,
                label="empirical cumulative (T={})".format(budget),
            )

            diag = empirical_draw_diagnostics(drawn, b_toy=table_a.B_toy)
            concentration = theoretical_concentration_diagnostics(pi_tilt_block, draw_budget=int(budget))
            report_lines.append(
                "T={}: empirical B_toy fraction={:.6f}, expected_unique={:.1f}, "
                "observed_unique={}, max_reuse={}".format(
                    budget, diag["empirical_b_toy_fraction"], concentration["expected_n_unique"],
                    diag["unique_rows_drawn"], diag["max_reuse_count"],
                )
            )

        ax.set_xlabel("rank (rows ordered by declared pi_tilt, descending)")
        ax.set_ylabel("cumulative probability mass")
        # The per-panel numeric report lives in the title (not an in-axes text
        # box) so it never overlaps the legend or the plotted curves -- both
        # the mild and strong tilt curves occupy most of the plot area,
        # leaving no reliably empty corner for an inline annotation.
        ax.set_title("\n".join(report_lines), fontsize=7, loc="left")
        ax.legend(loc="lower right", fontsize=7)
        # Shared, explicit y-limits (finding O5): cumulative probability mass
        # is inherently bounded in [0, 1] in both panels -- without this,
        # matplotlib's independent autoscaling gives each panel a different
        # range, making the two curves harder to compare at a glance.
        ax.set_ylim(-0.03, 1.05)

    fig.suptitle(
        "Direct alias-sampler validation against the declared pi_tilt table "
        "(PDG 13 only -- a sampler-implementation check, not a physics claim; "
        "PDG-independence is not the point of this figure)",
        fontsize=11, wrap=True,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save_png_and_pdf(fig, out_dir, "04_direct_sampler_validation")


# --- Figure 05 -------------------------------------------------------------


def _sorted_arena_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = list(payload.get("rows", []))

    def key(r: Dict[str, Any]):
        vid = r.get("variant_id")
        return (0, "") if vid == "NOMINAL_PHYSICAL" else (1, str(vid))

    return sorted(rows, key=key)


def plot_05_nf_preserves_tilt(
    arena_reports: Dict[Any, Dict[str, Any]], out_dir: Path, *, caveat: Optional[str] = None
) -> List[str]:
    """P0 vs exact target P_U vs NF-generated occupancy Q_theta, per trained variant."""

    plt = _figure()
    caveat_text = caveat if caveat else _DEFAULT_NF_CAVEAT

    pdg_ids = sorted(arena_reports.keys(), key=lambda k: (abs(int(k)), int(k)))
    n = max(len(pdg_ids), 1)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 7), squeeze=False)
    axes = axes[0]

    def _safe(vals):
        # None -> NaN, never 0.0: a missing/untrained value must never be
        # visually indistinguishable from a real, measured zero occupancy
        # (review round 1, finding I6).
        return [float(v) if isinstance(v, (int, float)) else float("nan") for v in vals]

    for ax, pdg_id in zip(axes, pdg_ids):
        payload = arena_reports[pdg_id]
        rows = _sorted_arena_rows(payload)
        labels = [str(r.get("variant_id")) for r in rows]
        x = np.arange(len(labels))
        width = 0.26

        p0_vals = _safe([r.get("nominal_b_toy_prevalence") for r in rows])
        pu_vals = _safe([r.get("theoretical_target_b_toy_prevalence") for r in rows])
        q_vals = _safe([r.get("generated_b_toy_occupancy") for r in rows])
        n_samples = [r.get("generated_sample_count") for r in rows]

        # Binomial standard error on each Q_theta bar (finding I2): occupancy
        # is a fraction of generated_sample_count independent flow samples,
        # so its own sampling uncertainty is directly computable and is not
        # negligible at the ~4000-sample budget used here, especially for
        # the mild tilts.
        q_err = [
            float(np.sqrt(q * (1.0 - q) / n)) if (np.isfinite(q) and isinstance(n, (int, float)) and n > 0)
            else 0.0
            for q, n in zip(q_vals, n_samples)
        ]

        for vals, offset, role, label in (
            (p0_vals, -width, "nominal", "P0: nominal_b_toy_prevalence"),
            (pu_vals, 0.0, "exact_target", "P_U: theoretical_target_b_toy_prevalence (exact)"),
        ):
            heights = np.nan_to_num(vals, nan=0.0)
            ax.bar(x + offset, heights, width, color=ROLE_STYLE[role]["color"], label=label)
            for xi, v in zip(x, vals):
                if not np.isfinite(v):
                    ax.text(xi + offset, 0.0, "N/A", fontsize=6, color="dimgray", ha="center", va="bottom", rotation=90)

        q_heights = np.nan_to_num(q_vals, nan=0.0)
        ax.bar(
            x + width, q_heights, width, yerr=q_err, capsize=3,
            color=ROLE_STYLE["nf"]["color"], hatch=ROLE_STYLE["nf"]["hatch"], edgecolor="black",
            label="Q_theta: generated_b_toy_occupancy (NF, empirical) +/- 1 binomial SE",
        )
        for xi, v in zip(x, q_vals):
            if not np.isfinite(v):
                ax.text(xi + width, 0.0, "N/A", fontsize=6, color="dimgray", ha="center", va="bottom", rotation=90)

        # symlog (not plain log): occupancy values legitimately span from
        # ~0.003 up to ~1.0, and a plain log axis cannot represent bars that
        # start at 0 -- symlog keeps mild-tilt bars visible instead of
        # degenerating to invisible slivers under a linear axis (finding I3).
        ax.set_yscale("symlog", linthresh=0.01)
        ax.set_ylim(0, 1.35)

        # A single compact summary line (not one floating annotation per bar)
        # avoids the top-of-panel text collisions the original per-bar
        # annotations produced when two or more bars reached similar heights
        # (finding O6) -- still clearly separate from the occupancy bars, per
        # the suptitle's "non-bar diagnostic" statement below.
        nll_parts = [
            "{}={:.2f}".format(str(r.get("variant_id")), r["final_train_nll"])
            for r in rows if isinstance(r.get("final_train_nll"), (int, float))
        ]
        if nll_parts:
            ax.text(
                0.5, 1.10, "final_train_nll: " + ", ".join(nll_parts),
                transform=ax.transAxes, fontsize=5.5, color="dimgray", ha="center", va="bottom",
                wrap=True,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("B_toy occupancy / prevalence (synthetic, symlog scale)")
        ax.set_title("PDG {}".format(pdg_id))
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)

    fig.suptitle(
        "NF-generated samples versus the exact declared tilt target, per "
        "trained variant (synthetic B_toy; NLL annotations are a separate, "
        "non-bar diagnostic -- not evidence of sampling agreement)",
        fontsize=11, wrap=True,
    )
    fig.text(0.5, 0.01, caveat_text, ha="center", fontsize=7, wrap=True)
    fig.tight_layout(rect=(0, 0.08, 1, 0.86))
    return _save_png_and_pdf(fig, out_dir, "05_nf_preserves_tilt")
