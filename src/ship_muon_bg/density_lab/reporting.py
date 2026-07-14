"""Report builder for a completed campaign directory (reads artifacts only).

Never retrains a model. Loads per-run ``metrics.json`` / ``run_status.json`` /
``experiment_config.json``, aggregates them (mean +/- std across seeds), and
writes summary tables (JSON/CSV/Markdown), a limitations note, and plots. Plots
use matplotlib (optional ``lab`` dependency, imported lazily) and never smooth
away or silently drop failed/catastrophic runs. No plot claims a SHiP
simulation speed-up.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

TARGET_ORDER = ["D0", "D1", "D2", "D3", "D4", "D5"]


def _target_label(target_id: str, variant: Optional[str]) -> str:
    if target_id == "D5" and variant:
        return "D5-{}".format(variant.replace("rare_", ""))
    return target_id


def load_run_records(campaign_dir: Path) -> List[Dict[str, Any]]:
    """Load one flat record per run directory (including failed runs)."""

    campaign_dir = Path(campaign_dir)
    records: List[Dict[str, Any]] = []
    for run_dir in sorted(p for p in campaign_dir.iterdir() if p.is_dir()):
        status_path = run_dir / "run_status.json"
        if not status_path.exists():
            continue
        status = json.loads(status_path.read_text())
        config = _maybe_json(run_dir / "experiment_config.json")
        metrics = _maybe_json(run_dir / "metrics.json")
        record: Dict[str, Any] = {
            "run_id": status.get("run_id", run_dir.name),
            "status": status.get("status"),
            "run_dir": str(run_dir),
        }
        if config:
            record.update(
                {
                    "target_id": config["target"]["target_id"],
                    "variant": config["target"].get("variant"),
                    "pdg_id": config["pdg_id"],
                    "feature_view": config["feature_view"]["view_id"],
                    "model": config["model"]["name"],
                    "seed": config["seed"],
                    "device": config.get("device"),
                }
            )
            record["target_label"] = _target_label(
                record["target_id"], record.get("variant")
            )
        if metrics and record["status"] == "completed":
            record["forward_kl"] = _dig(metrics, "forward_kl", "forward_kl")
            record["held_out_nll"] = _dig(metrics, "held_out", "held_out_nll")
            record["ess_over_n"] = _dig(metrics, "importance_ess", "ess_over_n")
            record["ess_catastrophic"] = _dig(metrics, "importance_ess", "catastrophic")
            record["c2st_accuracy"] = _dig(metrics, "c2st", "c2st_accuracy")
            record["parameter_count"] = metrics.get("parameter_count")
            record["fit_wall_time_seconds"] = metrics.get("fit_wall_time_seconds")
            record["q_rare_region_mass"] = _dig(metrics, "rare_mode", "q_rare_region_mass")
            record["target_rare_mass"] = _dig(metrics, "rare_mode", "target_rare_mass")
            record["observed_q_rare_sample_count"] = _dig(
                metrics, "rare_mode", "observed_q_rare_sample_count"
            )
        records.append(record)
    return records


def _maybe_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _dig(payload, *keys):
    node = payload
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def build_summary_tables(records: List[Dict[str, Any]], out_dir: Path) -> Dict[str, Any]:
    """Write benchmark_summary.{json,csv,md} and return the aggregated summary."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "target_label", "pdg_id", "feature_view", "model", "device", "seed",
        "status", "forward_kl", "ess_over_n", "ess_catastrophic", "c2st_accuracy",
        "held_out_nll", "parameter_count", "fit_wall_time_seconds",
        "q_rare_region_mass", "target_rare_mass",
    ]
    # CSV (one row per run, failed runs included)
    with (out_dir / "benchmark_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({c: record.get(c) for c in columns})

    # JSON aggregate (mean/std across seeds per target/view/model/pdg)
    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        if r.get("status") != "completed":
            continue
        # Device is part of the key: runs differing only by backend must not be
        # averaged together (backend-dependent metrics) or counted as extra
        # seeds when a config is run once on CPU and once on CUDA/auto.
        key = (
            r.get("target_label"), r.get("pdg_id"), r.get("feature_view"),
            r.get("model"), r.get("device"),
        )
        groups[key].append(r)
    aggregate = []
    for (target_label, pdg_id, view, model, device), rows in sorted(
        groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])
    ):
        aggregate.append(
            {
                "target_label": target_label,
                "pdg_id": pdg_id,
                "feature_view": view,
                "model": model,
                "device": device,
                "n_seeds": len(rows),
                "forward_kl_mean": _mean(rows, "forward_kl"),
                "forward_kl_std": _std(rows, "forward_kl"),
                "ess_over_n_mean": _mean(rows, "ess_over_n"),
                "ess_over_n_std": _std(rows, "ess_over_n"),
                "c2st_accuracy_mean": _mean(rows, "c2st_accuracy"),
                "any_catastrophic": any(bool(r.get("ess_catastrophic")) for r in rows),
            }
        )
    summary = {
        "n_runs": len(records),
        "n_completed": sum(1 for r in records if r.get("status") == "completed"),
        "n_failed": sum(1 for r in records if r.get("status") == "failed"),
        "aggregate": aggregate,
    }
    (out_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # Markdown
    lines = ["# Controlled Density Lab — Benchmark Summary", ""]
    lines.append("Runs: {} completed, {} failed.".format(summary["n_completed"], summary["n_failed"]))
    lines.append("")
    lines.append("| target | pdg | view | model | device | seeds | fKL (mean) | ESS/N (mean) | C2ST acc | catastrophic |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in aggregate:
        lines.append(
            "| {target_label} | {pdg_id} | {feature_view} | {model} | {device} | {n_seeds} | "
            "{fkl} | {ess} | {c2st} | {cat} |".format(
                fkl=_fmt(row["forward_kl_mean"]),
                ess=_fmt(row["ess_over_n_mean"]),
                c2st=_fmt(row["c2st_accuracy_mean"]),
                cat="yes" if row["any_catastrophic"] else "no",
                **row,
            )
        )
    (out_dir / "benchmark_summary.md").write_text("\n".join(lines) + "\n")
    return summary


def _values(rows, key):
    return [r[key] for r in rows if isinstance(r.get(key), (int, float))]


def _mean(rows, key):
    vals = _values(rows, key)
    return float(np.mean(vals)) if vals else None


def _std(rows, key):
    vals = _values(rows, key)
    return float(np.std(vals)) if len(vals) > 1 else 0.0


def _fmt(value):
    return "n/a" if value is None else "{:.4g}".format(value)


def write_limitations(records, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    n_failed = sum(1 for r in records if r.get("status") == "failed")
    n_cat = sum(1 for r in records if r.get("ess_catastrophic"))
    text = [
        "# Limitations and Non-goals",
        "",
        "These are numerical controlled benchmarks (D0-D5), not SHiP physics.",
        "",
        "- No SHiP background rate, simulation speed-up, or FairShip/GEANT4 claim.",
        "- No energy variable; no event-level four-momentum conservation.",
        "- pz is preserved by every target transform; no feature view is privileged.",
        "- Metrics are Monte-Carlo estimates with finite-sample uncertainty.",
        "- {} run(s) failed and are shown in the summary tables, not hidden.".format(n_failed),
        "- {} run(s) had catastrophic ESS/N (< 0.01); reported explicitly.".format(n_cat),
        "- Wall times mix CPU/GPU only within, never across, a single curve.",
        "- Provisional engineering gates are not preregistered physics criteria.",
    ]
    (out_dir / "limitations.md").write_text("\n".join(text) + "\n")


# --- plots ------------------------------------------------------------------


def _figure():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def build_plots(records, out_dir: Path) -> List[str]:
    plt = _figure()
    out_dir = Path(out_dir)
    completed = [r for r in records if r.get("status") == "completed"]
    written: List[str] = []

    # 1. quality_by_target: forward KL and ESS/N by target, stratified by pdg.
    written += _plot_quality_by_target(plt, completed, out_dir)
    # 2. rare_mode_recovery
    written += _plot_rare_mode(plt, completed, out_dir)
    # 3. feature_view_comparison
    written += _plot_feature_views(plt, completed, out_dir)
    # 4. capacity_frontier
    written += _plot_capacity(plt, completed, out_dir)
    return written


def _plot_quality_by_target(plt, rows, out_dir):
    if not rows:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for metric, ax, title in (
        ("forward_kl", axes[0], "Forward KL (lower better)"),
        ("ess_over_n", axes[1], "ESS/N (higher better)"),
    ):
        for pdg in sorted({r["pdg_id"] for r in rows}):
            labels, means, stds = _by_target(rows, metric, pdg)
            ax.errorbar(labels, means, yerr=stds, marker="o", capsize=3, label="pdg {}".format(pdg))
        ax.set_title(title)
        ax.set_xlabel("target")
        ax.set_ylabel(metric)
        ax.legend()
        if metric == "forward_kl":
            ax.set_yscale("symlog")
    fig.suptitle("Quality by target (exact controlled benchmarks; not SHiP physics)")
    fig.tight_layout()
    path = out_dir / "quality_by_target.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return [str(path)]


def _by_target(rows, metric, pdg):
    grouped = defaultdict(list)
    for r in rows:
        if r["pdg_id"] == pdg and isinstance(r.get(metric), (int, float)):
            grouped[r["target_label"]].append(r[metric])
    labels = sorted(grouped, key=lambda t: (TARGET_ORDER.index(t.split("-")[0]) if t.split("-")[0] in TARGET_ORDER else 99, t))
    means = [float(np.mean(grouped[l])) for l in labels]
    stds = [float(np.std(grouped[l])) if len(grouped[l]) > 1 else 0.0 for l in labels]
    return labels, means, stds


def _plot_rare_mode(plt, rows, out_dir):
    rare = [r for r in rows if isinstance(r.get("q_rare_region_mass"), (int, float))]
    if not rare:
        return []
    fig, ax = plt.subplots(figsize=(7, 6))
    for r in rare:
        target = r.get("target_rare_mass")
        recovered = r.get("q_rare_region_mass")
        marker = "x" if r.get("observed_q_rare_sample_count") == 0 else "o"
        ax.scatter(target, recovered, marker=marker, s=60)
    lims = [1e-4, 2e-2]
    ax.plot(lims, lims, "k--", alpha=0.5, label="ideal recovery")
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1e-5)
    ax.set_xlabel("target rare mass")
    ax.set_ylabel("recovered rare-region mass (x = zero count)")
    ax.set_title("Rare-mode recovery (D5)")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "rare_mode_recovery.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return [str(path)]


def _plot_feature_views(plt, rows, out_dir):
    views = sorted({r["feature_view"] for r in rows})
    if len(views) < 2:
        return []
    fig, ax = plt.subplots(figsize=(9, 5))
    targets = sorted({r["target_label"] for r in rows}, key=lambda t: (TARGET_ORDER.index(t.split("-")[0]) if t.split("-")[0] in TARGET_ORDER else 99, t))
    width = 0.8 / len(views)
    x = np.arange(len(targets))
    for i, view in enumerate(views):
        means = []
        for t in targets:
            vals = [r["forward_kl"] for r in rows if r["feature_view"] == view and r["target_label"] == t and isinstance(r.get("forward_kl"), (int, float))]
            means.append(float(np.mean(vals)) if vals else np.nan)
        ax.bar(x + i * width, means, width, label=view)
    ax.set_xticks(x + width * (len(views) - 1) / 2)
    ax.set_xticklabels(targets)
    ax.set_ylabel("forward KL (physical space)")
    ax.set_title("Feature-view comparison (matched rows/model/seed/budget)")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "feature_view_comparison.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return [str(path)]


def _plot_capacity(plt, rows, out_dir):
    pts = [r for r in rows if isinstance(r.get("parameter_count"), (int, float)) and isinstance(r.get("ess_over_n"), (int, float))]
    if not pts:
        return []
    fig, ax = plt.subplots(figsize=(8, 5))
    for target in sorted({r["target_label"] for r in pts}):
        sub = sorted([r for r in pts if r["target_label"] == target], key=lambda r: r["parameter_count"])
        ax.plot([r["parameter_count"] for r in sub], [r["ess_over_n"] for r in sub], marker="o", label=target)
    ax.set_xscale("log")
    ax.set_xlabel("parameter count")
    ax.set_ylabel("ESS/N")
    device_note = "hardware: see environment.json per run (CPU/GPU not mixed on one curve)"
    ax.set_title("Capacity frontier\n{}".format(device_note))
    ax.legend()
    fig.tight_layout()
    path = out_dir / "capacity_frontier.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return [str(path)]


def build_report(campaign_dir: Path, *, with_plots: bool = True) -> Dict[str, Any]:
    """Build all report outputs under ``<campaign_dir>/report/``."""

    campaign_dir = Path(campaign_dir)
    out_dir = campaign_dir / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    records = load_run_records(campaign_dir)
    summary = build_summary_tables(records, out_dir)
    write_limitations(records, out_dir)
    plot_paths: List[str] = []
    if with_plots:
        try:
            plot_paths = build_plots(records, out_dir)
        except Exception as exc:  # matplotlib missing / headless issue
            (out_dir / "plots_error.txt").write_text(str(exc))
    return {"summary": summary, "plots": plot_paths, "report_dir": str(out_dir)}
