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
        # "status" keeps its historical meaning: technical execution status.
        # technical_status mirrors it; scientific_status is separate.
        technical_status = status.get("technical_status", status.get("status"))
        record: Dict[str, Any] = {
            "run_id": status.get("run_id", run_dir.name),
            "status": status.get("status"),
            "technical_status": technical_status,
            "scientific_status": status.get("scientific_status"),
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
                    "target_stage": config["target"].get("stage", "transformed"),
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
            record["sampling_regime"] = metrics.get("sampling_regime")
            record["diagnostic_only"] = bool(metrics.get("diagnostic_only", False))
            record["fit_claim"] = metrics.get("fit_claim")
            record["train_main_nll"] = metrics.get("train_main_nll")
            record["train_rare_nll"] = metrics.get("train_rare_nll")
            record["validation_main_nll"] = metrics.get("validation_main_nll")
            record["validation_rare_nll"] = metrics.get("validation_rare_nll")
            record["fit_wall_time_seconds"] = metrics.get("fit_wall_time_seconds")
            record["q_rare_region_mass"] = _dig(metrics, "rare_mode", "q_rare_region_mass")
            record["target_rare_mass"] = _dig(metrics, "rare_mode", "target_rare_mass")
            record["rare_region_mass_ratio"] = _dig(
                metrics, "rare_mode", "rare_region_mass_ratio"
            )
            record["observed_q_rare_sample_count"] = _dig(
                metrics, "rare_mode", "observed_q_rare_sample_count"
            )
            # Prefer the scientific status recorded in metrics.json's gate block;
            # fall back to run_status.json (kept in sync by the campaign).
            gate_block = metrics.get("scientific_gates") or {}
            if gate_block.get("scientific_status") is not None:
                record["scientific_status"] = gate_block.get("scientific_status")
            record["scientific_failure_reasons"] = gate_block.get(
                "scientific_failure_reasons", []
            )
            record["gate_config_hash"] = gate_block.get("gate_config_hash")
            record["decision_scope"] = gate_block.get("decision_scope")
        else:
            record["scientific_failure_reasons"] = []
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
        "target_label", "target_stage", "sampling_regime", "diagnostic_only",
        "pdg_id", "feature_view", "model", "device", "seed",
        "technical_status", "scientific_status",
        "forward_kl", "ess_over_n", "ess_catastrophic", "c2st_accuracy",
        "held_out_nll", "parameter_count", "fit_wall_time_seconds",
        "q_rare_region_mass", "target_rare_mass", "rare_region_mass_ratio",
        "observed_q_rare_sample_count", "train_main_nll", "train_rare_nll",
        "validation_main_nll", "validation_rare_nll",
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
            r.get("model"), r.get("device"), r.get("sampling_regime"),
            r.get("diagnostic_only"), r.get("target_stage"),
        )
        groups[key].append(r)
    aggregate = []
    for (target_label, pdg_id, view, model, device, sampling_regime, diagnostic_only, target_stage), rows in sorted(
        groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])
    ):
        # Scientifically non-catastrophic rows only. A catastrophic run (e.g.
        # ESS/N collapse or D5 zero-rare) must never be averaged together with a
        # passing run into an unqualified mean: the "clean" means below are
        # explicitly pass/inconclusive-only, and catastrophic runs stay counted
        # and visible.
        clean_rows = [r for r in rows if r.get("scientific_status") != "catastrophic"]
        sci_counts: Dict[str, int] = defaultdict(int)
        for r in rows:
            sci_counts[r.get("scientific_status") or "unknown"] += 1
        aggregate.append(
            {
                "target_label": target_label,
                "pdg_id": pdg_id,
                "feature_view": view,
                "model": model,
                "device": device,
                "sampling_regime": sampling_regime,
                "diagnostic_only": diagnostic_only,
                "target_stage": target_stage,
                "fit_claim": (
                    "diagnostic_only_not_a_fit_to_original_target_density"
                    if diagnostic_only else "original_target_density"
                ),
                "n_seeds": len(rows),
                "scientific_status_counts": dict(sci_counts),
                "n_scientific_catastrophic": sci_counts.get("catastrophic", 0),
                "any_scientific_catastrophic": sci_counts.get("catastrophic", 0) > 0,
                # Means are computed over non-catastrophic runs only (qualified).
                "forward_kl_mean_noncatastrophic": _mean(clean_rows, "forward_kl"),
                "forward_kl_std_noncatastrophic": _std(clean_rows, "forward_kl"),
                "ess_over_n_mean_noncatastrophic": _mean(clean_rows, "ess_over_n"),
                "ess_over_n_std_noncatastrophic": _std(clean_rows, "ess_over_n"),
                "c2st_accuracy_mean_noncatastrophic": _mean(clean_rows, "c2st_accuracy"),
                "any_catastrophic": any(bool(r.get("ess_catastrophic")) for r in rows),
            }
        )
    summary = {
        "n_runs": len(records),
        "n_completed": sum(1 for r in records if r.get("status") == "completed"),
        "n_failed": sum(1 for r in records if r.get("status") == "failed"),
        "scientific_status_counts": _scientific_status_counts(records),
        "aggregate": aggregate,
    }
    (out_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # Markdown
    lines = ["# Controlled Density Lab — Benchmark Summary", ""]
    lines.append("Runs: {} completed, {} failed.".format(summary["n_completed"], summary["n_failed"]))
    lines.append("")
    lines.append(
        "Scientific status counts: {}".format(
            _fmt_counts(summary["scientific_status_counts"])
        )
    )
    lines.append("")
    lines.append(
        "Means below are computed over scientifically non-catastrophic runs only; "
        "catastrophic runs are counted separately and never averaged in."
    )
    lines.append("")
    lines.append(
        "| target | pdg | view | model | device | seeds | fKL (mean, non-cat) | "
        "ESS/N (mean, non-cat) | C2ST acc (non-cat) | sci catastrophic |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in aggregate:
        lines.append(
            "| {target_label} | {pdg_id} | {feature_view} | {model} | {device} | {n_seeds} | "
            "{fkl} | {ess} | {c2st} | {cat} |".format(
                fkl=_fmt(row["forward_kl_mean_noncatastrophic"]),
                ess=_fmt(row["ess_over_n_mean_noncatastrophic"]),
                c2st=_fmt(row["c2st_accuracy_mean_noncatastrophic"]),
                cat=row["n_scientific_catastrophic"],
                **row,
            )
        )
    (out_dir / "benchmark_summary.md").write_text("\n".join(lines) + "\n")
    return summary


def _scientific_status_counts(records) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for r in records:
        status = r.get("scientific_status")
        if status is None:
            continue
        counts[status] += 1
    return dict(counts)


def _fmt_counts(counts: Dict[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join("{}={}".format(k, counts[k]) for k in sorted(counts))


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


def build_scientific_gate_summary(
    records: List[Dict[str, Any]], out_dir: Path
) -> Dict[str, Any]:
    """Write scientific_gate_summary.{json,md} and return the summary payload.

    Distinguishes technical execution status from scientific status, counts runs
    by scientific status, enumerates catastrophic reasons and D5 zero-rare cases,
    and keeps every failed/catastrophic run visible.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    status_counts: Dict[str, int] = defaultdict(int)
    technical_counts: Dict[str, int] = defaultdict(int)
    catastrophic_runs: List[Dict[str, Any]] = []
    inconclusive_runs: List[Dict[str, Any]] = []
    d5_zero_rare_runs: List[Dict[str, Any]] = []
    per_run: List[Dict[str, Any]] = []

    for r in records:
        tech = r.get("technical_status") or r.get("status")
        sci = r.get("scientific_status")
        technical_counts[tech or "unknown"] += 1
        if sci is not None:
            status_counts[sci] += 1
        reasons = r.get("scientific_failure_reasons") or []
        row = {
            "run_id": r.get("run_id"),
            "target_label": r.get("target_label"),
            "model": r.get("model"),
            "seed": r.get("seed"),
            "sampling_regime": r.get("sampling_regime"),
            "diagnostic_only": r.get("diagnostic_only", False),
            "device": r.get("device"),
            "technical_status": tech,
            "scientific_status": sci,
            "ess_over_n": r.get("ess_over_n"),
            "observed_q_rare_sample_count": r.get("observed_q_rare_sample_count"),
            "rare_region_mass_ratio": r.get("rare_region_mass_ratio"),
            "scientific_failure_reasons": reasons,
        }
        per_run.append(row)
        if sci == "catastrophic":
            catastrophic_runs.append(row)
        if sci == "inconclusive":
            inconclusive_runs.append(row)
        if r.get("observed_q_rare_sample_count") == 0 and r.get("target_label", "").startswith("D5"):
            d5_zero_rare_runs.append(row)

    scopes = sorted({r.get("decision_scope") for r in records if r.get("decision_scope")})
    summary = {
        "n_runs": len(records),
        "decision_scope": scopes[0] if len(scopes) == 1 else scopes,
        "decision_scope_meaning": (
            "scientific_status='pass' means only that all currently active gates "
            "passed. It does not assert sufficient rare-mode fidelity, a validated "
            "minimum capacity, or final scientific acceptance."
        ),
        "technical_status_counts": dict(technical_counts),
        "scientific_status_counts": dict(status_counts),
        "n_catastrophic": len(catastrophic_runs),
        "n_inconclusive": len(inconclusive_runs),
        "n_d5_zero_rare": len(d5_zero_rare_runs),
        "catastrophic_runs": catastrophic_runs,
        "inconclusive_runs": inconclusive_runs,
        "d5_zero_rare_runs": d5_zero_rare_runs,
        "runs": per_run,
    }
    (out_dir / "scientific_gate_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )

    lines = ["# Scientific Gate Summary", ""]
    lines.append(
        "Technical execution status and scientific status are reported "
        "separately. A technically completed run can still be scientifically "
        "`catastrophic` or `inconclusive`; it is never relabeled a technical "
        "failure for failing a scientific gate."
    )
    lines.append("")
    lines.append(
        "**Decision scope (`{}`).** `scientific_status = pass` means **only** that "
        "all currently active gates passed. It does **not** mean sufficient "
        "rare-mode fidelity, validated minimum capacity, or final scientific "
        "acceptance. The rare-region mass ratio is report-only under this scope.".format(
            summary["decision_scope"] or "unknown"
        )
    )
    lines.append("")
    lines.append("Technical status counts: {}".format(_fmt_counts(dict(technical_counts))))
    lines.append("Scientific status counts: {}".format(_fmt_counts(dict(status_counts))))
    lines.append("")
    lines.append(
        "| run | target | model | seed | technical | scientific | ESS/N | rare count | reasons |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in per_run:
        reason_text = "; ".join(
            "{}:{}".format(x.get("threshold_class"), x.get("gate_id"))
            for x in row["scientific_failure_reasons"]
        ) or "—"
        lines.append(
            "| {run} | {target} | {model} | {seed} | {tech} | {sci} | {ess} | {rare} | {reasons} |".format(
                run=row["run_id"],
                target=row["target_label"],
                model=row["model"],
                seed=row["seed"],
                tech=row["technical_status"],
                sci=row["scientific_status"],
                ess=_fmt(row["ess_over_n"]),
                rare=row["observed_q_rare_sample_count"],
                reasons=reason_text,
            )
        )
    lines.append("")
    if catastrophic_runs:
        lines.append("## Catastrophic runs (kept visible, never averaged into clean means)")
        lines.append("")
        for row in catastrophic_runs:
            codes = ", ".join(
                "{} [{}]".format(x.get("gate_id"), x.get("threshold_class"))
                for x in row["scientific_failure_reasons"]
            )
            lines.append("- `{}` ({}): {}".format(row["run_id"], row["model"], codes))
        lines.append("")
    if d5_zero_rare_runs:
        lines.append("## D5 zero-rare-sample cases")
        lines.append("")
        for row in d5_zero_rare_runs:
            lines.append(
                "- `{}` ({}): generated 0 rare-region samples".format(
                    row["run_id"], row["model"]
                )
            )
        lines.append("")
    lines.append(
        "Threshold classes: `mathematical_invariant`, `catastrophic_guard`, "
        "`provisional_engineering_gate`, `preregistered_scientific_gate`. "
        "Provisional engineering thresholds are working references, not final "
        "physics criteria."
    )
    (out_dir / "scientific_gate_summary.md").write_text("\n".join(lines) + "\n")
    return summary


def write_limitations(records, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    n_failed = sum(1 for r in records if r.get("status") == "failed")
    n_cat = sum(1 for r in records if r.get("ess_catastrophic"))
    n_sci_cat = sum(1 for r in records if r.get("scientific_status") == "catastrophic")
    text = [
        "# Limitations and Non-goals",
        "",
        "These are numerical controlled benchmarks (D0-D5), not SHiP physics.",
        "",
        "- No SHiP background rate, simulation speed-up, or FairShip/GEANT4 claim.",
        "- No energy variable; no event-level four-momentum conservation.",
        "- pz is preserved by every target transform; no feature view is privileged.",
        "- Metrics are Monte-Carlo estimates with finite-sample uncertainty.",
        "- {} run(s) failed technically and are shown in the summary tables, not hidden.".format(n_failed),
        "- {} run(s) had catastrophic ESS/N (< 0.01); reported explicitly.".format(n_cat),
        "- {} run(s) are scientifically catastrophic; see scientific_gate_summary.md.".format(n_sci_cat),
        "- Technical status (completed/failed) is separate from scientific status",
        "  (pass/fail/catastrophic/inconclusive): a completed run may be catastrophic.",
        "- Wall times mix CPU/GPU only within, never across, a single curve.",
        "- Provisional engineering gates are not preregistered physics criteria.",
        "- stratified_unweighted_diagnostic rows are diagnostic-only and are never a fit claim for the original target density.",
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
    gate_summary = build_scientific_gate_summary(records, out_dir)
    summary["scientific_gate_summary"] = gate_summary
    write_limitations(records, out_dir)
    plot_paths: List[str] = []
    if with_plots:
        try:
            plot_paths = build_plots(records, out_dir)
        except Exception as exc:  # matplotlib missing / headless issue
            (out_dir / "plots_error.txt").write_text(str(exc))
    return {"summary": summary, "plots": plot_paths, "report_dir": str(out_dir)}
