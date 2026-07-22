"""D8 plotting (§6 training curves, §11 sample matrices, §11 pz diagnostics).

Matplotlib only, no seaborn. Never smooths five points, never fabricates a
history for one-shot (Gaussian/GMM) baselines, never plots test NLL as an
epoch curve.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ship_muon_bg.afterms.d8.registry import RunRecord


def _curve_title(record: RunRecord) -> str:
    return (
        f"{record.run_id}\n"
        f"family={record.model_family} pdg={record.pdg_policy} preprocessing={record.preprocessing_name} "
        f"target={record.target_measure}"
    )


def plot_loss_curve(record: RunRecord, output_path: Path) -> Optional[Path]:
    """One `training_curves/loss_curve__<run_id>.png` per NEURAL run with a
    genuine multi-epoch history. Baselines (one-shot fits, a single history
    entry) never get a fabricated epoch curve -- returns None instead."""

    if not record.history or len(record.history) < 2:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in record.history]
    train = [h.get("train_loss") for h in record.history]
    val = [h.get("validation_loss") for h in record.history]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, train, marker="o", label="train NLL")
    ax.plot(epochs, val, marker="o", label="validation NLL")
    if record.best_validation_epoch is not None:
        ax.axvline(record.best_validation_epoch, color="green", linestyle="--", label="best validation epoch")
    if record.final_epoch is not None:
        ax.axvline(record.final_epoch, color="gray", linestyle=":", label="final epoch")
    if record.available_checkpoint_epoch is not None:
        ax.axvline(record.available_checkpoint_epoch, color="red", linestyle="-.", label="available checkpoint epoch")
    ax.set_xlabel("epoch")
    coordinate_label = "quantile feature-space (undefined physical Jacobian)" if record.preprocessing_name == "quantile_normal_v0" else "feature-space (see report for physical-space test/derived-validation NLL)"
    ax.set_ylabel(f"NLL [{coordinate_label}]")
    ax.set_title(_curve_title(record), fontsize=8)
    ax.legend(fontsize=7)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_grouped_curves(
    records: List[RunRecord], output_path: Path, title: str
) -> Optional[Path]:
    """A single figure overlaying validation-NLL curves for a comparable group
    of runs (e.g. identity vs log1p at fixed PDG). Callers are responsible for
    only grouping runs that are actually comparable (same weighting policy,
    same coordinate-space class)."""

    plottable = [r for r in records if r.history and len(r.history) >= 2]
    if not plottable:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for r in plottable:
        epochs = [h["epoch"] for h in r.history]
        val = [h.get("validation_loss") for h in r.history]
        ax.plot(epochs, val, marker="o", label=f"{r.preprocessing_name}/{r.model_capacity or r.model_family}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation NLL [feature-space, run-specific coordinates]")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def write_all_curves(records: List[RunRecord], output_dir: Path) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    written = []
    skipped = []
    for r in records:
        path = output_dir / f"loss_curve__{r.run_id}.png"
        result = plot_loss_curve(r, path)
        if result is not None:
            written.append(str(result))
        else:
            skipped.append(r.run_id)

    # Grouped: identity vs log1p by PDG, unweighted only (comparable coordinate class).
    for pdg in ("pdg13", "pdg_minus13"):
        group = [
            r for r in records
            if r.pdg_policy == pdg and not r.weighting_policy
            and r.preprocessing_name in ("identity_standardized_v0", "cartesian_log1p_pz_v0")
            and r.model_family == "affine_coupling"
        ]
        if group:
            path = output_dir / f"grouped_identity_vs_log1p__{pdg}.png"
            result = plot_grouped_curves(group, path, f"identity vs log1p, {pdg}, unweighted")
            if result is not None:
                written.append(str(result))

    # Grouped: capacity tiers (job 09), PDG+13 unweighted identity.
    capacity_group = [
        r for r in records
        if r.job_id == "09" and r.model_family == "affine_coupling"
    ]
    if capacity_group:
        path = output_dir / "grouped_capacity_tiers__pdg13.png"
        result = plot_grouped_curves(capacity_group, path, "capacity tiny/small/medium, pdg13, unweighted")
        if result is not None:
            written.append(str(result))

    # Grouped: weighted vs unweighted, in separate panels (never overlaid in one axes).
    for pdg in ("pdg13", "pdg_minus13"):
        weighted = [r for r in records if r.pdg_policy == pdg and r.weighting_policy]
        unweighted_twin = [
            r for r in records
            if r.pdg_policy == pdg and not r.weighting_policy
            and r.preprocessing_name == "identity_standardized_v0" and r.model_capacity == "small"
        ]
        if weighted or unweighted_twin:
            _write_weighted_panel(weighted, unweighted_twin, output_dir / f"grouped_weighted_panels__{pdg}.png", pdg)
            written.append(str(output_dir / f"grouped_weighted_panels__{pdg}.png"))

    return {"written": written, "skipped_no_history": skipped}


# --- §11 sample matrices + pz diagnostics -----------------------------------

_MATRIX_VARIABLES = ("x", "y", "px", "py")
_FEATURE_INDEX = {"px": 0, "py": 1, "pz": 2, "x": 3, "y": 4}
_MATRIX_PLOT_SUBSET = 4000  # deterministic plotting subset (spec §11: "deterministic plotting subset")


def plot_sample_matrix(run_id: str, reference: np.ndarray, generated: np.ndarray, output_path: Path) -> Path:
    """diagonal: held-out vs generated histogram+ECDF; lower triangle: held-out;
    upper triangle: generated; shared axis limits per variable pair."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref = reference[:_MATRIX_PLOT_SUBSET]
    gen = generated[:_MATRIX_PLOT_SUBSET]
    n_vars = len(_MATRIX_VARIABLES)
    fig, axes = plt.subplots(n_vars, n_vars, figsize=(3 * n_vars, 3 * n_vars))

    limits = {}
    for var in _MATRIX_VARIABLES:
        idx = _FEATURE_INDEX[var]
        lo = min(np.quantile(ref[:, idx], 0.001), np.quantile(gen[:, idx], 0.001))
        hi = max(np.quantile(ref[:, idx], 0.999), np.quantile(gen[:, idx], 0.999))
        limits[var] = (lo, hi)

    for i, vi in enumerate(_MATRIX_VARIABLES):
        for j, vj in enumerate(_MATRIX_VARIABLES):
            ax = axes[i, j]
            idx_i, idx_j = _FEATURE_INDEX[vi], _FEATURE_INDEX[vj]
            if i == j:
                bins = np.linspace(*limits[vi], 40)
                ax.hist(ref[:, idx_i], bins=bins, alpha=0.5, density=True, label="held-out")
                ax.hist(gen[:, idx_i], bins=bins, alpha=0.5, density=True, label="generated")
                if i == 0:
                    ax.legend(fontsize=6)
            elif i > j:  # lower triangle: held-out
                ax.hexbin(ref[:, idx_j], ref[:, idx_i], gridsize=30, cmap="Blues")
                ax.set_xlim(limits[vj])
                ax.set_ylim(limits[vi])
            else:  # upper triangle: generated
                ax.hexbin(gen[:, idx_j], gen[:, idx_i], gridsize=30, cmap="Oranges")
                ax.set_xlim(limits[vj])
                ax.set_ylim(limits[vi])
            if i == n_vars - 1:
                ax.set_xlabel(vj)
            if j == 0:
                ax.set_ylabel(vi)

    fig.suptitle(f"sample matrix: {run_id}", fontsize=8)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_pz_diagnostics(run_id: str, reference: np.ndarray, generated: np.ndarray, output_path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pz_ref = reference[:, _FEATURE_INDEX["pz"]]
    pz_gen = generated[:, _FEATURE_INDEX["pz"]]
    pt_gen = np.sqrt(generated[:, _FEATURE_INDEX["px"]] ** 2 + generated[:, _FEATURE_INDEX["py"]] ** 2)
    r_gen = np.sqrt(generated[:, _FEATURE_INDEX["x"]] ** 2 + generated[:, _FEATURE_INDEX["y"]] ** 2)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    axes[0, 0].hist(pz_ref, bins=60, alpha=0.5, density=True, label="held-out")
    axes[0, 0].hist(pz_gen, bins=60, alpha=0.5, density=True, label="generated")
    axes[0, 0].set_title("pz marginal")
    axes[0, 0].legend(fontsize=6)

    axes[0, 1].hist(pz_ref, bins=60, alpha=0.5, density=True, label="held-out")
    axes[0, 1].hist(pz_gen, bins=60, alpha=0.5, density=True, label="generated")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("pz marginal (log-y tail)")

    for label, pz in (("held-out", pz_ref), ("generated", pz_gen)):
        sorted_pz = np.sort(pz)
        survival = 1.0 - np.arange(1, len(sorted_pz) + 1) / len(sorted_pz)
        axes[0, 2].plot(sorted_pz, survival, label=label)
    axes[0, 2].set_yscale("log")
    axes[0, 2].set_title("pz survival function")
    axes[0, 2].legend(fontsize=6)

    axes[1, 0].scatter(pz_gen, pt_gen, s=2, alpha=0.3)
    axes[1, 0].set_xlabel("pz (generated)")
    axes[1, 0].set_ylabel("pt (generated)")
    axes[1, 0].set_title("generated pz vs pt")

    axes[1, 1].scatter(pz_gen, r_gen, s=2, alpha=0.3)
    axes[1, 1].set_xlabel("pz (generated)")
    axes[1, 1].set_ylabel("transverse radius (generated)")
    axes[1, 1].set_title("generated pz vs transverse radius")

    neg_ref = pz_ref[pz_ref < 0]
    neg_gen = pz_gen[pz_gen < 0]
    axes[1, 2].hist(neg_ref, bins=30, alpha=0.5, density=True, label=f"held-out (n={neg_ref.size})")
    axes[1, 2].hist(neg_gen, bins=30, alpha=0.5, density=True, label=f"generated (n={neg_gen.size})")
    axes[1, 2].set_title("negative-pz diagnostic (never clipped)")
    axes[1, 2].legend(fontsize=6)

    fig.suptitle(f"pz diagnostics: {run_id}", fontsize=8)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _write_weighted_panel(weighted: List[RunRecord], unweighted: List[RunRecord], output_path: Path, pdg: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=False)
    for ax, group, label in ((axes[0], unweighted, "unweighted"), (axes[1], weighted, "weighted")):
        for r in group:
            if not r.history:
                continue
            epochs = [h["epoch"] for h in r.history]
            val = [h.get("validation_loss") for h in r.history]
            ax.plot(epochs, val, marker="o", label=r.run_id.split("__")[-1])
        ax.set_title(f"{label} ({pdg})")
        ax.set_xlabel("epoch")
        ax.set_ylabel("validation NLL")
        ax.legend(fontsize=6)
    fig.suptitle(f"weighted vs unweighted, separated panels, {pdg}")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
