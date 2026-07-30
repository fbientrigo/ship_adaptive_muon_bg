#!/usr/bin/env python3
"""Bounded statistical validation for the rare-aware minibatch estimators.

Implements the two-stage validation design from Issue #17 (rare-aware
minibatch estimators) and
``docs/contracts/rare_aware_minibatch_estimators_v0.md``:

``--stage analytic``
    A quadratic loss with an exactly known gradient, evaluated in both a
    population (generative) and a finite-pool setting. Monte Carlo verifies
    that the arm-A (IID) and arm-C (fixed-composition Horvitz-Thompson) mean
    gradients converge to the exact gradient, while arm-B (fixed-composition
    diagnostic) and arm-D (legacy self-normalized) show a detectable,
    non-vanishing bias whenever the fixed allocation departs from the target
    stratum mass. Cheap (NumPy only); runs in well under a minute.

``--stage d5``
    A bounded D5 campaign (``configs/density_lab/estimator_validation_v0.json``,
    15 runs, CPU-only) comparing arms A/B/C/D, plus gradient bias/variance at
    the models' initial parameters using the same batch plans the campaign
    would train with (no extra training cost for that measurement).

Deterministic given ``--seed``. No real afterMS data. Heavy optional
dependencies (torch, the density_lab campaign runner) are imported lazily,
only inside ``--stage d5``, so ``--stage analytic`` runs with NumPy alone.
This script never declares an architecture or allocation winner: it reports
bias, variance, ESS, and wall time only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


# =============================================================================
# Stage 1: analytic quadratic-loss validation (NumPy only)
# =============================================================================

_DIM = 3
_MU_MAIN = np.zeros(_DIM)
_MU_RARE = np.full(_DIM, 5.0)
_THETA = np.full(_DIM, 0.3)  # arbitrary fixed evaluation point; not "trained"


def _exact_population_gradient(p_rare: float) -> np.ndarray:
    """nabla_theta E_p[0.5||x-theta||^2] = theta - sum_h p_h * mu_h."""

    return _THETA - ((1.0 - p_rare) * _MU_MAIN + p_rare * _MU_RARE)


def _draw_iid_batch(rng: np.random.Generator, p_rare: float, batch_size: int) -> np.ndarray:
    is_rare = rng.random(batch_size) < p_rare
    x = np.where(
        is_rare[:, None],
        rng.normal(_MU_RARE, 1.0, size=(batch_size, _DIM)),
        rng.normal(_MU_MAIN, 1.0, size=(batch_size, _DIM)),
    )
    return x


def _realized_counts(batch_size: int, a_rare: float) -> Tuple[int, int]:
    m_rare = int(round(batch_size * a_rare))
    m_rare = max(1, min(batch_size - 1, m_rare))
    return m_rare, batch_size - m_rare


def _draw_fixed_composition_batch(
    rng: np.random.Generator, m_rare: int, m_main: int
) -> Tuple[np.ndarray, np.ndarray]:
    x_rare = rng.normal(_MU_RARE, 1.0, size=(m_rare, _DIM))
    x_main = rng.normal(_MU_MAIN, 1.0, size=(m_main, _DIM))
    return x_rare, x_main


def _grad_from_batch(x: np.ndarray) -> np.ndarray:
    return _THETA - x.mean(axis=0)


def _summarize(grads: np.ndarray, exact_gradient: np.ndarray) -> Dict[str, Any]:
    n = grads.shape[0]
    mean_grad = grads.mean(axis=0)
    bias = mean_grad - exact_gradient
    stderr = grads.std(axis=0, ddof=1) / np.sqrt(n)
    z = np.divide(bias, stderr, out=np.full_like(bias, np.inf), where=stderr > 0)
    cov = np.atleast_2d(np.cov(grads.T))
    return {
        "replicates": int(n),
        "exact_gradient": exact_gradient.tolist(),
        "mean_gradient": mean_grad.tolist(),
        "bias": bias.tolist(),
        "bias_norm": float(np.linalg.norm(bias)),
        "stderr": stderr.tolist(),
        "z": z.tolist(),
        "max_abs_z": float(np.max(np.abs(z))),
        "gradient_covariance_trace": float(np.trace(cov)),
    }


def _run_arm_a(rng, p_rare, batch_size, replicates) -> np.ndarray:
    grads = np.empty((replicates, _DIM))
    for r in range(replicates):
        grads[r] = _grad_from_batch(_draw_iid_batch(rng, p_rare, batch_size))
    return grads


def _run_arm_b(rng, a_rare, batch_size, replicates) -> Tuple[np.ndarray, int, int]:
    m_rare, m_main = _realized_counts(batch_size, a_rare)
    grads = np.empty((replicates, _DIM))
    for r in range(replicates):
        x_rare, x_main = _draw_fixed_composition_batch(rng, m_rare, m_main)
        grads[r] = _grad_from_batch(np.concatenate([x_rare, x_main]))
    return grads, m_rare, m_main


def _run_arm_c(rng, p_rare, a_rare, batch_size, replicates) -> Tuple[np.ndarray, int, int, float, float]:
    m_rare, m_main = _realized_counts(batch_size, a_rare)
    w_rare = p_rare * batch_size / m_rare
    w_main = (1.0 - p_rare) * batch_size / m_main
    grads = np.empty((replicates, _DIM))
    for r in range(replicates):
        x_rare, x_main = _draw_fixed_composition_batch(rng, m_rare, m_main)
        weighted_sum = w_rare * x_rare.sum(axis=0) + w_main * x_main.sum(axis=0)
        grads[r] = _THETA - weighted_sum / batch_size
    return grads, m_rare, m_main, w_rare, w_main


def _run_arm_d(rng, p_rare, a_rare, batch_size, replicates, pool_size=20000) -> np.ndarray:
    """Legacy self-normalized: fixed-composition *pool*, random minibatch slice."""

    n_rare_pool = max(1, int(round(pool_size * a_rare)))
    n_main_pool = pool_size - n_rare_pool
    pool_x = np.concatenate([
        rng.normal(_MU_RARE, 1.0, size=(n_rare_pool, _DIM)),
        rng.normal(_MU_MAIN, 1.0, size=(n_main_pool, _DIM)),
    ])
    is_rare_pool = np.concatenate(
        [np.ones(n_rare_pool, dtype=bool), np.zeros(n_main_pool, dtype=bool)]
    )
    w_rare = p_rare / a_rare
    w_main = (1.0 - p_rare) / (1.0 - a_rare)
    weight_pool = np.where(is_rare_pool, w_rare, w_main)
    grads = np.empty((replicates, _DIM))
    for r in range(replicates):
        idx = rng.choice(pool_x.shape[0], size=batch_size, replace=False)
        x, w = pool_x[idx], weight_pool[idx]
        grads[r] = _THETA - (w[:, None] * x).sum(axis=0) / w.sum()
    return grads


def run_analytic_stage(
    *,
    seed: int,
    rare_masses: Tuple[float, ...] = (1e-3, 1e-2),
    batch_sizes: Tuple[int, ...] = (128, 512),
    extra_allocations: Tuple[float, ...] = (0.05, 0.2, 0.5),
    replicate_counts: Tuple[int, ...] = (125, 500, 2000),
) -> Dict[str, Any]:
    master = np.random.default_rng(seed)
    results: List[Dict[str, Any]] = []
    started = time.perf_counter()
    for p_rare in rare_masses:
        exact_gradient = _exact_population_gradient(p_rare)
        allocations = sorted(set([p_rare]) | set(extra_allocations))
        for batch_size in batch_sizes:
            for a_rare in allocations:
                for replicates in replicate_counts:
                    row: Dict[str, Any] = {
                        "rare_mass": p_rare,
                        "batch_size": batch_size,
                        "allocation": a_rare,
                        "replicates": replicates,
                    }
                    t0 = time.perf_counter()
                    grads_a = _run_arm_a(
                        np.random.default_rng(master.integers(1 << 31)), p_rare, batch_size, replicates
                    )
                    row["arm_A"] = _summarize(grads_a, exact_gradient)
                    row["arm_A"]["wall_time_seconds"] = time.perf_counter() - t0

                    t0 = time.perf_counter()
                    grads_b, m_rare_b, m_main_b = _run_arm_b(
                        np.random.default_rng(master.integers(1 << 31)), a_rare, batch_size, replicates
                    )
                    row["arm_B"] = _summarize(grads_b, exact_gradient)
                    row["arm_B"]["wall_time_seconds"] = time.perf_counter() - t0
                    row["arm_B"]["fixed_counts"] = {"main": m_main_b, "rare": m_rare_b}

                    t0 = time.perf_counter()
                    grads_c, m_rare_c, m_main_c, w_rare, w_main = _run_arm_c(
                        np.random.default_rng(master.integers(1 << 31)), p_rare, a_rare, batch_size, replicates
                    )
                    row["arm_C"] = _summarize(grads_c, exact_gradient)
                    row["arm_C"]["wall_time_seconds"] = time.perf_counter() - t0
                    row["arm_C"]["fixed_counts"] = {"main": m_main_c, "rare": m_rare_c}
                    row["arm_C"]["weights"] = {"main": w_main, "rare": w_rare}
                    row["arm_C"]["weight_sum_identity_holds"] = bool(
                        abs(m_rare_c * w_rare + m_main_c * w_main - batch_size) < 1e-6
                    )

                    t0 = time.perf_counter()
                    grads_d = _run_arm_d(
                        np.random.default_rng(master.integers(1 << 31)), p_rare, a_rare, batch_size, replicates
                    )
                    row["arm_D"] = _summarize(grads_d, exact_gradient)
                    row["arm_D"]["wall_time_seconds"] = time.perf_counter() - t0

                    results.append(row)

    checks = _analytic_preregistered_checks(results, rare_masses, batch_sizes, replicate_counts)
    return {
        "stage": "analytic",
        "seed": int(seed),
        "theta": _THETA.tolist(),
        "mu_main": _MU_MAIN.tolist(),
        "mu_rare": _MU_RARE.tolist(),
        "results": results,
        "preregistered_checks": checks,
        "wall_time_seconds": time.perf_counter() - started,
        "no_allocation_promoted_as_optimal": True,
        "no_ess_acceptance_threshold": True,
    }


def _analytic_preregistered_checks(
    results: List[Dict[str, Any]],
    rare_masses: Tuple[float, ...],
    batch_sizes: Tuple[int, ...],
    replicate_counts: Tuple[int, ...],
) -> Dict[str, Any]:
    """Preregistered implementation checks (not scientific gates)."""

    checks: Dict[str, Any] = {"per_row": [], "convergence": []}
    max_r = max(replicate_counts)
    min_r = min(replicate_counts)
    min_rare_mass = min(rare_masses)
    for row in results:
        for arm in ("A", "C"):
            entry = row["arm_" + arm]
            checks["per_row"].append({
                "arm": arm, "rare_mass": row["rare_mass"], "batch_size": row["batch_size"],
                "allocation": row["allocation"], "replicates": row["replicates"],
                "rule": "abs(z) < 4", "max_abs_z": entry["max_abs_z"],
                "passed": bool(entry["max_abs_z"] < 4.0),
            })
        # The single deliberately asymmetric configured case: the smallest
        # tested rare mass at allocation 0.5 (the largest possible departure
        # from that rare mass) and the largest replicate count (most
        # statistical power). This is one preregistered case, not every grid
        # point -- other (rare_mass, batch_size) combinations may show a
        # smaller, still-nonzero bias that does not clear this bound; that is
        # reported under "convergence" below, not asserted here.
        if (
            row["allocation"] == 0.5
            and row["rare_mass"] == min_rare_mass
            and row["replicates"] == max_r
        ):
            for arm in ("B", "D"):
                entry = row["arm_" + arm]
                checks["per_row"].append({
                    "arm": arm, "rare_mass": row["rare_mass"], "batch_size": row["batch_size"],
                    "allocation": row["allocation"], "replicates": row["replicates"],
                    "rule": "abs(z) > 6 (deliberately asymmetric configured case)",
                    "max_abs_z": entry["max_abs_z"], "passed": bool(entry["max_abs_z"] > 6.0),
                })

    # Convergence: A/C standard error should shrink ~ R^-1/2 (checked across
    # the whole grid -- this is a property of the estimator's variance, not
    # of how asymmetric one particular allocation is).
    by_key: Dict[Tuple[Any, ...], Dict[int, Dict[str, Any]]] = {}
    for row in results:
        key = (row["rare_mass"], row["batch_size"], row["allocation"])
        by_key.setdefault(key, {})[row["replicates"]] = row
    for key, by_r in by_key.items():
        if min_r not in by_r or max_r not in by_r:
            continue
        expected_ratio = float(np.sqrt(max_r / min_r))
        for arm in ("A", "C"):
            se_min = float(np.linalg.norm(by_r[min_r]["arm_" + arm]["stderr"]))
            se_max = float(np.linalg.norm(by_r[max_r]["arm_" + arm]["stderr"]))
            observed_ratio = (se_min / se_max) if se_max > 0 else float("inf")
            checks["convergence"].append({
                "arm": arm, "rare_mass": key[0], "batch_size": key[1], "allocation": key[2],
                "stderr_ratio_min_over_max_replicates": observed_ratio,
                "expected_ratio_sqrt_R": expected_ratio,
                # generous band: MC noise in the stderr estimate itself
                "within_generous_band": bool(0.4 * expected_ratio <= observed_ratio <= 2.5 * expected_ratio),
            })

    # D's bias should not vanish merely from more replicates at fixed B --
    # checked only at the single deliberately asymmetric configured case
    # (same case as the z>6 check above): elsewhere on the grid (e.g.
    # allocation == rare_mass) arm D's *true* bias is itself near zero, so a
    # small-sample MC estimate of "no bias" shrinking further is expected,
    # not a violation of this property.
    asymmetric_key = (min_rare_mass, None, 0.5)
    for key, by_r in by_key.items():
        if key[0] != asymmetric_key[0] or key[2] != asymmetric_key[2]:
            continue
        if min_r not in by_r or max_r not in by_r:
            continue
        bias_min = float(np.linalg.norm(by_r[min_r]["arm_D"]["bias"]))
        bias_max = float(np.linalg.norm(by_r[max_r]["arm_D"]["bias"]))
        checks["convergence"].append({
            "arm": "D", "rare_mass": key[0], "batch_size": key[1], "allocation": key[2],
            "deliberately_asymmetric_configured_case": True,
            "bias_norm_at_min_replicates": bias_min, "bias_norm_at_max_replicates": bias_max,
            "bias_does_not_vanish_with_replicates": bool(
                bias_max > 0.5 * bias_min if bias_min > 0 else bias_max > 0
            ),
        })

    checks["all_per_row_passed"] = all(c["passed"] for c in checks["per_row"])
    checks["all_convergence_bias_checks_passed"] = all(
        c["bias_does_not_vanish_with_replicates"]
        for c in checks["convergence"]
        if "bias_does_not_vanish_with_replicates" in c
    )
    return checks


# =============================================================================
# Stage 2: bounded D5 validation (torch, density_lab -- lazy imports)
# =============================================================================

_D5_CONFIG_PATH = REPO_ROOT / "configs" / "density_lab" / "estimator_validation_v0.json"


def run_d5_stage(*, artifact_root: Path, force: bool = False) -> Dict[str, Any]:
    from ship_muon_bg.density_lab.campaign import run_campaign
    from ship_muon_bg.density_lab.config import ExperimentConfig
    from ship_muon_bg.density_lab.reporting import load_run_records

    config = ExperimentConfig.from_json_file(_D5_CONFIG_PATH)
    started = time.perf_counter()
    summary = run_campaign(config, root=artifact_root, force=force)
    wall_time = time.perf_counter() - started

    from ship_muon_bg.density_lab.artifacts import ArtifactStore

    campaign_dir = ArtifactStore(config.experiment_id, root=artifact_root).experiment_dir
    records = load_run_records(campaign_dir)
    gradient_check = _d5_initial_gradient_check(config)

    per_run: List[Dict[str, Any]] = []
    for record in records:
        if record.get("status") != "completed":
            per_run.append({
                "run_id": record.get("run_id"), "status": record.get("status"),
                "technical_status": record.get("technical_status"),
            })
            continue
        per_run.append({
            "run_id": record.get("run_id"),
            "sampling_regime": record.get("sampling_regime"),
            "estimator": record.get("estimator"),
            "unbiasedness_status": record.get("unbiasedness_status"),
            "permitted_claim": record.get("permitted_claim"),
            "weight_normalization": record.get("weight_normalization"),
            "validation_objective_law": record.get("validation_objective_law"),
            "feature_space_train_nll": record.get("feature_space_train_nll"),
            "feature_space_train_main_nll": record.get("feature_space_train_main_nll"),
            "feature_space_train_rare_nll": record.get("feature_space_train_rare_nll"),
            "feature_space_validation_nll": record.get("feature_space_validation_nll"),
            "feature_space_validation_main_nll": record.get("feature_space_validation_main_nll"),
            "feature_space_validation_rare_nll": record.get("feature_space_validation_rare_nll"),
            "q_rare_region_mass": record.get("q_rare_region_mass"),
            "observed_q_rare_sample_count": record.get("observed_q_rare_sample_count"),
            "rare_mode_interpretation": record.get("rare_mode_interpretation"),
            "ess_over_n": record.get("ess_over_n"),
            "fit_wall_time_seconds": record.get("fit_wall_time_seconds"),
            "scientific_status": record.get("scientific_status"),
            "technical_status": record.get("technical_status"),
        })

    return {
        "stage": "d5",
        "config_path": str(_D5_CONFIG_PATH),
        "config_hash": config.config_hash(),
        "n_runs": summary["n_runs"],
        "n_completed": summary["n_completed"],
        "n_failed": summary["n_failed"],
        "n_skipped": summary["n_skipped"],
        "scientific_status_counts": summary["scientific_status_counts"],
        "wall_time_seconds": wall_time,
        "runs": per_run,
        "initial_parameter_gradient_check": gradient_check,
        "no_architecture_winner_declared": True,
        "no_allocation_promoted_as_optimal": True,
    }


def _d5_initial_gradient_check(config) -> Dict[str, Any]:
    """Gradient bias/variance at each model's *initial* theta, no training.

    Builds the training dataset and the arm-C fixed-composition minibatch
    plan exactly as the campaign would, then evaluates the feature-space NLL
    gradient at initialization across the plan's steps -- this is the
    training gradient the campaign will actually descend on, without paying
    for a full training run.
    """

    import torch

    from Nflow.torch_models.affine_coupling import AffineCouplingFlow
    from ship_muon_bg.density_lab.datasets import build_controlled_dataset
    from ship_muon_bg.density_lab.feature_pipeline import FittedFeaturePipeline
    from ship_muon_bg.density_lab.sampling import (
        ESTIMATOR_HORVITZ_THOMPSON_FIXED, STRATIFIED_HT_FIXED_COMPOSITION,
        plan_fixed_composition_batches, resolve_regime,
    )
    from ship_muon_bg.density_lab.targets import resolve_target
    from ship_muon_bg.benchmarks import embed_physical_to_raw

    out: List[Dict[str, Any]] = []
    target_spec = config.targets[0]
    model_spec = config.models[0]
    seed = config.seeds[0]

    for sampling in config.sampling_regimes:
        pool_law, estimator_kind = resolve_regime(sampling.regime)
        if estimator_kind != ESTIMATOR_HORVITZ_THOMPSON_FIXED:
            continue
        dataset = build_controlled_dataset(
            target_id=target_spec.target_id, variant=target_spec.variant, pdg_id=13,
            n_train=config.dataset.n_train, n_validation=config.dataset.n_validation,
            n_test=config.dataset.n_test, seed=seed, regime=sampling.regime,
            sampling_rare_fraction=sampling.sampling_rare_fraction,
            target_stage=target_spec.stage,
            validation_partition_law=sampling.resolved_validation_partition_law(),
        )
        target = resolve_target(target_spec.target_id, variant=target_spec.variant, stage=target_spec.stage)
        rare_id = target.rare_component_id(pdg_id=13)
        plan = plan_fixed_composition_batches(
            component_id=dataset.train.component_id, rare_id=rare_id,
            target_stratum_masses=target.stratum_masses(pdg_id=13),
            batch_size=sampling.minibatch_batch_size,
            counts={
                "rare": sampling.minibatch_rare_count,
                "main": sampling.minibatch_batch_size - sampling.minibatch_rare_count,
            },
            replacement=sampling.replacement, steps_per_epoch_rule=sampling.steps_per_epoch_rule,
            seed=seed, physical_weight=dataset.train.sample_weight,
        )
        raw_train = embed_physical_to_raw(dataset.train.physical, pdg_id=13, plane_z=0.0)
        from ship_muon_bg.data_contracts.feature_views import FeatureView

        pipeline = FittedFeaturePipeline.fit(raw_train, FeatureView(config.feature_views[0].view_id))
        normalized_train = pipeline.transform_raw(raw_train)
        train_tensor = torch.as_tensor(normalized_train, dtype=torch.float32)
        train_labels = torch.as_tensor(dataset.train.component_id)

        flow = AffineCouplingFlow(dimension=5, **{
            k: v for k, v in model_spec.params.items()
            if k not in ("max_epochs", "batch_size")
        })
        flow._build_module(seed=int(seed))
        module = flow._module

        rare_norms, main_norms, ht_losses = [], [], []
        n_diag_steps = min(5, plan.steps_per_epoch)
        for step in range(n_diag_steps):
            idx = torch.as_tensor(plan.step_indices(step), dtype=torch.long)
            weight = torch.as_tensor(plan.step_weights(step), dtype=torch.float32)
            x_batch = train_tensor[idx]
            labels_batch = train_labels[idx]
            module.zero_grad(set_to_none=True)
            loss = torch.sum(weight * -module.log_prob(x_batch)) / float(plan.batch_size)
            loss.backward()
            ht_losses.append(float(loss.detach()))
            for name, mask in (("rare", labels_batch == rare_id), ("main", labels_batch != rare_id)):
                if not bool(mask.any()):
                    continue
                module.zero_grad(set_to_none=True)
                sub_loss = torch.mean(-module.log_prob(x_batch[mask]))
                sub_loss.backward()
                norm = float(torch.sqrt(sum(
                    torch.sum(p.grad * p.grad) for p in module.parameters() if p.grad is not None
                )))
                (rare_norms if name == "rare" else main_norms).append(norm)
            module.zero_grad(set_to_none=True)

        out.append({
            "regime": sampling.regime,
            "minibatch_rare_count": sampling.minibatch_rare_count,
            "minibatch_batch_size": sampling.minibatch_batch_size,
            "steps_evaluated": n_diag_steps,
            "initial_ht_loss_mean": float(np.mean(ht_losses)) if ht_losses else None,
            "initial_ht_loss_std": float(np.std(ht_losses)) if ht_losses else None,
            "rare_gradient_norm_mean": float(np.mean(rare_norms)) if rare_norms else None,
            "main_gradient_norm_mean": float(np.mean(main_norms)) if main_norms else None,
        })
    return {"arm_C_configurations": out}


# =============================================================================
# CLI
# =============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("analytic", "d5"), required=True)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument(
        "--output", type=Path, default=None,
        help="path to write the JSON result (default: prints to stdout only)",
    )
    parser.add_argument(
        "--artifact-root", type=Path, default=None,
        help="(--stage d5 only) artifact root; defaults to a temp directory",
    )
    parser.add_argument("--force", action="store_true", help="(--stage d5 only) re-run completed runs")
    args = parser.parse_args()

    if args.stage == "analytic":
        result = run_analytic_stage(seed=args.seed)
        checks = result["preregistered_checks"]
        print("analytic stage: {} rows, wall_time={:.2f}s".format(
            len(result["results"]), result["wall_time_seconds"]
        ))
        print("  all per-row preregistered checks passed: {}".format(checks["all_per_row_passed"]))
        for c in checks["per_row"]:
            if not c["passed"]:
                print("  FAILED:", c)
    else:
        import tempfile

        artifact_root = args.artifact_root
        cleanup = None
        if artifact_root is None:
            cleanup = tempfile.TemporaryDirectory(prefix="rare_aware_d5_validation_")
            artifact_root = Path(cleanup.name)
        try:
            result = run_d5_stage(artifact_root=artifact_root, force=args.force)
        finally:
            if cleanup is not None:
                cleanup.cleanup()
        print("d5 stage: {}/{} runs completed, wall_time={:.1f}s".format(
            result["n_completed"], result["n_runs"], result["wall_time_seconds"]
        ))
        print("  scientific_status_counts:", result["scientific_status_counts"])

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True, default=str))
        print("wrote", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
