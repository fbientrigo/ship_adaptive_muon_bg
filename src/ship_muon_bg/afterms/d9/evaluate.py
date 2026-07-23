"""D9 test-split evaluation (§14).

This is the ONLY module in the D9 package permitted to load a test shard --
``runner.py``/``train_candidate_seed`` has no parameter that could carry one
(required tests 22/23). ``evaluate_candidate_seed`` is called strictly after
training completes and selection is frozen (§12).

Reuses D8's campaign-independent statistics helpers
(``ship_muon_bg.afterms.d8.statistics``) rather than duplicating them; never
imports or modifies the frozen ``ship_muon_bg.afterms.d8.legacy_d7_adapter``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from ship_muon_bg.afterms.d8 import statistics as d8stats
from ship_muon_bg.afterms.preprocessing import PreprocessingPipeline
from ship_muon_bg.data_contracts import schema

from . import checkpoint as ckpt
from . import contract as d9contract
from . import runner as d9runner


def load_test_split(candidate_config: Dict[str, Any], shard_dir: Path) -> np.ndarray:
    """The only D9 function that touches test shards."""

    raw = d9runner.load_concatenated_shards(shard_dir, candidate_config["test_shards"])
    return d9runner.pdg_filter(raw, candidate_config["pdg_value"])


def _reconstruct_pipeline(preprocessing_path: Path) -> PreprocessingPipeline:
    """``preprocessing_path`` is the D9 preprocessing *contract* sidecar
    (§8: preprocessing_name, feature_order, fitted_state, training_split_hash,
    ...), not the bare ``PreprocessingPipeline.to_dict()`` payload -- the
    fitted pipeline state is nested under ``fitted_state``."""

    contract = json.loads(preprocessing_path.read_text(encoding="utf-8"))
    return PreprocessingPipeline.from_dict(contract["fitted_state"])


def negative_pz_diagnostics(generated_pz: np.ndarray) -> Dict[str, Any]:
    generated_pz = np.asarray(generated_pz, dtype=np.float64)
    if generated_pz.size == 0:
        return {
            "generated_domain_violation_count": 0,
            "generated_domain_violation_rate": None,
            "min_generated_pz": None,
        }
    return {
        "generated_domain_violation_count": int(np.count_nonzero(generated_pz < 0.0)),
        "generated_domain_violation_rate": float(np.mean(generated_pz < 0.0)),
        "min_generated_pz": float(np.min(generated_pz)),
    }


def generate_deterministic_samples(estimator, n: int, seed: int):
    import torch

    module = estimator._module
    module.eval()
    gen = torch.Generator(device=estimator.device)
    gen.manual_seed(int(seed))
    with torch.no_grad():
        z = torch.randn(int(n), 5, dtype=estimator.torch_dtype, device=estimator.device, generator=gen)
        return module(z).cpu().numpy()


def evaluate_candidate_seed(
    candidate_config: Dict[str, Any],
    *,
    run_dir: Path,
    shard_dir: Path,
    checkpoint_scope: str = ckpt.SCOPE_BEST,
    budget_name: str = "quick_validation_budget",
    device: str = "cpu",
    generation_seed: int = 20260720,
) -> Dict[str, Any]:
    """Evaluate one completed (candidate, seed) run on its test split.

    ``budget_name`` selects ``candidate_config["evaluation_policy"][budget_name]``
    (``quick_validation_budget`` or ``final_candidate_budget``); the caller
    chooses which to run, per §14.
    """

    from Nflow.registry import create_density_estimator

    budget = candidate_config["evaluation_policy"][budget_name]
    bundle_path = Path(run_dir, "checkpoints", ckpt.FILENAME_BY_SCOPE[checkpoint_scope])
    bundle = ckpt.load_bundle(bundle_path)

    estimator = create_density_estimator(
        {"family": candidate_config["model_family"], "params": d9runner._architecture_params(candidate_config)},
        dimension=5, device=device,
    )
    estimator._build_module(seed=int(bundle["seed"]))
    estimator._module.load_state_dict(bundle["model_state_dict"])
    estimator._module.eval()

    pipeline = _reconstruct_pipeline(Path(run_dir, "preprocessing", "preprocessing.json"))

    test_raw = load_test_split(candidate_config, shard_dir)
    test_norm = pipeline.transform(test_raw)

    import torch

    with torch.no_grad():
        test_lp = estimator._module.log_prob(
            torch.tensor(test_norm, dtype=estimator.torch_dtype, device=device)
        ).cpu().numpy().astype(np.float64)
    test_feature_nll = float(-np.mean(test_lp))

    physical_nll = None
    try:
        log_jac = pipeline.forward_log_abs_det_jacobian(test_raw)
        physical_nll = float(-np.mean(test_lp + log_jac))
    except Exception as exc:  # e.g. quantile_normal_v0 or an out-of-domain row
        physical_nll = None
        physical_nll_error = str(exc)
    else:
        physical_nll_error = None

    n_generated = budget["n_generated_samples"]
    generated = generate_deterministic_samples(estimator, n_generated, generation_seed)
    generated_physical = pipeline.inverse(generated)

    weighted = candidate_config["weighting_policy"] == "production_weighted"
    weights = (
        test_raw[:, schema.COLUMN_INDEX["w"]].astype(np.float64) if weighted else None
    )

    one_d = d8stats.run_1d_suite(
        test_raw[:, :5], generated_physical, weighted=weighted, weights=weights,
        n_boot=budget.get("one_d_n_boot", 200), seed=generation_seed,
    )
    two_d = d8stats.run_2d_suite(
        test_raw[:, :5], generated_physical, weighted=weighted,
        energy_sample_size=budget.get("two_d_test_subsample", 500),
        n_permutations=budget.get("two_d_n_permutations", 200), seed=generation_seed,
    )
    c2st_sample = budget.get("c2st_subsample", 500)
    ndim = d8stats.ndim_c2st(
        test_raw[:min(c2st_sample, test_raw.shape[0]), :5],
        generated_physical[:min(c2st_sample, generated_physical.shape[0])],
        sample_size=c2st_sample, seed=generation_seed,
        n_boot=budget.get("c2st_n_boot", 200), n_permutations=budget.get("c2st_n_permutations", 200),
    )

    result = {
        "candidate_id": candidate_config["candidate_id"],
        "seed": int(bundle["seed"]),
        "checkpoint_scope": checkpoint_scope,
        "budget_name": budget_name,
        "evaluation_policy_hash": d9contract.evaluation_policy_hash(
            evaluation_policy=candidate_config["evaluation_policy"],
        ),
        "test_feature_nll": test_feature_nll,
        "test_physical_nll": physical_nll,
        "test_physical_nll_error": physical_nll_error,
        "generation_seed": generation_seed,
        "generated_sample_count": int(generated_physical.shape[0]),
        "negative_pz_diagnostics": negative_pz_diagnostics(generated_physical[:, 2]),
        "one_dimensional_tests": one_d,
        "two_dimensional_tests": two_d,
        "ndimensional_c2st": ndim,
        "weighted_estimand_note": (
            "weighted 2D/C2ST inference deferred; see two_dimensional_tests.deferred" if weighted else None
        ),
    }

    eval_dir = Path(run_dir, "evaluation")
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / f"{budget_name}.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result
