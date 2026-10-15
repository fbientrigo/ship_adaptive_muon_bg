"""D9-5 family-agnostic evaluation (Sec 13).

Generalizes ``ship_muon_bg.afterms.d9.evaluate``'s statistics pipeline to any
:class:`~ship_muon_bg.afterms.d9_5.model_adapter.ModelAdapter`, so the same
budgets, the same reused ``d8.statistics`` suite, and the same physical-space
Jacobian convention (Sec 7) apply identically to NF_AC/GAUSS_DIAG/GAUSS_FULL/
GMM. Never imports test data itself -- callers (the CLI's ``validate``/
``evaluate-test`` subcommands) are solely responsible for test-split
isolation (Sec 12/required tests 27/28).
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np

from ship_muon_bg.afterms.d8 import statistics as d8stats
from ship_muon_bg.afterms.d9.evaluate import negative_pz_diagnostics


def peak_host_ram_bytes() -> Optional[int]:
    """Best-effort current-process peak working-set size (Windows), without
    adding a new dependency. Returns None off Windows or on any failure."""

    try:
        import ctypes
        import ctypes.wintypes as wintypes

        class _ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return int(counters.PeakWorkingSetSize)
    except Exception:
        pass
    return None


def peak_cuda_vram_bytes() -> Optional[int]:
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated())
    except Exception:
        pass
    return None


def support_diagnostics(
    real: np.ndarray, generated: np.ndarray, *, train_feature_range: Optional[Any] = None,
) -> Dict[str, Any]:
    """Sec 13.D. ``train_feature_range``, if given, is ``(lo, hi)`` per-feature
    arrays from the training scope; observed-range exceedance is reported as a
    diagnostic only, never as automatic OOD."""

    real_features = real[:, :5]
    negative_pz_real = int(np.count_nonzero(real_features[:, 2] < 0.0))
    non_finite_generated = int(np.count_nonzero(~np.isfinite(generated)))
    finite_mask = np.all(np.isfinite(generated), axis=1)
    finite_generated = generated[finite_mask]

    result: Dict[str, Any] = {
        "negative_pz_count_real": negative_pz_real,
        "negative_pz_fraction_real": (
            float(negative_pz_real / real_features.shape[0]) if real_features.shape[0] else None
        ),
        "non_finite_generated_count": non_finite_generated,
        "extreme_coordinate_summary": {
            "real_min": real_features.min(axis=0).tolist() if real_features.shape[0] else None,
            "real_max": real_features.max(axis=0).tolist() if real_features.shape[0] else None,
            "generated_min": finite_generated.min(axis=0).tolist() if finite_generated.shape[0] else None,
            "generated_max": finite_generated.max(axis=0).tolist() if finite_generated.shape[0] else None,
        },
        "real_covariance": np.cov(real_features, rowvar=False).tolist() if real_features.shape[0] > 1 else None,
        "generated_covariance": (
            np.cov(finite_generated, rowvar=False).tolist() if finite_generated.shape[0] > 1 else None
        ),
    }
    if train_feature_range is not None and finite_generated.shape[0]:
        lo, hi = train_feature_range
        outside = np.any((finite_generated < lo) | (finite_generated > hi), axis=1)
        result["generated_outside_training_range_count"] = int(np.count_nonzero(outside))
        result["generated_outside_training_range_fraction"] = float(np.mean(outside))
        result["observed_range_exceedance_note"] = "diagnostic only, not automatic OOD (Sec 13.D)"
    return result


def evaluate_adapter(
    adapter: Any,
    split_raw: np.ndarray,
    *,
    split_name: str,
    track_id: str,
    budget: Dict[str, Any],
    budget_name: str,
    generation_seed: int,
    train_feature_range: Optional[Any] = None,
) -> Dict[str, Any]:
    """Evaluate one already-fitted/loaded adapter on ``split_raw`` (raw (n, 8)
    rows already PDG-filtered to ``track_id``). ``split_name`` is
    ``"validation"`` or ``"test"`` and only controls which adapter method is
    called and how results are labeled -- it carries no isolation guarantee by
    itself; the caller is responsible for never passing test rows before a
    selection freeze exists."""

    log_prob_fn = adapter.test_log_prob if split_name == "test" else adapter.validation_log_prob
    start_lp = time.perf_counter()
    feature_log_prob = log_prob_fn(split_raw)
    log_prob_wall = time.perf_counter() - start_lp
    feature_nll = float(-np.mean(feature_log_prob))
    finite_log_prob_fraction = float(np.mean(np.isfinite(feature_log_prob)))

    physical_nll = None
    physical_nll_error = None
    try:
        log_abs_det_jacobian = adapter.pipeline.forward_log_abs_det_jacobian(split_raw)
        physical_nll = float(-np.mean(feature_log_prob + log_abs_det_jacobian))
    except Exception as exc:  # e.g. an out-of-domain row for a non-identity preprocessing variant
        physical_nll_error = str(exc)

    n_generated = int(budget["n_generated_samples"])
    start_gen = time.perf_counter()
    generated_physical = adapter.sample(n_generated, seed=generation_seed)
    generation_wall = time.perf_counter() - start_gen

    one_dimensional = d8stats.run_1d_suite(
        split_raw[:, :5], generated_physical, weighted=False, weights=None,
        n_boot=budget.get("one_d_n_boot", 200), seed=generation_seed,
    )
    two_dimensional = d8stats.run_2d_suite(
        split_raw[:, :5], generated_physical, weighted=False,
        energy_sample_size=budget.get("two_d_test_subsample", 500),
        n_permutations=budget.get("two_d_n_permutations", 200), seed=generation_seed,
    )
    c2st_sample = budget.get("c2st_subsample", 500)
    ndimensional_c2st = d8stats.ndim_c2st(
        split_raw[:min(c2st_sample, split_raw.shape[0]), :5],
        generated_physical[:min(c2st_sample, generated_physical.shape[0])],
        sample_size=c2st_sample, seed=generation_seed,
        n_boot=budget.get("c2st_n_boot", 200), n_permutations=budget.get("c2st_n_permutations", 200),
    )

    return {
        "track_id": track_id,
        "model_family_id": adapter.model_family_id,
        "model_config_id": adapter.model_config_id,
        "split": split_name,
        "budget_name": budget_name,
        "generation_seed": generation_seed,
        "feature_nll": feature_nll,
        "physical_nll": physical_nll,
        "physical_nll_error": physical_nll_error,
        "finite_log_prob_fraction": finite_log_prob_fraction,
        "negative_pz_diagnostics": negative_pz_diagnostics(generated_physical[:, 2]),
        "support_diagnostics": support_diagnostics(split_raw, generated_physical, train_feature_range=train_feature_range),
        "one_dimensional_tests": one_dimensional,
        "two_dimensional_tests": two_dimensional,
        "ndimensional_c2st": ndimensional_c2st,
        "engineering_metrics": {
            "log_prob_wall_time_seconds": log_prob_wall,
            "log_prob_throughput_rows_per_second": (
                float(split_raw.shape[0] / log_prob_wall) if log_prob_wall > 0 else None
            ),
            "sample_generation_wall_time_seconds": generation_wall,
            "samples_per_second": float(n_generated / generation_wall) if generation_wall > 0 else None,
            "generated_sample_count": int(generated_physical.shape[0]),
            "peak_host_ram_bytes": peak_host_ram_bytes(),
            "peak_cuda_vram_bytes": peak_cuda_vram_bytes(),
        },
    }
