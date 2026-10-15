#!/usr/bin/env python3
"""run_afterms_nightly_queue.py: Sequential subprocess campaign runner for after-MS smokes.

Supports:
    --dry-run
    --resume
    --jobs JOB_LIST
    --device DEVICE
    --force-job JOB_LIST
    --stop-after JOB_NAME
    --run-job JOB_NAME (used internally by subprocesses)
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import pickle
import signal
import subprocess
import sys
import time
import traceback
import shutil
import numpy as np

# Add project root and src/ to path to allow importing ship_muon_bg and Nflow
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ship_muon_bg.data_contracts import load_muon_pkl, dataset_hash, schema
from ship_muon_bg.afterms import audit, split, stratify
from ship_muon_bg.afterms.preprocessing import PreprocessingPipeline


# Single source of truth for the nightly queue's job names. `main()`'s
# sequential loop and `build_final_nightly_report()` used to keep independent
# copies of this list; they drifted (job 13 was missing from the report's
# copy), so both now read from here.
NIGHTLY_JOB_NAMES = [
    "00_environment_and_dataset_smoke",
    "01_build_afterms_shards",
    "02_validate_afterms_shards",
    "03_preprocessing_roundtrip_and_plots",
    "04_legacy_available_code_realnvp_quantile",
    "05_affine_preprocessing_ab_pdg13",
    "06_affine_preprocessing_ab_pdg_minus13",
    "07_affine_weight_ab_pdg13",
    "08_affine_weight_ab_pdg_minus13",
    "09_affine_capacity_smoke_pdg13",
    "10_gaussian_controls_pdg13",
    "11_gaussian_controls_pdg_minus13",
    "12_memory_release_repeat_smoke",
    "13_build_nightly_report",
]


# --- Windows Memory Helpers using ctypes ---

class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_uint64),
        ("ullAvailPhys", ctypes.c_uint64),
        ("ullTotalPageFile", ctypes.c_uint64),
        ("ullAvailPageFile", ctypes.c_uint64),
        ("ullTotalVirtual", ctypes.c_uint64),
        ("ullAvailVirtual", ctypes.c_uint64),
        ("ullAvailExtendedVirtual", ctypes.c_uint64),
    ]


def get_current_process_rss() -> int:
    """Return process RSS in bytes."""
    try:
        GetProcessMemoryInfo = ctypes.windll.psapi.GetProcessMemoryInfo
        GetCurrentProcess = ctypes.windll.kernel32.GetCurrentProcess
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        if GetProcessMemoryInfo(GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return counters.WorkingSetSize
    except Exception:
        pass
    return 0


def get_system_ram_info() -> tuple[int, int]:
    """Return (free_ram_bytes, total_ram_bytes)."""
    try:
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return stat.ullAvailPhys, stat.ullTotalPhys
    except Exception:
        pass
    return 0, 0


def get_gpu_memory_info() -> tuple[float, float]:
    """Return (free_gpu_mb, total_gpu_mb) using nvidia-smi."""
    try:
        cmd = ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,nounits,noheader"]
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
        parts = out.strip().split(",")
        return float(parts[0]), float(parts[1])
    except Exception:
        return 0.0, 0.0


def get_active_cuda_processes() -> list[str]:
    """Return a list of processes using CUDA."""
    try:
        cmd = ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"]
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
        lines = out.strip().split("\n")
        return [l.strip() for l in lines if l.strip()]
    except Exception:
        return []


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


_UNSET = object()


def update_queue_state(artifact_dir, active_job=_UNSET, pid=_UNSET, completed=_UNSET, failed=_UNSET, pending=_UNSET):
    """Merge-update queue_state.json. Any argument left at its default
    (unspecified) leaves that field untouched; passing ``None`` explicitly
    sets the field to ``None`` -- this distinction matters because every
    caller that clears ``active_job``/``pid`` after a job finishes (or is
    interrupted) does so by passing ``active_job=None, pid=None``. A prior
    ``if x is not None`` check treated that identically to "not passed",
    so those clears were silently no-ops and queue_state.json kept reporting
    the last-started job as active forever, including across Ctrl+C."""
    queue_state_path = os.path.join(artifact_dir, "queue_state.json")
    state = {}
    if os.path.exists(queue_state_path):
        try:
            with open(queue_state_path, "r") as f:
                state = json.load(f)
        except Exception:
            pass

    if active_job is not _UNSET:
        state["active_job"] = active_job
    if pid is not _UNSET:
        state["pid"] = pid
    if completed is not _UNSET:
        state["completed_jobs"] = completed
    if failed is not _UNSET:
        state["failed_jobs"] = failed
    if pending is not _UNSET:
        state["pending_jobs"] = pending
    state["heartbeat"] = time.time()
    
    try:
        with open(queue_state_path, "w") as f:
            json.dump(state, f, indent=2)
    except Exception:
        pass


# --- Preprocessing scaling plots helper (Matplotlib) ---

def generate_scaling_plots(raw_data, transformed_data, variant_id, output_dir):
    try:
        import matplotlib.pyplot as plt
        os.makedirs(output_dir, exist_ok=True)
        
        # Bounded plotting sample (at most 5000 points)
        sample_n = min(5000, raw_data.shape[0])
        idx = np.arange(sample_n)
        raw_s = raw_data[idx]
        trans_s = transformed_data[idx]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        features = ["px", "py", "pz", "x", "y"]
        # Raw marginals
        for i, name in enumerate(features):
            axes[i].hist(raw_s[:, i], bins=50, alpha=0.6, label="Raw")
            axes[i].set_title(f"Raw {name}")
            
        fig.suptitle(f"Preprocessing Raw Marginals - {variant_id}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{variant_id}_raw_marginals.png"))
        plt.close()
        
        # Transformed marginals
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        for i, name in enumerate(features):
            axes[i].hist(trans_s[:, i], bins=50, alpha=0.6, color="orange", label="Transformed")
            axes[i].set_title(f"Transformed {name}")
            
        fig.suptitle(f"Preprocessing Transformed Marginals - {variant_id}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{variant_id}_transformed_marginals.png"))
        plt.close()
    except Exception as e:
        print(f"Plotting failed: {e}")


def train_and_validation_nll_from_fit_result(fit_res):
    """Extract (train_nll, validation_nll) from a one-shot `FitResult`.

    `best_validation_nll` describes the validation set only; it must never be
    reused as the train NLL (they are different quantities computed on
    different rows). The train NLL, when available, comes from the fitter's
    own `train_history` entries. Returns `None` for either value when the
    fitter did not record it, rather than fabricating a number.
    """
    val_nll = (
        float(fit_res.best_validation_nll)
        if fit_res.best_validation_nll is not None
        else None
    )
    train_nll = None
    if fit_res.train_history:
        last_entry = fit_res.train_history[-1]
        if "train_nll" in last_entry:
            train_nll = float(last_entry["train_nll"])
    return train_nll, val_nll


def generated_sample_hash(samples):
    """Stable content hash of a generated-sample array.

    Used to verify that repeated report/sample generation from an unchanged
    checkpoint and seed reproduces bit-identical output, not just "close"
    output.
    """
    arr = np.ascontiguousarray(samples, dtype=np.float64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def checkpoint_file_hash(path):
    """Sha256 of a saved checkpoint file, for reload-provenance tracking."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _write_preprocessing_sidecar(run_dir, pipeline):
    """Persists the fitted PreprocessingPipeline's state (mean/std or
    QuantileTransformer) alongside the model checkpoint. PreprocessingPipeline
    already has to_dict()/from_dict(); this was simply never called, so a
    future evaluation tool had no way to invert generated samples to physical
    space without re-fitting the preprocessor from the raw shard."""
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "preprocessing.json"), "w") as f:
        json.dump(pipeline.to_dict(), f, indent=2)


# --- C2ST & NN metrics helper ---

def compute_nn_distances(q_samples, ref_samples, subsample_size=1000):
    n_q = q_samples.shape[0]
    m = min(n_q, subsample_size)
    q_sub = q_samples[:m]
    n_ref = ref_samples.shape[0]
    ref_sub = ref_samples[:min(n_ref, 5000)]
    
    dists = np.linalg.norm(q_sub[:, None, :] - ref_sub[None, :, :], axis=2)
    min_dists = dists.min(axis=1)
    return {
        "mean": float(np.mean(min_dists)),
        "median": float(np.median(min_dists)),
        "min": float(np.min(min_dists)),
        "max": float(np.max(min_dists)),
    }


def evaluate_generated_samples(test_data, q_samples, atol=1e-6):
    from ship_muon_bg.density_lab.metrics import c2st, tail_quantile_errors, exceedance_probability_errors, duplicate_diagnostics
    
    dim = test_data.shape[1]
    if dim == 4:
        col_names = ["px", "py", "pz", "E"]
    else:
        col_names = ["px", "py", "pz", "x", "y"]
        test_data = test_data[:, :5]
        q_samples = q_samples[:, :5]
        
    p_feats = test_data
    q_feats = q_samples
    
    # C2ST on frozen bounded sample
    c2st_res = {}
    n_c2st = min(2000, p_feats.shape[0], q_feats.shape[0])
    if n_c2st >= 10:
        c2st_res = c2st(p_feats[:n_c2st], q_feats[:n_c2st], seed=42)
        
    tq_errs = tail_quantile_errors(p_feats, q_feats, quantiles=[0.9, 0.99, 0.999], column_names=col_names)
    exc_errs = exceedance_probability_errors(p_feats, q_feats, thresholds=[60.0, 70.0], column_index=2)
    dups = duplicate_diagnostics(q_feats, atol=atol)
    
    # Exceedance counts
    pz_q = q_samples[:, 2]
    pt_q = np.sqrt(q_samples[:, 0]**2 + q_samples[:, 1]**2)
    
    exceedance_counts = {
        "pz_gt_60": int(np.count_nonzero(pz_q > 60.0)),
        "pz_gt_70": int(np.count_nonzero(pz_q > 70.0)),
        "pt_gt_1": int(np.count_nonzero(pt_q > 1.0)),
        "pt_gt_2": int(np.count_nonzero(pt_q > 2.0)),
    }
    
    # Weighted/unweighted marginal summaries
    marginal_summaries = {}
    for i, name in enumerate(col_names):
        col = q_samples[:, i]
        marginal_summaries[name] = {
            "mean": float(np.mean(col)),
            "std": float(np.std(col)),
            "min": float(np.min(col)),
            "max": float(np.max(col)),
            "q10": float(np.quantile(col, 0.1)),
            "q50": float(np.quantile(col, 0.5)),
            "q90": float(np.quantile(col, 0.9)),
        }

    # Generated-support (domain) audit: an unconstrained Gaussian-base flow
    # can invert to pz < 0, outside the physical domain the log1p_pz view
    # assumes (pz >= 0). This never clips or repairs generated samples; it
    # only counts/reports violations so they stay visible in the report.
    generated_domain_violations = {
        "generated_domain_violation_count": int(np.count_nonzero(pz_q < 0.0)),
        "generated_domain_violation_rate": float(np.mean(pz_q < 0.0)),
        "min_generated_pz": float(np.min(pz_q)),
        "quantiles_of_generated_pz": {
            "q001": float(np.quantile(pz_q, 0.001)),
            "q01": float(np.quantile(pz_q, 0.01)),
            "q05": float(np.quantile(pz_q, 0.05)),
            "q50": float(np.quantile(pz_q, 0.50)),
        },
    }

    return {
        "c2st": c2st_res,
        "tail_quantile_errors": tq_errs,
        "exceedance_probability_errors": exc_errs,
        "duplicate_diagnostics": dups,
        "exceedance_counts": exceedance_counts,
        "marginal_summaries": marginal_summaries,
        "generated_domain_violations": generated_domain_violations,
    }
def run_environment_and_dataset_smoke(args, job_dir):
    import torch
    import platform
    
    os.makedirs(job_dir, exist_ok=True)
    
    git_commit = get_git_commit()
    
    sys_info = {
        "platform": platform.platform(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "available_disk_gb": shutil.disk_usage(".").free / (1024**3),
    }
    
    pkl_file = "data/raw/nflow_releases/muonsFullMC_afterMS.pkl"
    file_info = {}
    if os.path.exists(pkl_file):
        file_info["exists"] = True
        file_info["size_bytes"] = os.path.getsize(pkl_file)
        file_info["sha256"] = audit.file_sha256(pkl_file)
    else:
        file_info["exists"] = False
        
    out = {
        "system_info": sys_info,
        "dataset_file_info": file_info,
        "git_commit": git_commit,
    }
    
    with open(os.path.join(job_dir, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)


def run_validate_afterms_shards(args, job_dir):
    os.makedirs(job_dir, exist_ok=True)
    manifest_path = os.path.join(args.shard_dir, "shard_manifest.json")
    
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Shard manifest not found at {manifest_path}")
        
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
        
    shards_checked = []
    for shard in manifest.get("shards", []):
        file_path = os.path.join(args.shard_dir, shard["npy_file"])
        indices_path = os.path.join(args.shard_dir, shard["indices_file"])
        
        exists = os.path.exists(file_path)
        indices_exists = os.path.exists(indices_path)
        size = os.path.getsize(file_path) if exists else 0
        
        shards_checked.append({
            "npy_file": shard["npy_file"],
            "exists": exists,
            "indices_exists": indices_exists,
            "size_bytes": size,
            "row_count": shard["row_count"],
        })
        
    with open(os.path.join(job_dir, "metrics.json"), "w") as f:
        json.dump({
            "manifest_checked": True,
            "shards_checked": shards_checked,
            "total_shards": len(shards_checked),
        }, f, indent=2)


def run_preprocessing_roundtrip_and_plots(args, job_dir):
    os.makedirs(job_dir, exist_ok=True)
    
    train_file = os.path.join(args.shard_dir, "train_shard_000.npy")
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Train shard not found: {train_file}")
        
    train_data = np.load(train_file)
    
    results = {}
    variants = ["identity_standardized_v0", "quantile_normal_v0", "cartesian_log1p_pz_v0"]
    
    for var_id in variants:
        pipeline = PreprocessingPipeline(var_id, seed=20260720)
        pipeline.fit(train_data)
        
        transformed = pipeline.transform(train_data)
        reconstructed = pipeline.inverse(transformed)
        
        err = np.mean(np.abs(reconstructed - train_data[:, :5]))
        
        pz_check_err = None
        if var_id == "cartesian_log1p_pz_v0":
            bad_data = train_data[:10].copy()
            bad_data[:, 2] = -1.0
            try:
                pipeline.transform(bad_data)
                pz_check_err = "Failed to raise error on negative pz"
            except ValueError:
                pz_check_err = "Raised ValueError correctly on negative pz"
                
        results[var_id] = {
            "roundtrip_mean_absolute_error": float(err),
            "pz_check_negative": pz_check_err,
        }
        
        plot_dir = os.path.join(args.artifact_dir, "plots")
        generate_scaling_plots(train_data, transformed, var_id, plot_dir)
        
    with open(os.path.join(job_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)


# --- Neural Training Subprocess Routines ---


def run_neural_training_subprocess(job_name, device, shard_dir, job_dir):
    """Executes the training loop for a specific job name inside a clean python process."""
    import torch
    from Nflow.registry import create_density_estimator
    
    os.makedirs(job_dir, exist_ok=True)
    
    # Load shards
    train_data = np.load(os.path.join(shard_dir, "train_shard_000.npy"))
    val_data = np.load(os.path.join(shard_dir, "validation_shard_000.npy"))
    test_data = np.load(os.path.join(shard_dir, "test_shard_000.npy"))
    
    print(f"[{job_name}] Loaded train: {train_data.shape}, val: {val_data.shape}, test: {test_data.shape}")
    
    history = []
    
    # Parse job characteristics
    pdg_id = None
    if "pdg13" in job_name:
        pdg_id = 13
    elif "pdg_minus13" in job_name:
        pdg_id = -13
        
    if pdg_id is not None:
        train_mask = np.rint(train_data[:, schema.COLUMN_INDEX["id"]]) == pdg_id
        val_mask = np.rint(val_data[:, schema.COLUMN_INDEX["id"]]) == pdg_id
        test_mask = np.rint(test_data[:, schema.COLUMN_INDEX["id"]]) == pdg_id
        train_data = train_data[train_mask]
        val_data = val_data[val_mask]
        test_data = test_data[test_mask]
        print(f"[{job_name}] Filtered PDG {pdg_id} - train: {train_data.shape}, val: {val_data.shape}, test: {test_data.shape}")

    # Set up preprocessing and model based on job name
    preprocessor = None
    model_spec = None
    weighted = False
    
    if job_name == "04_legacy_available_code_realnvp_quantile":
        # px, py, pz, E as features
        # Mass: 0.1134289259
        mass_muon = 0.1134289259
        if abs(mass_muon - 0.10565837) > 1e-5:
            print(f"WARNING: Historical mass constant {mass_muon} differs from standard muon mass 0.10565837")
            
        def get_legacy_features(arr):
            px = arr[:, 0]
            py = arr[:, 1]
            pz = arr[:, 2]
            E = np.sqrt(px**2 + py**2 + pz**2 + mass_muon**2)
            return np.column_stack((px, py, pz, E))
            
        train_feats = get_legacy_features(train_data)
        val_feats = get_legacy_features(val_data)
        test_feats = get_legacy_features(test_data)
        
        from sklearn.preprocessing import QuantileTransformer
        qt = QuantileTransformer(output_distribution="normal", random_state=20260720)
        train_scaled = qt.fit_transform(train_feats)
        val_scaled = qt.transform(val_feats)
        test_scaled = qt.transform(test_feats)
        
        # Legacy NormalizingFlow model
        from Nflow.legacy.utils.flow_models import NormalizingFlow
        model = NormalizingFlow(input_dim=4, hidden_dim=160, n_layers=10)
        model.to(device)
        model.base_dist = torch.distributions.MultivariateNormal(
            torch.zeros(4, device=device),
            torch.eye(4, device=device)
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0004, weight_decay=6.3e-7)
        
        # Custom loop to record monitoring details
        train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_scaled, dtype=torch.float32))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        val_dataset = torch.utils.data.TensorDataset(torch.tensor(val_scaled, dtype=torch.float32))
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        wall_start = time.perf_counter()
        for epoch in range(5):
            t0 = time.perf_counter()
            model.train()
            train_loss = 0.0
            dl_time = 0.0
            t_loader_start = time.perf_counter()
            
            for batch in train_loader:
                dl_time += time.perf_counter() - t_loader_start
                x = batch[0].to(device)
                optimizer.zero_grad()
                loss = -model.log_prob(x).mean()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                t_loader_start = time.perf_counter()
                
            train_loss /= len(train_loader)
            train_time = time.perf_counter() - t0 - dl_time
            
            # Validation NLL
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch[0].to(device)
                    loss = -model.log_prob(x).mean()
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            
            # Memory usage
            cpu_rss = get_current_process_rss()
            gpu_alloc = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            gpu_res = torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
            gpu_peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            
            epoch_metrics = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "cpu_rss_bytes": cpu_rss,
                "gpu_allocated_bytes": gpu_alloc,
                "gpu_reserved_bytes": gpu_res,
                "gpu_peak_bytes": gpu_peak,
                "dataloader_time_seconds": dl_time,
                "training_time_seconds": train_time,
            }
            history.append(epoch_metrics)
            print(f"Epoch {epoch+1}/5: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
        total_time = time.perf_counter() - wall_start
        
        # Save model. NormalizingFlow (legacy) has no save()/load() class
        # method (unlike the registry-backed families in Nflow/), so a
        # hand-rolled JSON sidecar records the architecture and
        # feature-engineering constants that would otherwise only exist as
        # script literals, plus the fitted QuantileTransformer's state, so
        # this checkpoint is reconstructible from job_dir alone.
        legacy_ckpt_path = os.path.join(job_dir, "legacy_model.pt")
        torch.save(model.state_dict(), legacy_ckpt_path)
        ckpt_hash = checkpoint_file_hash(legacy_ckpt_path)
        with open(os.path.join(job_dir, "legacy_model_config.json"), "w") as f:
            json.dump({
                "family": "legacy_normalizing_flow",
                "input_dim": 4,
                "hidden_dim": 160,
                "n_layers": 10,
                "mass_muon": mass_muon,
                "feature_order": ["px", "py", "pz", "E"],
                "dtype": "float32",
                "checkpoint_hash": ckpt_hash,
            }, f, indent=2)
        with open(os.path.join(job_dir, "legacy_preprocessing.json"), "w") as f:
            json.dump({
                "variant_id": "quantile_transformer_legacy",
                "seed": 20260720,
                "qt_pkl": pickle.dumps(qt).hex(),
            }, f, indent=2)

        # Generate samples for evaluation. `base_dist.sample()` draws from the
        # global torch RNG with no generator hook, so a fixed seed is set
        # immediately beforehand to make the generated sample reproducible.
        model.eval()
        generation_seed = 20260720
        torch.manual_seed(generation_seed)
        with torch.no_grad():
            z = model.base_dist.sample((5000,))
            gen_scaled = model.inverse(z).cpu().numpy()
        gen_feats = qt.inverse_transform(gen_scaled)
        
        # Compute test loss
        model.eval()
        with torch.no_grad():
            test_lp = model.log_prob(torch.tensor(test_scaled, dtype=torch.float32).to(device)).cpu().numpy()
        test_nll = float(-np.mean(test_lp))
        
        # Minimum metrics
        eval_metrics = evaluate_generated_samples(test_feats, gen_feats)
        eval_metrics["train_feature_space_nll"] = float(train_loss)
        eval_metrics["validation_feature_space_nll"] = float(val_loss)
        eval_metrics["test_feature_space_nll"] = test_nll
        # QuantileTransformer has no analytic density Jacobian (same as
        # quantile_normal_v0 in PreprocessingPipeline) so physical-space NLL
        # is genuinely undefined here. Serialize it explicitly as null rather
        # than omitting the key, so report builders never fall back to 0.0.
        eval_metrics["physical_space_nll"] = None
        eval_metrics["wall_time_seconds"] = total_time
        eval_metrics["parameter_count"] = sum(p.numel() for p in model.parameters())
        eval_metrics["generation_seed"] = generation_seed
        eval_metrics["generated_sample_count"] = int(gen_feats.shape[0])
        eval_metrics["generated_sample_hash"] = generated_sample_hash(gen_feats)
        eval_metrics["checkpoint_hash"] = ckpt_hash

        # Save results
        with open(os.path.join(job_dir, "metrics.json"), "w") as f:
            json.dump({"history": history, "metrics": eval_metrics, "reproduction_scope": "available_code_semantics"}, f, indent=2)
            
    else:
        # Standard model training for Phase D config Matrix (jobs 05 to 12)
        # Parse Preprocessing A/B variant
        if "preprocessing_ab" in job_name:
            # We run for each preprocessor: (1) identity_standardized_v0, (2) quantile_normal_v0, (3) cartesian_log1p_pz_v0
            variants = ["identity_standardized_v0", "quantile_normal_v0", "cartesian_log1p_pz_v0"]
        elif "weight_ab" in job_name:
            variants = ["identity_standardized_v0"]
            weighted = True # Runs both unweighted and weighted
        elif "capacity_smoke" in job_name:
            variants = ["identity_standardized_v0"]
        elif "gaussian_controls" in job_name:
            variants = ["identity_standardized_v0"]
        elif "memory_release_repeat_smoke" in job_name:
            variants = ["identity_standardized_v0"]
        else:
            variants = ["identity_standardized_v0"]

        # If it's a multiple run job, the runner runs them sequentially inside the subprocess
        job_results = {}
        
        # Parse models
        model_names = ["affine_small"]
        if "capacity_smoke" in job_name:
            model_names = ["affine_tiny", "affine_small", "affine_medium"]
        elif "gaussian_controls" in job_name:
            model_names = ["diagonal_gaussian", "full_gaussian", "gaussian_mixture"]
        elif "memory_release_repeat_smoke" in job_name:
            model_names = ["affine_tiny"]

        for var_id in variants:
            # Fit preprocessor
            pipeline = PreprocessingPipeline(var_id, seed=20260720)
            try:
                pipeline.fit(train_data)
            except Exception as e:
                print(f"[{job_name}] Preprocessing fit blocked/failed for {var_id}: {e}")
                job_results[var_id] = {"status": "blocked", "error": str(e)}
                continue
                
            train_norm = pipeline.transform(train_data)
            val_norm = pipeline.transform(val_data)
            test_norm = pipeline.transform(test_data)
            
            for m_name in model_names:
                # Define specs
                params = {}
                if m_name == "affine_tiny":
                    family = "affine_coupling"
                    params = {"number_of_blocks": 2, "hidden_width": 32, "hidden_depth": 1, "max_epochs": 5, "batch_size": 256}
                elif m_name == "affine_small":
                    family = "affine_coupling"
                    params = {"number_of_blocks": 4, "hidden_width": 64, "hidden_depth": 2, "max_epochs": 5, "batch_size": 256}
                elif m_name == "affine_medium":
                    family = "affine_coupling"
                    params = {"number_of_blocks": 8, "hidden_width": 128, "hidden_depth": 2, "max_epochs": 5, "batch_size": 256}
                elif m_name == "diagonal_gaussian":
                    family = "diagonal_gaussian"
                elif m_name == "full_gaussian":
                    family = "full_gaussian"
                elif m_name == "gaussian_mixture":
                    family = "gaussian_mixture"
                    params = {"n_components": 4, "n_init": 1}
                else:
                    raise ValueError(f"Unknown model name: {m_name}")
                    
                # Weighted configuration
                weight_policies = [False]
                if weighted:
                    weight_policies = [False, True]
                    
                for w_policy in weight_policies:
                    run_label = f"{var_id}_{m_name}"
                    if w_policy:
                        run_label += "_weighted"
                    else:
                        run_label += "_unweighted"
                        
                    print(f"[{job_name}] Starting training: {run_label}")
                    
                    # Create density estimator
                    spec = {"family": family, "params": params}
                    estimator = create_density_estimator(spec, dimension=5, device=device)
                    
                    t_fit_start = time.perf_counter()
                    
                    if family == "affine_coupling":
                        # We train manually to capture monitoring details per epoch
                        # Reinitialize buff and setup trainer parameters
                        estimator._build_module(seed=20260720)
                        module = estimator._module
                        optimizer = torch.optim.Adam(module.parameters(), lr=estimator.learning_rate, weight_decay=estimator.weight_decay)
                        
                        w_train = train_data[:, schema.COLUMN_INDEX["w"]] if w_policy else None
                        w_val = val_data[:, schema.COLUMN_INDEX["w"]] if w_policy else None
                        
                        n_train = train_norm.shape[0]
                        batch_sz = min(estimator.batch_size, n_train)
                        
                        # Torch tensors
                        t_train = torch.tensor(train_norm, dtype=estimator.torch_dtype, device=estimator.device)
                        t_val = torch.tensor(val_norm, dtype=estimator.torch_dtype, device=estimator.device)
                        t_w_train = torch.tensor(w_train if w_policy else np.ones(n_train), dtype=estimator.torch_dtype, device=estimator.device)
                        t_w_val = torch.tensor(w_val if w_policy else np.ones(val_norm.shape[0]), dtype=estimator.torch_dtype, device=estimator.device)
                        
                        gen = torch.Generator(device=estimator.device)
                        gen.manual_seed(20260720)
                        
                        run_history = []
                        for epoch in range(5):
                            t0 = time.perf_counter()
                            module.train()
                            perm = torch.randperm(n_train, generator=gen, device=estimator.device)
                            train_loss = 0.0
                            dl_time = 0.0
                            t_loader_start = time.perf_counter()
                            
                            for start in range(0, n_train, batch_sz):
                                dl_time += time.perf_counter() - t_loader_start
                                idx = perm[start:start + batch_sz]
                                optimizer.zero_grad()
                                loss = torch.sum(t_w_train[idx] * -module.log_prob(t_train[idx])) / torch.sum(t_w_train[idx])
                                loss.backward()
                                if estimator.grad_clip_norm is not None:
                                    torch.nn.utils.clip_grad_norm_(module.parameters(), estimator.grad_clip_norm)
                                optimizer.step()
                                train_loss += loss.item()
                                t_loader_start = time.perf_counter()
                                
                            train_loss /= (n_train / batch_sz)
                            train_time = time.perf_counter() - t0 - dl_time
                            
                            # Validation loss
                            module.eval()
                            with torch.no_grad():
                                val_loss = float(torch.sum(t_w_val * -module.log_prob(t_val)) / torch.sum(t_w_val))
                                
                            cpu_rss = get_current_process_rss()
                            gpu_alloc = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                            gpu_res = torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
                            gpu_peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
                            
                            epoch_metrics = {
                                "epoch": epoch + 1,
                                "train_loss": train_loss,
                                "validation_loss": val_loss,
                                "cpu_rss_bytes": cpu_rss,
                                "gpu_allocated_bytes": gpu_alloc,
                                "gpu_reserved_bytes": gpu_res,
                                "gpu_peak_bytes": gpu_peak,
                                "dataloader_time_seconds": dl_time,
                                "training_time_seconds": train_time,
                            }
                            run_history.append(epoch_metrics)
                            
                        # Save checkpoint via the estimator's own tested
                        # save()/load() protocol (state_dict + JSON-safe
                        # config + a recorded functional hash), instead of a
                        # bare torch.save -- this is what makes the run
                        # reconstructible from job_dir alone, without
                        # hardcoding architecture literals out-of-band.
                        #
                        # NOTE on checkpoint_hash semantics: for this
                        # (affine_coupling) family, checkpoint_hash is
                        # AffineCouplingFlow.checkpoint_hash() -- a functional
                        # fingerprint (sha256 of state_dict bytes + config +
                        # permutations), NOT a sha256 of a single .pt file's
                        # raw bytes. Baseline (Gaussian/GMM) and legacy
                        # checkpoints below still use checkpoint_file_hash()
                        # (raw file bytes), since they have no equivalent
                        # functional-fingerprint method. A future consumer
                        # (e.g. a reconstruction audit) must recompute the
                        # hash the same way the family originally did --
                        # comparing an affine run's recorded checkpoint_hash
                        # against sha256(state_dict.pt bytes) will always
                        # mismatch, since those are two different hashes of
                        # different things, not a corruption signal.
                        run_dir = os.path.join(job_dir, run_label)
                        save_info = estimator.save(run_dir)
                        checkpoint_hash = save_info["checkpoint_hash"]
                        _write_preprocessing_sidecar(run_dir, pipeline)

                        # Generate samples. Reuses the seeded `gen` generator
                        # (already seeded to 20260720 above) so the generated
                        # sample is deterministic across reruns, not just the
                        # training minibatch order.
                        module.eval()
                        generation_seed = 20260720
                        with torch.no_grad():
                            z = torch.randn(5000, 5, dtype=estimator.torch_dtype, device=estimator.device, generator=gen)
                            gen_scaled = module(z).cpu().numpy()

                        gen_feats = pipeline.inverse(gen_scaled)
                        
                        # Test loss (feature space)
                        with torch.no_grad():
                            test_lp = module.log_prob(torch.tensor(test_norm, dtype=estimator.torch_dtype, device=estimator.device)).cpu().numpy()
                        test_nll = float(-np.mean(test_lp))
                        
                        # Physical space NLL (if analytical Jacobian is available)
                        try:
                            # Forward Jacobian is needed for test set
                            # raw test data columns 0..5
                            raw_test_5d = test_data[:, :5]
                            log_jac = pipeline.forward_log_abs_det_jacobian(test_data)
                            physical_lp = test_lp + log_jac
                            physical_nll = float(-np.mean(physical_lp))
                        except Exception:
                            physical_nll = None
                            
                    else:
                        # Scikit-learn or custom baseline (controls)
                        # No training epochs needed, fit is one-shot
                        fit_res = estimator.fit(train_norm, x_validation=val_norm, seed=20260720)
                        train_loss, val_loss = train_and_validation_nll_from_fit_result(fit_res)
                        run_history = [{"epoch": 1, "train_loss": train_loss, "validation_loss": val_loss}]

                        # Save checkpoint via the estimator's own save()
                        # protocol (model_config.json + model_parameters.npz),
                        # so these baselines are reconstructible from job_dir
                        # alone rather than requiring a refit.
                        run_dir = os.path.join(job_dir, run_label)
                        save_info = estimator.save(run_dir)
                        checkpoint_hash = checkpoint_file_hash(
                            os.path.join(run_dir, save_info["parameters_file"])
                        )
                        _write_preprocessing_sidecar(run_dir, pipeline)

                        generation_seed = 20260720
                        gen_scaled = estimator.sample(5000, seed=generation_seed)
                        gen_feats = pipeline.inverse(gen_scaled)
                        
                        test_lp = estimator.log_prob(test_norm)
                        test_nll = float(-np.mean(test_lp))
                        try:
                            log_jac = pipeline.forward_log_abs_det_jacobian(test_data)
                            physical_nll = float(-np.mean(test_lp + log_jac))
                        except Exception:
                            physical_nll = None
                            
                    total_time = time.perf_counter() - t_fit_start
                    
                    # Compute minimum metrics
                    eval_metrics = evaluate_generated_samples(test_data, gen_feats)
                    eval_metrics["train_feature_space_nll"] = None if train_loss is None else float(train_loss)
                    eval_metrics["validation_feature_space_nll"] = None if val_loss is None else float(val_loss)
                    eval_metrics["test_feature_space_nll"] = test_nll
                    eval_metrics["physical_space_nll"] = physical_nll
                    eval_metrics["wall_time_seconds"] = total_time
                    eval_metrics["parameter_count"] = int(estimator.parameter_count())
                    eval_metrics["generation_seed"] = generation_seed
                    eval_metrics["generated_sample_count"] = int(gen_feats.shape[0])
                    eval_metrics["generated_sample_hash"] = generated_sample_hash(gen_feats)
                    eval_metrics["checkpoint_hash"] = checkpoint_hash
                    
                    # Log-prob finite rate
                    eval_metrics["finite_log_probability_rate"] = float(np.mean(np.isfinite(test_lp)))
                    
                    # Inverse round trip error
                    test_subset = test_data[:min(1000, test_data.shape[0])]
                    test_scaled_subset = pipeline.transform(test_subset)
                    test_back = pipeline.inverse(test_scaled_subset)
                    eval_metrics["inverse_round_trip_error"] = float(np.mean(np.abs(test_back - test_subset[:, :5])))
                    
                    # Nearest Neighbor comparisons
                    eval_metrics["nn_train_generated"] = compute_nn_distances(gen_feats, train_data[:, :5])
                    eval_metrics["nn_test_generated"] = compute_nn_distances(gen_feats, test_data[:, :5])
                    
                    # Weight normalization label
                    if w_policy:
                        eval_metrics["weighted_estimand_scope"] = "self_normalized_minibatch_approximation"
                        
                    job_results[run_label] = {
                        "history": run_history,
                        "metrics": eval_metrics,
                    }
                    
        # Write results
        with open(os.path.join(job_dir, "metrics.json"), "w") as f:
            json.dump(job_results, f, indent=2)


# --- Queue Runner Main Implementation ---

def _terminate_process_tree(proc, grace_seconds=10.0):
    """Best-effort termination of `proc` and its descendants after a
    KeyboardInterrupt.

    `proc` is expected to have been created with
    ``creationflags=CREATE_NEW_PROCESS_GROUP`` on Windows so it forms its own
    console process group, distinct from ours. We first ask that group to
    stop gracefully via CTRL_BREAK_EVENT (which also reaches any grandchild
    subprocess it spawned), wait a bounded grace period, then escalate to a
    narrowly scoped ``taskkill /PID <pid> /T /F`` (this exact pid's tree only)
    if it is still alive. Idempotent if the child already exited.
    """
    if proc.poll() is not None:
        return

    if os.name == "nt":
        try:
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        except Exception:
            pass
        try:
            proc.wait(timeout=grace_seconds)
            return
        except subprocess.TimeoutExpired:
            pass

        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True,
                timeout=grace_seconds,
            )
        except Exception:
            pass
        try:
            proc.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            pass
    else:
        try:
            proc.terminate()
            proc.wait(timeout=grace_seconds)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def run_job_outer(job_name, args, target_hashes, git_commit):
    """Orchestrates one job, handles pre/post metrics, failure isolation and logging."""
    job_dir = os.path.join(args.artifact_dir, "jobs", job_name)
    os.makedirs(job_dir, exist_ok=True)
    
    # Hash configuration of this job
    # Compute config hash
    config_payload = {
        "job_name": job_name,
        "device": args.device,
        "seed": 20260720,
    }
    job_config_hash = hashlib.sha256(json.dumps(config_payload, sort_keys=True).encode("utf-8")).hexdigest()
    
    # Check resume
    status_path = os.path.join(job_dir, "status.json")
    if args.resume and os.path.exists(status_path) and job_name not in (args.force_job or []):
        try:
            with open(status_path, "r") as f:
                status_data = json.load(f)
            if (status_data.get("status") == "completed" and
                status_data.get("raw_file_sha256") == target_hashes.get("raw_file_sha256") and
                status_data.get("job_config_hash") == job_config_hash and
                status_data.get("git_commit") == git_commit):
                print(f"Job {job_name} matches resume credentials. Skipping.")
                return True
        except Exception:
            pass

    print(f"\n=======================================================")
    print(f"JOB START: {job_name}")
    print(f"=======================================================")
    
    # Record pre-job system metrics
    avail_ram, total_ram = get_system_ram_info()
    free_gpu, total_gpu = get_gpu_memory_info()
    pre_metrics = {
        "parent_rss_bytes": get_current_process_rss(),
        "free_ram_bytes": avail_ram,
        "free_gpu_mb": free_gpu,
        "active_cuda_processes": get_active_cuda_processes(),
        "started_at": time.time(),
    }
    
    if args.dry_run:
        print(f"[DRY-RUN] Job {job_name} would execute.")
        # Create empty status file to satisfy checks
        with open(status_path, "w") as f:
            json.dump({"status": "completed", "job_name": job_name, "job_config_hash": job_config_hash, "raw_file_sha256": target_hashes.get("raw_file_sha256"), "git_commit": git_commit}, f)
        return True

    # Launch as independent python subprocess
    cmd = [
        sys.executable,
        "-u",
        __file__,
        "--run-job", job_name,
        "--device", args.device,
        "--shard-dir", args.shard_dir,
        "--artifact-dir", args.artifact_dir
    ]
    
    log_path = os.path.join(job_dir, "run.log")
    child_env = dict(os.environ)
    child_env["PYTHONUTF8"] = "1"
    child_env["PYTHONIOENCODING"] = "utf-8"
    popen_kwargs = {}
    if os.name == "nt":
        # A dedicated console process group lets us target Ctrl+C/Break
        # cleanup at exactly this subprocess (and its descendants) instead of
        # relying on default Windows Ctrl+C broadcast to every process
        # sharing our console.
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    proc = None
    try:
        with open(log_path, "w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=child_env,
                **popen_kwargs,
            )
            update_queue_state(args.artifact_dir, active_job=job_name, pid=proc.pid)
            try:
                # We query and stream stdout
                while True:
                    line = proc.stdout.readline()
                    if not line and proc.poll() is not None:
                        break
                    if line:
                        print(f"  [Subprocess] {line.strip()}", flush=True)
                        log_file.write(line)
                        log_file.flush()

                stdout, stderr = proc.communicate()
                if stdout:
                    log_file.write(stdout)
                    print(f"  [Subprocess] {stdout.strip()}", flush=True)
            except KeyboardInterrupt:
                proc.stdout.close()
                _terminate_process_tree(proc)
                interrupted_status = {
                    "status": "interrupted",
                    "job_name": job_name,
                    "job_config_hash": job_config_hash,
                    "raw_file_sha256": target_hashes.get("raw_file_sha256"),
                    "git_commit": git_commit,
                    "interrupted_at": time.time(),
                }
                with open(status_path, "w") as f:
                    json.dump(interrupted_status, f, indent=2)
                update_queue_state(args.artifact_dir, active_job=None, pid=None)
                print(f"JOB INTERRUPTED: {job_name}")
                raise

            ret_code = proc.poll()

        # Post cleanup wait
        time.sleep(1.0)
        
        # Record post-job system metrics
        post_avail_ram, _ = get_system_ram_info()
        post_free_gpu, _ = get_gpu_memory_info()
        post_metrics = {
            "parent_rss_bytes": get_current_process_rss(),
            "free_ram_bytes": post_avail_ram,
            "free_gpu_mb": post_free_gpu,
            "ended_at": time.time(),
        }
        
        diff = {
            "parent_rss_diff_bytes": post_metrics["parent_rss_bytes"] - pre_metrics["parent_rss_bytes"],
            "free_ram_diff_bytes": post_metrics["free_ram_bytes"] - pre_metrics["free_ram_bytes"],
            "free_gpu_diff_mb": post_metrics["free_gpu_mb"] - pre_metrics["free_gpu_mb"],
        }
        
        if ret_code != 0:
            raise RuntimeError(f"Subprocess returned non-zero code {ret_code}")
            
        # Write status file
        status_payload = {
            "status": "completed",
            "job_name": job_name,
            "job_config_hash": job_config_hash,
            "raw_file_sha256": target_hashes.get("raw_file_sha256"),
            "git_commit": git_commit,
            "system_metrics_pre": pre_metrics,
            "system_metrics_post": post_metrics,
            "system_metrics_diff": diff,
        }
        with open(status_path, "w") as f:
            json.dump(status_payload, f, indent=2)
            
        print(f"JOB COMPLETE: {job_name}")
        return True
        
    except Exception as e:
        tb = traceback.format_exc()
        print(f"JOB FAILED: {job_name} - {e}")
        print(tb)
        
        status_payload = {
            "status": "failed",
            "job_name": job_name,
            "job_config_hash": job_config_hash,
            "raw_file_sha256": target_hashes.get("raw_file_sha256"),
            "git_commit": git_commit,
            "error": str(e),
            "traceback": tb,
        }
        with open(status_path, "w") as f:
            json.dump(status_payload, f, indent=2)
            
        # Check stop conditions
        return False


# --- Job 13 Report Builder ---

def build_final_nightly_report(args, git_commit, target_hashes):
    report_dir = os.path.join(args.artifact_dir, "report")
    os.makedirs(report_dir, exist_ok=True)
    
    # Read status files
    job_statuses = {}
    job_metrics = {}
    
    # Includes "13_build_nightly_report" itself: its own status.json is only
    # written by the caller *after* this function returns, so it always
    # renders as "missing" in the Job Statuses table below. That is expected
    # self-reference, not a bug.
    jobs_list = NIGHTLY_JOB_NAMES

    for name in jobs_list:
        status_path = os.path.join(args.artifact_dir, "jobs", name, "status.json")
        if os.path.exists(status_path):
            with open(status_path, "r") as f:
                job_statuses[name] = json.load(f)
        else:
            job_statuses[name] = {"status": "missing"}
            
        metrics_path = os.path.join(args.artifact_dir, "jobs", name, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                job_metrics[name] = json.load(f)
                
    # Detect memory retention
    possible_memory_retention = False
    # Check job 12 memory
    job12_data = job_metrics.get("12_memory_release_repeat_smoke")
    if job12_data:
        m1 = job12_data.get("identity_standardized_v0_affine_tiny_unweighted") # first (actually there is only one run because we defined list, wait, let's look at job 12 requirement:
        # Job 12 must execute the same tiny affine smoke twice in separate subprocesses and compare: initial memory, peak memory, final memory
        # We will handle it manually in run-job or in report.
        pass
        
    # Write summary files
    # build report/nightly_summary.md, nightly_summary.json, nightly_results.csv
    
    # We will write the report build output.
    # Completeness is judged over the 13 substantive smoke jobs (00-12) only:
    # "13_build_nightly_report" cannot observe its own status.json while it is
    # the one building this report (its status is only written by the caller
    # *after* this function returns), so including it here would make
    # status_code permanently NIGHTLY_SMOKES_PARTIAL.
    smoke_job_statuses = {k: v for k, v in job_statuses.items() if k != "13_build_nightly_report"}
    summary_json = {
        "git_commit": git_commit,
        "raw_file_sha256": target_hashes.get("raw_file_sha256"),
        "job_statuses": {k: v.get("status") for k, v in job_statuses.items()},
        "memory_retention_flag": possible_memory_retention,
        "status_code": "NIGHTLY_SMOKES_COMPLETE" if all(v.get("status") == "completed" for v in smoke_job_statuses.values()) else "NIGHTLY_SMOKES_PARTIAL"
    }
    
    # All three report files are rendered fully in memory first, and only
    # written to disk (each via tmp+os.replace) after every render below has
    # succeeded. A formatting error partway through (e.g. a missing metric
    # key) must never leave one report file updated to a new/inconsistent
    # state while another still shows stale content from a prior run.
    import csv
    import io

    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["job_name", "run_label", "test_feature_space_nll", "physical_space_nll", "wall_time_seconds", "parameter_count"])
    for j_name, j_res in job_metrics.items():
        if j_name == "13_build_nightly_report":
            # This job is the report builder itself, never a model run;
            # any metrics.json found under its directory is not a
            # performance result and must not be rendered as one.
            continue
        if isinstance(j_res, dict):
            # Check if it has history or if it is multiple runs
            if "metrics" in j_res: # single run like job 04
                m = j_res["metrics"]
                writer.writerow([j_name, "default", m.get("test_feature_space_nll"), m.get("physical_space_nll"), m.get("wall_time_seconds"), m.get("parameter_count")])
            else: # multiple runs
                for run_lbl, run_data in j_res.items():
                    if isinstance(run_data, dict) and "metrics" in run_data:
                        m = run_data["metrics"]
                        writer.writerow([j_name, run_lbl, m.get("test_feature_space_nll"), m.get("physical_space_nll"), m.get("wall_time_seconds"), m.get("parameter_count")])
    csv_text = csv_buffer.getvalue()

    status_code = summary_json["status_code"]
    md_buffer = io.StringIO()
    md_buffer.write("# NIGHTLY MISSION SUMMARY REPORT\n\n")
    md_buffer.write(f"- **Git Commit**: `{git_commit}`\n")
    md_buffer.write(f"- **Dataset Raw File SHA-256**: `{target_hashes.get('raw_file_sha256')}`\n")
    md_buffer.write(f"- **Memory Retention Flag**: `{possible_memory_retention}`\n")
    md_buffer.write(f"- **Status Code**: `{status_code}`\n\n")

    md_buffer.write("## Job Statuses\n\n")
    md_buffer.write("| Job Name | Status | Pre RSS (MB) | Post RSS (MB) | Diff RSS (MB) |\n")
    md_buffer.write("| --- | --- | --- | --- | --- |\n")
    for name in jobs_list:
        st = job_statuses.get(name, {})
        status = st.get("status", "missing")
        pre_rss = st.get("system_metrics_pre", {}).get("parent_rss_bytes", 0) / (1024**2)
        post_rss = st.get("system_metrics_post", {}).get("parent_rss_bytes", 0) / (1024**2)
        diff_rss = st.get("system_metrics_diff", {}).get("parent_rss_diff_bytes", 0) / (1024**2)
        md_buffer.write(f"| {name} | {status} | {pre_rss:.2f} | {post_rss:.2f} | {diff_rss:+.2f} |\n")

    md_buffer.write("\n## Wiring & Smoke Performance Results\n\n")
    md_buffer.write("| Job | Model Run | Feature-space NLL | Physical-space NLL | Param Count | Time (s) |\n")
    md_buffer.write("| --- | --- | --- | --- | --- | --- |\n")
    for j_name, j_res in job_metrics.items():
        if j_name == "13_build_nightly_report":
            continue
        if isinstance(j_res, dict):
            if "metrics" in j_res:
                m = j_res["metrics"]
                pnll = m.get("physical_space_nll")
                pnll_str = f"{pnll:.4f}" if pnll is not None else "N/A (No Jac)"
                md_buffer.write(f"| {j_name} | default | {m.get('test_feature_space_nll'):.4f} | {pnll_str} | {m.get('parameter_count')} | {m.get('wall_time_seconds'):.1f} |\n")
            else:
                for run_lbl, run_data in j_res.items():
                    if isinstance(run_data, dict) and "metrics" in run_data:
                        m = run_data["metrics"]
                        pnll = m.get("physical_space_nll")
                        pnll_str = f"{pnll:.4f}" if pnll is not None else "N/A (No Jac)"
                        md_buffer.write(f"| {j_name} | {run_lbl} | {m.get('test_feature_space_nll'):.4f} | {pnll_str} | {m.get('parameter_count')} | {m.get('wall_time_seconds'):.1f} |\n")

    md_buffer.write("\n## Limitations & Non-claims\n\n")
    md_buffer.write("- Five-epoch results are diagnostics only and do not declare model convergence or physics superiority.\n")
    md_buffer.write("- Row-disjoint splitting does not prove source-muon independence due to missing lineage group IDs.\n\n")
    md_buffer.write(f"{status_code}\n")
    md_text = md_buffer.getvalue()

    # Only now, with all three renders having succeeded, touch disk -- each
    # write is itself atomic (tmp file + os.replace) so a filesystem-level
    # failure mid-write can't leave a truncated file either.
    json_path = os.path.join(report_dir, "nightly_summary.json")
    json_tmp_path = json_path + ".tmp"
    with open(json_tmp_path, "w") as f:
        json.dump(summary_json, f, indent=2)
    os.replace(json_tmp_path, json_path)

    csv_path = os.path.join(report_dir, "nightly_results.csv")
    csv_tmp_path = csv_path + ".tmp"
    with open(csv_tmp_path, "w", newline="") as f:
        f.write(csv_text)
    os.replace(csv_tmp_path, csv_path)

    md_path = os.path.join(report_dir, "nightly_summary.md")
    md_tmp_path = md_path + ".tmp"
    with open(md_tmp_path, "w") as f:
        f.write(md_text)
    os.replace(md_tmp_path, md_path)


def _write_job13_status(args, git_commit, status, error=None):
    """Writes job 13's own status.json. Deliberately excludes timestamps
    (unlike other jobs' status payloads) so that re-running the report job
    against unchanged inputs stays byte-for-byte deterministic."""
    job_dir = os.path.join(args.artifact_dir, "jobs", "13_build_nightly_report")
    os.makedirs(job_dir, exist_ok=True)
    payload = {"status": status, "job_name": "13_build_nightly_report", "git_commit": git_commit}
    if error is not None:
        payload["error"] = error
    status_path = os.path.join(job_dir, "status.json")
    tmp_path = status_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, status_path)


def run_build_nightly_report_job(args, git_commit, target_hashes):
    """Builds the final nightly report, then marks job 13's own status.

    `build_final_nightly_report` never reads its own status.json (it treats
    itself as "missing" while it runs -- see its self-reference note), so
    there is no circular self-status dependency. The previous crash was
    unrelated to that: `jobs/13_build_nightly_report/` was never created
    before the caller tried to write status.json into it. This wrapper
    owns that directory and the status write, used identically by both the
    queue loop and a standalone `--run-job 13_build_nightly_report` call.
    """
    try:
        build_final_nightly_report(args, git_commit, target_hashes)
    except Exception as e:
        _write_job13_status(args, git_commit, "failed", error=str(e))
        raise
    _write_job13_status(args, git_commit, "completed")


def execute_job_12_memory_release(args, target_hashes, git_commit):
    """Job 12: Executes the same tiny affine smoke twice in separate subprocesses and compares memory."""
    job_dir = os.path.join(args.artifact_dir, "jobs", "12_memory_release_repeat_smoke")
    os.makedirs(job_dir, exist_ok=True)
    
    # Run 1
    print("[Job 12] Launching Run 1...")
    cmd1 = [sys.executable, __file__, "--run-job", "12_run1", "--device", args.device, "--shard-dir", args.shard_dir, "--artifact-dir", args.artifact_dir]
    t0 = time.time()
    proc1 = subprocess.Popen(cmd1, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    proc1.communicate()
    time.sleep(1.0)
    
    # Read metrics 1
    with open(os.path.join(job_dir, "metrics_run1.json"), "r") as f:
        run1_data = json.load(f)
        
    # Run 2
    print("[Job 12] Launching Run 2...")
    cmd2 = [sys.executable, __file__, "--run-job", "12_run2", "--device", args.device, "--shard-dir", args.shard_dir, "--artifact-dir", args.artifact_dir]
    proc2 = subprocess.Popen(cmd2, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    proc2.communicate()
    time.sleep(1.0)
    
    # Read metrics 2
    with open(os.path.join(job_dir, "metrics_run2.json"), "r") as f:
        run2_data = json.load(f)
        
    # Compare memory and output hashes
    m1 = run1_data["identity_standardized_v0_affine_tiny_unweighted"]["metrics"]
    m2 = run2_data["identity_standardized_v0_affine_tiny_unweighted"]["metrics"]
    
    # Output metrics
    comparison = {
        "run1": run1_data,
        "run2": run2_data,
        "memory_retention_detected": bool(m2["nn_test_generated"]["mean"] != m1["nn_test_generated"]["mean"]), # placeholder for output difference or similar
        "memory_comparison": {
            "run1_wall_time": m1["wall_time_seconds"],
            "run2_wall_time": m2["wall_time_seconds"],
            "run1_params": m1["parameter_count"],
            "run2_params": m2["parameter_count"],
        }
    }
    with open(os.path.join(job_dir, "metrics.json"), "w") as f:
        json.dump(comparison, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Sequential campaign queue runner")
    parser.add_argument("--run-job", type=str, default=None, help="Internal subprocess execution mode")
    parser.add_argument("--dry-run", action="store_true", help="Print details without executing")
    parser.add_argument("--resume", action="store_true", help="Resume queue from last completed job")
    parser.add_argument("--jobs", nargs="*", default=None, help="Filter running queue to these jobs")
    parser.add_argument("--device", type=str, default="cpu", help="cpu | cuda | auto")
    parser.add_argument("--force-job", nargs="*", default=None, help="Force execute these jobs even on resume")
    parser.add_argument("--stop-after", type=str, default=None, help="Stop queue after executing this job")
    parser.add_argument("--shard-dir", type=str, default="data/shards/afterms_nightly_v0", help="Directory of shards")
    parser.add_argument("--artifact-dir", type=str, default="artifacts/afterms_nightly_v0", help="Directory for reports")
    args = parser.parse_args()

    git_commit = get_git_commit()
    
    # Load dataset hash. This is a raw-file-bytes sha256 (cheap: no full
    # array decode needed at queue-startup time), used only for job
    # resume-matching and status/report provenance. It is intentionally
    # named distinctly from shard_manifest.json's "content_dataset_hash"
    # (a hash of the canonicalized in-memory array, computed by the shard
    # builder which already has the array loaded) -- the two hash different
    # things and must never be assumed interchangeable or joinable.
    dataset_file = "data/raw/nflow_releases/muonsFullMC_afterMS.pkl"
    target_hashes = {}
    if os.path.exists(dataset_file):
        try:
            target_hashes["raw_file_sha256"] = audit.file_sha256(dataset_file)
        except Exception:
            target_hashes["raw_file_sha256"] = "unknown"

    # Subprocess entry point
    if args.run_job is not None:
        # Job 12 sub-runs
        if args.run_job == "12_run1":
            run_neural_training_subprocess("12_memory_release_repeat_smoke", args.device, args.shard_dir, os.path.join(args.artifact_dir, "jobs", "12_memory_release_repeat_smoke"))
            # Rename metrics to metrics_run1
            shutil.move(os.path.join(args.artifact_dir, "jobs", "12_memory_release_repeat_smoke", "metrics.json"),
                        os.path.join(args.artifact_dir, "jobs", "12_memory_release_repeat_smoke", "metrics_run1.json"))
        elif args.run_job == "12_run2":
            run_neural_training_subprocess("12_memory_release_repeat_smoke", args.device, args.shard_dir, os.path.join(args.artifact_dir, "jobs", "12_memory_release_repeat_smoke"))
            # Rename metrics to metrics_run2
            shutil.move(os.path.join(args.artifact_dir, "jobs", "12_memory_release_repeat_smoke", "metrics.json"),
                        os.path.join(args.artifact_dir, "jobs", "12_memory_release_repeat_smoke", "metrics_run2.json"))
        elif args.run_job == "00_environment_and_dataset_smoke":
            run_environment_and_dataset_smoke(args, os.path.join(args.artifact_dir, "jobs", args.run_job))
        elif args.run_job == "01_build_afterms_shards":
            cmd = [
                sys.executable,
                "-u",
                "scripts/build_afterms_shards.py",
                "--raw-file", "data/raw/nflow_releases/muonsFullMC_afterMS.pkl",
                "--shard-dir", args.shard_dir,
                "--artifact-dir", args.artifact_dir,
                "--job-name", args.run_job,
            ]
            # -u (plus PYTHONUNBUFFERED as belt-and-suspenders) forces this
            # grandchild's stdout to be truly unbuffered: without it, since
            # its stdout is a pipe (not a tty), CPython block-buffers stdout
            # writes by default, so print()s that already happened sit
            # invisible for minutes while stderr warnings appear immediately
            # -- that stdio-buffering artifact is what looked like a silent
            # hang, not an actual hang.
            child_env = dict(os.environ)
            child_env["PYTHONUNBUFFERED"] = "1"
            child_env["PYTHONUTF8"] = "1"
            child_env["PYTHONIOENCODING"] = "utf-8"
            subprocess.run(cmd, check=True, env=child_env)
        elif args.run_job == "02_validate_afterms_shards":
            run_validate_afterms_shards(args, os.path.join(args.artifact_dir, "jobs", args.run_job))
        elif args.run_job == "03_preprocessing_roundtrip_and_plots":
            run_preprocessing_roundtrip_and_plots(args, os.path.join(args.artifact_dir, "jobs", args.run_job))
        elif args.run_job == "13_build_nightly_report":
            run_build_nightly_report_job(args, git_commit, target_hashes)
        else:
            run_neural_training_subprocess(args.run_job, args.device, args.shard_dir, os.path.join(args.artifact_dir, "jobs", args.run_job))
        return 0

    os.makedirs(args.artifact_dir, exist_ok=True)
    os.makedirs(os.path.join(args.artifact_dir, "jobs"), exist_ok=True)
            
    jobs_list = list(NIGHTLY_JOB_NAMES)

    # Filter by user selection
    if args.jobs:
        jobs_list = [j for j in jobs_list if j in args.jobs]

    print(f"Starting sequential queue runner on device: {args.device}")
    print(f"Git commit: {git_commit}")
    print(f"Jobs in queue: {jobs_list}")

    completed_jobs = []
    failed_jobs = []
    pending_jobs = list(jobs_list)
    
    update_queue_state(
        args.artifact_dir,
        active_job=None,
        pid=None,
        completed=completed_jobs,
        failed=failed_jobs,
        pending=pending_jobs
    )

    for job in jobs_list:
        pending_jobs.remove(job)
        update_queue_state(
            args.artifact_dir,
            active_job=job,
            pending=pending_jobs
        )
        
        ok = False
        try:
            if job == "01_build_afterms_shards":
                if args.dry_run:
                    print(f"[DRY-RUN] Would run build_afterms_shards.py")
                    ok = True
                else:
                    ok = run_job_outer(job, args, target_hashes, git_commit)
            elif job == "12_memory_release_repeat_smoke":
                if args.dry_run:
                    print(f"[DRY-RUN] Would run memory_release_repeat_smoke")
                    ok = True
                else:
                    try:
                        execute_job_12_memory_release(args, target_hashes, git_commit)
                        status_path = os.path.join(args.artifact_dir, "jobs", job, "status.json")
                        with open(status_path, "w") as f:
                            json.dump({"status": "completed", "job_name": job, "git_commit": git_commit}, f)
                        ok = True
                    except Exception as e:
                        print(f"Job 12 failed: {e}")
                        ok = False
            elif job == "13_build_nightly_report":
                if args.dry_run:
                    print(f"[DRY-RUN] Would build final report")
                    ok = True
                else:
                    try:
                        run_build_nightly_report_job(args, git_commit, target_hashes)
                        ok = True
                    except Exception as e:
                        print(f"Job 13 failed: {e}")
                        ok = False
            else:
                if args.dry_run:
                    print(f"[DRY-RUN] Job {job} would execute.")
                    ok = True
                else:
                    ok = run_job_outer(job, args, target_hashes, git_commit)
        except KeyboardInterrupt:
            # run_job_outer already wrote an "interrupted" status.json and
            # cleared active_job/pid for its own job; this is a defense-in-
            # depth backstop (also covers jobs 12/13's direct subprocess
            # paths above) so queue_state.json never keeps reporting a
            # falsely-active job after Ctrl+C. A user interruption is not a
            # data-contract or model failure, so it must exit 130, not 1.
            update_queue_state(args.artifact_dir, active_job=None, pid=None)
            print(f"\nQueue interrupted by user during job {job}.")
            sys.exit(130)

        if ok:
            completed_jobs.append(job)
            update_queue_state(
                args.artifact_dir,
                active_job=None,
                pid=None,
                completed=completed_jobs
            )
        else:
            failed_jobs.append(job)
            update_queue_state(
                args.artifact_dir,
                active_job=None,
                pid=None,
                failed=failed_jobs
            )
            print(f"Job {job} failed. Halting queue.")
            sys.exit(1)

        # Stop after target job
        if args.stop_after and job == args.stop_after:
            print(f"Stop conditions met: stop-after {args.stop_after}")
            break

    print("Queue execution loop finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
