#!/usr/bin/env python
"""Runtime micro-benchmark for density models (fit / log_prob / sample).

Reports median and dispersion over repeated timed runs, with warm-up and CUDA
synchronization where relevant. Separates fit, log_prob and sample throughput
and records batch size, dtype and device. Does NOT assert on wall time and does
NOT mix different hardware on one curve -- CPU may legitimately be faster for
these small 5D models.

    python scripts/benchmark_density_runtime.py --family affine_coupling --device cpu
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from Nflow.registry import create_density_estimator  # noqa: E402


def _sync(device: str) -> None:
    if device in ("cuda", "auto"):
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass


def _timeit(fn, *, repeats: int, device: str):
    fn()  # warm up
    _sync(device)
    times = []
    for _ in range(repeats):
        _sync(device)
        start = time.perf_counter()
        fn()
        _sync(device)
        times.append(time.perf_counter() - start)
    return {
        "median_seconds": statistics.median(times),
        "min_seconds": min(times),
        "max_seconds": max(times),
        "stdev_seconds": statistics.pstdev(times) if len(times) > 1 else 0.0,
        "repeats": repeats,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", default="affine_coupling")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-train", type=int, default=8192)
    parser.add_argument("--n-eval", type=int, default=8192)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--params", default="{}", help="JSON model params override")
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    x_train = rng.standard_normal((args.n_train, 5))
    x_eval = rng.standard_normal((args.n_eval, 5))
    params = json.loads(args.params)
    model = create_density_estimator(
        {"family": args.family, "params": params}, dimension=5, device=args.device
    )

    fit_time = _timeit(
        lambda: model.fit(x_train, x_validation=None, seed=0),
        repeats=max(1, args.repeats // 2),
        device=args.device,
    )
    logprob_time = _timeit(lambda: model.log_prob(x_eval), repeats=args.repeats, device=args.device)
    sample_time = _timeit(lambda: model.sample(args.n_eval, seed=0), repeats=args.repeats, device=args.device)

    report = {
        "family": args.family,
        "device": args.device,
        "batch_size": args.n_eval,
        "dtype": params.get("dtype", "default"),
        "parameter_count": int(model.parameter_count()),
        "fit": fit_time,
        "log_prob": logprob_time,
        "sample": sample_time,
        "note": "CPU may be faster than GPU for small 5D models; not an error.",
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
