"""Controlled density laboratory (v0).

A modular, reproducible lab for fitting and evaluating density models against
the exact controlled targets D0-D5. The public import path is NumPy-only:
importing ``ship_muon_bg.density_lab`` must not import torch, scikit-learn,
matplotlib or mlflow. Heavy dependencies are imported lazily inside the
functions that need them (model registry, C2ST metric, reporting, tracking).
"""

from __future__ import annotations

from .feature_pipeline import FeaturePipelineError, FittedFeaturePipeline

__all__ = [
    "FittedFeaturePipeline",
    "FeaturePipelineError",
]
