"""Controlled density laboratory (v0).

A modular, reproducible lab for fitting and evaluating density models against
the exact controlled targets D0-D5. The public import path is NumPy-only:
importing ``ship_muon_bg.density_lab`` must not import torch, scikit-learn,
matplotlib or mlflow. Heavy dependencies are imported lazily inside the
functions that need them (model registry, C2ST metric, reporting, tracking).
"""

from __future__ import annotations

from .artifacts import ArtifactStore, derive_run_id
from .campaign import run_campaign, run_single
from .config import (
    CONFIG_SCHEMA_VERSION,
    PREDEFINED_SCIENTIFIC_SEEDS,
    DatasetSpec,
    EvaluationSpec,
    ExperimentConfig,
    FeatureViewSpec,
    ModelSpec,
    ResourceSpec,
    RunSpec,
    SamplingSpec,
    TargetSpec,
    TrackingSpec,
)
from .datasets import ControlledDataset, build_controlled_dataset
from .evaluator import evaluate_run
from .feature_pipeline import FeaturePipelineError, FittedFeaturePipeline
from .gates import (
    DECISION_SCOPE,
    GATE_SCHEMA_VERSION,
    SCIENTIFIC_STATUSES,
    STATUS_UNAVAILABLE,
    ScientificGateResult,
    ScientificGateSpec,
    evaluate_scientific_gates,
)
from .sampling import (
    IID_TARGET,
    SAMPLING_REGIMES,
    STRATIFIED_SELF_NORMALIZED_PROVISIONAL,
    STRATIFIED_DIAGNOSTIC,
    sample_controlled,
)
from .doe import generate_blocked_maximin_lhs

__all__ = [
    "FittedFeaturePipeline",
    "FeaturePipelineError",
    "ExperimentConfig",
    "RunSpec",
    "TargetSpec",
    "FeatureViewSpec",
    "ModelSpec",
    "DatasetSpec",
    "EvaluationSpec",
    "TrackingSpec",
    "ResourceSpec",
    "SamplingSpec",
    "ScientificGateSpec",
    "ScientificGateResult",
    "evaluate_scientific_gates",
    "GATE_SCHEMA_VERSION",
    "DECISION_SCOPE",
    "SCIENTIFIC_STATUSES",
    "STATUS_UNAVAILABLE",
    "CONFIG_SCHEMA_VERSION",
    "PREDEFINED_SCIENTIFIC_SEEDS",
    "build_controlled_dataset",
    "ControlledDataset",
    "evaluate_run",
    "ArtifactStore",
    "derive_run_id",
    "run_campaign",
    "run_single",
    "IID_TARGET",
    "STRATIFIED_DIAGNOSTIC",
    "STRATIFIED_SELF_NORMALIZED_PROVISIONAL",
    "SAMPLING_REGIMES",
    "sample_controlled",
    "generate_blocked_maximin_lhs",
]
