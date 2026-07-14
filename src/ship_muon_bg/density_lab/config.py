"""Typed experiment configuration for the controlled density lab.

JSON configs deserialize into frozen dataclasses. Every config validates
before execution, serializes canonically, has a deterministic SHA-256 hash and
records a schema version. No Hydra.

An :class:`ExperimentConfig` describes a campaign as a set of axis lists
(targets, pdg ids, feature views, models, seeds). :meth:`ExperimentConfig.runs`
expands the Cartesian product into :class:`RunSpec` points; each run's
``run_id`` is derived from canonical configuration hashes (see
``artifacts.py``), not a timestamp.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .gates import ScientificGateSpec

CONFIG_SCHEMA_VERSION = "0"

PREDEFINED_SCIENTIFIC_SEEDS: Tuple[int, ...] = (11, 22, 33, 44, 55)


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


class ConfigError(ValueError):
    """An invalid experiment configuration."""


@dataclass(frozen=True)
class DatasetSpec:
    n_train: int = 4096
    n_validation: int = 2048
    n_test: int = 4096

    def validate(self) -> None:
        for name in ("n_train", "n_validation", "n_test"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 2:
                raise ConfigError("DatasetSpec.{} must be an int >= 2".format(name))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TargetSpec:
    target_id: str
    variant: Optional[str] = None

    def validate(self) -> None:
        if not isinstance(self.target_id, str) or not self.target_id:
            raise ConfigError("TargetSpec.target_id must be a non-empty string")

    def to_dict(self) -> Dict[str, Any]:
        return {"target_id": self.target_id, "variant": self.variant}


@dataclass(frozen=True)
class FeatureViewSpec:
    view_id: str
    pz_unit_gev: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"view_id": self.view_id, "pz_unit_gev": self.pz_unit_gev}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    params: Dict[str, Any] = field(default_factory=dict)
    training_budget_id: str = "default"

    def validate(self) -> None:
        if not isinstance(self.family, str) or not self.family:
            raise ConfigError("ModelSpec.family must be a non-empty string")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "params": dict(self.params),
            "training_budget_id": self.training_budget_id,
        }

    def config_hash(self) -> str:
        return canonical_hash(self.to_dict())


@dataclass(frozen=True)
class EvaluationSpec:
    ess_sample_count: int = 20000
    c2st_sample_count: int = 4000
    tail_quantiles: Tuple[float, ...] = (0.9, 0.99, 0.999)
    exceedance_pz_thresholds: Tuple[float, ...] = (60.0, 70.0)
    catastrophic_ess_threshold: float = 0.01
    near_duplicate_atol: float = 1e-6

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["tail_quantiles"] = list(self.tail_quantiles)
        d["exceedance_pz_thresholds"] = list(self.exceedance_pz_thresholds)
        return d

    def config_hash(self) -> str:
        return canonical_hash(self.to_dict())


@dataclass(frozen=True)
class TrackingSpec:
    mode: str = "local"  # "local" | "mlflow"
    experiment_name: str = "controlled_density_lab"

    def validate(self) -> None:
        if self.mode not in ("local", "mlflow"):
            raise ConfigError("TrackingSpec.mode must be 'local' or 'mlflow'")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResourceSpec:
    device: str = "cpu"  # "cpu" | "cuda" | "auto"

    def validate(self) -> None:
        if self.device not in ("cpu", "cuda", "auto"):
            raise ConfigError("ResourceSpec.device must be 'cpu', 'cuda' or 'auto'")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunSpec:
    """One campaign matrix point."""

    experiment_id: str
    target: TargetSpec
    pdg_id: int
    feature_view: FeatureViewSpec
    model: ModelSpec
    seed: int
    dataset: DatasetSpec
    evaluation: EvaluationSpec
    device: str
    scientific_gates: ScientificGateSpec = field(default_factory=ScientificGateSpec)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "experiment_id": self.experiment_id,
            "target": self.target.to_dict(),
            "pdg_id": self.pdg_id,
            "feature_view": self.feature_view.to_dict(),
            "model": self.model.to_dict(),
            "seed": self.seed,
            "dataset": self.dataset.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "device": self.device,
            "scientific_gates": self.scientific_gates.to_dict(),
        }

    def resolved_gate_spec(self) -> ScientificGateSpec:
        """Gate spec with its ESS threshold resolved from the evaluation config."""

        return self.scientific_gates.resolve(self.evaluation)

    def config_hash(self) -> str:
        return canonical_hash(self.to_dict())


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    targets: List[TargetSpec]
    pdg_ids: List[int]
    feature_views: List[FeatureViewSpec]
    models: List[ModelSpec]
    seeds: List[int]
    dataset: DatasetSpec = field(default_factory=DatasetSpec)
    evaluation: EvaluationSpec = field(default_factory=EvaluationSpec)
    tracking: TrackingSpec = field(default_factory=TrackingSpec)
    resources: ResourceSpec = field(default_factory=ResourceSpec)
    scientific_gates: ScientificGateSpec = field(default_factory=ScientificGateSpec)
    description: str = ""
    schema_version: str = CONFIG_SCHEMA_VERSION

    def validate(self) -> None:
        if not self.experiment_id:
            raise ConfigError("experiment_id must be non-empty")
        for group_name in ("targets", "pdg_ids", "feature_views", "models", "seeds"):
            if not getattr(self, group_name):
                raise ConfigError("{} must be non-empty".format(group_name))
        self.dataset.validate()
        self.tracking.validate()
        self.resources.validate()
        # The ESS catastrophic threshold has one source of truth: EvaluationSpec.
        # The gate spec may only leave it None (inherit) or set an equal value.
        try:
            self.scientific_gates.validate(self.evaluation)
        except Exception as exc:  # normalize to ConfigError for callers
            raise ConfigError(str(exc)) from exc
        for target in self.targets:
            target.validate()
        for model in self.models:
            model.validate()
        for pdg_id in self.pdg_ids:
            if pdg_id not in (13, -13):
                raise ConfigError("pdg_id must be 13 or -13, got {}".format(pdg_id))

    def runs(self) -> List[RunSpec]:
        self.validate()
        runs: List[RunSpec] = []
        for target in self.targets:
            for pdg_id in self.pdg_ids:
                for view in self.feature_views:
                    for model in self.models:
                        for seed in self.seeds:
                            runs.append(
                                RunSpec(
                                    experiment_id=self.experiment_id,
                                    target=target,
                                    pdg_id=int(pdg_id),
                                    feature_view=view,
                                    model=model,
                                    seed=int(seed),
                                    dataset=self.dataset,
                                    evaluation=self.evaluation,
                                    device=self.resources.device,
                                    scientific_gates=self.scientific_gates,
                                )
                            )
        return runs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "description": self.description,
            "targets": [t.to_dict() for t in self.targets],
            "pdg_ids": list(self.pdg_ids),
            "feature_views": [v.to_dict() for v in self.feature_views],
            "models": [m.to_dict() for m in self.models],
            "seeds": list(self.seeds),
            "dataset": self.dataset.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "tracking": self.tracking.to_dict(),
            "resources": self.resources.to_dict(),
            "scientific_gates": self.scientific_gates.to_dict(),
        }

    def config_hash(self) -> str:
        return canonical_hash(self.to_dict())

    # -- (de)serialization ---------------------------------------------------

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ExperimentConfig":
        try:
            config = cls(
                experiment_id=payload["experiment_id"],
                description=payload.get("description", ""),
                targets=[
                    TargetSpec(t["target_id"], t.get("variant"))
                    for t in payload["targets"]
                ],
                pdg_ids=[int(p) for p in payload["pdg_ids"]],
                feature_views=[
                    FeatureViewSpec(v["view_id"], v.get("pz_unit_gev"))
                    for v in payload["feature_views"]
                ],
                models=[
                    ModelSpec(
                        name=m["name"],
                        family=m["family"],
                        params=dict(m.get("params", {})),
                        training_budget_id=m.get("training_budget_id", "default"),
                    )
                    for m in payload["models"]
                ],
                seeds=[int(s) for s in payload["seeds"]],
                dataset=DatasetSpec(**payload.get("dataset", {})),
                evaluation=EvaluationSpec(
                    **{
                        k: (tuple(v) if isinstance(v, list) else v)
                        for k, v in payload.get("evaluation", {}).items()
                    }
                ),
                tracking=TrackingSpec(**payload.get("tracking", {})),
                resources=ResourceSpec(**payload.get("resources", {})),
                scientific_gates=ScientificGateSpec(
                    **payload.get("scientific_gates", {})
                ),
            )
        except KeyError as exc:
            raise ConfigError("missing required config key: {}".format(exc)) from exc
        config.validate()
        return config

    @classmethod
    def from_json_file(cls, path) -> "ExperimentConfig":
        from pathlib import Path

        payload = json.loads(Path(path).read_text())
        return cls.from_dict(payload)
