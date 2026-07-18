"""Controlled benchmark datasets: independent train/validation/test draws.

For a given ``(target, variant, pdg_id, seed)`` the builder draws three
*independent* partitions (train / validation / test_nominal) from the exact
target. Partitions are view-independent physical rows, so matched A/B1/B2 runs
(different feature views) operate on exactly the same underlying physical rows.

Rare rows are never forced or oversampled: ``test_nominal`` is a plain draw
from the nominal target. Each partition records its provenance (target hash,
seed, raw dataset hash, component labels, rare-region mask where available).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from ..benchmarks import embed_physical_to_raw
from ..data_contracts import dataset_hash
from .sampling import IID_TARGET, sample_controlled
from .targets import resolve_target


def _partition_seeds(seed: int):
    """Return three independent integer seeds derived from ``seed``."""

    children = np.random.SeedSequence(int(seed)).spawn(3)
    return tuple(int(child.generate_state(1)[0]) for child in children)


@dataclass(frozen=True)
class DatasetPartition:
    partition: str
    physical: np.ndarray
    component_id: np.ndarray
    rare_region_mask: Optional[np.ndarray]
    seed: int
    raw_dataset_hash: str
    n_rows: int
    sample_weight: np.ndarray
    sampling_manifest: Dict[str, Any]

    def manifest(self) -> Dict[str, Any]:
        return {
            "partition": self.partition,
            "n_rows": self.n_rows,
            "seed": self.seed,
            "raw_dataset_hash": self.raw_dataset_hash,
            "has_component_labels": self.component_id is not None,
            "has_rare_region_mask": self.rare_region_mask is not None,
            "rare_region_count": (
                int(self.rare_region_mask.sum())
                if self.rare_region_mask is not None
                else None
            ),
            "sampling": self.sampling_manifest,
        }


@dataclass(frozen=True)
class ControlledDataset:
    target_id: str
    target_variant: Optional[str]
    target_config_hash: str
    pdg_id: int
    base_seed: int
    train: DatasetPartition
    validation: DatasetPartition
    test_nominal: DatasetPartition

    def manifest(self) -> Dict[str, Any]:
        hashes = [
            self.train.raw_dataset_hash,
            self.validation.raw_dataset_hash,
            self.test_nominal.raw_dataset_hash,
        ]
        return {
            "target_id": self.target_id,
            "target_variant": self.target_variant,
            "target_config_hash": self.target_config_hash,
            "pdg_id": self.pdg_id,
            "base_seed": self.base_seed,
            "validation_no_leakage": len(set(hashes)) == 3,
            "partitions": {
                "train": self.train.manifest(),
                "validation": self.validation.manifest(),
                "test_nominal": self.test_nominal.manifest(),
            },
        }


def _make_partition(
    target, *, partition, pdg_id, n_rows, seed, region_id, regime,
    sampling_rare_fraction,
) -> DatasetPartition:
    sampled = sample_controlled(
        target, pdg_id=pdg_id, n=n_rows, seed=seed, regime=regime,
        sampling_rare_fraction=sampling_rare_fraction,
    )
    physical = sampled.physical
    raw = embed_physical_to_raw(physical, pdg_id=pdg_id, plane_z=0.0)
    rare_mask = None
    if region_id is not None:
        rare_mask = target.region_mask(physical, pdg_id=pdg_id, region_id=region_id)
    return DatasetPartition(
        partition=partition,
        physical=physical,
        component_id=sampled.component_id,
        rare_region_mask=rare_mask,
        seed=seed,
        raw_dataset_hash=dataset_hash(raw),
        n_rows=int(n_rows),
        sample_weight=sampled.sample_weight,
        sampling_manifest=sampled.manifest,
    )


def build_controlled_dataset(
    *,
    target_id: str,
    variant: Optional[str],
    pdg_id: int,
    n_train: int,
    n_validation: int,
    n_test: int,
    seed: int,
    regime: str = IID_TARGET,
    sampling_rare_fraction: Optional[float] = None,
    target_stage: str = "transformed",
) -> ControlledDataset:
    """Build independent train/validation/test partitions for one target arm."""

    target = resolve_target(target_id, variant=variant, stage=target_stage)
    region_id = None
    if hasattr(target, "declared_regions") and target.declared_regions():
        region_id = target.declared_regions()[0]
    train_seed, val_seed, test_seed = _partition_seeds(seed)
    dataset = ControlledDataset(
        target_id=target_id,
        target_variant=getattr(target, "target_variant", variant),
        target_config_hash=target.config_hash(),
        pdg_id=int(pdg_id),
        base_seed=int(seed),
        train=_make_partition(
            target,
            partition="train",
            pdg_id=pdg_id,
            n_rows=n_train,
            seed=train_seed,
            region_id=region_id,
            regime=regime,
            sampling_rare_fraction=sampling_rare_fraction,
        ),
        validation=_make_partition(
            target,
            partition="validation",
            pdg_id=pdg_id,
            n_rows=n_validation,
            seed=val_seed,
            region_id=region_id,
            regime=regime,
            sampling_rare_fraction=sampling_rare_fraction,
        ),
        test_nominal=_make_partition(
            target,
            partition="test_nominal",
            pdg_id=pdg_id,
            n_rows=n_test,
            seed=test_seed,
            region_id=region_id,
            regime=IID_TARGET,
            sampling_rare_fraction=None,
        ),
    )
    if not dataset.manifest()["validation_no_leakage"]:
        raise RuntimeError("train/validation/test dataset hash collision")
    return dataset
