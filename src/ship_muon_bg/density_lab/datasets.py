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

from ..benchmarks import embed_physical_to_raw, make_controlled_target
from ..benchmarks.controlled_targets import TransformedControlledTarget
from ..data_contracts import dataset_hash


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
        return {
            "target_id": self.target_id,
            "target_variant": self.target_variant,
            "target_config_hash": self.target_config_hash,
            "pdg_id": self.pdg_id,
            "base_seed": self.base_seed,
            "partitions": {
                "train": self.train.manifest(),
                "validation": self.validation.manifest(),
                "test_nominal": self.test_nominal.manifest(),
            },
        }


def _make_partition(
    target, *, partition, pdg_id, n_rows, seed, region_id
) -> DatasetPartition:
    batch = target.sample(n_rows, pdg_id=pdg_id, seed=seed)
    physical = batch.physical
    raw = embed_physical_to_raw(physical, pdg_id=pdg_id, plane_z=0.0)
    rare_mask = None
    if region_id is not None:
        rare_mask = target.region_mask(physical, pdg_id=pdg_id, region_id=region_id)
    return DatasetPartition(
        partition=partition,
        physical=physical,
        component_id=batch.component_id,
        rare_region_mask=rare_mask,
        seed=seed,
        raw_dataset_hash=dataset_hash(raw),
        n_rows=int(n_rows),
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
) -> ControlledDataset:
    """Build independent train/validation/test partitions for one target arm."""

    target = make_controlled_target(target_id, variant=variant)
    region_id = None
    if isinstance(target, TransformedControlledTarget) and target.declared_regions():
        region_id = target.declared_regions()[0]
    train_seed, val_seed, test_seed = _partition_seeds(seed)
    return ControlledDataset(
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
        ),
        validation=_make_partition(
            target,
            partition="validation",
            pdg_id=pdg_id,
            n_rows=n_validation,
            seed=val_seed,
            region_id=region_id,
        ),
        test_nominal=_make_partition(
            target,
            partition="test_nominal",
            pdg_id=pdg_id,
            n_rows=n_test,
            seed=test_seed,
            region_id=region_id,
        ),
    )
