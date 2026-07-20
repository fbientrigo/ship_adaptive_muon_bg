"""Density-lab target-stage resolution without changing benchmark contracts."""

from __future__ import annotations

from typing import Optional

from ..benchmarks import make_controlled_target


class D5BaseStageTarget:
    """D5 labelled mixture before the D4 skew/banana transform."""

    def __init__(self, transformed) -> None:
        self._source = transformed
        self._base = transformed._base
        self.target_id = "D5_base"
        self.target_variant = transformed.target_variant
        self.rare_mass = transformed.rare_mass

    def sample(self, n, *, pdg_id, seed):
        return self._base.sample(n, pdg_id=pdg_id, seed=seed)

    def log_prob(self, physical, *, pdg_id):
        return self._base.log_prob(physical, pdg_id=pdg_id)

    def component_posterior(self, physical, *, pdg_id):
        return self._base.component_posterior(physical, pdg_id=pdg_id)

    def rare_component_id(self, *, pdg_id):
        return self._source.rare_component_id(pdg_id=pdg_id)

    def declared_regions(self):
        return self._source.declared_regions()

    def region_mask(self, physical, *, pdg_id, region_id):
        return self._source._regions_by_pdg_id[pdg_id][region_id].contains(physical)

    def manifest(self):
        payload = dict(self._source.manifest())
        payload.update({"target_id": self.target_id, "transform_stage": "base_before_d4"})
        return payload

    def config_hash(self):
        from .config import canonical_hash

        return canonical_hash(self.manifest())


def resolve_target(target_id: str, variant: Optional[str] = None, stage: str = "transformed"):
    """Resolve D3/D5 transformed targets or the labelled pre-transform D5 base."""

    if stage not in ("transformed", "base_before_d4"):
        raise ValueError("target stage must be 'transformed' or 'base_before_d4'")
    target = make_controlled_target(target_id, variant=variant)
    if stage == "base_before_d4":
        if target_id != "D5":
            raise ValueError("base_before_d4 is defined only for D5")
        return D5BaseStageTarget(target)
    return target
