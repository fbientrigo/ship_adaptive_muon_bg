"""Presentation-only track/model-family/model-config aliases (D9C).

Legacy candidate IDs such as ``A1``/``B2`` are opaque and were never meant to
be read as model names -- they identify a historical selection decision, not
an empirical track or an architecture. This module derives a small, stable
alias layer on top of them:

``track_id``
    which empirical distribution/estimand is being modeled (derived from
    ``pdg_value`` + ``weighting_policy`` + ``preprocessing_name``, never from
    the legacy candidate id itself).
``model_family_id``
    which generative model family (``NF_AC`` / ``GAUSS_DIAG`` / ``GAUSS_FULL``
    / ``GMM``).
``model_config_id``
    which concrete architecture (e.g. ``NF_AC_b08_w128_d02``), derived from
    the actual architecture values a run used, never from a capacity label or
    scale multiplier.

Everything here is pure presentation/lineage metadata read from already-frozen
configs, run manifests, or producer source. Nothing in this module trains,
refits, or mutates a model, and nothing it returns may be folded back into a
hashed training config -- see ``d9/contract.py``: ``semantic_training_hash``
hashes ``candidate_config`` wholesale, so an alias string injected into that
dict would silently change resume compatibility. Aliases are always looked up
alongside a config, never inside one.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

DEFAULT_REGISTRY_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "afterms" / "model_alias_registry_v0.json"
)

_GENERIC_FALLBACK = "GENERIC_FALLBACK"
_CURATED = "CURATED"
_UNRESOLVED = "UNRESOLVED"

_NF_AC_CONFIG_RE = re.compile(r"^NF_AC_b(\d+)_w(\d+)_d(\d+)$")


class AliasResolutionError(ValueError):
    """An alias could not be resolved from the evidence given."""


class ModelConfigCollisionError(ValueError):
    """Two distinct architectures resolved to the same ``model_config_id``."""


def load_alias_registry(path: Optional[Path] = None) -> Dict[str, Any]:
    registry_path = Path(path) if path is not None else DEFAULT_REGISTRY_PATH
    return json.loads(registry_path.read_text(encoding="utf-8"))


def _slug_upper(value: Any) -> str:
    text = str(value).strip().upper()
    return re.sub(r"[^A-Z0-9]+", "_", text).strip("_")


# ---------------------------------------------------------------------------
# Track resolution
# ---------------------------------------------------------------------------


def resolve_track(
    registry: Dict[str, Any],
    *,
    pdg_value: Optional[int],
    weighting_policy: str,
    preprocessing_name: str,
) -> Dict[str, Any]:
    """Resolve ``(pdg_value, weighting_policy, preprocessing_name)`` to a
    ``track_id`` + ``track_label``.

    Two candidates with the same three fields always resolve to the same
    track, regardless of legacy candidate id, model family, or architecture.
    Values present in the registry get short curated codes; unrecognized
    values fall back to a deterministic slug of the raw field so the
    function never raises on legitimate but not-yet-curated data (e.g.
    excluded-candidate preprocessing choices) -- callers can inspect
    ``resolution_status`` to see which path was taken.
    """

    pdg_key = "null" if pdg_value is None else str(int(pdg_value))
    pdg_codes = registry.get("pdg_codes", {})
    pdg_labels = registry.get("pdg_labels", {})
    weighting_codes = registry.get("weighting_codes", {})
    weighting_labels = registry.get("weighting_labels", {})
    preprocessing_codes = registry.get("preprocessing_codes", {})
    preprocessing_labels = registry.get("preprocessing_labels", {})

    curated = True

    if pdg_key in pdg_codes:
        pdg_code = pdg_codes[pdg_key]
        pdg_label = pdg_labels.get(pdg_key, pdg_key)
    else:
        curated = False
        pdg_code = "PDGNA" if pdg_value is None else f"PDG{_slug_upper(pdg_value)}"
        pdg_label = "PDG (combined/unspecified)" if pdg_value is None else f"PDG {pdg_value}"

    if weighting_policy in weighting_codes:
        weighting_code = weighting_codes[weighting_policy]
        weighting_label = weighting_labels.get(weighting_policy, weighting_policy)
    else:
        curated = False
        weighting_code = _slug_upper(weighting_policy)
        weighting_label = str(weighting_policy)

    if preprocessing_name in preprocessing_codes:
        preprocessing_code = preprocessing_codes[preprocessing_name]
        preprocessing_label = preprocessing_labels.get(preprocessing_name, preprocessing_name)
    else:
        curated = False
        preprocessing_code = _slug_upper(preprocessing_name)
        preprocessing_label = str(preprocessing_name)

    track_id = f"TRK_{pdg_code}_{weighting_code}_{preprocessing_code}"
    track_label = f"{pdg_label} · {weighting_label} · {preprocessing_label}"

    return {
        "track_id": track_id,
        "track_label": track_label,
        "alias_resolution_status": _CURATED if curated else _GENERIC_FALLBACK,
    }


# ---------------------------------------------------------------------------
# NF_AC (affine coupling) model-config resolution
# ---------------------------------------------------------------------------


def nf_ac_model_config_id(*, number_of_blocks: int, hidden_width: int, hidden_depth: int) -> str:
    return f"NF_AC_b{int(number_of_blocks):02d}_w{int(hidden_width):03d}_d{int(hidden_depth):02d}"


def parse_nf_ac_model_config_id(model_config_id: str) -> Dict[str, int]:
    match = _NF_AC_CONFIG_RE.match(model_config_id)
    if not match:
        raise AliasResolutionError(f"not a well-formed NF_AC model_config_id: {model_config_id!r}")
    blocks, width, depth = match.groups()
    return {
        "number_of_blocks": int(blocks),
        "hidden_width": int(width),
        "hidden_depth": int(depth),
    }


def nf_ac_model_config(architecture: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve an NF_AC ``model_config_id``/label/params from an actual
    architecture dict (as read from a training config or run manifest, never
    from a capacity label or scale multiplier)."""

    blocks = architecture["number_of_blocks"]
    width = architecture["hidden_width"]
    depth = architecture["hidden_depth"]
    model_config_id = nf_ac_model_config_id(number_of_blocks=blocks, hidden_width=width, hidden_depth=depth)
    # Round-trip through the parser: a genuine assertion that this specific
    # (blocks, width, depth) triple is exactly recoverable from the id string,
    # not merely "probably injective by inspection" of the format.
    parsed = parse_nf_ac_model_config_id(model_config_id)
    if parsed != {"number_of_blocks": int(blocks), "hidden_width": int(width), "hidden_depth": int(depth)}:
        raise ModelConfigCollisionError(
            f"NF_AC model_config_id {model_config_id!r} does not round-trip to ({blocks}, {width}, {depth})"
        )
    label = f"Affine-coupling NF, {blocks} blocks, width {width}, depth {depth}"
    return {
        "model_family_id": "NF_AC",
        "model_family_label": "Affine-coupling NF",
        "model_config_id": model_config_id,
        "model_config_label": label,
        "architecture_parameters": {
            "number_of_blocks": int(blocks),
            "hidden_width": int(width),
            "hidden_depth": int(depth),
        },
    }


def assert_no_model_config_collisions(records: Sequence[Dict[str, Any]]) -> None:
    """Raise if two distinct architectures resolved to the same
    ``model_config_id`` anywhere in ``records``. Same architecture on
    different tracks is expected and NOT a collision."""

    seen: Dict[str, Dict[str, Any]] = {}
    for record in records:
        config_id = record.get("model_config_id")
        arch = record.get("architecture_parameters")
        if config_id is None or arch is None:
            continue
        if config_id in seen and seen[config_id] != arch:
            raise ModelConfigCollisionError(
                f"model_config_id {config_id!r} maps to two different architectures: "
                f"{seen[config_id]!r} vs {arch!r}"
            )
        seen[config_id] = arch


# ---------------------------------------------------------------------------
# GAUSS (diagonal / full covariance Gaussian) model-config resolution
# ---------------------------------------------------------------------------


def gauss_model_config(*, family: str, dimension: int) -> Dict[str, Any]:
    if family == "diagonal_gaussian":
        model_config_id = f"GAUSS_DIAG_d{int(dimension):02d}"
        label = f"Diagonal Gaussian, dimension {dimension}"
    elif family == "full_gaussian":
        model_config_id = f"GAUSS_FULL_d{int(dimension):02d}"
        label = f"Full-covariance Gaussian, dimension {dimension}"
    else:
        raise AliasResolutionError(f"not a recognized Gaussian family: {family!r}")
    return {
        "model_family_id": "GAUSS_DIAG" if family == "diagonal_gaussian" else "GAUSS_FULL",
        "model_family_label": "Diagonal Gaussian" if family == "diagonal_gaussian" else "Full-covariance Gaussian",
        "model_config_id": model_config_id,
        "model_config_label": label,
        "architecture_parameters": {"dimension": int(dimension)},
    }


# ---------------------------------------------------------------------------
# GMM (Gaussian mixture) model-config resolution
# ---------------------------------------------------------------------------


def gmm_model_config(
    *,
    n_components: Optional[int],
    covariance_type: Optional[str],
    dimension: Optional[int],
    missing_fields: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Resolve a GMM ``model_config_id`` from verified evidence only.

    If any of ``n_components``/``covariance_type``/``dimension`` is missing,
    this never invents a value -- it returns the explicit
    ``GMM_LEGACY_CONFIG_UNRESOLVED`` alias and records exactly which fields
    were missing in ``alias_evidence_source``/``alias_resolution_status``.
    """

    missing = list(missing_fields) if missing_fields else []
    if n_components is None:
        missing.append("n_components")
    if covariance_type is None:
        missing.append("covariance_type")
    if dimension is None:
        missing.append("dimension")

    if missing:
        return {
            "model_family_id": "GMM",
            "model_family_label": "Gaussian mixture",
            "model_config_id": "GMM_LEGACY_CONFIG_UNRESOLVED",
            "model_config_label": "Gaussian mixture, configuration unresolved",
            "architecture_parameters": {
                "n_components": n_components,
                "covariance_type": covariance_type,
                "dimension": dimension,
            },
            "alias_resolution_status": _UNRESOLVED,
            "unresolved_missing_fields": sorted(set(missing)),
        }

    cov_code = _slug_upper(covariance_type)
    model_config_id = f"GMM_k{int(n_components):02d}_cov{cov_code}_d{int(dimension):02d}"
    label = f"Gaussian mixture, {n_components} components, {covariance_type} covariance, dimension {dimension}"
    return {
        "model_family_id": "GMM",
        "model_family_label": "Gaussian mixture",
        "model_config_id": model_config_id,
        "model_config_label": label,
        "architecture_parameters": {
            "n_components": int(n_components),
            "covariance_type": str(covariance_type),
            "dimension": int(dimension),
        },
        "alias_resolution_status": _CURATED,
    }


# ---------------------------------------------------------------------------
# Full alias-record assembly
# ---------------------------------------------------------------------------


def build_alias_record(
    *,
    internal_candidate_id: str,
    legacy_run_id: Optional[str],
    track: Dict[str, Any],
    model_config: Dict[str, Any],
    preprocessing_name: str,
    weighting_policy: str,
    pdg_value: Optional[int],
    modeled_features: Sequence[str],
    modeled_dimension: int,
    alias_registry_version: str,
    alias_evidence_source: str,
) -> Dict[str, Any]:
    """Assemble one full alias record per the D9C schema (docs section 4)."""

    status_parts = [track.get("alias_resolution_status", _CURATED)]
    if "alias_resolution_status" in model_config:
        status_parts.append(model_config["alias_resolution_status"])
    alias_resolution_status = _UNRESOLVED if _UNRESOLVED in status_parts else (
        _GENERIC_FALLBACK if _GENERIC_FALLBACK in status_parts else _CURATED
    )

    display_name = f"{model_config['model_config_id']} on {track['track_id']}"

    return {
        "internal_candidate_id": internal_candidate_id,
        "legacy_run_id": legacy_run_id,
        "track_id": track["track_id"],
        "track_label": track["track_label"],
        "model_family_id": model_config["model_family_id"],
        "model_family_label": model_config["model_family_label"],
        "model_config_id": model_config["model_config_id"],
        "model_config_label": model_config["model_config_label"],
        "display_name": display_name,
        "architecture_parameters": model_config["architecture_parameters"],
        "preprocessing_name": preprocessing_name,
        "weighting_policy": weighting_policy,
        "pdg_value": pdg_value,
        "modeled_features": list(modeled_features),
        "modeled_dimension": int(modeled_dimension),
        "alias_registry_version": alias_registry_version,
        "alias_resolution_status": alias_resolution_status,
        "alias_evidence_source": alias_evidence_source,
    }


def readable_report_line(record: Dict[str, Any]) -> str:
    return f"{record['model_config_label']} · {record['track_label']}"


def resolve_d9_training_candidate_alias(
    *,
    registry: Dict[str, Any],
    plan_entry: Dict[str, Any],
    training_entry: Dict[str, Any],
    evidence_source: str,
) -> Dict[str, Any]:
    """Resolve the alias record for one of the six enabled D9 candidates,
    combining the candidate plan (legacy id, D8 lineage) with the training
    config (actual architecture) -- never inferring architecture from a
    capacity label alone."""

    track = resolve_track(
        registry,
        pdg_value=training_entry["pdg_value"],
        weighting_policy=training_entry["weighting_policy"],
        preprocessing_name=training_entry["preprocessing_name"],
    )
    if training_entry["model_family"] != "affine_coupling":
        raise AliasResolutionError(
            f"expected affine_coupling model_family for D9 training candidate, got "
            f"{training_entry['model_family']!r}"
        )
    model_config = nf_ac_model_config(training_entry["architecture"])
    return build_alias_record(
        internal_candidate_id=plan_entry["candidate_id"],
        legacy_run_id=plan_entry.get("source_d8_run_id"),
        track=track,
        model_config=model_config,
        preprocessing_name=training_entry["preprocessing_name"],
        weighting_policy=training_entry["weighting_policy"],
        pdg_value=training_entry["pdg_value"],
        modeled_features=training_entry["modeled_features"],
        modeled_dimension=len(training_entry["modeled_features"]),
        alias_registry_version=registry["schema_version"],
        alias_evidence_source=evidence_source,
    )


def resolve_scout_variant_alias(
    *,
    registry: Dict[str, Any],
    variant_training_config: Dict[str, Any],
    base_candidate_id: str,
    capacity_scale: float,
    evidence_source: str,
) -> Dict[str, Any]:
    """Resolve the alias record for one AFFINE_COUPLING_CAPACITY_SCOUT_V0
    variant from its own per-run ``training_config.json`` snapshot -- the
    single most authoritative source for what that specific run actually
    trained (it is what ``train_candidate_seed`` wrote alongside the
    checkpoint, not a value recomputed from ``capacity_scale``).

    The track is derived from the variant's own recorded
    pdg_value/weighting_policy/preprocessing_name (identical to the base
    candidate's track, since scouting only varies capacity); the
    model_config_id is derived from the variant's own recorded architecture,
    never from ``capacity_scale`` or a scale-multiplied guess.
    """

    track = resolve_track(
        registry,
        pdg_value=variant_training_config["pdg_value"],
        weighting_policy=variant_training_config["weighting_policy"],
        preprocessing_name=variant_training_config["preprocessing_name"],
    )
    model_config = nf_ac_model_config(variant_training_config["architecture"])
    record = build_alias_record(
        internal_candidate_id=variant_training_config["candidate_id"],
        legacy_run_id=None,
        track=track,
        model_config=model_config,
        preprocessing_name=variant_training_config["preprocessing_name"],
        weighting_policy=variant_training_config["weighting_policy"],
        pdg_value=variant_training_config["pdg_value"],
        modeled_features=variant_training_config["modeled_features"],
        modeled_dimension=len(variant_training_config["modeled_features"]),
        alias_registry_version=registry["schema_version"],
        alias_evidence_source=evidence_source,
    )
    record["scout_legacy_base_candidate_id"] = base_candidate_id
    record["scout_legacy_capacity_scale"] = capacity_scale
    record["scout_experiment_id"] = "AFFINE_COUPLING_CAPACITY_SCOUT_V0"
    return record
