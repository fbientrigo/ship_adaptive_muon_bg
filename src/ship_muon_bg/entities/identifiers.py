"""Content-addressed semantic definition identifiers.

Implements ``docs/contracts/tagging_contract_v0.md`` ``CONF-01``: a
``*_definition_id`` must resolve, by construction, to exactly one semantic
definition forever. The same ``content`` always produces the same id; a
changed ``content`` always produces a different one. A human-readable
``label`` may prefix the id but is never load-bearing for identity.

Mirrors the pattern in ``src/ship_muon_bg/afterms/d9/contract.py``'s
``semantic_training_hash`` (canonical JSON + sha256 over a semantic
fingerprint, not a raw git ref). Implemented independently here rather than
imported from ``afterms`` — that package is an unrelated experiment track,
and ``entities/`` must not depend on it.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


def canonical_json(content: Mapping[str, Any]) -> str:
    """Deterministic JSON serialization: sorted keys, no incidental whitespace."""
    return json.dumps(content, sort_keys=True, separators=(",", ":"), default=str)


def content_hash(content: Mapping[str, Any]) -> str:
    """sha256 hex digest of ``content``'s canonical serialization."""
    return hashlib.sha256(canonical_json(content).encode("utf-8")).hexdigest()


def definition_id(label: str, content: Mapping[str, Any]) -> str:
    """A content-addressed ``*_definition_id``: ``"{label}@sha256:{digest}"``.

    ``label`` is a human-readable prefix only (e.g. ``"operational_selection_v0"``);
    identity and immutability come entirely from the hash of ``content``, so
    two calls with the same ``content`` always agree and any semantic change
    to ``content`` always produces a different id, even if ``label`` is left
    unchanged by mistake.
    """
    if not label:
        raise ValueError("label must be non-empty")
    return f"{label}@sha256:{content_hash(content)}"
