"""D9-5 frozen config loaders (Sec 6, Sec 10, Sec 11.1).

Thin, explicit readers over the two frozen JSON configs -- no defaults are
invented here; a missing field is a loud ``KeyError``, not a silent fallback.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_SCOPE_CONFIG_PATH = REPO_ROOT / "configs" / "afterms" / "d9_5_data_scope_v0.json"
MODEL_FAMILY_ARENA_CONFIG_PATH = REPO_ROOT / "configs" / "afterms" / "d9_5_model_family_arena_v0.json"

TRACK_IDS = ("TRK_PDG13_UW_ID", "TRK_PDGM13_UW_ID")
MODEL_CONFIG_IDS_BY_FAMILY = {
    "GAUSS_DIAG": "GAUSS_DIAG_d05",
    "GAUSS_FULL": "GAUSS_FULL_d05",
    "GMM": "GMM_k04_covFULL_d05",
}


def load_data_scope_config(path: Path = DATA_SCOPE_CONFIG_PATH) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_model_family_arena_config(path: Path = MODEL_FAMILY_ARENA_CONFIG_PATH) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def track_entry(data_scope_config: Dict[str, Any], track_id: str) -> Dict[str, Any]:
    for track in data_scope_config["tracks"]:
        if track["track_id"] == track_id:
            return track
    raise KeyError(f"unknown D9-5 track_id {track_id!r}; expected one of {TRACK_IDS}")
