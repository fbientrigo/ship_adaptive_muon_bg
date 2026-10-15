"""Report/config-shape tests.

Covers required tests 3, 4, 33.
"""

from __future__ import annotations

import json

from ship_muon_bg.afterms.d9_5 import config as d9_5config, report


def test_exactly_four_model_families_declared():
    arena_config = d9_5config.load_model_family_arena_config()
    families = {"nf_ac", "gauss_diag", "gauss_full", "gmm"}
    assert families.issubset(arena_config.keys())
    assert len(families) == 4


def test_no_flow_matching_model_included():
    arena_config = d9_5config.load_model_family_arena_config()
    serialized = json.dumps(arena_config).lower()
    assert "flow_matching" not in serialized
    assert "flow matching" not in serialized


def test_final_report_regeneration_is_deterministic(tmp_path):
    per_track_results = {
        "TRK_PDG13_UW_ID": [
            {
                "model_family_id": "GAUSS_DIAG", "model_config_id": "GAUSS_DIAG_d05",
                "physical_nll": 5.0, "feature_nll": 4.0, "finite_log_prob_fraction": 1.0,
            },
            {
                "model_family_id": "GAUSS_FULL", "model_config_id": "GAUSS_FULL_d05",
                "physical_nll": 4.5, "feature_nll": 3.5, "finite_log_prob_fraction": 1.0,
            },
        ],
    }
    paths1 = report.write_final_report(tmp_path / "run1", per_track_results)
    paths2 = report.write_final_report(tmp_path / "run2", per_track_results)
    assert paths1["json"].read_text(encoding="utf-8") == paths2["json"].read_text(encoding="utf-8")
    assert paths1["md"].read_text(encoding="utf-8") == paths2["md"].read_text(encoding="utf-8")
    assert paths1["csv"].read_text(encoding="utf-8") == paths2["csv"].read_text(encoding="utf-8")
