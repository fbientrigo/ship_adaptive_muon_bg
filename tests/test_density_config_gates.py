"""ExperimentConfig.from_dict must reject malformed scientific_gates config.

Regression coverage for Codex P2 on ScientificGateSpec.require_d5_rare_metrics:
a non-bool value (notably 0/1, since bool is an int subclass in Python) must
be rejected before it can reach D5 gate activation, whether the spec is built
directly or parsed from a JSON-shaped config payload.
"""

from __future__ import annotations

import pytest

from ship_muon_bg.density_lab.config import ConfigError, ExperimentConfig
from ship_muon_bg.density_lab.gates import GateConfigError, ScientificGateSpec


def _base_payload(**scientific_gates_overrides):
    return {
        "experiment_id": "cfg-gate-test",
        "targets": [{"target_id": "D2"}],
        "pdg_ids": [13],
        "feature_views": [{"view_id": "identity_cartesian_v0"}],
        "models": [{"name": "diagonal_gaussian", "family": "diagonal_gaussian"}],
        "seeds": [11],
        "scientific_gates": scientific_gates_overrides,
    }


_VALID_VALUES = [True, False]
_INVALID_VALUES = [None, 0, 1, 0.0, "true", "false", [], {}]


@pytest.mark.parametrize("value", _VALID_VALUES)
def test_from_dict_accepts_valid_require_d5_rare_metrics(value):
    config = ExperimentConfig.from_dict(
        _base_payload(require_d5_rare_metrics=value)
    )
    assert config.scientific_gates.require_d5_rare_metrics is value


@pytest.mark.parametrize("value", _INVALID_VALUES)
def test_from_dict_rejects_non_bool_require_d5_rare_metrics(value):
    payload = _base_payload(require_d5_rare_metrics=value)
    with pytest.raises(ConfigError, match="require_d5_rare_metrics must be a bool"):
        ExperimentConfig.from_dict(payload)


def test_from_dict_default_omits_scientific_gates_key():
    # No "scientific_gates" key at all -> default spec (require_d5_rare_metrics=True).
    payload = _base_payload()
    del payload["scientific_gates"]
    config = ExperimentConfig.from_dict(payload)
    assert config.scientific_gates.require_d5_rare_metrics is True


def test_true_to_false_changes_canonical_config_hash():
    true_config = ExperimentConfig.from_dict(_base_payload(require_d5_rare_metrics=True))
    false_config = ExperimentConfig.from_dict(_base_payload(require_d5_rare_metrics=False))
    assert true_config.config_hash() != false_config.config_hash()


def test_direct_spec_validate_raises_gate_config_error_not_config_error():
    # ScientificGateSpec.validate() itself raises GateConfigError; only the
    # ExperimentConfig.validate() boundary normalizes it to ConfigError.
    spec = ScientificGateSpec(require_d5_rare_metrics=0)
    with pytest.raises(GateConfigError, match="require_d5_rare_metrics must be a bool"):
        spec.validate()
    assert not issubclass(GateConfigError, ConfigError)


def test_existing_valid_default_config_still_validates():
    config = ExperimentConfig.from_dict(_base_payload())
    assert config.scientific_gates.require_d5_rare_metrics is True
    config.validate()  # no error
