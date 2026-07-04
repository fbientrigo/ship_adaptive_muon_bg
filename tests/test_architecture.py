"""Guard-rail tests for the three-module architecture.

Covers the module packages (``Nflow``, ``ProxyTagger``) and the
``simulation_backend`` boundary (``ship_muon_bg.simulation``): imports stay
light (no torch/ROOT/FairShip in non-legacy code), the ``DummyProxy``
honors the ``U(x)`` contract, and the outcome taxonomy invariants hold.
Requires no FairShip, no ROOT, no GPU.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

import Nflow
import ProxyTagger
from Nflow.interfaces import DensityModel  # noqa: F401 - import must succeed
from ProxyTagger import SCORE_MAX, SCORE_MIN, DummyProxy, ProxyScorer
from ship_muon_bg.data_contracts import load_muon_pkl
from ship_muon_bg.simulation import (
    FlowProposalRecord,
    OutcomeCategory,
    SimulationBackend,  # noqa: F401 - import must succeed
    SimulationResult,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO_ROOT, "src")


# 1. The module packages import without pulling in heavy/physics deps.
def test_module_imports_pull_no_heavy_deps():
    code = (
        "import sys\n"
        "import Nflow, ProxyTagger, ship_muon_bg.simulation\n"
        "banned = [m for m in ('torch', 'ROOT', 'h5py') if m in sys.modules]\n"
        "assert not banned, f'heavy modules imported: {banned}'\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [REPO_ROOT, SRC, env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


# 2. DummyProxy honors the ProxyScorer contract on the committed fixture.
def test_dummy_proxy_scores_fixture(tiny_pkl_path):
    array = load_muon_pkl(tiny_pkl_path)
    proxy = DummyProxy()
    assert isinstance(proxy, ProxyScorer)
    assert proxy.is_physical is False  # placeholders never report results

    scores = proxy.score(array)
    assert scores.shape == (array.shape[0],)
    assert np.isfinite(scores).all()
    assert (scores >= SCORE_MIN).all() and (scores <= SCORE_MAX).all()
    # Deterministic: same input, same scores.
    np.testing.assert_array_equal(scores, DummyProxy().score(array))


def test_dummy_proxy_rejects_non_positive_scale():
    with pytest.raises(ValueError):
        DummyProxy(scale=0.0)


# 3. Outcome taxonomy: exactly three mutually exclusive categories.
def test_outcome_taxonomy_is_exactly_three_categories():
    assert {c.value for c in OutcomeCategory} == {
        "technical_failure",
        "physics_rejection",
        "accepted_candidate",
    }


def test_technical_failure_never_carries_a_dis_tag():
    ok = SimulationResult(
        candidate_id="c0", outcome=OutcomeCategory.TECHNICAL_FAILURE
    )
    assert ok.dis is None
    with pytest.raises(ValueError):
        SimulationResult(
            candidate_id="c1",
            outcome=OutcomeCategory.TECHNICAL_FAILURE,
            dis=False,
        )


def test_physics_outcomes_carry_dis_tag():
    result = SimulationResult(
        candidate_id="c2",
        outcome=OutcomeCategory.ACCEPTED_CANDIDATE,
        dis=True,
    )
    assert result.dis is True


def test_flow_proposal_record_is_immutable():
    record = FlowProposalRecord(
        candidate_id="c0",
        px=0.1, py=0.2, pz=40.0,
        x=0.0, y=0.0, z=30.0,
        pdg_id=13,
        weight=1.0,
    )
    with pytest.raises(Exception):
        record.px = 5.0  # frozen dataclass


# 4. No ROOT / FairShip import in the module packages (non-legacy code).
#    Mirrors test_data_contracts.test_no_root_or_fairship_import_in_core,
#    which already covers src/ship_muon_bg (including simulation/).
def test_no_root_or_fairship_import_in_module_packages():
    scan_dirs = [
        os.path.join(REPO_ROOT, "Nflow"),
        os.path.join(REPO_ROOT, "ProxyTagger"),
    ]
    legacy_dir = os.path.join(REPO_ROOT, "Nflow", "legacy")
    offenders = []
    for scan_dir in scan_dirs:
        for dirpath, _dirs, files in os.walk(scan_dir):
            if dirpath.startswith(legacy_dir):
                continue  # quarantined fork code is exempt (still no ROOT there)
            for name in files:
                if not name.endswith(".py"):
                    continue
                full = os.path.join(dirpath, name)
                with open(full, "r", encoding="utf-8") as handle:
                    for lineno, line in enumerate(handle, start=1):
                        stripped = line.strip()
                        if not (
                            stripped.startswith("import ")
                            or stripped.startswith("from ")
                        ):
                            continue
                        lowered = stripped.lower()
                        if (
                            "import root" in lowered
                            or "fairship" in lowered
                            or stripped == "import ROOT"
                        ):
                            offenders.append(f"{full}:{lineno}: {stripped}")
    assert not offenders, (
        "module packages must not import ROOT/FairShip:\n" + "\n".join(offenders)
    )
