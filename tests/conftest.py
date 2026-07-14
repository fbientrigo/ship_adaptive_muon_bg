"""Test configuration: make the repo importable uninstalled, expose fixtures.

The suite must pass both with ``pip install -e .`` and from a bare clone, so
``src/`` (for ``ship_muon_bg``) and the repo root (for ``Nflow`` and
``ProxyTagger``) are added to ``sys.path`` as a fallback.
"""

from __future__ import annotations

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO_ROOT, "src")
for path in (SRC, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

FIXTURE_DIR = os.path.join(REPO_ROOT, "tests", "fixtures")


def _module_available(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


def pytest_collection_modifyitems(config, items):
    """Auto-skip optional-stack tests when their dependency is absent.

    This keeps the single command ``python -m pytest -q`` valid in *both* the
    NumPy-only core environment (flow/lab tests skip) and the full
    ``.[dev,flow,lab]`` environment (they run). It does not change the command.
    """

    skip_flow = None
    skip_lab = None
    if not _module_available("torch"):
        skip_flow = pytest.mark.skip(reason="requires optional torch stack (.[flow])")
    if not _module_available("sklearn"):
        skip_lab = pytest.mark.skip(reason="requires optional scikit-learn (.[lab])")
    for item in items:
        if skip_flow is not None and "flow" in item.keywords:
            item.add_marker(skip_flow)
        if skip_lab is not None and "lab" in item.keywords:
            item.add_marker(skip_lab)


@pytest.fixture
def tiny_pkl_path():
    return os.path.join(FIXTURE_DIR, "muon_sample_tiny.pkl.gz")
