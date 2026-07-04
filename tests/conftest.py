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


@pytest.fixture
def tiny_pkl_path():
    return os.path.join(FIXTURE_DIR, "muon_sample_tiny.pkl.gz")
