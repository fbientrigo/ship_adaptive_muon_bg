"""Test configuration: make ``src/`` importable and expose fixture paths."""

from __future__ import annotations

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO_ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

FIXTURE_DIR = os.path.join(REPO_ROOT, "tests", "fixtures")


@pytest.fixture
def tiny_pkl_path():
    return os.path.join(FIXTURE_DIR, "muon_sample_tiny.pkl.gz")
