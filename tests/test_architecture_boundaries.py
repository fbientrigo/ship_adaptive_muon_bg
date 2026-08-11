"""Precise dependency-boundary guard tests.

Replaces the retired ``tests/test_data_contracts.py::test_no_root_or_fairship_import_in_core``,
which swept the *entire* ``src/ship_muon_bg`` tree for any import line
containing ``"fairship"``/``"import root"``. That invariant is obsolete now
that ``src/ship_muon_bg/adapters/fairship/`` is the explicitly sanctioned
FairShip adapter boundary (``docs/architecture/scientific_architecture_v2.md``
§2 "placement note", ``docs/contracts/tagging_contract_v0.md`` §0).

The invariant this file enforces instead: *backend-independent* packages
under ``src/ship_muon_bg/`` — today ``entities/``, ``data_contracts/``,
``simulation/`` — must never import ROOT, FairShip, or
``ship_muon_bg.adapters.fairship`` itself. ``src/ship_muon_bg/adapters/fairship/``
is exempt by construction: it is simply never a scan root, not special-cased
by the scanner. Requires no ROOT, no FairShip, no GPU.
"""

from __future__ import annotations

import ast
import os
import textwrap
from typing import List, Tuple

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE_ROOT = os.path.join(REPO_ROOT, "src", "ship_muon_bg")

# Backend-independent package roots that exist today. Extend this list as
# tagging/, proxy/, proposal/ land (migration steps in
# scientific_architecture_v2.md §11). Never add "adapters" here.
BACKEND_INDEPENDENT_PACKAGES = ("entities", "data_contracts", "simulation")

# The one adapter boundary this contract currently sanctions
# (scientific_architecture_v2.md §2).
ALLOWED_ADAPTER_BOUNDARIES = (os.path.join("adapters", "fairship"),)


def _segment_is_forbidden(segment: str) -> bool:
    """A dotted-import segment is forbidden if it is exactly ``root``
    (catches ``import ROOT``) or starts with ``fairship`` (catches
    ``fairship``, ``fairship_adapter``, ``FairShipRunner``-style module
    names, and — deliberately — ``ship_muon_bg.adapters.fairship`` itself,
    whose last segment is ``fairship``). Segment-exact/prefix matching on
    AST-parsed dotted names avoids the old test's whole-line substring
    matching, which could false-positive on an unrelated identifier merely
    containing "fairship" as a substring anywhere on an import line.
    """
    lowered = segment.lower()
    return lowered == "root" or lowered.startswith("fairship")


def find_forbidden_imports(root: str) -> List[Tuple[str, int, str]]:
    """Return ``(file, lineno, dotted_module_name)`` for every import under
    ``root`` whose dotted module name has a forbidden segment. Uses the
    ``ast`` module rather than line-substring matching, per the requirement
    to prefer AST inspection over fragile repository-wide greps.
    """
    offenders: List[Tuple[str, int, str]] = []
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            if not name.endswith(".py"):
                continue
            full = os.path.join(dirpath, name)
            with open(full, "r", encoding="utf-8") as handle:
                source = handle.read()
            tree = ast.parse(source, filename=full)
            for node in ast.walk(tree):
                dotted_names = []
                if isinstance(node, ast.Import):
                    dotted_names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    dotted_names = [node.module]
                for dotted in dotted_names:
                    if any(_segment_is_forbidden(seg) for seg in dotted.split(".")):
                        offenders.append((full, node.lineno, dotted))
    return offenders


# ---------------------------------------------------------------------------
# A. Backend-independent package purity.
# ---------------------------------------------------------------------------


def test_backend_independent_packages_have_no_root_or_fairship_imports():
    offenders: List[Tuple[str, int, str]] = []
    scanned_any = False
    for package in BACKEND_INDEPENDENT_PACKAGES:
        package_dir = os.path.join(CORE_ROOT, package)
        if not os.path.isdir(package_dir):
            continue
        scanned_any = True
        offenders.extend(find_forbidden_imports(package_dir))
    assert scanned_any, "expected at least one backend-independent package to exist"
    assert not offenders, (
        "backend-independent packages must not import ROOT/FairShip "
        "(including ship_muon_bg.adapters.fairship):\n"
        + "\n".join(f"{p}:{ln}: {name}" for p, ln, name in offenders)
    )


def test_adapters_directory_is_never_a_backend_independent_scan_root():
    assert "adapters" not in BACKEND_INDEPENDENT_PACKAGES
    for package in BACKEND_INDEPENDENT_PACKAGES:
        assert not package.startswith("adapters")


def test_adapters_fairship_is_a_declared_allowed_adapter_boundary():
    assert os.path.join("adapters", "fairship") in ALLOWED_ADAPTER_BOUNDARIES


# ---------------------------------------------------------------------------
# B. Scanner correctness against synthetic fixtures (does not require ROOT
#    or a real FairShip checkout — and does not require adapters/fairship/
#    to contain real code yet, since it does not in this slice).
# ---------------------------------------------------------------------------


def test_scanner_flags_root_and_fairship_imports_when_pointed_at_them(tmp_path):
    offending = tmp_path / "runner.py"
    offending.write_text("import ROOT\nfrom fairship import Something\n")
    offenders = find_forbidden_imports(str(tmp_path))
    assert len(offenders) == 2


def test_adapters_fairship_subtree_is_exempt_by_construction(tmp_path):
    root = tmp_path / "ship_muon_bg"
    clean_pkg = root / "entities"
    clean_pkg.mkdir(parents=True)
    (clean_pkg / "__init__.py").write_text("import numpy as np\n")

    adapter_pkg = root / "adapters" / "fairship"
    adapter_pkg.mkdir(parents=True)
    (adapter_pkg / "runner.py").write_text("import ROOT\nfrom fairship import Something\n")

    # Sweeping only entities/ (as the real purity test above does) finds
    # nothing, even though a FairShip-importing file exists elsewhere in
    # the same tree — exemption comes from never pointing the sweep at
    # adapters/fairship/, not from the scanner special-casing that path.
    assert not find_forbidden_imports(str(clean_pkg))

    # The scanner itself is not blind to ROOT/FairShip imports in general:
    # pointed directly at the adapter subtree, it does flag them. This is
    # exactly why the purity sweep's exemption must come from root
    # selection, not from a silently-more-permissive scanner.
    assert len(find_forbidden_imports(str(adapter_pkg))) == 2


def test_reverse_import_of_fairship_adapter_from_scientific_layer_is_rejected(tmp_path):
    offending = tmp_path / "tagging_stub.py"
    offending.write_text(
        textwrap.dedent(
            """
            from ship_muon_bg.adapters.fairship import FairShipRunner
            """
        )
    )
    offenders = find_forbidden_imports(str(tmp_path))
    assert len(offenders) == 1
    _, _, module_name = offenders[0]
    assert module_name == "ship_muon_bg.adapters.fairship"


def test_scanner_does_not_false_positive_on_unrelated_root_prefixed_name(tmp_path):
    benign = tmp_path / "benign.py"
    benign.write_text("from rootcause_toolkit import analyze\nimport roots_are_fine\n")
    assert not find_forbidden_imports(str(tmp_path))
