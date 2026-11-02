"""Run FairShip's ``macro/run_simScript.py --MuonBack`` unmodified, and keep the unfiltered event tree.

``--MuonBack`` ends by copying only events with a hit in a non-scoring ``*Point`` branch and then ``rm`` + ``mv``-ing the
copy over the output (``run_simScript.py`` "remove empty events", 26.10: ``os.system("rm  " + outFile)``). A muon that
crosses P2 and nothing else is therefore absent from the standard output. This driver runs the script with
``runpy`` (no source rewriting) and turns that one ``rm`` into a rename to ``<out>.unfiltered.root``, so the run
yields both the standard (filtered) file and the lossless one. Because it relies on that internal command, the script is
pinned by sha256: an unknown ``run_simScript.py`` is refused until someone re-reads the filter and adds its hash.

    pixi run --manifest-path <FairShip>/pixi.toml python scripts/run_stage2_p2.py --fairship <FairShip> -- <run_simScript args>

The native fix is an option in FairShip (see ``notes/proposals/fairship_muonback_keep_scoring.md``); this driver is the
interim, pinned route.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import runpy
import sys
from pathlib import Path

#: sha256 of ``macro/run_simScript.py`` whose empty-event filter was read and found to match ``FILTER_SIGNATURE``.
PINNED = {"8a1cc53ecdd55b99cceec1a35346ee6b8803a250": "6eed97e07f278ec6070ce3a312ca51bbaf8b6d2cd36fd2d51401777885ee26f3"}
FILTER_SIGNATURE = ('# remove empty events\nif options.muonback:', 'os.system("rm  " + outFile)', 'os.system("mv " + tmpFile + " " + outFile)')
_RM = re.compile(r"^rm\s+(\S+\.root)$")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_pinned(script: Path, pins: dict[str, str] = PINNED) -> None:
    """Refuse a ``run_simScript.py`` that is not pinned, or whose filter no longer has the shape this driver interposes on."""
    text = script.read_text(encoding="utf-8")
    missing = [s for s in FILTER_SIGNATURE if s not in text]
    if missing:
        raise RuntimeError(f"{script}: empty-event filter changed, not found: {missing}")
    if sha256(script) not in pins.values():
        raise RuntimeError(f"{script}: sha256 {sha256(script)} is not pinned; re-read the filter, then add it to PINNED")


def keep_unfiltered(system):
    """Wrap ``os.system``: the first ``rm <x>.root`` becomes ``mv <x>.root <x>.unfiltered.root``; everything else passes through."""
    state = {"done": False}

    def wrapped(cmd):
        m = _RM.match(cmd) if isinstance(cmd, str) else None
        if m and not state["done"]:
            state["done"] = True
            src = Path(m.group(1))
            os.replace(src, src.with_suffix(".unfiltered.root"))
            return 0
        return system(cmd)

    wrapped.state = state
    return wrapped


def run(script: Path, argv: list[str]) -> bool:
    """Run ``script`` with ``argv``; True if the unfiltered file was kept (the filter ran)."""
    original, wrapper = os.system, keep_unfiltered(os.system)
    os.system, sys.argv = wrapper, [str(script), *argv]
    sys.path.insert(0, str(script.parent))
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        os.system = original
    return wrapper.state["done"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fairship", required=True, type=Path)
    ap.add_argument("--allow-unpinned", action="store_true", help="diagnostics only")
    ap.add_argument("args", nargs=argparse.REMAINDER)
    ns = ap.parse_args()
    script = ns.fairship / "macro" / "run_simScript.py"
    if not ns.allow_unpinned:
        check_pinned(script)
    argv = ns.args[1:] if ns.args[:1] == ["--"] else ns.args
    kept = run(script, argv)
    print("run_stage2_p2: unfiltered file kept" if kept else "run_stage2_p2: WARNING the empty-event filter did not run (no --MuonBack?)")
    return 0 if kept else 2


if __name__ == "__main__":
    raise SystemExit(main())
