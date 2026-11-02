"""The stage-2 driver on a stand-in ``run_simScript.py`` (no ROOT, no FairShip): interposition, pinning, signature."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from scripts import run_stage2_p2 as drv

FAKE = '''
import os, sys
out = sys.argv[1]
open(out, "w").write("RAW")
if True:
    tmp = out + "tmp"
    open(tmp, "w").write("FILTERED")
    # remove empty events
    os.system("rm  " + out)
    os.system("mv " + tmp + " " + out)
'''
# the real script has this exact shape; the driver checks the same strings
REAL_SHAPE = '# remove empty events\nif options.muonback:\n    os.system("rm  " + outFile)\n    os.system("mv " + tmpFile + " " + outFile)\n'


def test_rm_becomes_rename_and_the_standard_output_is_still_the_filtered_one(tmp_path):
    script, out = tmp_path / "run_simScript.py", tmp_path / "sim.root"
    script.write_text(FAKE)
    assert drv.run(script, [str(out)]) is True
    assert out.read_text() == "FILTERED" and (tmp_path / "sim.unfiltered.root").read_text() == "RAW"
    assert not (tmp_path / "sim.rootmp").exists()


def test_a_script_without_the_filter_reports_it_and_os_system_is_restored(tmp_path):
    import os

    before = os.system
    script = tmp_path / "run_simScript.py"
    script.write_text('import sys\nopen(sys.argv[1], "w").write("RAW")\n')
    assert drv.run(script, [str(tmp_path / "sim.root")]) is False and os.system is before


def test_only_the_first_rm_is_interposed():
    calls = []
    w = drv.keep_unfiltered(lambda c: calls.append(c) or 0)
    assert w("echo hi") == 0 and calls == ["echo hi"]


def test_unpinned_or_changed_scripts_are_refused(tmp_path):
    s = tmp_path / "run_simScript.py"
    s.write_text(REAL_SHAPE)
    with pytest.raises(RuntimeError, match="not pinned"):
        drv.check_pinned(s, {"x": "0" * 64})
    drv.check_pinned(s, {"x": drv.sha256(s)})
    s.write_text(REAL_SHAPE.replace("rm  ", "rm "))
    with pytest.raises(RuntimeError, match="filter changed"):
        drv.check_pinned(s, {"x": drv.sha256(s)})
