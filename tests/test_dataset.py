"""Parquet export is lossless, snapshots are immutable and refuse undeclared or mixed definitions (no ROOT)."""

import importlib.util
import json
from pathlib import Path

import duckdb
import pytest

from ship_muon_bg import dataset as ds
from ship_muon_bg.adapters.pr1432_mudis import build_bundle, stage_ids
from ship_muon_bg.simulation.evaluation import bundle_records

_spec = importlib.util.spec_from_file_location("pr1432_helpers", Path(__file__).parent / "fairship" / "test_pr1432_mudis.py")
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)
FILTER = stage_ids(H.COMMIT, 4, 1)["filter"]
POLICY = {"u_version": "u0_v0", "sampling": "test"}


def write_bundles(tmp: Path) -> list[Path]:
    cases = [("row-1", 11, {}), ("row-1", 12, {"kept_ms": ()}), ("row-2", 13, {"failure": {"prepare": "boom"}})]
    paths = []
    for subject, seed, kw in cases:
        b = build_bundle(H.request(seed=seed, subject=subject), H.facts(seed=seed, **kw))
        p = tmp / f"{subject}_{seed}" / "bundle.json"
        p.parent.mkdir()
        p.write_text(json.dumps(bundle_records(b)))
        paths.append(p)
    return paths


def test_export_keeps_every_record_and_its_source(tmp_path):
    paths = write_bundles(tmp_path)
    m = ds.export(paths, tmp_path / "d", {"campaign": "t"})
    for kind in ds.KINDS:
        assert m["tables"][kind]["rows"] == sum(len(json.loads(p.read_text())[kind]) for p in paths)
    rows = duckdb.sql(f"select execution_status, count(*) from '{tmp_path}/d/executions.parquet' group by 1 order by 1").fetchall()
    assert rows == [("succeeded", 2), ("technical_failure", 1)]  # the failure is kept, not dropped
    assert {r[0] for r in duckdb.sql(f"select bundle_file from '{tmp_path}/d/decisions.parquet'").fetchall()} <= {str(p) for p in paths}


def test_export_refuses_duplicate_requests(tmp_path):
    p = write_bundles(tmp_path)[0]
    with pytest.raises(ValueError, match="duplicate"):
        ds.export([p, p], tmp_path / "d", {})


def test_snapshot_aggregates_per_state_and_is_immutable(tmp_path):
    ds.export(write_bundles(tmp_path), tmp_path / "d", {})
    snap = ds.snapshot([tmp_path / "d"], FILTER, tmp_path / "s", policy=POLICY, split_seed=1)
    spec = json.loads((snap / "snapshot.json").read_text())
    assert spec["stage_definition_id"] == FILTER and spec["estimator_definition_id"].startswith("eta_hat_positive_over_valid_v0@")
    rows = {r[0]: r for r in duckdb.sql(f"select subject_id, executions, valid, positive, technical_failure, eta_hat from '{snap}/rows.parquet'").fetchall()}
    assert rows["row-1"][1:] == (2, 2, 1, 0, 0.5)  # two executions of one state: one pass, one reject
    assert rows["row-2"][1:4] == (1, 0, 0) and rows["row-2"][5] is None  # a technical failure is unknown, not zero
    before = (snap / "rows.parquet").read_bytes()
    assert ds.snapshot([tmp_path / "d"], FILTER, tmp_path / "s", policy=POLICY, split_seed=1) == snap
    assert (snap / "rows.parquet").read_bytes() == before
    assert ds.snapshot([tmp_path / "d"], FILTER, tmp_path / "s", policy={**POLICY, "u_version": "u1"}, split_seed=1) != snap


def test_snapshot_refuses_absent_stage_and_tampered_tables(tmp_path):
    ds.export(write_bundles(tmp_path), tmp_path / "d", {})
    with pytest.raises(ValueError, match="not in the evidence"):
        ds.snapshot([tmp_path / "d"], stage_ids(H.COMMIT, 10, 1)["filter"], tmp_path / "s", policy=POLICY, split_seed=1)
    (tmp_path / "d" / "decisions.parquet").write_bytes(b"x")
    with pytest.raises(ValueError, match="manifest"):
        ds.load_records([tmp_path / "d"])
