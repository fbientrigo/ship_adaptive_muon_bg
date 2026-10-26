"""Per-DIS filter pass rate per straight-ray class, all minimal-loop evidence (EXPERIMENT analysis, iteration 3/4).

usage: PYTHONPATH=src .venv/bin/python ~/thesis/scratch/minimal-loop-v0/per_dis.py > per_dis.json
"""
import glob
import json
import sys

import duckdb
import numpy as np
from scipy import stats

from ship_muon_bg import minimal_loop as ml

S = "/home/fabian/thesis/scratch/minimal-loop-v0"
ds = [d for d in glob.glob(f"{S}/data/*") + glob.glob(f"{S}/iter2/data/G*") + glob.glob(f"{S}/iter[34]/data_*/*")
      if not d.endswith(("snapshots", ".json"))]
L = lambda t: "[" + ",".join(f"'{d}/{t}.parquet'" for d in ds) + "]"  # noqa: E731
s = np.load(f"{S}/iter2/score/pool_scores.npz")
cls = ml.geo_class(s["u0"], s["ray_upstream"], s["ray_tracker"])
rows = duckdb.sql(f"""
with dec as (select subject_ref, stage_definition_id sid, evaluation_status st, decision from read_parquet({L('decisions')})),
rz as (select r.execution_id, count(*) n, sum(d.decision::int) k from read_parquet({L('realizations')}) r
       join dec d on d.subject_ref = r.realization_id and d.sid like 'pr1432_filter_dis_v0@%' group by 1),
fa as (select subject_ref execution_id, decision any_ from dec where sid like 'pr1432_filter_any_v0@%' and st = 'evaluated'),
dpm as (select subject_ref execution_id, cast(json_extract(payload, '$.value') as int) ndis
        from read_parquet({L('observations')}) where observation_definition_id like 'pr1432:dis_requested_per_material%'),
ch as (select subject_ref execution_id, cast(json_extract(payload, '$.value') as double) chain_s
       from read_parquet({L('observations')}) where observation_definition_id like 'pr1432:wall_seconds_total%'),
c as (select parse_dirpath(filename) dir, request_id, inject_wall_s, total_wall_s from read_parquet({L('cost')}, filename = true)),
x as (select parse_dirpath(filename) dir, * exclude (filename) from read_parquet({L('executions')}, filename = true))
select x.subject_id, x.execution_id, dpm.ndis, rz.n, rz.k, fa.any_, c.inject_wall_s, ch.chain_s, c.total_wall_s
from x join rz using (execution_id) join fa using (execution_id) join dpm using (execution_id)
join ch using (execution_id) join c using (dir, request_id)""").fetchall()
assert len({r[1] for r in rows}) == len(rows)
assert all((r[4] > 0) == bool(r[5]) for r in rows), "filter_any is not any(per-DIS filter pass)"
out = {"executions": len(rows), "filter_any_equals_any_per_dis": True, "by_dis_per_material": {}, "cost_by_dis_in_run": {}, "by_class": {}}
for n in sorted({r[3] for r in rows}):
    sel = [r for r in rows if r[3] == n]
    out["cost_by_dis_in_run"][n] = {"runs": len(sel), "inject_s": round(float(np.mean([r[6] for r in sel])), 1),
                                    "chain_s": round(float(np.mean([r[7] for r in sel])), 1),
                                    "total_s": round(float(np.mean([r[8] for r in sel])), 1)}
for nd in sorted({r[2] for r in rows}):
    sel = [r for r in rows if r[2] == nd]
    out["by_dis_per_material"][nd] = {"runs": len(sel), "mean_dis": float(np.mean([r[3] for r in sel])),
                                      "inject_s": float(np.mean([r[6] for r in sel])), "chain_s": float(np.mean([r[7] for r in sel])),
                                      "total_s": float(np.mean([r[8] for r in sel]))}
for k, c in enumerate(ml.CLASSES):
    st: dict = {}
    for sid, _, nd, n, kk, *_ in rows:
        if cls[int(sid.removeprefix("row-"))] == k:
            v = st.setdefault(sid, [0, 0, 0])
            v[0] += 1; v[1] += n; v[2] += kk  # noqa: E702
    if not st:
        continue
    N, K = sum(v[1] for v in st.values()), sum(v[2] for v in st.values())
    lam = K / N
    lo = float(stats.beta.ppf(0.025, K, N - K + 1)) if K else 0.0
    hi = float(stats.beta.ppf(0.975, K + 1, N - K))
    exp = np.array([v[1] * lam for v in st.values()])
    chi = float((((np.array([v[2] for v in st.values()]) - exp) ** 2) / np.where(exp > 0, exp, 1)).sum())
    out["by_class"][c] = {"states": len(st), "runs": sum(v[0] for v in st.values()), "dis": N, "dis_pass": K,
                          "per_dis_rate": lam, "cp95": [lo, hi],
                          "homogeneity_chi2": chi, "dof": len(st) - 1, "p": float(stats.chi2.sf(chi, len(st) - 1)) if K else None,
                          "per_state": {sid: {"runs": v[0], "dis": v[1], "pass": v[2]} for sid, v in sorted(st.items())}}
json.dump(out, sys.stdout, indent=1)
