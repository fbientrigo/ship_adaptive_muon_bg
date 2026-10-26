"""One iteration of the minimal utility-guided loop on the PR #1432 filter stage.

  score  (FairShip env)  U0 + TGeo initial volume for every pool row  -> <out>/pool_scores.npz + .json
  draw   (any env)       eligibility gate + reference/tilt draws        -> <out>/draws.csv + draws.json
  run    (FairShip env)  per draw: TTree injection + PR #1432 chain     -> <runs>/<draw_id>/{inject,chain}/, bundle.json, cost.json
  export (core venv)     bundles -> one Parquet dataset per arm         -> <data>/<arm>/*.parquet + manifest.json
  report (core venv)     per-arm efficiency, multiplicity, coverage     -> <data>/report.json + report.md

``run`` is resumable: a draw with a bundle.json is never re-run.  Each draw is one execution with its own seed; an
injection failure is a TECHNICAL_FAILURE execution, never a physics negative.

Every path and hash the result depends on is written next to it.  See ``ship_muon_bg.minimal_loop`` for the semantics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from ship_muon_bg.adapters.current_main_mudis import _git_head
from ship_muon_bg.adapters.fairship import LEGACY_AFTERMS_TRANSFORM as TRANSFORM
from ship_muon_bg.adapters.fairship import CandidateInjectionRecord
from ship_muon_bg import minimal_loop as ml


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(a: np.ndarray, i: int) -> CandidateInjectionRecord:
    px, py, pz, x, y, z, pid, w = (float(v) for v in a[i])
    return CandidateInjectionRecord(f"row-{i}", px, py, pz, x, y, z, int(pid), w, 1.0)


def score(args) -> None:
    import audit_fairship_tgeo as A  # scripts/ sibling; needs the FairShip ROOT env

    A.configure_fairship(args.fairship)
    import ROOT  # type: ignore

    geo_file = ROOT.TFile.Open(str(args.geo))  # keep the file alive: a chained Open().Get() segfaults
    geom = geo_file.Get("FAIRGeom")
    a = ml.load_pool()
    u0 = np.full(len(a), np.nan)
    volume = np.full(len(a), "", dtype=object)
    upstream, tracker = np.zeros(len(a), bool), np.zeros(len(a), bool)
    for i in np.where(ml.valid_rows(a))[0]:
        c = record(a, i)
        x, y, z = TRANSFORM.apply(c)  # the exact placement the injection uses
        info, rows = A.trajectory(geom, dict(px=c.px, py=c.py, pz=c.pz, x=x, y=y, z=z))
        u0[i] = float(len(info["ray_sensitive_volumes"]) > 0)
        volume[i] = info["initial_volume"]
        names = {r["volume"].split("_")[0] for r in rows}
        upstream[i], tracker[i] = "Upstream" in names, bool(names & {"Tr1", "Tr2", "Tr3", "Tr4"})
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "pool_scores.npz", u0=u0, initial_volume=volume.astype(str), ray_upstream=upstream, ray_tracker=tracker)
    meta = {"u0_definition": "straight TGeo ray from the transformed state crosses a LiSc* (SBT) volume; no field, no scattering",
            "ray_flags": "ray_upstream: ray enters an Upstream* (UBT) volume; ray_tracker: ray enters Tr1..Tr4",
            "pool": str(ml.POOL_NPZ), "pool_sha256": sha256(ml.POOL_NPZ), "geofile": str(args.geo), "geofile_sha256": sha256(args.geo),
            "fairship_dir": str(args.fairship), "fairship_commit": _git_head(args.fairship), "transform": TRANSFORM.manifest(),
            "n_rows": len(a), "n_scored": int(np.isfinite(u0).sum()), "n_u0_1": int((u0 == 1).sum()),
            "scores_sha256": sha256(args.out / "pool_scores.npz")}
    (args.out / "pool_scores.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta, indent=1))


def checked_scores(path: Path) -> dict:
    """The score manifest, after checking that the scores, the pool and the transform are the ones it records."""
    meta = json.loads(Path(path).with_suffix(".json").read_text())
    if sha256(path) != meta["scores_sha256"] or sha256(ml.POOL_NPZ) != meta["pool_sha256"] or meta["transform"] != TRANSFORM.manifest():
        raise SystemExit(f"{path}: scores, pool or transform differ from {Path(path).with_suffix('.json')}")
    return meta


def checked_draws(path: Path) -> tuple[list[dict], dict]:
    meta = json.loads(path.with_suffix(".json").read_text())
    if sha256(path) != meta["draws_sha256"]:
        raise SystemExit(f"{path} differs from the hash in {path.with_suffix('.json')}")
    checked_scores(Path(meta["scores"]))
    return read_draws(path), meta


def draw(args) -> None:
    a = ml.load_pool()
    scores_meta = checked_scores(args.scores)
    s = np.load(args.scores, allow_pickle=False)
    u0, volume = s["u0"], s["initial_volume"]
    status = ml.eligibility(a, u0, volume)
    rows = ml.draw(status, u0, args.n, args.seed)
    for r in rows:
        r["status"], r["initial_volume"] = status[r["row_index"]], str(volume[r["row_index"]])
        r["p_gev"] = float(np.linalg.norm(a[r["row_index"], :3]))
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "draws.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    per_arm = {arm: Counter(r["row_index"] for r in rows if r["arm"] == arm) for arm in ml.ARMS}
    meta = {"scores": str(args.scores), "scores_sha256": sha256(args.scores), "scores_meta": scores_meta, "seed": args.seed, "n_per_arm": args.n,
            "delta": ml.DELTA, "alpha": ml.ALPHA, "eligible_volumes": list(ml.ELIGIBLE_VOLUMES), "gate_status": "PROVISIONAL",
            "eligibility_counts": dict(Counter(status.tolist())),
            "ineligible_volume_counts": dict(Counter(volume[status == "INELIGIBLE_IN_SHIELD_MAGNET"].tolist())),
            "invalid_or_unscored_rows": {int(i): status[i] for i in np.where(np.isin(status, ["INVALID_ROW", "UNSCORED"]))[0]},
            "u0_1_rows": {int(i): status[i] for i in np.where(u0 == 1)[0]},
            "concentration": ml.concentration(status, u0),
            "drawn": {arm: {"executions": sum(c.values()), "distinct_rows": len(c), "multiplicity": {str(k): v for k, v in c.most_common()}}
                      for arm, c in per_arm.items()},
            "draws_sha256": sha256(args.out / "draws.csv")}
    (args.out / "draws.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps({k: meta[k] for k in ("eligibility_counts", "concentration", "drawn")}, indent=1))


CONFIG = "pr1432_400c0fb_TRY_2026"
STATE_DEFINITION = "legacy_row_plus2p5916m_provisional"  # same subject semantics as run_pr1432_stage_smoke.py


def read_draws(path: Path) -> list[dict]:
    with path.open() as f:
        return [{**r, **{k: int(r[k]) for k in ("row_index", "sim_seed")}, **{k: float(r[k]) for k in ("proposal_prob", "u0", "h", "p_gev")}}
                for r in csv.DictReader(f)]


def run(args) -> None:
    from ship_muon_bg.adapters.fairship import FairShipTTreeConnector
    from ship_muon_bg.adapters.pr1432_mudis import PR1432MuonDISRunner, build_bundle
    from ship_muon_bg.entities.subject import TagSubject
    from ship_muon_bg.simulation.evaluation import EvaluationRequest, bundle_records

    draws, draws_meta = checked_draws(args.draws)
    if _git_head(args.fairship) != draws_meta["scores_meta"]["fairship_commit"]:
        raise SystemExit(f"{args.fairship} is not the checkout the eligibility gate and U0 were scored with")
    a = ml.load_pool()
    runner = PR1432MuonDISRunner(args.fairship, timeout=args.timeout)
    options = {"n_dis_per_material": args.n_dis, "filter_option": args.filter_option, "shield": "TRY_2026", "strawDesign": 10,
               "last_step": args.last_step}
    for d in draws:
        out = args.runs / d["draw_id"]
        identity = {"draw": d, "options": options, "pool_sha256": draws_meta["scores_meta"]["pool_sha256"]}
        if (out / "bundle.json").exists():
            if json.loads((out / "draw.json").read_text()) != identity:
                raise SystemExit(f"{out} holds a run of a different draw/options; use a fresh --runs directory")
            continue
        if out.exists():  # interrupted attempt: keep its artifacts aside, retry in a fresh directory
            n = 1
            while out.with_name(f"{out.name}.attempt-{n}").exists():
                n += 1
            out.rename(out.with_name(f"{out.name}.attempt-{n}"))
        out.mkdir(parents=True)
        (out / "draw.json").write_text(json.dumps(identity) + "\n")
        if d["status"] != "ELIGIBLE":
            raise SystemExit(f"{d['draw_id']}: row {d['row_index']} is {d['status']}; draws.csv was not produced by the gate")
        request = EvaluationRequest(d["draw_id"], (TagSubject(f"row-{d['row_index']}", "post_shield_muon", STATE_DEFINITION),), CONFIG,
                                    d["sim_seed"], options=options)
        t0 = time.monotonic()
        t_inject = None
        try:
            inj = FairShipTTreeConnector(args.fairship, python_executable=sys.executable, timeout=args.timeout).run(
                [record(a, d["row_index"])], out / "inject", transform=TRANSFORM, seed=d["sim_seed"])
            t_inject = time.monotonic() - t0
            if not inj.ok:
                raise RuntimeError(f"injection failed: {inj.technical_failures or inj.verification.get('status')}")
            bundle = runner.run(request, sim_connector=inj.output_root, geofile=out / "inject" / "geo_connector.root",
                                output_directory=out / "chain", n_dis_per_material=args.n_dis, filter_option=args.filter_option,
                                last_step=args.last_step)
        except Exception as exc:  # noqa: BLE001 - any failure before the chain concludes is technical, never a negative
            bundle = build_bundle(request, {"fairship_commit": _git_head(args.fairship), "n_dis": args.n_dis, "filter_option": args.filter_option,
                                            "seed": d["sim_seed"], "steps": {}, "failure": {"prepare": f"{type(exc).__name__}: {exc}"[:500]}})
        (out / "cost.json").write_text(json.dumps({"inject_wall_s": t_inject, "total_wall_s": time.monotonic() - t0}) + "\n")
        (out / "bundle.json").write_text(json.dumps(bundle_records(bundle), indent=1) + "\n")
        e = bundle.executions[0]
        print(d["draw_id"], d["row_index"], e.execution_status.value, len(bundle.realizations), "DIS", f"{time.monotonic() - t0:.0f}s", flush=True)


def export(args) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from ship_muon_bg import dataset as ds

    draws, draws_meta = checked_draws(args.draws)
    meta = {"draws": str(args.draws), "draws_sha256": sha256(args.draws), "draws_meta": draws_meta}
    for d in draws:  # every bundle must be the run of exactly this draw
        run_dir = args.runs / d["draw_id"]
        b = json.loads((run_dir / "bundle.json").read_text()) if (run_dir / "bundle.json").exists() else None
        if b is not None and (json.loads((run_dir / "draw.json").read_text())["draw"] != d or b["request_id"] != d["draw_id"]
                              or [(e["subject_id"], e["seed"]) for e in b["executions"]] != [(f"row-{d['row_index']}", d["sim_seed"])]):
            raise SystemExit(f"{run_dir}: bundle does not belong to draw {d['draw_id']}")
    for arm in dict.fromkeys(d["arm"] for d in draws):
        rows = [d for d in draws if d["arm"] == arm]
        missing = [d["draw_id"] for d in rows if not (args.runs / d["draw_id"] / "bundle.json").exists()]
        if missing:
            raise SystemExit(f"{arm}: draws without a bundle (run not finished?): {missing}")
        out = args.data / arm
        cost = [{"request_id": d["draw_id"], **json.loads((args.runs / d["draw_id"] / "cost.json").read_text())} for d in rows]
        out.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), out / "draws.parquet")  # proposal origin of every request_id
        pq.write_table(pa.Table.from_pylist(cost), out / "cost.parquet")
        ds.export([args.runs / d["draw_id"] / "bundle.json" for d in rows], out,
                  {**meta, "arm": arm, "draws_parquet_sha256": sha256(out / "draws.parquet"), "cost_parquet_sha256": sha256(out / "cost.parquet")})
        print(arm, json.loads((out / "manifest.json").read_text())["tables"])


def upper95(n: int) -> float | None:
    """One-sided 95 % upper bound on a rate after 0 positives in ``n`` trials (exact binomial)."""
    return None if n == 0 else 1 - 0.05 ** (1 / n)


def report(args) -> None:
    import duckdb
    from ship_muon_bg import dataset as ds
    from ship_muon_bg.adapters.pr1432_mudis import stage_ids

    _, draws_meta = checked_draws(args.draws)
    scores = Path(draws_meta["scores"])
    a, u0 = ml.load_pool(), np.load(scores)["u0"]
    status = ml.eligibility(a, u0, np.load(scores)["initial_volume"])
    pool = a[status == "ELIGIBLE"]
    feats = {"p_gev": lambda m: np.linalg.norm(m[:, :3], axis=1), "pt_gev": lambda m: np.hypot(m[:, 0], m[:, 1]),
             "x_m": lambda m: m[:, 3], "y_m": lambda m: m[:, 4]}
    edges = {k: np.quantile(f(pool), np.linspace(0, 1, 11)) for k, f in feats.items()}
    policy_u = draws_meta.get("u_policy") or {"u0_scores_sha256": sha256(scores), "u0_definition": "straight-ray SBT crossing (pool_scores.json)", "delta": ml.DELTA, "alpha": ml.ALPHA}
    rep: dict = {"scope": "PR #1432 filter stage only; not Y^(k), not reconstruction, not B. Exploratory smoke, no power claim.",
                 "reference_is": "uniform over eligible rows (empirical), not the nominal physical measure", "arms": {}}
    arms = list(dict.fromkeys(r for r, in duckdb.sql(f"select arm from read_csv('{args.draws}', header=true)").fetchall()))
    for arm in arms:
        d = args.data / arm
        prov = {tuple(sorted((k, v) for k, v in json.loads(r).items() if k in ("fairship_commit", "n_dis_per_material", "filter_option")))
                for r, in duckdb.sql(f"select provenance from '{d}/executions.parquet'").fetchall()}
        if len(prov) != 1:
            raise SystemExit(f"{arm}: executions disagree on commit/budget/filter: {prov}")
        k = dict(prov.pop())
        filt = stage_ids(k["fairship_commit"], int(k["n_dis_per_material"]), int(k["filter_option"]))["filter"]
        rows = duckdb.execute(f"""
            select x.request_id, x.subject_id, x.execution_status, f.evaluation_status, f.decision, c.total_wall_s, dr.u0
            from '{d}/executions.parquet' x
            join '{d}/cost.parquet' c using (request_id) join '{d}/draws.parquet' dr on dr.draw_id = x.request_id
            left join (select * from '{d}/decisions.parquet' where stage_definition_id = ?) f on f.subject_ref = x.execution_id""",
            [filt]).fetchall()
        n = len(rows)
        tech = sum(r[2] == "technical_failure" or r[3] == "technically_unavailable" for r in rows)
        not_eval = sum(r[2] == "succeeded" and r[3] in (None, "not_evaluated") for r in rows)
        valid = [r for r in rows if r[3] == "evaluated"]
        passes = sum(bool(r[4]) for r in valid)
        hours = sum(r[5] for r in rows) / 3600
        snap = ds.snapshot([d], filt, args.data / "snapshots", split_seed=20261025,
                           policy={"u": policy_u, "sampling": {"arm": arm, "draws_sha256": draws_meta["draws_sha256"], "seed": draws_meta["seed"]}})
        states = duckdb.sql(f"select count(*), count(*) filter (where positive > 0), count(*) filter (where valid > 0) from '{snap}/rows.parquet'").fetchone()
        drawn = a[[int(r) for r, in duckdb.sql(f"select row_index from '{d}/draws.parquet'").fetchall()]]
        strata = {}
        for k in (0.0, 1.0):
            v = [r for r in valid if r[6] == k]
            pk = sum(bool(r[4]) for r in v)
            strata[f"u0_{int(k)}"] = {"valid": len(v), "pass": pk, "upper95_if_zero": upper95(len(v)) if pk == 0 else None}
        rep["arms"][arm] = {
            "executions": n, "distinct_states": states[0], "max_multiplicity": max(Counter(r[1] for r in rows).values()),
            "technical_failure": tech, "not_evaluated": not_eval, "valid": len(valid), "filter_pass_executions": passes,
            "states_with_a_pass": states[1], "states_with_a_valid_execution": states[2],
            "wall_hours": round(hours, 3), "pass_per_wall_hour": round(passes / hours, 2) if hours else None,
            "pass_per_valid_execution": round(passes / len(valid), 3) if valid else None,
            "upper95_pass_rate_if_zero": upper95(len(valid)) if passes == 0 else None,
            "by_u0": strata, "snapshot": str(snap),
            "coverage_deciles_of_eligible_pool": {k: np.histogram(np.clip(f(drawn), edges[k][0], edges[k][-1]), edges[k])[0].tolist()
                                                  for k, f in feats.items()},
        }
    if arms == list(ml.ARMS):
        ref, tilt = (ml.arm_probabilities(arm, status, u0) for arm in ml.ARMS)
        u0_0 = (status == "ELIGIBLE") & (u0 == 0)
        rep["missed_region"] = {"reference_u0_0": rep["arms"]["reference_uniform_eligible"]["by_u0"]["u0_0"],
                                "tilt_over_reference_prob_u0_0": float(tilt[u0_0][0] / ref[u0_0][0]),  # normalised, per row
                                "within_tilt_weight_ratio_u0_0_over_u0_1": ml.DELTA ** ml.ALPHA,
                                "tilt_mass_on_u0_0": float(tilt[u0_0].sum()), "reference_mass_on_u0_0": float(ref[u0_0].sum())}
    (args.data / "report.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))


def state_outcomes(snapshots: list[Path]) -> tuple[dict[str, bool], list[str]]:
    """Distinct states with >= 1 valid execution -> any positive.  Pools a state's executions across snapshots of one
    stage/configuration only; refuses snapshots that disagree on stage, estimator or configuration."""
    import duckdb

    specs = [json.loads((Path(s) / "snapshot.json").read_text()) for s in snapshots]
    keys = {(x["stage_definition_id"], x["estimator_definition_id"], x["fs_sim_configuration_id"], x["options_digest"]) for x in specs}
    if len(keys) != 1:
        raise SystemExit(f"snapshots disagree on stage/estimator/configuration: {keys}")
    for x, s in zip(specs, snapshots):
        if sha256(Path(s) / "rows.parquet") != x["rows_sha256"]:
            raise SystemExit(f"{s}/rows.parquet does not match its snapshot.json")
    files = ", ".join(f"'{Path(s) / 'rows.parquet'}'" for s in snapshots)
    rows = duckdb.sql(f"select subject_id, sum(valid), sum(positive) from read_parquet([{files}]) group by 1").fetchall()
    return {sid: pos > 0 for sid, valid, pos in rows if valid > 0}, [x["snapshot_id"] for x in specs]


def classes(scores: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = np.load(scores)
    a = ml.load_pool()
    eligible = (ml.eligibility(a, s["u0"], s["initial_volume"]) == "ELIGIBLE") & (np.linalg.norm(a[:, :3], axis=1) >= ml.MIN_P_GEV)
    return ml.geo_class(s["u0"], s["ray_upstream"], s["ray_tracker"]), eligible, a


def update_u(args) -> None:
    """EXPERIMENT (U definition) on a FEATURE path: U is fitted only from frozen snapshots and written as a versioned file."""
    checked_scores(args.scores)
    cls, _, _ = classes(args.scores)
    outcome, snap_ids = state_outcomes(args.snapshots)
    state_class = {sid: int(cls[int(sid.removeprefix("row-"))]) for sid in outcome}
    u = ml.fit_u(state_class, outcome)
    spec = {"u_definition": "Laplace rate of filter-PASS states per straight-ray class " + "/".join(ml.CLASSES), "classes": u,
            "fit_snapshots": snap_ids, "fit_states": sorted(outcome), "scores_sha256": sha256(args.scores)}
    spec["u_id"] = "u-" + hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]
    if args.evaluate:  # out-of-sample check of an earlier U on states it was not fitted on
        prev = json.loads(args.evaluate.read_text())
        fresh = {s for s in outcome if s not in set(prev["fit_states"])}
        spec["evaluation_of_previous"] = {"u_id": prev["u_id"], "fresh_states": len(fresh), "by_class": {
            name: {"predicted_u": prev["classes"][name]["u"],
                   "observed": f"{sum(outcome[s] for s in fresh if state_class[s] == k)}/{sum(state_class[s] == k for s in fresh)}"}
            for k, name in enumerate(ml.CLASSES)}}
    args.out.write_text(json.dumps(spec, indent=1) + "\n")
    print(json.dumps(spec, indent=1))


def draw_strata(args) -> None:
    """FEATURE: strategic use of expensive runs -- no run on a known outcome (p < 2 GeV, already evaluated states),
    without replacement, an exploration floor per class, the rest proportional to U."""
    import duckdb

    scores_meta = checked_scores(args.scores)
    cls, eligible, a = classes(args.scores)
    s = np.load(args.scores)
    u = json.loads(args.u.read_text())
    exclude = {int(r.removeprefix("row-")) for d in args.exclude_data
               for r, in duckdb.sql(f"select distinct subject_id from '{d}/executions.parquet'").fetchall()}
    available = {name: int(sum(1 for i in np.where(eligible & (cls == k))[0] if int(i) not in exclude)) for k, name in enumerate(ml.CLASSES)}
    alloc = ml.allocate({k: v["u"] for k, v in u["classes"].items()}, available, args.n, args.floor)
    rows = ml.draw_strata(cls, eligible, alloc, exclude, args.seed)
    for r in rows:
        i = r["row_index"]
        r.update(u0=float(s["u0"][i]), h=u["classes"][r["arm"]]["u"], status="ELIGIBLE", initial_volume=str(s["initial_volume"][i]),
                 p_gev=float(np.linalg.norm(a[i, :3])))
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "draws.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    meta = {"scores": str(args.scores), "scores_sha256": sha256(args.scores), "scores_meta": scores_meta, "seed": args.seed,
            "design": "stratified by straight-ray class, without replacement; h column = U of the class",
            "u_policy": {"u_id": u["u_id"], "u_sha256": sha256(args.u)}, "n": args.n, "floor": args.floor,
            "excluded_already_evaluated": len(exclude), "min_p_gev": ml.MIN_P_GEV, "available": available, "allocation": alloc,
            "draws_sha256": sha256(args.out / "draws.csv")}
    (args.out / "draws.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps({k: meta[k] for k in ("available", "allocation", "u_policy")}, indent=1))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("score")
    p.add_argument("--fairship", type=Path, required=True)
    p.add_argument("--geo", type=Path, required=True, help="geo_connector.root of an injection on the same checkout")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("draw")
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=20261025)
    p = sub.add_parser("run")
    p.add_argument("--fairship", type=Path, required=True)
    p.add_argument("--draws", type=Path, required=True)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--n-dis", type=int, default=10)
    p.add_argument("--filter-option", type=int, default=1)
    p.add_argument("--last-step", default="filter", choices=["filter", "replay", "reco"])
    p.add_argument("--timeout", type=float, default=1800)
    for name in ("export", "report"):
        p = sub.add_parser(name)
        p.add_argument("--draws", type=Path, required=True)
        p.add_argument("--runs", type=Path, required=True)
        p.add_argument("--data", type=Path, required=True)
    p = sub.add_parser("update-u")
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--snapshots", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--evaluate", type=Path, help="earlier u.json to check on the states it was not fitted on")
    p = sub.add_parser("draw-strata")
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--u", type=Path, required=True)
    p.add_argument("--exclude-data", type=Path, nargs="*", default=[], help="dataset dirs whose states are already evaluated")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--floor", type=int, default=5)
    p.add_argument("--seed", type=int, default=20261026)
    args = ap.parse_args(argv)
    {"score": score, "draw": draw, "run": run, "export": export, "report": report, "update-u": update_u,
     "draw-strata": draw_strata}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
