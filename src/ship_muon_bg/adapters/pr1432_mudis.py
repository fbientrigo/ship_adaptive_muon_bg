"""PR #1432 Muon DIS chain adapter: prepare -> filter -> Geant4 replay -> ShipReco.

One source state per call.  ROOT is touched only by the ``read_*`` functions,
which return plain dicts ("facts"); ``build_bundle`` is a pure function from
facts to canonical records, so every status rule is testable without FairShip.

Status vocabulary (existing records, no new schema):

* PASS / PHYSICS_REJECTED  = ``StageDecision`` EVALUATED with ``True`` / ``False``
* NOT_EVALUATED            = ``StageDecision`` NOT_EVALUATED (stage never applied)
* TECHNICAL_FAILURE        = ``FSSimExecution`` TECHNICAL_FAILURE (run) or a
  ``StageDecision`` TECHNICALLY_UNAVAILABLE (stage evidence could not be trusted)

The chain's native ``wDIS`` is stored only as an observation named
``..._not_event_weight``; it carries no cross section and, for injected states,
no source weight (``docs/1012_FairShipUpdate``).
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from ship_muon_bg.adapters.current_main_mudis import _git_head, _sha256, _write_json
from ship_muon_bg.entities.decision import (
    DecisionEvaluationStatus as DS,
    StageDecision,
    stage_decision_id,
)
from ship_muon_bg.entities.identifiers import content_hash, definition_id
from ship_muon_bg.entities.lineage import (
    OPTIONS_DIGEST_PROVENANCE_KEY,
    ExecutionStatus,
    FSSimExecution,
    InteractionRealization,
    ReconstructedCandidate,
)
from ship_muon_bg.entities.observation import (
    ObservationEnvelope,
    ObservationEvaluationStatus,
    ScalarObservationPayload,
)
from ship_muon_bg.simulation.evaluation import (
    EvaluationBundle,
    EvaluationRequest,
    verify_evaluation_bundle,
)

MATERIALS = ("MS", "UBT", "SBTsens", "SBTfr", "SSTsens", "SSTfr", "HE", "AIR", "HeBalloon", "REST")
#: ``MuDISProcessor.cxx:437`` skips muons with p < ``fMinPythiaP`` (default 2, ``:44``): a declared generator support limit.
GENERATOR_MIN_P_GEV = 2.0
STEPS = ("prepare", "filter", "replay", "reco")


# ---------------------------------------------------------------- ROOT readers
def _open(path: Path, tree: str):
    import ROOT  # type: ignore

    root_file = ROOT.TFile.Open(str(path), "READ")  # keep a reference: a chained Open().Get() segfaults
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"unreadable ROOT file: {path}")
    found = root_file.Get(tree)
    if not found:
        raise RuntimeError(f"{path}: no tree {tree}")
    return root_file, found


def _entry(tree, index: int, path: Path) -> None:
    """A failed ``GetEntry`` leaves stale/empty buffers that read as zeros; that must never become evidence."""
    if tree.GetEntry(index) <= 0:
        raise RuntimeError(f"{path}: entry {index} unreadable")


def read_input_muon(path: Path) -> dict[str, float]:
    f, t = _open(path, "cbmsim")
    if int(t.GetEntries()) < 1:
        raise RuntimeError(f"{path}: no entries")
    _entry(t, 0, path)
    track = t.MCTrack[0]
    out = {"p_gev": float(track.GetP()), "w": float(track.GetWeight())}
    f.Close()
    return out


def read_muondis(path: Path) -> Optional[dict[str, Any]]:
    """Per-material counts and DIS vertices of a prepared/filtered ``MuonDIS`` file; ``None`` if it has no entry."""
    f, t = _open(path, "MuonDIS")
    n = int(t.GetEntries())
    if n == 0:
        f.Close()
        return None
    if n != 1:
        raise RuntimeError(f"{path}: expected one muon entry, found {n}")
    _entry(t, 0, path)
    materials: dict[str, Any] = {}
    for m in MATERIALS:
        n_dis, length = int(getattr(t, f"muon_nDISevt_{m}")), float(getattr(t, f"muon_path_length_{m}"))
        if n_dis == 0 and length <= 0:
            continue
        materials[m] = {
            "path_length_cm": length, "n_dis": n_dis, "n_generated": int(getattr(t, f"muon_nDISGenerated_{m}")),
            "wDIS_native": float(getattr(t, f"muon_wDIS_{m}")),
            "vertices": [tuple(float(getattr(t, f"mudis_DISv{c}_{m}")[i]) for c in "xyzt") for i in range(n_dis)],
        }
    f.Close()
    return {"materials": materials}


def read_replay(path: Path) -> list[tuple[str, int, int]]:
    f, t = _open(path, "cbmsim")
    events = []
    for i in range(int(t.GetEntries())):
        _entry(t, i, path)
        events.append((str(t.muDIS_material), int(t.muDIS_muEntry), int(t.muDIS_disIndex)))
    f.Close()
    return events


def read_reco(path: Path) -> list[dict[str, int]]:
    f, t = _open(path, "ship_reco_sim")
    events = []
    for i in range(int(t.GetEntries())):
        _entry(t, i, path)
        events.append({"n_fit_tracks": len(t.FitTracks), "n_particles": len(t.Particles)})
    f.Close()
    return events


# --------------------------------------------------------------- pure builder
def _obs(oid: str, ref: str, definition: str, units: str, value: float, evidence: str, digest: str) -> ObservationEnvelope:
    return ObservationEnvelope(oid, ref, definition, units, ObservationEvaluationStatus.COMPUTED, evidence,
                               ScalarObservationPayload(float(value)), {OPTIONS_DIGEST_PROVENANCE_KEY: digest})


def stage_ids(fairship_commit: str, n_dis: int, filter_option: int) -> dict[str, str]:
    """Content-addressed stage ids: a different commit, trial budget or filter never shares an id."""
    base = {"fairship_commit": fairship_commit, "n_dis_per_material": n_dis, "filter_option": filter_option}
    return {
        "generation": definition_id("pr1432_dis_generation_v0", {**base, "min_muon_p_gev": GENERATOR_MIN_P_GEV}),
        "filter": definition_id("pr1432_filter_any_v0", base),
        "filter_dis": definition_id("pr1432_filter_dis_v0", base),
        "replay": definition_id("pr1432_replay_complete_v0", base),
        "reco": definition_id("pr1432_reco_candidate_any_v0", {**base, "candidate": "ShipReco Particles entry"}),
    }


def _match_kept(generated: Sequence[tuple], kept: Sequence[tuple]) -> Optional[list[int]]:
    """Indices into ``generated`` of each kept vertex, or ``None`` if the (ordered, exact) join is not unique."""
    if len(set(generated)) != len(generated):
        return None
    out, pos = [], 0
    for vertex in kept:
        while pos < len(generated) and generated[pos] != vertex:
            pos += 1
        if pos == len(generated):
            return None
        out.append(pos)
        pos += 1
    return out


def build_bundle(request: EvaluationRequest, facts: Mapping[str, Any], *, is_physical: bool = True) -> EvaluationBundle:
    """Canonical records from one chain run.  ``facts`` keys: fairship_commit, n_dis, filter_option, seed, muon_p_gev,
    steps{name:{rc,wall_s}}, prepare, filter, replay, reco, last_step, failure{step:reason}."""
    subject = request.subjects[0]
    digest, seed = request.options_digest, request.seed
    if facts.get("seed") != seed:  # the recorded simulation seed is the identity of the trial; it must be the request's
        raise ValueError(f"facts were produced with seed {facts.get('seed')!r}, request says {seed}")
    ex_id = "exec:" + content_hash({"subject": subject.subject_id, "config": request.fs_sim_configuration_id, "options_digest": digest, "seed": seed})[:20]
    steps, failure = facts.get("steps", {}), dict(facts.get("failure", {}))

    def attested(step: str) -> bool:  # a stage may only be concluded from a step that is recorded as having returned 0
        return steps.get(step, {}).get("rc") == 0 and step not in failure

    def block(raw):  # JSON round trips turn vertex tuples into lists
        return None if raw is None else {"materials": {m: {**v, "vertices": [tuple(x) for x in v["vertices"]]} for m, v in raw["materials"].items()}}

    prepare, filt = block(facts.get("prepare")), block(facts.get("filter"))
    replay = None if facts.get("replay") is None else [tuple(e) for e in facts["replay"]]
    prov = {OPTIONS_DIGEST_PROVENANCE_KEY: digest, "runner": "pr1432_mudis_v0", "fairship_commit": facts["fairship_commit"],
            "n_dis_per_material": str(facts["n_dis"]), "filter_option": str(facts["filter_option"]),
            "pythia6_seed": str(seed), "geant4_seed": str(seed), "geofile_sha256": str(facts.get("geofile_sha256", "")),
            **{f"wall_s_{k}": f"{v['wall_s']:.2f}" for k, v in steps.items() if v.get("wall_s") is not None}}
    if not attested("prepare"):  # nothing trustworthy exists: the only whole-run technical failure
        ex = FSSimExecution(ex_id, subject.subject_id, request.fs_sim_configuration_id, ExecutionStatus.TECHNICAL_FAILURE,
                            seed, failure.get("prepare", "prepare not recorded as returning 0"), prov)
        return EvaluationBundle(request.request_id, "pr1432_mudis_v0", is_physical, executions=(ex,))
    ex = FSSimExecution(ex_id, subject.subject_id, request.fs_sim_configuration_id, ExecutionStatus.SUCCEEDED, seed, provenance=prov)
    sid = stage_ids(facts["fairship_commit"], facts["n_dis"], facts["filter_option"])
    obs: list[ObservationEnvelope] = []
    decisions: list[StageDecision] = []

    def ex_obs(name: str, units: str, value: float) -> str:
        oid = f"obs:{name}:{ex_id}"
        obs.append(_obs(oid, ex_id, f"pr1432:{name}_v0", units, value, f"{ex_id}#{name}", digest))
        return oid

    broken = False  # set once any stage is TECHNICALLY_UNAVAILABLE: downstream stages are then unavailable too, not "not evaluated"

    def decide(key: str, status: DS, value: Optional[bool], reason: str, evidence: Sequence[str] = (), ref: str = ex_id) -> None:
        nonlocal broken
        broken = broken or status is DS.TECHNICALLY_UNAVAILABLE
        decisions.append(StageDecision(stage_decision_id(stage_definition_id=sid[key], subject_ref=ref, evidence_references=tuple(evidence),
                                                         evaluation_status=status),
                                       ref, sid[key], status, value, reason, tuple(evidence), {OPTIONS_DIGEST_PROVENANCE_KEY: digest}))

    def skip(key: str, reason: str) -> None:
        decide(key, DS.TECHNICALLY_UNAVAILABLE if broken else DS.NOT_EVALUATED, None, reason)

    ex_obs("wall_seconds_total", "s", sum(v["wall_s"] for v in steps.values() if v.get("wall_s") is not None))
    ex_obs("dis_requested_per_material", "count", facts["n_dis"])

    # ---- generation
    realizations: list[InteractionRealization] = []
    gen_index: dict[tuple, str] = {}  # (material, idx) -> realization id
    gen_vertices: dict[str, list] = {}
    bad = [m for m, v in (prepare or {"materials": {}})["materials"].items() if len(v["vertices"]) != v["n_dis"]]
    if prepare is None:
        low = facts["muon_p_gev"] < GENERATOR_MIN_P_GEV
        decide("generation", DS.NOT_EVALUATED if low else DS.TECHNICALLY_UNAVAILABLE, None,
               f"no prepared entry; muon p={facts['muon_p_gev']:.3f} GeV " + ("is below the generator support cut (not physics)" if low else "is above the cut: unexplained"))
    elif bad:
        decide("generation", DS.TECHNICALLY_UNAVAILABLE, None, f"vertex list length != nDISevt in {bad}")
    else:
        definition = definition_id("pr1432_muon_dis_v0", {"fairship_commit": facts["fairship_commit"]})
        for m, v in prepare["materials"].items():
            gen_vertices[m] = v["vertices"]
            for i in range(v["n_dis"]):
                rid = f"real:{ex_id}:{m}:{i}"
                realizations.append(InteractionRealization(rid, ex_id, "muon_dis", definition))
                gen_index[(m, i)] = rid
                obs.append(_obs(f"obs:vz:{rid}", rid, "pr1432:dis_vertex_z_cm_v0", "cm", v["vertices"][i][2], f"{rid}#vz", digest))
            ex_obs(f"generated_{m}", "count", v["n_dis"])
            ex_obs(f"path_length_{m}", "cm", v["path_length_cm"])
            ex_obs(f"wDIS_native_not_event_weight_{m}", "FairShip-native", v["wDIS_native"])
        ev = ex_obs("generated_total", "count", len(realizations))
        decide("generation", DS.EVALUATED, len(realizations) >= 1,
               "DIS generated" if realizations else "muon processed but crossed no DIS target path", [ev])
    n_generated = len(realizations)

    # ---- filter (execution level + one decision per generated realization)
    kept_map: dict[tuple, str] = {}  # (material, kept index) -> realization id
    flog = facts.get("filter_log")
    if n_generated == 0:
        skip("filter", "no generated DIS to filter")
    elif not attested("filter"):
        decide("filter", DS.TECHNICALLY_UNAVAILABLE, None, failure.get("filter", "filter not recorded as returning 0"))
    elif not flog or flog.get("skipped") != 0 or flog.get("saved") != (1 if filt else 0):
        # FairShip's filter skips unreadable/malformed entries and still exits 0 with an empty tree: that is not "nothing passed"
        decide("filter", DS.TECHNICALLY_UNAVAILABLE, None, f"filter log does not attest a clean pass: {flog}")
    else:
        kept_vertices = {m: (filt["materials"][m]["vertices"] if filt and m in filt["materials"] else []) for m in gen_vertices}
        joins = {m: _match_kept(gen_vertices[m], kept_vertices[m]) for m in gen_vertices}
        extra = set(filt["materials"]) - set(gen_vertices) if filt else set()
        if any(j is None for j in joins.values()) or extra or (filt and any(len(v["vertices"]) != v["n_dis"] for v in filt["materials"].values())):
            decide("filter", DS.TECHNICALLY_UNAVAILABLE, None, f"kept->generated lineage join failed (materials {[m for m, j in joins.items() if j is None]}, extra {sorted(extra)})")
        else:
            for m, j in joins.items():
                for k, g in enumerate(j):
                    kept_map[(m, k)] = gen_index[(m, g)]
            for rid in gen_index.values():
                decide("filter_dis", DS.EVALUATED, rid in kept_map.values(), "", [f"obs:vz:{rid}"], ref=rid)
            decide("filter", DS.EVALUATED, bool(kept_map), "", [ex_obs("kept_total", "count", len(kept_map))])
    n_kept = len(kept_map)

    # ---- replay
    replay_ok = False
    if n_kept == 0:
        skip("replay", "no kept DIS to replay")
    elif facts.get("last_step", "reco") == "filter":
        decide("replay", DS.NOT_EVALUATED, None, "replay not requested (last_step=filter)")
    elif not attested("replay") or replay is None:
        decide("replay", DS.TECHNICALLY_UNAVAILABLE, None, failure.get("replay", "replay not recorded as returning 0, or output missing"))
    elif sorted((m, i) for m, _, i in replay) != sorted(kept_map) or any(e != 0 for _, e, _ in replay):
        decide("replay", DS.TECHNICALLY_UNAVAILABLE, None, f"replay events {len(replay)} do not match kept DIS {n_kept}")
    else:
        replay_ok = True
        decide("replay", DS.EVALUATED, True, "every kept DIS replayed with matching lineage", [ex_obs("replayed_events", "count", len(replay))])

    # ---- reconstruction
    candidates: list[ReconstructedCandidate] = []
    reco = facts.get("reco")
    if n_kept > 0 and facts.get("last_step", "reco") in ("filter", "replay") and (replay_ok or facts.get("last_step") == "filter"):
        decide("reco", DS.NOT_EVALUATED, None, f"reconstruction not requested (last_step={facts['last_step']})")
    elif not replay_ok:
        skip("reco", "no replayed DIS")
    elif not attested("reco") or reco is None or len(reco) != len(replay):
        decide("reco", DS.TECHNICALLY_UNAVAILABLE, None, failure.get("reco", "reco not recorded as returning 0, output missing, or event count != replay events"))
    else:
        for (m, _, i), r in zip(replay, reco):
            rid = kept_map[(m, i)]
            obs.append(_obs(f"obs:fit_tracks:{rid}", rid, "pr1432:n_fit_tracks_v0", "count", r["n_fit_tracks"], f"{rid}#FitTracks", digest))
            candidates += [ReconstructedCandidate(f"cand:{rid}:{c}", ex_id, rid, c) for c in range(r["n_particles"])]
        ev = ex_obs("reco_candidates_total", "count", len(candidates))
        decide("reco", DS.EVALUATED, bool(candidates), "ShipReco ran" + ("" if candidates else " and produced no Particles"), [ev])
    bundle = EvaluationBundle(request.request_id, "pr1432_mudis_v0", is_physical, executions=(ex,), realizations=tuple(realizations),
                              candidates=tuple(candidates), observations=tuple(obs), decisions=tuple(decisions))
    verify_evaluation_bundle(bundle, request)
    return bundle


# ------------------------------------------------------------------- runner
class PR1432MuonDISRunner:
    """Run prepare -> filter -> replay -> reco for one source state (``sim_connector.root`` + geometry)."""

    name = "pr1432_mudis_v0"

    def __init__(self, fairship_dir: Path | str, *, python_executable: str = sys.executable, timeout: Optional[float] = None) -> None:
        self.fairship_dir, self.python_executable, self.timeout = Path(fairship_dir).resolve(), python_executable, timeout

    def run(self, request: EvaluationRequest, *, sim_connector: Path | str, geofile: Path | str, output_directory: Path | str,
            n_dis_per_material: int = 10, filter_option: int = 1, last_step: str = "reco") -> EvaluationBundle:
        if len(request.subjects) != 1 or request.replications_per_subject != 1:
            raise ValueError("one source state and one replication per call")
        if last_step not in STEPS[1:]:
            raise ValueError(f"last_step must be one of {STEPS[1:]}")
        if not 1 <= request.seed <= 899999999:
            raise ValueError("seed must be in [1, 899999999] (prepareEvents wraps others)")
        src, geo, out = Path(sim_connector).resolve(), Path(geofile).resolve(), Path(output_directory).resolve()
        out.mkdir(parents=True, exist_ok=True)
        fs, py, s = self.fairship_dir, self.python_executable, str(request.seed)
        prepared, selected, replay_dir = out / "prepared.root", out / "selected.root", out / "replay"
        if any(path.exists() for path in (prepared, selected, replay_dir)):  # stale outputs would be read as this run's
            raise FileExistsError(f"{out} already holds outputs of an earlier run; use a fresh output directory")
        commands = {
            "prepare": [py, str(fs / "newMuonDIS/prepareEvents.py"), "-f", str(src), "-o", str(prepared), "-n", "1", "-d", str(n_dis_per_material), "--seed", s, "-g", str(geo)],
            "filter": [py, str(fs / "newMuonDIS/filterEvents.py"), "-f", str(prepared), "-o", str(selected), "-g", str(geo), "--filter-option", str(filter_option)],
            "replay": [py, str(fs / "macro/run_simScript.py"), "--NewMuDIS", "-f", str(selected), "-i", "0", "-n", "1", "-o", str(replay_dir), "--tag", "dis",
                       "-s", s, "-r", s, "--shieldName", "TRY_2026", "--strawDesign", "10", "--noSND"],
            "reco": [py, str(fs / "macro/ShipReco.py"), "-f", str(replay_dir / "sim_dis.root"), "-g", str(replay_dir / "geo_dis.root")],
        }
        facts: dict[str, Any] = {"fairship_commit": _git_head(fs), "n_dis": n_dis_per_material, "filter_option": filter_option,
                                 "seed": request.seed, "last_step": last_step, "steps": {}, "failure": {}, "muon_p_gev": read_input_muon(src)["p_gev"],
                                 "geofile_sha256": _sha256(geo)}
        readers = {"prepare": lambda: read_muondis(prepared), "filter": lambda: read_muondis(selected),
                   "replay": lambda: read_replay(replay_dir / "sim_dis.root"), "reco": lambda: read_reco(replay_dir / "sim_dis_rec.root")}
        for step in STEPS:
            if STEPS.index(step) > STEPS.index(last_step):
                break
            if step == "replay" and not (facts.get("filter") and facts["filter"]["materials"]):
                break  # nothing kept: replay is not applicable (build_bundle records NOT_EVALUATED)
            replay_dir.mkdir(exist_ok=True)
            t0, rc = time.monotonic(), None  # rc stays None if the process itself could not run
            try:
                done = subprocess.run(commands[step], cwd=replay_dir if step == "reco" else out, capture_output=True, text=True,
                                      timeout=self.timeout, check=False)
                (out / f"{step}.log").write_text(done.stdout + "\n--stderr--\n" + done.stderr, encoding="utf-8")
                rc = done.returncode
                if rc:
                    facts["failure"][step] = f"{step} returned {rc}"
                else:
                    facts[step] = readers[step]()
                    if step == "filter":
                        found = re.search(r"saved (\d+) muon entries; skipped (\d+)", done.stdout + done.stderr)
                        facts["filter_log"] = {"saved": int(found[1]), "skipped": int(found[2])} if found else None
            except Exception as exc:  # noqa: BLE001 - any reader/subprocess error is a technical failure, never a physics result
                facts["failure"][step] = f"{step} technical failure: {type(exc).__name__}: {exc}"
            facts["steps"][step] = {"rc": rc, "wall_s": time.monotonic() - t0}
            if step in facts["failure"]:
                break
        hashes = {p.name: _sha256(p) for p in (src, geo, prepared, selected, replay_dir / "sim_dis.root") if p.is_file()}
        _write_json(out / "facts.json", {**facts, "hashes": hashes, "commands": commands})  # before building: facts outlive a builder error
        return build_bundle(request, facts)
