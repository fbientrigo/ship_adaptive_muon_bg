"""Stage-outcome semantics of the PR #1432 adapter, on synthetic facts (no ROOT, no FairShip).

Real-artifact agreement with the ROOT files is the ``local_env`` test at the bottom.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ship_muon_bg.adapters.pr1432_mudis import build_bundle, stage_ids
from ship_muon_bg.entities.decision import DecisionEvaluationStatus as DS
from ship_muon_bg.entities.subject import TagSubject
from ship_muon_bg.simulation.evaluation import EvaluationBundle, EvaluationRequest, bundle_records
from ship_muon_bg.tagging.aggregation import ExecutionStageOutcome as O
from ship_muon_bg.tagging.aggregation import aggregate_state_stage_evaluations

COMMIT = "400c0fb"


def request(seed: int = 11, subject: str = "src-1", options: dict | None = None, config: str = "cfg") -> EvaluationRequest:
    return EvaluationRequest(f"req-{subject}-{seed}", (TagSubject(subject, "post_shield_muon", "state_v0"),), config, seed,
                             options=options or {"n_dis": 4})


def vertices(n: int, base: float = 100.0) -> list:
    return [(1.0 * i, 2.0, base + i, 0.5) for i in range(n)]


def material(n: int, base: float = 100.0) -> dict:
    return {"path_length_cm": 20.0, "n_dis": n, "n_generated": n, "wDIS_native": 1.5, "vertices": vertices(n, base)}


def facts(*, n_ms=4, n_sbt=2, kept_ms=(0, 2), kept_sbt=(), replay="auto", reco="auto", failure=None, p=10.0, prepare="auto", n_dis=4, seed=11, **extra) -> dict:
    prep = {"materials": {"MS": material(n_ms), "SBTsens": material(n_sbt, 500.0)}} if prepare == "auto" else prepare
    filt_ms = [vertices(n_ms)[i] for i in kept_ms]
    filt_sbt = [vertices(n_sbt, 500.0)[i] for i in kept_sbt]
    filt = {"materials": {}}
    if filt_ms:
        filt["materials"]["MS"] = {**material(len(filt_ms)), "vertices": filt_ms}
    if filt_sbt:
        filt["materials"]["SBTsens"] = {**material(len(filt_sbt), 500.0), "vertices": filt_sbt}
    kept = [("MS", 0, i) for i in range(len(filt_ms))] + [("SBTsens", 0, i) for i in range(len(filt_sbt))]
    last = extra.get("last_step", "reco")
    steps = {"prepare": {"rc": 0, "wall_s": 5.0}, "filter": {"rc": 0, "wall_s": 12.0}}
    if kept and last != "filter" and replay is not None:
        steps["replay"] = {"rc": 0, "wall_s": 40.0}
        if last == "reco" and reco is not None:
            steps["reco"] = {"rc": 0, "wall_s": 15.0}
    for step in failure or {}:
        steps[step] = {"rc": 1, "wall_s": 1.0}
    return {"fairship_commit": COMMIT, "n_dis": n_dis, "filter_option": 1, "seed": seed, "muon_p_gev": p, "last_step": "reco",
            "steps": steps, "filter_log": {"saved": 1 if kept else 0, "skipped": 0},
            "prepare": prep, "filter": filt if kept else None,
            "replay": kept if replay == "auto" else replay,
            "reco": [{"n_fit_tracks": 1, "n_particles": 0} for _ in kept] if reco == "auto" else reco,
            "failure": failure or {}, **extra}


def status(bundle: EvaluationBundle, key: str, request_: EvaluationRequest | None = None):
    sid = stage_ids(COMMIT, 4, 1)[key]
    execution_level = {item.execution_id for item in bundle.executions}
    found = [d for d in bundle.decisions if d.stage_definition_id == sid and d.subject_ref in execution_level]
    assert len(found) == 1, (key, found)
    return found[0].evaluation_status, found[0].decision


def test_one_source_many_realizations_with_lineage() -> None:
    bundle = build_bundle(request(), facts())
    assert len(bundle.realizations) == 6 and len({r.realization_id for r in bundle.realizations}) == 6
    assert status(bundle, "generation") == (DS.EVALUATED, True)
    assert status(bundle, "filter") == (DS.EVALUATED, True)
    assert status(bundle, "replay") == (DS.EVALUATED, True)
    per_dis = {d.subject_ref: d.decision for d in bundle.decisions if d.stage_definition_id == stage_ids(COMMIT, 4, 1)["filter_dis"]}
    assert len(per_dis) == 6 and sum(per_dis.values()) == 2  # kept MS 0 and 2 only
    assert all(r.realization_id in per_dis for r in bundle.realizations)


def test_zero_one_and_many_candidates_are_preserved() -> None:
    zero = build_bundle(request(), facts())
    assert status(zero, "reco") == (DS.EVALUATED, False) and not zero.candidates  # ran, produced nothing: a real zero
    many = build_bundle(request(), facts(reco=[{"n_fit_tracks": 2, "n_particles": 1}, {"n_fit_tracks": 2, "n_particles": 3}]))
    assert status(many, "reco") == (DS.EVALUATED, True) and len(many.candidates) == 4
    assert {c.realization_id for c in many.candidates} == {r.realization_id for r in many.realizations if r.realization_id.endswith(("MS:0", "MS:2"))}
    assert sorted(c.candidate_index for c in many.candidates if c.realization_id.endswith("MS:2")) == [0, 1, 2]
    one = build_bundle(request(), facts(reco=[{"n_fit_tracks": 2, "n_particles": 1}, {"n_fit_tracks": 0, "n_particles": 0}]))
    assert len(one.candidates) == 1


def test_empty_filter_is_a_physics_rejection_and_downstream_not_evaluated() -> None:
    bundle = build_bundle(request(), facts(kept_ms=()))
    assert status(bundle, "filter") == (DS.EVALUATED, False)
    assert status(bundle, "replay")[0] is DS.NOT_EVALUATED and status(bundle, "reco")[0] is DS.NOT_EVALUATED
    assert not bundle.candidates


def test_generator_support_skip_is_not_evaluated_not_negative() -> None:
    bundle = build_bundle(request(), facts(prepare=None, p=1.2))
    assert status(bundle, "generation")[0] is DS.NOT_EVALUATED
    assert all(status(bundle, k)[0] is DS.NOT_EVALUATED for k in ("filter", "replay", "reco"))
    unexplained = build_bundle(request(), facts(prepare=None, p=9.0))
    assert status(unexplained, "generation")[0] is DS.TECHNICALLY_UNAVAILABLE  # above the cut: cannot be called physics
    assert status(unexplained, "filter")[0] is DS.TECHNICALLY_UNAVAILABLE


def test_no_target_path_is_evaluated_false_generation() -> None:
    bundle = build_bundle(request(), facts(prepare={"materials": {}}))
    assert status(bundle, "generation") == (DS.EVALUATED, False) and not bundle.realizations


def test_prepare_failure_is_execution_technical_failure_without_physics_records() -> None:
    bundle = build_bundle(request(), facts(failure={"prepare": "prepare returned 1"}, prepare=None))
    assert bundle.executions[0].execution_status.value == "technical_failure"
    assert not bundle.realizations and not bundle.candidates and not bundle.decisions


def test_replay_crash_keeps_upstream_and_marks_downstream_unavailable() -> None:
    bundle = build_bundle(request(), facts(replay=None, reco=None, failure={"replay": "replay returned 139"}))
    assert bundle.executions[0].execution_status.value == "succeeded" and len(bundle.realizations) == 6
    assert status(bundle, "filter") == (DS.EVALUATED, True)
    assert status(bundle, "replay")[0] is DS.TECHNICALLY_UNAVAILABLE
    assert status(bundle, "reco")[0] is DS.TECHNICALLY_UNAVAILABLE  # a crash is never "no candidates"


def test_reco_not_requested_or_failed() -> None:
    assert status(build_bundle(request(), facts(last_step="replay", reco=None)), "reco")[0] is DS.NOT_EVALUATED
    stopped = build_bundle(request(), facts(last_step="filter", replay=None, reco=None))
    assert status(stopped, "filter") == (DS.EVALUATED, True)  # a not-requested stage is not a failure and not a negative
    assert status(stopped, "replay")[0] is DS.NOT_EVALUATED and status(stopped, "reco")[0] is DS.NOT_EVALUATED
    failed = build_bundle(request(), facts(reco=None, failure={"reco": "reco returned 1"}))
    assert status(failed, "reco")[0] is DS.TECHNICALLY_UNAVAILABLE and status(failed, "replay") == (DS.EVALUATED, True)


def test_lineage_mismatch_is_unavailable_never_false() -> None:
    f = facts()
    f["filter"]["materials"]["MS"]["vertices"][0] = (9.0, 9.0, 9.0, 9.0)  # a kept vertex that was never generated
    bundle = build_bundle(request(), f)
    assert status(bundle, "filter")[0] is DS.TECHNICALLY_UNAVAILABLE
    assert status(bundle, "replay")[0] is DS.TECHNICALLY_UNAVAILABLE and status(bundle, "reco")[0] is DS.TECHNICALLY_UNAVAILABLE
    assert not any(d.stage_definition_id == stage_ids(COMMIT, 4, 1)["filter_dis"] for d in bundle.decisions)
    duplicate = facts()
    duplicate["prepare"]["materials"]["MS"]["vertices"][1] = duplicate["prepare"]["materials"]["MS"]["vertices"][0]
    assert status(build_bundle(request(), duplicate), "filter")[0] is DS.TECHNICALLY_UNAVAILABLE  # ambiguous join
    swapped = facts(kept_ms=(2, 0))  # order must be preserved by the filter
    assert status(build_bundle(request(), swapped), "filter")[0] is DS.TECHNICALLY_UNAVAILABLE


def test_replay_count_or_index_mismatch_is_unavailable() -> None:
    assert status(build_bundle(request(), facts(replay=[("MS", 0, 0)])), "replay")[0] is DS.TECHNICALLY_UNAVAILABLE
    assert status(build_bundle(request(), facts(replay=[("MS", 0, 0), ("MS", 0, 5)])), "replay")[0] is DS.TECHNICALLY_UNAVAILABLE
    assert status(build_bundle(request(), facts(replay=[("MS", 1, 0), ("MS", 1, 1)])), "replay")[0] is DS.TECHNICALLY_UNAVAILABLE


def test_trial_budget_and_commit_change_the_stage_ids() -> None:
    assert stage_ids(COMMIT, 4, 1) != stage_ids(COMMIT, 10, 1) != stage_ids("other", 10, 1)
    assert stage_ids(COMMIT, 4, 1)["filter"] != stage_ids(COMMIT, 4, 0)["filter"]
    b4, b10 = build_bundle(request(seed=1), facts(seed=1)), build_bundle(request(seed=2), facts(seed=2, n_dis=10))
    merged = EvaluationBundle("m", "pr1432_mudis_v0", True, executions=b4.executions + b10.executions, realizations=b4.realizations + b10.realizations,
                              observations=b4.observations + b10.observations, decisions=b4.decisions + b10.decisions)
    stages = [stage_ids(COMMIT, 4, 1)["filter"], stage_ids(COMMIT, 10, 1)["filter"]]
    ds = aggregate_state_stage_evaluations(executions=merged.executions, candidates=merged.candidates, decisions=merged.decisions, stage_definition_ids=stages)
    # the other budget's run is silent for this stage id -> NOT_EVALUATED, never pooled into valid_count
    assert {(row.stage_definition_id, row.valid_count, row.not_evaluated_count) for row in ds.rows} == {(stages[0], 1, 1), (stages[1], 1, 1)}


def test_repeats_are_conditional_one_source_row_two_executions_and_failures_excluded() -> None:
    bundles = [build_bundle(request(seed=1), facts(seed=1)), build_bundle(request(seed=2), facts(seed=2, kept_ms=())),
               build_bundle(request(seed=3), facts(seed=3, failure={"prepare": "boom"}, prepare=None))]
    merged = EvaluationBundle("m", "pr1432_mudis_v0", True, executions=tuple(e for b in bundles for e in b.executions),
                              realizations=tuple(r for b in bundles for r in b.realizations), candidates=tuple(c for b in bundles for c in b.candidates),
                              observations=tuple(o for b in bundles for o in b.observations), decisions=tuple(d for b in bundles for d in b.decisions))
    stage = stage_ids(COMMIT, 4, 1)["filter"]
    ds = aggregate_state_stage_evaluations(executions=merged.executions, candidates=merged.candidates, decisions=merged.decisions, stage_definition_ids=[stage])
    (row,) = ds.rows
    assert (row.execution_count, row.valid_count, row.positive_count, row.negative_count, row.technical_failure_count) == (3, 2, 1, 1, 1)
    assert row.eta_hat == 0.5  # the crashed run is not a negative in the denominator
    assert len({e.subject_id for e in merged.executions}) == 1


def test_repeated_seed_collides_instead_of_faking_an_independent_repeat() -> None:
    a, b = build_bundle(request(seed=5), facts(seed=5)), build_bundle(request(seed=5), facts(seed=5))
    assert a.executions[0].execution_id == b.executions[0].execution_id
    with pytest.raises(ValueError):
        EvaluationBundle("m", "x", True, executions=a.executions + b.executions)
    assert build_bundle(request(seed=6), facts(seed=6)).executions[0].execution_id != a.executions[0].execution_id


def test_incompatible_options_under_one_config_do_not_pool() -> None:
    b1 = build_bundle(request(seed=1, options={"geo": "a"}), facts(seed=1))
    b2 = build_bundle(request(seed=2, options={"geo": "b"}), facts(seed=2))
    merged = EvaluationBundle("m", "x", True, executions=b1.executions + b2.executions, decisions=b1.decisions + b2.decisions)
    ds = aggregate_state_stage_evaluations(executions=merged.executions, candidates=(), decisions=merged.decisions, stage_definition_ids=[stage_ids(COMMIT, 4, 1)["filter"]])
    assert len(ds.rows) == 2
    with pytest.raises(ValueError):
        ds.require_single_configuration()


def test_native_wdis_is_not_a_weight_and_cost_is_recorded() -> None:
    bundle = build_bundle(request(), facts())
    names = {o.observation_definition_id for o in bundle.observations}
    assert any("wDIS_native_not_event_weight" in n for n in names) and not any("weight" in n.replace("not_event_weight", "") for n in names)
    total = next(o for o in bundle.observations if o.observation_definition_id.endswith("wall_seconds_total_v0"))
    assert total.payload.value == pytest.approx(72.0)  # prepare 5 + filter 12 + replay 40 + reco 15
    assert bundle.executions[0].provenance["fairship_commit"] == COMMIT and bundle.executions[0].seed == 11


def test_records_roundtrip_to_json_and_mock_flag_is_carried() -> None:
    bundle = build_bundle(request(), facts(), is_physical=False)
    assert bundle.is_physical is False
    dumped = json.loads(json.dumps(bundle_records(bundle)))
    assert len(dumped["decisions"]) == len(bundle.decisions) and len(dumped["observations"]) == len(bundle.observations)


def test_runner_refuses_an_output_directory_with_stale_outputs(tmp_path: Path) -> None:
    from ship_muon_bg.adapters.pr1432_mudis import PR1432MuonDISRunner

    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "selected.root").write_bytes(b"stale")
    with pytest.raises(FileExistsError):
        PR1432MuonDISRunner(tmp_path).run(request(), sim_connector=tmp_path, geofile=tmp_path, output_directory=tmp_path / "out")


def test_a_stage_is_concluded_only_from_a_step_recorded_as_returning_zero() -> None:
    no_filter_step = facts()
    del no_filter_step["steps"]["filter"]
    crashed_silently = facts()
    crashed_silently["steps"]["filter"]["rc"] = 139  # non-zero rc, but nobody wrote a failure entry
    for f in (no_filter_step, crashed_silently):
        bundle = build_bundle(request(), f)
        assert status(bundle, "filter")[0] is DS.TECHNICALLY_UNAVAILABLE and status(bundle, "reco")[0] is DS.TECHNICALLY_UNAVAILABLE
        assert not any(d.stage_definition_id == stage_ids(COMMIT, 4, 1)["filter_dis"] for d in bundle.decisions)
    reco_missing = facts()
    del reco_missing["steps"]["reco"]
    assert status(build_bundle(request(), reco_missing), "reco")[0] is DS.TECHNICALLY_UNAVAILABLE
    no_prepare = facts()
    del no_prepare["steps"]["prepare"]
    assert build_bundle(request(), no_prepare).executions[0].execution_status.value == "technical_failure"
    stage = stage_ids(COMMIT, 4, 1)["filter"]
    ok, bad = build_bundle(request(seed=1), facts(seed=1)), build_bundle(request(seed=2), {**crashed_silently, "seed": 2})
    ds = aggregate_state_stage_evaluations(executions=ok.executions + bad.executions, candidates=(), decisions=ok.decisions + bad.decisions, stage_definition_ids=[stage])
    (row,) = ds.rows
    assert (row.negative_count, row.technically_censored_count, row.eta_hat) == (0, 1, 1.0)  # not a negative, not in the denominator


def test_filter_that_skipped_the_entry_does_not_attest_a_zero() -> None:
    for log in ({"saved": 0, "skipped": 1}, None, {"saved": 1, "skipped": 0}):  # malformed-skip, no log line, log/tree disagree
        bundle = build_bundle(request(), facts(kept_ms=(), filter_log=log))
        assert status(bundle, "filter")[0] is DS.TECHNICALLY_UNAVAILABLE, log
    assert status(build_bundle(request(), facts(kept_ms=())), "filter") == (DS.EVALUATED, False)  # clean zero stays a physics rejection


def test_recorded_seed_must_match_the_request() -> None:
    with pytest.raises(ValueError):
        build_bundle(request(seed=2), facts(seed=1))


def test_facts_survive_a_json_round_trip() -> None:
    f = facts()
    again = build_bundle(request(), json.loads(json.dumps(f)))
    assert [d.evaluation_status for d in again.decisions] == [d.evaluation_status for d in build_bundle(request(), f).decisions]


def test_execution_ids_do_not_collide_on_delimiters() -> None:
    a = build_bundle(request(subject="a:b", config="c"), facts())
    b = build_bundle(request(subject="a", config="b:c"), facts())
    assert a.executions[0].execution_id != b.executions[0].execution_id


def test_runner_turns_reader_errors_into_technical_failure_and_keeps_upstream(tmp_path: Path, monkeypatch) -> None:
    import ship_muon_bg.adapters.pr1432_mudis as mod

    (tmp_path / "geo.root").write_bytes(b"geo")
    ok = type("R", (), {"returncode": 0, "stdout": "MuDISFilter: saved 1 muon entries; skipped 0 unreadable or malformed entries.", "stderr": ""})()
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: ok)
    monkeypatch.setattr(mod, "_git_head", lambda path: "testcommit")
    monkeypatch.setattr(mod, "read_input_muon", lambda path: {"p_gev": 10.0, "w": 1.0})
    monkeypatch.setattr(mod, "read_muondis", lambda path: {"materials": {"MS": material(2)}} if path.name == "prepared.root" else (_ for _ in ()).throw(IndexError("truncated vertex vector")))
    bundle = mod.PR1432MuonDISRunner(tmp_path).run(request(), sim_connector=tmp_path / "geo.root", geofile=tmp_path / "geo.root", output_directory=tmp_path / "out")
    sid = stage_ids("testcommit", 10, 1)
    by_stage = {d.stage_definition_id: d for d in bundle.decisions if d.subject_ref == bundle.executions[0].execution_id}
    assert len(bundle.realizations) == 2 and by_stage[sid["generation"]].decision is True
    assert by_stage[sid["filter"]].evaluation_status is DS.TECHNICALLY_UNAVAILABLE and by_stage[sid["reco"]].evaluation_status is DS.TECHNICALLY_UNAVAILABLE
    saved = json.loads((tmp_path / "out" / "facts.json").read_text())
    assert "IndexError" in saved["failure"]["filter"] and saved["geofile_sha256"]


# ---- original-artifact agreement (needs ROOT + the 2026-10-12 chain outputs) ----------------------------------------
CHAIN = Path("/home/fabian/thesis/scratch/fairship-update-1012/pr1432_chain")


@pytest.mark.local_env
@pytest.mark.parametrize("row,generated,kept,replayed", [("25740", 60, 7, 7), ("32029", 60, 2, 2), ("17005", 30, 0, None)])
def test_readers_agree_with_the_original_root_files(row: str, generated: int, kept: int, replayed) -> None:
    pytest.importorskip("ROOT")
    from ship_muon_bg.adapters.pr1432_mudis import read_muondis, read_replay

    base = CHAIN / row
    if not (base / "prepared.root").is_file():
        pytest.skip("1012 chain artifacts not present")
    prepared, selected = read_muondis(base / "prepared.root"), read_muondis(base / "selected_opt1.root")
    assert sum(m["n_dis"] for m in prepared["materials"].values()) == generated
    assert sum(m["n_dis"] for m in selected["materials"].values()) == kept if selected else kept == 0
    if replayed is not None:
        assert len(read_replay(base / "replay_opt1" / "sim_dis.root")) == replayed
