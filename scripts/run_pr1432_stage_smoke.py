"""Bounded real-FairShip smoke for the PR #1432 stage-outcome adapter.

Runs ``ROW:SEED`` pairs against existing injected ``sim_connector.root`` files, writes one ``bundle.json`` per
execution and a per-source ``aggregates.csv`` (the label table a later U update would consume).  Run it inside the
FairShip pixi environment.  Real output only: it refuses to aggregate a bundle that declares ``is_physical=False``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from ship_muon_bg.adapters.pr1432_mudis import PR1432MuonDISRunner, stage_ids
from ship_muon_bg.entities.subject import TagSubject
from ship_muon_bg.simulation.evaluation import EvaluationBundle, EvaluationRequest, bundle_records
from ship_muon_bg.tagging.aggregation import aggregate_state_stage_evaluations


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fairship", required=True, type=Path)
    ap.add_argument("--inputs", required=True, type=Path, help="dir holding <row>/sim_connector.root and geo_connector.root")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--runs", required=True, help="comma list ROW:SEED")
    ap.add_argument("--n-dis", type=int, default=10)
    ap.add_argument("--filter-option", type=int, default=1)
    ap.add_argument("--last-step", default="reco", choices=["filter", "replay", "reco"], help="stop the chain here; later stages are recorded NOT_EVALUATED")
    args = ap.parse_args(argv)

    runner = PR1432MuonDISRunner(args.fairship)
    options = {"n_dis_per_material": args.n_dis, "filter_option": args.filter_option, "shield": "TRY_2026", "strawDesign": 10, "last_step": args.last_step}
    bundles: list[EvaluationBundle] = []
    for pair in args.runs.split(","):
        row, seed = pair.split(":")
        request = EvaluationRequest(f"req-{row}-{seed}", (TagSubject(f"row-{row}", "post_shield_muon", "legacy_row_plus2p5916m_provisional"),),
                                    "pr1432_400c0fb_TRY_2026", int(seed), options=options)
        out = args.out / f"{row}_s{seed}"
        bundle = runner.run(request, sim_connector=args.inputs / row / "sim_connector.root", geofile=args.inputs / row / "geo_connector.root",
                            output_directory=out, n_dis_per_material=args.n_dis, filter_option=args.filter_option, last_step=args.last_step)
        (out / "bundle.json").write_text(json.dumps(bundle_records(bundle), indent=1) + "\n")
        bundles.append(bundle)
        print(f"{row}:{seed} -> {bundle.executions[0].execution_status.value}, {len(bundle.realizations)} DIS, {len(bundle.candidates)} candidates", flush=True)

    if not all(b.is_physical for b in bundles):
        raise SystemExit("refusing to report: a bundle declares is_physical=False")
    merged = EvaluationBundle("smoke", runner.name, True, **{k: tuple(x for b in bundles for x in getattr(b, k))
                                                              for k in ("executions", "realizations", "candidates", "observations", "decisions")})
    commit = merged.executions[0].provenance["fairship_commit"]
    sid = stage_ids(commit, args.n_dis, args.filter_option)
    names = {sid[k]: k for k in ("generation", "filter", "replay", "reco")}
    dataset = aggregate_state_stage_evaluations(executions=merged.executions, candidates=merged.candidates, decisions=merged.decisions,
                                                stage_definition_ids=list(names))
    dataset.require_single_configuration()
    with (args.out / "aggregates.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject_id", "stage", "stage_definition_id", "fs_sim_configuration_id", "options_digest", "n_dis_per_material", "executions", "valid", "positive", "negative", "technical_failure", "censored", "not_evaluated", "eta_hat_observed"])
        for r in dataset.rows:
            w.writerow([r.subject_id, names[r.stage_definition_id], r.stage_definition_id, r.fs_sim_configuration_id, r.options_digest, args.n_dis, r.execution_count, r.valid_count, r.positive_count, r.negative_count,
                        r.technical_failure_count, r.technically_censored_count, r.not_evaluated_count, "" if r.eta_hat is None else r.eta_hat])
    print((args.out / "aggregates.csv").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
