# Lossless P2 extraction (task 3, 2026-11-02)

FairShip `8a1cc53ec` (26.10). Code: `scripts/run_stage2_p2.py` (driver), `scripts/p2_live_smoke.py`, `adapters/fairship_p2.py:compare_with_standard`;
tests: `tests/fairship/test_run_stage2_p2.py` (4, portable), `tests/fairship/test_p2_lossless.py` (7, `local_env`; 1 of them runs FairShip for ~1 min).
Runs (all in `scratch/gen-from-scratch-v0/`, none uses `PlaneMS`): `p2_real_nofollow/` (seeds 201-203, real stage 1), `synth_p2only/{u_*,nf_*}` (12-event mechanism input, see below),
`synth_p2only/nf_native` (proposed option). Small: 3+2+2+2+1 simulations of 40-50 s.

## How the filter works (read, then measured)

- `macro/run_simScript.py:1200-1275`: after the run, with `--MuonBack`, copy only events with a non-empty `*Point` branch, excluding `CaloScoringPlanePoint` and `UpstreamTaggerScoringPoint` (`:1226`), then `rm` + `mv` over the output (`:1270-1271`). There is no option to keep empty events and no hook; the unfiltered tree is deleted.
- Native ways to keep a P2-only event: none in 26.10. A muon survives only through another sensitive volume. With `--FollowMuon` (`:1039`, `Veto.SetFollowMuon`) the cavern is sensitive: every `vetoPoint` of the synthetic runs is detID 1 in node `Cavern_1`, z 5497-9643 cm, one per muon. Without it, only the UBT, SBT, straws and timing detector can rescue an event.

## 1. Generated / kept / extracted P2 crossings

| Configuration | generated (unfiltered) | kept (standard) | lost | extracted (driver) | lost EventIDs |
|---|---|---|---|---|---|
| real stage 1, seeds 201-203, `--FollowMuon` (earlier runs) | 19 | 19 | 0 | 19 | - |
| **real stage 1, seeds 201-203, no `--FollowMuon`** | 19 | 9 | **10** | 19 | s201: 9, 14, 15; s202: 16, 36; s203: 26 |
| synthetic (12 events), `--FollowMuon` | 117 | 117 | 0 | 117 | - |
| **synthetic (12 events), no `--FollowMuon`** | 117 | 19 | **98** | 117 | 1-10 |
| synthetic, no `--FollowMuon`, proposed native option (copy of the script) | 117 | 117 | 0 | - | - |

Real-sample crossings, with trajectory and stage-1 origin (all 19 `RESOLVED`; `LOST` = absent from the standard file, present in the unfiltered one):

| seed | EventID | st2 trk | st1 entry (POT evt) | st1 mu trk | split set | clone w | x,y (cm) | p (GeV) | in standard? |
|---|---|---|---|---|---|---|---|---|---|
| 201 | 9 | 4 | 8 (5046) | 4 | - | 1 | 188, -12 | 10.43 | **LOST** (RESOLVED) |
| 201 | 10 | 32 | 9 (5110) | 25 | 0 | 0.1 | -187, -9 | 25.52 | yes (RESOLVED) |
| 201 | 10 | 31 | 9 (5110) | 24 | 0 | 0.1 | -226, 14 | 40.74 | yes (RESOLVED) |
| 201 | 10 | 30 | 9 (5110) | 23 | 0 | 0.1 | -199, -6 | 45.07 | yes (RESOLVED) |
| 201 | 10 | 29 | 9 (5110) | 22 | 0 | 0.1 | -209, -13 | 24.45 | yes (RESOLVED) |
| 201 | 10 | 28 | 9 (5110) | 21 | 0 | 0.1 | -197, -12 | 33.24 | yes (RESOLVED) |
| 201 | 10 | 27 | 9 (5110) | 20 | 0 | 0.1 | -203, -6 | 34.90 | yes (RESOLVED) |
| 201 | 10 | 26 | 9 (5110) | 18 | 0 | 0.1 | -211, 11 | 37.84 | yes (RESOLVED) |
| 201 | 10 | 25 | 9 (5110) | 17 | 0 | 0.1 | -213, -8 | 38.38 | yes (RESOLVED) |
| 201 | 10 | 24 | 9 (5110) | 16 | 0 | 0.1 | -152, -68 | 4.90 | yes (RESOLVED) |
| 201 | 14 | 8 | 13 (6073) | 18 | 0 | 0.1 | -189, -78 | 2.18 | **LOST** (RESOLVED) |
| 201 | 14 | 7 | 13 (6073) | 16 | 0 | 0.1 | -188, -7 | 14.55 | **LOST** (RESOLVED) |
| 201 | 15 | 15 | 14 (6172) | 13 | 0 | 0.1 | -213, -0 | 41.77 | **LOST** (RESOLVED) |
| 201 | 15 | 14 | 14 (6172) | 11 | 0 | 0.1 | -191, 5 | 32.88 | **LOST** (RESOLVED) |
| 201 | 15 | 13 | 14 (6172) | 10 | 0 | 0.1 | -183, 3 | 5.21 | **LOST** (RESOLVED) |
| 201 | 15 | 12 | 14 (6172) | 9 | 0 | 0.1 | -198, 52 | 13.69 | **LOST** (RESOLVED) |
| 202 | 16 | 7 | 15 (7494) | 11 | 0 | 0.1 | 179, -41 | 6.44 | **LOST** (RESOLVED) |
| 202 | 36 | 9 | 35 (18303) | 8 | - | 1 | -206, -20 | 59.41 | **LOST** (RESOLVED) |
| 203 | 26 | 7 | 25 (14766) | 18 | 0 | 0.1 | -182, -40 | 7.83 | **LOST** (RESOLVED) |
generated/kept/lost: [19, 9, 10]

Counts differ between the `--FollowMuon` and no-`--FollowMuon` real runs event by event (EventID 10 has 10 crossings in one and 9 in the other; EventID 30 has one in one and none in the other): the sensitive cavern changes the random stream. The loss is measured inside each run (unfiltered vs standard), never across runs.

## VERIFIED

- The standard `--MuonBack` output loses P2-only events: 10 of 19 real crossings (6 of 7 events) and 98 of 117 synthetic ones, without `--FollowMuon`; none with it (19/19 real, all synthetic). Mechanism: `Cavern_1` points of `--FollowMuon`.
- `scripts/run_stage2_p2.py` yields the standard file (identical to a stock run: same `(EventID, track)` keys, same p and x to 1e-9 on the 12-event `--FollowMuon` run; on the no-`--FollowMuon` run the crossing count of the standard file equals the stock one) plus `sim_*.unfiltered.root` with every crossing; every crossing keeps a unique stage-1 origin. No `PlaneMS` is involved.
- The driver does not rewrite FairShip text: it runs the script with `runpy` and replaces one `os.system("rm  <out>.root")` by a rename. Refuses any `run_simScript.py` whose sha256 is not pinned (`PINNED`, 8a1cc53) or whose filter lines changed.
- Live smoke (`test_live_smoke_...`, 52 s with its two FairShip processes): P2-only events lost in the standard file and kept in the unfiltered one, `--FollowMuon` run loses none, all provenance `RESOLVED`, P2 centre = shield end + 10.0 cm read from the produced geofile.
- Regression alarms that fail on FairShip changes: pin of `run_simScript.py` (`test_installed_run_simscript_is_pinned...`), the filter signature, the live smoke (placement, detID 2, filter behaviour, EventID join), geometry/detID/EventID tests of tasks 1-2.

## PROVISIONAL

- The synthetic input (copies of one stage-1 entry with `PlaneHAPoint` px shifted per event, MCTrack history and split flags unchanged) is a mechanism test: its momenta are not a physical sample, and nothing here estimates a physical P2-only rate. The real-sample loss rate (10/19 without `--FollowMuon`) is 3 seeds.
- Interposing on `os.system` relies on the shell command string of the filter; the pin and signature turn a change into a failure, not into silent behaviour.
- The proposed native option was run on a copy of the script placed outside the checkout with `PYTHONPATH=<checkout>/macro`.

## OPEN

- Whether campaigns should run with or without `--FollowMuon`: it changes the random stream, not P2 itself. Measured on the 12-event synthetic input (driver, two parallel runs): wall 38.4 s both, so at this size the cost is FairShip start-up, not tracking; a per-event cost needs a larger run. Not measured at campaign scale.
- Disk cost of unfiltered output: 12 events, `.unfiltered.root` 40 kB (no `--FollowMuon`) / 49 kB (`--FollowMuon`) vs standard 27 / 48 kB. Campaign scale not measured.
- Upstream acceptance of the option, and on LXPLUS: a new `run_simScript.py` hash must be read and pinned (the live smoke is the pre-flight check).

## DECISION (user, 2026-11-02)

Time cost is low and disk is not a concern for now: campaigns use the driver and keep `*.unfiltered.root` always; optimise space only if it becomes a problem.
