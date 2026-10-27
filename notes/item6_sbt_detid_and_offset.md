# Queue item 6: SBT detID range and the +2.5916 m offset

Checked 2026-10-27, read-only. FairShip pins: `f73a305` (geometry file used), `8df35bc`, PR #1432 `400c0fb`.
Labels: OBSERVED (read from code/data), INFERENCE, OPEN.

## A. SBT predicate `1000 < detID < 999999 AND |pdg| == 13 AND p > 3 GeV`

| Claim | Label | Evidence |
|---|---|---|
| Identical predicate in all three pinned commits | OBSERVED | `muonDIS/make_nTuple_SBT.py` line 251 (`f73a305`), 250 (`8df35bc`, `400c0fb`); `P_threshold = 3` line 193 |
| vetoPoint detID = `veto::liscId` = `shape*100000 + block*10000 + Zlayer*100 + number*10 + position`; shapes 1..6; unknown shape returns sentinel `999999` | OBSERVED | `veto/veto.cxx` (`400c0fb`) `liscId`, ~l.338-361; vetoPoint filled with `CurrentVolID` at l.819-830 |
| So the upper bound `< 999999` rejects the invalid-shape sentinel; valid ids are 100000..699999 | INFERENCE from the two rows above | same |
| All 75 distinct detIDs in our TRY_2026 audit data lie in 210111..521022, i.e. shapes 2, 4, 5 | OBSERVED | `artifacts/fairship_tgeo_audit_v0/detector_observations.csv` (1020 hits, 12 NF candidates + 2 controls) |
| 62 hits qualify, all `pdg -13`/13 and p>3 GeV | OBSERVED | same csv, column `sbt_qualifying` |
| Lower bound `1000`: ids below it are not LiSc ids; no observed hit is near it | INFERENCE | observed min is 210111 |

Verdict: the range is not SBT geometry "ranges"; it is "valid LiSc id". It holds for the pinned commits and our data.
OPEN: shapes 1, 3, 6 never appeared in our hits (small sample), so their ids are only read from code, not seen.

## B. Offset `z + 2.5916 m` (sample plane 28.905 m)

| Claim | Label | Evidence |
|---|---|---|
| Sample z lies in 28.895..28.905 m (not a single value) | OBSERVED | `data/samples/muonsFullMC_afterMS_sample.npz`, column 5 |
| In the TRY_2026 geometry of `f73a305` (geometry_id `7588f2e6...`), Magn7 iron spans z = 27.93..31.93 m | OBSERVED | TGeo read of `artifacts/fairship_connector_v0/p3_row_25740/geo_connector.root`; script `~/thesis/scratch/item6/list_magn_z.py` (units cm: 2793..3193) |
| 28.905 + 2.5916 = 31.4966 m, i.e. 0.433 m before the Magn7 exit at 31.93 m, still inside the iron z range | OBSERVED (arithmetic) | the two rows above |
| The transform name says "shield exit plus 1 cm" = 31.94 m, which would need +3.035 m, not +2.5916 m | OBSERVED (name vs number) | `adapters/fairship.py` `LEGACY_AFTERMS_TRANSFORM` |
| Rows 25740 / 32029 start in `Magn7_MiddleMag{R,L}` iron | OBSERVED | `origin/chore/publish-validation-artifacts` manifest (`initial_volume`) |
| The offset is not derived anywhere in the repo | OBSERVED | only `docs/decisions/canonical_post_ms_source_state.md:27` calls it "legacy/provisional" |
| The transform is not independently anchored | OBSERVED | same manifest: `common_frame_anchor=false`, decision `BLOCKED` |

Verdict: **NOT verified, and the numbers disagree with the transform's own name.** Do not claim physical validity.
OPEN (needs the source of the offset, not another run): what plane does 28.905 m refer to in the original MS frame,
and what is the intended FairShip plane? +3.035 m is only an arithmetic reading of the name, not a result.
Impact: the eligibility gate in `minimal_loop.py` (cave/Cavern only) hides rows that start in iron; it does not
fix a wrong plane (its own docstring says so, B5).
