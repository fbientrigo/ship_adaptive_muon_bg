# Historical dataset scope correction — decision record v0

Date: 2026-10-12. Status: `PROJECT DECISION` (the user's scope correction, recorded here). Other labels in this note: VERIFIED, PROVISIONAL, OPEN. Documentation only; no code, FairShip or tracked file was changed to write it.
Scientific authority: `SHIP-EMPIRICAL-FS-SIM-CONTRACT-V0-SOURCE`, cited by ID, **not inspected locally**. This note adds no clause to it.
Refines (does not replace) `canonical_post_ms_source_state.md`, which already says the same for the coordinate question (lines 8-10, 25-27).

## PROJECT DECISION

1. **Martina's `muonsFullMC_afterMS.pkl` / `muons_FullMC.pkl` are a historical reference and feasibility dataset.** Their one role: a reference distribution to show that a generative model can learn muon states resembling a supplied sample.
2. They are **not**: the operational source distribution; an authoritative `P0`; the normalization of new FairShip simulations; a required input of the iterative loop; a dataset whose historical weights must be recovered before work proceeds.
3. **Open questions about those weights are non-blocking.** They are kept, not deleted: the audit, the classes, the match to `extractMuonsAndUpdateWeight.py:15-23` and the OPEN items (`muon_weight_provenance.md`, label `HISTORICAL / NON-BLOCKING FOR CURRENT PIPELINE`). No further historical-provenance work is planned.
4. **Historical weights are never substituted for the weights of new simulations.** The statistical meaning of weights is set for the declared current FairShip configuration (see "What the current configuration must declare").
5. The thesis aim is a working, iterative, utility-guided pipeline on a declared, reproducible FairShip configuration, measured as **useful candidates per unit simulation cost against compatible baselines**. It does **not** estimate the physical background rate. A physical-rate estimator is a possible later stage, not a prerequisite.

```
P0(S) -> U(S) -> PU(S) -> Q_theta(S) -> FairShip/GEANT4 -> Y -> U_updated(S)
```

## The five things that must not be conflated

| Item | Role | Status |
|---|---|---|
| Martina's historical dataset | reference for generative-model feasibility; legacy evidence | PROJECT DECISION: non-blocking |
| Operational source distribution | states captured from the **current** FairShip configuration at a geometry-derived `Sigma_postMS` (#43) | not yet generated (OPEN, the main gap) |
| Utility-guided generation | `U`, `PU ∝ P0·h(U)`, `Q_theta`; `h` is a tilt, never a physical weight | pilot only (U0 geometric proxy, circular) |
| Downstream simulation and empirical evaluation | FairShip/GEANT4 is the oracle; intermediate `Y^(k)` (SBT eligibility) is not `B` | VERIFIED at pilot scale |
| Physical-rate estimation | needs σ, exposure, a source measure; **out of scope** | not attempted |

## What the current configuration must declare (replaces "recover Martina's weights")

For the configuration that generates `P0`, record: the generated source measure; any sampling or simulation bias applied; the statistical sampling unit (the repository's candidate: the input FairShip entry, `source_entry_key` in `source_bundle_architecture_v0.md`); what is an original draw versus a repeated realization (repeats are conditional, `canonical_post_ms_source_state.md:152-153`); what `wDIS` is (`weight_ledger_v0.md`: not a complete event weight); source weight, proposal probability `q_i` and utility multiplier as separate fields; independent configurations and technical failures recorded apart (a failure is never a physics negative). Status: PROVISIONAL (names exist in `weight_ledger_v0.md`; `q_i` has no field in the connector yet).

## Audit: where the old framing survives

Actual current requirements are marked CURRENT; outdated wording is marked STALE. Nothing below was edited except the scoped banners in row 6 and the banner on `muon_weight_provenance.md`.

| # | Location | What it says | Verdict | Minimal correction (proposal, not applied) |
|---|---|---|---|---|
| 1 | `docs/1011_U0_DirectedPilot/README.md:10-11`, `:47` (§5), `:73` (§9) | Pilot draws `P0` from the 40k legacy sample; `w` "OPEN", used as a PROVISIONAL nominal weight; asks Oliver/Maksym what `w` means | STALE as a gate. The pilot is a legitimate **mechanism test on legacy states** (CURRENT), but its `P0` is a stand-in | One line under the title: "P0 here = legacy reference sample, not the operational P0; `w` question non-blocking (see `historical_dataset_scope_correction_v0.md`)" |
| 2 | `docs/1012_FairShipUpdate/README.md:10` and `:144-152` (§6 Q2) | "The meaning of the weight `w` (OPEN)" listed as one of two open issues "for the next step"; Q2 asks what `w` is in the pkl | STALE for Q2. Q1 (is the post-shield input region the one the new chain expects) is CURRENT but should be restated for the **canonical** `Sigma_postMS` state | Mark Q2 as optional; restate Q1 for the canonical state when it exists |
| 3 | `ship_adaptive_muon_bg/README.md:17`, `:28` | Goal and workflow start from "post-muon-shield muon datasets / sample" | AMBIGUOUS: reads as the pkl being the source | Reword to "post-muon-shield source states (canonical: #43; historical `.pkl` = reference)" |
| 4 | `ship_adaptive_muon_bg/AGENTS.md:19-88` (D9.5 Gate D nightly GPU training on the afterMS arena) | An unattended multi-block campaign trains model families on the pkl | AMBIGUOUS. It is feasibility work (CURRENT as such) but its heading reads as production | One line: "Gate D = feasibility on the historical reference; not a prerequisite of the FairShip loop". Whether to continue the campaign is the user's call |
| 5 | `phd-thesis/thesis_material_audit/dataset_registry.json:24`, `:112` (the two Martina pkl entries) | `weight_semantics: physical_cross_section`; `status: BLOCKED` | `BLOCKED` is per-use (`blocked_use`, e.g. `:31`) and already fits the decision. The `physical_cross_section` label on these two entries has no source (`muon_weight_provenance.md` §2, UNSUPPORTED bullet). The GAN entries (`:68`, `:156`, `generated_weights`) and `sc_v6_merged` (`:199`, ROOT `CrossSection` column, not audited) are outside this finding | None now: registry semantics are not changed in this step. Recorded for a later, separate decision |
| 6 | `docs/contracts/weight_ledger_v0.md`, `docs/decisions/weight_contract_amendment_v0.md` | Mix historical content with current rules | Split: the Martina-specific `w_source` classes, lineage, HA-exit normalization, `N_spills` and amendment clauses 6–7 are HISTORICAL; the `w_source`/`w_transport`/`wDIS`/`nDISevts`/`q_i`/`h(U)` separation and clauses 1–5 are CURRENT (PROVISIONAL), and line 31 above relies on them | Scoped banner added to each (below) |
| 7 | `docs/decisions/canonical_post_ms_source_state.md:9-10`, `:25-27` | Already: canonical state independent of the historical pkl; pkl = legacy | CURRENT, consistent. File has pre-existing uncommitted edits | Optional one-line link to this note (not made) |
| 8 | `docs/contracts/fairship_empirical_endpoint_v0.md` | `Y^(k)` = SBT eligibility, `Y^(k) != B` | CURRENT, correctly scoped. But it is an SBT-hit label, not a Muon DIS outcome | See B7 below |
| 9 | Plans requiring a rate estimator before benchmarking | none found in `README.md:62`, the benchmark contract, or the endpoint contract; all say no `B` | CURRENT, no mistake found | none |

Not found: any document that states recovering Martina's normalization as a precondition. The inheritance was by omission (the OPEN `w` item travelled into the 1011/1012 questions and the registry `BLOCKED` flag), not by an explicit requirement.

## What remains genuinely blocking for the FairShip loop

| Question | Class | Evidence |
|---|---|---|
| B1. Geometry-derived `Sigma_postMS` and capture from a full current-FairShip event (#43). No capture code exists in `src/`, `scripts/` or `tests/` (grep for `Sigma_postMS`: no hit); `exitHadronAbsorber` is a candidate, adequacy unverified | OPEN, **blocking** | `canonical_post_ms_source_state.md` OPEN 1 |
| B2. Source-state schema and support: single muon vs interaction bundle; whether cutting at the plane loses upstream context | OPEN, **blocking** | same file OPEN 2-3; `source_bundle_architecture_v0.md` (design only) |
| B3. Exact reinjection round trip, with a frozen `roundtrip_validation_plan_id` before acceptance data are inspected | OPEN, **blocking** | same file OPEN 5 |
| B4. Which configuration is the declared one. Evidence: runs exist on `f73a305` (TRY_2026 geometry, not shown equal to the SHiP "24.09" geometry) and on main `8df35bc`. Main has only the legacy DIS path (`makeMuonDIS.py`, `run_simScript.py:767`); the new prepare/filter/replay chain exists only in unmerged PR #1432 @`400c0fb` | OPEN. Proposed default (not decided): main `8df35bc` + PR #1432 @`400c0fb` for the DIS chain; pin recorded per run. Unresolved conflict: `canonical_post_ms_source_state.md:178` still pins the #43 geometry work to `f73a305` | `1012_FairShipUpdate/README.md` §2-3 |
| B5. Whether the canonical state fits the input region of the MS-DIS path (the PR caps MS DIS at 20 cm of accumulated material; README asks for input "before the last interaction length" of the shield) | OPEN, **blocking for the MS-DIS part** | `1012` README §3 |
| B6. Repeated-DIS lineage: each DIS realization maps back to its source state and entry key through prepare, filter, replay | PROVISIONAL; one-muon lineage verified only for single runs | `docs/0808_MuonDIS_Compatibility/lineage.md`; `1012` README §3 |
| B7. A Muon-DIS-level intermediate outcome per source state (today only `Y^(k)`, an SBT-hit label). The relation between `Y^(k)` and the PR #1432 filter outcome is **unproven**: on the 3 tested states they agree (25740: 7 kept, `Y=1`; 32029: 2 kept, `Y=1`; 17005: 30 candidates before the filter, 0 kept, `Y=0`), which neither shows nor excludes divergence | OPEN, needs a project decision; `Y^(k)` must not be used as a substitute for the DIS objective | `1012` README §3 |

Safe to leave unresolved for now: meaning of `w`, the `w = 1.0` class, the /100 classes, Martina lineage, afterMS exposure, PoT per spill, `σ_DIS` reliability (needed only for a rate), the 7.5 m x/y limit, Muons & Matter equivalence, pile-up (Level 2).

## Limits of this note

- Evidence is the repository, the 1011/1012 write-ups and the earlier provenance audit. No FairShip run, no data download and no model training was done for it.
- The audit table is a text search over `*.md`, `AGENTS.md` and the registry JSON; it can miss wording that avoids the searched terms.
- Its findings are PROJECT DECISION where the user decided, otherwise as labelled. Codex review #1 (`vault/reviews/historical_dataset_scope_correction_v0.1.md`) failed; rows 4–6, 8, B4, B7 and both banners were corrected in response.
