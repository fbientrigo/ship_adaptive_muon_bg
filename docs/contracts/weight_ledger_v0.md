# Weight ledger v0

> **Scope (project decision 2026-10-12, `docs/decisions/historical_dataset_scope_correction_v0.md`).** HISTORICAL / NON-BLOCKING: the Martina-specific content only — the `w_source` value classes and their meaning, the lineage, the 1.80178·10¹⁰ HA-exit normalization, and the `N_spills` exposure. CURRENT PIPELINE (PROVISIONAL): the separation of `w_source`, `w_transport`, `wDIS`, `nDISevts`, `q_i` and `h(U)`. For new simulations, `w_source` is whatever the declared configuration records. Findings and OPEN items below are unchanged.

Status: `PROVISIONAL`. It records names and boundaries only. It creates no new scientific contract. The authoritative contract is `SHIP-EMPIRICAL-FS-SIM-CONTRACT-V0-SOURCE`, cited by ID and **not inspected locally**. Evidence: `docs/investigations/muon_weight_provenance.md` (called PROV below).

No field called just "weight" may stand for several of these. Each row is a separate quantity.

| Name | Origin | Unit | Statistical meaning | Enters at | Must not |
|---|---|---|---|---|---|
| `w_source` (pkl column `w`; code field `physical_source_weight`) | Empirical dataset, column 7 of `[px,py,pz,x,y,z,id,w]`. Value classes match FairShip `extractMuonsAndUpdateWeight.py` constants (PROV §3–4). Stored `dimensionless`, with the original README unverified | dimensionless (as stored); physical unit **OPEN** | Per-row importance of the production, not verified as per spill, per event or per proton interaction (PROV §5 #10–11). A historical HA-exit weight sum of 1.80178·10¹⁰ per nominal spill exists for the 2018 production (note p.14); its relation to this dataset is OPEN | Proposal, conditional on the sampling design (draw vs inclusion probability, support of `q_i`): `w_source/q_i` in an estimator over selected source states | Be sent to FairShip as a transport weight; be merged with `h(U)` or `wDIS`; be assigned the 1.80178·10¹⁰ normalization of the HA exit |
| `w_transport` (`fairship_transport_weight`; TTree `MCTrack[0].fW`) | The connector writes 1.0 | 1 | Neutral transport value | Injection (`fairship.py:76-77`, `:313-315`) | Differ from 1.0 in the current connector; carry `w_source` |
| `wDIS` | PR #1432: `muW/nDISevts·Σρ_iL_i` (`MuDISProcessor.cxx:362-363`; `MuonPath.cxx:28-39`; `Σ` runs over the sampled interval only: for MS the last 20 cm of accumulated material z, `MuDISProcessor.cxx:24-30`, `MuDISDefs.h:29-30`) | density × length ÷ N per unit of `muW`; unit system **not verified** (`MuonPath.h:85` accumulates `step·density`) | Per-DIS-realization path factor for one input muon and one material. No σ, no N_A (README:27) | Code behavior: the same factor is given to every realization of that muon and material, while vertices are sampled uniformly in length (`MuDISProcessor.cxx:307-310`, `:362-363`; slices keep their own densities, `MuonPath.cxx:163-174`). Using it as a density-weighted yield needs a constant-density condition or a sampling correction, **not demonstrated** | Be read as a complete event weight; be multiplied by `1/N_per muon` again; carry `w_source` when `muW = 1` |
| `σ_DIS(E_µ)` | Historical: Pythia6 via `makeMuonDIS.py`, one converged value per muon (note pp.15–16). PR #1432 stores successive `GetPARI(1)` values (`MuDISProcessor.cxx:323-324`); use of a single converged value is not demonstrated | cross section | Physical | Eq (2) `P_DIS` | Be taken from PR #1432 replay: `muDIS_xsec = -1` (`NewMuDISGenerator.cxx:273` comment, assignment `:327`) |
| `N_A` | Note eqs (1)–(2) | 1/mol | Constant of the DIS probability | `P_DIS`, separately from `wDIS` (which contains no `N_A`) | Be omitted when computing a probability; be applied twice |
| `ρ_domain`, `l_domain` (`ρl`) | Note eqs (1)–(3) | usual | Material budget | Already inside `wDIS` as `Σρ_iL_i` over the sampled interval | Be reapplied on top of `wDIS` |
| `nDISevts` / `N_per muon` | PR #1432 `-d` (requested) vs `nDISevts` (generated, can skip iterations, `MuDISProcessor.cxx:312-319`) vs retained after filtering (the filter keeps `wDIS` and `nDISevtsGenerated`, `MuDISFilter.cxx:968-969`, but rebuilds `nDISevts` from accepted interactions, `:1023`); note eq (5) `N_per muon` | count | Repeated DIS realizations from **one** muon state | The generated count is the denominator in `wDIS`; the requested count is not | Be treated as independent source draws; be confused with the requested or retained counts |
| `N_spills` | Note eq (5) | spills | Exposure scale: n-spills per 15-year run in the historical study (n=1 LoI, 10 current) | Eq (5) only | Be applied to a dataset whose exposure is unknown |
| `q_i` | Proposal probability of candidate selection | probability | Probability of drawing source state `i` in the pilot's selection | Denominator of the correction `w_source/q_i`. Recorded **independently** of `w_source` | Be inferred from `w_source`; be dropped when sampling non-uniformly |
| `h(U_i)` | Utility tilt used to build `q` | ≥ 0, dimensionless | Exploration preference | Only inside `q_i` | Be treated as a physical weight or multiplied into `w_source` |

## Acceptance checks (traceable)

| Criterion | Status | Evidence |
|---|---|---|
| Every physical-weight claim has a primary source or data evidence | Note facts and code constants: primary. Data classes: observed. The meaning of `w` in the pkl: **not established**, it rests on numerical match and secondary text | PROV |
| Martina ← 2018 lineage | OPEN | PROV §5 #4–5 |
| Meaning of all source weight classes | 6 classes match code constants (PROVISIONAL lineage), `w=1.0` and the /100 process OPEN | PROV §4–5 |
| The adapter does not erase source weights | VERIFIED in the connector | `fairship.py:47-48`, `:313-315`; caveat: `:93` alias `"weight"` |
| Proposal probability recorded apart from source weight | **Not enforced anywhere yet**: `q_i` has no field in the connector bookkeeping (PROV §6). A requirement, not a fact | Gap |
| Repeated DIS cannot pass as new independent source events | Only a rule so far (`canonical_post_ms_source_state.md:152-153`, "conditional repeats, not new P0 draws"); no test checked | Gap |
| `wDIS` not a complete weight | Documented here and in README:27; the code does not prevent misuse | PROV §6 |
| No double weighting | Requires the unique application points in the table; not enforced by code | Gap |
| Historical exposure, current configuration, subsets not mixed | Requires `N_spills` / exposure of the afterMS sample, which is OPEN | PROV §5 #10 |
