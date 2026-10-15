# Muon weight provenance — `muonsFullMC_afterMS.pkl` column `w`

> **HISTORICAL / NON-BLOCKING FOR CURRENT PIPELINE** (project decision 2026-10-12, `docs/decisions/historical_dataset_scope_correction_v0.md`). This covers Martina's historical dataset only. It is not the weight definition of the new FairShip simulations. Findings and OPEN items below are unchanged.

Status: investigation record, 2026-10-12. Tags: **VERIFIED** / **PROJECT DECISION** / **PROVISIONAL** / **OPEN** / **REFUTED**.
Authoritative contract: `SHIP-EMPIRICAL-FS-SIM-CONTRACT-V0-SOURCE`, referenced by ID only. Its text is **not available locally** and was not inspected (see `docs/decisions/canonical_post_ms_source_state.md:145`).
This note makes no estimate of the physical background rate and does not claim a normalized P0.

## 1. SHiP note (October 2024) — [VERIFIED]
Source: Bezshyiko, Lantwin, Ovchynnikov, Reghunath, *Signal and background studies for ECN3*, 1 Oct 2024 (copy: `NFlow/study_material/2410_IMPORTANT_SHiP_signal_background_note.pdf`). Section 2 "Muon DIS".

| Fact | Where |
|---|---|
| Target + hadron-absorber (HA) pre-production, 67 files in `/eos/experiment/ship/data/Mbias/background-prod-2018/`; each muon has a weight `w_µ` with Σ_67 Σ_µ `w_µ` = `N_µ^spill` = 1.80178·10¹⁰, "the number of muons exiting the HA per nominal SHiP spill" | printed p.14 |
| Muons are re-propagated "starting from the target" with the fast-muon option. LoI post-shield file `…/sc_v6_full_spill/sc_v6_merged.root` (≈O(10⁵) unweighted muons); 10-spill file `…/sc_v6_10_spills/sc_v6_MERGE.root` | p.15 |
| Front/side/cavern classes. `n_front`, `n_side` = 35, 6072 (1 spill) and 392, 62010 (10 spills). `N_per muon` front/side = 7000/700 (LoI), 10000/100 (current) | p.15–16 |
| (1) `dP_DIS/dl = σ_DIS(E_µ)·A·n_domain = σ_DIS(E_µ)·N_A·ρ_domain` | p.16 |
| (2) `P_DIS = Σ_domains (dP/dl)·l_domain ≡ σ_DIS(E_µ)·N_A·ρl·Σ_domain ω_domain`; (3) `ρl ≡ Σ(ρl)_domain`, `ω_domain = (ρl)_domain/ρl` | p.16 |
| (4) `N_DIS,expected = Σ_{i=front,side} N^i_DIS,expected`; (5) `N^i = (N_spills / N^i_per muon)·Σ_j ω_µ^(j)·P_DIS^(j)`, `N_spills` = number of n-spills per 15-year run (n=1 LoI, n=10 current), sum over the sampled (n×N)_j events | p.17 |

Boundary: the note's Muon DIS section describes a FairShip production and its sc_v6 post-shield files. It does **not** mention "Muons & Matter" or `muonsFullMC_*.pkl` (the note does name M. Ferrillo elsewhere, p.25, in the EventCalc work). 1.80178·10¹⁰ is an **HA-exit** count and must not be transferred to a post-shield sample.

## 2. Martina's dataset documentation
- Original README `/eos/user/m/mfership/public/SHiP/MC/Muons/README`: **not accessible** (no `/eos` here). Direct verification of its text is **OPEN**. Nothing local is a verbatim copy.
- Secondary text (the project's own release notes, not Martina's):
  - `fbientrigo/NFlow` releases `v1.0.0-fullmc` and `v1.0.0-fullmc-afterms` (2025-09-18): "FairShip full-simulation muons propagated through the Muon Shield using the Muons&Matter tool", columns `[px, py, pz, x, y, z, id, w]`, `w` "event weight (relative importance)".
  - `NFlow/scripts/07_analysis_on_data.py:22-26`: "`w` – event weight".
- Ferrillo, *SHiP 33rd Collaboration Meeting*, 5 June 2025 (`NFlow/study_material/2506_MFerrillo_ML_proposal.pdf`, 34 pp.; read as text via `pdftotext`; slide numbers not recorded) — [VERIFIED as quotes]:
  - "FairShip can already read Muons & Matter input data from NTuples (and adding support for new formats trivial)".
  - Under "USING MUONS & MATTER": "Propagating muons inside the MS (optimal design for the warm magnet option) using standalone tool [Luis Felipe's talk]". So M&M is a standalone muon-shield propagation tool, and the attribution to a talk by a third person is the slide's.
  - "Currently generated MC sample corresponds to only about 1 spill [*] of real data", with footnote "[*] Simulated sample statistics for the minimum bias corresponding to 6.5 × 10¹⁰ PoT, i.e. 2.5 × 10⁻³ spills [/eos/experiment/ship/data/Mbias/background-prod-2018/README]".
  - The slides do not mention the pkl files, `w`, or the EOS path of this dataset.
  - **Tension (OPEN):** 6.5·10¹⁰ PoT = 2.5·10⁻³ spills implies about 2.6·10¹³ PoT per spill, while the code comment normalizes `weightMbias` to "5E13 pot" (=1 spill, ratio 768.75). The production README was not available to resolve this.
- A public web search (AGY research job, 2026-10-12; summary only, no transcript kept in the repo) found **nothing** public on "Muons & Matter", the EOS path, or the meaning of `w`. This is absence of a public source found in one search, not evidence about the content.
- **UNSUPPORTED (no source; not falsified):** `phd-thesis/thesis_material_audit/dataset_registry.json` and the wiki pages `05_Dataset_Provenance/*` state `weight_semantics: physical_cross_section (mb)`. No cited source supports this for the pkl files. Its origin is unknown (a guess: the `CrossSection` column of `sc_v6_merged.root`, untraced). Reported only; the registry was not edited.

## 3. FairShip code that defines the weight classes — [VERIFIED as code facts]
`muonShieldOptimization/extractMuonsAndUpdateWeight.py`, read at `e8627d1de` (canonical) and checked against the exact pin `8a72b3941a1c41238a0a25ed0559e8e6dd17c099` (2026-10-10). The weight constants are identical in both. The pin only adds a `process_muons_only` option to `processFile`, which changes which events are kept, not the weights. File added by T. Ruf in `258994cee` (2018-03-05); `afcf7c14c` (2018-04-10) modified it.

| Line | Content |
|---|---|
| `:15-16` | `# for 10GeV Yandex Production 65.041 Billion PoT, weight = 768.75 for 5E13 pot` → `weightMbias = 768.75` |
| `:18-20` | `# for 10GeV charm Production 102.2 Billion PoT equivalent, weight = 489.24` / `# added another cycle` → `weightCharm = 489.24 * 2.0 / 3.0` (= 326.16). The factor 2/3 is explained only by the comment "added another cycle" (an equivalent exposure of about 153.3·10⁹ PoT would give 326.16; this is an inference) |
| `:22-23` | `# for 10GeV charm Production 5336 Billion PoT equivalent, weight = 9.37` → `weightBeauty = 9.37` (the comment says "charm" for the beauty line; the variable and file name `…beauty_5336B…` say beauty) |
| `:30-50` `muonUpdateWeight` | for each muon found through `sTree.vetoPoint` (so only muons with a veto point are reweighted; `:32`): `SetWeight(weight)`; ×1/`diMuboost` for "Hadronic inelastic", or for primary/decay muons whose mother is η, ω, φ, ρ⁰ or η′ (`muSources`); ×1/`xSecboost` for "Lepton pair" or "Positron annihilation". With `noCharm`, muons whose mother is in `charmExtern` are skipped (heavy flavours come from separate productions) |
| `:53-91` `PoT` | `diMuboost`, `xSecboost`, PoT are parsed from the `FileHeader` title; the code comment gives the example "diMu100.0 X100.0" |
| `run()` | loops `range(0, 67000, 1000)` = 67 files, `_mu.root` outputs in `background-prod-2018/` |

Arithmetic (derived here, not stated in the source): 5·10¹³/65.041·10⁹ = 768.75 and 5·10¹³/5336·10⁹ = 9.37; 5·10¹³/102.2·10⁹ = 489.24, which is then multiplied by 2/3 for charm. So `weightMbias` and `weightBeauty` are "5·10¹³ PoT ÷ simulated PoT"; for charm that holds only for the base 489.24.

`muonShieldOptimization/exitHadronAbsorber.cxx` (pin) carries its own per-particle weights (decay-survival and clone weights, `:160-300`). Whether this code existed when the 2018 files were made, and how it relates to `weightMbias`, is **OPEN** (not checked in git history).

## 4. Audit of the actual data — [VERIFIED as observations]
Script: `scripts/audit_weight_provenance.py`. Raw outputs: `~/thesis/scratch/weight-provenance/audit_{beforeMS,afterMS,afterMS_40k}.json`.
Files: GitHub release assets of `fbientrigo/NFlow`. sha256 match the manifests:
- `muonsFullMC_afterMS.pkl` `fb251be2cbfb04aed52bee215c23cbd0898eccae989523952d3b774c004f974c`, 13,779,080 rows.
- `muons_FullMC.pkl` `91273fe62e30a2bc5647d60ec51c7212b8659df31eaf670e9fc4a21d232ceb9c`, 16,533,515 rows.

Weight classes (the values are float32 numbers, for example `9.369999885…`):

| w | rows before | rows after | after/before | Σw after | note |
|---|---|---|---|---|---|
| 768.75 | 454,906 | 362,983 | 0.798 | 2.790e8 | = `weightMbias` |
| 326.16 | 362,183 | 288,452 | 0.796 | 9.408e7 | = `weightCharm` |
| 9.37 | 5,509 | 4,514 | 0.819 | 4.230e4 | = `weightBeauty` |
| 7.6875 | 15,687,580 | 13,104,800 | 0.835 | 1.007e8 | = 768.75/100 |
| 3.2616 | 22,512 | 17,704 | 0.786 | 5.774e4 | = 326.16/100 |
| 0.0937 | 108 | 76 | 0.704 | 7.12 | = 9.37/100 |
| **1.0** | 716 | 550 | 0.768 | 550 | **not predicted** |
| 0 (−0.0) | 1 | 1 | — | 0 | one row in each file |

Other observations:
- **The prediction was partly falsified.** The hypothesized set {768.75, 326.16, 9.37}·{1, 1/100} ∪ {0} missed one class: **w = 1.0** (716 / 550 rows). Its meaning is **OPEN**. It may be a default or unreweighted weight, but the data cannot say that.
- Σw: beforeMS 5.886·10⁸ (3.27% of 1.80178·10¹⁰), afterMS 4.740·10⁸ (2.63%); weighted afterMS/beforeMS = 0.805. Effective sample size (Σw)²/Σw² ≈ 1.12·10⁶ (before) and 9.1·10⁵ (after), about 6.6–6.8% of the row count.
- The weight classes appear at the same values before and after Muons&Matter, and the per-class row-count ratio after/before (0.70–0.84; not a measured survival probability, because row identities between the files are unknown) is similar across classes. This is **consistent with** `w` being carried unchanged through the transport row by row. It does not prove it.
- 95% of the rows (both files) have the /100 class 7.6875 but they hold only 21% of Σw; 768.75 and 326.16 together hold 79%.
- No exact duplicate rows in either file.
- beforeMS: 55.5% of the rows have x = y = 0 exactly; z ∈ [−2.545, 3.888] m. afterMS: z ∈ [28.895, 28.905] m, x and y at exactly ±7.5 m for 66 / 19 rows (the mechanism — a box cut, clipping, a transport boundary or a selection — is **OPEN**; the audit only counts coordinates).
- The 40k committed sample has classes 768.75 (1044), 326.16 (863), 9.37 (11), 7.6875 (38028), 3.2616 (53) and −0.0 (1). It has no 1.0 and no 0.0937 rows. Their absence is observed, not shown to be a defect: a uniform 40k draw expects about 1.6 `w=1` and 0.2 `w=0.0937` rows (the sample uses `uniform_core_plus_range_anchors`, `subsampling.py:54`).

## 5. Decision record

| # | Claim | Status | Evidence / what closes it |
|---|---|---|---|
| 1 | The 2018 pre-production weights sum to 1.80178·10¹⁰ muons per nominal spill at the HA exit | VERIFIED | Note p.14 |
| 2 | The constants 768.75 / 326.16 / 9.37 exist in the code, and 768.75 and 9.37 equal 5·10¹³ PoT ÷ the stated PoT (rounded arithmetic) | VERIFIED (constants and arithmetic only) | Code `:15-23` |
| 2b | Each constant is a "one nominal spill" factor; the charm factor (489.24 × 2/3, `run4Charm` handles three cycles `:147`, `mergeCharm` two `:185`) and the PoT-per-spill convention (see §2 tension) | PROVISIONAL | Production README, author |
| 3 | The observed classes {768.75, 7.6875, 326.16, 3.2616, 9.37, 0.0937} equal the code constants (×1 or /100) | VERIFIED (numerical match) | Section 4 |
| 4 | The classes were generated by this scheme in the files Martina used | PROVISIONAL | A match is not lineage. Needs README, scripts or the author |
| 5 | Martina's files derive from background-prod-2018 | OPEN | Sum is only 3.3% of the 67-file total; 1 GeV production also exists (`compactingBackgroundProduction.py:26-31`) |
| 6 | Which process (hadronic/decay vs lepton-pair) a /100 class represents | OPEN | The factor 100 is ambiguous (`diMu100 X100` example) |
| 7 | `w` = 1.0 class meaning | OPEN | Not in code constants |
| 8 | `w` is a cross-section in mb | UNSUPPORTED (not independently falsified) | No source found for it; the weight set equals production-constant classes. The suggested origin in the `sc_v6` `CrossSection` column is a guess with no traced evidence |
| 9 | `w` unchanged through Muons&Matter | PROVISIONAL | Same class values, similar survival. Needs M&M code or output check |
| 10 | afterMS Σw is a nominal physical flux per spill | OPEN (do not assume) | Exposure, selection and subset undefined |
| 11 | Weights belong to rows or to events / proton interactions | OPEN | No event-ID column; the pkl cannot show ancestry |
| 12 | With the current connector (TTree `w = 1.0`) `wDIS` carries no `w_source` | VERIFIED, conditional | `fairship.py:76-77,313-315`; PR #1432 `MuDISProcessor.cxx:433,362-363` (pin 400c0fb) multiplies whatever `MCTrack[0]` weight it reads; the PR does not enforce `muW = 1` |

## 6. Pipeline interaction (code facts)
- Connector (`src/ship_muon_bg/adapters/fairship.py`): `physical_source_weight` and `fairship_transport_weight` are separate fields (`:47-48`); transport weight must be 1.0 (`:76-77`); bookkeeping stores both plus `ttree_w = 1.0` (`:313-315`). **Finding, not fixed:** `:93` falls back to the key `"weight"` as `physical_source_weight`, an ambiguous name that can import any "weight" as a source weight.
- PR #1432 @ 400c0fb: `muW = MCTrack[0].GetWeight()` (`MuDISProcessor.cxx:433`); `wDIS = muW / nDISevts · Σ ρ_i L_i` (`:362-363`, `MuonPath.cxx:28-39`; the sum covers only the sampled interval, for MS the last 20 cm of accumulated material z, `:24-30`; `nDISevts` belongs to one muon/material branch); no σ, no N_A (README:27); replay writes `muDIS_xsec = -1` (placeholder comment `NewMuDISGenerator.cxx:273`; branch `:317`, assignment `:327`); the filter sums `nDISevts·wDIS` (`MuDISFilter.cxx:1045-1047`).
- Consequence: `1/nDISevts` in `wDIS` plays the role of `1/N_per muon` in eq (5). An analysis that also divides by `N_per muon` double-counts it. `wDIS` is a path-density factor, not a complete event weight.
- The ledger is in `docs/contracts/weight_ledger_v0.md`; proposed amendments in `docs/decisions/weight_contract_amendment_v0.md`.

## 7. Minimum questions for the SHiP team
1. (Martina) README of `…/Muons/`: which FairShip file(s) and production (background-prod-2018 10 GeV, 1 GeV, or sc_v6) fed `muonsFullMC_beforeMS`, and is `w` the `_mu.root` weight copied unchanged?
2. (Martina) Does `muonsFullMC_beforeMS` hold the 67 files or a subset (its Σw is 3.3% of 1.80178·10¹⁰)? What is the 55.5% of rows with x = y = 0, and why do x and y end at ±7.5 m after the shield?
3. (Oliver/Maksym) What produces w = 1.0 (716 / 550 rows), and which process does each /100 class stand for?
4. (Oliver/Maksym) The June 2025 slides give 6.5·10¹⁰ PoT = 2.5·10⁻³ spills (≈2.6·10¹³ PoT/spill), while `weightMbias` is normalized to 5·10¹³ PoT. Which PoT per spill defines `w`?
5. (Oliver/Maksym) Is the afterMS sample meant to be normalized per nominal spill, and what exposure (PoT, spills) does its Σw correspond to?

## 8. Reproduce
```bash
cd ship_adaptive_muon_bg
python scripts/audit_weight_provenance.py --input <pkl> --label afterMS --out audit_afterMS.json
python -m pytest tests/test_weight_provenance_audit.py
```
Downloads: `gh release download v1.0.0-fullmc-afterms -R fbientrigo/NFlow -p muonsFullMC_afterMS.pkl` (and `v1.0.0-fullmc`, `muons_FullMC.pkl`). Peak memory about 3.4 GB.
