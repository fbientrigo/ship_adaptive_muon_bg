# Proposed weight clauses (amendment v0)

> **Scope (project decision 2026-10-12, `docs/decisions/historical_dataset_scope_correction_v0.md`).** Clauses 1–5 are CURRENT-PIPELINE rules (PROVISIONAL). Clauses 6–7 and "Evidence to close" concern Martina's historical dataset only and are HISTORICAL / NON-BLOCKING. Findings and OPEN items below are unchanged.

Status: `PROPOSAL`. **Not checked against `SHIP-EMPIRICAL-FS-SIM-CONTRACT-V0-SOURCE`**, whose text is not available locally. They may duplicate or conflict with it. Nothing here changes an existing repository contract.
Evidence: `docs/investigations/muon_weight_provenance.md`; names: `docs/contracts/weight_ledger_v0.md`.

1. **Separate names.** `w_source`, `w_transport`, `wDIS`, `q_i` and `h(U_i)` are distinct stored fields. The key `"weight"` is not accepted as `physical_source_weight` (current alias at `src/ship_muon_bg/adapters/fairship.py:93`).
2. **`w_source` is applied once**, as `w_source/q_i`, in the estimator. It is never sent to FairShip (TTree `w = 1.0`) and never multiplied into `h(U_i)` or `wDIS`.
3. **`q_i` is recorded at selection time**, independently of `w_source`, for every candidate, including the uniform case.
4. **`wDIS` is not an event weight.** It has no cross section and no exposure. Any yield built from it needs σ_DIS and an exposure statement, and must not divide by `N_per muon` again.
5. **Repeated DIS realizations** of one `SourceState` are conditional repeats grouped by that state. They never count as new draws (this restates `canonical_post_ms_source_state.md:152-153`).
6. **No transfer of the HA-exit normalization.** 1.80178·10¹⁰ muons per nominal spill (note p.14) is not assigned to the post-shield sample. A `Σw` of the afterMS file is reported with its subset and exposure labelled OPEN until the lineage is established.
7. **Weight-class labels in reports** state the value and the matching code constant, with the status PROVISIONAL until lineage is confirmed.

Evidence to close: the Martina README and production configuration; confirmation from the SHiP team of the `w = 1.0` class and the /100 classes.
