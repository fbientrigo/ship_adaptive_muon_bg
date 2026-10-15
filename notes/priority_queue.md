# Priority queue (working notes, not authoritative)

Ordered by ROI. At most **3 IN PROGRESS** at a time. Two lanes: **ORDENAR** (git/branches/worktrees,
moving one definition to one place) comes before **LIMPIAR** (deleting or archiving code).
Nothing is deleted or moved to `archive/` without its own grill-me decision.
Long-term learnings go to `~/thesis/vault`; this file tracks current work.

## Decisions so far (grill-me, 2026-10-14/15)

- DIS engine stays behind an adapter (`adapters/pr1432_mudis.py`), pinned to PR #1432 `400c0fb`, PROVISIONAL; swapping engines = one import.
- `afterms/` and `density_lab/` stay: they hold the NF capability (S5 kept, not overridden).
- Docs are not deleted. `docs/` = how things work + plans; `notes/` = agent working memory (versioned).
- Branch rule: at most 2-3 open feature branches. Commits only after the user sees target branch + content; at most ~10 grouped commits per cleanup.
- AGENT_POLICY: user agrees "approve per decision, with a 'what you learn' note" beats a line budget; amendment to be proposed, not applied.
- `.venv` holds the declared `lab` + `flow` extras (2026-10-15): scikit-learn, scipy, matplotlib, torch 2.13.0+cpu
  (same build as `ship-pi-demo-v0/.venv`). `pyproject.toml` unchanged. Core suite 746 passed; afterms 360 passed, 19 failed.
- 2026-10-15 (round 3): stage-outcomes and weight-provenance work committed and merged; canonical checkout now on `main`;
  feature work lives in worktrees (max 2-3). Worktree `artifacts/` (ignored, real FairShip outputs) copied into the
  canonical `artifacts/` before a worktree is removed. AGENT_POLICY amended (decision-based approval, grill-me).

## Queue

| # | Lane | Item | Status |
|---|---|---|---|
| 1 | ORDENAR | Branch/worktree consolidation: commit pending work, merge to `main`, drop merged branches/worktrees | DONE 2026-10-15 (2 rebased duplicates await `-D` approval) |
| 2 | ORDENAR | Study the 2 unmerged branches: `chore/publish-validation-artifacts` (afterMS->FairShip transform validator, 2.5916 m) and `demo/pi-utility-guided-v0` (`density_lab/utility_tilt.py`, the NF utility-tilt sampler the pilot's checkpoints come from) | IN PROGRESS (grill-me) |
| 3 | ORDENAR | One SBT predicate in `benchmarks/fairship_endpoint.py`; detID range, pdg, p threshold and `z + 2.5916 m` as named constants imported by pilot and `audit_fairship_tgeo.py`; semantics unchanged | NEXT |
| 4 | ORDENAR | AGENT_POLICY amendment (decision-based approval) | DONE 2026-10-15 |
| 5 | LOOP | Loop v0 first run: U0 -> sample with delta floor -> PR #1432 adapter -> `aggregates.csv` -> histogram U update, few bounded rounds; NF arm after | after 1-3 |
| 6 | OPEN | Verify SBT detID ranges for TRY_2026 geometry and the provisional `+2.5916 m` frame offset | after 3 |
| 7 | OPEN | PR #1432 assumes muons before the last shield interaction length; post-shield states never get MS DIS (B5) | backlog |
| 8 | LIMPIAR | Audit findings 2026-10-14 (dead constants, duplicated helpers, legacy protocols, `graphify-out/`), each by its own decision | backlog |
| 9 | LIMPIAR | 19 afterms tests fail: missing local `artifacts/afterms_d8_evaluation_v0/` and Windows-only process code in the nightly runner (`.venv/Scripts/python.exe`, PID tree) | backlog |
