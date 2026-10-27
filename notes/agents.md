# Agents: who did what, and why

Working note, started 2026-10-25. Worktree rule (2026-10-27): the canonical checkout `ship_adaptive_muon_bg` belongs to whoever has a feature branch checked out there (now `thesis-2a`, `feat/minimal-loop-v0`). Every other session works in its own `worktrees/<name>` on its own branch. Several Claude sessions work in this repo. Git author is always `fbientrigo`,
so the **`Claude-Session` trailer** in each commit is the only way to tell sessions apart.
Rule: before editing a file listed under another session, send it a message (`SendMessage`, name below) and wait.
Each session adds or updates its own row. Nothing here is deleted; mark rows DONE instead.

## Sessions

| Session (name, id) | Role | Owns (edits only these) | Purpose | Status |
|---|---|---|---|---|
| `stage-wise-dis-outcomes-plan` (`session_01TLGPf7sSETuevgZBSaYeZu`, earlier `01MbEBFKxoNbViEcirfAhQ7c` and `017Jru3ZmhJRBYyRkf1qPmaT`) | Decisions, branch audit, metadata design (grill-me with the learner) | worktree `worktrees/ship-notes-triage`, branch `notes/branch-triage-v0` (from `main` `504a3d2`): `notes/branches.md`, `notes/priority_queue.md`, `notes/agents.md`; `.vibe-wise/` stays in the canonical checkout (untracked, learner state) | Queue items 10 (Parquet/ROOT metadata), 11 (prune + merge demo), 14 (one NF for both charges); issue-45 triage | ACTIVE |
| `thesis-2a` / minimal-loop (`session_01Reu2gBssY9d6aorTRGhHQ6`) | Minimal loop code, on branch `feat/minimal-loop-v0` (user-approved 2026-10-25; the canonical checkout is NOT on `main`) | `src/ship_muon_bg/{minimal_loop,dataset}.py`, `scripts/run_minimal_loop.py`, `tests/test_{minimal_loop,dataset}.py`, `adapters/fairship.py` (constant `LEGACY_AFTERMS_TRANSFORM` only, ~line 165), pilot script (import of it), later `artifacts/minimal_loop_v0/` and README Status | User goal "minimal-loop-v0": pool -> U0 -> tilt -> FairShip PR #1432 filter -> Parquet -> snapshot, one real adaptive iteration. Commits only its own files by explicit path; no push (its own report, 2026-10-25) | IN PROGRESS |

## Commits by session (checked 2026-10-27; `b3edcfa`, `504a3d2` are on `main`, `f7ad11b` only on `feat/minimal-loop-v0`)

| Commit | Session | What | Why |
|---|---|---|---|
| `b3edcfa` | `01MbEBFKxoNbViEcirfAhQ7c` | `notes/enrichment_tradeoff.md` | Record the exact/computed/measured enrichment trade-off for the alpha/delta scan (queue 13) |
| `504a3d2` | `017Jru3ZmhJRBYyRkf1qPmaT` | `notes/branches.md` (new), `notes/priority_queue.md` (+5 lines) | Branch inventory and queue items 14 and the worktree decision, before simplification |
| `f7ad11b` | `01Reu2gBssY9d6aorTRGhHQ6` | `minimal_loop.py`, `run_minimal_loop.py`, `test_minimal_loop.py`, `LEGACY_AFTERMS_TRANSFORM`, pilot uses it | Recover U0 scoring + eligibility gate (Magn7_* start volume -> `INELIGIBLE_IN_SHIELD_MAGNET`, never a negative) + tilt draws with per-draw seeds. Rescore matched `scratch/fairship-update-1012/u0_main.npz` on 39998 rows (claim from the commit message, not re-run here) |

## Overlap to watch

- `adapters/fairship.py`: `thesis-2a` added a constant near line 170. Branch `origin/fix/issue-45-provenance-correctness`
  (group A, `9b7e2dd`) changes `candidates[0]` at line ~356. Different hunks, but merge in one order and re-run tests.
- `simulation/evaluation.py`: group B of issue-45 conflicts only in the imports; nobody else edits it today.
- `notes/priority_queue.md` is shared. Append rows, do not reorder.

## Open questions (answered 2026-10-25 by `thesis-2a`: no queue number; will not edit `fairship.py:~356` nor `evaluation.py`)

1. Which queue item are you on, and which files will you edit next?
2. Does the eligibility gate change what PR #1432 receives (it excludes rows that start in iron)? Queue item 6 (the +2.5916 m offset) is still OPEN.

## Pitfall: the shared `.venv` imports the canonical checkout's `src`

`ship_adaptive_muon_bg/.venv` is installed editable from `ship_adaptive_muon_bg/src`. In any other worktree run
`PYTHONPATH=src ../../ship_adaptive_muon_bg/.venv/bin/python -m pytest ...`, or the tests silently check the other
session's branch. (Found 2026-10-27: a run without it printed `ship_muon_bg.__file__` in the canonical checkout.)
