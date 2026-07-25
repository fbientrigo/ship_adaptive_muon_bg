# Agent Instructions

## Running D9-5 Gate D nightly GPU training blocks

Gate D (the six-run NF_AC production arena, `configs/afterms/d9_5_model_family_arena_v0.json`)
runs in restartable 8-hour blocks via the D9-5N nightly runner. Any agent picking up
this work should use the commands below — do not invoke the training CLI
(`scripts/run_afterms_d9_5_model_family_arena.py fit ...`) directly for production
Gate D runs; the nightly runner owns process ownership, locking, CPU/GPU policy,
and exact epoch-boundary resume.

Full workflow, operational notes, and subcommand reference:
`docs/reviews/afterms_d9_5_nightly_execution_v0.md`.

**Start (or resume) an 8-hour block** — detached, survives the terminal closing:

```
.venv\Scripts\python.exe scripts\run_afterms_d9_5_nightly.py start
```

**Check status** (safe, read-only; use this instead of tailing logs when just
checking whether a block is still alive):

```
.venv\Scripts\python.exe scripts\run_afterms_d9_5_nightly.py status
```

`status` reports `lock.live`, the per-run queue state, and the current run's
last completed epoch. A block that has ended cleanly shows `lock.live: false`
and the current run in `INTERRUPTED_AT_BLOCK_DEADLINE` (normal, resumable) or
`COMPLETED`.

**Do not poll status more than once per hour while a block is running.** A
block runs for up to 8 real hours; checking every few minutes burns tokens for
no new information. Check hourly, or every 4 hours, and rely on background
task notifications rather than manual polling loops.

**Stop-here procedure between blocks**: once `status` shows `lock.live: false`
and the current run has advanced to `COMPLETED` or
`INTERRUPTED_AT_BLOCK_DEADLINE`, it is safe to launch the next `start` — it
resumes automatically from the next epoch. Never delete or edit files under
`artifacts/afterms_d9_5_model_family_arena_v0/nightly_runner/` or
`artifacts/afterms_d9_5_model_family_arena_v0/runs/` between blocks.

Rules that must not be violated by any agent running this workflow:

- Never read or open the test split, and never start Gate E, while Gate D is
  in progress.
- Never modify the frozen scientific config
  (`configs/afterms/d9_5_model_family_arena_v0.json`) or execution policy to
  make a block "fit" — a `FAILED_BLOCKED` state must be surfaced to a human,
  not silently worked around.
- Never kill processes by name (`taskkill /IM ...`, `Stop-Process -Name ...`).
  Use `scripts/run_afterms_d9_5_nightly.py abort`, which is PID-scoped to the
  supervisor's own recorded lock.
