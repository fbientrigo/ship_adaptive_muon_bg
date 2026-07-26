# D9-5N — Nightly Execution Workflow (Gate D)

## Normal nightly workflow

**Preferred: run the whole remaining queue unattended.** `start-campaign`
chains consecutive 8-hour blocks automatically, with no manual re-`start`
required between them, until Gate D is fully drained
(`GATE_D_COMPLETE_READY_FOR_VALIDATION_REVIEW`) or a blocking failure needs a
human:

```
cd C:\Users\Asus\Documents\FisicoFabi\tesis\ship_adaptive_muon_bg
.venv\Scripts\python.exe scripts\run_afterms_d9_5_nightly.py start-campaign
```

This launches one detached process for the entire remaining campaign — check
on it every few hours via `status`, not every block.

**Single block start** (only if you deliberately want to stop after one block
instead of running the whole remaining queue): run this once, then close the
terminal — training continues detached:

```
cd C:\Users\Asus\Documents\FisicoFabi\tesis\ship_adaptive_muon_bg
.venv\Scripts\python.exe scripts\run_afterms_d9_5_nightly.py start
```

Or double-click `run_d9_5_night.cmd` in the repo root (equivalent to a single
`start`).

`start` and `start-campaign` each refuse to run while the other is already
live (a single-block lock at `locks/supervisor.lock`, plus a campaign lock at
`locks/campaign.lock` while a chain is in progress) — `status` reports both.

**Check status at any time:**

```
.venv\Scripts\python.exe scripts\run_afterms_d9_5_nightly.py status
```

**Watch the live log:**

```
.venv\Scripts\python.exe scripts\run_afterms_d9_5_nightly.py tail --follow
```

**Ask it to stop gracefully after the current epoch** (before going to bed early,
or before a planned reboot):

```
.venv\Scripts\python.exe scripts\run_afterms_d9_5_nightly.py stop-after-epoch
```

**Morning check:**

```
.venv\Scripts\python.exe scripts\run_afterms_d9_5_nightly.py status
```

If the queue isn't fully drained, just run `start` again the next night — it
resumes automatically from the exact next epoch of whichever run was in
progress. Repeat until `status`/`start` prints:

```
GATE_D_COMPLETE_READY_FOR_VALIDATION_REVIEW
```

---

## Important operational notes

- **It is safe to reboot** the machine after a nightly block has stopped (or
  after `stop-after-epoch` has been honored). Every epoch's checkpoint is
  written atomically (temp file + `os.replace`) before the next epoch starts,
  so there is never a half-written checkpoint on disk.
- **Running `start` the next night resumes automatically.** The supervisor
  reconstructs its state from disk (`status.json` + checkpoint files), not
  from a cached in-memory assumption, so a hard reboot or crash never loses
  track of where training was.
- **Never manually delete or edit files under**
  `artifacts/afterms_d9_5_model_family_arena_v0/nightly_runner/` or
  `artifacts/afterms_d9_5_model_family_arena_v0/runs/`. Doing so can corrupt
  the resume contract. If something looks wrong, run `reconcile` and read the
  `status` output first.
- **Recognizing a BLOCKED state**: `status` shows the current run's state as
  `FAILED_BLOCKED`, and the campaign will not advance to the next run until a
  human resolves it. This means the supervisor hit a declared blocking
  condition (CUDA OOM, non-finite loss, checkpoint hash mismatch, etc.) — it
  never modifies hyperparameters or skips the run on its own.
- **Incident reports** are written to
  `artifacts/afterms_d9_5_model_family_arena_v0/nightly_runner/incidents/` as
  matching `.json` and `.md` files, one pair per incident. Each `.md` includes
  an exact manual recovery command (`fit ... --resume`) at the bottom.
- **After an unexpected reboot or crash**, run:

  ```
  .venv\Scripts\python.exe scripts\run_afterms_d9_5_nightly.py reconcile
  ```

  This resyncs the lock and queue state against disk without launching
  anything, so you can inspect `status` before deciding to `start` again.
- **Checking that no GPU process remains** after a block has stopped:

  ```
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
  ```

  An empty result (no rows) means no CUDA process is active. You can also
  confirm no stray supervisor is holding the lock via `status` (`lock.live`
  should be `false` once a block has cleanly ended).
- **One command wrapper**: `run_d9_5_night.cmd` at the repo root resolves its
  own directory and the local `.venv`, so it can be run from any working
  directory or double-clicked directly. It forwards any extra arguments to
  `start`.

## What `start` does by default

- Duration: 8 hours (soft-stop at 7.5h, hard exit margin of 0.5h)
- Detached: yes — the supervisor survives the terminal that launched it
  closing
- Device: `cuda`
- CPU policy: 2 threads (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS` all set to 2 on the training
  child process)
- Phase: Gate D only (the six NF_AC production runs); it never opens the test
  split and never starts Gate E on its own

## Full subcommand reference

| Command | Purpose | Mutates state? |
|---|---|---|
| `init` | Create the nightly state tree and frozen 6-run queue. Idempotent. | Yes (first run only) |
| `doctor` | Environment checks (CUDA, disk, python, psutil). | No |
| `start` | Acquire the lock and launch a detached supervisor for one 8-hour block. | Yes |
| `run-foreground` | The actual supervisor loop (used internally by `start`, or directly for foreground debugging). | Yes |
| `start-campaign` | Acquire the campaign lock and launch a detached loop that chains consecutive 8-hour blocks unattended until the queue drains or a blocking failure needs a human. | Yes |
| `run-campaign-foreground` | The actual campaign loop (repeated blocks; used internally by `start-campaign`, or directly for foreground debugging). | Yes |
| `status` | Campaign/lock/queue report (includes both the per-block lock and the campaign lock). | No |
| `tail [--follow]` | Tail the current block's log file. | No |
| `stop-after-epoch` | Request a graceful stop at the next epoch boundary. | Writes a flag file only |
| `abort` | PID-specific termination of the live supervisor + its one active `fit` child. Never a by-name kill. | Yes |
| `reconcile` | Force a state resync against disk without launching anything. | Yes (lock/queue bookkeeping only) |
| `history` | List past nightly blocks and their outcomes. | No |
