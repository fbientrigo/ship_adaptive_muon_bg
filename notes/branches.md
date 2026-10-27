# Branches (thesis repo `fbientrigo/ship_adaptive_muon_bg`)

Checked 2026-10-20 against `main` (`b3edcfa`); recounted 2026-10-25 on `504a3d2`. "In main" = every commit is an ancestor of `main`, or every
changed file is byte-identical in `main`. Nothing here is deleted; GitHub cleanup (deleting the LEGACY rows)
is a later, separate decision that needs an explicit push request.

| Branch | Status | Where its content is |
|---|---|---|
| `main` | LIVE | - |
| `demo/pi-utility-guided-v0` | LIVE | to be pruned (queue 11), then merged |
| `feat/minimal-loop-v0` | LIVE | other session (`thesis-2a`), see `notes/agents.md`; not mine to touch |
| `notes/branch-triage-v0` | LIVE | this notes worktree; merge to `main` after the user approves |
| `chore/publish-validation-artifacts` | LIVE | afterMS -> FairShip transform validator (queue 2, 6) |
| 23 remote branches, 2026-06-29..09-27 (`development`, `sprint1/*`, `integration/*`, `gate/d5pilot`, `experiment/d5-*`, `experiment/d7-*`, `experiment/d9-5n-*`, `claude/*`, `docs/*`, `feat/architecture`, `feat/ml-proposal`, `feat/feature-views-v0`, `feat/controlled-targets-d0-d2-v0`, `fix/unvalidated-d5-rare-gate-bypass`, `ci/numpy-pytest-v0`, `chore/project-state-consolidation`) | LEGACY | all commits in `main` |
| `fix/ci-portable-test-scope`, `rebase/sprint2-issue31-ci` | LEGACY | different hashes, every file identical in `main` |
| `experiment/d9-utility-tilt-arena-v1`, `experiment/d9-full-data-replay-v0` | LEGACY after demo merge | all commits in demo |
| `experiment/d9-conditional-charge-nf-v0`, `feat/d9-conditional-charge-sampling-v1` (same tip `eea984e`), `experiment/d9-conditional-charge-sampling-v1` | LEGACY | contained in `experiment/d9-conditional-utility-tilt-v0` |
| `experiment/d9-conditional-utility-tilt-v0` | REFERENCE | one NF conditioned on charge; queue 14 |
| `experiment/d9-direct-utility-sampling-v0` | LEGACY after demo merge | demo + 1 commit (52-line doc `docs/d7_full_local_replay_v0.md`) |
| `fix/issue-45-provenance-correctness` | NOT EXTRACTED | 16 commits of fixes absent from `main`; under review |
| `audit/initial-inventory` | LEGACY | June 2026 repo snapshot (tree, status, log) |
| `research/u-definition-fairship-audit` | LEGACY | July U(x) audit, superseded by stage outcomes; its section 7 lists candidate fields (DOCA, IP, fiducial, SBT/UBT) still absent from our output |
| `data/full-mc-small-fixtures` | LEGACY | same NFlow releases as `data/samples/`; adds a release downloader + CI workflow |
| `feat/distilled_data` | LEGACY | 1M / 1.7M-row distilled datasets (~80 MB) with a two-sample fidelity test; replaces `data/samples/` |
