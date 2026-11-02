# Audit of the current tree against the split (ponytail-audit, read-only)

Sizes from `git ls-files` at `e4c901c` (branch `feat/p2-extraction-v0`). Out of scope: correctness, performance.
Earlier I proposed deleting `afterms`/`density_lab`/`benchmarks`; WRONG, they are the NF line. Retracted.

## Where things go (VERIFIED import edges)
| Package | Lines | Imports from the rest | Goes to |
|---|---|---|---|
| `afterms` | 8281 | `data_contracts` (5), `density_lab` (1, `d8/statistics.py`) | A |
| `density_lab` | 4153 | none | A |
| `Nflow` | 2661 (with `ProxyTagger`) | - | A |
| `data_contracts` | 1298 | none | split: A keeps the row/feature-view part; the shared contract is rewritten (see 04) |
| `benchmarks` | 2303 | `tagging`, `entities` (1 each) | split by what they benchmark (user) |
| `adapters` (P2 etc.) | 1371 | `entities`, `simulation` | B |
| `entities`, `tagging`, `simulation` | 518 / 1372 / 1950 | `entities` only | B |

## Cuts and redesigns, ranked
1. `yagni:` two plan lines in one tree. `afterms` has D8/D9/D9.5/D9b stages (`d9_5/nightly_runner.py` 908, `d9b/gpu_gate.py` 779, `scripts/run_afterms_nightly_queue.py` 1628, `watch_*`, `run_d9_5_night.cmd`). In repo A replace by one config-driven runner + a queue that is the pipe (stdin/stdout of jobs), not a nightly scheduler.
2. `yagni:` `afterms/d9_5/{gaussian,gmm,nf_ac}_adapter.py` (236+197+305) are wrappers over `Nflow/baselines/*` and the flow, behind `model_adapter.py`. Checked: they import the baselines (not duplicates). In A, one model protocol and the models implement it directly; the wrapper layer goes.
3. `delete:` legacy HDF5-era `Nflow/legacy/*` (untested, `legacy` extra: h5py, seaborn, pandas, optuna, tensorboard, pyyaml). Do not carry to A; keep in the old repo's history.
4. `yagni:` `afterms/model_naming.py` (426) and the alias-inventory script. A model id should be a hash of its config, not a naming scheme.
5. `shrink:` `afterms/d8/legacy_d7_adapter.py` (600) exists to read an older layout; greenfield removes the need.
6. `delete:` in B, `simulation/fake_fairship.py` (719) + `stub_backend.py` (254): a fake backend for the old runner protocol (contract OPEN-11). Replace with the real pinned FairShip smoke (as `p2_live_smoke.py`) plus a tiny fixture.
7. `yagni:` `tagging/aggregation.py` (1005) + `simulation/evaluation.py` (803): rich, but with no real producer on `main`; only PR #1432 smoke wrote to it. Port in B only what the minimal loop consumes; re-grow on demand.
8. `shrink:` three copies of the SBT rule and string outcomes in `benchmarks/fairship_contract.py` (568): one function, one status enum (PASS / PHYSICS_REJECTED / NOT_EVALUATED / TECHNICAL_FAILURE).
9. `shrink:` `adapters/fairship.py` carries `z_offset_m`/`z_scale` transform parameters (lines ~128-160, 384). Greenfield: the state carries its plane and geometry id; no free offset parameter.
10. `delete:` docs: 34 files under `docs/` + 9 notes, many superseded (`fairship_adapter_contract_v0`, `ml_skeleton_local_pkl_v0`). Keep as history, do not copy.
11. `shrink:` two hashing helpers with different jobs (checked): `entities/identifiers.py` (sha256 of canonical JSON) and `data_contracts/hashing.py` (array/file hashing). Not a duplicate; keep one of each in the shared contract.
12. `native:` `mlflow` tracking extra: the local file store is the canonical one. Drop the extra.

net: roughly -6k to -9k lines of code and 4-5 deps in the carried set, before docs; mostly by not duplicating baselines/stages and not porting the fake backend. (Estimate; not measured.)

## What is already right and should be copied
- `fairship_p2.py`, `fairship_p2_source.py`, `run_stage2_p2.py`, `p2_live_smoke.py` and their tests: facts-only readers, content joins, pinned script, live smoke.
- Status vocabulary (PASS / PHYSICS_REJECTED / NOT_EVALUATED / TECHNICAL_FAILURE) and `TagSubject`.
