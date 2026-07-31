# D9 branch integration and full-data preflight v0

Status: `FULL_DATA_PREFLIGHT_VERIFIED`

This document records the integration decision, portable golden-test hygiene,
and the no-training full-data Table A/Table B preflight for
`experiment/d9-full-data-replay-v0`.

## Branch integration decision

- Current branch: `experiment/d9-full-data-replay-v0`
- Starting commit: `787a7a40bf290f00595b3d6780c04ad56963d4bc`
- Historical branch: `origin/experiment/d9-5n-afterms-nightly-gpu-runner-v0`
  at `aca59a45d5fb9aad6629f3b7834f7e734092814d`
- Utility-tilt parent: `origin/experiment/d9-utility-tilt-arena-v1` at
  `787a7a40bf290f00595b3d6780c04ad56963d4bc`
- Merge base: `c629d035ba426c48fbaa31660e5351dfa26ce663`

The current line is the physical-weight direct-sampling and utility-tilt
line. Its relevant commits are the fixed-composition estimator parent,
fixture-to-full-data replay, direct utility-tilt tables, and the controlled
utility-tilt fixture arena. The historical line is the row-empirical
afterMS model-family arena and multi-day unattended GPU runner.

The whole historical branch was rejected: it adds the row-empirical D9-5
scientific semantics, CUDA/nightly execution and campaign chaining, a large
afterMS runner/reporting surface, and unrelated historical documentation. It
also contains generated `graphify-out/` artifacts. That would cross the
weighted sampling target boundary and is unnecessary for a full-data table
preflight.

No historical commit was cherry-picked. The audit identified no operationally
required commit for full-data loading, checkpointing, long-running execution,
conditional model support, or reusable reporting in this task. The current
line already provides the shared loader/split contracts, table construction,
table hashes/manifests, and the tables-only CLI. Conditional-charge NF work
is intentionally not part of this branch.

## Golden-test hygiene

`tests/test_rare_aware_estimators.py` now has two layers:

1. `test_golden_arm_a_portable_functional_regression` always checks fit status,
   best step, history length, `sum_weights`, train and validation NLL,
   state-dict names and shapes, parameter count, finite state values,
   deterministic repeated local executions, and `log_prob` on a fixed probe
   with tight numerical tolerances.
2. `test_golden_arm_a_reference_state_dict_hash` is marked `reference_golden`
   and preserves the historical exact hash. It is skipped unless
   `SHIP_RUN_REFERENCE_GOLDEN=1` is explicitly set, with a message that exact
   model-byte identity requires the frozen reference environment.

The preserved exact hash is
`afcaa5215fc4c24113ae42a91ef473d2847eaac7e044edb2787fa95258c2d025`.
The local deterministic hash is
`710a07ed14b57dafefeeb3c3ff5838141eca2b3ba3df8a21637ed3ca9978f5d9` and was
not substituted for the reference hash.

Commands:

```text
pytest tests/ -q
pytest -m reference_golden -q
```

The exact frozen reference environment fingerprint is `OPEN`; the repository
does not contain enough evidence to reconstruct it. The live runner used for
this preflight reported Windows, Python 3.12.10, NumPy 2.5.0, PyTorch
2.13.0+cpu, and MKL available. CUDA was not available to this interpreter.

## Full-data validation and resources

The local source was the trusted gzip-PKL afterMS dataset supplied to the
preflight. The validate-only command completed in 47 seconds and wrote the
report under the isolated artifact root:

```text
python scripts/build_dataset_report.py --dataset <local-full-afterMS.pkl> --validate-only --allow-zero-weight --seed 1234 --output artifacts/density_lab/d9_full_data_replay_v0/full_dataset_validation.json
```

Validation results:

- file SHA-256: `FB251BE2CBFB04AED52BEE215C23CBD0898ECCAE989523952D3B774C004F974C`
- canonical dataset hash: `44336e8e3629149c813026cf21334c6fc2db75ae9ffa12e78ed2b064f3a54579`
- rows/columns: `13,779,080 x 8`
- PDG 13 / -13: `6,854,911` / `6,924,169`
- weights: negative `0`, non-finite `0`, zero `1`, maximum `768.75`
- validation checks: shape, finite, weights-positive-under-allow-zero,
  integer IDs, and unit bounds all passed
- duplicate-row check: not run because the full row count exceeds the bounded
  2,000,000-row check; this is an explicit technical limitation, not a failed
  validation check
- raw float64 array estimate: `881,861,120` bytes (`841.008 MiB`)
- split feasibility: passed with 20% validation and 20% test fractions

Before table construction the machine had approximately 5.33 GiB free RAM
and 23.50 GiB free disk. Table construction was sequential. Peak child RSS
was measured at 2,256,875,520 bytes for PDG 13 and 2,097,127,424 bytes for
PDG -13. No resource-pressure stop was required.

## Table preflight

The command used one layer only and never entered the training path:

```text
python scripts/run_utility_tilt_campaign.py --dataset <local-full-afterMS.pkl> --pdg-ids <13-or--13> --seed 11 --tilt-ids UA_d0p9_a04 --build-tables --tables-only --artifact-root artifacts/density_lab/d9_full_data_replay_v0
```

Both runs used source hash
`44336e8e3629149c813026cf21334c6fc2db75ae9ffa12e78ed2b064f3a54579`,
train-only weighted thresholds, and the same deterministic three-way split
contract. `pi_nominal` and the one `pi_tilt` block each normalized to `1.0`.

| PDG | train / validation / test | split hash | train-only `t_pT`, `t_R` | nominal `p(B_toy)` | nominal `N_eff` / `N_eff/n_train` | top-10 / top-100 mass | zero-probability fraction | expected unique rows (`T=n_train`) | Table A hash | Table B hash | runtime / peak RSS |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---|---:|
| 13 | 4,387,143 / 1,096,786 / 1,370,982 | `53cd73d8...aeb7d4` | 3.8532100703, 1.4810009099 | 0.0245097040 | 270,729.8944 / 0.0617 | 0.0000562541 / 0.0005625413 | 0.0 | 1,100,650.3530 | `4e783529...3713e` | `2f062ba2...cb094` | 58.039 s / 2.257 GB |
| -13 | 4,431,468 / 1,107,867 / 1,384,834 | `ee7e52e2...ccfa5c` | 4.0466979247, 1.4615513325 | 0.0265183421 | 314,495.3294 / 0.0710 | 0.0000460143 / 0.0004601426 | 0.0 | 1,009,216.3437 | `07e34e9b...574c8` | `9a392f14...fedbe` | 57.428 s / 2.097 GB |

The full hashes are in the generated manifests under
`artifacts/density_lab/d9_full_data_replay_v0/d9_utility_tilt_v0/tables/`.
Table A artifact sizes were 82,737,067 bytes (PDG 13) and 83,654,892 bytes
(PDG -13). The one-layer Table B sizes were 9,504,300 bytes and 9,654,442
bytes, respectively; each manifest was below 1.2 KiB.

For the requested `UA_d0p9_a04` Table B layer, the additional concentration
was small and is reported separately from the physical `w_mc` concentration:

- PDG 13: `N_eff=272,814.1142`, top-10 `0.0000846526`, top-100
  `0.0008465264`, expected unique `1,115,563.0560`.
- PDG -13: `N_eff=315,298.6873`, top-10 `0.0000691715`, top-100
  `0.0006917153`, expected unique `1,020,229.9775`.

Low `N_eff` is retained as a concentration diagnostic, not an automatic
failure. Expected row reuse is a consequence of the sampling law and is not
labelled OOD.

## Status boundaries

- `VERIFIED`: branch starting state, historical/utility branch identities,
  no-merge decision, dataset hashes and validation, sequential Table A plus
  one-layer Table B for both PDGs, normalization, hashes, and resource probe.
- `PROJECT DECISION`: no historical cherry-pick; no model training, no
  FairShip, no test-split model evaluation, and no all-20 Table B materialization.
- `PROVISIONAL`: concentration and expected-unique values are diagnostic
  summaries only; duplicate-source-event identity is not represented by this
  row schema.
- `OPEN`: frozen reference-environment fingerprint, conditional-charge NF
  implementation, full-data training, and any broader scientific
  interpretation of `B_toy`.
