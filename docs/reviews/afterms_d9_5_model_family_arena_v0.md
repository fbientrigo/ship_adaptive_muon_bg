# D9-5: Modern Same-Track Generative Model Family Arena

Status: **Gate A complete** (data-scope audit + common model-adapter contract +
Gaussian/GMM/NF_AC adapters + evaluation/aggregation/report layer + CLI +
tests). Gates B-E (actual fitting/training/evaluation on the full declared
scope) are tracked separately -- see
`docs/reviews/afterms_d9_5_execution_runbook_v0.md` for exact resume commands
and current status.

Branch: `experiment/d9-5-afterms-model-family-arena-v0`, forked from D9C HEAD
`22d38b5`.

## 1. Scientific question

Under the same empirical after-MS target, the same five-dimensional state,
the same data splits and the same evaluation protocol, how do affine-coupling
NF, diagonal Gaussian, full Gaussian and a four-component full-covariance GMM
compare? This is the first modern, direct, same-track comparison across these
four families in this repository. It is explicitly **not** a FairShip
downstream-signal comparison -- see Sec 10 (Limitations) below.

## 2. Data source and five-dimensional state

Source dataset: `data/shards/afterms_nightly_v1` (dataset hash
`44336e8e3629149c813026cf21334c6fc2db75ae9ffa12e78ed2b064f3a54579`), the same
frozen shard set D9/D9B/D9C already use. Raw rows are
`(px, py, pz, x, y, z, id, w)`; the modeled state is exactly

    px, py, pz, x, y

(dimension 5). `id` (PDG code) selects the track; `w` (source weight) is
recorded as metadata only and never enters a feature vector, a loss, or a
sampling weight anywhere in D9-5.

## 3. Track definitions

Two independent, never-pooled, never-cross-ranked tracks:

- `TRK_PDG13_UW_ID` -- PDG +13, row-empirical-unweighted, identity-standardized.
- `TRK_PDGM13_UW_ID` -- PDG -13, row-empirical-unweighted, identity-standardized.

Both resolve through `ship_muon_bg.afterms.model_naming.resolve_track` against
`configs/afterms/model_alias_registry_v0.json` -- the same alias machinery
D9C introduced; D9-5 adds no new alias codes. log1p-pz and production-weighted
tracks are out of scope (weighting interpretation remains OPEN, tracked
outside D9-5).

## 4. Exact rows/shards used

Declared policy: `configs/afterms/d9_5_data_scope_v0.json` (`shard_scope:
"complete_declared_scope"` -- all 22 train / 3 validation / 3 test shards, no
row subsampling, no silent negative-pz dropping). Resolved evidence: the audit
run in `artifacts/afterms_d9_5_model_family_arena_v0/data_scope/` records,
per split and track, the concrete shard names/hashes, raw and PDG-filtered
row counts, per-feature finite-value counts, negative-pz counts and a content
hash of the filtered scope.

Resolved row counts from the frozen audit (`data_scope_manifest.json`):

| track | train rows | validation rows | test rows |
|---|---|---|---|
| TRK_PDG13_UW_ID | 5,484,126 | 685,646 | 685,139 |
| TRK_PDGM13_UW_ID | 5,539,934 | 691,239 | 692,996 |

No negative-`pz` rows exist anywhere in this dataset (0 across every
split/track); the negative-pz-preservation machinery in D9-5's pipeline is
therefore inert on this data but remains active/tested, since it is a
declared invariant, not an assumption about this one dataset.

A host-RAM preflight (`ram_preflight.json`, Sec 6.1 of the mission) confirms
the complete declared scope fits comfortably in host RAM for every family
(peak estimates in the few-hundred-MB range per track); **no chunked/streaming
GMM implementation was required and `D9_5_BLOCKED_BY_GMM_DATA_SCALE` does not
apply**.

## 5. Preprocessing

`identity_standardized_v0` for every family (mean/std of the training split's
5 features, fit once per track on the full training scope). Physical-space
NLL is the primary cross-family likelihood metric:

    log p_x(x) = log p_z(T(x)) + log|det J_T|,  log|det J_T| = -sum(log(std))

Every adapter's `test_log_prob`/`validation_log_prob` returns **feature-space**
log p(z) only; the shared evaluation layer (`d9_5/evaluation.py`) adds the
Jacobian once, so no family can silently double-count or drop that constant.
This is verified directly: `tests/afterms/d9_5/test_gaussian_adapter.py::
test_gaussian_physical_nll_matches_independent_raw_space_calculation` checks
that (standardized-space log p(z) + Jacobian) exactly matches an
independently, directly computed raw-space Gaussian NLL on the same rows.

## 6. Model configurations

| model_config_id | family | notes |
|---|---|---|
| `NF_AC_...` (per track) | affine-coupling NF | derived from the named `AFFINE_COUPLING_CAPACITY_SCOUT_V0` report, never hardcoded -- see Sec 7 |
| `GAUSS_DIAG_d05` | diagonal Gaussian | deterministic single fit, two-pass streaming sufficient statistics |
| `GAUSS_FULL_d05` | full-covariance Gaussian | deterministic single fit, two-pass streaming sufficient statistics |
| `GMM_k04_covFULL_d05` | 4-component full-covariance GMM | 3 seeds (20260720/21/22), sklearn EM driven step-by-step to record a genuine lower-bound curve |

Exact settings (regularization, `n_init`, `max_iter`, `tol`, seeds) are frozen
in `configs/afterms/d9_5_model_family_arena_v0.json`.

## 7. NF_AC configuration selection (scout-promoted)

`ship_muon_bg.afterms.d9_5.nf_ac_adapter.select_scout_promoted_nf_config`
resolves the architecture per track from
`artifacts/afterms_d9_model_arena_v0/named/run_inventory_named.json` (the
named `AFFINE_COUPLING_CAPACITY_SCOUT_V0` report): `argmin(best_validation_metric)`
over that track's `status == "completed"` capacity variants. Confirmed
resolution:

- `TRK_PDG13_UW_ID` -> `NF_AC_b08_w128_d02` (scout val NLL 1.3992097402270696, run `A1_capacity_medium_identity_pdg13_unweighted__arena_cap_1.00x`)
- `TRK_PDGM13_UW_ID` -> `NF_AC_b06_w096_d02` (scout val NLL 1.4511939737719584, run `B2_capacity_small_identity_pdg_minus13_unweighted__arena_cap_1.50x`)

The scout used a single seed (20260720) and 10 epochs; it selected
**architecture capacity only**, on a single training shard, not a converged
final model. D9-5 labels this `scout_promoted_nf_config` (never "final
champion") and trains three new seeds from initialization on the full
declared scope for each track -- the scout's own checkpoints are never
reused (structurally: D9-5's run directories live entirely under
`artifacts/afterms_d9_5_model_family_arena_v0/`, isolated from the scout's
`artifacts/afterms_d9_model_arena_v0/` tree).

The alias/label string `scout_promoted_nf_config` is presentation metadata
only -- it is never folded into the hashed `candidate_config` that feeds
`semantic_training_hash` (see `nf_ac_adapter.py`'s `_build_candidate_config`
and required test 25).

## 8. Fitting/training policies

- **Gaussian controls**: deterministic single fit, two-pass chunk-bounded
  streaming sufficient statistics (verified algebraically identical to direct
  `mean`/covariance computation on the same rows -- required tests 13/14).
- **GMM**: 3 deterministic seeds (20260720/21/22), `n_components=4`,
  `covariance_type="full"`, `reg_covar=1e-6`, `n_init=1`, `max_iter=200`,
  `tol=1e-3` -- every setting frozen in
  `configs/afterms/d9_5_model_family_arena_v0.json`, never silently changed.
  Non-convergence is recorded, never hidden (`converged: false` +
  a warning), and is not automatically interpreted as poor physics.
- **NF_AC**: execution policy frozen *before* the first seed of either track
  trains: `minimum_epochs=20`, `maximum_epochs=100`,
  `early_stopping_patience=25` (see the config file's `justification` field
  for the reasoning and Sec 11.1 bounds). Three seeds (20260720/21/22) trained
  from initialization, sequentially, one CUDA process at a time, float32, no
  AMP, `num_workers=0` (Windows / RTX 2060 Laptop, 6 GB VRAM).

## 9. Likelihood semantics, validation-only selection, frozen test evaluation

All four families share one preprocessing contract and one Jacobian
convention (Sec 5). Validation-only selection is a hard gate:
`ship_muon_bg.afterms.d9_5.report.require_frozen_selection` raises unless
`family_selection_manifest.json` already exists (written only by
`freeze-selection --execute`); `evaluate-test` calls this before reading any
test shard (required tests 27/28). The primary validation metric is
physical-space mean NLL, explicitly stated as primary-but-not-only
(`report.build_family_selection_manifest`'s `primary_metric_statement`).
Ranking is always within-track; no code path produces or implies a global
cross-track ranking (required test 31).

## 10. Limitations

- This is a pre-FairShip, pre-GEANT4, proxy-free comparison of generative
  fits to an empirical after-MS distribution. It says nothing about which
  family better reproduces the downstream DIS-relevant subset, and no D9-5
  output should be read as a FairShip readiness signal.
- The NF_AC architecture is scout-promoted (single seed, 10 epochs, one
  shard) -- a capacity choice with real evidence behind it, but a much
  smaller evidence base than the full D9-5 multi-seed campaign that trains
  on top of it.
- GMM `k=4` and Gaussian model classes are fixed, not searched; this is a
  same-track comparison of these four specific configurations, not a
  hyperparameter search over model families.
- Source weight `w`'s physical interpretation remains an open question
  outside D9-5's scope; production-weighted tracks are excluded entirely.
- PDG +13 and PDG -13 are two separate empirical targets and are never
  ranked against each other.
