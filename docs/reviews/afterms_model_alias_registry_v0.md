# D9C — Model and track alias registry

## 1. Why this exists

Candidate ids like `A1`, `A2`, `B1`, `B2`, `C1`, `D1` are legacy internal
identifiers assigned during D9 candidate selection (`configs/afterms/d9_candidate_plan_v0.json`).
They encode a historical *selection decision* (which D8 arena run became a D9
champion/challenger), not a model family or an architecture, and they are
**not renamed or removed** by this change — every internal id, run directory,
checkpoint path, and hash remains exactly as it was. This registry adds a
presentation/lineage layer on top of them so a reviewer can read "affine
coupling, 8 blocks, width 128, depth 2, on PDG +13 unweighted identity
preprocessing" instead of "A1".

## 2. Concepts

- **`track_id`** — which empirical distribution/estimand is being modeled.
  Derived only from `pdg_value` + `weighting_policy` + `preprocessing_name`
  (never from a legacy candidate id). Two runs with the same three fields
  always resolve to the same track.
- **`model_family_id`** — which generative model family: `NF_AC`
  (affine-coupling normalizing flow), `GAUSS_DIAG`, `GAUSS_FULL`, or `GMM`.
  Flow Matching / CNF / diffusion families are **not** registered yet
  (`configs/afterms/model_alias_registry_v0.json`'s `not_yet_registered_model_families`).
- **`model_config_id`** — which concrete architecture, e.g. `NF_AC_b08_w128_d02`
  (8 blocks, hidden width 128, depth 2). Always derived from the actual
  architecture values a run used — never from a capacity label
  (`"medium"`/`"small"`) or a capacity-scale multiplier.
- **`run_id`** — unchanged: track + model configuration + seed + training
  policy, exactly as `d9/runner.run_id_for` already computes it.

Note: the candidate plan already has an unrelated field literally named
`track_id` (e.g. `"A_row_empirical_unweighted_pdg13"`), which groups by
`pdg_value` + `weighting_policy` only. The new `TRK_*` ids introduced here are
strictly finer — they also split by `preprocessing_name` — because two D9
candidates on that same legacy track (e.g. A1/A2) use different
preprocessing and are not the same empirical target for ranking purposes.
The two notions are deliberately kept distinct; do not conflate them.

## 3. Where the code lives

- `src/ship_muon_bg/afterms/model_naming.py` — pure resolver functions
  (`resolve_track`, `nf_ac_model_config`, `gauss_model_config`,
  `gmm_model_config`, `build_alias_record`, `assert_no_model_config_collisions`,
  and the two `resolve_d9_training_candidate_alias`/`resolve_scout_variant_alias`
  convenience wrappers).
- `configs/afterms/model_alias_registry_v0.json` — versioned lookup tables
  (pdg/weighting/preprocessing → short code + label, registered model
  families). Curated values get short codes (`ID`, `LOGPZ`, `UW`, `W`,
  `PDG13`, `PDGM13`); anything not yet curated (e.g. `quantile_normal_v0`)
  falls back to a deterministic slug of the raw field, never a fabricated
  abbreviation, and the record's `alias_resolution_status` is set to
  `GENERIC_FALLBACK` so callers can tell curated from best-effort aliases.

## 4. Hash and resume safety

`d9/contract.py`'s `semantic_training_hash` hashes the identity-relevant
subset of `candidate_config` **wholesale**
(`d9/runner.py::_identity_relevant_config`). Anything added to that dict
would silently change resume/checkpoint compatibility. Aliases are therefore
computed **externally**, alongside a config, never inside one:

- `model_naming.py` never mutates the `candidate_config`/training-config
  dicts it's given — it only reads fields off them.
- The 30-run capacity scout's per-run `training_config.json` snapshots (the
  file `train_candidate_seed` writes next to the checkpoint) are read-only
  evidence for aliasing; nothing writes back into them.
- New alias output only lands under `artifacts/afterms_d9_model_arena_v0/named/`
  (scout relabeling) and `artifacts/afterms_model_alias_inventory_v0/`
  (Gaussian/GMM inventory) — both under `artifacts/`, both gitignored.
- `tests/afterms/d9/test_model_alias_candidates.py` proves this directly:
  it snapshots the identity-relevant config subset before/after alias
  resolution and asserts byte-for-byte equality, recomputes
  `execution_policy_hash` before/after and asserts equality, and reads
  `configs/afterms/d9_training_v0.json` off disk before/after and asserts
  the bytes never changed.
- `d9plan.verify_source_hashes` (already the frozen mechanism the candidate
  plan uses to pin D8 lineage) is re-run in tests here too, confirming this
  change touches none of the D7/D8 frozen artifact hashes.

## 5. The recent 30-run scout

`scripts/run_afterms_d9_model_arena.py` ran 30 completed CUDA runs (6 D9
candidates × 5 capacity variants, seed `20260720`, 10 epochs). This is
**`AFFINE_COUPLING_CAPACITY_SCOUT_V0`** — a per-candidate capacity sweep, not
a model-family arena: every variant trained is `NF_AC`. The candidate ids
(`A1`…`D1`) identify empirical tracks/selection lineage, not model families.

The script gained a read-only `--relabel-existing` flag. It:

1. Resolves the base alias for each of the 6 D9 candidates from the frozen
   `d9_candidate_plan_v0.json` + `d9_training_v0.json`.
2. Discovers each candidate's existing scout variant run directories by glob
   (`<candidate_id>__arena_cap_<scale>x`) — never by recomputing a scale
   multiplier.
3. Reads each variant's own `training_config.json` (written by
   `train_candidate_seed` at training time) for its actual
   `number_of_blocks`/`hidden_width`/`hidden_depth`, and derives
   `model_config_id` from those recorded values.
4. Writes `artifacts/afterms_d9_model_arena_v0/named/{alias_snapshot.json,
   run_inventory_named.{json,csv}, per_track_model_comparison.{md,csv},
   affine_capacity_scout_summary.md}`.

It never renames an existing run directory, never rewrites `arena_manifest.json`
or any per-run `status.json`/`training_config.json`, and never trains (no
`torch` import, no call into `train_candidate_seed`, on this code path).

Within a track, capacity variants are ranked by validation feature-space NLL.
**Across tracks, no single global ranking is produced** — each of the six
tracks models a different empirical target (different PDG sign, different
weighting, or different preprocessing), so their NLLs are not on the same
axis; the champion line for each track is reported independently.

## 6. Mapping table — six enabled D9 candidates

| legacy id | track_id | model_family_id | model_config_id | display_name | evidence source | status |
|---|---|---|---|---|---|---|
| A1_capacity_medium_identity_pdg13_unweighted | TRK_PDG13_UW_ID | NF_AC | NF_AC_b08_w128_d02 | NF_AC_b08_w128_d02 on TRK_PDG13_UW_ID | configs/afterms/d9_training_v0.json | CURATED |
| A2_capacity_small_cartesian_pdg13_unweighted | TRK_PDG13_UW_LOGPZ | NF_AC | NF_AC_b04_w064_d02 | NF_AC_b04_w064_d02 on TRK_PDG13_UW_LOGPZ | configs/afterms/d9_training_v0.json | CURATED |
| B1_capacity_small_cartesian_pdg_minus13_unweighted | TRK_PDGM13_UW_LOGPZ | NF_AC | NF_AC_b04_w064_d02 | NF_AC_b04_w064_d02 on TRK_PDGM13_UW_LOGPZ | configs/afterms/d9_training_v0.json | CURATED |
| B2_capacity_small_identity_pdg_minus13_unweighted | TRK_PDGM13_UW_ID | NF_AC | NF_AC_b04_w064_d02 | NF_AC_b04_w064_d02 on TRK_PDGM13_UW_ID | configs/afterms/d9_training_v0.json | CURATED |
| C1_weighted_small_identity_pdg13 | TRK_PDG13_W_ID | NF_AC | NF_AC_b04_w064_d02 | NF_AC_b04_w064_d02 on TRK_PDG13_W_ID | configs/afterms/d9_training_v0.json | CURATED |
| D1_weighted_small_identity_pdg_minus13 | TRK_PDGM13_W_ID | NF_AC | NF_AC_b04_w064_d02 | NF_AC_b04_w064_d02 on TRK_PDGM13_W_ID | configs/afterms/d9_training_v0.json | CURATED |

Note that A2/B1/B2/C1/D1 share `NF_AC_b04_w064_d02` (same architecture) but
each sits on a distinct track — this is expected: `model_config_id` is
architecture-only, `track_id` is estimand-only, and only their combination
(`display_name`) identifies a specific scientific run.

## 7. D7/D8 Gaussian/GMM controls

`scripts/build_afterms_model_alias_inventory.py` reads the frozen
`artifacts/afterms_d8_evaluation_v0/registry/run_registry.json` (never writes
to it) and resolves:

| legacy id (D8 run_id) | track_id | model_family_id | model_config_id | evidence source | status |
|---|---|---|---|---|---|
| 10_gaussian_controls_pdg13__…_diagonal_gaussian_unweighted | TRK_PDG13_UW_ID | GAUSS_DIAG | GAUSS_DIAG_d05 | D8 run registry `modeled_dimension` | CURATED |
| 10_gaussian_controls_pdg13__…_full_gaussian_unweighted | TRK_PDG13_UW_ID | GAUSS_FULL | GAUSS_FULL_d05 | D8 run registry `modeled_dimension` | CURATED |
| 10_gaussian_controls_pdg13__…_gaussian_mixture_unweighted | TRK_PDG13_UW_ID | GMM | GMM_k04_covFULL_d05 | D8 registry `architecture.n_components=4` + `Nflow/baselines/gmm.py` hardcoded `covariance_type="full"` | CURATED |
| 11_gaussian_controls_pdg_minus13__…_diagonal_gaussian_unweighted | TRK_PDGM13_UW_ID | GAUSS_DIAG | GAUSS_DIAG_d05 | D8 run registry `modeled_dimension` | CURATED |
| 11_gaussian_controls_pdg_minus13__…_full_gaussian_unweighted | TRK_PDGM13_UW_ID | GAUSS_FULL | GAUSS_FULL_d05 | D8 run registry `modeled_dimension` | CURATED |
| 11_gaussian_controls_pdg_minus13__…_gaussian_mixture_unweighted | TRK_PDGM13_UW_ID | GMM | GMM_k04_covFULL_d05 | D8 registry `architecture.n_components=4` + `Nflow/baselines/gmm.py` hardcoded `covariance_type="full"` | CURATED |

`covariance_type="full"` is cited from `Nflow/baselines/gmm.py`
(`GaussianMixtureEstimator.fit` hardcodes it in the `sklearn.mixture.GaussianMixture`
constructor call) — not a configurable field of that estimator, so it cannot
vary across these runs, and citing it is producer-code evidence rather than
an invented value. The D8 registry mislabels `weighting_policy`/`target_measure`
for these six rows (`weighting_policy: false`); job ids 10/11 do not appear in
`legacy_d7_adapter.JOB_WEIGHTED`, so both are asserted unweighted directly
rather than trusting that field.

All six carry forward `reconstruction_status: MISSING_HISTORICAL_CHECKPOINT`
and `checkpoint_path: null` from the frozen D8 registry unchanged — no model
is refit, and no checkpoint is invented. Zero entries were unresolved; if a
future Gaussian/GMM control lacks evidence for `n_components`/`covariance_type`/
`dimension`, `gmm_model_config` reports the explicit
`GMM_LEGACY_CONFIG_UNRESOLVED` alias and records exactly which fields were
missing, rather than guessing.

Output lands under `artifacts/afterms_model_alias_inventory_v0/` (gitignored,
not committed): `existing_model_inventory.{json,csv}`,
`unresolved_aliases.json`, `alias_inventory_report.md`.

## 8. What this does not do

- Flow Matching (`FM_*`), CNF, and diffusion aliases are deliberately not
  registered yet (section 3.5 of the mission scope) — only `NF_AC`, `GAUSS`,
  and `GMM` exist in `model_alias_registry_v0.json`.
- No model is trained, refit, or reconstructed by anything in this change.
- No existing run directory, checkpoint, or internal candidate/run id is
  renamed.
