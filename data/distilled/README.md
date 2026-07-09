# Muon NTuples v1.0 — distilled datasets

Committable **distilled** copies of the two Muon NTuples v1.0 full–Monte-Carlo
releases, one per release, so you can code on the cloud or any PC without
downloading the multi-hundred-MB originals. "Distilled" means **the same
distribution with far fewer rows** — bimodality, ranges, means, and std are
preserved and verified by a two-sample hypothesis test against the corresponding
full release (each distilled file is compared only to its *own* full dataset —
after-shield with after-shield, scoring-plane with scoring-plane, never mixed).

These are working datasets for exploring data shape and behaviour, not a
physics-validity claim. The full releases stay out of git (repo README;
`docs/architecture/ml_skeleton_local_pkl_v0.md` §9).

## Datasets

| File | Stage | Release | Distilled rows | Full rows | % of full | Size |
| --- | --- | --- | --- | --- | --- | --- |
| `muons_FullMC_distilled.npz` | Scoring plane (upstream of Muon Shield) | [`v1.0.0-fullmc`](https://github.com/fbientrigo/NFlow/releases/tag/v1.0.0-fullmc) | 1,700,000 | 16,533,515 | 10.3% | ~39 MB |
| `muonsFullMC_afterMS_distilled.npz` | After Muon Shield (Muons&Matter propagation) | [`v1.0.0-fullmc-afterms`](https://github.com/fbientrigo/NFlow/releases/tag/v1.0.0-fullmc-afterms) | 1,000,000 | 13,779,080 | 7.3% | ~40 MB |

Full-file sha256 (verified at distill time):
`muons_FullMC.pkl` → `91273fe62e30a2bc5647d60ec51c7212b8659df31eaf670e9fc4a21d232ceb9c`;
`muonsFullMC_afterMS.pkl` → `fb251be2cbfb04aed52bee215c23cbd0898eccae989523952d3b774c004f974c`.

## Column schema (fixed order)

Matches `src/ship_muon_bg/data_contracts/schema.py`.

| Index | Name | Unit | Meaning |
| --- | --- | --- | --- |
| 0 | `px` | GeV/c | momentum x |
| 1 | `py` | GeV/c | momentum y |
| 2 | `pz` | GeV/c | momentum z |
| 3 | `x` | m | position x |
| 4 | `y` | m | position y |
| 5 | `z` | m | position z |
| 6 | `id` | PDG code (int-valued) | particle id (`±13` = muon) |
| 7 | `w` | dimensionless | event weight |

Positions are in **metres**, `z = 0` at the entrance of the Muon Shield. The
after-shield file is already in that convention (`z ≈ 28.9 m`); the scoring-plane
file spans `z ≈ [-2.5, 3.9] m` around the entrance.

## Files

```
data/distilled/
  muons_FullMC_distilled.npz                         # scoring plane, array key "muons"
  muonsFullMC_afterMS_distilled.npz                  # after shield
  *_distilled_manifest.json                          # provenance: source tag+sha256, seed, hashes, fraction
  *_distilled_distribution_report.json               # full-vs-distilled KS + moments + bimodality
  reports/
    *_full_report.json                               # full-dataset ranges/quantiles/id-histogram
```

## Loading

```python
from ship_muon_bg.data_contracts import load_muon_npz
muons = load_muon_npz("data/distilled/muons_FullMC_distilled.npz")   # (1_700_000, 8) float64
# or with plain numpy:  numpy.load(path)["muons"]
```

`.npz` is used because `numpy.load` does not execute code (unlike pickle). Arrays
are stored under the key `muons`.

## How the distill is built

`scripts/make_distilled_dataset.py` → `representative_subset` (in
`ship_muon_bg.data_contracts.subsampling`): a deterministic seeded **uniform
random core** — an unbiased sample, so moments/quantiles/bimodality match the full
within sampling noise — plus **per-column argmin/argmax range anchors** so the
distilled min/max equal the full dataset's exactly (16 rows, zero quantile
distortion). Given the same source file, `--seed`, and `--n-rows`, the output is
bit-for-bit reproducible; the manifest records the resulting `distilled_dataset_hash`.

Reproduce:

```bash
python scripts/make_distilled_dataset.py \
  --input /path/to/muons_FullMC.pkl \
  --output-stem data/distilled/muons_FullMC_distilled \
  --n-rows 1700000 --seed 1234 \
  --source-tag v1.0.0-fullmc \
  --source-sha256 91273fe62e30a2bc5647d60ec51c7212b8659df31eaf670e9fc4a21d232ceb9c
```

## Fidelity (distilled vs full)

Two-sample **Kolmogorov–Smirnov** of the distilled set against the full
complement (`full \ distilled` — two independent uniform samples of the same law),
plus a moment/bimodality comparison, from the `*_distribution_report.json` files.
For identically distributed samples of these sizes the expected KS `D` is ≈ 0.001,
and that is what we see — the distilled sets are statistically indistinguishable
from the full data on every kinematic column, with the **bimodality coefficient
(Sarle's BC) preserved**, including the strongly bimodal after-shield `x`.

**Scoring plane** (KS D · BC full → distilled):

| col | KS D | mean full → dist | std full → dist | BC full → dist |
| --- | --- | --- | --- | --- |
| px | 0.00086 | −4.8e-5 → 7.4e-5 | 0.3681 → 0.3686 | 0.206 → 0.179 |
| pz | 0.00104 | 45.35 → 45.40 | 31.02 → 31.08 | 0.633 → 0.633 |
| z  | 0.00064 | −2.120 → −2.120 | 0.2079 → 0.2079 | 0.449 → 0.404 |

**After Muon Shield**:

| col | KS D | mean full → dist | std full → dist | BC full → dist |
| --- | --- | --- | --- | --- |
| px | 0.00083 | −0.0104 → −0.0124 | 2.525 → 2.525 | 0.219 → 0.219 |
| pz | 0.00059 | 24.34 → 24.34 | 25.30 → 25.31 | 0.606 → 0.606 |
| **x** | 0.00071 | −0.0107 → −0.0126 | 2.421 → 2.421 | **0.675 → 0.675** (bimodal) |

Worst KS `D` across all six kinematic columns is 0.00104 (scoring) / 0.00091
(after-shield). Full per-column numbers are in the distribution reports.

## Validation note

Each full file contains exactly one `w == 0` row; it is a per-column extreme
(`argmin(w)`), so it is kept as a range anchor and the distilled files are
validated with `allow_zero_weight=True`. All other contract checks (shape,
finiteness, integer `id`, units-sanity bounds) pass with defaults.
