# Muon NTuples v1.0 — committed representative samples

Small, git-committable subsets of the **Muon NTuples v1.0** full–Monte-Carlo
datasets, so anyone cloning this repo can inspect the data shape, units, and
ranges without downloading the multi-hundred-MB originals. These samples are for
**understanding the data envelope only** — they are not a physics-validity claim
and are not sized for training.

The full datasets stay out of git (see the repo README and
`docs/architecture/ml_skeleton_local_pkl_v0.md` §9: large data out of git; only
hashes, manifests, and small fixtures committed).

## Source datasets (GitHub release assets on `fbientrigo/NFlow`)

| Sample here | Stage | Release | Full file | Full size | Full rows | Full-file sha256 |
| --- | --- | --- | --- | --- | --- | --- |
| `muons_FullMC_sample.*` | Scoring plane (upstream of Muon Shield) | [`v1.0.0-fullmc`](https://github.com/fbientrigo/NFlow/releases/tag/v1.0.0-fullmc) | `muons_FullMC.pkl` | 336,172,545 B | 16,533,515 | `91273fe62e30a2bc5647d60ec51c7212b8659df31eaf670e9fc4a21d232ceb9c` |
| `muonsFullMC_afterMS_sample.*` | After Muon Shield (Muons&Matter propagation) | [`v1.0.0-fullmc-afterms`](https://github.com/fbientrigo/NFlow/releases/tag/v1.0.0-fullmc-afterms) | `muonsFullMC_afterMS.pkl` | 564,941,616 B | 13,779,080 | `fb251be2cbfb04aed52bee215c23cbd0898eccae989523952d3b774c004f974c` |

Each full file is a **gzip-pickled** `numpy.ndarray` of shape `(N, 8)`.

## Column schema (fixed order)

Matches `src/ship_muon_bg/data_contracts/schema.py` exactly.

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

Coordinate note: positions are in **metres**, with `z = 0` at the entrance of
the Muon Shield. In the after-Muon-Shield sample the coordinates are already
shifted to that convention (`z ≈ 28.9 m`, the shield exit / scoring plane there),
while the scoring-plane sample spans `z ≈ [-2.5, 3.9] m` around the entrance.

## Files

```
data/samples/
  muons_FullMC_sample.pkl.gz / .npz          # scoring plane, 40,000 rows
  muonsFullMC_afterMS_sample.pkl.gz / .npz    # after Muon Shield, 40,000 rows
  *_sample_manifest.json                      # provenance (source tag+sha256, seed, hashes)
  reports/
    *_full_dataset_report.json                # ranges/quantiles/id-hist of the FULL datasets
    *_sample_dataset_report.json              # same report computed on the committed sample
```

Two on-disk formats are committed for each sample:

- **`.npz`** — preferred. `numpy.load` does **not** execute code; the array is
  under the key `muons`.
- **`.pkl.gz`** — gzip-pickled, for compatibility with the existing loaders
  (`ship_muon_bg.data_contracts.load_muon_pkl`, `NFlow.utils.data_handling`).
  Pickle executes arbitrary code on load — only load trusted local files.

The `.npz` and `.pkl.gz` copies of a given sample contain the identical array
(checked in `tests/test_samples.py`).

## How the sample was built

`scripts/make_muon_subset.py` → `ship_muon_bg.data_contracts.subsampling.representative_subset`:
a deterministic seeded **uniform core** (so the marginals and quantiles resemble
the full data) **plus range anchors** — the argmin/argmax row of every column.
The anchors make the committed sample span the **same observed envelope** as the
full dataset (the `min`/`max` in the `*_sample_dataset_report.json` match the
`*_full_dataset_report.json`), which a plain uniform sample would understate,
while adding only two rows per column — far too few to distort the quantiles.

Reproduce a sample from the full file:

```bash
python scripts/make_muon_subset.py \
  --input /path/to/muons_FullMC.pkl \
  --output-stem data/samples/muons_FullMC_sample \
  --n-rows 40000 --seed 1234 \
  --source-tag v1.0.0-fullmc \
  --source-sha256 91273fe62e30a2bc5647d60ec51c7212b8659df31eaf670e9fc4a21d232ceb9c
```

Given the same source file, `--seed`, and `--n-rows`, the subset is bit-for-bit
reproducible; the manifest records the resulting `subset_dataset_hash` so it can
be verified.

## Loading

```python
# NPZ (preferred — no code execution on load)
from ship_muon_bg.data_contracts import load_muon_npz
muons = load_muon_npz("data/samples/muons_FullMC_sample.npz")   # (40000, 8) float64

# gzip-PKL (legacy-compatible)
from ship_muon_bg.data_contracts import load_muon_pkl
muons = load_muon_pkl("data/samples/muons_FullMC_sample.pkl.gz")
```

## Validation note

Each full file contains exactly one row with weight `w == 0`. That row is a
per-column extreme (`argmin(w)`) and is therefore included in the sample, so the
samples are reported and validated with `allow_zero_weight=True`
(`validate_muon_array(..., allow_zero_weight=True)`). All other contract checks
(shape, finiteness, integer `id`, units-sanity bounds) pass with defaults.
