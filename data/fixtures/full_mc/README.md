# Full-MC muon fixture datasets

This directory is reserved for two small development fixtures derived from the `fbientrigo/NFlow` full-MC release assets:

| Fixture | Source release | Intended meaning |
| --- | --- | --- |
| `after_muon_shield.pkl.gz` | `fbientrigo/NFlow@v1.0.0-fullmc-afterms` | Muon NTuple sample after the muon shield. |
| `scoring_plane.pkl.gz` | `fbientrigo/NFlow@v1.0.0-fullmc` | Muon NTuple sample at the scoring plane. |

The files are capped at 40 MiB each by `scripts/sample_muon_contract_fixture.py` and are intended only for:

- data-loader tests;
- schema and range inspection;
- support-audit development;
- ML input plumbing and agent context.

They are **not** valid for physics-rate estimates, background-yield estimates, or final SHiP sensitivity studies.

## Contract

Both fixtures use the repository v0 muon contract:

```text
[px, py, pz, x, y, z, id, w]
```

with units:

```text
px, py, pz: GeV/c
x, y, z: m
id: PDG code
w: dimensionless
```

The workflow default applies `--position-scale 0.01`, converting legacy centimetre coordinates to metres. It applies no z-origin shift by default (`--z-shift-m 0.0`) because the geometry/origin convention must be explicit before changing frames.

## Generated metadata

The generation workflow also writes:

```text
full_mc_fixtures_manifest.json
after_muon_shield_contract/dataset_report.json
after_muon_shield_contract/split_manifest.json
after_muon_shield_contract/normalization.json
scoring_plane_contract/dataset_report.json
scoring_plane_contract/split_manifest.json
scoring_plane_contract/normalization.json
```

These metadata files record source paths, dataset hashes, column statistics, validation results, deterministic splits, and train-only normalization parameters.

## Regeneration

The branch `data/full-mc-small-fixtures` contains a manual/push-triggered workflow:

```text
.github/workflows/build_full_mc_fixtures.yml
```

The workflow downloads the exact NFlow release tags, builds the capped fixtures, validates the repo tests, and commits generated files back to the same branch.
