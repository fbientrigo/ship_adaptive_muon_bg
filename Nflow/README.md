# Nflow — normalizing-flow proposal module (module 2)

Learns the post-shield muon distribution and proposes new candidate points
`x`, biased toward regions that `ProxyTagger`'s `U(x)` score marks as likely
to produce Deep Inelastic Scattering (DIS).

## Layout

| Path | Status | Contents |
| --- | --- | --- |
| `interfaces.py` | contract | `DensityModel` (fit / log_prob / sample) and `BiasStrategy` (the A/B seam for biasing mechanisms). |
| `legacy/` | quarantined, **untested** | The RealNVP trainer inherited from the `mferril/NFlow` fork: `deepflow.py` entry point, `config.yaml`, and `utils/` (flow model, training loop, HDF5 data handling, plotting, run management). |

## The `BiasStrategy` seam (A/B testing)

*How* `U(x)` should bias the proposal is an open question — data
aggregation, a modified loss, both, or something else. Each candidate
mechanism implements `BiasStrategy` (`resample` for the data side,
`loss_weights` for the loss side) so campaigns can compare strategies under
identical seeds, data, and artifacts. Never hardcode a biasing mechanism
into a model; keep it behind this interface.

## Legacy status and promotion rules

The code under `legacy/` was written for HDF5 mother/daughter data with a
4-feature layout, assumes CUDA, and has zero test coverage. Rules from
`docs/architecture/ml_skeleton_local_pkl_v0.md`:

1. **Do not** import `Nflow.legacy` from tested core code.
2. The RealNVP in `legacy/utils/flow_models.py` is a *candidate* to promote
   into `Nflow/` proper — but only adapted to the `(N, 8)` PKL contract,
   behind `DensityModel`, and with a tiny-overfit smoke test in the same
   commit ("overfit first, then regularize").
3. To run the legacy trainer as-is (needs `pip install -e .[legacy]`, a GPU,
   and an HDF5 dataset):

   ```bash
   cd Nflow/legacy && python deepflow.py   # reads ./config.yaml
   ```

## Next steps for this module

See `docs/architecture/repo_architecture_v1.md` (best-ROI roadmap). Step 1
lives here: adapt the legacy RealNVP to the PKL contract behind
`DensityModel` with the tiny-overfit test.
