# PI-ready utility-guided sampling demonstration

This bundle demonstrates the already-implemented support-preserving
utility-guided proposal:

```
pi_nominal_i = w_i / sum_j w_j                       (nominal direct-sampling law)
h_{alpha,delta}(U_i) = [delta + (1-delta) U_i]^alpha  (utility transform)
pi_tilt_i = w_i h(U_i) / sum_j w_j h(U_j)             (tilted law)
```

implemented in `src/ship_muon_bg/density_lab/utility_tilt.py` (mathematics,
alias sampler, direct-sampling training) and extracted/plotted here by
`src/ship_muon_bg/density_lab/pi_demo.py` and `pi_demo_plots.py`. Nothing in
this bundle reimplements the mathematics — it only reads the existing
tables and arena results and reports them.

**`B_toy` is a project-defined synthetic diagnostic region** (thresholds on
`pT` and `R_xy` fit from the training split), never a validated FairShip
danger metric, never a SHiP background endpoint, never a physical background
rate, never a validated DIS label. `U_A`/`U_P` are synthetic utility scores,
never calibrated probabilities. This bundle validates the *mechanism*, not
any physics claim.

## Dataset

| | |
|---|---|
| Path | `data/samples/muonsFullMC_afterMS_sample.npz` |
| Scope | **sample** — a 40,000-row committed subset of a 13,779,080-row parent file (`muonsFullMC_afterMS.pkl`, not present on this machine; hosted as a GitHub release asset) |
| Hash | `718b9197a3965f6842574fec1f3f6b1bf4c16949ff8a1f27dca6232817c03c32` |
| PDG tracks | `13` (12,664 train rows) and `-13` (12,936 train rows) — **processed and reported independently, never pooled** |
| Seed / sampler seed | `11` / `7` |

Full-data replay is a one-flag change (`--dataset .../muonsFullMC_afterMS.pkl`,
per `docs/d7_full_local_replay_v0.md`'s D9 quickstart) — no code changes are
needed, only more RAM and wall time.

## Numerical validation (from `run_manifest.json`)

- `sum_i pi_nominal_i == 1` for both PDG tracks: **true**
- `sum_i pi_tilt_i == 1` for all 20 tilt configs x 2 PDG tracks: **true, all 40**
- `max |pU_closed_form - pU_direct|` across all 40 rows: **2.22e-16** (machine epsilon)
- No NaN/Inf anywhere, including the most extreme configuration
  (`alpha=16, delta=0.1`, where `r_negative` is ~1e-16)

## Figures

### 01 — `01_tilt_enrichment_vs_alpha`
**Question.** Does the implemented tilt produce exactly the mathematically
predicted distribution?
**What is shown.** For each PDG track, `pU(B_toy)` vs `alpha in {1,2,4,8,16}`,
one line per `(U_A/U_P, delta)` combination, overlaying the closed-form value,
the direct-summation value, and the empirical alias-sampler occupancy, plus a
nominal `p0` reference line.
**Observation.** All three estimates coincide to within `2.22e-16` everywhere
plotted; the empirical alias-draw points show small, expected sampling scatter
at the strongest tilt.
**What it does not prove.** That any tilt is scientifically preferable, or
that `B_toy` reflects a real background process.

### 02 — `02_enrichment_vs_concentration`
**Question.** What is the trade-off between enrichment and sampling
concentration?
**What is shown.** `enrichment = pU_direct/p0` (log scale) vs `N_eff/N`, all 20
tilts x 2 PDG tracks, encoded by mode/delta/alpha.
**Observation.** Enrichment ranges from ~1.1x to ~38x as `N_eff/N` falls from
~0.07 to ~0.003 — increasing exploitation (alpha) can increase enrichment
while concentrating sampling probability onto fewer effective rows.
**What it does not prove.** No ESS threshold is drawn or implied; concentration
here is not labeled "OOD" or "collapse" — this is a descriptive trade-off, not
a pass/fail criterion.

### 03 — `03_support_preservation`
**Question.** Why introduce `delta > 0`?
**What is shown.** (a) The empirical `pi_tilt/pi_nominal` ratio for the
strongest available tilt (`UA_d0p1_a16`), sorted, log scale, both PDG tracks.
(b) `h_alpha_delta(u; alpha=16)` with `delta=0.1` vs a `delta=0` reference.
**Observation.** The ratio stays strictly positive and finite across many
orders of magnitude with `delta=0.1`; the `delta=0` reference curve visibly
touches zero at `u=0`, which `delta=0.1` never does.
**What it does not prove.** Support preservation is relative to the finite
empirical nominal population. It does not prove that the available dataset
covers all physically relevant SHiP regions.

### 04 — `04_direct_sampler_validation`
**Question.** Does the Walker alias sampler actually draw from the declared
`pi_tilt` law?
**What is shown.** For PDG 13, one mild tilt (`UA_d0p9_a01`) and one strong
tilt (`UA_d0p1_a04`): declared cumulative `pi_tilt` mass vs empirical
cumulative mass from alias draws at two draw budgets (`T=12,664` matching the
training convention, and `T=200,000` to separate sampler error from
Monte-Carlo noise), plus target/empirical `B_toy` prevalence, expected vs.
observed unique rows, and max reuse count.
**Observation.** Declared and empirical curves are visually indistinguishable
at both budgets for both tilts; reported prevalences agree to 3 decimal
places.
**What it does not prove.** This validates the sampler implementation only,
not physics. PDG `-13` was intentionally omitted from this figure (a
sampler-implementation check does not need both tracks); it is covered by
every other figure.

### 05 — `05_nf_preserves_tilt`
**Question.** Does a trained normalizing flow preserve the declared tilted
target, rather than reverting toward the nominal population?
**What is shown.** For each PDG track and each of 5 trained variants
(`NOMINAL_PHYSICAL` + the 4 named tilts), three quantities: `P_0` (nominal
prevalence), `P_U` (exact declared target), and `Q_theta` (occupancy measured
from the flow's own generated samples, mapped back to physical space).
**Observation.** `Q_theta` tracks the direction and (roughly) the magnitude of
`P_U` for every trained variant — e.g. PDG 13, `UP_d0p1_a04`: `P_U=0.995` vs
`Q_theta=0.943`; PDG -13, same tilt: `P_U=0.995` vs `Q_theta=0.896`. Mild tilts
stay close to nominal in both `P_U` and `Q_theta`.
**Status.** `NF SUPPORTED` (not blocked). **Bounded CPU-scale affine-coupling
flow** (8 blocks, width 96, <=40 epochs, early stopping) — not a converged
production model. `Q_theta` approaches but does not exactly reach `P_U`, which
is expected at this training budget and is not evidence of a sampler or
mathematics bug (those are validated separately and exactly via the
closed-form/direct-summation/alias-sampler path in Figures 01 and 04).
`final_train_nll` is shown as a small separate annotation and is **not**
equivalent to, or evidence for, sampling agreement — only the occupancy bars
speak to that question.

## Terminology (enforced throughout)

- `w_mc` — physical MC weight.
- `r_utility = h(U)` — a synthetic utility multiplier, **never** called or
  used as a physical weight.
- `pi_tilt` — the tilted sampling probability actually used for training draws.

## Reproduction

```bash
cd worktrees/ship-pi-demo-v0   # branch demo/pi-utility-guided-v0, base a53eec4
.venv/bin/pip install -e ".[dev,lab]"
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu

# 1. Build Table A (nominal) / Table B (20-tilt grid), both PDG tracks.
.venv/bin/python scripts/run_utility_tilt_campaign.py \
    --dataset data/samples/muonsFullMC_afterMS_sample.npz \
    --pdg-ids 13 -13 --seed 11 --sampler-seed 7 \
    --experiment-id pi_demo_v0 --build-tables --tables-only

# 2. Build the summary CSV + run manifest (Figures 01-04 data).
.venv/bin/python -m ship_muon_bg.density_lab.pi_demo

# 3. Train the bounded NF arena (Figure 05 data): NOMINAL_PHYSICAL + 4 tilts x 2 PDG.
.venv/bin/python scripts/run_utility_tilt_campaign.py \
    --dataset data/samples/muonsFullMC_afterMS_sample.npz \
    --pdg-ids 13 -13 --seed 11 --sampler-seed 7 \
    --experiment-id pi_demo_v0 \
    --model-config configs/density_lab/utility_tilt/pi_demo_cpu_v0.json \
    --train-tilt-ids NOMINAL_PHYSICAL UA_d0p9_a01 UA_d0p1_a04 UP_d0p9_a01 UP_d0p1_a04 \
    --device cpu --arena-report

# 4. Render all figures.
.venv/bin/python scripts/run_pi_demo.py --figures 01 02 03 04 05
```

`utility_guided_summary.csv`'s `nf_*` columns are backfilled from step 3's
arena reports for the 4 trained tilts x 2 PDG (8 of 40 rows); the other 16
rows per PDG were never trained and are left empty by design.

## Files

```
artifacts/pi_demo/
├── figures/
│   ├── 01_tilt_enrichment_vs_alpha.{png,pdf}
│   ├── 02_enrichment_vs_concentration.{png,pdf}
│   ├── 03_support_preservation.{png,pdf}
│   ├── 04_direct_sampler_validation.{png,pdf}
│   └── 05_nf_preserves_tilt.{png,pdf}
├── utility_guided_summary.csv    (40 rows: 20 tilt configs x 2 PDG tracks)
├── run_manifest.json             (provenance, seeds, validation gate, NF arena status)
└── README.md                     (this file)
```
