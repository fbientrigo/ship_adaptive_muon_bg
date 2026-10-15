# Enrichment vs effective rows (utility tilt)

Working note, 2026-10-15. Use it for the alpha/delta scan (queue item 13) and the presentation.
Code: `density_lab/utility_tilt.py` on `demo/pi-utility-guided-v0` (`pi_tilt`, `h_alpha_delta`, `draw_rows`).
Each claim is labelled by where it comes from: EQUATION, COMPUTED (on data), MEASURED (by sampling), INFERENCE.

## The model

```
h(U)       = [delta + (1 - delta) * U] ** alpha        U in [0, 1], delta > 0 keeps every row possible
pi_nominal = w_i / sum_j w_j
pi_tilt    = w_i * h(U_i) / sum_j w_j * h(U_j)         h multiplies each weight, then normalise
p0         = sum_i pi_nominal_i * B_i                  useful fraction without tilt
pU         = sum_i pi_tilt_i * B_i                     useful fraction with tilt
enrichment = pU / p0
N_eff      = 1 / sum_i pi_i ** 2                       Kish: how many rows the probability is spread over
```

Rows are then drawn with replacement: `numpy.random.default_rng(seed).choice(N, size=T, p=pi)`.

## Worked example (4 rows, delta = 0.1, alpha = 4, hard U)

| row | w | B | U | h(U) | w*h | pi_nominal | pi_tilt |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 1 | 1 | 1 | 1 | 0.25 | 1/1.0003 = 0.9997 |
| 2 | 1 | 0 | 0 | 0.1^4 = 0.0001 | 0.0001 | 0.25 | 0.0001 |
| 3 | 1 | 0 | 0 | 0.0001 | 0.0001 | 0.25 | 0.0001 |
| 4 | 1 | 0 | 0 | 0.0001 | 0.0001 | 0.25 | 0.0001 |

p0 = 0.25, pU = 0.9997, enrichment = 3.999; N_eff goes from 4 (nominal) to ~1.0 (tilted).

## What is exact and what is not

| Claim | Source |
|---|---|
| Ceiling: enrichment <= 1/p0 (B in {0,1} and sum pi = 1, so pU <= 1). Approached when delta**alpha -> 0 | EQUATION |
| Binary U: pU = p0 r+ / (p0 r+ + (1 - p0) r-), with r+ = h(U_pos), r- = h(U_neg). Hard U: pU = p0 / (p0 + (1 - p0) delta**alpha) | EQUATION |
| N_eff = 1 / sum pi^2 | EQUATION (definition) |
| Enrichment and N_eff move together: more enrichment concentrates pi on the B rows | EQUATION (follows from the two above) |
| Martina 40k sample, PDG 13: p0 = 0.0252, ceiling ~40x; delta 0.1 alpha 4 -> 39.5x, N_eff/N 0.0035; delta 0.9 alpha 1 -> 1.1x, N_eff/N 0.062 | COMPUTED |
| The MC weights alone already concentrate: near-nominal N_eff/N = 0.06 | COMPUTED |
| Strong tilt: 200k draws touch only ~1,345 distinct rows (= the B rows) | MEASURED (2026-10-15, 20 seeds) |
| Mild tilt: 12,664 draws touch 3,619 distinct rows of 19,788 | MEASURED |
| The NF then learns from few repeated rows and may memorise them | INFERENCE, not measured |

`U` and `B_toy` here are synthetic (`B_toy = 1[pT > weighted q95, R_xy < weighted q05]`, U_A = 1/0, U_P = 0.99/0.01).
The trade-off is a property of the tilt, not of the physics.

## What to check in the alpha/delta scan

For each (alpha, delta), compare the model's prediction with what is observed:

1. Predicted pU (closed form) vs the draw fraction (sampler check, MC noise only).
2. Predicted pU vs the NF's own B fraction `Q_theta` (does the flow keep the tilt?). Earlier result: strong tilts track pU, mild tilts were inconclusive at 4,000 samples (seed noise larger than the signal).
3. Predicted N_eff and distinct rows drawn vs NF quality outside B (does coverage collapse?).
4. Error bars: binomial SE of `Q_theta` and seed-to-seed spread, before reading any trend.
