# NF seed-robustness check

**Question.** Is an observed single-seed NF result (e.g. Figure 05's
per-PDG `generated_b_toy_occupancy`) a real, seed-independent signal, or an
artifact of that one training run's particular initialization/optimization
trajectory ("seed luck")? A single seed cannot answer this — Figure 05
(and the rest of `artifacts/pi_demo/`) uses `seed=11` only. This is a
follow-up diagnostic, not a P0 deliverable: it does not modify any figure,
CSV, or manifest under `artifacts/pi_demo/`.

**What was done.** The same bounded NF arena (`NOMINAL_PHYSICAL` + the 4
named tilts, both PDG tracks) was retrained at seeds `11` (reused from the
existing P0 run), `12`, and `13`, using the unmodified
`scripts/run_utility_tilt_campaign.py --arena-report` CLI — no new training
code, no new tables, same config
(`configs/density_lab/utility_tilt/pi_demo_cpu_v0.json`).

## Result

See `seed_robustness_summary.csv` (`src/ship_muon_bg/density_lab/pi_demo_seed_robustness.py`
computed it; `scripts/run_pi_demo_seed_robustness.py` ran it). Three findings:

1. **Even the untilted `NOMINAL_PHYSICAL` arm undershoots `p0` at every
   seed, for both PDG tracks** (PDG 13: mean `Q=0.0148` vs `p0=0.0259`;
   PDG -13: mean `Q=0.0149` vs `p0=0.0265`). Sampling directly from
   `pi_nominal` should reproduce `p0` in expectation, so this is a
   **systematic rare-region under-coverage of the flow itself at this
   training budget** — not something specific to any one tilt or PDG
   track, and not evidence the tilt mechanism is broken (Figures 01 and 04
   validate the mechanism exactly, independent of any NF).
2. **Mild-tilt seed-to-seed spread is comparable to, or larger than, the
   effect the tilt is supposed to produce.** The mild tilts move the exact
   target from `p0≈0.026` to `pU≈0.029` — a ~0.003 shift — while the
   3-seed standard deviation of `Q_theta` is `0.0046–0.0094`, i.e. **larger
   than the signal being measured**. PDG -13's `UA_d0p9_a01` even flips
   sides of `p0` across seeds (`all_seeds_same_side_of_p0=False`:
   `Q` = 0.0148, 0.0265, 0.0290 across the three seeds). This directly
   confirms the mild-tilt "seed luck" hypothesis from the main README's
   Figure 05 discussion — it is not resolvable with a single seed.
3. **Strong-tilt results are directionally consistent but not tightly
   pinned.** All 5 strong-tilt (pdg, variant) combinations land on the
   correct side of `p0` at every seed, and 3 of them have a tight spread
   (`std<0.04`), but PDG -13's `UP_d0p1_a04` ranges `0.665–0.896` across
   seeds (`std=0.095`) — the single-seed number reported in the main
   Figure 05 (`0.896`) happens to be this run's best case, not
   necessarily representative.

**What this does not change.** The main bundle's mathematical claims
(closed-form/direct-summation/alias-sampler agreement) are exact and
seed-independent by construction — nothing here touches them. What this
diagnostic adds is exactly the caveat the main README's Figure 05 section
already carries in words ("not reliably distinguishable from sampling
noise at this budget"), now backed by actual multi-seed numbers instead of
a single-seed argument.

## Reproduction

```bash
.venv/bin/python scripts/run_pi_demo_seed_robustness.py \
    --dataset data/samples/muonsFullMC_afterMS_sample.npz \
    --pdg-ids 13 -13 --seeds 11 12 13 --sampler-seed 7 \
    --model-config configs/density_lab/utility_tilt/pi_demo_cpu_v0.json \
    --device cpu --reuse-seed 11 --reuse-experiment-id pi_demo_v0
```

Add more seeds (`--seeds 11 12 13 14 15`) for a tighter spread estimate;
each additional seed costs one more full arena run (~1-2 minutes on CPU).
