# D9 conditional utility-tilt fixture arena (v0)

**Scientific scope.** This fixture arena tests whether one shared charge-conditional flow trained from a utility-tilted empirical proposal reproduces the declared tilted target for both charges while retaining measurable nominal coverage. It is not a physical-rate estimate, does not treat B_toy as a physical endpoint, does not treat U_A as a calibrated acceptance probability, does not claim FairShip acceptance, and declares no universal winner. FairShip/GEANT4 remains the final physical oracle.

- Completion: `CONDITIONAL_UTILITY_ARENA_VERIFIED` (12/12 runs)
- Charge convention: PDG 13 (mu-) -> -1, PDG -13 (mu+) -> +1
- Loss: ordinary unweighted NLL after direct sampling (`sample_weight_applied_to_loss = false`, `utility_weight_applied_to_loss = false`)
- Preprocessing: one fixed nominal macro-weighted train-only transform per (split, model seed), shared by every variant
- Test payload: never materialized; never used for thresholds

| variant | seed | PDG | nominal val NLL | tilted val NLL | nominal p0(B) | tilted pU(B) | generated B | gen/nom | gen/tilt | gen SE | N_eff | cum unique | max reuse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NOMINAL_PHYSICAL | 11 | 13 | 5.7752 | 5.7752 | 0.0241853 | 0.0241853 | 0.0217285 | 0.898417 | 0.898417 | 0.00161083 | 630.264 | 9661 | 22 |
| NOMINAL_PHYSICAL | 11 | -13 | 6.24658 | 6.24658 | 0.0263835 | 0.0263835 | 0.0349121 | 1.32326 | 1.32326 | 0.00202804 | 742.402 | 9290 | 21 |
| NOMINAL_PHYSICAL | 12 | 13 | 5.26592 | 5.26592 | 0.0235175 | 0.0235175 | 0.012085 | 0.51387 | 0.51387 | 0.00120722 | 622.007 | 8418 | 25 |
| NOMINAL_PHYSICAL | 12 | -13 | 5.18392 | 5.18392 | 0.02143 | 0.02143 | 0.0205078 | 0.956967 | 0.956967 | 0.0015659 | 714.285 | 7935 | 19 |
| NOMINAL_PHYSICAL | 13 | 13 | 4.99374 | 4.99374 | 0.0270605 | 0.0270605 | 0.019165 | 0.708229 | 0.708229 | 0.00151481 | 638.018 | 8724 | 22 |
| NOMINAL_PHYSICAL | 13 | -13 | 4.71852 | 4.71852 | 0.0275399 | 0.0275399 | 0.0107422 | 0.390059 | 0.390059 | 0.00113895 | 745.78 | 8053 | 20 |
| UA_d0p9_a04 | 11 | 13 | 5.49575 | 5.54569 | 0.0241853 | 0.0364008 | 0.0218506 | 0.903464 | 0.600277 | 0.00161525 | 638.813 | 9686 | 23 |
| UA_d0p9_a04 | 11 | -13 | 6.18087 | 6.20278 | 0.0263835 | 0.0396641 | 0.0256348 | 0.971621 | 0.646297 | 0.00174615 | 744.066 | 9331 | 22 |
| UA_d0p9_a04 | 12 | 13 | 5.24475 | 5.28546 | 0.0235175 | 0.035408 | 0.0281982 | 1.19903 | 0.796381 | 0.00182896 | 627.132 | 8553 | 28 |
| UA_d0p9_a04 | 12 | -13 | 4.96382 | 5.04588 | 0.02143 | 0.0322999 | 0.0275879 | 1.28735 | 0.854117 | 0.00180963 | 719.652 | 7970 | 25 |
| UA_d0p9_a04 | 13 | 13 | 4.8697 | 4.92249 | 0.0270605 | 0.0406677 | 0.0380859 | 1.40744 | 0.936517 | 0.00211473 | 645.529 | 9696 | 24 |
| UA_d0p9_a04 | 13 | -13 | 4.6708 | 4.73863 | 0.0275399 | 0.0413779 | 0.0202637 | 0.735793 | 0.489723 | 0.00155675 | 748.023 | 9285 | 21 |
| UA_d0p9_a08 | 11 | 13 | 5.53692 | 5.64599 | 0.0241853 | 0.0544419 | 0.045166 | 1.8675 | 0.829619 | 0.00229443 | 645.939 | 9030 | 38 |
| UA_d0p9_a08 | 11 | -13 | 5.77633 | 5.84022 | 0.0263835 | 0.0592231 | 0.0396729 | 1.5037 | 0.669888 | 0.00215656 | 732.906 | 8440 | 29 |
| UA_d0p9_a08 | 12 | 13 | 5.21313 | 5.27832 | 0.0235175 | 0.052984 | 0.0527344 | 2.24234 | 0.995289 | 0.00246938 | 627.077 | 8823 | 35 |
| UA_d0p9_a08 | 12 | -13 | 5.17236 | 5.34509 | 0.02143 | 0.0484106 | 0.0395508 | 1.84558 | 0.816987 | 0.00215338 | 719.579 | 8231 | 31 |
| UA_d0p9_a08 | 13 | 13 | 4.91285 | 5.02804 | 0.0270605 | 0.0606902 | 0.0532227 | 1.9668 | 0.876956 | 0.00248015 | 648.732 | 8885 | 38 |
| UA_d0p9_a08 | 13 | -13 | 4.50965 | 4.65846 | 0.0275399 | 0.0617276 | 0.0202637 | 0.735793 | 0.328276 | 0.00155675 | 737.422 | 8225 | 32 |
| UA_d0p1_a01 | 11 | 13 | 5.53404 | 5.85859 | 0.0241853 | 0.19862 | 0.157959 | 6.53119 | 0.795282 | 0.00402943 | 496.511 | 9308 | 115 |
| UA_d0p1_a01 | 11 | -13 | 6.2914 | 6.18874 | 0.0263835 | 0.213208 | 0.112427 | 4.26125 | 0.527309 | 0.00349013 | 390.402 | 8811 | 87 |
| UA_d0p1_a01 | 12 | 13 | 5.82453 | 5.90039 | 0.0235175 | 0.194094 | 0.140259 | 5.96401 | 0.722634 | 0.00383667 | 409.585 | 8009 | 108 |
| UA_d0p1_a01 | 12 | -13 | 5.03863 | 5.5667 | 0.02143 | 0.179651 | 0.128174 | 5.98104 | 0.713461 | 0.00369334 | 479.477 | 7474 | 92 |
| UA_d0p1_a01 | 13 | 13 | 5.09723 | 5.39331 | 0.0270605 | 0.217608 | 0.160889 | 5.94552 | 0.739351 | 0.00405955 | 440.022 | 8379 | 96 |
| UA_d0p1_a01 | 13 | -13 | 4.73504 | 5.19472 | 0.0275399 | 0.220697 | 0.13562 | 4.92449 | 0.614508 | 0.00378285 | 394.333 | 7683 | 81 |

Concentration diagnostics (N_eff, top-k mass, uniqueness, reuse) are diagnostic only: no ESS threshold and no automatic rejection criterion is defined. Generated occupancy must be read together with its binomial standard error and across seeds.
