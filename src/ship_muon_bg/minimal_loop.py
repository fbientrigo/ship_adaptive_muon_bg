"""Minimal utility-guided resampling loop: eligible pool -> U0 -> tilt h -> draws (no ROOT, no FairShip here).

Recovered from ``~/thesis/docs/1011_U0_DirectedPilot/u0.py`` (``pool``, ``h_of``, ``probs``).  Logic kept; changes:

* eligibility gate (PROJECT DECISION 2026-10-25, PROVISIONAL): only rows whose TGeo initial volume is ``cave`` or
  ``Cavern`` under the pinned geometry and the legacy ``+2.5916 m`` transform are drawn.  Rows starting inside a
  ``Magn7_*`` volume are ``INELIGIBLE_IN_SHIELD_MAGNET``: excluded and counted, never negatives.  Being in the cave
  does not prove a row lies on the post-shield surface PR #1432 expects (OPEN, B5).
* no hard ``p0 <= 3 GeV`` exclusion (that was a necessary condition for ``Y^(k)`` only); the PR #1432 generator's own
  ``p < 2 GeV`` limit shows up downstream as NOT_EVALUATED.
* reference arm is uniform over eligible rows: the physical weight ``w`` is not applied because its lineage and
  exposure semantics are OPEN (``docs/investigations/muon_weight_provenance.md``).  It is an *empirical* reference,
  not the nominal physical measure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

POOL_NPZ = Path(__file__).resolve().parents[2] / "data/samples/muonsFullMC_afterMS_sample.npz"
ELIGIBLE_VOLUMES = ("cave", "Cavern")
DELTA, ALPHA = 0.1, 4.0
ARMS = ("reference_uniform_eligible", "tilt_u0")


def load_pool(path: Path = POOL_NPZ) -> np.ndarray:
    return np.load(path)["muons"]  # (N, 8): px, py, pz [GeV], x, y, z [m], pdg, w


def valid_rows(a: np.ndarray) -> np.ndarray:
    p = np.linalg.norm(a[:, :3], axis=1)
    return np.isfinite(a).all(1) & (p > 0) & (a[:, 7] > 0)


def eligibility(a: np.ndarray, u0: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """One status per row; only ``ELIGIBLE`` rows can be drawn."""
    status = np.full(len(a), "INELIGIBLE_OTHER_VOLUME", dtype=object)
    status[np.isin(volume, ELIGIBLE_VOLUMES)] = "ELIGIBLE"
    status[np.char.startswith(volume.astype(str), "Magn7_")] = "INELIGIBLE_IN_SHIELD_MAGNET"
    status[~np.isfinite(u0)] = "UNSCORED"
    status[~valid_rows(a)] = "INVALID_ROW"  # p == 0, w <= 0 or non-finite: never scored, never drawn
    return status


def h(u0: np.ndarray, delta: float = DELTA, alpha: float = ALPHA) -> np.ndarray:
    return (delta + (1 - delta) * u0) ** alpha


def arm_probabilities(arm: str, status: np.ndarray, u0: np.ndarray) -> np.ndarray:
    ok = status == "ELIGIBLE"
    if arm == "reference_uniform_eligible":
        q = ok.astype(float)
    elif arm == "tilt_u0":
        q = np.where(ok, h(np.nan_to_num(u0)), 0.0)
    else:
        raise ValueError(f"unknown arm {arm!r}")
    return q / q.sum()


def draw(status: np.ndarray, u0: np.ndarray, n_per_arm: int, seed: int) -> list[dict]:
    """With-replacement draws per arm.  Each draw gets its own simulation seed; a repeated row is a repeated
    execution of one state, not a new independent draw from the reference measure."""
    rng = np.random.default_rng(seed)
    sim_seeds = rng.choice(899_999_999, size=n_per_arm * len(ARMS), replace=False) + 1  # PR #1432 range, not materialised
    rows = []
    for arm in ARMS:
        pi = arm_probabilities(arm, status, u0)
        for k, i in enumerate(rng.choice(len(pi), n_per_arm, replace=True, p=pi)):
            rows.append(dict(draw_id=f"{arm}-{k:03d}", arm=arm, row_index=int(i), sim_seed=int(sim_seeds[len(rows)]),
                             proposal_prob=float(pi[i]), u0=float(u0[i]), h=float(h(u0[i]))))
    return rows


def concentration(status: np.ndarray, u0: np.ndarray) -> dict[str, dict[str, float]]:
    """Per arm: probability mass on U0=1, effective number of rows (1/sum pi^2), rows carrying 90 % of the mass."""
    out = {}
    for arm in ARMS:
        pi = arm_probabilities(arm, status, u0)
        top = np.sort(pi)[::-1]
        out[arm] = {"mass_u0_1": float(pi[u0 == 1].sum()), "effective_rows": float(1 / (pi ** 2).sum()),
                    "rows_for_90pct_mass": int(np.searchsorted(np.cumsum(top), 0.9) + 1)}
    return out


# ---- iteration 2+: geometric strata, U fitted from snapshots, stratified allocation of expensive runs -------------
#: The PR #1432 generator skips muons below 2 GeV (``adapters/pr1432_mudis.py`` GENERATOR_MIN_P_GEV): such a run is a
#: known NOT_EVALUATED, so stratified draws never spend a simulation on it.  It is a support limit, not a negative.
MIN_P_GEV = 2.0
CLASSES = ("G0_none", "G1_upstream", "G2_sbt", "G3_tracker")


def geo_class(u0: np.ndarray, ray_upstream: np.ndarray, ray_tracker: np.ndarray) -> np.ndarray:
    """Straight-ray class, highest first: crosses Tr1-4 > crosses SBT LiSc > crosses Upstream (UBT) > none."""
    return np.where(ray_tracker, 3, np.where(u0 == 1, 2, np.where(ray_upstream, 1, 0)))


def fit_u(state_class: dict[str, int], state_eta: dict[str, float]) -> dict[str, dict]:
    """U per class = (sum of per-state pass rates + 1) / (states + 2).  The unit is the distinct *state* (each counts
    once however often it ran) and the endpoint is a pass *per run*, so U matches the one-run-per-state allocation."""
    out = {}
    for k, name in enumerate(CLASSES):
        ids = [s for s, c in state_class.items() if c == k]
        eta = sum(state_eta[s] for s in ids)
        out[name] = {"valid_states": len(ids), "positive_states": sum(state_eta[s] > 0 for s in ids), "sum_state_eta": eta,
                     "u": (eta + 1) / (len(ids) + 2)}
    return out


def allocate(u: dict[str, float], available: dict[str, int], n: int, floor: int) -> dict[str, int]:
    """``floor`` runs per non-empty class (exploration, keeps every region measured), the rest proportional to U
    (expected positives per run; costs are equal per run), capped by availability, largest-remainder rounding."""
    alloc = {k: min(floor, available[k]) for k in u}
    if sum(alloc.values()) > n:
        raise ValueError(f"floors need {sum(alloc.values())} runs, budget is {n}")
    while (left := n - sum(alloc.values())) > 0:
        open_ = [k for k in u if alloc[k] < available[k]]
        if not open_:
            break
        share = {k: left * u[k] / sum(u[j] for j in open_) for k in open_}
        base = {k: min(int(share[k]), available[k] - alloc[k]) for k in open_}
        for k in open_:
            alloc[k] += base[k]
        if any(base[k] < int(share[k]) for k in open_):
            continue  # a class hit its cap: redistribute what is left among the others
        for k in sorted(open_, key=lambda k: share[k] - int(share[k]), reverse=True)[:n - sum(alloc.values())]:
            if alloc[k] < available[k]:
                alloc[k] += 1
    return alloc


def draw_strata(cls: np.ndarray, eligible: np.ndarray, alloc: dict[str, int], exclude: set[int], seed: int) -> list[dict]:
    """Without replacement inside each class; ``proposal_prob`` is the inclusion probability n_k / N_k."""
    rng = np.random.default_rng(seed)
    sim_seeds = rng.choice(899_999_999, size=sum(alloc.values()), replace=False) + 1
    rows = []
    for k, name in enumerate(CLASSES):
        pool = np.array([i for i in np.where(eligible & (cls == k))[0] if int(i) not in exclude], dtype=int)
        for j, i in enumerate(rng.choice(pool, alloc.get(name, 0), replace=False) if alloc.get(name) else []):
            rows.append(dict(draw_id=f"{name}-{j:03d}", arm=name, row_index=int(i), sim_seed=int(sim_seeds[len(rows)]),
                             proposal_prob=alloc[name] / len(pool)))
    return rows


def draw_repeats(candidates: dict[tuple[str, int], list[int]], n: dict[tuple[str, int], int], k: int, used_seeds: set[int],
                 seed: int) -> list[dict]:
    """``k`` more executions of ``n[g]`` states drawn at random per group ``g = (class, previous outcome)`` from states
    already evaluated, each execution with a seed never used before: measures the seed variance inside one state.
    ``proposal_prob`` is the state's inclusion probability inside its group."""
    rng = np.random.default_rng(seed)
    rows = []
    for (name, out), ids in candidates.items():
        m = n.get((name, out), 0)
        for i in rng.choice(sorted(ids), m, replace=False) if m else []:
            rows += [dict(draw_id=f"repeat-{name}-{'pass' if out else 'fail'}-{i}-{j}", arm="seed_repeat", row_index=int(i),
                          proposal_prob=m / len(ids)) for j in range(k)]
    seeds: list[int] = []
    while len(seeds) < len(rows):
        s = int(rng.integers(1, 900_000_000))  # PR #1432 seed range
        if s not in used_seeds and s not in seeds:
            seeds.append(s)
    return [{**r, "sim_seed": s} for r, s in zip(rows, seeds)]
