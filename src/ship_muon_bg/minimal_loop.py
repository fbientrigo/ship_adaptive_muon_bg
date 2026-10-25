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
    sim_seeds = rng.choice(np.arange(1, 899_999_999), size=n_per_arm * len(ARMS), replace=False)  # PR #1432 seed range
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
