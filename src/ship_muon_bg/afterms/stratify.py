"""Phase B: coarse stratification and round-robin shard assignment (§5.2-5.3).

Within one split, rows are grouped into strata by (exact PDG id, coarse
weight-quantile bin, coarse log1p(pz) bin, coarse log1p(pt) bin, coarse rxy
bin), each stratum is deterministically shuffled by the shard seed, then rows
are distributed round-robin across shards. This preserves common modes and
spreads rare strata across shards as far as their counts allow -- a stratum
with fewer rows than shards cannot appear in every shard; that shortfall is
recorded, not hidden.
"""

from __future__ import annotations

import numpy as np

N_QUANTILE_BINS = 8

# Edge buckets tracked separately from the coarse stratification (§5.3): each
# entry is (name, column_or_derived, upper_tail_fraction) or an explicit
# boolean condition, applied after shard assignment for reporting only.
TAIL_SPECS = (
    ("top1pct_pz", "pz", 0.01),
    ("top0p1pct_pz", "pz", 0.001),
    ("top1pct_pt", "pt", 0.01),
    ("top0p1pct_pt", "pt", 0.001),
    ("top1pct_abs_x", "abs_x", 0.01),
    ("top1pct_abs_y", "abs_y", 0.01),
    ("top1pct_weight", "w", 0.01),
)


def _derived_columns(array, schema_index):
    px = array[:, schema_index["px"]]
    py = array[:, schema_index["py"]]
    pz = array[:, schema_index["pz"]]
    x = array[:, schema_index["x"]]
    y = array[:, schema_index["y"]]
    w = array[:, schema_index["w"]]
    pt = np.sqrt(px**2 + py**2)
    rxy = np.sqrt(x**2 + y**2)
    return {
        "pz": pz,
        "pt": pt,
        "abs_x": np.abs(x),
        "abs_y": np.abs(y),
        "rxy": rxy,
        "w": w,
        "id": array[:, schema_index["id"]],
    }


def _quantile_bin_edges(values, n_bins=N_QUANTILE_BINS):
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(values, quantiles)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _digitize_bin(values, edges):
    return np.clip(np.digitize(values, edges[1:-1], right=False), 0, len(edges) - 2)


def compute_strata(array, schema_index, *, n_bins=N_QUANTILE_BINS):
    """Return (stratum_ids: (N,) int array, stratum_table: dict[id -> descriptor])."""
    derived = _derived_columns(array, schema_index)
    pdg = np.rint(derived["id"]).astype(np.int64)

    weight_edges = _quantile_bin_edges(derived["w"], n_bins)
    weight_bin = _digitize_bin(derived["w"], weight_edges)

    log_pz = np.log1p(np.clip(derived["pz"], a_min=0.0, a_max=None))
    pz_edges = _quantile_bin_edges(log_pz, n_bins)
    pz_bin = _digitize_bin(log_pz, pz_edges)

    log_pt = np.log1p(derived["pt"])
    pt_edges = _quantile_bin_edges(log_pt, n_bins)
    pt_bin = _digitize_bin(log_pt, pt_edges)

    rxy_edges = _quantile_bin_edges(derived["rxy"], n_bins)
    rxy_bin = _digitize_bin(derived["rxy"], rxy_edges)

    # Encode the 5-tuple stratum key as one integer for fast grouping.
    base = n_bins + 1
    pdg_codes, pdg_index = np.unique(pdg, return_inverse=True)
    stratum_ids = (
        ((pdg_index.astype(np.int64) * base + weight_bin) * base + pz_bin) * base
        + pt_bin
    ) * base + rxy_bin

    table = {
        "n_bins": int(n_bins),
        "pdg_codes": pdg_codes.tolist(),
        "weight_bin_edges": weight_edges.tolist(),
        "log_pz_bin_edges": pz_edges.tolist(),
        "log_pt_bin_edges": pt_edges.tolist(),
        "rxy_bin_edges": rxy_edges.tolist(),
    }
    return stratum_ids, table


def assign_shards(row_indices, stratum_ids, *, n_shards, shard_seed):
    """Deterministic round-robin shard assignment within each stratum.

    Returns (shard_of_row: (N,) int array, understrata_report: dict).
    """
    if n_shards < 1:
        raise ValueError("n_shards must be >= 1")
    row_indices = np.asarray(row_indices)
    stratum_ids = np.asarray(stratum_ids)
    shard_of_row = np.full(row_indices.shape, -1, dtype=np.int64)

    order = np.argsort(stratum_ids, kind="stable")
    sorted_strata = stratum_ids[order]
    boundaries = np.flatnonzero(np.diff(sorted_strata)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [sorted_strata.size]))

    understaffed = {}
    for start, end in zip(starts, ends):
        members = order[start:end]
        stratum_key = int(sorted_strata[start])
        rng = np.random.default_rng((int(shard_seed), stratum_key))
        shuffled = members[rng.permutation(members.size)]
        shard_of_row[shuffled] = np.arange(shuffled.size) % n_shards
        if members.size < n_shards:
            understaffed[str(stratum_key)] = {
                "row_count": int(members.size),
                "shards_missing_this_stratum": int(n_shards - members.size),
            }

    assert np.all(shard_of_row >= 0)
    return shard_of_row, {
        "n_shards": int(n_shards),
        "n_strata": int(starts.size),
        "understaffed_strata": understaffed,
    }


def compute_tail_buckets(array, schema_index):
    """Boolean membership mask per tail spec in TAIL_SPECS, plus a pz==0 mask."""
    derived = _derived_columns(array, schema_index)
    masks = {}
    for name, col, frac in TAIL_SPECS:
        values = derived[col]
        threshold = np.quantile(values, 1.0 - frac)
        masks[name] = values >= threshold
    masks["pz_equals_zero"] = derived["pz"] == 0.0
    return masks


def n_shards_for_split(n_rows, *, target_rows_per_shard=500_000):
    return max(1, int(round(n_rows / target_rows_per_shard)))
