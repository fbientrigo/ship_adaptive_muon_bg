import numpy as np

from ship_muon_bg.afterms import stratify
from ship_muon_bg.data_contracts import schema


def _fake_array(n, rng):
    px = rng.normal(size=n)
    py = rng.normal(size=n)
    pz = rng.uniform(0, 50, size=n)
    x = rng.normal(scale=0.5, size=n)
    y = rng.normal(scale=0.5, size=n)
    z = rng.normal(size=n)
    ids = rng.choice([13, -13], size=n).astype(np.float64)
    w = rng.uniform(0.1, 2.0, size=n)
    return np.column_stack([px, py, pz, x, y, z, ids, w])


def test_deterministic_shard_assignment():
    rng = np.random.default_rng(1)
    array = _fake_array(20_000, rng)
    stratum_ids, _ = stratify.compute_strata(array, schema.COLUMN_INDEX)
    idx = np.arange(array.shape[0])

    shards_a, _ = stratify.assign_shards(idx, stratum_ids, n_shards=4, shard_seed=99)
    shards_b, _ = stratify.assign_shards(idx, stratum_ids, n_shards=4, shard_seed=99)
    np.testing.assert_array_equal(shards_a, shards_b)


def test_all_rows_assigned_to_valid_shard():
    rng = np.random.default_rng(2)
    array = _fake_array(15_000, rng)
    stratum_ids, _ = stratify.compute_strata(array, schema.COLUMN_INDEX)
    idx = np.arange(array.shape[0])
    shards, report = stratify.assign_shards(idx, stratum_ids, n_shards=5, shard_seed=1)
    assert shards.min() >= 0
    assert shards.max() < 5
    assert shards.shape[0] == array.shape[0]
    assert report["n_shards"] == 5


def test_pdg_proportions_roughly_preserved_per_shard():
    rng = np.random.default_rng(3)
    array = _fake_array(40_000, rng)
    stratum_ids, _ = stratify.compute_strata(array, schema.COLUMN_INDEX)
    idx = np.arange(array.shape[0])
    shards, _ = stratify.assign_shards(idx, stratum_ids, n_shards=4, shard_seed=7)

    pdg = np.rint(array[:, schema.COLUMN_INDEX["id"]])
    overall_frac_13 = np.mean(pdg == 13)
    for s in range(4):
        mask = shards == s
        shard_frac_13 = np.mean(pdg[mask] == 13)
        assert abs(shard_frac_13 - overall_frac_13) < 0.05


def test_understaffed_strata_reported_not_hidden():
    rng = np.random.default_rng(4)
    array = _fake_array(200, rng)  # small dataset -> some strata smaller than n_shards
    stratum_ids, _ = stratify.compute_strata(array, schema.COLUMN_INDEX, n_bins=8)
    idx = np.arange(array.shape[0])
    _, report = stratify.assign_shards(idx, stratum_ids, n_shards=10, shard_seed=1)
    assert report["n_strata"] > 0
    # With 200 rows split across up to 8**4 * 2 strata and 10 shards, some
    # strata are necessarily smaller than n_shards.
    assert len(report["understaffed_strata"]) > 0


def test_tail_buckets_flag_extremes():
    rng = np.random.default_rng(5)
    array = _fake_array(10_000, rng)
    masks = stratify.compute_tail_buckets(array, schema.COLUMN_INDEX)
    assert masks["top1pct_pz"].sum() == pytest_approx_count(array.shape[0], 0.01)
    assert "pz_equals_zero" in masks


def pytest_approx_count(n, frac):
    # np.quantile-based thresholding gives an approx top-frac count; just
    # assert it's in a sane ballpark rather than exact.
    lo = int(n * frac * 0.5)
    hi = int(n * frac * 2.0) + 1
    return AnyInRange(lo, hi)


class AnyInRange:
    def __init__(self, lo, hi):
        self.lo, self.hi = lo, hi

    def __eq__(self, other):
        return self.lo <= other <= self.hi

    def __repr__(self):
        return f"AnyInRange({self.lo}, {self.hi})"
