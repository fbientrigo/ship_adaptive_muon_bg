"""Deterministic, range-preserving subsampling of muon datasets.

The full ``Muon NTuples v1.0`` datasets are hundreds of MB (GitHub release
assets) and must stay out of git. To let anyone understand the data shape,
units, and ranges from a clone, a small representative subset is committed.

The subset is built from two parts, then deduplicated:

1. **Uniform core** — a deterministic seeded uniform random sample of rows, so
   the committed marginals (and quantiles) resemble the full distribution.
2. **Range anchors** — the argmin and argmax row of every column, so the sample's
   per-column min/max equal the full dataset's. This preserves the observed
   *envelope* (which a small uniform sample would understate) while adding only
   two rows per column, far too few to distort the distribution's quantiles.

All randomness is driven by an explicit integer ``seed`` via
``numpy.random.default_rng`` — no wall-clock seeding, mirroring
:mod:`ship_muon_bg.data_contracts.splitting`.

New on-disk samples are also written as NPZ. Per the storage strategy in
``docs/architecture/ml_skeleton_local_pkl_v0.md`` (§9), NPZ is preferred over
pickle for *new* samples because pickle executes arbitrary code on load; the
gzip-PKL copy is kept only for compatibility with the existing legacy loaders.
"""

from __future__ import annotations

import gzip
import pickle

import numpy as np

from . import schema
from .errors import LoaderError, ShapeError

# NPZ array key for a saved muon subset.
NPZ_ARRAY_KEY = "muons"


def _range_anchor_indices(array):
    """Row indices whose values define the per-column envelope.

    The argmin and argmax of every one of the 8 columns — the rows a
    representative sample must not drop if the sample is to report the same
    per-column min/max as the full dataset.
    """
    idx = set()
    for col in range(array.shape[1]):
        idx.add(int(np.argmin(array[:, col])))
        idx.add(int(np.argmax(array[:, col])))
    return idx


def representative_subset(array, n_rows, *, seed):
    """Return a deterministic, range-preserving subset of ``array``.

    Parameters
    ----------
    array : numpy.ndarray
        Full ``(N, 8)`` muon array (``[px, py, pz, x, y, z, id, w]``).
    n_rows : int
        Target number of rows in the subset. If ``n_rows >= N`` the whole array
        is returned.
    seed : int
        Explicit deterministic seed; never derived from wall-clock time.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        ``(subset_rows, selected_indices)`` where ``selected_indices`` are the
        sorted row indices into the original ``array`` (provenance). Row order of
        ``subset_rows`` matches ``selected_indices``.
    """
    if array.ndim != 2 or array.shape[1] != schema.N_COLUMNS:
        raise ShapeError(
            f"expected a 2-D ({schema.N_COLUMNS}-column) array, got shape {array.shape}"
        )
    if not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an explicit integer")
    if n_rows < 1:
        raise ValueError("n_rows must be >= 1")

    n_total = array.shape[0]
    if n_rows >= n_total:
        return np.array(array, copy=True), np.arange(n_total)

    forced = _range_anchor_indices(array)

    # Fill the remainder with a deterministic uniform sample of the other rows.
    rng = np.random.default_rng(int(seed))
    remaining = max(0, n_rows - len(forced))
    if remaining > 0:
        pool = np.setdiff1d(
            np.arange(n_total), np.fromiter(forced, dtype=int), assume_unique=False
        )
        take = min(remaining, pool.size)
        chosen = rng.choice(pool, size=take, replace=False)
        forced.update(int(i) for i in chosen)

    selected = np.array(sorted(forced), dtype=int)
    return np.array(array[selected], copy=True), selected


def save_subset_npz(path, array):
    """Save a muon subset as a compressed NPZ under :data:`NPZ_ARRAY_KEY`."""
    canonical = np.ascontiguousarray(array, dtype=np.float64)
    np.savez_compressed(path, **{NPZ_ARRAY_KEY: canonical})


def save_subset_pkl_gz(path, array):
    """Save a muon subset as a gzip-pickled ``float64`` array (legacy format)."""
    canonical = np.ascontiguousarray(array, dtype=np.float64)
    with gzip.open(path, "wb") as handle:
        pickle.dump(canonical, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_muon_npz(path, *, key=NPZ_ARRAY_KEY):
    """Load a muon subset from NPZ into a ``float64`` ``(N, 8)`` array.

    The NPZ counterpart of :func:`ship_muon_bg.data_contracts.loader.load_muon_pkl`.
    Unlike pickle, NPZ does not execute code on load, so this is the preferred
    reader for new samples. Full contract validation still belongs to
    :mod:`validation`.

    Raises
    ------
    LoaderError
        If the file cannot be read, lacks ``key``, or is not a 2-D array.
    """
    try:
        with np.load(path, allow_pickle=False) as data:
            if key not in data:
                raise LoaderError(
                    f"NPZ at {path!r} has no array {key!r}; keys={list(data.keys())}"
                )
            array = np.ascontiguousarray(data[key], dtype=np.float64)
    except LoaderError:
        raise
    except (OSError, ValueError) as exc:
        raise LoaderError(f"could not load NPZ at {path!r}: {exc}") from exc

    if array.ndim != 2:
        raise LoaderError(f"NPZ payload at {path!r} is not 2-D (got ndim={array.ndim})")
    return array
