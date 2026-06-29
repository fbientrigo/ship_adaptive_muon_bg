"""Loader for trusted local legacy muon PKL files.

Historical muon samples are gzip-compressed pickle files whose payload is a
single NumPy array of shape ``(N, 8)``.

Security note: ``pickle`` executes arbitrary code on load. This loader is for
**trusted local legacy files only**. Never point it at untrusted or remote
inputs. New on-disk samples should prefer NPZ (see the planning doc, §9).
"""

from __future__ import annotations

import gzip
import pickle

import numpy as np

from .errors import LoaderError


def load_muon_pkl(path):
    """Load a trusted local gzip-PKL muon file into a ``float64`` NumPy array.

    The payload is decompressed, unpickled, and coerced to a contiguous
    ``float64`` array. This function performs only the minimal structural check
    that the payload is array-like and 2-D; full contract validation
    (shape/units/finite/weights/id) is the job of :mod:`validation`.

    Parameters
    ----------
    path : str or os.PathLike
        Filesystem path to the ``*.pkl.gz`` file. Caller-supplied; no path is
        hardcoded.

    Returns
    -------
    numpy.ndarray
        A C-contiguous ``float64`` array.

    Raises
    ------
    LoaderError
        If the payload cannot be decompressed/unpickled or is not a 2-D
        array-like object.
    """
    try:
        with gzip.open(path, "rb") as handle:
            payload = pickle.load(handle)
    except OSError as exc:  # decompression / IO failure
        raise LoaderError(f"could not open/decompress PKL at {path!r}: {exc}") from exc
    except pickle.UnpicklingError as exc:
        raise LoaderError(f"could not unpickle PKL at {path!r}: {exc}") from exc

    try:
        array = np.ascontiguousarray(payload, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise LoaderError(
            f"PKL payload at {path!r} is not coercible to a float array: {exc}"
        ) from exc

    if array.ndim != 2:
        raise LoaderError(
            f"PKL payload at {path!r} is not 2-D (got ndim={array.ndim})"
        )

    return array
