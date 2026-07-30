"""PDG-id row selection for the local post-shield muon PKL contract.

Models are trained separately per PDG id (no charge conditioning is built in
anywhere downstream, see ``Nflow.interfaces``), so selecting the rows for one
PDG id is a data-contract concern, not a training concern. This module is the
single place that does it, so the repository sample and any full local
dataset select rows the same way.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from . import schema
from .errors import IdError


def filter_by_pdg(array: np.ndarray, pdg_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(filtered_rows, source_indices)`` for rows whose ``id == pdg_id``.

    ``pdg_id`` must be one of :data:`schema.EXPECTED_MUON_IDS` (``13`` or
    ``-13``) -- this is a PDG-id selector, not a general integer filter.
    ``id`` values are compared after rounding to the nearest integer, the same
    tolerance :func:`validation.validate_id_integer` already enforces
    elsewhere in the contract. ``source_indices`` are the sorted row indices
    into ``array`` (ascending, since ``array`` order is unchanged by this
    selection), preserved for provenance and duplicate-row auditing.
    """

    if pdg_id not in schema.EXPECTED_MUON_IDS:
        raise IdError(
            "pdg_id must be one of {}, got {!r}".format(schema.EXPECTED_MUON_IDS, pdg_id)
        )
    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != schema.N_COLUMNS:
        raise IdError(
            "expected a 2-D ({}, )-column array, got shape {}".format(
                schema.N_COLUMNS, array.shape
            )
        )
    ids = np.rint(array[:, schema.COLUMN_INDEX[schema.ID_COLUMN]]).astype(np.int64)
    indices = np.flatnonzero(ids == int(pdg_id))
    return np.array(array[indices], copy=True), indices


def pdg_counts(array: np.ndarray) -> Dict[int, int]:
    """Return ``{pdg_id: row_count}`` for every distinct integer id present.

    Includes ids outside :data:`schema.EXPECTED_MUON_IDS`; callers that only
    want the expected muon ids should intersect with
    :data:`schema.EXPECTED_MUON_IDS` explicitly rather than assume this
    function filters them.
    """

    array = np.asarray(array, dtype=np.float64)
    ids = np.rint(array[:, schema.COLUMN_INDEX[schema.ID_COLUMN]]).astype(np.int64)
    values, counts = np.unique(ids, return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts)}
