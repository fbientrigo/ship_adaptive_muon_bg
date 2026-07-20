"""After-MS nightly experiment path: audit, sharding, log1p-pz preprocessing.

Dedicated to ``data/raw/nflow_releases/muonsFullMC_afterMS.pkl``. Reuses
``ship_muon_bg.data_contracts`` (loader, schema, validation, hashing) rather
than reimplementing the raw-data contract.
"""

from __future__ import annotations
