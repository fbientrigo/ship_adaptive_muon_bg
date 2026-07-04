"""Quarantined legacy code from the ``mferril/NFlow`` fork.

Untested, HDF5-era (mother/daughter 4-feature layout), and CUDA-assuming.
It stays runnable for reference but must not be imported by ``Nflow``
core modules or wired into the tested pipeline until it is adapted to the
``(N, 8)`` PKL data contract and gains a tiny-overfit smoke test (see
``docs/architecture/ml_skeleton_local_pkl_v0.md``, section 5).

Importing anything under this package requires the ``legacy`` extra
(torch, h5py, scikit-learn, matplotlib, seaborn, pandas, optuna,
tensorboard, pyyaml).
"""
