"""Analytic Gaussian density baselines for the controlled density lab.

These estimators are exact-MLE Gaussian family models behind the common
``DensityEstimator`` interface. They import only NumPy (the Gaussian mixture
uses scikit-learn lazily, only when fitted). Canonical artifacts are explicit
parameter arrays (NPZ + JSON), never pickled estimators.
"""

from __future__ import annotations

from .gaussian import DiagonalGaussian, FullGaussian
from .gmm import GaussianMixtureEstimator

__all__ = [
    "DiagonalGaussian",
    "FullGaussian",
    "GaussianMixtureEstimator",
]
