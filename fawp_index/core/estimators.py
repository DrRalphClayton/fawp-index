"""
fawp-index: Core MI Estimators and Null Controls
Based on OATS validation suite by Ralph Clayton (2026)
DOI: https://doi.org/10.5281/zenodo.18663547
"""

import numpy as np
from typing import Tuple, Optional


def mi_from_rho(rho: float) -> float:
    if not np.isfinite(rho):
        return 0.0
    rho = float(np.clip(rho, -0.99999, 0.99999))
    return (-0.5 * np.log(1.0 - rho ** 2)) / np.log(2.0)


def mi_from_arrays(x: np.ndarray, y: np.ndarray, min_n: int = 30) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < min_n:
        return 0.0
    rho = np.corrcoef(x, y)[0, 1]
    return mi_from_rho(rho)


def null_shuffle(x, y, n_null=200, beta=0.99, rng=None, min_n=30):
    if rng is None:
        rng = np.random.default_rng(42)
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    null_mis = [mi_from_arrays(x, rng.permutation(y), min_n=min_n) for _ in range(n_null)]
    null_mis = np.array(null_mis)
    return float(np.mean(null_mis)), float(np.quantile(null_mis, beta))


def null_shift(x, y, n_null=200, beta=0.99, rng=None, min_n=30):
    if rng is None:
        rng = np.random.default_rng(43)
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = len(y)
    null_mis = [mi_from_arrays(x, np.roll(y, int(rng.integers(1, max(2, n)))), min_n=min_n) for _ in range(n_null)]
    null_mis = np.array(null_mis)
    return float(np.mean(null_mis)), float(np.quantile(null_mis, beta))


def conservative_null_floor(x, y, n_null=200, beta=0.99, rng=None, min_n=30):
    _, q_shuf = null_shuffle(x, y, n_null=n_null, beta=beta, rng=rng, min_n=min_n)
    _, q_shift = null_shift(x, y, n_null=n_null, beta=beta, rng=rng, min_n=min_n)
    return max(q_shuf, q_shift)


def null_corrected_mi(x, y, n_null=200, beta=0.99, rng=None, min_n=30):
    raw = mi_from_arrays(x, y, min_n=min_n)
    floor = conservative_null_floor(x, y, n_null=n_null, beta=beta, rng=rng, min_n=min_n)
    return max(0.0, raw - floor), floor
