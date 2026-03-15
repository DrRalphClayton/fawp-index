"""
JIT speed test — compares MI estimation with and without Numba.

Run with:
    python examples/jit_speedtest.py

Install Numba first for the speedup:
    pip install "fawp-index[fast]"
"""

import time
import numpy as np
from fawp_index.core.estimators import (
    mi_from_arrays, null_shuffle, null_shift, has_numba
)

print(f"Numba JIT active: {has_numba()}")
print()

rng = np.random.default_rng(42)
n   = 500
x   = rng.normal(0, 1, n)
y   = 0.4 * x + rng.normal(0, 1, n)

# Warm up JIT if active
_ = mi_from_arrays(x, y)
_ = null_shuffle(x, y, n_null=10)

# Benchmark: single MI
trials = 10_000
t0 = time.perf_counter()
for _ in range(trials):
    mi_from_arrays(x, y)
t1 = time.perf_counter()
per_call_us = (t1 - t0) / trials * 1e6
print(f"mi_from_arrays ({n} samples × {trials} calls)")
print(f"  Total  : {t1-t0:.3f}s")
print(f"  Per call: {per_call_us:.2f} µs")
print()

# Benchmark: null shuffle (the real hot path in watchlist scans)
n_null = 200
trials_null = 50
t0 = time.perf_counter()
for _ in range(trials_null):
    null_shuffle(x, y, n_null=n_null)
t1 = time.perf_counter()
per_null_ms = (t1 - t0) / trials_null * 1e3
print(f"null_shuffle (n_null={n_null}, n={n}, × {trials_null} calls)")
print(f"  Total  : {t1-t0:.3f}s")
print(f"  Per call: {per_null_ms:.1f} ms")
print()

# Benchmark: null shift
t0 = time.perf_counter()
for _ in range(trials_null):
    null_shift(x, y, n_null=n_null)
t1 = time.perf_counter()
per_shift_ms = (t1 - t0) / trials_null * 1e3
print(f"null_shift   (n_null={n_null}, n={n}, × {trials_null} calls)")
print(f"  Total  : {t1-t0:.3f}s")
print(f"  Per call: {per_shift_ms:.1f} ms")
print()

print("Expected speedup vs pure NumPy:")
print("  mi_from_arrays  : 3–6×")
print("  null_shuffle    : 5–15× (JIT loop over 200 permutations)")
print("  null_shift      : 5–15×")
print()
print("Install Numba: pip install 'fawp-index[fast]'")
