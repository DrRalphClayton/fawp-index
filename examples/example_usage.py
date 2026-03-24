"""
fawp-index — Quick Example
Demonstrates the FAWP Alpha Index on synthetic E8-style data.

Running this script
-------------------
Option A — from a cloned repo (recommended):
    git clone https://github.com/DrRalphClayton/fawp-index
    cd fawp-index
    pip install -e .
    python examples/example_usage.py

Option B — from the examples/ directory inside the repo:
    cd fawp-index/examples
    python example_usage.py

Note for PyPI installs
----------------------
If you installed via ``pip install fawp-index`` this script is *not* in
your working directory.  Either clone the repo and follow Option A above,
or copy-paste the code from:
    https://github.com/DrRalphClayton/fawp-index/blob/main/examples/example_usage.py
"""
import sys, os as _os
# Allow running from repo root OR from examples/ directory
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fawp_index import FAWPAlphaIndex
from fawp_index.io.feeds import load_synthetic_demo

print("fawp-index — FAWP Alpha Index Example")
print("Ralph Clayton (2026) | doi:10.5281/zenodo.18673949")
print()

data = load_synthetic_demo('seismic', seed=42)
print(f"Loaded {len(data.pred_series):,} observations ({data.metadata['source']})")

result = FAWPAlphaIndex(n_null=200).compute(
    pred_series   = data.pred_series,
    future_series = data.future_series,
    action_series = data.action_series,
    obs_series    = data.obs_series,
    tau_grid      = list(range(1, 16)),
)

print(result.summary())

print(f"{'τ':>4} {'Pred MI':>10} {'Steer MI':>10} {'Alpha':>10} {'FAWP':>6}")
print("-" * 45)
for i, tau in enumerate(result.tau):
    flag = " ← ✓" if result.in_fawp[i] else ""
    print(f"{int(tau):>4} {result.pred_mi_raw[i]:>10.4f} "
          f"{result.steer_mi_raw[i]:>10.4f} "
          f"{result.alpha_index[i]:>10.4f}{flag}")

print()
print("To plot: result.plot()  (requires: pip install matplotlib)")
