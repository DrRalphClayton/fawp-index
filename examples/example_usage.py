"""
fawp-index — Quick Example
Demonstrates the FAWP Alpha Index on synthetic E8-style data.

Run after installing:
    pip install .
    python example_usage.py

Or via PyPI:
    pip install fawp-index
    python example_usage.py
"""

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
