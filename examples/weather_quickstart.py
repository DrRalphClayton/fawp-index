"""
Weather FAWP quickstart — detect the Information-Control Exclusion Principle
in ERA5 reanalysis data for any location on Earth.

No API key needed. Uses Open-Meteo (free, open data).

Install:
    pip install "fawp-index[weather]"
"""

from fawp_index.weather import fawp_from_open_meteo, scan_weather_grid

# ── Single location ────────────────────────────────────────────────────────
print("=== Single location: London temperature ===")
result = fawp_from_open_meteo(
    latitude    = 51.5,
    longitude   = -0.1,
    variable    = "temperature_2m",
    start_date  = "2015-01-01",
    end_date    = "2024-12-31",
    horizon_days = 7,
)
print(result.summary())

# ── Multi-location grid scan ───────────────────────────────────────────────
print("\n=== Grid scan: 5 major cities ===")
cities = [
    {"lat": 51.5,  "lon": -0.1,  "name": "London"},
    {"lat": 48.9,  "lon":  2.4,  "name": "Paris"},
    {"lat": 40.7,  "lon": -74.0, "name": "New York"},
    {"lat": 35.7,  "lon": 139.7, "name": "Tokyo"},
    {"lat": -33.9, "lon": 151.2, "name": "Sydney"},
]
results = scan_weather_grid(
    cities,
    variable     = "temperature_2m",
    start_date   = "2015-01-01",
    end_date     = "2024-12-31",
    horizon_days = 7,
)
print(f"\n{'Location':<20} {'FAWP':>6}  {'Gap (bits)':>10}  {'ODW'}")
print("-" * 50)
for r in results:
    flag = "🔴 YES" if r.fawp_found else "—"
    odw  = f"τ {r.odw_start}-{r.odw_end}" if r.fawp_found else "—"
    print(f"{r.location:<20} {flag:>6}  {r.peak_gap_bits:>10.4f}  {odw}")

# ── From your own forecast data ────────────────────────────────────────────
import numpy as np
from fawp_index.weather import fawp_from_forecast

print("\n=== Synthetic NWP example ===")
rng = np.random.default_rng(42)
n   = 2000
# Simulate a NWP model with degrading intervention effectiveness
true_temp    = 15 + 5 * np.sin(np.linspace(0, 12*np.pi, n)) + rng.normal(0, 1, n)
forecast     = true_temp + rng.normal(0, 0.8, n)
observed     = true_temp
# Steering (model nudge) loses effectiveness over time
nudge_effect = np.exp(-0.003 * np.arange(n))
intervention = nudge_effect * rng.normal(0, 1, n)

result = fawp_from_forecast(
    forecast     = forecast,
    observed     = observed,
    intervention = intervention,
    horizon_days = 5,
    variable     = "temperature_2m",
    location     = "synthetic_NWP",
)
print(result.summary())
