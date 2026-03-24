"""
Weather FAWP quickstart — detect the Information-Control Exclusion Principle
in ERA5 reanalysis data for any location on Earth.

No API key needed. Uses Open-Meteo (free, open data).

Install:
    pip install "fawp-index[weather]"
"""
import sys, os as _os
# Allow running from repo root OR from examples/ directory
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── 1. Simplest possible usage ─────────────────────────────────────────────
from fawp_index.weather import fetch_openmeteo, to_fawp_dataframe, fawp_from_forecast

df = fetch_openmeteo("London", days=365*5, var="temperature_2m")
print(df.head())
#    date        pred   future  action    obs
# 0  2019-01-01  6.3    7.1     0.8       6.3

result = fawp_from_forecast(
    forecast     = df["pred"].values,
    observed     = df["future"].values,
    intervention = df["action"].values,
    variable     = "temperature_2m",
    location     = "London",
)
result.explain()    # plain-English explanation
print(result.summary())

# ── 2. Named city scan via ERA5 (one-liner) ────────────────────────────────
from fawp_index.weather import fawp_from_open_meteo

result2 = fawp_from_open_meteo(
    latitude=51.5, longitude=-0.1,
    variable="temperature_2m",
    start_date="2015-01-01",
    end_date="2024-12-31",
    horizon_days=7,
)
print(result2.summary())

# ── 3. Multi-city grid scan ────────────────────────────────────────────────
from fawp_index.weather import scan_weather_grid

cities = [
    {"lat": 51.5,  "lon": -0.1,  "name": "London"},
    {"lat": 48.9,  "lon":  2.4,  "name": "Paris"},
    {"lat": 40.7,  "lon": -74.0, "name": "New York"},
    {"lat": 35.7,  "lon": 139.7, "name": "Tokyo"},
    {"lat": -33.9, "lon": 151.2, "name": "Sydney"},
]
results = scan_weather_grid(cities, variable="temperature_2m",
                            start_date="2015-01-01", end_date="2024-12-31")
print(f"\n{'Location':<20} {'FAWP':>6}  {'Gap (bits)':>10}  {'ODW'}")
print("-" * 50)
for r in results:
    flag = "🔴 YES" if r.fawp_found else "—"
    odw  = f"τ {r.odw_start}-{r.odw_end}" if r.fawp_found else "—"
    print(f"{r.location:<20} {flag:>6}  {r.peak_gap_bits:>10.4f}  {odw}")

# ── 4. From your own forecast arrays ──────────────────────────────────────
import numpy as np
from fawp_index.weather import fawp_from_forecast

rng = np.random.default_rng(42)
n   = 2000
true_temp    = 15 + 5*np.sin(np.linspace(0, 12*np.pi, n)) + rng.normal(0, 1, n)
forecast     = true_temp + rng.normal(0, 0.8, n)
observed     = true_temp
intervention = np.exp(-0.003*np.arange(n)) * rng.normal(0, 1, n)

result3 = fawp_from_forecast(forecast, observed, intervention,
                              horizon_days=5, variable="temperature_2m",
                              location="synthetic_NWP")
print(result3.explain())

# ── 5. CLI usage ──────────────────────────────────────────────────────────
# fawp-weather scan --location london --variable temperature_2m --horizon 7
# fawp-weather scan --location "new york" --variable precipitation_sum
# fawp-weather scan --lat 51.5 --lon -0.1 --out result.json
# fawp-weather grid --cities london paris newyork tokyo sydney
# fawp-weather list-variables
