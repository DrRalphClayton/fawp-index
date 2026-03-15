# Weather & Climate FAWP Detection

Detect the Information-Control Exclusion Principle in atmospheric forecast systems.

## The core question

> "Is there a regime where the atmosphere remains forecastable,
>  but weather-modifying interventions have already lost their effect?"

That is FAWP in the weather domain: **prediction persists, control has collapsed.**

## Install

```bash
pip install "fawp-index[weather]"
```

## Three entry points

### 1. From ERA5 reanalysis data (no API key)

```python
from fawp_index.weather import fawp_from_open_meteo

result = fawp_from_open_meteo(
    latitude    = 51.5,    # London
    longitude   = -0.1,
    variable    = "temperature_2m",
    start_date  = "2015-01-01",
    end_date    = "2024-12-31",
    horizon_days = 7,      # 7-day forecast horizon
)
print(result.summary())
```

### 2. Multi-location grid scan

```python
from fawp_index.weather import scan_weather_grid

cities = [
    {"lat": 51.5,  "lon": -0.1,  "name": "London"},
    {"lat": 48.9,  "lon":  2.4,  "name": "Paris"},
    {"lat": 40.7,  "lon": -74.0, "name": "New York"},
]
results = scan_weather_grid(cities, variable="temperature_2m",
                            start_date="2015-01-01", end_date="2024-12-31")
for r in results:
    flag = "🔴 FAWP" if r.fawp_found else "—"
    print(f"{r.location:<20} {flag}  gap={r.peak_gap_bits:.4f}b")
```

### 3. From your own NWP forecast arrays

```python
from fawp_index.weather import fawp_from_forecast

result = fawp_from_forecast(
    forecast     = nwp_output,       # model forecast values
    observed     = verification,     # actual measurements
    intervention = model_nudge,      # forecast adjustment / ensemble spread
    horizon_days = 5,
    variable     = "temperature_2m",
    location     = "50.0N 0.0E",
)
print(result.summary())
```

## Supported variables (ERA5 via Open-Meteo)

| Variable | Description |
|----------|-------------|
| `temperature_2m` | 2m air temperature (°C) |
| `precipitation_sum` | Daily precipitation (mm) |
| `wind_speed_10m` | 10m wind speed (m/s) |
| `surface_pressure` | Surface pressure (hPa) |
| `cloud_cover` | Cloud cover fraction (%) |
| `et0_fao_evapotranspiration` | Reference ET (mm) |
| `shortwave_radiation` | Solar radiation (W/m²) |

## Interpreting results

| Field | Meaning |
|-------|---------|
| `fawp_found` | FAWP regime detected — prediction persists but control collapsed |
| `peak_gap_bits` | Maximum leverage gap (bits) — larger = stronger FAWP |
| `odw_start/end` | Operational Detection Window (τ range) |
| `tau_h_plus` | Post-zero agency horizon — where control first vanishes |
| `tau_f` | Failure cliff — where the system becomes fully uncontrollable |

## Physical interpretations

- **High `peak_gap_bits`**: The forecast model retains skill at lags where
  interventions (nudges, corrections, warnings) have already lost effect.
- **Narrow ODW**: FAWP window is short — small lead time available before cliff.
- **FAWP in precipitation**: Predictability persists into a regime where
  cloud-seeding or model re-initialization no longer changes outcomes.
- **FAWP in wind energy**: Grid operator can forecast output but can no longer
  route or curtail fast enough to affect the outcome.

## Papers

- E1–E7: doi:10.5281/zenodo.18663547
- E8: doi:10.5281/zenodo.18673949
- E9 (SPHERE_15): Experiment 9 confirmation suite
