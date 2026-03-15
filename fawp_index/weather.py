"""
fawp_index.weather — FAWP detection for weather and climate forecast systems.

Applies the Information-Control Exclusion Principle to meteorological data:
  - Prediction channel: model forecast skill (how well forecasts verify)
  - Steering channel:   intervention effectiveness (can you still nudge the system?)

The core question FAWP asks in the weather domain:
  "Is there a regime where the atmosphere remains forecastable,
   but weather-modifying interventions (seeding, routing, warning response)
   have already lost their effect?"

Three entry points:

1. ``fawp_from_forecast`` — works from model forecast + verification arrays.
   The most common case: NWP model output vs actual observations.

2. ``fawp_from_skill_series`` — works from a pre-computed skill time series
   (e.g. anomaly correlation, RMSE) alongside an intervention effectiveness series.

3. ``fawp_from_reanalysis`` — loads ERA5 or open-meteo data (no API key needed)
   and constructs the FAWP channels automatically.

Domains supported:
  - Temperature forecasting (NWP skill collapse)
  - Precipitation forecasting
  - Wind / renewable energy production forecasts
  - Wildfire weather windows
  - Climate model drift detection

Install optional dependencies:
    pip install "fawp-index[weather]"   # adds xarray, netcdf4, openmeteo-requests

Ralph Clayton (2026) · doi:10.5281/zenodo.18673949
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from fawp_index import __version__ as _VERSION
from fawp_index.constants import (
    BETA_NULL_QUANTILE,
    EPSILON_STEERING_RAW,
    MARKET_TAU_MAX,
    PERSISTENCE_RULE_M,
    PERSISTENCE_RULE_N,
)
from fawp_index.detection.odw import ODWDetector, ODWResult



def _fmt_loc(lat: float, lon: float) -> str:
    """Format lat/lon with correct N/S/E/W hemisphere labels."""
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"({abs(lat):.2f}{ns}, {abs(lon):.2f}{ew})"


# ── Internal MI computation (mirrors market.py pattern) ──────────────────────

def _weather_mi(x: np.ndarray, y: np.ndarray, min_n: int = 20) -> float:
    """Pearson-based MI estimate."""
    from fawp_index.core.estimators import mi_from_arrays
    return mi_from_arrays(x, y, min_n=min_n)


def _weather_null_floor(x, y, n_null: int, beta: float, min_n: int = 20) -> float:
    """Conservative null floor for weather MI."""
    from fawp_index.core.estimators import conservative_null_floor
    return conservative_null_floor(x, y, n_null=n_null, beta=beta, min_n=min_n)


def _compute_weather_mi_curves(
    pred_series:    np.ndarray,
    future_series:  np.ndarray,
    steer_series:   np.ndarray,
    tau_max:        int = 40,
    delta:          int = 5,
    epsilon:        float = 0.01,
    beta:           float = 0.99,
    n_null:         int = 100,
    min_n:          int = 20,
) -> tuple:
    """
    Compute null-corrected pred/steer MI curves and run ODW detector.
    Returns (odw_result, tau, pred_mi, steer_mi).
    """
    from fawp_index.detection.odw import ODWDetector

    n       = len(pred_series)
    tau_arr = np.arange(1, tau_max + 1)
    pred_mi  = np.zeros(len(tau_arr))
    steer_mi = np.zeros(len(tau_arr))

    for i, tau in enumerate(tau_arr):
        n_usable = n - max(delta, int(tau) + 1)
        if n_usable < min_n:
            continue

        # Pred: I(pred_t ; future_{t+delta})
        xp = pred_series[:n_usable]
        yp = future_series[delta:delta + n_usable]
        mn = min(len(xp), len(yp))
        raw_p  = _weather_mi(xp[:mn], yp[:mn], min_n)
        floor_p = _weather_null_floor(xp[:mn], yp[:mn], n_null, beta, min_n) if n_null > 0 else 0.0
        pred_mi[i] = max(0.0, raw_p - floor_p)

        # Steer: I(steer_t ; future_{t+tau+1})
        xs = steer_series[:n_usable]
        ys = future_series[int(tau) + 1:int(tau) + 1 + n_usable]
        mn2 = min(len(xs), len(ys))
        raw_s   = _weather_mi(xs[:mn2], ys[:mn2], min_n)
        floor_s = _weather_null_floor(xs[:mn2], ys[:mn2], n_null, beta, min_n) if n_null > 0 else 0.0
        steer_mi[i] = max(0.0, raw_s - floor_s)

    fail_rate = np.zeros(len(tau_arr))
    det = ODWDetector(epsilon=epsilon,
                      persistence_m=PERSISTENCE_RULE_M,
                      persistence_n=PERSISTENCE_RULE_N)
    odw = det.detect(tau=tau_arr, pred_corr=pred_mi,
                     steer_corr=steer_mi, fail_rate=fail_rate)
    return odw, tau_arr, pred_mi, steer_mi


# ── WeatherFAWPResult ─────────────────────────────────────────────────────────

@dataclass
class WeatherFAWPResult:
    """
    FAWP detection result for a weather/climate forecast system.

    Attributes
    ----------
    variable : str
        Meteorological variable (e.g. "temperature_2m", "precipitation").
    location : str
        Human-readable location or grid point label.
    odw_result : ODWResult
        Core detection output: fawp_found, tau_h_plus, tau_f, ODW, peak_gap_bits.
    tau : np.ndarray
        Tau grid used for detection.
    pred_mi : np.ndarray
        Corrected prediction MI per tau (forecast skill channel).
    steer_mi : np.ndarray
        Corrected steering MI per tau (intervention channel).
    skill_metric : str
        Name of the skill metric used (e.g. "correlation", "mae_reduction").
    n_obs : int
        Number of observations used.
    horizon_days : int
        Forecast horizon Δ in days.
    date_range : tuple of str
        (start_date, end_date) of the data.
    metadata : dict
        Source information, parameters, etc.
    """
    variable:      str
    location:      str
    odw_result:    ODWResult
    tau:           np.ndarray
    pred_mi:       np.ndarray
    steer_mi:      np.ndarray
    skill_metric:  str
    n_obs:         int
    horizon_days:  int
    date_range:    Tuple[str, str]
    metadata:      Dict = field(default_factory=dict)

    @property
    def fawp_found(self) -> bool:
        return self.odw_result.fawp_found

    @property
    def peak_gap_bits(self) -> float:
        return self.odw_result.peak_gap_bits

    @property
    def odw_start(self) -> Optional[int]:
        return self.odw_result.odw_start

    @property
    def odw_end(self) -> Optional[int]:
        return self.odw_result.odw_end

    def summary(self) -> str:
        lines = [
            "=" * 60,
            f"  FAWP Weather Result — {self.variable} @ {self.location}",
            "=" * 60,
            f"  FAWP detected    : {'YES 🔴' if self.fawp_found else 'NO ✅'}",
            f"  Peak gap         : {self.peak_gap_bits:.4f} bits",
            f"  Forecast horizon : {self.horizon_days} day(s)",
            f"  Skill metric     : {self.skill_metric}",
            f"  Observations     : {self.n_obs:,}",
            f"  Date range       : {self.date_range[0]} → {self.date_range[1]}",
        ]
        if self.fawp_found:
            lines += [
                f"  τ⁺ₕ (horizon)   : τ = {self.odw_result.tau_h_plus}",
                f"  τf  (cliff)      : τ = {self.odw_result.tau_f}",
                f"  ODW              : τ = {self.odw_result.odw_start}–{self.odw_result.odw_end}",
                "",
                "  Interpretation:",
                "  The forecast system retains predictive skill, but the",
                "  ability to intervene on the outcome has already collapsed.",
                "  This is the FAWP regime — prediction without control.",
            ]
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "variable":      self.variable,
            "location":      self.location,
            "fawp_found":    self.fawp_found,
            "peak_gap_bits": round(self.peak_gap_bits, 6),
            "odw_start":     self.odw_start,
            "odw_end":       self.odw_end,
            "tau_h_plus":    self.odw_result.tau_h_plus,
            "tau_f":         self.odw_result.tau_f,
            "horizon_days":  self.horizon_days,
            "skill_metric":  self.skill_metric,
            "n_obs":         self.n_obs,
            "date_range":    list(self.date_range),
            "metadata":      self.metadata,
            "fawp_index_version": _VERSION,
        }


# ── Core detection functions ──────────────────────────────────────────────────

def fawp_from_forecast(
    forecast:      np.ndarray,
    observed:      np.ndarray,
    intervention:  np.ndarray,
    horizon_days:  int = 5,
    variable:      str = "temperature",
    location:      str = "unknown",
    tau_max:       int = MARKET_TAU_MAX,
    epsilon:       float = EPSILON_STEERING_RAW,
    beta:          float = BETA_NULL_QUANTILE,
    n_null:        int = 100,
    skill_metric:  str = "correlation",
    dates:         Optional[pd.DatetimeIndex] = None,
    metadata:      Optional[dict] = None,
) -> WeatherFAWPResult:
    """
    Detect FAWP from NWP forecast output and observations.

    Maps the weather domain to the FAWP framework:
      - Prediction channel: I(forecast_t ; observed_{t+Δ})
        How much does today's forecast predict the future state?
      - Steering channel:   I(intervention_t ; observed_{t+τ+1})
        How much does an intervention still influence outcomes at lag τ?

    Parameters
    ----------
    forecast : array-like, shape (N,)
        Model forecast values (e.g. temperature forecast issued at t).
    observed : array-like, shape (N,)
        Verification observations (actual measurements at t+Δ).
    intervention : array-like, shape (N,)
        Proxy for steering/intervention (e.g. forecast adjustment, model nudge,
        warning issued, cloud-seeding event flag, ensemble spread, etc.)
    horizon_days : int
        Forecast horizon Δ. Default 5 (medium-range weather).
    variable : str
        Variable name for labelling (e.g. "temperature_2m", "precipitation_mm").
    location : str
        Location label (e.g. "London UK", "50.0N 0.0E").
    tau_max : int
        Maximum lag τ for steering MI. Default 40.
    epsilon : float
        Detectability threshold (bits). Default 0.01.
    beta : float
        Null-quantile level. Default 0.99.
    n_null : int
        Null permutations for floor estimation. Default 100.
    skill_metric : str
        Label describing the skill metric used.
    dates : DatetimeIndex, optional
        Date index for the time series.
    metadata : dict, optional
        Additional metadata to attach to the result.

    Returns
    -------
    WeatherFAWPResult

    Example
    -------
    ::

        import numpy as np
        from fawp_index.weather import fawp_from_forecast

        rng = np.random.default_rng(42)
        n = 1000
        true_temp   = 15 + 5 * np.sin(np.linspace(0, 6*np.pi, n)) + rng.normal(0, 1, n)
        forecast    = true_temp + rng.normal(0, 0.5, n)
        observed    = true_temp
        intervention = rng.normal(0, 0.1, n)   # e.g. model nudge magnitude

        result = fawp_from_forecast(forecast, observed, intervention,
                                    horizon_days=5, variable="temperature_2m",
                                    location="synthetic_NWP")
        print(result.summary())
    """
    forecast     = np.asarray(forecast,     dtype=float)
    observed     = np.asarray(observed,     dtype=float)
    intervention = np.asarray(intervention, dtype=float)

    n = min(len(forecast), len(observed), len(intervention))
    forecast     = forecast[:n]
    observed     = observed[:n]
    intervention = intervention[:n]

    # Forecast error as the prediction channel
    # I(forecast_t ; observed_{t+Δ})
    pred_series   = forecast
    future_series = observed

    date_start = str(dates[0].date()) if dates is not None else "unknown"
    date_end   = str(dates[-1].date()) if dates is not None else "unknown"

    # Run ODW detector
    odw_result, tau, pred_mi, steer_mi = _compute_weather_mi_curves(
        pred_series   = pred_series,
        future_series = future_series,
        steer_series  = intervention,
        tau_max       = tau_max,
        delta         = horizon_days,
        epsilon       = epsilon,
        beta          = beta,
        n_null        = n_null,
    )

    return WeatherFAWPResult(
        variable     = variable,
        location     = location,
        odw_result   = odw_result,
        tau          = tau,
        pred_mi      = pred_mi,
        steer_mi     = steer_mi,
        skill_metric = skill_metric,
        n_obs        = n,
        horizon_days = horizon_days,
        date_range   = (date_start, date_end),
        metadata     = metadata or {},
    )


def fawp_from_skill_series(
    skill_series:         np.ndarray,
    intervention_series:  np.ndarray,
    skill_name:           str = "anomaly_correlation",
    variable:             str = "geopotential",
    location:             str = "unknown",
    tau_max:              int = MARKET_TAU_MAX,
    epsilon:              float = EPSILON_STEERING_RAW,
    beta:                 float = BETA_NULL_QUANTILE,
    n_null:               int = 100,
    dates:                Optional[pd.DatetimeIndex] = None,
    metadata:             Optional[dict] = None,
) -> WeatherFAWPResult:
    """
    Detect FAWP from pre-computed skill and intervention time series.

    Use this when you already have a rolling skill metric (e.g. 30-day
    rolling anomaly correlation of 500hPa geopotential height, or a
    time series of RMSE reduction) and a parallel intervention series.

    Parameters
    ----------
    skill_series : array-like, shape (N,)
        Time series of forecast skill (higher = better forecast, e.g. ACC).
        This is the PREDICTION channel proxy.
    intervention_series : array-like, shape (N,)
        Time series of intervention effectiveness (e.g. ensemble spread,
        model correction magnitude, warning response rate).
        This is the STEERING channel proxy.
    skill_name : str
        Name of the skill metric (for labelling).
    variable, location, tau_max, epsilon, beta, n_null, dates, metadata
        See fawp_from_forecast for parameter descriptions.

    Returns
    -------
    WeatherFAWPResult

    Example
    -------
    ::

        import numpy as np
        from fawp_index.weather import fawp_from_skill_series

        # Rolling ACC for a climate model
        n = 500
        rng = np.random.default_rng(0)
        acc     = 0.8 - 0.001 * np.arange(n) + rng.normal(0, 0.05, n)
        spread  = 0.5 * np.exp(-0.01 * np.arange(n)) + rng.normal(0, 0.02, n)

        result = fawp_from_skill_series(acc, spread,
                                        skill_name="ACC_500hPa",
                                        variable="geopotential_500hPa")
        print(result.summary())
    """
    skill    = np.asarray(skill_series,        dtype=float)
    steering = np.asarray(intervention_series, dtype=float)

    n = min(len(skill), len(steering))
    skill    = skill[:n]
    steering = steering[:n]

    date_start = str(dates[0].date()) if dates is not None else "unknown"
    date_end   = str(dates[-1].date()) if dates is not None else "unknown"

    odw_result, tau, pred_mi, steer_mi = _compute_weather_mi_curves(
        pred_series   = skill[:-1],
        future_series = skill[1:],
        steer_series  = steering[:-1],
        tau_max       = tau_max,
        delta         = 1,
        epsilon       = epsilon,
        beta          = beta,
        n_null        = n_null,
    )

    return WeatherFAWPResult(
        variable     = variable,
        location     = location,
        odw_result   = odw_result,
        tau          = tau,
        pred_mi      = pred_mi,
        steer_mi     = steer_mi,
        skill_metric = skill_name,
        n_obs        = n,
        horizon_days = 1,
        date_range   = (date_start, date_end),
        metadata     = metadata or {},
    )


def fawp_from_open_meteo(
    latitude:        float,
    longitude:       float,
    variable:        str = "temperature_2m",
    start_date:      str = "2020-01-01",
    end_date:        str = "2024-12-31",
    horizon_days:    int = 5,
    tau_max:         int = 30,
    epsilon:         float = EPSILON_STEERING_RAW,
    n_null:          int = 100,
) -> WeatherFAWPResult:
    """
    Fetch ERA5 reanalysis data from Open-Meteo (free, no API key) and
    run FAWP detection automatically.

    Uses ERA5 hourly data resampled to daily. The steering channel is
    constructed from the day-over-day change in the variable — a proxy
    for how much "corrective information" each day adds.

    Parameters
    ----------
    latitude, longitude : float
        Grid point coordinates.
    variable : str
        Meteorological variable. Options:
        "temperature_2m", "precipitation_sum", "wind_speed_10m",
        "surface_pressure", "cloud_cover", "et0_fao_evapotranspiration"
    start_date, end_date : str
        Date range "YYYY-MM-DD".
    horizon_days : int
        Forecast horizon Δ for the prediction channel.
    tau_max : int
        Maximum steering lag τ.
    epsilon : float
        Detectability threshold (bits).
    n_null : int
        Null permutations.

    Returns
    -------
    WeatherFAWPResult

    Requires
    --------
    pip install openmeteo-requests requests-cache retry-requests

    Example
    -------
    ::

        from fawp_index.weather import fawp_from_open_meteo

        result = fawp_from_open_meteo(
            latitude=51.5, longitude=-0.1,   # London
            variable="temperature_2m",
            start_date="2015-01-01",
            end_date="2024-12-31",
            horizon_days=7,
        )
        print(result.summary())

        # Example: NYC precipitation
        result = fawp_from_open_meteo(
            latitude=40.71, longitude=-74.01,
            variable="precipitation_sum",
            start_date="2010-01-01",
            end_date="2024-12-31",
        )
    """
    try:
        import openmeteo_requests
        import requests_cache
        from retry_requests import retry
    except ImportError:
        raise ImportError(
            "Open-Meteo client not installed.\n"
            "Install with: pip install openmeteo-requests requests-cache retry-requests\n"
            "Or: pip install 'fawp-index[weather]'"
        )

    # ── Fetch ERA5 reanalysis ──────────────────────────────────────────────
    cache = requests_cache.CachedSession(".fawp_weather_cache", expire_after=3600)
    session = retry(cache, retries=5, backoff_factor=0.2)
    om = openmeteo_requests.Client(session=session)

    # Map variable names to Open-Meteo ERA5 hourly names
    _ERA5_MAP = {
        "temperature_2m":                "temperature_2m",
        "precipitation_sum":             "precipitation",
        "wind_speed_10m":                "wind_speed_10m",
        "surface_pressure":              "surface_pressure",
        "cloud_cover":                   "cloud_cover",
        "et0_fao_evapotranspiration":    "et0_fao_evapotranspiration",
        "shortwave_radiation":           "shortwave_radiation",
    }
    api_var = _ERA5_MAP.get(variable, variable)

    print(f"Fetching ERA5: {variable} @ ({latitude:.2f}, {longitude:.2f}) "
          f"{start_date} → {end_date}")

    url    = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":        latitude,
        "longitude":       longitude,
        "start_date":      start_date,
        "end_date":        end_date,
        "daily":           api_var,
        "timezone":        "UTC",
    }

    # Try daily first; fall back to hourly resampled
    try:
        responses = om.weather_api(url, params=params)
        resp = responses[0]
        daily = resp.Daily()
        values = daily.Variables(0).ValuesAsNumpy().astype(float)
        times  = pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left",
        ).tz_localize(None)
    except Exception as exc:
        raise RuntimeError(
            f"Open-Meteo fetch failed: {exc}\n"
            "Check your internet connection and variable name."
        ) from exc

    # ── Build FAWP channels ───────────────────────────────────────────────
    values = np.where(np.isfinite(values), values, np.nan)
    # Forward-fill short gaps
    series = pd.Series(values, index=times).interpolate(method="linear", limit=3)
    values = series.values

    n = len(values) - horizon_days
    if n < 100:
        raise ValueError(
            f"Not enough data after alignment: {n} rows. "
            f"Try a longer date range."
        )

    # Prediction channel: today's value → value in horizon_days
    pred_series   = values[:n]
    future_series = values[horizon_days:horizon_days + n]

    # Steering channel proxy: day-over-day change (forecast correction signal)
    # This captures how much "intervention-like information" enters each day
    day_changes  = np.diff(values)
    steer_proxy  = day_changes[:n] if len(day_changes) >= n else np.zeros(n)

    result = fawp_from_forecast(
        forecast     = pred_series,
        observed     = future_series,
        intervention = steer_proxy,
        horizon_days = horizon_days,
        variable     = variable,
        location     = _fmt_loc(latitude, longitude),
        tau_max      = tau_max,
        epsilon      = epsilon,
        n_null       = n_null,
        skill_metric = "ERA5_reanalysis_MI",
        dates        = times[:n],
        metadata     = {
            "source":    "Open-Meteo ERA5",
            "latitude":  latitude,
            "longitude": longitude,
            "variable":  variable,
            "start":     start_date,
            "end":       end_date,
        },
    )
    return result


def scan_weather_grid(
    locations:    List[Dict],
    variable:     str = "temperature_2m",
    start_date:   str = "2015-01-01",
    end_date:     str = "2024-12-31",
    horizon_days: int = 5,
    tau_max:      int = 30,
    n_null:       int = 50,
    verbose:      bool = True,
) -> List[WeatherFAWPResult]:
    """
    Scan multiple grid points or weather stations for FAWP regimes.

    Parameters
    ----------
    locations : list of dict
        Each dict must have "lat", "lon", and optionally "name".
        Example: [{"lat": 51.5, "lon": -0.1, "name": "London"}]
    variable : str
        ERA5 variable to scan.
    start_date, end_date : str
        Date range.
    horizon_days : int
        Forecast horizon.
    tau_max : int
        Maximum steering lag.
    n_null : int
        Null permutations (keep low for grid scans).
    verbose : bool
        Print progress.

    Returns
    -------
    list of WeatherFAWPResult, sorted by peak_gap_bits descending.

    Example
    -------
    ::

        from fawp_index.weather import scan_weather_grid

        locations = [
            {"lat": 51.5,  "lon": -0.1,  "name": "London"},
            {"lat": 48.9,  "lon":  2.4,  "name": "Paris"},
            {"lat": 40.7,  "lon": -74.0, "name": "New York"},
            {"lat": 35.7,  "lon": 139.7, "name": "Tokyo"},
            {"lat": -33.9, "lon": 151.2, "name": "Sydney"},
        ]
        results = scan_weather_grid(locations, variable="temperature_2m",
                                    start_date="2015-01-01",
                                    end_date="2024-12-31")
        for r in results:
            flag = "🔴 FAWP" if r.fawp_found else "—"
            print(f"{r.location:<20} {flag}  gap={r.peak_gap_bits:.4f}b")
    """
    results = []
    for i, loc in enumerate(locations):
        lat  = loc["lat"]
        lon  = loc["lon"]
        name = loc.get("name", f"({lat:.1f},{lon:.1f})")
        if verbose:
            print(f"[{i+1}/{len(locations)}] Scanning {name}…")
        try:
            r = fawp_from_open_meteo(
                latitude     = lat,
                longitude    = lon,
                variable     = variable,
                start_date   = start_date,
                end_date     = end_date,
                horizon_days = horizon_days,
                tau_max      = tau_max,
                n_null       = n_null,
            )
            # Override location with friendly name
            r = WeatherFAWPResult(
                variable     = r.variable,
                location     = name,
                odw_result   = r.odw_result,
                tau          = r.tau,
                pred_mi      = r.pred_mi,
                steer_mi     = r.steer_mi,
                skill_metric = r.skill_metric,
                n_obs        = r.n_obs,
                horizon_days = r.horizon_days,
                date_range   = r.date_range,
                metadata     = {**r.metadata, "name": name},
            )
            results.append(r)
            if verbose:
                flag = "🔴 FAWP" if r.fawp_found else "✅ clear"
                print(f"       → {flag}  gap={r.peak_gap_bits:.4f}b")
        except Exception as exc:
            if verbose:
                print(f"       ⚠  Failed: {exc}")

    return sorted(results, key=lambda r: r.peak_gap_bits, reverse=True)
