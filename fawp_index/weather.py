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


def _deseasonalise(series: np.ndarray, period: int = 365) -> np.ndarray:
    """
    Remove the annual seasonal cycle using a rolling mean.

    Subtracts the running centred mean over `period` days from the raw
    series. This removes the dominant annual cycle so the FAWP detector
    operates on anomalies rather than the raw signal.

    Parameters
    ----------
    series : np.ndarray
    period : int
        Seasonal period in samples. 365 for daily ERA5 data.

    Returns
    -------
    np.ndarray — anomaly series (same length as input)
    """
    import pandas as _pd
    s = _pd.Series(series)
    # Centred rolling mean with min_periods so edges are handled
    trend = s.rolling(window=period, center=True, min_periods=period // 2).mean()
    # Fill any remaining NaNs at edges with the global mean
    trend = trend.fillna(s.mean())
    return (s - trend).values



# ── Internal MI computation (mirrors market.py pattern) ──────────────────────

def _weather_mi(
    x: np.ndarray,
    y: np.ndarray,
    min_n: int = 20,
    estimator: str = "pearson",
) -> float:
    """
    MI estimate for weather/seismic data.

    Parameters
    ----------
    estimator : str
        "pearson" (default, fast, Gaussian assumption) or
        "knn"     (non-parametric, slower, better for non-Gaussian data
                  such as seismic energy or precipitation — see E9 methods).
    """
    if len(x) < min_n or len(y) < min_n:
        return 0.0
    if estimator == "knn":
        try:
            from sklearn.feature_selection import mutual_info_regression
            return float(mutual_info_regression(
                x.reshape(-1, 1), y, n_neighbors=3, random_state=42
            )[0])
        except ImportError:
            pass  # fall back to pearson if sklearn not installed
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
    estimator:      str = "pearson",
) -> tuple:
    """
    Compute null-corrected pred/steer MI curves and run ODW detector.
    Returns (odw_result, tau, pred_mi, steer_mi).

    Parameters
    ----------
    estimator : str
        "pearson" (default) or "knn" (non-Gaussian, requires sklearn).
        Use "knn" for seismic energy, precipitation, or other heavy-tailed data.
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
        raw_p  = _weather_mi(xp[:mn], yp[:mn], min_n, estimator)
        floor_p = _weather_null_floor(xp[:mn], yp[:mn], n_null, beta, min_n) if n_null > 0 else 0.0
        pred_mi[i] = max(0.0, raw_p - floor_p)

        # Steer: I(steer_t ; future_{t+tau+1})
        xs = steer_series[:n_usable]
        ys = future_series[int(tau) + 1:int(tau) + 1 + n_usable]
        mn2 = min(len(xs), len(ys))
        raw_s   = _weather_mi(xs[:mn2], ys[:mn2], min_n, estimator)
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

    def explain(self) -> str:
        """
        Plain-English explanation of the FAWP result.
        Suitable for non-technical audiences.
        """
        if self.fawp_found:
            return (
                f"FAWP detected for {self.variable} at {self.location}.\n"
                f"The forecast system retains predictive skill "
                f"(agency horizon τ⁺ₕ = {self.odw_result.tau_h_plus}), "
                f"but the ability to intervene and change outcomes has already collapsed.\n"
                f"Peak leverage gap: {self.peak_gap_bits:.4f} bits. "
                f"Operational Detection Window: τ = {self.odw_start}–{self.odw_end}.\n"
                f"In plain terms: you can see it coming better than you can still change it."
            )
        else:
            return (
                f"No FAWP regime detected for {self.variable} at {self.location}.\n"
                f"Predictive and steering coupling collapse together — "
                f"no persistent pre-cliff gap was found.\n"
                f"Peak gap: {self.peak_gap_bits:.4f} bits."
            )

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



# ── ERA5 variable mapping + shared hourly→daily fetch helper ──────────────
# All ERA5 variables must be fetched as HOURLY and resampled to daily.
# Sending them as `daily=` causes "invalid String value" errors from the API.

_HOURLY_ERA5_MAP = {
    "temperature_2m":             "temperature_2m",
    "precipitation_sum":          "precipitation",
    "wind_speed_10m":             "wind_speed_10m",
    "surface_pressure":           "surface_pressure",
    "cloud_cover":                "cloud_cover",
    "et0_fao_evapotranspiration": "et0_fao_evapotranspiration",
    "shortwave_radiation":        "shortwave_radiation",
    # aliases
    "temp":   "temperature_2m",
    "precip": "precipitation",
    "wind":   "wind_speed_10m",
}

# Variables that should be summed (not averaged) when resampling hourly → daily
_SUM_VARS = {"precipitation_sum", "et0_fao_evapotranspiration", "precipitation"}


def _fetch_openmeteo_daily_series(om, latitude, longitude, start_date, end_date, variable):
    """
    Fetch a single ERA5 variable as HOURLY data and resample to daily.

    All UI-exposed variables (temperature_2m, precipitation_sum, etc.) must go
    through the hourly endpoint — the ERA5 archive does not support them as
    `daily=` parameters, which causes ``invalid String value`` errors.

    Parameters
    ----------
    om : openmeteo_requests.Client
    latitude, longitude : float
    start_date, end_date : str  YYYY-MM-DD
    variable : str  UI variable name (e.g. "temperature_2m", "precipitation_sum")

    Returns
    -------
    times : pd.DatetimeIndex  (tz-naive, daily)
    values : np.ndarray       (daily values, NaN-filled gaps interpolated)
    """
    api_var = _HOURLY_ERA5_MAP.get(variable, variable)

    resp = om.weather_api(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude":   latitude,
            "longitude":  longitude,
            "start_date": start_date,
            "end_date":   end_date,
            "hourly":     api_var,
            "timezone":   "UTC",
        },
    )[0]

    hourly = resp.Hourly()
    raw    = hourly.Variables(0).ValuesAsNumpy().astype(float)
    times_hr = pd.date_range(
        start=pd.to_datetime(hourly.Time(),    unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    )

    s = pd.Series(np.where(np.isfinite(raw), raw, np.nan), index=times_hr)

    # Sum for accumulation variables, mean for everything else
    if variable in _SUM_VARS:
        daily = s.resample("D").sum(min_count=1)
    else:
        daily = s.resample("D").mean()

    daily = daily.interpolate(method="linear", limit=3)
    return daily.index.tz_localize(None), daily.values

def fawp_from_open_meteo(
    latitude:          float,
    longitude:         float,
    variable:          str = "temperature_2m",
    start_date:        str = "2020-01-01",
    end_date:          str = "2024-12-31",
    horizon_days:      int = 5,
    tau_max:           int = 30,
    epsilon:           float = EPSILON_STEERING_RAW,
    n_null:            int = 100,
    remove_seasonality: bool = False,
    estimator:          str  = "pearson",
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
    remove_seasonality : bool
        If True, subtract the 365-day rolling mean before computing MI.
        Recommended for temperature to remove the annual cycle and detect
        FAWP in the anomaly signal rather than the raw signal. Default False.

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

    # Fetch ERA5 via shared hourly→daily helper
    # (All variables must use hourly endpoint; daily= causes API errors)
    try:
        times, values = _fetch_openmeteo_daily_series(
            om, latitude, longitude, start_date, end_date, variable
        )
    except Exception as exc:
        raise RuntimeError(
            f"Open-Meteo fetch failed for {variable!r}: {exc}\n"
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

    # Optional: remove annual seasonal cycle
    if remove_seasonality:
        values = _deseasonalise(values, period=365)

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


# ── Convenience helpers ────────────────────────────────────────────────────────

def fetch_openmeteo(
    location: str,
    days: int = 14,
    var: str = "temperature_2m",
    end_date: Optional[str] = None,
) -> "pd.DataFrame":
    """
    Fetch ERA5 reanalysis from Open-Meteo for a named city or lat/lon string.

    Returns a DataFrame with columns: pred, future, action, obs
    ready to pass directly to FAWPAlphaIndexV2 or fawp_from_forecast().

    Parameters
    ----------
    location : str
        City name (e.g. "London", "Paris", "New York") or "lat,lon" string.
    days : int
        Number of days of data to fetch (counting back from end_date).
    var : str
        ERA5 variable name. Options: temperature_2m, precipitation_sum,
        wind_speed_10m, surface_pressure, cloud_cover, shortwave_radiation.
    end_date : str, optional
        End date "YYYY-MM-DD". Defaults to yesterday.

    Returns
    -------
    pd.DataFrame with columns: pred, future, action, obs, date

    Requires
    --------
    pip install "fawp-index[weather]"

    Example
    -------
    ::

        from fawp_index.weather import fetch_openmeteo, to_fawp_dataframe

        df = fetch_openmeteo("London", days=365, var="temperature_2m")
        print(df.head())
        # pred    future   action    obs         date
        # 12.3    13.1     0.8       12.3   2024-01-01
    """
    import datetime as _dt

    # Resolve location
    _CITY_COORDS = {
        "london":    (51.50, -0.10),
        "paris":     (48.86,  2.35),
        "new york":  (40.71,-74.01),
        "newyork":   (40.71,-74.01),
        "tokyo":     (35.69,139.69),
        "sydney":   (-33.87,151.21),
        "dubai":     (25.20, 55.27),
        "berlin":    (52.52, 13.40),
        "chicago":   (41.88,-87.63),
        "mumbai":    (19.08, 72.88),
        "beijing":   (39.91,116.40),
        "moscow":    (55.75, 37.62),
        "toronto":   (43.65,-79.38),
        "cairo":     (30.04, 31.24),
        "singapore": ( 1.35,103.82),
        "amsterdam": (52.37,  4.90),
        "madrid":    (40.42, -3.70),
        "rome":      (41.90, 12.50),
        "bangkok":   (13.75,100.52),
        "seoul":     (37.57,126.98),
    }

    loc_key = location.strip().lower()
    if "," in loc_key:
        parts = loc_key.split(",")
        try:
            lat, lon = float(parts[0].strip()), float(parts[1].strip())
        except ValueError:
            raise ValueError(f"Cannot parse lat,lon from: {location!r}")
    elif loc_key in _CITY_COORDS:
        lat, lon = _CITY_COORDS[loc_key]
    else:
        raise ValueError(
            f"Unknown city: {location!r}. Use lat,lon string or one of: "
            + ", ".join(_CITY_COORDS.keys())
        )

    if end_date is None:
        end_date = (_dt.date.today() - _dt.timedelta(days=1)).isoformat()
    start_dt  = _dt.date.fromisoformat(end_date) - _dt.timedelta(days=days)
    start_date = start_dt.isoformat()

    # Use existing Open-Meteo fetch logic
    try:
        import openmeteo_requests
        import requests_cache
        from retry_requests import retry
    except ImportError:
        raise ImportError(
            "Install weather dependencies: pip install \'fawp-index[weather]\'"
        )

    cache   = requests_cache.CachedSession(".fawp_weather_cache", expire_after=3600)
    session = retry(cache, retries=5, backoff_factor=0.2)
    om      = openmeteo_requests.Client(session=session)

    # Fetch via shared hourly→daily helper (daily= endpoint causes API errors)
    times, values = _fetch_openmeteo_daily_series(om, lat, lon, start_date, end_date, var)
    series = values
    n      = len(series) - 1
    df = pd.DataFrame({
        "date":   times[:n],
        "pred":   series[:n],
        "future": series[1:],
        "action": np.diff(series),   # day-over-day change = intervention proxy
        "obs":    series[:n],
    })
    df.attrs["variable"] = var
    df.attrs["location"] = location
    df.attrs["lat"]      = lat
    df.attrs["lon"]      = lon
    return df


def to_fawp_dataframe(
    df: "pd.DataFrame",
    pred_col: str = "pred",
    future_col: str = "future",
    action_col: str = "action",
    obs_col: str = "obs",
) -> "pd.DataFrame":
    """
    Normalise any DataFrame into the standard FAWP input format.

    Renames columns to pred / future / action / obs and drops NaNs.
    Use when you have your own forecast/observation data and want
    to pass it to fawp_from_forecast().

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with forecast / observation columns.
    pred_col, future_col, action_col, obs_col : str
        Column names in df to map to the FAWP input format.

    Returns
    -------
    pd.DataFrame with columns: pred, future, action, obs

    Example
    -------
    ::

        from fawp_index.weather import fetch_openmeteo, to_fawp_dataframe, fawp_from_forecast

        raw = fetch_openmeteo("Tokyo", days=730, var="temperature_2m")
        df  = to_fawp_dataframe(raw)

        result = fawp_from_forecast(
            forecast     = df["pred"].values,
            observed     = df["future"].values,
            intervention = df["action"].values,
            variable     = "temperature_2m",
            location     = "Tokyo",
        )
        result.explain()
    """
    rename = {pred_col: "pred", future_col: "future",
              action_col: "action", obs_col: "obs"}
    out = df.rename(columns=rename)[["pred", "future", "action", "obs"]].dropna()
    return out.reset_index(drop=True)







def fawp_from_nwp_csvs(
    forecast_path:   str,
    observed_path:   str,
    forecast_col:    str = "forecast",
    observed_col:    str = "observed",
    date_col:        str = "date",
    intervention_col: Optional[str] = None,
    variable:        str = "temperature",
    location:        str = "uploaded",
    horizon_days:    int = 1,
    tau_max:         int = 40,
    epsilon:         float = 0.01,
    n_null:          int = 100,
    remove_seasonality: bool = False,
    estimator:          str  = "pearson",
) -> WeatherFAWPResult:
    """
    Run FAWP detection on real NWP forecast vs observation data from CSV files.

    This is the scientific-credibility mode — uses actual forecast verification
    data rather than the ERA5 proxy. Upload a model forecast CSV and an
    observation CSV, and FAWP is computed on the real skill gap.

    Parameters
    ----------
    forecast_path : str   Path to forecast CSV.
    observed_path : str   Path to observation CSV.
    forecast_col : str    Column name for forecast values.
    observed_col : str    Column name for observed values.
    date_col : str        Column name for dates (used to align the two files).
    intervention_col : str, optional
        Column in forecast CSV to use as intervention proxy (e.g. ensemble spread,
        model correction). If None, uses day-over-day forecast change.
    variable : str        Variable label for the result.
    location : str        Location label for the result.
    horizon_days : int    Forecast horizon Δ (for labelling).
    tau_max, epsilon, n_null : detection settings.
    remove_seasonality : bool

    Returns
    -------
    WeatherFAWPResult

    Example
    -------
    ::

        from fawp_index.weather import fawp_from_nwp_csvs

        result = fawp_from_nwp_csvs(
            forecast_path    = "ecmwf_t2m_forecast.csv",
            observed_path    = "station_t2m_obs.csv",
            forecast_col     = "t2m_forecast",
            observed_col     = "t2m_obs",
            date_col         = "date",
            intervention_col = "ensemble_spread",   # or None
            variable         = "temperature_2m",
            location         = "London Heathrow",
            horizon_days     = 5,
        )
        print(result.summary())
        print(result.explain())
    """
    fc_df  = pd.read_csv(forecast_path, parse_dates=[date_col])
    obs_df = pd.read_csv(observed_path, parse_dates=[date_col])

    # Align on date
    merged = pd.merge(fc_df, obs_df, on=date_col, how="inner").sort_values(date_col)

    if forecast_col not in merged.columns:
        raise ValueError(f"Forecast column '{forecast_col}' not found. "
                         f"Available: {merged.columns.tolist()}")
    if observed_col not in merged.columns:
        raise ValueError(f"Observed column '{observed_col}' not found. "
                         f"Available: {merged.columns.tolist()}")

    forecast = merged[forecast_col].values.astype(float)
    observed = merged[observed_col].values.astype(float)

    if intervention_col and intervention_col in merged.columns:
        intervention = merged[intervention_col].values.astype(float)
    else:
        # Day-over-day forecast change as proxy
        intervention = np.concatenate([[0], np.diff(forecast)])

    if remove_seasonality:
        forecast     = _deseasonalise(forecast)
        observed     = _deseasonalise(observed)

    dates = merged[date_col]
    date_start = str(dates.iloc[0].date())  if hasattr(dates.iloc[0], 'date') else str(dates.iloc[0])[:10]
    date_end   = str(dates.iloc[-1].date()) if hasattr(dates.iloc[-1], 'date') else str(dates.iloc[-1])[:10]

    return fawp_from_forecast(
        forecast     = forecast,
        observed     = observed,
        intervention = intervention,
        horizon_days = horizon_days,
        variable     = variable,
        location     = location,
        tau_max      = tau_max,
        epsilon      = epsilon,
        n_null       = n_null,
        skill_metric = "NWP_verification",
        dates        = pd.DatetimeIndex(dates),
        metadata     = {
            "source":        "NWP CSV upload",
            "forecast_file": str(forecast_path),
            "observed_file": str(observed_path),
            "n_rows":        len(merged),
        },
    )


def compare_locations(
    location_a:      dict,
    location_b:      dict,
    variable:        str = "temperature_2m",
    start_date:      str = "2010-01-01",
    end_date:        str = "2024-12-31",
    horizon_days:    int = 7,
    tau_max:         int = 30,
    epsilon:         float = 0.01,
    n_null:          int = 50,
    remove_seasonality: bool = False,
) -> "tuple[WeatherFAWPResult, WeatherFAWPResult]":
    """
    Compare FAWP detection between two locations for the same variable.

    Parameters
    ----------
    location_a, location_b : dict
        Each must have "lat", "lon", and optionally "name".
        Example: {"lat": 51.5, "lon": -0.1, "name": "London"}
    variable, start_date, end_date, horizon_days, tau_max, epsilon, n_null,
    remove_seasonality : see fawp_from_open_meteo()

    Returns
    -------
    tuple of (WeatherFAWPResult, WeatherFAWPResult)

    Example
    -------
    ::

        from fawp_index.weather import compare_locations

        r_lon, r_nyc = compare_locations(
            {"lat": 51.5, "lon": -0.1,  "name": "London"},
            {"lat": 40.7, "lon": -74.0, "name": "New York"},
            variable="temperature_2m",
        )
        print(r_lon.summary())
        print(r_nyc.summary())
    """
    def _run(loc):
        return fawp_from_open_meteo(
            latitude     = loc["lat"],
            longitude    = loc["lon"],
            variable     = variable,
            start_date   = start_date,
            end_date     = end_date,
            horizon_days = horizon_days,
            tau_max      = tau_max,
            epsilon      = epsilon,
            n_null       = n_null,
        )

    name_a = location_a.get("name", _fmt_loc(location_a["lat"], location_a["lon"]))
    name_b = location_b.get("name", _fmt_loc(location_b["lat"], location_b["lon"]))

    r_a = _run(location_a)
    r_b = _run(location_b)

    # Override location labels with friendly names
    r_a = WeatherFAWPResult(
        variable=r_a.variable, location=name_a, odw_result=r_a.odw_result,
        tau=r_a.tau, pred_mi=r_a.pred_mi, steer_mi=r_a.steer_mi,
        skill_metric=r_a.skill_metric, n_obs=r_a.n_obs,
        horizon_days=r_a.horizon_days, date_range=r_a.date_range,
        metadata={**r_a.metadata, "name": name_a},
    )
    r_b = WeatherFAWPResult(
        variable=r_b.variable, location=name_b, odw_result=r_b.odw_result,
        tau=r_b.tau, pred_mi=r_b.pred_mi, steer_mi=r_b.steer_mi,
        skill_metric=r_b.skill_metric, n_obs=r_b.n_obs,
        horizon_days=r_b.horizon_days, date_range=r_b.date_range,
        metadata={**r_b.metadata, "name": name_b},
    )
    return r_a, r_b


def fawp_rolling_timeline(
    latitude:        float,
    longitude:       float,
    variable:        str = "temperature_2m",
    start_date:      str = "2000-01-01",
    end_date:        str = "2024-12-31",
    window_years:    int = 2,
    step_months:     int = 6,
    horizon_days:    int = 7,
    tau_max:         int = 30,
    epsilon:         float = 0.01,
    n_null:          int = 20,
    remove_seasonality: bool = False,
) -> "pd.DataFrame":
    """
    Compute FAWP metrics across rolling time windows for a single location.

    Slides a window of `window_years` years across the full date range,
    stepping by `step_months` months, and computes FAWP detection for each
    window. Returns a DataFrame showing how peak_gap_bits, fawp_found,
    odw_start, and odw_end evolve over time.

    Answers: "Is FAWP getting worse at this location over time?"

    Parameters
    ----------
    latitude, longitude : float
    variable : str
    start_date, end_date : str   Full date range to scan.
    window_years : int           Rolling window size in years. Default 2.
    step_months : int            Step between windows in months. Default 6.
    horizon_days : int           Forecast horizon Δ.
    tau_max, epsilon, n_null : detection settings
    remove_seasonality : bool

    Returns
    -------
    pd.DataFrame with columns:
        window_start, window_end, fawp_found, peak_gap_bits,
        odw_start, odw_end, tau_h_plus, tau_f

    Example
    -------
    ::

        from fawp_index.weather import fawp_rolling_timeline

        df = fawp_rolling_timeline(
            51.5, -0.1, variable="temperature_2m",
            start_date="2000-01-01", end_date="2024-12-31",
            window_years=2, step_months=6,
        )
        print(df[df.fawp_found])
    """
    import datetime as _dt

    start = _dt.date.fromisoformat(start_date)
    end   = _dt.date.fromisoformat(end_date)
    win   = _dt.timedelta(days=window_years * 365)
    step  = _dt.timedelta(days=step_months * 30)

    rows = []
    cursor = start
    while cursor + win <= end:
        w_start = cursor.isoformat()
        w_end   = (cursor + win).isoformat()
        try:
            r = fawp_from_open_meteo(
                latitude     = latitude,
                longitude    = longitude,
                variable     = variable,
                start_date   = w_start,
                end_date     = w_end,
                horizon_days = horizon_days,
                tau_max      = tau_max,
                epsilon      = epsilon,
                n_null       = n_null,
            )
            rows.append({
                "window_start":  w_start,
                "window_end":    w_end,
                "fawp_found":    r.fawp_found,
                "peak_gap_bits": r.peak_gap_bits,
                "odw_start":     r.odw_start,
                "odw_end":       r.odw_end,
                "tau_h_plus":    r.odw_result.tau_h_plus,
                "tau_f":         r.odw_result.tau_f,
                "n_obs":         r.n_obs,
            })
        except Exception as exc:
            rows.append({
                "window_start": w_start, "window_end": w_end,
                "fawp_found": None, "peak_gap_bits": float("nan"),
                "odw_start": None, "odw_end": None,
                "tau_h_plus": None, "tau_f": None, "n_obs": 0,
                "error": str(exc),
            })
        cursor += step

    return pd.DataFrame(rows)


def plot_weather_map(
    results: "List[WeatherFAWPResult]",
    title:   str = "FAWP Agency Horizon Map",
    zoom:    int = 2,
) -> "plotly.graph_objects.Figure":
    """
    Interactive map of FAWP scan results — coloured by leverage gap.

    Each location is shown as a circle marker:
      - 🔴 Red  = FAWP detected (gap above threshold)
      - 🟢 Green = No FAWP detected
    Marker size scales with peak_gap_bits.
    Hover shows: location, variable, gap, ODW, τ⁺ₕ, τf.

    Parameters
    ----------
    results : list of WeatherFAWPResult
        Output from scan_weather_grid() or a list of fawp_from_open_meteo() calls.
    title : str
        Map title.
    zoom : int
        Initial zoom level (1=world, 4=country, 8=city).

    Returns
    -------
    plotly.graph_objects.Figure — use .show() or st.plotly_chart()

    Requires
    --------
    pip install "fawp-index[plotly]"

    Example
    -------
    ::

        from fawp_index.weather import scan_weather_grid, plot_weather_map

        locs = [
            {"lat": 51.5,  "lon": -0.1,  "name": "London"},
            {"lat": 48.9,  "lon":  2.4,  "name": "Paris"},
            {"lat": 40.7,  "lon": -74.0, "name": "New York"},
            {"lat": 35.7,  "lon": 139.7, "name": "Tokyo"},
        ]
        results = scan_weather_grid(locs, variable="temperature_2m",
                                    start_date="2015-01-01", end_date="2024-12-31")
        fig = plot_weather_map(results)
        fig.show()
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError(
            "plotly is required for map visualisation.\\n"
            "Install with: pip install \\'fawp-index[plotly]\\'"
        )

    lats, lons, names, gaps, colors, sizes, hovers = [], [], [], [], [], [], []

    for r in results:
        meta = r.metadata or {}
        lat  = meta.get("latitude",  meta.get("lat", 0.0))
        lon  = meta.get("longitude", meta.get("lon", 0.0))
        name = meta.get("name", r.location)

        gap   = r.peak_gap_bits
        fawp  = r.fawp_found
        color = "#C0111A" if fawp else "#1DB954"
        size  = max(8, min(40, 8 + gap * 60))

        odw   = f"τ {r.odw_start}–{r.odw_end}" if fawp else "—"
        tauh  = str(r.odw_result.tau_h_plus) if r.odw_result.tau_h_plus else "—"
        tauf  = str(r.odw_result.tau_f)      if r.odw_result.tau_f      else "—"
        hover = (
            f"<b>{name}</b><br>"
            f"FAWP: {'🔴 YES' if fawp else '✅ NO'}<br>"
            f"Variable: {r.variable}<br>"
            f"Gap: {gap:.4f} bits<br>"
            f"ODW: {odw}<br>"
            f"τ⁺ₕ: {tauh} &nbsp; τf: {tauf}<br>"
            f"Period: {r.date_range[0]} → {r.date_range[1]}"
        )

        lats.append(lat); lons.append(lon); names.append(name)
        gaps.append(gap); colors.append(color)
        sizes.append(size); hovers.append(hover)

    # Centre map on mean of locations
    centre_lat = float(np.mean(lats)) if lats else 30.0
    centre_lon = float(np.mean(lons)) if lons else 0.0

    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lat          = lats,
        lon          = lons,
        text         = hovers,
        hoverinfo    = "text",
        mode         = "markers+text",
        textposition = "top center",
        textfont     = dict(size=10, color="#EDF0F8"),
        marker       = dict(
            size        = sizes,
            color       = colors,
            opacity     = 0.85,
            line        = dict(width=1, color="#07101E"),
            symbol      = "circle",
        ),
        name         = "FAWP scan results",
    ))

    fig.update_layout(
        title       = dict(text=title, font=dict(size=15, color="#D4AF37"),
                           x=0.5, xanchor="center"),
        paper_bgcolor = "#07101E",
        plot_bgcolor  = "#07101E",
        geo = dict(
            showland      = True, landcolor      = "#0D1729",
            showocean     = True, oceancolor     = "#070E1A",
            showcoastlines= True, coastlinecolor = "#182540",
            showframe     = False,
            showcountries = True, countrycolor   = "#1E2E4A",
            projection_type = "natural earth",
            center      = dict(lat=centre_lat, lon=centre_lon),
        ),
        margin = dict(l=0, r=0, t=50, b=0),
        legend = dict(bgcolor="#0D1729", font=dict(color="#7A90B8")),
        height = 520,
    )
    return fig
