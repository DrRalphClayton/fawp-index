"""
Tests for fawp_index.weather — weather/climate FAWP detection.

Tests cover the two offline entry points (fawp_from_forecast,
fawp_from_skill_series) and the helper utilities. The ERA5
Open-Meteo function (fawp_from_open_meteo) and scan_weather_grid
require network access and are skipped in CI unless FAWP_WEATHER_NET=1.
"""

import os
import numpy as np
import pandas as pd
import pytest

from fawp_index.weather import (
    WeatherFAWPResult,
    _fmt_loc,
    fawp_from_forecast,
    fawp_from_skill_series,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_nwp():
    """Synthetic NWP forecast with degrading intervention effectiveness."""
    rng = np.random.default_rng(42)
    n   = 800
    true_temp    = 15 + 5 * np.sin(np.linspace(0, 12 * np.pi, n)) + rng.normal(0, 1, n)
    forecast     = true_temp + rng.normal(0, 0.5, n)
    observed     = true_temp
    # Intervention loses effectiveness — decaying amplitude
    intervention = np.exp(-0.005 * np.arange(n)) * rng.normal(0, 1, n)
    return forecast, observed, intervention


@pytest.fixture
def synthetic_skill():
    """Synthetic ACC-like skill series with decaying spread."""
    rng = np.random.default_rng(7)
    n   = 500
    skill  = 0.9 - 0.001 * np.arange(n) + rng.normal(0, 0.04, n)
    spread = 0.5 * np.exp(-0.008 * np.arange(n)) + rng.normal(0, 0.02, n)
    return skill, spread


# ── WeatherFAWPResult structure ───────────────────────────────────────────────

class TestWeatherFAWPResult:
    def test_fawp_from_forecast_returns_result(self, synthetic_nwp):
        fc, obs, intv = synthetic_nwp
        r = fawp_from_forecast(fc, obs, intv, variable="temperature_2m",
                               location="test_nwp", n_null=0)
        assert isinstance(r, WeatherFAWPResult)

    def test_result_has_required_fields(self, synthetic_nwp):
        fc, obs, intv = synthetic_nwp
        r = fawp_from_forecast(fc, obs, intv, n_null=0)
        assert isinstance(r.fawp_found, bool)
        assert isinstance(r.peak_gap_bits, float)
        assert r.peak_gap_bits >= 0.0
        assert len(r.tau) > 0
        assert len(r.pred_mi) == len(r.tau)
        assert len(r.steer_mi) == len(r.tau)
        assert r.n_obs > 0

    def test_summary_string(self, synthetic_nwp):
        fc, obs, intv = synthetic_nwp
        r = fawp_from_forecast(fc, obs, intv, n_null=0)
        s = r.summary()
        assert "FAWP" in s
        assert r.variable in s

    def test_to_dict(self, synthetic_nwp):
        fc, obs, intv = synthetic_nwp
        r = fawp_from_forecast(fc, obs, intv, n_null=0)
        d = r.to_dict()
        assert "fawp_found" in d
        assert "peak_gap_bits" in d
        assert "variable" in d
        assert "location" in d
        assert "n_obs" in d
        assert d["peak_gap_bits"] >= 0.0

    def test_variable_and_location_preserved(self, synthetic_nwp):
        fc, obs, intv = synthetic_nwp
        r = fawp_from_forecast(fc, obs, intv,
                               variable="wind_speed_10m",
                               location="50.0N_0.0E",
                               n_null=0)
        assert r.variable == "wind_speed_10m"
        assert r.location == "50.0N_0.0E"

    def test_horizon_days_preserved(self, synthetic_nwp):
        fc, obs, intv = synthetic_nwp
        r = fawp_from_forecast(fc, obs, intv, horizon_days=10, n_null=0)
        assert r.horizon_days == 10

    def test_odw_fields_when_fawp_found(self, synthetic_nwp):
        fc, obs, intv = synthetic_nwp
        r = fawp_from_forecast(fc, obs, intv, n_null=0)
        if r.fawp_found:
            assert r.odw_start is not None
            assert r.odw_end   is not None
            assert r.odw_start <= r.odw_end

    def test_pred_steer_arrays_nonneg(self, synthetic_nwp):
        fc, obs, intv = synthetic_nwp
        r = fawp_from_forecast(fc, obs, intv, n_null=0)
        assert np.all(r.pred_mi  >= -1e-9)
        assert np.all(r.steer_mi >= -1e-9)


# ── fawp_from_skill_series ────────────────────────────────────────────────────

class TestFAWPFromSkillSeries:
    def test_returns_result(self, synthetic_skill):
        skill, spread = synthetic_skill
        r = fawp_from_skill_series(skill, spread, n_null=0)
        assert isinstance(r, WeatherFAWPResult)

    def test_skill_name_preserved(self, synthetic_skill):
        skill, spread = synthetic_skill
        r = fawp_from_skill_series(skill, spread,
                                   skill_name="ACC_500hPa",
                                   n_null=0)
        assert r.skill_metric == "ACC_500hPa"

    def test_unequal_length_arrays(self):
        rng  = np.random.default_rng(0)
        skill   = rng.normal(0.7, 0.05, 300)
        spread  = rng.normal(0.5, 0.03, 250)   # shorter
        r = fawp_from_skill_series(skill, spread, n_null=0)
        assert r.n_obs == 250

    def test_to_dict_has_version(self, synthetic_skill):
        skill, spread = synthetic_skill
        r = fawp_from_skill_series(skill, spread, n_null=0)
        assert "fawp_index_version" in r.to_dict()


# ── _fmt_loc helper ───────────────────────────────────────────────────────────

class TestFmtLoc:
    def test_northern_eastern(self):
        s = _fmt_loc(51.5, 0.1)
        assert "N" in s and "E" in s
        assert "S" not in s and "W" not in s

    def test_southern_western(self):
        s = _fmt_loc(-33.9, -70.7)   # Santiago
        assert "S" in s and "W" in s
        assert "N" not in s and "E" not in s

    def test_northern_western(self):
        s = _fmt_loc(40.7, -74.0)    # New York
        assert "N" in s and "W" in s

    def test_southern_eastern(self):
        s = _fmt_loc(-33.9, 151.2)   # Sydney
        assert "S" in s and "E" in s

    def test_zero_lat_lon(self):
        s = _fmt_loc(0.0, 0.0)
        assert "N" in s and "E" in s   # zero treated as positive

    def test_abs_values_shown(self):
        s = _fmt_loc(-33.9, -70.7)
        assert "-" not in s   # negatives stripped, replaced with S/W


# ── fawp_from_forecast edge cases ─────────────────────────────────────────────

class TestForecastEdgeCases:
    def test_short_series_handled(self):
        rng = np.random.default_rng(1)
        n   = 60
        fc  = rng.normal(0, 1, n)
        obs = rng.normal(0, 1, n)
        intv = rng.normal(0, 1, n)
        # Should not raise
        r = fawp_from_forecast(fc, obs, intv, tau_max=5, n_null=0)
        assert isinstance(r, WeatherFAWPResult)

    def test_metadata_attached(self):
        rng  = np.random.default_rng(2)
        n    = 200
        r = fawp_from_forecast(rng.normal(size=n), rng.normal(size=n),
                               rng.normal(size=n), n_null=0,
                               metadata={"source": "test", "station": "XYZ"})
        assert r.metadata["source"] == "test"
        assert r.metadata["station"] == "XYZ"

    def test_mismatched_lengths_truncated(self):
        rng  = np.random.default_rng(3)
        fc   = rng.normal(size=300)
        obs  = rng.normal(size=250)
        intv = rng.normal(size=200)
        r = fawp_from_forecast(fc, obs, intv, n_null=0)
        assert r.n_obs == 200   # truncated to shortest


# ── Network tests (skipped in CI by default) ──────────────────────────────────

_NET = os.environ.get("FAWP_WEATHER_NET", "0") == "1"

@pytest.mark.skipif(not _NET, reason="requires network (set FAWP_WEATHER_NET=1)")
def test_open_meteo_london():
    from fawp_index.weather import fawp_from_open_meteo
    r = fawp_from_open_meteo(
        latitude=51.5, longitude=-0.1,
        variable="temperature_2m",
        start_date="2020-01-01",
        end_date="2023-12-31",
        horizon_days=5,
        tau_max=20,
        n_null=20,
    )
    assert isinstance(r, WeatherFAWPResult)
    assert r.n_obs > 100
    assert "N" in r.location   # London is northern hemisphere


@pytest.mark.skipif(not _NET, reason="requires network (set FAWP_WEATHER_NET=1)")
def test_scan_weather_grid_returns_sorted():
    from fawp_index.weather import scan_weather_grid
    locs = [
        {"lat": 51.5,  "lon": -0.1,  "name": "London"},
        {"lat": 48.9,  "lon":  2.4,  "name": "Paris"},
    ]
    results = scan_weather_grid(locs, variable="temperature_2m",
                                start_date="2020-01-01", end_date="2022-12-31",
                                n_null=10)
    assert len(results) == 2
    # Sorted by peak_gap_bits descending
    assert results[0].peak_gap_bits >= results[1].peak_gap_bits
