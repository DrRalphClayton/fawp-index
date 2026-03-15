"""Tests for fawp_index.leaderboard."""

import numpy as np
import pytest
from unittest.mock import MagicMock


def _make_asset(ticker="SPY", score=0.05, gap=0.06, active=True,
                odw_start=5, odw_end=15, days=10, age=2, n_windows=20):
    """Build a minimal AssetResult mock."""
    asset = MagicMock()
    asset.ticker        = ticker
    asset.timeframe     = "1d"
    asset.error         = None
    asset.latest_score  = score
    asset.peak_score    = score
    asset.peak_gap_bits = gap
    asset.regime_active = active
    asset.regime_start  = None
    asset.days_in_regime= days
    asset.signal_age_days = age
    asset.peak_odw_start  = odw_start
    asset.peak_odw_end    = odw_end

    # Minimal scan with windows
    window = MagicMock()
    window.regime_score = score
    window.fawp_found   = active
    tau = np.arange(0, 40)
    pred_mi  = np.exp(-0.04 * tau) * 1.5
    steer_mi = np.exp(-0.20 * tau) * 1.5
    window.pred_mi  = pred_mi
    window.steer_mi = steer_mi

    scan = MagicMock()
    scan.windows = [window] * n_windows
    scan.latest  = window
    asset.scan   = scan
    return asset


def _make_watchlist_result(assets):
    wl = MagicMock()
    wl.assets    = assets
    wl.n_assets  = len(assets)
    wl.n_flagged = sum(1 for a in assets if a.regime_active)
    return wl


class TestLeaderboard:
    def test_from_watchlist_returns_leaderboard(self):
        from fawp_index.leaderboard import Leaderboard
        assets = [
            _make_asset("SPY", score=0.08, active=True),
            _make_asset("QQQ", score=0.05, active=False),
            _make_asset("GLD", score=0.02, active=False),
        ]
        wl = _make_watchlist_result(assets)
        lb = Leaderboard.from_watchlist(wl)
        assert isinstance(lb, Leaderboard)

    def test_top_fawp_only_active(self):
        from fawp_index.leaderboard import Leaderboard
        assets = [
            _make_asset("SPY", score=0.08, active=True),
            _make_asset("QQQ", score=0.05, active=False),
        ]
        wl = _make_watchlist_result(assets)
        lb = Leaderboard.from_watchlist(wl)
        tickers = [e.ticker for e in lb.top_fawp]
        assert "SPY" in tickers
        assert "QQQ" not in tickers

    def test_summary_is_string(self):
        from fawp_index.leaderboard import Leaderboard
        assets = [_make_asset("SPY", score=0.08, active=True)]
        wl = _make_watchlist_result(assets)
        lb = Leaderboard.from_watchlist(wl)
        s = lb.summary()
        assert isinstance(s, str)
        assert "SPY" in s

    def test_to_dict_structure(self):
        from fawp_index.leaderboard import Leaderboard
        assets = [_make_asset("SPY", score=0.08, active=True)]
        wl = _make_watchlist_result(assets)
        lb = Leaderboard.from_watchlist(wl)
        d = lb.to_dict()
        assert "meta" in d
        assert "top_fawp" in d
        assert "rising_risk" in d
        assert "collapsing_control" in d
        assert "strongest_odw" in d
        assert d["meta"]["n_flagged"] == 1

    def test_entry_fields(self):
        from fawp_index.leaderboard import Leaderboard
        assets = [_make_asset("SPY", score=0.08, active=True, odw_start=5, odw_end=12)]
        wl = _make_watchlist_result(assets)
        lb = Leaderboard.from_watchlist(wl)
        if lb.top_fawp:
            e = lb.top_fawp[0]
            assert e.ticker == "SPY"
            assert e.rank == 1
            assert e.score >= 0

    def test_empty_watchlist(self):
        from fawp_index.leaderboard import Leaderboard
        wl = _make_watchlist_result([])
        lb = Leaderboard.from_watchlist(wl)
        assert lb.top_fawp == []
        assert lb.n_assets == 0

    def test_top_n_respected(self):
        from fawp_index.leaderboard import Leaderboard
        assets = [_make_asset(f"T{i}", score=0.1 - i*0.01, active=True)
                  for i in range(8)]
        wl = _make_watchlist_result(assets)
        lb = Leaderboard.from_watchlist(wl, top_n=3)
        assert len(lb.top_fawp) <= 3

    def test_to_csv(self, tmp_path):
        from fawp_index.leaderboard import Leaderboard
        import pandas as pd
        assets = [_make_asset("SPY", score=0.08, active=True)]
        wl = _make_watchlist_result(assets)
        lb = Leaderboard.from_watchlist(wl)
        p = tmp_path / "lb.csv"
        lb.to_csv(p)
        df = pd.read_csv(p)
        assert "ticker" in df.columns
        assert "category" in df.columns

    def test_to_json(self, tmp_path):
        from fawp_index.leaderboard import Leaderboard
        import json
        assets = [_make_asset("SPY", score=0.08, active=True)]
        wl = _make_watchlist_result(assets)
        lb = Leaderboard.from_watchlist(wl)
        p = tmp_path / "lb.json"
        lb.to_json(p)
        data = json.loads(p.read_text())
        assert "top_fawp" in data

    def test_to_html(self, tmp_path):
        from fawp_index.leaderboard import Leaderboard
        assets = [_make_asset("SPY", score=0.08, active=True)]
        wl = _make_watchlist_result(assets)
        lb = Leaderboard.from_watchlist(wl)
        p = tmp_path / "lb.html"
        lb.to_html(p)
        html = p.read_text()
        assert "<!DOCTYPE html>" in html
        assert "Leaderboard" in html
