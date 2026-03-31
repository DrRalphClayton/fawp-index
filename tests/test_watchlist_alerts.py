"""
Tests for fawp_index.watchlist and fawp_index.alerts
"""
import json
import pytest
import fawp_index
import numpy as np
import pandas as pd
from datetime import datetime

from fawp_index.watchlist import (
    WatchlistScanner, scan_watchlist,
    WatchlistResult, AssetResult,
    _resample_df,
)
from fawp_index.alerts import (
    AlertEngine, FAWPAlert, AlertType,
    _TerminalBackend,
)
from fawp_index import WatchlistScanner as _WS, scan_watchlist as _sw


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_df(n=400, seed=42, vol=True):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    df = pd.DataFrame({"Close": prices}, index=dates)
    if vol:
        df["Volume"] = rng.integers(1_000_000, 5_000_000, size=n).astype(float)
    return df


@pytest.fixture(scope="module")
def small_dfs():
    return {
        "AAA": _make_df(300, seed=0),
        "BBB": _make_df(300, seed=1),
        "CCC": _make_df(300, seed=2),
    }


@pytest.fixture(scope="module")
def fast_wl(small_dfs):
    return scan_watchlist(
        small_dfs, window=100, step=50, tau_max=10, verbose=False
    )


# ── Resample helper ───────────────────────────────────────────────────────────

class TestResample:
    def test_daily_passthrough(self):
        df = _make_df(200)
        out = _resample_df(df, "1d", "Close", "Volume")
        assert len(out) == len(df)

    def test_weekly_shorter(self):
        df = _make_df(200)
        out = _resample_df(df, "1wk", "Close", "Volume")
        assert len(out) < len(df)
        assert len(out) > 0

    def test_monthly_shorter(self):
        df = _make_df(300)
        out = _resample_df(df, "1mo", "Close", "Volume")
        assert len(out) < len(df)
        assert len(out) > 0


# ── AssetResult ───────────────────────────────────────────────────────────────

class TestAssetResult:
    def test_to_dict_keys(self, fast_wl):
        a = fast_wl.assets[0]
        d = a.to_dict()
        for key in ("ticker", "timeframe", "latest_score", "peak_gap_bits",
                    "regime_active", "days_in_regime", "signal_age_days"):
            assert key in d

    def test_scores_nonneg(self, fast_wl):
        for a in fast_wl.assets:
            if not a.error:
                assert a.latest_score >= 0
                assert a.peak_score   >= 0
                assert a.peak_gap_bits >= 0

    def test_signal_age_nonneg(self, fast_wl):
        for a in fast_wl.assets:
            if not a.error:
                assert a.signal_age_days >= 0

    def test_days_in_regime_nonneg(self, fast_wl):
        for a in fast_wl.assets:
            if not a.error:
                assert a.days_in_regime >= 0


# ── WatchlistResult ───────────────────────────────────────────────────────────

class TestWatchlistResult:
    def test_is_watchlist_result(self, fast_wl):
        assert isinstance(fast_wl, WatchlistResult)

    def test_n_assets(self, fast_wl, small_dfs):
        assert fast_wl.n_assets == len(small_dfs)

    def test_n_flagged_in_range(self, fast_wl):
        assert 0 <= fast_wl.n_flagged <= fast_wl.n_assets

    def test_rank_by_score(self, fast_wl):
        ranked = fast_wl.rank_by("score")
        scores = [a.latest_score for a in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_rank_by_gap(self, fast_wl):
        ranked = fast_wl.rank_by("gap")
        gaps = [a.peak_gap_bits for a in ranked]
        assert gaps == sorted(gaps, reverse=True)

    def test_rank_by_freshness(self, fast_wl):
        ranked = fast_wl.rank_by("freshness")
        ages = [a.signal_age_days for a in ranked]
        assert ages == sorted(ages)  # ascending (lower = fresher)

    def test_rank_by_persistence(self, fast_wl):
        ranked = fast_wl.rank_by("persistence")
        days = [a.days_in_regime for a in ranked]
        assert days == sorted(days, reverse=True)

    def test_rank_unknown_metric_raises(self, fast_wl):
        with pytest.raises(ValueError, match="Unknown metric"):
            fast_wl.rank_by("banana")

    def test_top_n(self, fast_wl):
        top2 = fast_wl.top_n(2, "score")
        assert len(top2) <= 2
        all_ranked = fast_wl.rank_by("score")
        for a in top2:
            assert a in all_ranked[:2]

    def test_active_regimes_all_flagged(self, fast_wl):
        assert all(a.regime_active for a in fast_wl.active_regimes())

    def test_summary_str(self, fast_wl):
        s = fast_wl.summary()
        assert "AAA" in s or "BBB" in s
        assert "FAWP" in s

    def test_to_dataframe(self, fast_wl):
        df = fast_wl.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == fast_wl.n_assets
        assert "ticker" in df.columns

    def test_to_csv(self, fast_wl, tmp_path):
        p = fast_wl.to_csv(tmp_path / "wl.csv")
        df = pd.read_csv(p)
        assert len(df) == fast_wl.n_assets

    def test_to_json(self, fast_wl, tmp_path):
        p = fast_wl.to_json(tmp_path / "wl.json")
        d = json.loads(p.read_text())
        assert d["meta"]["fawp_index_version"] == fawp_index.__version__
        assert d["summary"]["n_assets"] == fast_wl.n_assets
        assert len(d["assets"]) == fast_wl.n_assets

    def test_to_html(self, fast_wl, tmp_path):
        p = fast_wl.to_html(tmp_path / "wl.html")
        text = p.read_text()
        assert "<!DOCTYPE html>" in text
        assert text.strip().endswith("</html>")


# ── WatchlistScanner ──────────────────────────────────────────────────────────

class TestWatchlistScanner:
    def test_scan_returns_result(self, small_dfs):
        scanner = WatchlistScanner(window=100, step=50, tau_max=10, verbose=False)
        result  = scanner.scan(small_dfs)
        assert isinstance(result, WatchlistResult)

    def test_multi_timeframe(self, small_dfs):
        scanner = WatchlistScanner(
            window=100, step=50, tau_max=10,
            timeframes=["1d", "1wk"], verbose=False
        )
        result = scanner.scan(small_dfs)
        # Should have 2× as many entries
        assert result.n_assets == len(small_dfs) * 2

    def test_scanned_at_is_datetime(self, fast_wl):
        assert isinstance(fast_wl.scanned_at, datetime)

    def test_yfinance_import_error_when_missing(self, tmp_path):
        """If yfinance not installed, passing a list should raise ImportError."""
        scanner = WatchlistScanner(window=100, step=50, verbose=False)
        try:
            import yfinance  # noqa
            pytest.skip("yfinance installed, can't test missing-import path")
        except ImportError:
            with pytest.raises(ImportError, match="yfinance"):
                scanner.scan(["SPY", "QQQ"])

    def test_error_asset_graceful(self):
        """Bad DataFrame (no Close col) produces error entry rather than crash."""
        bad_df = pd.DataFrame({"NotClose": [1, 2, 3]},
                              index=pd.date_range("2020-01-01", periods=3, freq="B"))
        scanner = WatchlistScanner(window=2, step=1, tau_max=2, verbose=False)
        result  = scanner.scan({"BAD": bad_df})
        assert result.n_assets == 1
        assert result.assets[0].error is not None


# ── scan_watchlist convenience ────────────────────────────────────────────────

class TestConvenience:
    def test_basic(self, small_dfs):
        result = scan_watchlist(small_dfs, window=100, step=50, tau_max=8, verbose=False)
        assert isinstance(result, WatchlistResult)

    def test_top_level_import(self):
        assert callable(_WS) and callable(_sw)


# ── AlertEngine ───────────────────────────────────────────────────────────────

class TestAlertEngine:
    def _make_wl_result(self, active_tickers=("AAA",)):
        """Build a minimal WatchlistResult for alert testing."""
        assets = []
        for ticker in ("AAA", "BBB", "CCC"):
            active = ticker in active_tickers
            assets.append(AssetResult(
                ticker=ticker, timeframe="1d", scan=None,
                latest_score=0.15 if active else 0.001,
                peak_score=0.2, peak_gap_bits=0.08 if active else 0.001,
                regime_active=active,
                regime_start=pd.Timestamp("2024-01-01") if active else None,
                days_in_regime=10 if active else 0,
                signal_age_days=0 if active else 99,
                peak_odw_start=31, peak_odw_end=33,
            ))
        return WatchlistResult(assets=assets, scanned_at=datetime.now())

    def test_new_fawp_alert(self):
        collected = []
        engine = AlertEngine()
        engine.add_callback(collected.append)
        # No previous state → AAA entering is NEW_FAWP
        engine.check(self._make_wl_result(active_tickers=("AAA",)))
        new_fawp = [a for a in collected if a.alert_type == AlertType.NEW_FAWP]
        assert any(a.ticker == "AAA" for a in new_fawp)

    def test_regime_end_alert(self):
        collected = []
        engine = AlertEngine()
        engine.add_callback(collected.append)
        # First check: AAA active
        engine.check(self._make_wl_result(active_tickers=("AAA",)))
        collected.clear()
        # Second check: AAA no longer active
        engine.check(self._make_wl_result(active_tickers=()))
        end_alerts = [a for a in collected if a.alert_type == AlertType.REGIME_END]
        assert any(a.ticker == "AAA" for a in end_alerts)

    def test_no_duplicate_active_alert(self):
        """Second check with same state should not fire NEW_FAWP again."""
        collected = []
        engine = AlertEngine()
        engine.add_callback(collected.append)
        wl = self._make_wl_result(active_tickers=("AAA",))
        engine.check(wl)
        collected.clear()
        engine.check(wl)  # same state
        new_fawp = [a for a in collected if a.alert_type == AlertType.NEW_FAWP]
        assert len(new_fawp) == 0

    def test_gap_threshold_alert(self):
        collected = []
        engine = AlertEngine(gap_threshold=0.05)  # our asset has 0.08
        engine.add_callback(collected.append)
        engine.check(self._make_wl_result(active_tickers=("AAA",)))
        gap_alerts = [a for a in collected if a.alert_type == AlertType.GAP_THRESHOLD]
        assert any(a.ticker == "AAA" for a in gap_alerts)

    def test_gap_threshold_not_triggered_below(self):
        collected = []
        engine = AlertEngine(gap_threshold=0.99)  # threshold too high
        engine.add_callback(collected.append)
        engine.check(self._make_wl_result(active_tickers=("AAA",)))
        gap_alerts = [a for a in collected if a.alert_type == AlertType.GAP_THRESHOLD]
        assert len(gap_alerts) == 0

    def test_daily_summary(self):
        collected = []
        engine = AlertEngine()
        engine.add_callback(collected.append)
        wl = self._make_wl_result(active_tickers=("AAA", "BBB"))
        alert = engine.daily_summary(wl)
        assert alert.alert_type == AlertType.DAILY_SUMMARY
        assert "SUMMARY" in alert.message

    def test_backends_list(self):
        engine = AlertEngine()
        engine.add_terminal()
        assert "terminal" in engine.backends

    def test_state_persistence(self, tmp_path):
        state_file = tmp_path / "fawp_state.json"
        engine = AlertEngine(state_path=state_file)
        engine.add_callback(lambda a: None)
        engine.check(self._make_wl_result(active_tickers=("AAA",)))
        assert state_file.exists()
        state = json.loads(state_file.read_text())
        assert state.get("AAA|1d")

    def test_alert_to_dict(self):
        alert = FAWPAlert(
            ticker="SPY", timeframe="1d",
            alert_type=AlertType.NEW_FAWP,
            score=0.12, gap_bits=0.08,
            odw_start=31, odw_end=33,
            timestamp=datetime(2024, 3, 1, 9, 30),
            message="test message",
        )
        d = alert.to_dict()
        assert d["ticker"]     == "SPY"
        assert d["alert_type"] == "NEW_FAWP"
        assert d["score"]      == pytest.approx(0.12)

    def test_suppress_errors(self):
        """Backend that raises should not crash when suppress_errors=True."""
        def bad_fn(alert):
            raise RuntimeError("intentional test error")

        engine = AlertEngine(suppress_errors=True)
        engine.add_callback(bad_fn)
        # Should not raise
        engine.check(self._make_wl_result(active_tickers=("AAA",)))

    def test_suppress_errors_false_raises(self):
        """With suppress_errors=False, backend exceptions propagate."""
        def bad_fn(alert):
            raise RuntimeError("intentional")

        engine = AlertEngine(suppress_errors=False)
        engine.add_callback(bad_fn)
        with pytest.raises(RuntimeError, match="intentional"):
            engine.check(self._make_wl_result(active_tickers=("AAA",)))
