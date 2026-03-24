"""
Tests for fawp_index.market — FAWPMarketScanner
"""
import json
import pytest
import numpy as np
import pandas as pd

from fawp_index.market import (
    FAWPMarketScanner,
    scan_fawp_market,
    MarketScanConfig,
    MarketScanSeries,
    MarketWindowResult,
    _mi, _null_floor, _log_returns, _signed_flow,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_price_df(n=400, seed=42, with_volume=True):
    """Synthetic price DataFrame with DatetimeIndex."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    df = pd.DataFrame({"Close": prices}, index=dates)
    if with_volume:
        df["Volume"] = rng.integers(1_000_000, 5_000_000, size=n).astype(float)
    return df


@pytest.fixture(scope="module")
def df_with_vol():
    return _make_price_df(400, with_volume=True)


@pytest.fixture(scope="module")
def df_no_vol():
    return _make_price_df(400, with_volume=False)


@pytest.fixture(scope="module")
def fast_scan(df_with_vol):
    """Fast scan with small window for tests."""
    return scan_fawp_market(
        df_with_vol,
        ticker="TEST",
        window=100,
        step=20,
        tau_max=15,
        n_null=0,
        verbose=False,
    )


# ── MI helpers ────────────────────────────────────────────────────────────────

class TestMIHelpers:
    def test_mi_correlated(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=200)
        y = x + rng.normal(scale=0.1, size=200)
        assert _mi(x, y) > 0.5

    def test_mi_uncorrelated(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=200)
        y = rng.normal(size=200)
        assert _mi(x, y) < 0.2

    def test_mi_too_few_returns_zero(self):
        assert _mi(np.ones(5), np.ones(5)) == 0.0

    def test_mi_constant_returns_zero(self):
        assert _mi(np.ones(100), np.ones(100)) == 0.0

    def test_log_returns_length(self):
        prices = np.array([100.0, 101.0, 99.0, 102.0])
        r = _log_returns(prices)
        assert len(r) == 4
        assert np.isnan(r[0])
        assert np.isfinite(r[1])

    def test_signed_flow(self):
        prices  = np.array([100.0, 102.0, 101.0, 103.0])
        volumes = np.array([1e6, 2e6, 1.5e6, 2.5e6])
        flow = _signed_flow(prices, volumes)
        assert len(flow) == 4
        assert np.isnan(flow[0])
        assert flow[1] > 0   # price went up, flow positive
        assert flow[2] < 0   # price went down, flow negative

    def test_null_floor_zero_when_n_null_zero(self):
        rng = np.random.default_rng(0)
        x = np.random.randn(100)
        y = np.random.randn(100)
        assert _null_floor(x, y, n_null=0, beta=0.99, rng=rng) == 0.0


# ── MarketScanConfig ──────────────────────────────────────────────────────────

class TestMarketScanConfig:
    def test_defaults(self):
        cfg = MarketScanConfig()
        assert cfg.window == 252
        assert cfg.step == 5
        assert cfg.tau_max == 40
        assert cfg.n_null == 0
        assert cfg.epsilon == 0.01

    def test_custom(self):
        cfg = MarketScanConfig(window=100, step=10, tau_max=20, n_null=50)
        assert cfg.window == 100
        assert cfg.n_null == 50


# ── MarketWindowResult ────────────────────────────────────────────────────────

class TestMarketWindowResult:
    def test_to_dict_keys(self, fast_scan):
        w = fast_scan.windows[0]
        d = w.to_dict()
        for key in ("date", "fawp_found", "regime_score", "tau_h_plus",
                    "odw_start", "odw_end", "peak_gap_bits", "n_obs"):
            assert key in d

    def test_regime_score_in_range(self, fast_scan):
        for w in fast_scan.windows:
            assert 0.0 <= w.regime_score <= 1.0

    def test_arrays_correct_length(self, fast_scan):
        w = fast_scan.windows[0]
        assert len(w.tau) == len(w.pred_mi) == len(w.steer_mi)
        assert len(w.pred_mi_raw) == len(w.steer_mi_raw) == len(w.tau)

    def test_pred_mi_nonneg(self, fast_scan):
        for w in fast_scan.windows:
            assert (w.pred_mi >= 0).all()
            assert (w.steer_mi >= 0).all()

    def test_n_obs_correct(self, fast_scan):
        for w in fast_scan.windows:
            assert w.n_obs == fast_scan.config.window


# ── MarketScanSeries ──────────────────────────────────────────────────────────

class TestMarketScanSeries:
    def test_returns_series(self, fast_scan):
        assert isinstance(fast_scan, MarketScanSeries)

    def test_windows_not_empty(self, fast_scan):
        assert len(fast_scan.windows) > 0

    def test_dates_length(self, fast_scan):
        assert len(fast_scan.dates) == len(fast_scan.windows)

    def test_regime_scores_length(self, fast_scan):
        assert len(fast_scan.regime_scores) == len(fast_scan.windows)

    def test_fawp_flags_bool(self, fast_scan):
        assert fast_scan.fawp_flags.dtype == bool

    def test_fawp_fraction_in_range(self, fast_scan):
        assert 0.0 <= fast_scan.fawp_fraction <= 1.0

    def test_fawp_fraction_consistent(self, fast_scan):
        computed = float(np.mean(fast_scan.fawp_flags))
        assert abs(fast_scan.fawp_fraction - computed) < 1e-9

    def test_latest_property(self, fast_scan):
        assert fast_scan.latest is fast_scan.windows[-1]

    def test_peak_property(self, fast_scan):
        peak = fast_scan.peak
        assert peak.regime_score == max(w.regime_score for w in fast_scan.windows)

    def test_fawp_windows_subset(self, fast_scan):
        fw = fast_scan.fawp_windows
        assert all(w.fawp_found for w in fw)
        assert len(fw) == int(np.sum(fast_scan.fawp_flags))

    def test_summary_str(self, fast_scan):
        s = fast_scan.summary()
        assert "TEST" in s
        assert "windows" in s.lower()

    def test_to_dataframe(self, fast_scan):
        df = fast_scan.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(fast_scan.windows)
        assert "fawp_found" in df.columns
        assert "regime_score" in df.columns

    def test_to_csv(self, fast_scan, tmp_path):
        p = tmp_path / "scan.csv"
        fast_scan.to_csv(p)
        df = pd.read_csv(p)
        assert len(df) == len(fast_scan.windows)
        assert "fawp_found" in df.columns

    def test_to_json(self, fast_scan, tmp_path):
        p = tmp_path / "scan.json"
        fast_scan.to_json(p)
        d = json.loads(p.read_text())
        assert d["meta"]["fawp_index_version"] == "2.5.0"
        assert d["meta"]["ticker"] == "TEST"
        assert len(d["windows"]) == len(fast_scan.windows)
        assert "fawp_fraction" in d["summary"]

    def test_to_html(self, fast_scan, tmp_path):
        p = tmp_path / "scan.html"
        fast_scan.to_html(p)
        text = p.read_text()
        assert "<!DOCTYPE html>" in text
        assert "TEST" in text
        assert text.strip().endswith("</html>")


# ── FAWPMarketScanner ────────────────────────────────────────────────────────

class TestFAWPMarketScanner:
    def test_scan_returns_series(self, df_with_vol):
        scanner = FAWPMarketScanner(ticker="X", window=100, step=20, tau_max=10)
        result = scanner.scan(df_with_vol, verbose=False)
        assert isinstance(result, MarketScanSeries)

    def test_correct_number_of_windows(self, df_with_vol):
        scanner = FAWPMarketScanner(window=100, step=25, tau_max=10)
        result = scanner.scan(df_with_vol, verbose=False)
        n_rows = len(df_with_vol)
        expected = len(list(range(0, n_rows - 100 + 1, 25)))
        assert len(result.windows) == expected

    def test_no_volume_fallback(self, df_no_vol):
        """Scanner works without volume column."""
        scanner = FAWPMarketScanner(
            window=100, step=20, tau_max=10,
            volume_col=None
        )
        result = scanner.scan(df_no_vol, verbose=False)
        assert isinstance(result, MarketScanSeries)
        assert len(result.windows) > 0

    def test_custom_config_object(self, df_with_vol):
        cfg = MarketScanConfig(window=100, step=20, tau_max=10, n_null=0)
        scanner = FAWPMarketScanner(config=cfg, ticker="CFG")
        result = scanner.scan(df_with_vol, verbose=False)
        assert result.config.window == 100
        assert result.ticker == "CFG"

    def test_wrong_close_col_raises(self, df_with_vol):
        scanner = FAWPMarketScanner(window=100, step=20, close_col="NOTHERE")
        with pytest.raises(ValueError, match="close_col"):
            scanner.scan(df_with_vol, verbose=False)

    def test_too_small_df_raises(self):
        df = _make_price_df(50)
        scanner = FAWPMarketScanner(window=100, step=10)
        with pytest.raises(ValueError, match="window"):
            scanner.scan(df, verbose=False)

    def test_custom_pred_col(self):
        """User-supplied pred_col is used instead of returns."""
        df = _make_price_df(300)
        rng = np.random.default_rng(7)
        df["my_signal"] = rng.normal(size=len(df))
        scanner = FAWPMarketScanner(
            window=100, step=20, tau_max=10,
            pred_col="my_signal",
        )
        result = scanner.scan(df, verbose=False)
        assert isinstance(result, MarketScanSeries)

    def test_custom_steer_col(self):
        """User-supplied steer_col is used."""
        df = _make_price_df(300)
        rng = np.random.default_rng(8)
        df["my_flow"] = rng.normal(size=len(df))
        scanner = FAWPMarketScanner(
            window=100, step=20, tau_max=10,
            steer_col="my_flow",
        )
        result = scanner.scan(df, verbose=False)
        assert isinstance(result, MarketScanSeries)

    def test_non_datetime_index_with_date_col(self):
        """Accepts date_col instead of DatetimeIndex."""
        df = _make_price_df(300)
        df = df.reset_index().rename(columns={"index": "Date"})
        scanner = FAWPMarketScanner(
            window=100, step=20, tau_max=10,
            date_col="Date",
        )
        result = scanner.scan(df, verbose=False)
        assert isinstance(result, MarketScanSeries)


# ── scan_fawp_market convenience function ────────────────────────────────────

class TestConvenienceFunction:
    def test_basic_call(self, df_with_vol):
        result = scan_fawp_market(
            df_with_vol, ticker="SPY",
            window=100, step=20, tau_max=10, verbose=False,
        )
        assert isinstance(result, MarketScanSeries)
        assert result.ticker == "SPY"

    def test_no_volume(self, df_no_vol):
        result = scan_fawp_market(
            df_no_vol, volume_col=None,
            window=100, step=20, tau_max=10, verbose=False,
        )
        assert isinstance(result, MarketScanSeries)

    def test_volume_col_not_present_silent(self, df_with_vol):
        """If volume_col is named but missing from df, falls back silently."""
        df = df_with_vol.drop(columns=["Volume"])
        result = scan_fawp_market(
            df, volume_col="Volume",
            window=100, step=20, tau_max=10, verbose=False,
        )
        assert isinstance(result, MarketScanSeries)

    def test_top_level_import(self):
        from fawp_index import FAWPMarketScanner as S, scan_fawp_market as f
        assert callable(S) and callable(f)

    def test_with_null_correction(self, df_with_vol):
        """n_null > 0 runs without error (slow path)."""
        result = scan_fawp_market(
            df_with_vol,
            window=100, step=50, tau_max=5,
            n_null=10,
            verbose=False,
        )
        assert isinstance(result, MarketScanSeries)


# ── Plot smoke test ───────────────────────────────────────────────────────────

class TestPlot:
    def test_plot_no_prices(self, fast_scan, tmp_path):
        try:
            fig = fast_scan.plot(show=False, save_path=str(tmp_path / "scan.png"))
            assert fig is not None
        except ImportError:
            pytest.skip("matplotlib not available")

    def test_plot_with_prices(self, fast_scan, df_with_vol, tmp_path):
        try:
            fig = fast_scan.plot(
                prices=df_with_vol["Close"],
                show=False,
                save_path=str(tmp_path / "scan_price.png"),
            )
            assert fig is not None
        except ImportError:
            pytest.skip("matplotlib not available")

    def test_html_contains_chart(self, fast_scan, tmp_path):
        try:
            p = tmp_path / "s.html"
            fast_scan.to_html(p)
            text = p.read_text()
            assert "data:image/png;base64," in text
        except ImportError:
            pytest.skip("matplotlib not available")
