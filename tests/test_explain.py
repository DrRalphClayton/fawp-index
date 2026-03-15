"""Tests for fawp_index.explain — explain_asset, confidence_badge, attribution."""

import numpy as np
import pytest
from unittest.mock import MagicMock


def _make_window(score=0.05, fawp=True, n_tau=40, odw_start=5, odw_end=15):
    w = MagicMock()
    w.regime_score = score
    w.fawp_found   = fawp
    tau = np.arange(0, n_tau)
    w.tau      = tau
    w.pred_mi  = np.exp(-0.04 * tau) * 1.5
    w.steer_mi = np.exp(-0.20 * tau) * 1.5
    odw = MagicMock()
    odw.odw_start    = odw_start
    odw.odw_end      = odw_end
    odw.peak_gap_bits= 0.08
    odw.tau_h_plus   = 4
    odw.tau_f        = 35
    w.odw_result = odw
    return w


def _make_asset(ticker="SPY", score=0.05, active=True, n_windows=15):
    asset = MagicMock()
    asset.ticker          = ticker
    asset.timeframe       = "1d"
    asset.error           = None
    asset.latest_score    = score
    asset.peak_gap_bits   = 0.08
    asset.regime_active   = active
    asset.days_in_regime  = 10 if active else 0
    asset.signal_age_days = 2
    asset.peak_odw_start  = 5
    asset.peak_odw_end    = 15

    win = _make_window(score=score, fawp=active)
    scan = MagicMock()
    scan.windows = [win] * n_windows
    scan.latest  = win
    asset.scan   = scan
    return asset


class TestExplainAsset:
    def test_returns_string(self):
        from fawp_index.explain import explain_asset
        a = _make_asset()
        result = explain_asset(a)
        assert isinstance(result, str)

    def test_contains_ticker(self):
        from fawp_index.explain import explain_asset
        a = _make_asset("NVDA")
        result = explain_asset(a)
        assert "NVDA" in result

    def test_contains_score(self):
        from fawp_index.explain import explain_asset
        a = _make_asset(score=0.081)
        result = explain_asset(a)
        # Score 0–100
        assert "/100" in result

    def test_fawp_active_shown(self):
        from fawp_index.explain import explain_asset
        a = _make_asset(active=True)
        result = explain_asset(a)
        assert "FAWP" in result or "active" in result.lower()

    def test_clear_state(self):
        from fawp_index.explain import explain_asset
        a = _make_asset(score=0.001, active=False)
        result = explain_asset(a)
        assert isinstance(result, str)

    def test_verbose_false(self):
        from fawp_index.explain import explain_asset
        a = _make_asset()
        r1 = explain_asset(a, verbose=True)
        r2 = explain_asset(a, verbose=False)
        assert len(r1) >= len(r2)

    def test_no_scan(self):
        from fawp_index.explain import explain_asset
        a = _make_asset()
        a.scan = None
        result = explain_asset(a)
        assert isinstance(result, str)


class TestConfidenceBadge:
    def test_returns_dict(self):
        from fawp_index.explain import confidence_badge
        a = _make_asset(score=0.05, active=True)
        badge = confidence_badge(a)
        assert isinstance(badge, dict)

    def test_required_keys(self):
        from fawp_index.explain import confidence_badge
        a = _make_asset()
        badge = confidence_badge(a)
        for k in ("tier", "score", "persistence", "concentration", "stability", "n_windows"):
            assert k in badge

    def test_tier_values(self):
        from fawp_index.explain import confidence_badge
        a = _make_asset(score=0.08, active=True)
        badge = confidence_badge(a)
        assert badge["tier"] in ("HIGH", "MEDIUM", "LOW", "INSUFFICIENT")

    def test_score_range(self):
        from fawp_index.explain import confidence_badge
        a = _make_asset()
        badge = confidence_badge(a)
        assert 0.0 <= badge["score"] <= 1.0
        assert 0.0 <= badge["persistence"]   <= 1.0
        assert 0.0 <= badge["concentration"] <= 1.0
        assert 0.0 <= badge["stability"]     <= 1.0

    def test_no_scan_returns_insufficient(self):
        from fawp_index.explain import confidence_badge
        a = _make_asset()
        a.scan = None
        badge = confidence_badge(a)
        assert badge["tier"] == "INSUFFICIENT"

    def test_inactive_capped_at_medium(self):
        from fawp_index.explain import confidence_badge
        a = _make_asset(score=0.001, active=False)
        badge = confidence_badge(a)
        assert badge["tier"] != "HIGH"

    def test_n_windows_capped_at_10(self):
        from fawp_index.explain import confidence_badge
        a = _make_asset(n_windows=50)
        badge = confidence_badge(a)
        assert badge["n_windows"] <= 10


class TestAttributionFunctions:
    def test_attribute_gap_returns_dict(self):
        from fawp_index.explain import attribute_gap
        w = _make_window()
        result = attribute_gap(w, top_n=3)
        assert isinstance(result, dict)
        assert "tau" in result
        assert "gap" in result
        assert "top_tau" in result
        assert "odw_share" in result
        assert len(result["top_tau"]) <= 3

    def test_attribute_gap_shares_sum_to_one(self):
        from fawp_index.explain import attribute_gap
        w = _make_window()
        result = attribute_gap(w)
        assert abs(sum(result["share"]) - 1.0) < 0.01

    def test_attribute_windows_returns_dict(self):
        from fawp_index.explain import attribute_windows
        a = _make_asset(n_windows=15)
        result = attribute_windows(a)
        assert isinstance(result, dict)
        assert "onset_date" in result
        assert "peak_score" in result
        assert "score_slope" in result
        assert "n_fawp_windows" in result

    def test_attribute_windows_no_scan(self):
        from fawp_index.explain import attribute_windows
        a = _make_asset()
        a.scan = None
        result = attribute_windows(a)
        assert result == {}

    def test_attribution_report_is_string(self):
        from fawp_index.explain import attribution_report
        a = _make_asset(n_windows=15)
        result = attribution_report(a)
        assert isinstance(result, str)
        assert "Attribution" in result
        assert a.ticker in result

    def test_explain_dispatcher_asset(self):
        from fawp_index.explain import explain
        a = _make_asset()
        result = explain(a)
        assert isinstance(result, str)
