"""
fawp_index.watchlist — Multi-asset, multi-timeframe FAWP watchlist scanner
===========================================================================

Scan a whole watchlist of tickers (or pre-loaded DataFrames) across multiple
timeframes and rank results by any FAWP signal metric.

Quick start
-----------
With your own DataFrames::

    from fawp_index.watchlist import WatchlistScanner

    dfs = {
        "SPY": spy_df,      # any DataFrame with Close + optional Volume
        "QQQ": qqq_df,
        "BTC": btc_df,
    }
    scanner = WatchlistScanner()
    result  = scanner.scan(dfs)

    print(result.summary())
    result.rank_by("score")         # strongest current FAWP signal
    result.rank_by("gap")           # widest leverage gap
    result.rank_by("persistence")   # longest regime duration
    result.rank_by("freshness")     # most recent new signal
    result.top_n(5, "score")
    result.active_regimes()

    result.to_html("watchlist.html")
    result.to_csv("watchlist.csv")

With yfinance (requires ``pip install yfinance``)::

    from fawp_index.watchlist import scan_watchlist

    result = scan_watchlist(
        tickers    = ["SPY", "QQQ", "GLD", "BTC-USD", "ETH-USD"],
        period     = "2y",
        timeframes = ["1d", "1wk"],
    )
    print(result.summary())

Multi-timeframe::

    result = scanner.scan(dfs, timeframes=["1d", "1wk"])
    df = result.to_dataframe()        # (ticker, timeframe) rows

Ranking metrics
---------------
``"score"``       — latest window regime score (0–1)
``"gap"``         — peak leverage gap in bits
``"entry"``       — earliest regime start (calendar days ago; lower = entered later)
``"persistence"`` — days in current regime (higher = longer-running)
``"freshness"``   — days since last FAWP signal (lower = more recent)

Ralph Clayton (2026) · https://doi.org/10.5281/zenodo.18673949
"""

from __future__ import annotations

import json
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date as _date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .market import (
    FAWPMarketScanner,
    MarketScanConfig,
    MarketScanSeries,
    scan_fawp_market,
)

from fawp_index import __version__ as _VERSION
_DOI     = "https://doi.org/10.5281/zenodo.18673949"
_GITHUB  = "https://github.com/DrRalphClayton/fawp-index"

RANK_METRICS = ("score", "gap", "entry", "persistence", "freshness")

# ─────────────────────────────────────────────────────────────────────────────
# Timeframe resampling
# ─────────────────────────────────────────────────────────────────────────────

_TF_RULES = {
    "1d":  None,        # no resample — use as-is
    "1wk": "W-FRI",
    "1mo": "ME",
    "4h":  "4h",
    "1h":  "1h",
}

def _resample_df(df: pd.DataFrame, timeframe: str, close_col: str, volume_col: Optional[str]) -> pd.DataFrame:
    """Resample a daily (or finer) DataFrame to the requested timeframe."""
    rule = _TF_RULES.get(timeframe)
    if rule is None:
        return df.copy()
    cols = {close_col: "last"}
    if volume_col and volume_col in df.columns:
        cols[volume_col] = "sum"
    resampled = df.resample(rule).agg(cols).dropna(subset=[close_col])
    return resampled


# ─────────────────────────────────────────────────────────────────────────────
# Per-asset result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AssetResult:
    """
    FAWP scan result for one (ticker, timeframe) combination.

    Attributes
    ----------
    ticker : str
    timeframe : str
        e.g. "1d", "1wk"
    scan : MarketScanSeries
        Full rolling scan output.
    latest_score : float
        Regime score of the most recent window (0–1).
    peak_score : float
        Highest regime score across all windows.
    peak_gap_bits : float
        Peak leverage gap in bits (from peak window).
    regime_active : bool
        True if the most recent window is flagged FAWP.
    regime_start : pd.Timestamp or None
        Start date of the current active regime (None if not active).
    days_in_regime : int
        Calendar days in the current active regime (0 if not active).
    signal_age_days : int
        Calendar days since the most recent FAWP window.
        0 = latest window is FAWP. Large = stale signal.
    peak_odw_start : int or None
    peak_odw_end : int or None
    error : str or None
        Non-None if scan failed (used for graceful degradation in batch runs).
    """
    ticker:         str
    timeframe:      str
    scan:           Optional[MarketScanSeries]
    latest_score:   float
    peak_score:     float
    peak_gap_bits:  float
    regime_active:  bool
    regime_start:   Optional[pd.Timestamp]
    days_in_regime: int
    signal_age_days: int
    peak_odw_start: Optional[int]
    peak_odw_end:   Optional[int]
    error:          Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "ticker":          self.ticker,
            "timeframe":       self.timeframe,
            "latest_score":    round(self.latest_score, 6),
            "peak_score":      round(self.peak_score, 6),
            "peak_gap_bits":   round(self.peak_gap_bits, 6),
            "regime_active":   self.regime_active,
            "regime_start":    str(self.regime_start.date()) if self.regime_start else None,
            "days_in_regime":  self.days_in_regime,
            "signal_age_days": self.signal_age_days,
            "peak_odw_start":  self.peak_odw_start,
            "peak_odw_end":    self.peak_odw_end,
            "error":           self.error,
        }


def _asset_result_from_scan(
    ticker: str,
    timeframe: str,
    scan: MarketScanSeries,
) -> AssetResult:
    """Derive AssetResult from a completed MarketScanSeries."""
    latest    = scan.latest
    peak      = scan.peak
    today     = pd.Timestamp.today().normalize()

    # Current regime start and duration
    regime_active = bool(latest.fawp_found)
    regime_start  = None
    days_in_regime = 0

    if regime_active:
        # Walk back to find contiguous run of fawp_found
        for w in reversed(scan.windows):
            if not w.fawp_found:
                break
            regime_start = w.date
        if regime_start is not None:
            days_in_regime = max(0, (today - regime_start).days)

    # Days since last FAWP signal
    signal_age_days = 999
    for w in reversed(scan.windows):
        if w.fawp_found:
            signal_age_days = max(0, (today - w.date).days)
            break

    return AssetResult(
        ticker         = ticker,
        timeframe      = timeframe,
        scan           = scan,
        latest_score   = float(latest.regime_score),
        peak_score     = float(peak.regime_score),
        peak_gap_bits  = float(peak.odw_result.peak_gap_bits),
        regime_active  = regime_active,
        regime_start   = regime_start,
        days_in_regime = days_in_regime,
        signal_age_days= signal_age_days,
        peak_odw_start = peak.odw_result.odw_start,
        peak_odw_end   = peak.odw_result.odw_end,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Watchlist result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WatchlistResult:
    """
    Aggregate result for a full watchlist scan.

    Attributes
    ----------
    assets : list of AssetResult
    scanned_at : datetime
    n_assets : int
    n_flagged : int
        Number of assets currently in a FAWP regime.
    """
    assets:     List[AssetResult]
    scanned_at: datetime

    @property
    def n_assets(self) -> int:
        return len(self.assets)

    @property
    def n_flagged(self) -> int:
        return sum(1 for a in self.assets if a.regime_active and not a.error)

    # ── Ranking ──────────────────────────────────────────────────────────────

    def rank_by(self, metric: str = "score", ascending: bool = False) -> List[AssetResult]:
        """
        Sort all assets by the requested metric.

        Parameters
        ----------
        metric : str
            One of: ``"score"``, ``"gap"``, ``"entry"``, ``"persistence"``,
            ``"freshness"``.
        ascending : bool
            Default False (best first) for score/gap/persistence;
            automatically True for freshness/entry.

        Returns
        -------
        list of AssetResult, sorted best-first.
        """
        _asc_defaults = {"freshness": True, "entry": True}
        asc = _asc_defaults.get(metric, ascending)
        key_fn = {
            "score":       lambda a: a.latest_score,
            "gap":         lambda a: a.peak_gap_bits,
            "entry":       lambda a: a.days_in_regime if a.regime_active else -1,
            "persistence": lambda a: a.days_in_regime,
            "freshness":   lambda a: a.signal_age_days,
        }.get(metric)
        if key_fn is None:
            raise ValueError(f"Unknown metric {metric!r}. Choose from: {RANK_METRICS}")
        valid = [a for a in self.assets if not a.error]
        return sorted(valid, key=key_fn, reverse=not asc)

    def top_n(self, n: int = 10, metric: str = "score") -> List[AssetResult]:
        """Return top-N assets by metric."""
        return self.rank_by(metric)[:n]

    def active_regimes(self) -> List[AssetResult]:
        """Return only assets currently in a FAWP regime, sorted by score."""
        return [a for a in self.rank_by("score") if a.regime_active]

    # ── Summary ──────────────────────────────────────────────────────────────

    def summary(self, n: int = 20) -> str:
        ranked = self.rank_by("score")[:n]
        lines = [
            "=" * 68,
            f"  FAWP Watchlist Scan  —  {self.scanned_at.strftime('%Y-%m-%d %H:%M')}",
            "=" * 68,
            f"  Assets scanned : {self.n_assets}",
            f"  FAWP active    : {self.n_flagged}",
            "",
            f"  {'TICKER':<10} {'TF':<5} {'SCORE':>6} {'GAP(b)':>8} {'ACTIVE':>7} "
            f"{'DAYS':>5} {'FRESH':>6} {'ODW':>8}",
            "  " + "-" * 62,
        ]
        for a in ranked:
            if a.error:
                lines.append(f"  {a.ticker:<10} {a.timeframe:<5} {'ERROR':>6}  {a.error[:30]}")
                continue
            odw = (f"{a.peak_odw_start}–{a.peak_odw_end}"
                   if a.peak_odw_start is not None else "—")
            flag = "🔴" if a.regime_active else "  "
            lines.append(
                f"  {flag}{a.ticker:<8} {a.timeframe:<5} {a.latest_score:>6.4f} "
                f"{a.peak_gap_bits:>8.4f} {'YES' if a.regime_active else 'no':>7} "
                f"{a.days_in_regime:>5} {a.signal_age_days:>6} {odw:>8}"
            )
        lines.append("=" * 68)
        return "\n".join(lines)

    # ── Exports ──────────────────────────────────────────────────────────────

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([a.to_dict() for a in self.assets])

    def to_csv(self, path: Union[str, Path]) -> Path:
        p = Path(path)
        self.to_dataframe().to_csv(p, index=False)
        return p

    def to_json(self, path: Union[str, Path], indent: int = 2) -> Path:
        p = Path(path)
        payload = {
            "meta": {
                "scanned_at":         self.scanned_at.isoformat(),
                "generated_date":     _date.today().isoformat(),
                "fawp_index_version": _VERSION,
                "doi":                _DOI,
            },
            "summary": {
                "n_assets":  self.n_assets,
                "n_flagged": self.n_flagged,
            },
            "assets": [a.to_dict() for a in self.rank_by("score")],
        }
        p.write_text(json.dumps(payload, indent=indent))
        return p

    def to_html(self, path: Union[str, Path]) -> Path:
        p = Path(path)
        p.write_text(_watchlist_html(self))
        return p


# ─────────────────────────────────────────────────────────────────────────────
# WatchlistScanner
# ─────────────────────────────────────────────────────────────────────────────

class WatchlistScanner:
    """
    Scan a dict of DataFrames (or a list of tickers via yfinance) across
    one or more timeframes and rank by FAWP signal.

    Parameters
    ----------
    config : MarketScanConfig, optional
        Shared scan config applied to every asset. Defaults are sensible.
    timeframes : list of str
        Timeframes to scan. Options: ``"1d"``, ``"1wk"``, ``"1mo"``.
        Only meaningful for daily-or-finer input data. Default ``["1d"]``.
    max_workers : int
        Parallel workers for multi-asset scanning. Default 4.
    verbose : bool
        Print progress. Default True.
    **config_kwargs
        Convenience: pass MarketScanConfig fields directly.

    Examples
    --------
    ::

        scanner = WatchlistScanner(window=252, step=5, tau_max=30)

        dfs = {"SPY": spy_df, "QQQ": qqq_df, "GLD": gld_df}
        result = scanner.scan(dfs)

        print(result.summary())
        for a in result.active_regimes():
            print(a.ticker, a.regime_score, a.days_in_regime)

        result.to_html("watchlist.html")
    """

    def __init__(
        self,
        config: Optional[MarketScanConfig] = None,
        timeframes: Optional[List[str]] = None,
        max_workers: int = 4,
        verbose: bool = True,
        **config_kwargs,
    ):
        self.config = config or MarketScanConfig(**{
            k: v for k, v in config_kwargs.items()
            if k in MarketScanConfig.__dataclass_fields__
        })
        self.timeframes  = timeframes or ["1d"]
        self.max_workers = max_workers
        self.verbose     = verbose

    def scan(
        self,
        data: Union[Dict[str, pd.DataFrame], List[str]],
        period: str = "2y",
    ) -> WatchlistResult:
        """
        Run the watchlist scan.

        Parameters
        ----------
        data : dict of str → DataFrame, or list of ticker strings
            If a list of strings, yfinance is used to fetch data.
            Each DataFrame must have a DatetimeIndex and at least
            ``config.close_col``.
        period : str
            yfinance period string (only used when data is a list of tickers).
            Default ``"2y"``.

        Returns
        -------
        WatchlistResult
        """
        if isinstance(data, list):
            data = self._fetch_yfinance(data, period)

        jobs = [
            (ticker, tf, df)
            for ticker, df in data.items()
            for tf in self.timeframes
        ]

        total = len(jobs)
        if self.verbose:
            print(
                f"WatchlistScanner — {len(data)} assets × "
                f"{len(self.timeframes)} timeframe(s) = {total} scans"
            )

        results: List[AssetResult] = []
        done = 0

        def _run(ticker, tf, df):
            resampled = _resample_df(
                df, tf,
                self.config.close_col,
                self.config.volume_col,
            )
            scanner = FAWPMarketScanner(config=self.config, ticker=ticker)
            scan    = scanner.scan(resampled, verbose=False)
            return _asset_result_from_scan(ticker, tf, scan)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(_run, t, tf, df): (t, tf) for t, tf, df in jobs}
            for fut in as_completed(futures):
                ticker, tf = futures[fut]
                done += 1
                try:
                    ar = fut.result()
                    results.append(ar)
                    if self.verbose:
                        flag = "🔴" if ar.regime_active else "🟢"
                        print(
                            f"  [{done:>3}/{total}] {flag} {ticker:<10} {tf:<5} "
                            f"score={ar.latest_score:.4f}  gap={ar.peak_gap_bits:.4f}"
                        )
                except Exception as exc:
                    results.append(AssetResult(
                        ticker=ticker, timeframe=tf, scan=None,
                        latest_score=0, peak_score=0, peak_gap_bits=0,
                        regime_active=False, regime_start=None,
                        days_in_regime=0, signal_age_days=999,
                        peak_odw_start=None, peak_odw_end=None,
                        error=str(exc),
                    ))
                    if self.verbose:
                        print(f"  [{done:>3}/{total}] ❌ {ticker:<10} {tf:<5} ERROR: {exc}")

        n_flagged = sum(1 for a in results if a.regime_active and not a.error)
        if self.verbose:
            print(f"\n  Done. {n_flagged}/{total} scans flagged FAWP.")

        return WatchlistResult(assets=results, scanned_at=datetime.now())

    def _fetch_yfinance(self, tickers: List[str], period: str) -> Dict[str, pd.DataFrame]:
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError(
                "yfinance is required to fetch tickers automatically.\n"
                "Install it with: pip install yfinance\n"
                "Or pass a dict of DataFrames directly."
            )
        data = {}
        for ticker in tickers:
            try:
                df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
                if df.empty:
                    warnings.warn(f"yfinance returned empty data for {ticker!r}")
                    continue
                # yfinance column names vary by version — normalise
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                df.index   = pd.to_datetime(df.index)
                data[ticker] = df
            except Exception as e:
                warnings.warn(f"Failed to fetch {ticker!r}: {e}")
        if not data:
            raise ValueError("No data fetched. Check tickers and network access.")
        return data


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function
# ─────────────────────────────────────────────────────────────────────────────

def scan_watchlist(
    tickers:    Union[List[str], Dict[str, pd.DataFrame]],
    period:     str = "2y",
    timeframes: Optional[List[str]] = None,
    *,
    window:         int   = 252,
    step:           int   = 5,
    tau_max:        int   = 40,
    n_null:         int   = 0,
    max_workers:    int   = 4,
    verbose:        bool  = True,
    **scan_kwargs,
) -> WatchlistResult:
    """
    One-call watchlist scan.

    Parameters
    ----------
    tickers : list of str or dict of str → DataFrame
        Ticker symbols (requires yfinance) or pre-loaded DataFrames.
    period : str
        yfinance period. Default ``"2y"``.
    timeframes : list of str
        ``["1d"]``, ``["1d", "1wk"]``, etc. Default ``["1d"]``.
    window, step, tau_max, n_null : int
        Passed to MarketScanConfig.
    max_workers : int
        Parallel scan threads. Default 4.
    verbose : bool
        Print progress.
    **scan_kwargs
        Any other MarketScanConfig fields.

    Returns
    -------
    WatchlistResult

    Examples
    --------
    ::

        from fawp_index.watchlist import scan_watchlist

        # With yfinance:
        result = scan_watchlist(
            ["SPY", "QQQ", "GLD", "BTC-USD"],
            period="2y",
            timeframes=["1d", "1wk"],
        )

        # With DataFrames:
        result = scan_watchlist({"SPY": spy_df, "QQQ": qqq_df})

        print(result.summary())
        result.to_html("watchlist.html")
    """
    cfg = MarketScanConfig(
        window  = window,
        step    = step,
        tau_max = tau_max,
        n_null  = n_null,
        **{k: v for k, v in scan_kwargs.items()
           if k in MarketScanConfig.__dataclass_fields__},
    )
    scanner = WatchlistScanner(
        config      = cfg,
        timeframes  = timeframes or ["1d"],
        max_workers = max_workers,
        verbose     = verbose,
    )
    return scanner.scan(tickers, period=period)


# ─────────────────────────────────────────────────────────────────────────────
# HTML renderer
# ─────────────────────────────────────────────────────────────────────────────

def _watchlist_html(result: WatchlistResult) -> str:
    ranked = result.rank_by("score")

    rows = ""
    for a in ranked:
        if a.error:
            rows += (
                f'<tr style="background:#fff8f0">'
                f'<td style="padding:6px 10px;font-weight:700">{a.ticker}</td>'
                f'<td>{a.timeframe}</td>'
                f'<td colspan="8" style="color:#c0390b">ERROR: {a.error}</td>'
                f'</tr>\n'
            )
            continue

        bg     = "#fff5f5" if a.regime_active else "#ffffff"
        flag   = '<span style="color:#C0111A;font-weight:700">🔴 YES</span>' if a.regime_active else "—"
        odw    = (f"{a.peak_odw_start}–{a.peak_odw_end}"
                  if a.peak_odw_start is not None else "—")
        bar_w  = int(a.latest_score * 160)
        score_bar = (
            f'<div style="display:flex;align-items:center;gap:6px">'
            f'<div style="width:{bar_w}px;height:10px;background:'
            f'{"#C0111A" if a.regime_active else "#1a7a1a"};'
            f'border-radius:3px;flex-shrink:0"></div>'
            f'<span>{a.latest_score:.4f}</span>'
            f'</div>'
        )
        rows += (
            f'<tr style="background:{bg}">'
            f'<td style="padding:6px 10px;font-weight:700;color:#0E2550">{a.ticker}</td>'
            f'<td style="padding:6px 10px">{a.timeframe}</td>'
            f'<td style="padding:6px 10px">{score_bar}</td>'
            f'<td style="padding:6px 10px">{a.peak_gap_bits:.4f}</td>'
            f'<td style="padding:6px 10px">{flag}</td>'
            f'<td style="padding:6px 10px">'
            f'{str(a.regime_start.date()) if a.regime_start else "—"}</td>'
            f'<td style="padding:6px 10px">{a.days_in_regime}</td>'
            f'<td style="padding:6px 10px">{a.signal_age_days}</td>'
            f'<td style="padding:6px 10px">{odw}</td>'
            f'</tr>\n'
        )

    active_count = result.n_flagged

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>FAWP Watchlist — {result.scanned_at.strftime('%Y-%m-%d')}</title>
<style>
  body {{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
         max-width:1100px;margin:0 auto;padding:2em 1.5em;
         background:#fafafa;color:#222;line-height:1.5}}
  header {{background:#0E2550;color:white;padding:1.6em 2em 1.2em;
           border-radius:8px;margin-bottom:1.5em}}
  header h1 {{margin:0 0 0.2em;font-size:1.5em}}
  header p  {{margin:0.2em 0;font-size:0.88em;color:#aac}}
  .stat {{display:inline-block;background:rgba(255,255,255,0.12);
          border-radius:20px;padding:0.3em 1em;margin:0.3em 0.3em 0 0;
          font-size:0.95em}}
  .stat.red {{background:#C0111A}}
  h2 {{color:#0E2550;border-bottom:2px solid #D4AF37;padding-bottom:4px;margin-top:2em}}
  table {{width:100%;border-collapse:collapse;box-shadow:0 1px 4px rgba(0,0,0,0.07);
          border-radius:6px;overflow:hidden;margin-top:1em}}
  thead th {{background:#0E2550;color:white;padding:9px 10px;text-align:left;font-size:0.86em}}
  tbody tr:hover {{background:#f0f4ff !important}}
  footer {{margin-top:3em;padding-top:1em;border-top:1px solid #ddd;font-size:0.8em;color:#888}}
  a {{color:#0E2550}}
</style>
</head>
<body>
<header>
  <h1>FAWP Watchlist Scan</h1>
  <p>Scanned {result.scanned_at.strftime('%Y-%m-%d %H:%M')} &bull;
     fawp-index v{_VERSION}</p>
  <p><a href="{_DOI}" style="color:#D4AF37">{_DOI}</a></p>
  <div>
    <span class="stat">{result.n_assets} assets</span>
    <span class="stat {'red' if active_count else ''}">{active_count} FAWP active</span>
  </div>
</header>

<h2>Ranked by Current Score</h2>
<table>
  <thead>
    <tr>
      <th>Ticker</th>
      <th>TF</th>
      <th>Score (latest)</th>
      <th>Peak gap (bits)</th>
      <th>FAWP active</th>
      <th>Regime start</th>
      <th>Days in regime</th>
      <th>Signal age (days)</th>
      <th>Peak ODW</th>
    </tr>
  </thead>
  <tbody>
    {rows}
  </tbody>
</table>

<footer>
  <a href="{_GITHUB}">fawp-index</a> &bull;
  Ralph Clayton (2026) &bull;
  <a href="{_DOI}">{_DOI}</a>
</footer>
</body>
</html>
"""
