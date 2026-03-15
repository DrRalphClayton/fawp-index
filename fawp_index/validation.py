"""
fawp_index.validation — Forward-return validation for FAWP signals.

Answers the core quant question: "After FAWP triggers, what happens next?"

For each signal date detected in a MarketScanSeries or AssetResult,
compute forward returns at multiple horizons and aggregate into:

  - Hit rate        (% of signals where forward return > 0 or < 0)
  - Average return  (mean forward return by horizon)
  - Median return
  - Max Adverse Excursion  (worst outcome per signal)
  - Max Favorable Excursion (best outcome per signal)
  - Tier breakdown  (by regime severity: HIGH / MEDIUM / LOW)
  - Regime vs no-regime comparison

Usage::

    import yfinance as yf
    from fawp_index.watchlist import scan_watchlist
    from fawp_index.validation import validate_signals

    prices = yf.download("SPY", period="5y", auto_adjust=True)["Close"]
    result = scan_watchlist({"SPY": prices.to_frame("Close")})
    asset  = result.rank_by("score")[0]

    report = validate_signals(asset, prices)
    print(report.summary())
    report.to_html("spy_validation.html")

Ralph Clayton (2026) · doi:10.5281/zenodo.18673949
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from fawp_index import __version__ as _VERSION


# ── HorizonStats ──────────────────────────────────────────────────────────────

@dataclass
class HorizonStats:
    """Forward-return statistics for one horizon."""
    horizon:     int
    n_signals:   int
    mean_return: float
    median_return: float
    hit_rate:    float          # fraction of signals with return > 0
    mae:         float          # mean max adverse excursion (worst drawdown)
    mfe:         float          # mean max favorable excursion (best run-up)
    std_return:  float
    pct_5:       float          # 5th percentile
    pct_95:      float          # 95th percentile

    def to_dict(self) -> dict:
        return {
            "horizon":       self.horizon,
            "n_signals":     self.n_signals,
            "mean_return":   round(self.mean_return,   4),
            "median_return": round(self.median_return, 4),
            "hit_rate":      round(self.hit_rate,      4),
            "mae":           round(self.mae,           4),
            "mfe":           round(self.mfe,           4),
            "std_return":    round(self.std_return,    4),
            "pct_5":         round(self.pct_5,         4),
            "pct_95":        round(self.pct_95,        4),
        }


# ── ValidationReport ──────────────────────────────────────────────────────────

@dataclass
class ValidationReport:
    """
    Full validation report for one asset's FAWP signals.

    Attributes
    ----------
    ticker : str
    timeframe : str
    horizons : list of HorizonStats
        One entry per forecast horizon tested.
    regime_mean_return : dict
        {horizon: mean_return} for FAWP-active windows only.
    baseline_mean_return : dict
        {horizon: mean_return} for non-FAWP windows (baseline comparison).
    n_signals : int
        Number of FAWP signal dates found.
    n_prices : int
        Number of price bars available.
    signal_dates : list of str
        ISO dates when FAWP was active.
    generated_date : str
    """
    ticker:               str
    timeframe:            str
    horizons:             List[HorizonStats]
    regime_mean_return:   Dict[int, float]
    baseline_mean_return: Dict[int, float]
    n_signals:            int
    n_prices:             int
    signal_dates:         List[str]
    generated_date:       str = field(
        default_factory=lambda: date.today().isoformat()
    )

    # ── summary ──────────────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = [
            "=" * 66,
            f"  Validation Report — {self.ticker} [{self.timeframe}]",
            f"  {self.n_signals} FAWP signals · {self.n_prices} price bars",
            f"  Generated {self.generated_date}",
            "=" * 66,
            "",
            f"  {'Horizon':>8}  {'N':>5}  {'Mean%':>7}  {'Median%':>8}  "
            f"{'HitRate':>8}  {'MAE%':>7}  {'MFE%':>7}",
            "  " + "-" * 60,
        ]
        for h in self.horizons:
            lines.append(
                f"  {h.horizon:>8}  {h.n_signals:>5}  "
                f"{h.mean_return*100:>7.2f}  {h.median_return*100:>8.2f}  "
                f"{h.hit_rate*100:>7.1f}%  "
                f"{h.mae*100:>7.2f}  {h.mfe*100:>7.2f}"
            )

        if self.regime_mean_return and self.baseline_mean_return:
            lines += ["", "  FAWP vs Baseline (mean return %):", "  " + "-" * 40]
            for hz in sorted(self.regime_mean_return):
                reg = self.regime_mean_return.get(hz, 0.0)
                base = self.baseline_mean_return.get(hz, 0.0)
                diff = reg - base
                arrow = "▲" if diff > 0 else "▼"
                lines.append(
                    f"  {hz:>6} bars:  FAWP {reg*100:+.2f}%  "
                    f"Base {base*100:+.2f}%  diff {diff*100:+.2f}% {arrow}"
                )

        lines += ["", "=" * 66]
        return "\n".join(lines)

    # ── export ────────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "meta": {
                "ticker":             self.ticker,
                "timeframe":          self.timeframe,
                "n_signals":          self.n_signals,
                "n_prices":           self.n_prices,
                "generated_date":     self.generated_date,
                "fawp_index_version": _VERSION,
            },
            "horizons":             [h.to_dict() for h in self.horizons],
            "regime_mean_return":   {str(k): round(v, 4)
                                     for k, v in self.regime_mean_return.items()},
            "baseline_mean_return": {str(k): round(v, 4)
                                     for k, v in self.baseline_mean_return.items()},
            "signal_dates":         self.signal_dates,
        }

    def to_json(self, path: Union[str, Path], indent: int = 2) -> Path:
        p = Path(path)
        p.write_text(json.dumps(self.to_dict(), indent=indent))
        return p

    def to_csv(self, path: Union[str, Path]) -> Path:
        p = Path(path)
        pd.DataFrame([h.to_dict() for h in self.horizons]).to_csv(p, index=False)
        return p

    def to_html(self, path: Union[str, Path]) -> Path:
        p = Path(path)
        p.write_text(_validation_html(self))
        return p


# ── Core computation ──────────────────────────────────────────────────────────

def validate_signals(
    asset,
    prices:    pd.Series,
    horizons:  Optional[List[int]] = None,
    min_signals: int = 3,
) -> ValidationReport:
    """
    Compute forward-return statistics for all FAWP signal dates.

    Parameters
    ----------
    asset : AssetResult
        Output of scan_watchlist or WatchlistScanner.
    prices : pd.Series
        Close price series indexed by datetime. Must overlap with scan dates.
        Use ``yf.download(ticker)["Close"]`` or pass the Close column directly.
    horizons : list of int, optional
        Forward horizons in bars to test. Default: [1, 5, 10, 20, 40].
    min_signals : int
        Minimum signals required; returns empty report if fewer found.

    Returns
    -------
    ValidationReport

    Examples
    --------
    ::

        import yfinance as yf
        from fawp_index.watchlist import scan_watchlist
        from fawp_index.validation import validate_signals

        df = yf.download("SPY", period="5y", auto_adjust=True)
        prices = df["Close"].squeeze()
        result = scan_watchlist({"SPY": df[["Close"]]})
        asset  = result.rank_by("score")[0]
        report = validate_signals(asset, prices)
        print(report.summary())
    """
    if horizons is None:
        horizons = [1, 5, 10, 20, 40]

    prices = prices.squeeze()
    if not isinstance(prices.index, pd.DatetimeIndex):
        prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index().dropna()

    # ── Get signal dates from scan windows ────────────────────────────────────
    signal_dates   = []
    nonsignal_dates = []

    if asset.scan is not None:
        for w in asset.scan.windows:
            dt = pd.Timestamp(w.date).normalize()
            if w.fawp_found:
                signal_dates.append(dt)
            else:
                nonsignal_dates.append(dt)

    ticker = asset.ticker
    timeframe = asset.timeframe

    if len(signal_dates) < min_signals:
        return ValidationReport(
            ticker=ticker, timeframe=timeframe,
            horizons=[], regime_mean_return={}, baseline_mean_return={},
            n_signals=len(signal_dates), n_prices=len(prices),
            signal_dates=[str(d.date()) for d in signal_dates],
        )

    # ── Forward return helper ──────────────────────────────────────────────────
    price_arr = prices.values.astype(float)
    price_idx = prices.index

    def _fwd_returns(dates_list: List[pd.Timestamp], h: int) -> np.ndarray:
        rets = []
        for dt in dates_list:
            # Find nearest price date on or after signal date
            pos = price_idx.searchsorted(dt)
            if pos + h >= len(price_arr):
                continue
            entry = price_arr[pos]
            if entry <= 0:
                continue
            # Forward return
            fwd   = price_arr[pos + h] / entry - 1.0
            rets.append(fwd)
        return np.array(rets) if rets else np.array([])

    def _mxe(dates_list: List[pd.Timestamp], h: int):
        """Max adverse and favorable excursion per signal."""
        maes, mfes = [], []
        for dt in dates_list:
            pos = price_idx.searchsorted(dt)
            end = min(pos + h + 1, len(price_arr))
            if end <= pos + 1:
                continue
            entry   = price_arr[pos]
            if entry <= 0:
                continue
            window  = price_arr[pos:end] / entry - 1.0
            maes.append(float(np.min(window)))
            mfes.append(float(np.max(window)))
        return np.array(maes), np.array(mfes)

    # ── Compute per-horizon stats ──────────────────────────────────────────────
    horizon_stats = []
    regime_mean:   Dict[int, float] = {}
    baseline_mean: Dict[int, float] = {}

    for h in horizons:
        sig_rets  = _fwd_returns(signal_dates,    h)
        base_rets = _fwd_returns(nonsignal_dates, h)

        if len(sig_rets) < 1:
            continue

        mae_arr, mfe_arr = _mxe(signal_dates, h)

        hs = HorizonStats(
            horizon       = h,
            n_signals     = len(sig_rets),
            mean_return   = float(np.mean(sig_rets)),
            median_return = float(np.median(sig_rets)),
            hit_rate      = float(np.mean(sig_rets > 0)),
            mae           = float(np.mean(mae_arr)) if len(mae_arr) else 0.0,
            mfe           = float(np.mean(mfe_arr)) if len(mfe_arr) else 0.0,
            std_return    = float(np.std(sig_rets)),
            pct_5         = float(np.percentile(sig_rets, 5)),
            pct_95        = float(np.percentile(sig_rets, 95)),
        )
        horizon_stats.append(hs)
        regime_mean[h]   = hs.mean_return
        baseline_mean[h] = float(np.mean(base_rets)) if len(base_rets) > 0 else 0.0

    return ValidationReport(
        ticker               = ticker,
        timeframe            = timeframe,
        horizons             = horizon_stats,
        regime_mean_return   = regime_mean,
        baseline_mean_return = baseline_mean,
        n_signals            = len(signal_dates),
        n_prices             = len(prices),
        signal_dates         = [str(d.date()) for d in signal_dates],
    )


# ── HTML renderer ─────────────────────────────────────────────────────────────

def _validation_html(r: ValidationReport) -> str:
    hdr = "#0E2550"
    rows = ""
    for h in r.horizons:
        hit_color = "#1DB954" if h.hit_rate >= 0.55 else (
                    "#D4AF37" if h.hit_rate >= 0.45 else "#C0111A")
        ret_color = "#1DB954" if h.mean_return > 0 else "#C0111A"
        rows += (
            f"<tr>"
            f"<td><b>{h.horizon}</b></td>"
            f"<td>{h.n_signals}</td>"
            f"<td style='color:{ret_color}'>{h.mean_return*100:+.2f}%</td>"
            f"<td>{h.median_return*100:+.2f}%</td>"
            f"<td style='color:{hit_color}'>{h.hit_rate*100:.1f}%</td>"
            f"<td style='color:#C0111A'>{h.mae*100:.2f}%</td>"
            f"<td style='color:#1DB954'>{h.mfe*100:.2f}%</td>"
            f"<td>{h.std_return*100:.2f}%</td>"
            f"<td>{h.pct_5*100:.2f}%</td>"
            f"<td>{h.pct_95*100:.2f}%</td>"
            f"</tr>"
        )

    comp_rows = ""
    if r.regime_mean_return and r.baseline_mean_return:
        for hz in sorted(r.regime_mean_return):
            reg  = r.regime_mean_return.get(hz, 0.0)
            base = r.baseline_mean_return.get(hz, 0.0)
            diff = reg - base
            diff_color = "#1DB954" if diff > 0 else "#C0111A"
            comp_rows += (
                f"<tr>"
                f"<td><b>{hz}</b></td>"
                f"<td>{reg*100:+.2f}%</td>"
                f"<td>{base*100:+.2f}%</td>"
                f"<td style='color:{diff_color};font-weight:700'>{diff*100:+.2f}%</td>"
                f"</tr>"
            )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>FAWP Validation — {r.ticker}</title>
<style>
  body{{font-family:-apple-system,sans-serif;background:#07101E;color:#EDF0F8;
       margin:0;padding:0}}
  header{{background:{hdr};padding:1.6em 2em 1.2em;margin-bottom:1.5em}}
  header h1{{margin:0 0 0.3em;font-size:1.35em}}
  header p{{margin:0.2em 0;font-size:0.85em;color:#aac}}
  .container{{padding:0 2em 2em}}
  h2{{font-size:1em;color:#D4AF37;border-bottom:1px solid #6A5518;
      padding-bottom:5px;margin:1.5em 0 0.7em;text-transform:uppercase;
      letter-spacing:.1em}}
  table{{width:100%;border-collapse:collapse;font-size:0.85em;margin-bottom:1.5em}}
  th{{text-align:left;padding:.4em .8em;color:#7A90B8;font-size:.72em;
      text-transform:uppercase;letter-spacing:.07em;
      border-bottom:1px solid #182540}}
  td{{padding:.4em .8em;border-bottom:1px solid #111E35}}
  tr:last-child td{{border-bottom:none}}
  tr:hover td{{background:rgba(255,255,255,.03)}}
  footer{{text-align:center;padding:1.5em;color:#3A4E70;font-size:.8em}}
</style></head><body>
<header>
  <h1>FAWP Validation — {r.ticker} [{r.timeframe}]</h1>
  <p>{r.n_signals} signal dates · {r.n_prices} price bars · {r.generated_date}</p>
  <p>fawp-index v{_VERSION}</p>
</header>
<div class="container">
<h2>Forward-return statistics by horizon</h2>
<table><thead><tr>
  <th>Horizon</th><th>N</th><th>Mean ret</th><th>Median ret</th>
  <th>Hit rate</th><th>Avg MAE</th><th>Avg MFE</th>
  <th>Std dev</th><th>p5</th><th>p95</th>
</tr></thead><tbody>{rows}</tbody></table>
{"<h2>FAWP vs baseline comparison</h2><table><thead><tr><th>Horizon</th><th>FAWP mean</th><th>Baseline mean</th><th>Difference</th></tr></thead><tbody>" + comp_rows + "</tbody></table>" if comp_rows else ""}
</div>
<footer>fawp-index v{_VERSION} · Ralph Clayton (2026) ·
doi:10.5281/zenodo.18673949</footer>
</body></html>"""
