"""
fawp_index.compare — Benchmark comparison: FAWP vs classic signals.

Computes FAWP regime score alongside RSI, realised volatility,
momentum (rolling return), and moving-average slope, then measures
correlation and forward-return lift for each.

Usage::

    from fawp_index.watchlist import scan_watchlist
    from fawp_index.compare import compare_signals

    prices = pd.Series(...)   # Close price series
    result = scan_watchlist({"SPY": df}, period="2y")
    asset  = result.rank_by("score")[0]
    report = compare_signals(asset, prices)
    print(report.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ── Signal builders ────────────────────────────────────────────────────────────

def _rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    delta = prices.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _realised_vol(prices: pd.Series, window: int = 21) -> pd.Series:
    return prices.pct_change().rolling(window).std() * (252 ** 0.5)


def _momentum(prices: pd.Series, window: int = 20) -> pd.Series:
    return prices.pct_change(window)


def _ma_slope(prices: pd.Series, window: int = 20) -> pd.Series:
    ma = prices.rolling(window).mean()
    return ma.diff(5) / ma.shift(5)


# ── CompareReport ──────────────────────────────────────────────────────────────

@dataclass
class SignalStats:
    name:          str
    correlation:   float   # Pearson r with FAWP score
    fwd_return_1:  float   # mean fwd return when signal is extreme (top 20%)
    fwd_return_5:  float   # mean fwd return at 5-bar horizon
    fwd_return_20: float   # mean fwd return at 20-bar horizon
    hit_rate_20:   float   # % of extreme-signal bars where 20-bar return > 0
    n_obs:         int

    def to_dict(self) -> dict:
        return {
            "signal":       self.name,
            "corr_fawp":    round(self.correlation,   3),
            "fwd_ret_1":    round(self.fwd_return_1,  4),
            "fwd_ret_5":    round(self.fwd_return_5,  4),
            "fwd_ret_20":   round(self.fwd_return_20, 4),
            "hit_rate_20":  round(self.hit_rate_20,   3),
            "n_obs":        self.n_obs,
        }


@dataclass
class CompareReport:
    ticker:     str
    timeframe:  str
    signals:    List[SignalStats]
    fawp_fwd_return_1:  float
    fawp_fwd_return_5:  float
    fawp_fwd_return_20: float
    fawp_hit_rate_20:   float
    n_obs:      int

    def summary(self) -> str:
        w = 68
        lines = [
            "=" * w,
            f"  Signal Comparison — {self.ticker} [{self.timeframe}]",
            f"  {self.n_obs} observations",
            "=" * w,
            f"  {'Signal':<18} {'Corr(FAWP)':>10}  {'Ret@1':>7}  "
            f"{'Ret@5':>7}  {'Ret@20':>7}  {'Hit%@20':>8}",
            "  " + "-" * 60,
        ]
        # FAWP row
        lines.append(
            f"  {'FAWP score':<18} {'—':>10}  "
            f"{self.fawp_fwd_return_1*100:>6.2f}%  "
            f"{self.fawp_fwd_return_5*100:>6.2f}%  "
            f"{self.fawp_fwd_return_20*100:>6.2f}%  "
            f"{self.fawp_hit_rate_20*100:>7.1f}%"
        )
        for s in self.signals:
            lines.append(
                f"  {s.name:<18} {s.correlation:>+10.3f}  "
                f"{s.fwd_return_1*100:>6.2f}%  "
                f"{s.fwd_return_5*100:>6.2f}%  "
                f"{s.fwd_return_20*100:>6.2f}%  "
                f"{s.hit_rate_20*100:>7.1f}%"
            )
        lines += ["", "=" * w]
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        rows = [{"signal": "FAWP score",
                 "corr_fawp": None,
                 "fwd_ret_1":  round(self.fawp_fwd_return_1,  4),
                 "fwd_ret_5":  round(self.fawp_fwd_return_5,  4),
                 "fwd_ret_20": round(self.fawp_fwd_return_20, 4),
                 "hit_rate_20": round(self.fawp_hit_rate_20,  3),
                 "n_obs": self.n_obs}]
        rows += [s.to_dict() for s in self.signals]
        return pd.DataFrame(rows)


# ── Core function ──────────────────────────────────────────────────────────────

def compare_signals(
    asset,
    prices: pd.Series,
    horizons: Optional[List[int]] = None,
    extreme_pct: float = 0.20,
) -> CompareReport:
    """
    Compare FAWP regime score against RSI, vol, momentum, MA-slope.

    Parameters
    ----------
    asset : AssetResult
    prices : pd.Series   Close prices, DatetimeIndex
    horizons : list of int, optional   Default [1, 5, 20]
    extreme_pct : float   Top fraction to define "extreme" signal. Default 0.20.

    Returns
    -------
    CompareReport
    """
    if horizons is None:
        horizons = [1, 5, 20]

    prices = prices.squeeze().sort_index().dropna()
    if not isinstance(prices.index, pd.DatetimeIndex):
        prices.index = pd.to_datetime(prices.index)

    # ── FAWP score series from scan windows ──────────────────────────────────
    if asset.scan is None:
        return CompareReport(
            ticker=asset.ticker, timeframe=asset.timeframe,
            signals=[], n_obs=0,
            fawp_fwd_return_1=0.0, fawp_fwd_return_5=0.0,
            fawp_fwd_return_20=0.0, fawp_hit_rate_20=0.0,
        )

    fawp_dates  = [pd.Timestamp(w.date).normalize() for w in asset.scan.windows]
    fawp_scores = [float(w.regime_score) for w in asset.scan.windows]
    fawp_s = pd.Series(fawp_scores, index=fawp_dates).sort_index()
    fawp_s = fawp_s[~fawp_s.index.duplicated(keep="last")]

    # ── Build comparison signals on price index ──────────────────────────────
    sigs: Dict[str, pd.Series] = {
        "RSI-14":          _rsi(prices, 14),
        "Realised vol-21": _realised_vol(prices, 21),
        "Momentum-20":     _momentum(prices, 20),
        "MA slope-20":     _ma_slope(prices, 20),
    }

    # Align all to FAWP dates
    def _align(s: pd.Series) -> pd.Series:
        aligned = s.reindex(fawp_s.index, method="ffill")
        return aligned

    aligned = {k: _align(v) for k, v in sigs.items()}

    # ── Forward returns ───────────────────────────────────────────────────────
    price_arr = prices.values.astype(float)
    price_idx = prices.index

    def _fwd(dt: pd.Timestamp, h: int) -> Optional[float]:
        pos = price_idx.searchsorted(dt)
        if pos + h >= len(price_arr) or price_arr[pos] <= 0:
            return None
        return price_arr[pos + h] / price_arr[pos] - 1.0

    fwd: Dict[int, List[float]] = {h: [] for h in horizons}
    for dt in fawp_s.index:
        for h in horizons:
            v = _fwd(dt, h)
            if v is not None:
                fwd[h].append(v)

    def _mean_safe(lst):
        return float(np.mean(lst)) if lst else 0.0
    def _hit_safe(lst):
        return float(np.mean([r > 0 for r in lst])) if lst else 0.0

    # FAWP extreme = top extreme_pct by score
    fawp_arr = fawp_s.values
    thresh   = np.quantile(fawp_arr, 1 - extreme_pct)
    fawp_extreme_idx = np.where(fawp_arr >= thresh)[0]

    def _extreme_fwd(scores_arr: np.ndarray, h: int) -> tuple:
        ext_thresh = np.quantile(scores_arr[np.isfinite(scores_arr)], 1 - extreme_pct)
        ext_idx    = np.where(scores_arr >= ext_thresh)[0]
        rets       = []
        for i in ext_idx:
            if i < len(fawp_s.index):
                v = _fwd(fawp_s.index[i], h)
                if v is not None:
                    rets.append(v)
        return _mean_safe(rets), _hit_safe(rets), len(rets)

    # ── Build SignalStats ─────────────────────────────────────────────────────
    signal_stats = []
    for name, sig in aligned.items():
        sig_arr = sig.values.astype(float)
        valid   = np.isfinite(sig_arr) & np.isfinite(fawp_arr)
        if valid.sum() < 10:
            continue
        corr = float(np.corrcoef(fawp_arr[valid], sig_arr[valid])[0, 1])

        r1,  h1,  n1  = _extreme_fwd(sig_arr, horizons[0])
        r5,  h5,  n5  = _extreme_fwd(sig_arr, horizons[1] if len(horizons) > 1 else 5)
        r20, h20, n20 = _extreme_fwd(sig_arr, horizons[2] if len(horizons) > 2 else 20)

        signal_stats.append(SignalStats(
            name          = name,
            correlation   = corr,
            fwd_return_1  = r1,
            fwd_return_5  = r5,
            fwd_return_20 = r20,
            hit_rate_20   = h20,
            n_obs         = n20,
        ))

    # FAWP extreme stats
    fr1,  _,   _  = _extreme_fwd(fawp_arr, horizons[0])
    fr5,  _,   _  = _extreme_fwd(fawp_arr, horizons[1] if len(horizons) > 1 else 5)
    fr20, fh20, fn = _extreme_fwd(fawp_arr, horizons[2] if len(horizons) > 2 else 20)

    return CompareReport(
        ticker     = asset.ticker,
        timeframe  = asset.timeframe,
        signals    = signal_stats,
        fawp_fwd_return_1  = fr1,
        fawp_fwd_return_5  = fr5,
        fawp_fwd_return_20 = fr20,
        fawp_hit_rate_20   = fh20,
        n_obs      = fn,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Original ODW comparison API — compare_fawp / ComparisonResult
# Compares two FAWP detection results side by side (ODW, AlphaV2, or Benchmark)
# ─────────────────────────────────────────────────────────────────────────────

import json as _json
from pathlib import Path as _Path
from typing import Union as _Union


@dataclass
class ComparisonRow:
    """One metric row in a side-by-side comparison."""
    field:   str
    value_a: str
    value_b: str
    winner:  str   # "A", "B", "tie", or ""


@dataclass
class ComparisonResult:
    """
    Side-by-side comparison of two FAWP detection results.

    Attributes
    ----------
    label_a, label_b : str
        Display names for the two results.
    result_type : str
        "ODW", "AlphaV2", or "Benchmark".
    rows : list of ComparisonRow
        Per-metric comparison rows.
    winner_overall : str
        "A", "B", or "tie".
    score_a, score_b : int
        Count of metrics won by each side.
    """
    label_a:        str
    label_b:        str
    result_type:    str
    rows:           list
    winner_overall: str
    score_a:        int
    score_b:        int

    def summary(self) -> str:
        w = 60
        lines = [
            "=" * w,
            f"  FAWP Comparison: {self.label_a}  vs  {self.label_b}",
            f"  Type: {self.result_type}",
            "=" * w,
            f"  {'Metric':<24} {'A':>12} {'B':>12} {'Winner':>6}",
            "  " + "-" * 56,
        ]
        for r in self.rows:
            lines.append(
                f"  {r.field:<24} {r.value_a:>12} {r.value_b:>12} "
                f"{'→'+r.winner if r.winner else '':>6}"
            )
        lines += [
            "  " + "-" * 56,
            f"  Score   {self.label_a}: {self.score_a}  "
            f"{self.label_b}: {self.score_b}",
            f"  Winner overall: {self.winner_overall}",
            "=" * w,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "label_a":        self.label_a,
            "label_b":        self.label_b,
            "result_type":    self.result_type,
            "winner_overall": self.winner_overall,
            "score_a":        self.score_a,
            "score_b":        self.score_b,
            "rows": [
                {"field": r.field, "value_a": r.value_a,
                 "value_b": r.value_b, "winner": r.winner}
                for r in self.rows
            ],
        }

    def to_json(self, path: _Union[str, _Path], indent: int = 2) -> _Path:
        p = _Path(path)
        p.write_text(_json.dumps(self.to_dict(), indent=indent))
        return p

    def to_html(self, path: _Union[str, _Path]) -> _Path:
        p = _Path(path)
        p.write_text(_comparison_html(self))
        return p


def _odw_rows(a, b) -> list:
    """Build comparison rows from two ODW results."""
    import numpy as _np

    def _fmt_bool(v): return "YES" if v else "NO"
    def _fmt_opt(v, fmt="{:.4f}"): return fmt.format(v) if v is not None else "—"

    rows  = []
    score_a, score_b = 0, 0

    def _row(field, va, vb, winner=""):
        rows.append(ComparisonRow(field=field, value_a=va, value_b=vb, winner=winner))
        return winner

    # fawp_found
    fa = a.fawp_found if hasattr(a, "fawp_found") else False
    fb = b.fawp_found if hasattr(b, "fawp_found") else False
    w  = "A" if fa and not fb else ("B" if fb and not fa else "tie")
    _row("FAWP found", _fmt_bool(fa), _fmt_bool(fb), w)
    if w == "A": score_a += 1
    elif w == "B": score_b += 1

    # peak_gap_bits
    ga = getattr(a, "peak_gap_bits", None) or 0.0
    gb = getattr(b, "peak_gap_bits", None) or 0.0
    w  = "A" if ga > gb else ("B" if gb > ga else "tie")
    _row("Peak gap (bits)", f"{ga:.4f}", f"{gb:.4f}", w)
    if w == "A": score_a += 1
    elif w == "B": score_b += 1

    # tau_h_plus
    tha = getattr(a, "tau_h_plus", getattr(a, "odw_result", a))
    thb = getattr(b, "tau_h_plus", getattr(b, "odw_result", b))
    tha = getattr(tha, "tau_h_plus", None) if not isinstance(tha, (int, float, type(None))) else tha
    thb = getattr(thb, "tau_h_plus", None) if not isinstance(thb, (int, float, type(None))) else thb
    w   = "A" if (tha or 0) > (thb or 0) else ("B" if (thb or 0) > (tha or 0) else "tie")
    _row("τ⁺ₕ (agency horizon)", _fmt_opt(tha, "{:.0f}"), _fmt_opt(thb, "{:.0f}"), w)
    if w == "A": score_a += 1
    elif w == "B": score_b += 1

    # ODW width
    def _width(x):
        s = getattr(x, "odw_start", None)
        e = getattr(x, "odw_end",   None)
        if s is not None and e is not None: return e - s + 1
        return None
    wa = _width(a) or _width(getattr(a, "odw_result", a))
    wb = _width(b) or _width(getattr(b, "odw_result", b))
    w  = "A" if (wa or 0) > (wb or 0) else ("B" if (wb or 0) > (wa or 0) else "tie")
    _row("ODW width (τ)", _fmt_opt(wa, "{:.0f}"), _fmt_opt(wb, "{:.0f}"), w)
    if w == "A": score_a += 1
    elif w == "B": score_b += 1

    return rows, score_a, score_b


def compare_fawp(
    result_a,
    result_b,
    label_a: str = "A",
    label_b: str = "B",
) -> ComparisonResult:
    """
    Compare two FAWP detection results side by side.

    Accepts ODWResult, ODWDetector, FAWPAlphaIndexV2, or BenchmarkResult objects.
    Both results must be of the same type.

    Parameters
    ----------
    result_a, result_b : FAWP result objects
    label_a, label_b : str   Display names.

    Returns
    -------
    ComparisonResult

    Raises
    ------
    TypeError if result types are incompatible.

    Example
    -------
    ::

        from fawp_index import ODWDetector, compare_fawp
        a = ODWDetector.from_e9_2_data(steering="u")
        b = ODWDetector.from_e9_2_data(steering="xi")
        cmp = compare_fawp(a, b, label_a="u-steer", label_b="xi-steer")
        print(cmp.summary())
    """
    cls_a = type(result_a).__name__
    cls_b = type(result_b).__name__

    # Normalise: unwrap detectors to their result objects
    def _unwrap(r):
        # ODWDetector → run if needed
        if hasattr(r, "result") and r.result is not None:
            return r.result
        if hasattr(r, "compute"):
            return r.compute()
        # BenchmarkResult → use its odw_result
        if hasattr(r, "odw_result"):
            return r.odw_result
        return r

    ua = _unwrap(result_a)
    ub = _unwrap(result_b)

    # Check type compatibility
    cls_ua = type(ua).__name__
    cls_ub = type(ub).__name__
    if cls_ua != cls_ub:
        # Allow BenchmarkResult → ODWResult normalization
        if "ODW" not in cls_ua or "ODW" not in cls_ub:
            raise TypeError(
                f"Cannot compare {cls_ua} with {cls_ub}. "
                f"Both results must be the same type."
            )

    # Determine result_type
    if "AlphaV2" in cls_a or "AlphaV2" in cls_b:
        result_type = "AlphaV2"
    else:
        result_type = "ODW"

    rows, score_a, score_b = _odw_rows(ua, ub)

    if score_a > score_b:
        winner = "A"
    elif score_b > score_a:
        winner = "B"
    else:
        winner = "tie"

    return ComparisonResult(
        label_a        = label_a,
        label_b        = label_b,
        result_type    = result_type,
        rows           = rows,
        winner_overall = winner,
        score_a        = score_a,
        score_b        = score_b,
    )


def _comparison_html(r: ComparisonResult) -> str:
    """Generate a self-contained HTML comparison report."""
    import base64, io
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import numpy as np
        fig, ax = plt.subplots(figsize=(6, 2.5))
        fig.patch.set_facecolor("#07101E")
        ax.set_facecolor("#0D1729")
        metrics = [row.field for row in r.rows]
        a_wins  = [1 if row.winner == "A" else 0 for row in r.rows]
        b_wins  = [1 if row.winner == "B" else 0 for row in r.rows]
        x = np.arange(len(metrics))
        ax.bar(x - 0.2, a_wins, 0.4, color="#D4AF37", label=r.label_a)
        ax.bar(x + 0.2, b_wins, 0.4, color="#4A7FCC", label=r.label_b)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=20, ha="right", fontsize=7, color="#7A90B8")
        ax.tick_params(colors="#7A90B8")
        ax.legend(fontsize=7, facecolor="#0D1729", labelcolor="#EDF0F8")
        for sp in ax.spines.values(): sp.set_edgecolor("#182540")
        plt.tight_layout(pad=0.4)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", facecolor="#07101E")
        plt.close(fig)
        b64 = base64.b64encode(buf.getvalue()).decode()
        chart_html = f'<img src="data:image/png;base64,{b64}" style="max-width:100%">'
    except Exception as _chart_err:
        import warnings
        warnings.warn(
            f"fawp-index: chart render failed in _comparison_html: {_chart_err}")
        chart_html = (
            '<p style="color:#c0392b;font-size:.85em">&#9888; Chart unavailable: '
            f'{_chart_err}</p>'
        )

    rows_html = "".join(
        f"<tr><td>{row.field}</td><td>{row.value_a}</td>"
        f"<td>{row.value_b}</td>"
        f"<td style='color:#D4AF37;font-weight:700'>"
        f"{'→'+row.winner if row.winner and row.winner != 'tie' else row.winner}</td></tr>"
        for row in r.rows
    )
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>FAWP Comparison</title>
<style>
body{{font-family:-apple-system,sans-serif;background:#07101E;color:#EDF0F8;padding:2em}}
h1{{color:#D4AF37;font-size:1.2em}}
table{{border-collapse:collapse;width:100%;font-size:.88em;margin-top:1em}}
th{{color:#7A90B8;text-align:left;padding:.4em .8em;border-bottom:1px solid #182540;font-size:.75em;text-transform:uppercase}}
td{{padding:.4em .8em;border-bottom:1px solid #111E35}}
</style></head><body>
<h1>FAWP Comparison — {r.label_a} vs {r.label_b}</h1>
<p style="color:#7A90B8;font-size:.85em">Winner: <b style="color:#D4AF37">{r.winner_overall}</b>
({r.label_a}: {r.score_a} · {r.label_b}: {r.score_b})</p>
{chart_html}
<table><thead><tr><th>Metric</th><th>{r.label_a}</th><th>{r.label_b}</th><th>Winner</th></tr></thead>
<tbody>{rows_html}</tbody></table>
</body></html>"""
