"""
fawp_index.market — Rolling FAWP detection on market data
==========================================================

Scans a price (and optionally volume) DataFrame for FAWP regimes —
windows where directional forecastability persists while market-impact
effectiveness has already collapsed.

Financial interpretation
------------------------
In a healthy market, you can forecast direction AND your orders move
price.  A FAWP regime means:

  • pred channel  I(return_t ; return_{t+Δ}) stays above null      — you
                  can still tell which way the market is going

  • steer channel I(signed_flow_t ; return_{t+τ}) has collapsed     — but
                  your orders no longer move price the way they used to

This is the market analogue of the lab result: information advantage
survives after execution leverage is gone.

Quick start
-----------
    import pandas as pd
    from fawp_index.market import scan_fawp_market

    df = pd.read_csv("SPY.csv", parse_dates=["Date"], index_col="Date")
    scan = scan_fawp_market(df, close_col="Close", volume_col="Volume")

    print(scan.summary())
    scan.plot()
    scan.to_html("spy_fawp.html")
    scan.to_csv("spy_fawp.csv")

    # Inspect each window
    for w in scan.windows:
        if w.fawp_found:
            print(w.date, w.regime_score, w.odw_result.odw_start, w.odw_result.odw_end)

Works without volume
--------------------
    scan = scan_fawp_market(df, close_col="Close")
    # steer channel falls back to lagged-return autocorrelation

Works with pre-built signal columns
------------------------------------
    # Supply your own pred and steer series directly
    scan = scan_fawp_market(
        df,
        pred_col  = "my_forecast_signal",
        steer_col = "my_impact_signal",
    )

Speed vs rigour
---------------
    # Fast (no null correction, default):
    scan = scan_fawp_market(df, n_null=0)

    # Rigorous (shuffle + shift null at each tau):
    scan = scan_fawp_market(df, n_null=100, beta_null=0.99)

Ralph Clayton (2026) · https://doi.org/10.5281/zenodo.18673949
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date as _date
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from fawp_index import __version__ as _VERSION
_DOI     = "https://doi.org/10.5281/zenodo.18673949"
_GITHUB  = "https://github.com/DrRalphClayton/fawp-index"


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MarketScanConfig:
    """
    All tuning parameters for FAWPMarketScanner in one place.

    Parameters
    ----------
    window : int
        Rolling window length in rows (trading days). Default 252 (≈1 year).
    step : int
        How many rows to advance between scans. Default 5 (weekly).
    delta_pred : int
        Forecast horizon Δ for predictive MI. Default 20 (≈1 month).
    tau_min : int
        Minimum steering lag to test. Default 1.
    tau_max : int
        Maximum steering lag to test. Default 40.
    tau_step : int
        Step between tau values. Default 1.
    epsilon : float
        MI threshold below which a channel is considered collapsed. Default 0.01.
    n_null : int
        Null permutations per tau. 0 = no null correction (fast).
        ≥50 recommended for rigorous use. Default 0.
    beta_null : float
        Null quantile for conservative floor. Default 0.99.
    min_n : int
        Minimum paired observations required to compute MI. Default 30.
    persistence_m : int
        Persistence rule numerator (m-of-n). Default 3.
    persistence_n : int
        Persistence rule denominator. Default 4.
    seed : int
        Random seed for null sampling. Default 42.
    close_col : str
        Column name for close prices. Default "Close".
    volume_col : str or None
        Column name for volume. None = no volume (steer fallback). Default "Volume".
    pred_col : str or None
        Pre-built predictor signal column. Overrides default return-based pred.
    steer_col : str or None
        Pre-built steering signal column. Overrides default flow-based steer.
    date_col : str or None
        Date column name if index is not already datetime. Default None.
    returns_log : bool
        Use log returns instead of simple returns. Default True.
    """
    window:        int   = 252
    step:          int   = 5
    delta_pred:    int   = 20
    tau_min:       int   = 1
    tau_max:       int   = 40
    tau_step:      int   = 1
    epsilon:       float = 0.01
    n_null:        int   = 0
    beta_null:     float = 0.99
    min_n:         int   = 30
    persistence_m: int   = 3
    persistence_n: int   = 4
    seed:          int   = 42
    close_col:     str   = "Close"
    volume_col:    Optional[str] = "Volume"
    pred_col:      Optional[str] = None
    steer_col:     Optional[str] = None
    date_col:      Optional[str] = None
    returns_log:   bool  = True


# ─────────────────────────────────────────────────────────────────────────────
# Per-window result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MarketWindowResult:
    """
    FAWP detection result for a single rolling window.

    Attributes
    ----------
    date : pd.Timestamp
        End date of the window.
    window_start : pd.Timestamp
        Start date of the window.
    fawp_found : bool
        Whether FAWP was detected in this window.
    regime_score : float
        Continuous signal 0–1.  Peak leverage gap / (1 + peak leverage gap).
        0 = no leverage gap; higher = stronger FAWP signal.
    odw_result : ODWResult
        Full ODW detection result.
    tau : np.ndarray
        Tau grid used.
    pred_mi : np.ndarray
        Null-corrected pred MI per tau.
    steer_mi : np.ndarray
        Null-corrected steer MI per tau.
    pred_mi_raw : np.ndarray
        Raw (uncorrected) pred MI per tau.
    steer_mi_raw : np.ndarray
        Raw (uncorrected) steer MI per tau.
    n_obs : int
        Number of observations in the window.
    """
    date:          pd.Timestamp
    window_start:  pd.Timestamp
    fawp_found:    bool
    regime_score:  float
    odw_result:    object   # ODWResult
    tau:           np.ndarray
    pred_mi:       np.ndarray
    steer_mi:      np.ndarray
    pred_mi_raw:   np.ndarray
    steer_mi_raw:  np.ndarray
    n_obs:         int

    @property
    def fawp_score(self) -> dict:
        """One-glance FAWP score. Keys: score(0-100), prediction, control, regime, gap_bits, odw."""
        score      = int(round(float(self.regime_score) * 100))
        pred_mean  = float(self.pred_mi.mean())  if len(self.pred_mi)  else 0.0
        steer_mean = float(self.steer_mi.mean()) if len(self.steer_mi) else 0.0
        pred_tier  = "HIGH" if pred_mean  > 0.05 else ("MEDIUM" if pred_mean  > 0.01 else "LOW")
        steer_tier = "LOW"  if steer_mean < 0.01 else ("MEDIUM" if steer_mean < 0.05 else "HIGH")
        regime     = "FAWP" if self.fawp_found else ("WATCHING" if score > 5 else "CLEAR")
        odw = (f"\u03c4 {self.odw_result.odw_start}\u2013{self.odw_result.odw_end}"
               if self.odw_result.odw_start is not None else "\u2014")
        return {"score": score, "prediction": pred_tier, "control": steer_tier,
                "regime": regime, "gap_bits": round(float(self.odw_result.peak_gap_bits), 4), "odw": odw}

    def fawp_score_str(self) -> str:
        """Human-readable one-glance FAWP score card."""
        s = self.fawp_score
        return (f"FAWP Score  : {s['score']}/100\n"
                f"Prediction  : {s['prediction']}\n"
                f"Control     : {s['control']}\n"
                f"Regime      : {s['regime']}\n"
                f"Leverage gap: {s['gap_bits']} bits\n"
                f"ODW         : {s['odw']}")

    def to_dict(self) -> dict:
        r = self.odw_result
        return {
            "date":          str(self.date.date()),
            "window_start":  str(self.window_start.date()),
            "fawp_found":    bool(self.fawp_found),
            "regime_score":  round(float(self.regime_score), 6),
            "tau_h_plus":    r.tau_h_plus,
            "tau_f":         r.tau_f,
            "odw_start":     r.odw_start,
            "odw_end":       r.odw_end,
            "odw_size":      r.odw_size,
            "peak_gap_bits": round(float(r.peak_gap_bits), 6),
            "n_obs":         self.n_obs,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Scan result (full time series)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MarketScanSeries:
    """
    Complete rolling scan result.

    Attributes
    ----------
    windows : list of MarketWindowResult
        One entry per scan step.
    config : MarketScanConfig
    ticker : str
        Identifier string (passed through from scanner).
    dates : pd.DatetimeIndex
        End dates of all windows.
    regime_scores : np.ndarray
        Continuous FAWP signal (0–1) per window.
    fawp_flags : np.ndarray of bool
        Binary FAWP flag per window.
    fawp_fraction : float
        Fraction of windows where FAWP was detected.
    """
    windows:        List[MarketWindowResult]
    config:         MarketScanConfig
    ticker:         str
    dates:          pd.DatetimeIndex
    regime_scores:  np.ndarray
    fawp_flags:     np.ndarray
    fawp_fraction:  float

    # ── summary ──────────────────────────────────────────────────────────────

    def summary(self) -> str:
        n_fawp = int(np.sum(self.fawp_flags))
        n_total = len(self.windows)
        peak_w = max(self.windows, key=lambda w: w.regime_score)
        lines = [
            "=" * 60,
            f"  FAWP Market Scan — {self.ticker}",
            "=" * 60,
            f"  Windows scanned : {n_total}",
            f"  FAWP detected   : {n_fawp}  ({self.fawp_fraction*100:.1f}%)",
            f"  Rolling window  : {self.config.window} bars",
            f"  Scan step       : {self.config.step} bars",
            f"  Tau range       : {self.config.tau_min}–{self.config.tau_max}",
            f"  Null correction : {'n_null=' + str(self.config.n_null) if self.config.n_null else 'none (fast mode)'}",
            "",
            f"  Peak window     : {peak_w.date.date()}",
            f"  Peak score      : {peak_w.regime_score:.4f}",
            f"  Peak ODW        : {peak_w.odw_result.odw_start}–{peak_w.odw_result.odw_end}",
            f"  Peak gap (bits) : {peak_w.odw_result.peak_gap_bits:.4f}",
            "=" * 60,
        ]
        return "\n".join(lines)

    # ── to_dataframe ─────────────────────────────────────────────────────────

    def to_dataframe(self) -> pd.DataFrame:
        """Return all window results as a tidy DataFrame."""
        rows = [w.to_dict() for w in self.windows]
        return pd.DataFrame(rows)

    # ── to_csv ───────────────────────────────────────────────────────────────

    def to_csv(self, path: Union[str, Path]) -> Path:
        """Write tidy result DataFrame to CSV."""
        p = Path(path)
        self.to_dataframe().to_csv(p, index=False)
        return p

    # ── to_json ──────────────────────────────────────────────────────────────

    def to_json(self, path: Union[str, Path], indent: int = 2) -> Path:
        """Write full scan result to JSON."""
        p = Path(path)
        payload = {
            "meta": {
                "generated_date":    _date.today().isoformat(),
                "fawp_index_version": _VERSION,
                "doi":               _DOI,
                "ticker":            self.ticker,
            },
            "config": {k: v for k, v in self.config.__dict__.items()},
            "summary": {
                "n_windows":      len(self.windows),
                "n_fawp":         int(np.sum(self.fawp_flags)),
                "fawp_fraction":  float(self.fawp_fraction),
                "peak_score":     float(self.regime_scores.max()),
                "peak_date":      str(self.dates[int(np.argmax(self.regime_scores))].date()),
            },
            "windows": [w.to_dict() for w in self.windows],
        }
        p.write_text(json.dumps(payload, indent=indent))
        return p

    # ── plot ─────────────────────────────────────────────────────────────────

    def plot(
        self,
        prices: Optional[pd.Series] = None,
        show: bool = True,
        save_path: Optional[str] = None,
        figsize: tuple = (13, 6),
    ):
        """
        Two-panel chart:
        - Top: price (if supplied) or regime score shaded by FAWP flag
        - Bottom: continuous regime score with FAWP windows shaded red

        Parameters
        ----------
        prices : pd.Series, optional
            Close price series indexed by date.  If None, only score panel shown.
        show : bool
            Call plt.show() after plotting.
        save_path : str, optional
            Save figure to this path.
        figsize : tuple
            Figure size.

        Returns
        -------
        matplotlib Figure
        """
        try:
            import matplotlib
            matplotlib.use("Agg" if not show else matplotlib.get_backend())
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            raise ImportError("pip install fawp-index[plot]")

        has_prices = prices is not None and len(prices) > 0
        nrows = 2 if has_prices else 1
        fig, axes = plt.subplots(
            nrows, 1,
            figsize=figsize,
            sharex=False,
            gridspec_kw={"height_ratios": [2, 1] if has_prices else [1]},
        )
        if nrows == 1:
            axes = [axes]

        dates_num = np.arange(len(self.dates))

        # ── Price panel ───────────────────────────────────────────────────
        if has_prices:
            ax_price = axes[0]
            # Align prices to scan dates
            aligned = prices.reindex(self.dates, method="ffill")
            ax_price.plot(dates_num, aligned.values, lw=1.5,
                          color="#0E2550", label=self.ticker or "Price")
            # Shade FAWP windows
            for i, w in enumerate(self.windows):
                if w.fawp_found:
                    ax_price.axvspan(i - 0.5, i + 0.5, alpha=0.18,
                                     color="#C0111A", zorder=0)
            ax_price.set_ylabel("Price", fontsize=9)
            ax_price.set_title(
                f"FAWP Market Scan — {self.ticker}  "
                f"({self.fawp_fraction*100:.1f}% windows flagged)",
                fontsize=10,
            )
            ax_price.legend(fontsize=8)
            ax_price.grid(True, alpha=0.2)
            ax_price.set_xticks([])

        # ── Regime score panel ────────────────────────────────────────────
        ax_score = axes[-1]
        colors = ["#C0111A" if f else "#1a7a1a" for f in self.fawp_flags]
        ax_score.bar(dates_num, self.regime_scores, color=colors,
                     alpha=0.75, width=1.0, edgecolor="none")
        ax_score.axhline(0, color="black", lw=0.7)
        # Threshold line at epsilon-equivalent score
        eps_score = self.config.epsilon / (1 + self.config.epsilon)
        ax_score.axhline(eps_score, color="grey", lw=0.8, ls="--",
                         label=f"ε threshold")

        # x-axis labels: show ~8 evenly spaced dates
        n = len(self.dates)
        ticks = np.linspace(0, n - 1, min(8, n), dtype=int)
        ax_score.set_xticks(ticks)
        ax_score.set_xticklabels(
            [str(self.dates[i].date()) for i in ticks],
            rotation=25, ha="right", fontsize=7,
        )
        ax_score.set_ylabel("Regime score", fontsize=9)
        ax_score.set_ylim(0, min(1.05, self.regime_scores.max() * 1.15 + 0.05))
        ax_score.legend(fontsize=7)
        ax_score.grid(True, alpha=0.2, axis="y")

        fawp_patch   = mpatches.Patch(color="#C0111A", alpha=0.75, label="FAWP detected")
        no_fawp_patch = mpatches.Patch(color="#1a7a1a", alpha=0.75, label="No FAWP")
        ax_score.legend(handles=[fawp_patch, no_fawp_patch], fontsize=7, loc="upper left")

        if not has_prices:
            ax_score.set_title(
                f"FAWP Market Scan — {self.ticker}  "
                f"({self.fawp_fraction*100:.1f}% windows flagged)",
                fontsize=10,
            )

        fig.text(0.99, 0.01, f"fawp-index v{_VERSION} | Clayton (2026)",
                 ha="right", fontsize=7, color="grey", style="italic")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            try:
                plt.show()
            except Exception:
                pass
        return fig

    # ── to_html ──────────────────────────────────────────────────────────────

    def to_html(self, path: Union[str, Path]) -> Path:
        """Write self-contained HTML report. Returns Path."""
        p = Path(path)
        p.write_text(_scan_html(self))
        return p

    # ── latest ───────────────────────────────────────────────────────────────

    @property
    def latest(self) -> MarketWindowResult:
        """Most recent window result."""
        return self.windows[-1]

    @property
    def peak(self) -> MarketWindowResult:
        """Window with highest regime score."""
        return max(self.windows, key=lambda w: w.regime_score)

    @property
    def fawp_windows(self) -> List[MarketWindowResult]:
        """All windows where FAWP was detected."""
        return [w for w in self.windows if w.fawp_found]


# ─────────────────────────────────────────────────────────────────────────────
# MI helpers (self-contained, no dependency on significance.py internals)
# ─────────────────────────────────────────────────────────────────────────────

def _mi(x: np.ndarray, y: np.ndarray, min_n: int = 20) -> float:
    """Gaussian MI from Pearson r (bits).  Returns 0 if insufficient data."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < min_n:
        return 0.0
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    rho = float(np.corrcoef(x, y)[0, 1])
    if not np.isfinite(rho):
        return 0.0
    rho = float(np.clip(rho, -0.999999, 0.999999))
    return float(-0.5 * np.log(1.0 - rho ** 2) / np.log(2.0))


def _null_floor(
    x: np.ndarray,
    y: np.ndarray,
    n_null: int,
    beta: float,
    rng: np.random.Generator,
    min_n: int = 20,
) -> float:
    """Conservative null floor = max(q_beta shuffle, q_beta shift)."""
    if n_null == 0:
        return 0.0
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = x.size
    if n < min_n or n < 3:
        return 0.0
    sh = [_mi(x, rng.permutation(y), min_n) for _ in range(n_null)]
    sf = [_mi(x, np.roll(y, int(rng.integers(1, n))), min_n) for _ in range(n_null)]
    return float(max(np.quantile(sh, beta), np.quantile(sf, beta)))


# ─────────────────────────────────────────────────────────────────────────────
# Signal builders
# ─────────────────────────────────────────────────────────────────────────────

def _log_returns(prices: np.ndarray) -> np.ndarray:
    """Log returns, NaN at index 0."""
    r = np.full_like(prices, np.nan, dtype=float)
    r[1:] = np.log(prices[1:] / prices[:-1])
    return r


def _signed_flow(prices: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    """
    Signed-volume flow proxy (simplified OBV-style):
      flow_t = sign(return_t) * volume_t

    Represents the direction in which volume was deployed.
    """
    ret = _log_returns(prices)
    flow = np.sign(ret) * volumes
    flow[0] = np.nan
    return flow


def _build_pred_steer(
    window_df:  pd.DataFrame,
    cfg:        MarketScanConfig,
) -> tuple:
    """
    Build pred and steer series arrays from a window slice.

    Returns
    -------
    pred_arr : np.ndarray — predictor values (returns or custom)
    steer_arr : np.ndarray — steering values (flow or lagged-return or custom)
    """
    n = len(window_df)

    # ── Predictor ────────────────────────────────────────────────────────────
    if cfg.pred_col and cfg.pred_col in window_df.columns:
        pred_arr = window_df[cfg.pred_col].values.astype(float)
    else:
        closes = window_df[cfg.close_col].values.astype(float)
        if cfg.returns_log:
            pred_arr = _log_returns(closes)
        else:
            pred_arr = np.full(n, np.nan, dtype=float)
            pred_arr[1:] = (closes[1:] - closes[:-1]) / closes[:-1]

    # ── Steering ─────────────────────────────────────────────────────────────
    if cfg.steer_col and cfg.steer_col in window_df.columns:
        steer_arr = window_df[cfg.steer_col].values.astype(float)
    elif cfg.volume_col and cfg.volume_col in window_df.columns:
        closes = window_df[cfg.close_col].values.astype(float)
        vols   = window_df[cfg.volume_col].values.astype(float)
        steer_arr = _signed_flow(closes, vols)
    else:
        # Fallback: lagged return (autocorrelation proxy)
        # steer[t] = return[t-1], outcome = return[t+tau]
        steer_arr = np.roll(pred_arr, 1)
        steer_arr[0] = np.nan

    return pred_arr, steer_arr


# ─────────────────────────────────────────────────────────────────────────────
# Per-window detector
# ─────────────────────────────────────────────────────────────────────────────

def _scan_window(
    window_df:  pd.DataFrame,
    cfg:        MarketScanConfig,
    tau_arr:    np.ndarray,
    rng:        np.random.Generator,
) -> MarketWindowResult:
    """Compute FAWP detection for a single rolling window."""
    from fawp_index.detection.odw import ODWDetector

    n = len(window_df)
    pred_arr, steer_arr = _build_pred_steer(window_df, cfg)

    pred_mi_raw  = np.zeros(len(tau_arr))
    steer_mi_raw = np.zeros(len(tau_arr))
    pred_mi      = np.zeros(len(tau_arr))
    steer_mi     = np.zeros(len(tau_arr))

    delta = cfg.delta_pred

    for i, tau in enumerate(tau_arr):
        n_usable = n - max(delta, tau + 1)
        if n_usable < cfg.min_n:
            continue

        # Pred: I(return_t ; return_{t+delta})
        x_pred = pred_arr[:n_usable]
        y_pred = pred_arr[delta:delta + n_usable]
        raw_p  = _mi(x_pred, y_pred, cfg.min_n)
        pred_mi_raw[i] = raw_p

        if cfg.n_null > 0:
            floor_p = _null_floor(x_pred, y_pred, cfg.n_null,
                                  cfg.beta_null, rng, cfg.min_n)
        else:
            floor_p = 0.0
        pred_mi[i] = max(0.0, raw_p - floor_p)

        # Steer: I(steer_t ; return_{t+tau+1})
        x_steer = steer_arr[:n_usable]
        y_steer = pred_arr[tau + 1:tau + 1 + n_usable]
        # clip y_steer to same length as x_steer
        min_len = min(len(x_steer), len(y_steer))
        x_steer = x_steer[:min_len]
        y_steer = y_steer[:min_len]
        raw_s   = _mi(x_steer, y_steer, cfg.min_n)
        steer_mi_raw[i] = raw_s

        if cfg.n_null > 0:
            floor_s = _null_floor(x_steer, y_steer, cfg.n_null,
                                  cfg.beta_null, rng, cfg.min_n)
        else:
            floor_s = 0.0
        steer_mi[i] = max(0.0, raw_s - floor_s)

    # fail_rate: zeros (no cliff reference in market context by default)
    fail_rate = np.zeros(len(tau_arr))

    det = ODWDetector(
        epsilon       = cfg.epsilon,
        persistence_m = cfg.persistence_m,
        persistence_n = cfg.persistence_n,
    )
    odw = det.detect(
        tau        = tau_arr,
        pred_corr  = pred_mi,
        steer_corr = steer_mi,
        fail_rate  = fail_rate,
    )

    # Regime score: continuous signal 0→1
    gap = float(odw.peak_gap_bits)
    regime_score = gap / (1.0 + gap)

    return MarketWindowResult(
        date         = window_df.index[-1],
        window_start = window_df.index[0],
        fawp_found   = bool(odw.fawp_found),
        regime_score = regime_score,
        odw_result   = odw,
        tau          = tau_arr.copy(),
        pred_mi      = pred_mi,
        steer_mi     = steer_mi,
        pred_mi_raw  = pred_mi_raw,
        steer_mi_raw = steer_mi_raw,
        n_obs        = n,
    )


# ─────────────────────────────────────────────────────────────────────────────
# FAWPMarketScanner
# ─────────────────────────────────────────────────────────────────────────────

class FAWPMarketScanner:
    """
    Rolling FAWP detector for financial price (and optionally volume) data.

    Parameters
    ----------
    config : MarketScanConfig, optional
        All tuning params. Created with defaults if not supplied.
    ticker : str
        Label for outputs. Default "asset".
    **config_kwargs
        Convenience: pass any MarketScanConfig field as keyword arg.

    Examples
    --------
    With defaults::

        from fawp_index.market import FAWPMarketScanner
        import pandas as pd

        df = pd.read_csv("SPY.csv", parse_dates=["Date"], index_col="Date")
        scanner = FAWPMarketScanner(ticker="SPY")
        scan = scanner.scan(df)
        print(scan.summary())
        scan.plot(prices=df["Close"])
        scan.to_html("spy_fawp.html")

    Custom config::

        scanner = FAWPMarketScanner(
            ticker  = "BTC-USD",
            window  = 180,
            step    = 7,
            tau_max = 30,
            n_null  = 50,
        )
        scan = scanner.scan(df)

    Inspect windows::

        for w in scan.fawp_windows:
            print(w.date, w.regime_score, w.odw_result.odw_start)
    """

    def __init__(
        self,
        config: Optional[MarketScanConfig] = None,
        ticker: str = "asset",
        **config_kwargs,
    ):
        if config is not None:
            self.config = config
        else:
            self.config = MarketScanConfig(**{
                k: v for k, v in config_kwargs.items()
                if k in MarketScanConfig.__dataclass_fields__
            })
        self.ticker = ticker

    def scan(
        self,
        df: pd.DataFrame,
        verbose: bool = True,
    ) -> MarketScanSeries:
        """
        Run rolling FAWP scan over a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must have a DatetimeIndex (or set date_col in config).
            Must contain config.close_col.
            Optionally config.volume_col, config.pred_col, config.steer_col.
        verbose : bool
            Print progress. Default True.

        Returns
        -------
        MarketScanSeries
        """
        cfg = self.config

        # ── Prepare DataFrame ─────────────────────────────────────────────
        df = df.copy()
        if cfg.date_col and cfg.date_col in df.columns:
            df.index = pd.to_datetime(df[cfg.date_col])
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        if cfg.close_col not in df.columns:
            raise ValueError(
                f"close_col={cfg.close_col!r} not found. "
                f"Columns: {list(df.columns)}"
            )

        n_rows = len(df)
        if n_rows < cfg.window:
            raise ValueError(
                f"DataFrame has {n_rows} rows but window={cfg.window}. "
                "Reduce window or supply more data."
            )

        tau_arr = np.arange(cfg.tau_min, cfg.tau_max + 1, cfg.tau_step, dtype=int)
        rng = np.random.default_rng(cfg.seed)

        # ── Rolling scan ──────────────────────────────────────────────────
        starts  = range(0, n_rows - cfg.window + 1, cfg.step)
        n_steps = len(list(starts))

        if verbose:
            print(
                f"FAWPMarketScanner — {self.ticker}\n"
                f"  {n_rows} rows, window={cfg.window}, step={cfg.step}\n"
                f"  {n_steps} windows · tau {cfg.tau_min}–{cfg.tau_max} · "
                f"null={'none' if cfg.n_null==0 else cfg.n_null}"
            )

        results: List[MarketWindowResult] = []
        for i, start in enumerate(range(0, n_rows - cfg.window + 1, cfg.step)):
            window_df = df.iloc[start: start + cfg.window]
            w = _scan_window(window_df, cfg, tau_arr, rng)
            results.append(w)

            if verbose and (i % max(1, n_steps // 10) == 0 or i == n_steps - 1):
                flag = "🔴 FAWP" if w.fawp_found else "🟢 none"
                print(
                    f"  [{i+1:>4}/{n_steps}]  {w.date.date()}  "
                    f"score={w.regime_score:.3f}  {flag}"
                )

        dates         = pd.DatetimeIndex([w.date for w in results])
        regime_scores = np.array([w.regime_score for w in results])
        fawp_flags    = np.array([w.fawp_found   for w in results], dtype=bool)
        fawp_fraction = float(np.mean(fawp_flags))

        if verbose:
            n_fawp = int(np.sum(fawp_flags))
            print(f"\n  Done. {n_fawp}/{n_steps} windows flagged ({fawp_fraction*100:.1f}%)")

        return MarketScanSeries(
            windows       = results,
            config        = cfg,
            ticker        = self.ticker,
            dates         = dates,
            regime_scores = regime_scores,
            fawp_flags    = fawp_flags,
            fawp_fraction = fawp_fraction,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function
# ─────────────────────────────────────────────────────────────────────────────

def scan_fawp_market(
    df: pd.DataFrame,
    ticker: str = "asset",
    *,
    close_col:  str  = "Close",
    volume_col: Optional[str] = "Volume",
    pred_col:   Optional[str] = None,
    steer_col:  Optional[str] = None,
    window:     int  = 252,
    step:       int  = 5,
    delta_pred: int  = 20,
    tau_min:    int  = 1,
    tau_max:    int  = 40,
    tau_step:   int  = 1,
    epsilon:    float = 0.01,
    n_null:     int  = 0,
    beta_null:  float = 0.99,
    min_n:      int  = 30,
    persistence_m: int = 3,
    persistence_n: int = 4,
    date_col:    Optional[str] = None,
    returns_log: bool = True,
    seed:        int  = 42,
    verbose:     bool = True,
) -> MarketScanSeries:
    """
    One-call FAWP market scan.

    Parameters
    ----------
    df : pd.DataFrame
        Price (and optionally volume) data. DatetimeIndex or date_col.
    ticker : str
        Asset label for outputs.
    close_col : str
        Column name for close prices. Default "Close".
    volume_col : str or None
        Column name for volume. None = use lagged-return fallback.
    pred_col : str or None
        Pre-built predictor signal column. Overrides return-based pred.
    steer_col : str or None
        Pre-built steering signal column. Overrides flow-based steer.
    window : int
        Rolling window length. Default 252 (≈1 year of daily bars).
    step : int
        Bars to advance between scans. Default 5 (weekly cadence).
    delta_pred : int
        Forecast horizon Δ for pred MI. Default 20 (≈1 month).
    tau_min, tau_max, tau_step : int
        Steering lag grid. Default 1–40, step 1.
    epsilon : float
        FAWP detection threshold. Default 0.01 bits.
    n_null : int
        Null permutations per tau. 0 = fast (no null correction).
    beta_null : float
        Null quantile. Default 0.99.
    min_n : int
        Minimum paired observations for MI. Default 30.
    persistence_m, persistence_n : int
        ODW persistence rule (m-of-n). Default 3-of-4.
    returns_log : bool
        Use log returns. Default True.
    seed : int
        Random seed. Default 42.
    verbose : bool
        Print progress. Default True.

    Returns
    -------
    MarketScanSeries

    Examples
    --------
    ::

        import pandas as pd
        from fawp_index.market import scan_fawp_market

        df = pd.read_csv("SPY.csv", parse_dates=["Date"], index_col="Date")

        # Fast (no null correction):
        scan = scan_fawp_market(df, ticker="SPY", close_col="Adj Close")

        # With null correction:
        scan = scan_fawp_market(df, ticker="SPY", n_null=50)

        print(scan.summary())
        scan.plot(prices=df["Adj Close"])
        scan.to_html("spy_fawp.html")
        scan.to_csv("spy_fawp.csv")
        scan.to_json("spy_fawp.json")
    """
    cfg = MarketScanConfig(
        close_col    = close_col,
        volume_col   = volume_col if (volume_col and volume_col in df.columns) else None,
        pred_col     = pred_col,
        steer_col    = steer_col,
        window       = window,
        step         = step,
        delta_pred   = delta_pred,
        tau_min      = tau_min,
        tau_max      = tau_max,
        tau_step     = tau_step,
        epsilon      = epsilon,
        n_null       = n_null,
        beta_null    = beta_null,
        min_n        = min_n,
        persistence_m = persistence_m,
        persistence_n = persistence_n,
        returns_log  = returns_log,
        date_col     = date_col,
        seed         = seed,
    )
    return FAWPMarketScanner(config=cfg, ticker=ticker).scan(df, verbose=verbose)


# ─────────────────────────────────────────────────────────────────────────────
# HTML renderer
# ─────────────────────────────────────────────────────────────────────────────

def _scan_html(scan: MarketScanSeries) -> str:
    cfg = scan.config
    n_fawp = int(np.sum(scan.fawp_flags))
    n_total = len(scan.windows)
    peak = scan.peak

    # Embed chart
    chart_html = ""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import base64
        import io

        fig = scan.plot(show=False)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        b64 = base64.b64encode(buf.getvalue()).decode()
        chart_html = (
            f'<img src="data:image/png;base64,{b64}" '
            'style="max-width:100%;border:1px solid #ddd;border-radius:4px">'
        )
    except Exception:
        pass

    # Summary table rows
    def row(k, v, bg="#fff"):
        return (
            f'<tr style="background:{bg}">'
            f'<td style="padding:6px 12px;font-weight:500;color:#444">{k}</td>'
            f'<td style="padding:6px 12px;font-weight:700;color:#0E2550">{v}</td>'
            f'</tr>'
        )

    summary_rows = "".join([
        row("Ticker",                   scan.ticker,                      "#f8f8f8"),
        row("Windows scanned",          n_total),
        row("FAWP windows",             f"{n_fawp} ({scan.fawp_fraction*100:.1f}%)", "#f8f8f8"),
        row("Rolling window",           f"{cfg.window} bars"),
        row("Scan step",                f"{cfg.step} bars",               "#f8f8f8"),
        row("Tau range",                f"{cfg.tau_min}–{cfg.tau_max}"),
        row("Null correction",          f"n_null={cfg.n_null}" if cfg.n_null else "none (fast mode)", "#f8f8f8"),
        row("Epsilon threshold",        f"{cfg.epsilon} bits"),
        row("Persistence rule",         f"{cfg.persistence_m}-of-{cfg.persistence_n}", "#f8f8f8"),
        row("Peak date",                str(peak.date.date())),
        row("Peak regime score",        f"{peak.regime_score:.4f}",       "#f8f8f8"),
        row("Peak ODW",
            f"{peak.odw_result.odw_start}–{peak.odw_result.odw_end}"
            if peak.odw_result.odw_start is not None else "—"),
        row("Peak gap (bits)",          f"{peak.odw_result.peak_gap_bits:.4f}", "#f8f8f8"),
    ])

    # Window table (most recent 50)
    show_windows = scan.windows[-50:]
    win_rows = ""
    for w in reversed(show_windows):
        bg = "#fff5f5" if w.fawp_found else "#fff"
        flag = '<span style="color:#C0111A;font-weight:700">FAWP</span>' if w.fawp_found else "—"
        odw_str = (f"{w.odw_result.odw_start}–{w.odw_result.odw_end}"
                   if w.odw_result.odw_start is not None else "—")
        win_rows += (
            f'<tr style="background:{bg}">'
            f'<td style="padding:4px 10px">{w.date.date()}</td>'
            f'<td style="padding:4px 10px">{flag}</td>'
            f'<td style="padding:4px 10px">{w.regime_score:.4f}</td>'
            f'<td style="padding:4px 10px">{odw_str}</td>'
            f'<td style="padding:4px 10px">{w.odw_result.peak_gap_bits:.4f}</td>'
            f'<td style="padding:4px 10px">{w.n_obs}</td>'
            f'</tr>\n'
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>FAWP Market Scan — {scan.ticker}</title>
<style>
  body {{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
         max-width:980px;margin:0 auto;padding:2em 1.5em;
         background:#fafafa;color:#222;line-height:1.6}}
  header {{background:#0E2550;color:white;padding:1.8em 2em 1.4em;
           border-radius:8px;margin-bottom:1.5em}}
  header h1 {{margin:0 0 0.3em;font-size:1.45em}}
  header p  {{margin:0.2em 0;font-size:0.88em;color:#aac}}
  .badge {{display:inline-block;padding:0.35em 1em;border-radius:14px;
           font-weight:700;font-size:0.95em;color:white;margin:0.6em 0.4em 0.6em 0}}
  .badge.fawp    {{background:#C0111A}}
  .badge.no-fawp {{background:#1a7a1a}}
  h2 {{color:#0E2550;border-bottom:2px solid #D4AF37;padding-bottom:4px;margin-top:2em}}
  table {{width:100%;border-collapse:collapse;margin:1em 0;
          box-shadow:0 1px 4px rgba(0,0,0,0.07);border-radius:6px;overflow:hidden}}
  thead th {{background:#0E2550;color:white;padding:8px 10px;text-align:left;font-size:0.88em}}
  footer {{margin-top:3em;padding-top:1em;border-top:1px solid #ddd;font-size:0.8em;color:#888}}
  a {{color:#0E2550}}
</style>
</head>
<body>
<header>
  <h1>FAWP Market Scan — {scan.ticker}</h1>
  <p>Generated {_date.today().isoformat()} &bull; fawp-index v{_VERSION}</p>
  <p><a href="{_DOI}" style="color:#D4AF37">{_DOI}</a></p>
</header>

<div>
  <span class="badge fawp">{n_fawp} FAWP windows</span>
  <span class="badge no-fawp">{n_total - n_fawp} clean windows</span>
  <span style="color:#888;font-size:0.9em">{scan.fawp_fraction*100:.1f}% flagged</span>
</div>

<h2>Chart</h2>
{chart_html}

<h2>Configuration &amp; Summary</h2>
<table>
  <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
  <tbody>{summary_rows}</tbody>
</table>

<h2>Window Results (most recent 50)</h2>
<table>
  <thead>
    <tr>
      <th>End date</th><th>FAWP</th><th>Regime score</th>
      <th>ODW</th><th>Peak gap (bits)</th><th>N obs</th>
    </tr>
  </thead>
  <tbody>{win_rows}</tbody>
</table>

<footer>
  <a href="{_GITHUB}">fawp-index</a> &bull;
  Ralph Clayton (2026) &bull;
  <a href="{_DOI}">{_DOI}</a>
</footer>
</body>
</html>
"""
