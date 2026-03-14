"""
fawp_index.scanner.presets — Preset asset-class watchlists
============================================================

Pre-configured ticker lists for common use cases, each with sensible
defaults for window size and timeframes.

All functions require yfinance (``pip install yfinance``).
"""

from __future__ import annotations
from typing import List, Optional

# ── Preset ticker lists ───────────────────────────────────────────────────────

PRESETS = {
    "crypto": {
        "tickers":    ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "BNB-USD",
                       "ADA-USD", "AVAX-USD", "DOGE-USD"],
        "period":     "1y",
        "window":     90,
        "step":       3,
        "tau_max":    20,
        "timeframes": ["1d"],
        "label":      "Crypto Pairs",
    },
    "equities": {
        "tickers":    ["SPY", "QQQ", "IWM", "DIA", "VTI", "VXUS"],
        "period":     "2y",
        "window":     252,
        "step":       5,
        "tau_max":    40,
        "timeframes": ["1d", "1wk"],
        "label":      "US Equity Indices",
    },
    "sectors": {
        "tickers":    ["XLK", "XLF", "XLE", "XLV", "XLI",
                       "XLC", "XLY", "XLP", "XLB", "XLRE", "XLU"],
        "period":     "2y",
        "window":     252,
        "step":       5,
        "tau_max":    40,
        "timeframes": ["1d"],
        "label":      "US Sector ETFs",
    },
    "etfs": {
        "tickers":    ["SPY", "QQQ", "GLD", "SLV", "TLT", "HYG",
                       "EEM", "EFA", "VNQ", "DBC", "UUP"],
        "period":     "2y",
        "window":     252,
        "step":       5,
        "tau_max":    40,
        "timeframes": ["1d"],
        "label":      "Broad ETFs",
    },
    "macro": {
        "tickers":    ["TLT", "TIP", "HYG", "LQD", "DXY",
                       "GLD", "USO", "UUP"],
        "period":     "3y",
        "window":     252,
        "step":       10,
        "tau_max":    60,
        "timeframes": ["1wk"],
        "label":      "Macro Instruments",
    },
}


# ── Shared runner ─────────────────────────────────────────────────────────────

def _run_preset(
    preset_key:  str,
    tickers:     Optional[List[str]] = None,
    period:      Optional[str]       = None,
    timeframes:  Optional[List[str]] = None,
    window:      Optional[int]       = None,
    step:        Optional[int]       = None,
    tau_max:     Optional[int]       = None,
    n_null:      int                 = 0,
    max_workers: int                 = 4,
    verbose:     bool                = True,
    **kwargs,
):
    """Run a scan from a named preset, with optional overrides."""
    from fawp_index.watchlist import scan_watchlist

    cfg = PRESETS[preset_key].copy()
    tickers    = tickers    or cfg["tickers"]
    period     = period     or cfg["period"]
    timeframes = timeframes or cfg["timeframes"]
    window     = window     or cfg["window"]
    step       = step       or cfg["step"]
    tau_max    = tau_max    or cfg["tau_max"]

    if verbose:
        print(f"fawp-scan: {cfg['label']} — {len(tickers)} tickers × {timeframes}")

    return scan_watchlist(
        tickers,
        period      = period,
        timeframes  = timeframes,
        window      = window,
        step        = step,
        tau_max     = tau_max,
        n_null      = n_null,
        max_workers = max_workers,
        verbose     = verbose,
        **kwargs,
    )


# ── Public preset functions ───────────────────────────────────────────────────

def scan_crypto(
    tickers:    Optional[List[str]] = None,
    period:     str                 = "1y",
    timeframes: Optional[List[str]] = None,
    **kwargs,
):
    """
    Scan major crypto pairs for FAWP regimes.

    Default tickers: BTC-USD, ETH-USD, SOL-USD, XRP-USD, BNB-USD,
                     ADA-USD, AVAX-USD, DOGE-USD

    Requires: ``pip install yfinance``

    Examples
    --------
    ::

        from fawp_index.scanner import scan_crypto
        result = scan_crypto()
        result = scan_crypto(["BTC-USD", "ETH-USD"], period="6mo")
        print(result.summary())
        result.to_html("crypto_scan.html")
    """
    return _run_preset("crypto", tickers=tickers, period=period,
                       timeframes=timeframes, **kwargs)


def scan_equities(
    tickers:    Optional[List[str]] = None,
    period:     str                 = "2y",
    timeframes: Optional[List[str]] = None,
    **kwargs,
):
    """
    Scan US equity index ETFs for FAWP regimes.

    Default tickers: SPY, QQQ, IWM, DIA, VTI, VXUS

    Examples
    --------
    ::

        from fawp_index.scanner import scan_equities
        result = scan_equities()
        result = scan_equities(["SPY", "QQQ"], timeframes=["1d","1wk"])
        print(result.summary())
    """
    return _run_preset("equities", tickers=tickers, period=period,
                       timeframes=timeframes, **kwargs)


def scan_sectors(
    tickers:    Optional[List[str]] = None,
    period:     str                 = "2y",
    timeframes: Optional[List[str]] = None,
    **kwargs,
):
    """
    Scan US sector ETFs for FAWP regimes.

    Default tickers: XLK XLF XLE XLV XLI XLC XLY XLP XLB XLRE XLU

    Examples
    --------
    ::

        from fawp_index.scanner import scan_sectors
        result = scan_sectors()
        print(result.summary())
        result.rank_by("gap")       # which sector has widest leverage gap?
    """
    return _run_preset("sectors", tickers=tickers, period=period,
                       timeframes=timeframes, **kwargs)


def scan_etfs(
    tickers:    Optional[List[str]] = None,
    period:     str                 = "2y",
    timeframes: Optional[List[str]] = None,
    **kwargs,
):
    """
    Scan a broad cross-asset ETF universe.

    Default tickers: SPY QQQ GLD SLV TLT HYG EEM EFA VNQ DBC UUP

    Examples
    --------
    ::

        from fawp_index.scanner import scan_etfs
        result = scan_etfs()
        result.to_html("etf_scan.html")
    """
    return _run_preset("etfs", tickers=tickers, period=period,
                       timeframes=timeframes, **kwargs)


def scan_macro(
    tickers:    Optional[List[str]] = None,
    period:     str                 = "3y",
    timeframes: Optional[List[str]] = None,
    **kwargs,
):
    """
    Scan macro instruments (rates, credit, commodities, dollar).

    Default tickers: TLT TIP HYG LQD DXY GLD USO UUP
    Default timeframe: weekly (1wk)

    Examples
    --------
    ::

        from fawp_index.scanner import scan_macro
        result = scan_macro()
        result.rank_by("persistence")
    """
    return _run_preset("macro", tickers=tickers, period=period,
                       timeframes=timeframes, **kwargs)
