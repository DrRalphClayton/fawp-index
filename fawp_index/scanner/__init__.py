"""
fawp_index.scanner — Preset scanners for common asset classes
=============================================================

Quick access to pre-configured watchlist scans for equities, sectors,
crypto pairs, ETFs, and macro series.

Usage::

    from fawp_index.scanner import scan_crypto, scan_equities, scan_sectors

    result = scan_crypto()          # BTC, ETH, SOL, XRP, BNB
    result = scan_equities()        # SPY, QQQ, IWM, DIA, VTI
    result = scan_sectors()         # XLK, XLF, XLE, XLV, XLI, ...

    print(result.summary())
    result.to_html("scan.html")

Or via the fawp-scan CLI::

    fawp-scan BTC-USD
    fawp-scan SPY QQQ GLD --out scan.html
    fawp-scan --preset crypto
    fawp-scan --preset sectors --timeframe 1wk
"""

from .presets import (
    scan_crypto,
    scan_equities,
    scan_sectors,
    scan_etfs,
    scan_macro,
    PRESETS,
)

__all__ = [
    "scan_crypto",
    "scan_equities",
    "scan_sectors",
    "scan_etfs",
    "scan_macro",
    "PRESETS",
]
