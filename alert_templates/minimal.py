"""
Minimal preset — plain text, no emoji, clean for webhooks and log files.
Best for: custom webhooks, log ingestion, downstream systems.
"""

MINIMAL = {
    "NEW_FAWP": (
        "FAWP {ticker} {timeframe} score={score:.4f} gap={gap:.4f} "
        "odw={odw} severity={severity} ts={timestamp}"
    ),
    "REGIME_END": (
        "CLEARED {ticker} {timeframe} score={score:.4f} ts={timestamp}"
    ),
    "GAP_THRESHOLD": (
        "GAP_ALERT {ticker} {timeframe} gap={gap:.4f} score={score:.4f} ts={timestamp}"
    ),
    "HORIZON_COLLAPSE": (
        "HORIZON {ticker} {timeframe} score={score:.4f} ts={timestamp}"
    ),
    "DAILY_SUMMARY": (
        "SUMMARY {ticker} score={score:.4f} ts={timestamp} v={version}"
    ),
}
