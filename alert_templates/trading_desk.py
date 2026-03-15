"""
Trading Desk preset — concise, emoji-rich, channel-friendly.
Best for: Telegram, Discord, terminal, quick Slack notifications.
"""

TRADING_DESK = {
    "NEW_FAWP": (
        "🔴 FAWP | {ticker} [{timeframe}] | "
        "score {score:.4f} · gap {gap:.4f}b | ODW {odw} | {severity}"
    ),
    "REGIME_END": (
        "🟢 CLEARED | {ticker} [{timeframe}] | "
        "score {score:.4f} | {timestamp}"
    ),
    "GAP_THRESHOLD": (
        "⚡ GAP ≥ threshold | {ticker} [{timeframe}] | "
        "gap {gap:.4f}b · score {score:.4f} | {severity}"
    ),
    "HORIZON_COLLAPSE": (
        "⚠️ HORIZON | {ticker} [{timeframe}] | "
        "agency horizon collapsed | score {score:.4f}"
    ),
    "DAILY_SUMMARY": (
        "📋 DAILY | {ticker} | score {score:.4f} | {timestamp}"
    ),
}
