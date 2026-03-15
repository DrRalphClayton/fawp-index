"""
fawp_index.alert_template_presets — Built-in alert message template presets.

Three presets for common use-cases:

  TRADING_DESK  — concise, emoji-rich (Telegram / Discord / terminal)
  RESEARCH      — verbose with math notation (email / Slack research)
  MINIMAL       — plain text, no emoji (webhooks / log files)

Usage::

    from fawp_index.alerts import AlertEngine
    from fawp_index.alert_template_presets import TRADING_DESK, RESEARCH, MINIMAL

    engine = AlertEngine(gap_threshold=0.05, state_path="state.json")
    engine.add_slack("https://hooks.slack.com/services/...")

    # Apply a full preset
    for alert_type, template in TRADING_DESK.items():
        engine.set_template(alert_type, template)

    # Or override just one type
    engine.set_template("NEW_FAWP", RESEARCH["NEW_FAWP"])

Ralph Clayton (2026) · doi:10.5281/zenodo.18673949
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

RESEARCH = {
    "NEW_FAWP": (
        "[FAWP DETECTED] {ticker} [{timeframe}]  "
        "α₂ score = {score:.6f}  |  leverage gap = {gap:.6f} bits  |  "
        "ODW: {odw}  |  severity = {severity}  |  "
        "I(Dt;Xt+Δ) > η ∧ I(At;Ot+τ+1) ≤ ε  |  "
        "fawp-index v{version}  |  {timestamp}"
    ),
    "REGIME_END": (
        "[FAWP CLEARED] {ticker} [{timeframe}]  "
        "Steering-prediction coupling restored.  "
        "score = {score:.6f}  |  {timestamp}"
    ),
    "GAP_THRESHOLD": (
        "[GAP ALERT] {ticker} [{timeframe}]  "
        "Leverage gap = {gap:.6f} bits ≥ configured threshold.  "
        "score = {score:.6f}  |  severity = {severity}  |  "
        "ODW: {odw}  |  {timestamp}"
    ),
    "HORIZON_COLLAPSE": (
        "[HORIZON] {ticker} [{timeframe}]  "
        "Agency horizon τ⁺ₕ collapsed below warning threshold.  "
        "score = {score:.6f}  |  "
        "doi:10.5281/zenodo.18663547  |  {timestamp}"
    ),
    "DAILY_SUMMARY": (
        "[DAILY SUMMARY] {ticker}  "
        "FAWP-index daily digest  |  score = {score:.4f}  |  {timestamp}  |  "
        "fawp-index v{version}"
    ),
}

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

ALL_PRESETS = {
    "trading_desk": TRADING_DESK,
    "research":     RESEARCH,
    "minimal":      MINIMAL,
}
