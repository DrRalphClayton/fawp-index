"""
Research preset — verbose, includes math notation.
Best for: email, Slack research channels, logging to file.
"""

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
