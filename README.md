# fawp-index

[![PyPI version](https://badge.fury.io/py/fawp-index.svg)](https://badge.fury.io/py/fawp-index)
[![PyPI downloads](https://img.shields.io/pypi/dm/fawp-index.svg)](https://pypi.org/project/fawp-index/)
[![Python 3.9–3.12](https://img.shields.io/badge/python-3.9–3.12-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/DrRalphClayton/fawp-index/actions/workflows/ci.yml/badge.svg)](https://github.com/DrRalphClayton/fawp-index/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18673949.svg)](https://doi.org/10.5281/zenodo.18673949)

---

## Screenshots

### Finance Scanner
![FAWP Finance Scanner — SPY FAWP detection](https://raw.githubusercontent.com/DrRalphClayton/fawp-index/main/docs/images/scanner_result.png)

### Weather Scanner
![FAWP Weather Scanner — London temperature](https://raw.githubusercontent.com/DrRalphClayton/fawp-index/main/docs/images/weather_result.png)



<div align="center">

### 🔴 Try it live — no install needed

**[→ fawp-scanner.info](https://fawp-scanner.info)**

Scan any stock, ETF, or crypto in your browser.
Enter tickers → Fetch data → See regime detection in real time.

*Powered by fawp-index v1.1.5 · [pip install it](#install) for full local control*

</div>

---

## Your model still predicts. But you've already lost control.

**fawp-index** detects the moment a system crosses into the
*Information-Control Exclusion Principle* regime — where **predictive
information persists** but **the ability to act on it has collapsed**.

| Domain | What predicts | What collapses |
|--------|--------------|----------------|
| 📈 Quant finance | Factor alpha signal | Market execution edge (crowding) |
| 🌀 Dynamical systems | State forecasts | Stabilising control authority |
| 🌊 Weather / climate | Forecast skill | Intervention window |
| 🌍 Seismic | Precursor signal | Stress release control |
| 🤖 ML systems | Model predictions | Ability to retrain / intervene |


---

## What it looks like

<!-- Add dashboard screenshot here: docs/images/scanner_tab.png -->
<!-- To generate: sign in → run scan on SPY QQQ GLD BTC → screenshot Scanner tab -->

**Scanner tab** — severity pills, sparklines, confidence badges, ODW bars

Each asset row shows:
- **Severity pill** — FAWP 🔴 / HIGH / WATCH / CLEAR
- **Score** — regime score (0–1) colour-coded by tier  
- **Sparkline** — 6-window score trend with ▲/▼ arrow
- **Confidence badge** — HIGH / MED / LOW based on persistence + ODW concentration
- **Gap (bits)** — leverage gap magnitude
- **ODW bar** — proportional bar showing detection window within τ range

Click any flagged asset → **"Why flagged?"** expander with full attribution.

**Compare tab** — FAWP vs RSI, realised vol, momentum, MA slope

Forward-return lift at 1 / 5 / 20 bars when each signal is in extreme zone.
FAWP row highlighted in gold. Export to CSV.

**Validation tab** — forward-return statistics after signal fires

Hit rate · mean return · MAE · MFE · p5/p95 · FAWP vs baseline comparison.

**History tab** — score timeline per asset across all saved scans

---

## Live demo

**[fawp-scanner.info](https://fawp-scanner.info)** — interactive dashboard running on live data.
No install required. Scan equities, crypto, and sectors in your browser.

---

## Install

```bash
pip install fawp-index                   # core
pip install "fawp-index[plot]"           # + matplotlib figures
pip install "fawp-index[dashboard]"      # + Streamlit dashboard
pip install "fawp-index[fast]"           # + Numba JIT (5–15× faster null scans)
pip install "fawp-index[all]"            # everything
```

---

## One-command demo

```bash
pip install "fawp-index[dashboard]"
fawp-demo                          # opens browser with synthetic data instantly
fawp-demo --asset BTC-USD SPY QQQ  # real tickers via yfinance
```

No CSV. No API key. No config. Just install and run.

---

## 60-second quickstart

```python
import numpy as np
from fawp_index import FAWPAlphaIndex

# Simulate: strong prediction, collapsed steering = FAWP
pred   = np.random.randn(5000)
future = pred[20:] + np.random.randn(4980) * 0.3   # forecastable
action = np.random.randn(4980) * 0.001              # near-zero = no steering
obs    = np.random.randn(4980) * 0.1

result = FAWPAlphaIndex().compute(pred[:4980], future, action, obs)
print(result.summary())
result.plot()   # pip install "fawp-index[plot]"
```

Output:
```
==================================================
FAWP Alpha Index v2.1 — Results Summary
==================================================
Agency Horizon (tau_h):  1
Peak Alpha Index:        0.2847
Peak Alpha at tau:       3
FAWP regime detected:   YES
FAWP tau range:          [1, 2, 3, 4, 5]
==================================================
```

---

## Run the benchmarks (zero data needed)

The fastest way to see it work:

```bash
pip install "fawp-index[plot]"
python -c "
from fawp_index import run_benchmarks
suite = run_benchmarks()
print(suite.summary())
suite.verify_all()
"
```

```
Case                  Expected   Detected   Result
----------------------------------------------------
clean_control         FAWP       FAWP       ✅ PASS
prediction_only       none       none       ✅ PASS
control_only          none       none       ✅ PASS
noisy_false_positive  none       none       ✅ PASS
delayed_collapse      FAWP       FAWP       ✅ PASS
All 5 assertions passed.
```

---

## Market scanner

```python
import pandas as pd
from fawp_index.market import scan_fawp_market

df   = pd.read_csv("SPY.csv", parse_dates=["Date"], index_col="Date")
scan = scan_fawp_market(df, ticker="SPY", close_col="Close", volume_col="Volume")

print(scan.summary())
scan.plot(prices=df["Close"])
scan.to_html("spy_fawp.html")
```

**Financial interpretation:**
- **pred channel** `I(returnₜ ; returnₜ₊Δ)` — is the market still forecastable?
- **steer channel** `I(signed_flowₜ ; returnₜ₊τ)` — do your orders still move price?
- **FAWP window** — you can forecast direction, but your orders no longer move price

---

## Watchlist scanner

```python
from fawp_index.watchlist import scan_watchlist

# Fetch and scan automatically via yfinance:
result = scan_watchlist(
    ["SPY", "QQQ", "GLD", "BTC-USD", "ETH-USD"],
    period     = "2y",
    timeframes = ["1d", "1wk"],
)

result.rank_by("score")        # strongest current regime score
result.rank_by("gap")          # widest leverage gap (bits)
result.rank_by("persistence")  # longest active regime
result.active_regimes()        # only currently flagged assets
result.top_n(5, "score")

result.to_html("watchlist.html")
```

---

## Saved watchlists

```bash
# Create and persist named watchlists
fawp-watchlist create tech AAPL MSFT NVDA AMD
fawp-watchlist create crypto BTC-USD ETH-USD SOL-USD --period 1y
fawp-watchlist scan tech --leaderboard --explain --out tech.html
fawp-watchlist list
fawp-watchlist show tech
```

---

## Leaderboard

```python
from fawp_index.leaderboard import Leaderboard

result = scan_watchlist(["SPY", "QQQ", "GLD", "BTC-USD"], period="2y")
lb = Leaderboard.from_watchlist(result)

print(lb.summary())
lb.to_html("leaderboard.html")
```

Four ranked categories: **Top FAWP** · **Rising Risk** · **Collapsing Control** · **Strongest ODW**

---

## Explain score

```python
from fawp_index.explain import explain_asset

result = scan_watchlist(["SPY", "QQQ"], period="2y")
top = result.rank_by("score")[0]
print(explain_asset(top))
```

```
============================================================
  SPY  [1d]
============================================================
  FAWP Score   : 81/100
  Status       : 🔴 HIGH  —  FAWP ACTIVE  (12 days)
------------------------------------------------------------
  Prediction   : elevated (0.142 bits)
  Steering     : collapsed (0.0001 bits ≈ zero)
  Leverage gap : large (0.141 bits)
  ODW          : detected (τ = 1–12)
------------------------------------------------------------
  Why flagged:
    • FAWP active 12 days — score rising 3 consecutive windows
    • Steering MI below ε at 11 of 12 tau values
    • Leverage gap 0.141 bits
    • ODW spans 12 tau steps (τ 1–12)
------------------------------------------------------------
  Recommendation:
  Prediction persists but execution edge has collapsed.
  Reduce size or investigate crowding / latency conditions.
============================================================
```

---

## Alerts

```python
from fawp_index.alerts import AlertEngine, AlertSeverity

engine = AlertEngine(
    gap_threshold=0.05,
    cooldown_hours=4,                        # suppress repeats within 4h
    min_consecutive_windows=2,               # only fire after 2 flagged windows
    score_change_threshold=0.02,             # only if score changed ≥ 0.02
    min_severity=AlertSeverity.MEDIUM,       # ignore LOW signals
    state_path="fawp_state.json",
)
engine.add_terminal()
engine.add_telegram(token="BOT_TOKEN", chat_id="CHAT_ID")
engine.add_discord(webhook_url="https://discord.com/api/webhooks/...")
engine.add_email(smtp_host="smtp.gmail.com", username="you@gmail.com",
                 password="app_password", to_addrs=["you@gmail.com"])
engine.add_webhook("https://hooks.slack.com/services/...")

result = scan_watchlist(dfs)
alerts = engine.check(result)
engine.daily_summary(result)
```

---

## Significance testing

```python
from fawp_index import ODWDetector, fawp_significance

odw = ODWDetector.from_e9_2_data()
sig = fawp_significance(odw, n_bootstrap=200)

print(sig.summary())
# p_fawp=1.000  p_null=0.145  significant=YES
# ci_tau_h=[31,31]  ci_peak_gap=[1.538, 1.562] bits
```

---

## Benchmark suite

```python
from fawp_index import run_benchmarks

suite = run_benchmarks()
suite.verify_all()
suite.to_html("benchmarks.html")
```

| Case | Expected | Description |
|------|:--------:|-------------|
| `clean_control` | ✅ FAWP | Textbook: steering collapses, prediction survives |
| `prediction_only` | ❌ None | Predictable system, no steering channel |
| `control_only` | ❌ None | Active controller, no predictive horizon |
| `noisy_false_positive` | ❌ None | Noisy stable — designed to trap detectors |
| `delayed_collapse` | ✅ FAWP | Fast-collapsing unstable system, narrow ODW |

---

## Dashboard

```bash
pip install "fawp-index[dashboard]"
fawp-dashboard                     # opens on http://localhost:8501

# Or from repo:
cd dashboard && streamlit run app.py
```

**Dashboard features (v1.1.5):**
- Severity pills (FAWP / HIGH / WATCH / CLEAR) with pulsing indicators
- Sparkline score trend per asset with ▲/▼ arrows
- ODW proportional bar showing window position in τ range
- Filter bar (All / FAWP only / Watching / Rising)
- Inline "Why flagged?" explain cards
- Mini leaderboard (Top FAWP · Rising Risk · Collapsing Control · Strongest ODW)
- Scan metadata: timestamp, duration, ε, window, τmax

---

## CLI

```bash
# Detect FAWP in a CSV:
fawp-index detect   data.csv --state price --action trade_size --plot

# Rolling market scan:
fawp-index market   SPY.csv --close Close --volume Volume --out report.html

# Scan a watchlist:
fawp-index watchlist spy.csv qqq.csv gld.csv --labels SPY QQQ GLD --out wl.html

# Scan with leaderboard + explain:
fawp-scan --preset equities --leaderboard --explain

# Saved watchlists:
fawp-watchlist create tech AAPL MSFT NVDA AMD
fawp-watchlist scan tech --rank-by gap --out tech.html

# Significance test:
fawp-index significance data.csv --state price --action trade

# Benchmark suite:
fawp-index benchmarks --verify

# Version:
fawp-index version
```

---

## DataFrame API

```python
from fawp_index import fawp_from_dataframe, fawp_rolling

# Single detection:
result = fawp_from_dataframe(df, pred_col="factor", action_col="trade", future_col="fwd_return")

# Rolling — adds fawp_pred_mi, fawp_steer_mi, fawp_gap, fawp_in_regime columns:
df_annotated = fawp_rolling(df, pred_col="returns", action_col="volume")
df_annotated[df_annotated["fawp_in_regime"]]
```

---

## Calibration constants

All calibration anchors are in `fawp_index.constants`, derived directly from the published papers:

```python
from fawp_index.constants import (
    EPSILON_STEERING_RAW,      # 0.01 bits — raw steering threshold (E8/E9 standard)
    EPSILON_STEERING_CORRECTED,# 1e-4 bits — post-null-correction threshold
    BETA_NULL_QUANTILE,        # 0.99 — conservative null floor quantile
    PERSISTENCE_WINDOW_M,      # 5 — Sm(τ) window width
    PERSISTENCE_RULE_M,        # 3 — m-of-n gate (E9.1 confirmed)
    PERSISTENCE_RULE_N,        # 4
    FLAGSHIP_A,                # 1.02 — E8/E9 canonical unstable regime
    FLAGSHIP_K,                # 0.8
    TAU_PLUS_H_E9,             # 31 — E9.2 agency horizon
    TAU_F_E9,                  # 36 — E9.2 functional cliff
    PEAK_GAP_BITS_E9,          # 1.55 bits — E9.2 peak leverage gap
)
```

---

## Citation

```bibtex
@software{clayton2026fawpindex,
  author  = {Ralph Clayton},
  title   = {fawp-index: Information-Control Exclusion Principle detector},
  year    = {2026},
  version = {1.1.5},
  url     = {https://github.com/DrRalphClayton/fawp-index},
  doi     = {10.5281/zenodo.18673949}
}

@article{clayton2026agency,
  author = {Ralph Clayton},
  title  = {Forecasting Without Power: Agency Horizons and the Leverage Gap},
  year   = {2026},
  doi    = {10.5281/zenodo.18663547}
}
```

---

## Links

- 🌐 **Live demo:** [fawp-scanner.info](https://fawp-scanner.info)
- 📦 **PyPI:** [pypi.org/project/fawp-index](https://pypi.org/project/fawp-index/)
- 📂 **GitHub:** [github.com/DrRalphClayton/fawp-index](https://github.com/DrRalphClayton/fawp-index)
- 📄 **Paper (E1–E7):** [doi:10.5281/zenodo.18663547](https://doi.org/10.5281/zenodo.18663547)
- 📄 **Paper (E8):** [doi:10.5281/zenodo.18673949](https://doi.org/10.5281/zenodo.18673949)
- 📗 **Book:** [*Forecasting Without Power*](https://www.amazon.com/dp/B0GS1ZVNM7/) — Ralph Clayton (2026)
- 📊 **Docs:** [`docs/`](docs/) — quickstart, examples, market, watchlist, alerts, significance, benchmarks, dashboard

---

## Module docs

| File | Contents |
|------|---------|
| [`docs/quickstart.md`](docs/quickstart.md) | Getting started guide |
| [`docs/examples.md`](docs/examples.md) | Example gallery — common use cases |
| [`docs/market.md`](docs/market.md) | Market scanner API |
| [`docs/watchlist.md`](docs/watchlist.md) | Watchlist scanner + ranking |
| [`docs/alerts.md`](docs/alerts.md) | Alert engine + channel setup |
| [`docs/significance.md`](docs/significance.md) | Significance testing |
| [`docs/benchmarks.md`](docs/benchmarks.md) | Benchmark cases + verification |
| [`docs/dashboard.md`](docs/dashboard.md) | Dashboard install + deploy |

---

*MIT License · Ralph Clayton · 2026*
