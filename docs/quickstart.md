# Quickstart

## Install

```bash
pip install fawp-index
pip install "fawp-index[plot]"       # + charts
pip install "fawp-index[dashboard]"  # + Streamlit dashboard
```

---

## 1. Detect FAWP in any time series

```python
import numpy as np
from fawp_index import ODWDetector

# Your data — any domain (finance, climate, control systems, ML)
n = 2000
pred   = np.random.randn(n)                          # predictor signal
future = pred[20:] + np.random.randn(n - 20) * 0.3  # what it's forecasting
action = np.random.randn(n - 20) * 0.001            # your control/steering signal
obs    = np.random.randn(n - 20) * 0.1              # system response

odw = ODWDetector().detect_from_arrays(
    pred[:n-20], future, action, obs,
    tau_grid=list(range(1, 41)),
)

print(f"FAWP detected : {odw.fawp_found}")
print(f"Agency horizon: τ_h = {odw.tau_h_plus}")
print(f"ODW           : τ {odw.odw_start}–{odw.odw_end}")
print(f"Peak gap      : {odw.peak_gap_bits:.4f} bits")
```

---

## 2. Scan a price series for FAWP regimes

```python
import pandas as pd
from fawp_index.market import scan_fawp_market

df   = pd.read_csv("prices.csv", parse_dates=["Date"], index_col="Date")
scan = scan_fawp_market(df, ticker="SPY", close_col="Close", volume_col="Volume")

print(scan.summary())
# FAWP Market Scan — SPY
# Windows scanned : 49
# FAWP windows    : 22  (44.9%)
# Latest score    : 0.0312
# Peak gap        : 0.0471 bits  at 2021-06-14
# Peak ODW        : τ 31–33

scan.plot(prices=df["Close"])
scan.to_html("spy_fawp.html")
```

---

## 3. Scan a whole watchlist

```python
from fawp_index.watchlist import scan_watchlist

result = scan_watchlist(
    {"SPY": spy_df, "QQQ": qqq_df, "BTC": btc_df, "GLD": gld_df},
    timeframes=["1d", "1wk"],
)

print(result.summary())

result.rank_by("score")        # strongest FAWP right now
result.rank_by("gap")          # widest leverage gap
result.rank_by("persistence")  # longest active regime
result.rank_by("freshness")    # most recent new signal
result.active_regimes()        # only currently flagged

result.to_html("watchlist.html")
```

With yfinance auto-fetch:

```python
result = scan_watchlist(
    ["SPY", "QQQ", "GLD", "BTC-USD", "ETH-USD"],
    period="2y",
)
```

---

## 4. Set up alerts

```python
from fawp_index.alerts import AlertEngine

engine = AlertEngine(gap_threshold=0.05, state_path="fawp_state.json")
engine.add_terminal()
engine.add_telegram(token="BOT_TOKEN", chat_id="CHAT_ID")
engine.add_discord(webhook_url="https://discord.com/api/webhooks/...")

alerts = engine.check(result)       # fires only on regime transitions
engine.daily_summary(result)        # condensed daily digest
```

---

## 5. Quick scan with fawp-scan

```bash
# Scan a single ticker (requires yfinance):
fawp-scan BTC-USD
fawp-scan SPY --period 2y --timeframe 1d

# Scan a sector ETF watchlist:
fawp-scan XLK XLF XLE XLV --out sector_scan.html

# Scan crypto pairs:
fawp-scan BTC-USD ETH-USD SOL-USD --out crypto_scan.html
```

---

## 6. Command-line tools

```bash
fawp-index detect   data.csv --state price --action volume --plot
fawp-index market   SPY.csv  --close Close --out spy_fawp.html
fawp-index watchlist spy.csv qqq.csv --labels SPY QQQ --out wl.html
fawp-index significance data.csv --state price --action volume
fawp-index benchmarks --verify
fawp-dashboard                       # launch Streamlit dashboard
```

---

## 7. Use the bundled E9.2 data

```python
from fawp_index import ODWDetector

# Reproduce the published E9.2 detection exactly:
odw = ODWDetector.from_e9_2_data()

print(f"τ_h  = {odw.tau_h_plus}")     # 31
print(f"τ_f  = {odw.tau_f}")          # 36
print(f"ODW  = {odw.odw_start}–{odw.odw_end}")  # 31–33
print(f"gap  = {odw.peak_gap_bits:.4f} bits")   # 1.5489
```

---

## 8. Test the detector on known cases

```python
from fawp_index import run_benchmarks

suite = run_benchmarks()
print(suite.summary())
suite.verify_all()   # all 5 canonical cases must pass
```

---

## Next steps

- [`docs/market.md`](market.md) — market scanner parameters
- [`docs/watchlist.md`](watchlist.md) — watchlist scanner + ranking
- [`docs/alerts.md`](alerts.md) — alert channels + scheduling
- [`docs/significance.md`](significance.md) — significance testing
- [`docs/benchmarks.md`](benchmarks.md) — benchmark cases
- [`docs/dashboard.md`](dashboard.md) — Streamlit dashboard
