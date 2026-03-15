# Examples — fawp-index

Common use cases with copy-paste code. All runnable after `pip install "fawp-index[all]"`.

---

## 1. Detect FAWP on any time series (5 lines)

The minimal case — no real data needed:

```python
import numpy as np
from fawp_index import FAWPAlphaIndex

# pred = your monitoring stream (e.g. current price/state)
# future = what you're trying to forecast (pre-aligned)
# action = what you do (trade size, control input)
# obs = delayed consequence of your action
pred   = np.random.randn(5000)
future = pred[20:] + np.random.randn(4980) * 0.3
action = np.random.randn(4980) * 0.001   # near-zero → no steering
obs    = np.random.randn(4980) * 0.1

result = FAWPAlphaIndex().compute(pred[:4980], future, action, obs)
print(result.summary())
```

---

## 2. Detect FAWP on BTC daily data

```python
import yfinance as yf
from fawp_index.market import scan_fawp_market

df = yf.download("BTC-USD", period="2y", auto_adjust=True)
df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

scan = scan_fawp_market(df, ticker="BTC-USD", close_col="Close", volume_col="Volume")

print(scan.summary())
scan.plot(prices=df["Close"])
scan.to_html("btc_fawp.html")
```

---

## 3. Scan SPY + QQQ + GLD and rank by FAWP signal

```python
from fawp_index.watchlist import scan_watchlist

result = scan_watchlist(
    ["SPY", "QQQ", "GLD", "TLT", "BTC-USD"],
    period     = "2y",
    timeframes = ["1d", "1wk"],
)

# Ranked tables
print(result.summary())
print("\nTop 5 by score:")
for a in result.rank_by("score")[:5]:
    print(f"  {a.ticker} [{a.timeframe}]  score={a.latest_score:.4f}  gap={a.peak_gap_bits:.4f}b")

# Export
result.to_html("watchlist.html")
result.to_csv("watchlist.csv")
```

---

## 4. Alert when a regime starts on your watchlist

```python
from fawp_index.watchlist import scan_watchlist
from fawp_index.alerts import AlertEngine, AlertSeverity

engine = AlertEngine(
    gap_threshold=0.05,
    cooldown_hours=6,
    min_consecutive_windows=2,
    min_severity=AlertSeverity.MEDIUM,
    state_path="fawp_state.json",   # persists across runs
)
engine.add_terminal()
engine.add_telegram(token="YOUR_TOKEN", chat_id="YOUR_CHAT_ID")

# Run this on a schedule (cron, GitHub Actions, etc.)
result = scan_watchlist(["SPY", "QQQ", "GLD", "BTC-USD"], period="2y")
alerts = engine.check(result)
engine.daily_summary(result)

print(f"Fired {len(alerts)} alert(s)")
```

---

## 5. Explain why an asset is flagged

```python
from fawp_index.watchlist import scan_watchlist
from fawp_index.explain import explain_asset
from fawp_index.leaderboard import Leaderboard

result = scan_watchlist(["SPY", "QQQ", "GLD"], period="2y")

# Explain top asset
top = result.rank_by("score")[0]
print(explain_asset(top))

# Full leaderboard
lb = Leaderboard.from_watchlist(result)
print(lb.summary())
lb.to_html("leaderboard.html")
```

---

## 6. Reproduce the E9 flagship result

```python
from fawp_index import ODWDetector, fawp_significance
from fawp_index.constants import TAU_PLUS_H_E9, TAU_F_E9, PEAK_GAP_BITS_E9

odw = ODWDetector.from_e9_2_data(steering="u")
print(odw.summary())

# Should print:
# tau_h_plus = 31
# tau_f      = 36
# ODW        = 31–33
# peak gap   = ~1.55 bits

assert odw.tau_h_plus == TAU_PLUS_H_E9
assert odw.tau_f      == TAU_F_E9

sig = fawp_significance(odw, n_bootstrap=200)
print(sig.summary())
# p_fawp=1.000  significant=YES
```

---

## 7. Use the benchmarks as ground truth

```python
from fawp_index import run_benchmarks, clean_control, delayed_collapse

# Run all five and verify
suite = run_benchmarks()
suite.verify_all()
suite.to_html("benchmarks.html")

# Inspect a single case
r = clean_control()
print(r.summary())
r.plot()   # pip install "fawp-index[plot]"

# Delayed collapse — narrow ODW
r2 = delayed_collapse()
print(f"ODW: {r2.odw_result.odw_start}–{r2.odw_result.odw_end}")
```

---

## 8. DataFrame rolling API (pandas integration)

```python
import pandas as pd
from fawp_index import fawp_rolling

df = pd.read_csv("my_data.csv", parse_dates=["date"], index_col="date")

# Adds fawp_pred_mi, fawp_steer_mi, fawp_gap, fawp_in_regime columns
df_out = fawp_rolling(df, pred_col="returns", action_col="volume")

# Filter to FAWP windows only
fawp_windows = df_out[df_out["fawp_in_regime"]]
print(fawp_windows[["fawp_pred_mi", "fawp_steer_mi", "fawp_gap"]].describe())
```

---

## 9. Saved watchlists (persist between runs)

```python
from fawp_index.watchlist_store import WatchlistStore

store = WatchlistStore()

# Create once
store.create("equities", ["SPY", "QQQ", "IWM", "DIA"])
store.create("crypto",   ["BTC-USD", "ETH-USD", "SOL-USD"], period="1y")

# Scan anytime
result = store.scan("equities")
print(result.summary())

# List all saved watchlists
print(store.summary())
```

Or via CLI:

```bash
fawp-watchlist create equities SPY QQQ IWM DIA
fawp-watchlist scan equities --rank-by gap --leaderboard --out equities.html
fawp-watchlist list
```

---

## 10. sklearn pipeline

```python
from fawp_index.sklearn_api import FAWPTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("fawp",   FAWPTransformer(pred_col=0, action_col=1, delta=20)),
])

pipe.fit(X_train)
X_annotated = pipe.transform(X_test)
```

---

## See also

- [`docs/quickstart.md`](quickstart.md) — getting started
- [`docs/market.md`](market.md) — market scanner parameters
- [`docs/watchlist.md`](watchlist.md) — watchlist scanner + ranking
- [`docs/alerts.md`](alerts.md) — alert engine setup
- [`docs/benchmarks.md`](benchmarks.md) — benchmark cases
- [`docs/dashboard.md`](dashboard.md) — Streamlit dashboard

---

## 11. Per-tau attribution — which lags drive the gap?

```python
from fawp_index.watchlist import scan_watchlist
from fawp_index.explain import attribute_gap, attribute_windows, attribution_report

result = scan_watchlist(["SPY", "QQQ"], period="2y")
top = result.rank_by("score")[0]

# Per-tau: which specific lag values are driving the divergence?
attr = attribute_gap(top.scan.latest, top_n=5)
print(f"Peak gap at τ={attr['peak_tau']}")
print(f"ODW captures {attr['odw_share']*100:.1f}% of total gap")
for tau, share in zip(attr['top_tau'], attr['top_shares']):
    print(f"  τ={tau:>3}  share={share*100:.1f}%")

# Per-window: when was the regime most active?
attr_w = attribute_windows(top)
print(f"\nFAWP in {attr_w['n_fawp_windows']}/{attr_w['n_total_windows']} windows")
print(f"Onset: {attr_w['onset_date']}  Peak: {attr_w['peak_date']}")
print(f"Score trend: {attr_w['score_slope']:+.4f}/window")

# Full combined report
print(attribution_report(top))
```

Output:
```
Peak gap at τ=9
ODW captures 78.3% of total gap
  τ=  9  share=24.1%
  τ= 10  share=19.8%
  τ=  8  share=15.2%
  τ= 11  share=12.7%
  τ=  7  share=9.4%

FAWP in 12/48 windows
Onset: 2026-03-02  Peak: 2026-03-08
Score trend: +0.0012/window
```
