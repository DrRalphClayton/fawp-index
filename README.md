# fawp-index

[![PyPI version](https://badge.fury.io/py/fawp-index.svg)](https://badge.fury.io/py/fawp-index)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18673949.svg)](https://doi.org/10.5281/zenodo.18673949)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

---

## Your model still predicts. But you've already lost control.

**fawp-index** detects the moment when a system crosses into the
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

## Install

```bash
pip install fawp-index                   # core
pip install "fawp-index[plot]"           # + matplotlib figures
pip install "fawp-index[dashboard]"      # + Streamlit dashboard
pip install "fawp-index[all]"            # everything
```

---

## What's in the package

| Module | What it does |
|--------|-------------|
| `fawp_index` | Core FAWP detector, ODW detection, DataFrame API |
| `fawp_index.market` | Rolling FAWP scan on price / volume data |
| `fawp_index.watchlist` | Multi-asset, multi-timeframe watchlist scanner |
| `fawp_index.alerts` | Telegram / Discord / email / webhook alert engine |
| `fawp_index.significance` | Bootstrap significance testing |
| `fawp_index.compare` | Side-by-side comparison of two detections |
| `fawp_index.benchmarks` | Five canonical ground-truth benchmark cases |
| `fawp_index.report` | PDF report generator |
| `dashboard/app.py` | Streamlit visual dashboard |
| `fawp-index` CLI | Six subcommands (detect / market / watchlist / significance / benchmarks / version) |

---

## 60-second quickstart

```python
import numpy as np
from fawp_index import FAWPAlphaIndex

pred   = np.random.randn(5000)
future = pred[20:] + np.random.randn(4980) * 0.3
action = np.random.randn(4980) * 0.001   # near-zero = FAWP
obs    = np.random.randn(4980) * 0.1

result = FAWPAlphaIndex().compute(pred[:4980], future, action, obs)
print(result.summary())
result.plot()   # pip install "fawp-index[plot]"
```

---

## Market scanner

Scan a price CSV for FAWP regimes across a rolling window:

```python
import pandas as pd
from fawp_index.market import scan_fawp_market

df   = pd.read_csv("SPY.csv", parse_dates=["Date"], index_col="Date")
scan = scan_fawp_market(df, ticker="SPY", close_col="Close", volume_col="Volume")

print(scan.summary())
scan.plot(prices=df["Close"])
scan.to_html("spy_fawp.html")   # self-contained HTML report
scan.to_csv("spy_fawp.csv")
```

**Financial interpretation:**
- **pred channel** `I(returnₜ ; returnₜ₊Δ)` — is the market still forecastable?
- **steer channel** `I(signed_flowₜ ; returnₜ₊τ)` — do your orders still move price?
- **FAWP window** — yes you can forecast direction, but your orders no longer move price

Works without volume (falls back to lagged-return autocorrelation).

---

## Watchlist scanner

Scan a whole watchlist and rank every asset by FAWP signal:

```python
from fawp_index.watchlist import scan_watchlist

# With your own DataFrames:
result = scan_watchlist({"SPY": spy_df, "QQQ": qqq_df, "GLD": gld_df, "BTC": btc_df})

# Or fetch automatically with yfinance:
result = scan_watchlist(
    ["SPY", "QQQ", "GLD", "BTC-USD", "ETH-USD"],
    period     = "2y",
    timeframes = ["1d", "1wk"],
)

result.rank_by("score")        # strongest current regime score
result.rank_by("gap")          # widest leverage gap (bits)
result.rank_by("persistence")  # longest active regime
result.rank_by("freshness")    # most recent new signal
result.active_regimes()        # only currently flagged assets
result.top_n(5, "score")

result.to_html("watchlist.html")
result.to_csv("watchlist.csv")
result.to_json("watchlist.json")
```

---

## Alerts

Fire alerts the moment a regime starts, ends, or crosses a threshold:

```python
from fawp_index.alerts import AlertEngine

engine = AlertEngine(gap_threshold=0.05, state_path="fawp_state.json")
engine.add_terminal()
engine.add_telegram(token="BOT_TOKEN", chat_id="CHAT_ID")
engine.add_discord(webhook_url="https://discord.com/api/webhooks/...")
engine.add_email(smtp_host="smtp.gmail.com", username="you@gmail.com",
                 password="app_password", to_addrs=["you@gmail.com"])
engine.add_webhook("https://hooks.slack.com/services/...")

result = scan_watchlist(dfs)
alerts = engine.check(result)       # NEW_FAWP / REGIME_END / GAP_THRESHOLD
engine.daily_summary(result)        # condensed digest
```

**State-aware:** `NEW_FAWP` fires once when a regime is first detected, not on
every subsequent scan. Set `state_path` to persist state across runs.

---

## Significance testing

Test whether a detected regime is statistically significant:

```python
from fawp_index import ODWDetector, fawp_significance

odw = ODWDetector.from_e9_2_data()
sig = fawp_significance(odw, n_bootstrap=200)

print(sig.summary())
# p_fawp=1.000  p_null=0.145  significant=YES
# ci_tau_h=[31,31]  ci_peak_gap=[1.538, 1.562] bits

sig.to_html("significance.html")
sig.plot()
```

---

## Benchmark suite

Five canonical ground-truth cases that verify the detector is working:

```python
from fawp_index import run_benchmarks

suite = run_benchmarks()
print(suite.summary())
suite.verify_all()              # raises if any case fails
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

Visual five-tab tool — Scanner, Curves, Heatmap, Significance, Export:

```bash
pip install "fawp-index[dashboard]"
fawp-dashboard                     # opens on http://localhost:8501

# Or run directly from the repo:
cd dashboard && streamlit run app.py
```

---

## Command line

```bash
# Detect FAWP in a CSV:
fawp-index detect   data.csv --state price --action trade_size --plot

# Rolling market scan:
fawp-index market   SPY.csv --close Close --volume Volume --out report.html

# Scan a whole watchlist:
fawp-index watchlist spy.csv qqq.csv gld.csv --labels SPY QQQ GLD --out wl.html
fawp-index watchlist *.csv --rank-by gap --top 5

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

## Sklearn compatible

```python
from fawp_index.sklearn_api import FAWPTransformer
from sklearn.pipeline import Pipeline

pipe = Pipeline([("fawp", FAWPTransformer(pred_col=0, action_col=1, delta=20))])
pipe.fit(X)
```

---

## Reproduce published figures

The E8 experimental data is bundled with the package:

```python
from fawp_index import ODWDetector
odw = ODWDetector.from_e9_2_data()
odw.plot()
```

To reproduce full figure sets, clone the repo and run from there:

```bash
git clone https://github.com/DrRalphClayton/fawp-index
cd fawp-index
python examples/reproduce_e8.py --save
python examples/reproduce_e1_e7.py --save
```

---

## The mathematics

The **FAWP Alpha Index v2.1**:

```
α₂(τ) = I[τ≥1] · g(τ) · (Sₘ(τ) − Ĩ_steer(τ)) · (1 + κ · R_log(τ))
```

- `g(τ)` — gate: fires when pred MI > η AND steer MI ≤ ε
- `Sₘ(τ)` — windowed-min corrected predictive MI (persistence)
- `Ĩ_steer(τ)` — null-corrected steering MI
- `R_log(τ)` — log-slope resonance amplifier near the horizon

The **agency horizon τ_h** is where steering MI first falls below ε.
Near τ_h, predictive MI does not fall — it surges. This is the empirical
signature of the Information-Control Exclusion Principle.

---

## Citation

```bibtex
@software{clayton2026fawpindex,
  author  = {Ralph Clayton},
  title   = {fawp-index: Information-Control Exclusion Principle detector},
  year    = {2026},
  version = {0.10.0},
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

- 📦 **PyPI:** [pypi.org/project/fawp-index](https://pypi.org/project/fawp-index/)
- 📂 **GitHub:** [github.com/DrRalphClayton/fawp-index](https://github.com/DrRalphClayton/fawp-index)
- 📄 **Paper (E1–E7):** [doi:10.5281/zenodo.18663547](https://doi.org/10.5281/zenodo.18663547)
- 📄 **Paper (E8):** [doi:10.5281/zenodo.18673949](https://doi.org/10.5281/zenodo.18673949)
- 📗 **Book:** [*Forecasting Without Power*](https://www.amazon.com/dp/B0GS1ZVNM7/) — Ralph Clayton (2026)
- 📊 **Docs:** [`docs/`](docs/) — market, watchlist, alerts, significance, benchmarks, dashboard

---

## Module docs

| File | Contents |
|------|---------|
| [`docs/market.md`](docs/market.md) | Market scanner API + parameters |
| [`docs/watchlist.md`](docs/watchlist.md) | Watchlist scanner + ranking |
| [`docs/alerts.md`](docs/alerts.md) | Alert engine + channel setup |
| [`docs/significance.md`](docs/significance.md) | Significance testing methods |
| [`docs/benchmarks.md`](docs/benchmarks.md) | Benchmark cases + verification |
| [`docs/dashboard.md`](docs/dashboard.md) | Dashboard install + deploy |

---

*MIT License · Ralph Clayton · 2026*
