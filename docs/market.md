# fawp_index.market — Rolling FAWP Market Scanner

Scan a price (and optionally volume) DataFrame for FAWP regimes — windows
where directional forecastability persists while market-impact effectiveness
has already collapsed.

## Financial interpretation

| Channel | Signal | Meaning |
|---------|--------|---------|
| **pred** | `I(returnₜ ; returnₜ₊Δ)` | Is the market still forecastable? |
| **steer** | `I(signed_flowₜ ; returnₜ₊τ)` | Do your orders still move price? |

A **FAWP window** means you can still forecast direction, but your orders
no longer move price the way they used to.

## Quick start

```python
import pandas as pd
from fawp_index.market import scan_fawp_market

df = pd.read_csv("SPY.csv", parse_dates=["Date"], index_col="Date")

# Fast scan (no null correction):
scan = scan_fawp_market(df, ticker="SPY", close_col="Close", volume_col="Volume")

# Rigorous (shuffle + shift null at every tau):
scan = scan_fawp_market(df, ticker="SPY", n_null=50)

print(scan.summary())
scan.plot(prices=df["Close"])
scan.to_html("spy_fawp.html")
scan.to_csv("spy_fawp.csv")
scan.to_json("spy_fawp.json")
```

## CLI

```bash
fawp-index market SPY.csv --close Close --volume Volume --out report.html
fawp-index market BTC.csv --window 180 --tau-max 30 --out btc_fawp.json
```

## Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window` | 252 | Rolling window in bars (≈1 year of daily data) |
| `step` | 5 | Bars to advance between scans (weekly cadence) |
| `delta_pred` | 20 | Forecast horizon Δ (≈1 month) |
| `tau_max` | 40 | Maximum steering lag to test |
| `epsilon` | 0.01 | MI threshold for FAWP condition |
| `n_null` | 0 | Null permutations. `0` = fast; `50+` = rigorous |
| `beta_null` | 0.99 | Conservative null quantile |

## Works without volume

```python
# Steer channel falls back to lagged-return autocorrelation
scan = scan_fawp_market(df, close_col="Close", volume_col=None)
```

## Custom signal columns

```python
# Supply your own predictor and steering signals
scan = scan_fawp_market(
    df,
    pred_col  = "my_forecast_signal",
    steer_col = "my_impact_signal",
)
```

## MarketScanSeries properties

| Property | Description |
|----------|-------------|
| `.windows` | List of `MarketWindowResult` (one per scan step) |
| `.dates` | DatetimeIndex of window end dates |
| `.regime_scores` | Continuous FAWP signal 0–1 per window |
| `.fawp_flags` | Boolean array |
| `.fawp_fraction` | Fraction of windows flagged |
| `.latest` | Most recent window |
| `.peak` | Window with highest regime score |
| `.fawp_windows` | All flagged windows |

## Exports

| Method | Output |
|--------|--------|
| `.to_html(path)` | Self-contained HTML report with embedded chart |
| `.to_csv(path)` | Tidy per-window DataFrame as CSV |
| `.to_json(path)` | Full result with metadata as JSON |
| `.plot(prices=...)` | Two-panel matplotlib figure |
