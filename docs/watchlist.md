# fawp_index.watchlist — Multi-asset Watchlist Scanner

Scan a whole watchlist across multiple timeframes and rank results by any
FAWP signal metric.

## Quick start

```python
from fawp_index.watchlist import scan_watchlist

# With your own DataFrames:
dfs = {"SPY": spy_df, "QQQ": qqq_df, "GLD": gld_df, "BTC": btc_df}
result = scan_watchlist(dfs)

# With yfinance auto-fetch:
result = scan_watchlist(
    ["SPY", "QQQ", "GLD", "BTC-USD", "ETH-USD"],
    period="2y",
)

# Multi-timeframe:
result = scan_watchlist(dfs, timeframes=["1d", "1wk"])

print(result.summary())
result.to_html("watchlist.html")
```

## CLI

```bash
fawp-index watchlist spy.csv qqq.csv gld.csv --labels SPY QQQ GLD --out watchlist.html
fawp-index watchlist *.csv --rank-by gap --top 5
```

## Ranking metrics

```python
result.rank_by("score")        # strongest current regime score
result.rank_by("gap")          # widest peak leverage gap (bits)
result.rank_by("persistence")  # longest days in active regime
result.rank_by("freshness")    # most recent new FAWP signal
result.rank_by("entry")        # earliest regime start

result.top_n(5, "score")       # top 5 by score
result.active_regimes()        # only currently flagged assets
```

## WatchlistResult properties

| Property | Description |
|----------|-------------|
| `.assets` | List of `AssetResult` |
| `.n_assets` | Total assets scanned |
| `.n_flagged` | Currently active FAWP regimes |
| `.scanned_at` | Scan timestamp |

## AssetResult fields

| Field | Description |
|-------|-------------|
| `ticker` | Asset label |
| `timeframe` | e.g. `"1d"`, `"1wk"` |
| `latest_score` | Current regime score (0–1) |
| `peak_gap_bits` | Peak leverage gap in bits |
| `regime_active` | Currently in FAWP? |
| `regime_start` | Start date of current regime |
| `days_in_regime` | Calendar days in current regime |
| `signal_age_days` | Days since last FAWP signal |
| `scan` | Full `MarketScanSeries` for this asset |

## Exports

```python
result.to_html("watchlist.html")   # HTML table with score bars
result.to_csv("watchlist.csv")     # Tidy DataFrame
result.to_json("watchlist.json")   # Full JSON with metadata
result.to_dataframe()              # pandas DataFrame
```
