# FAWP Dashboard — Streamlit App

A five-tab visual tool for interactive FAWP analysis.

## Installation

```bash
pip install 'fawp-index[dashboard]'
```

Or install dependencies manually:
```bash
pip install fawp-index streamlit matplotlib yfinance
```

## Launch

```bash
# Via entry point (after pip install):
fawp-dashboard

# Custom port:
fawp-dashboard --port 8502

# Via Streamlit directly (from repo):
cd dashboard
streamlit run app.py
```

## Tabs

### 🔍 Scanner
- KPI cards: assets scanned, FAWP active count, highest score
- Ranked table with color-coded FAWP flags
- Inline threshold alerts

### 📈 Curves
- Per-asset regime score time series
- MI curves (pred vs steer) for any selected window
- Leverage gap bar chart (τ-wise)
- Window statistics JSON panel

### 🟥 Heatmap
- Regime score heatmap (assets × timeframes)
- Peak leverage gap heatmap

### 🔬 Significance
- Bootstrap significance test for any selected asset
- p-value display + bootstrap distribution plots

### 💾 Export
- Download watchlist CSV / JSON / HTML
- Per-asset scan CSVs

## Data sources

| Source | How |
|--------|-----|
| Demo data | Built-in (5 synthetic assets, 600 bars) |
| Upload CSV | One file per asset (Close + optional Volume) |
| yfinance | Enter ticker symbols + period |

## Deploy to Streamlit Cloud

1. Push the repo to GitHub (dashboard/ is already included)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Point to `dashboard/app.py`
4. Use `dashboard/requirements.txt`
