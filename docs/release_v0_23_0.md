# fawp-index v1.1.9

**Information-Control Exclusion Principle detector for finance, climate, and ML systems.**

🔴 [Try live → fawp-scanner.info](https://fawp-scanner.info)  
📦 `pip install "fawp-index[dashboard]"`  
📄 [doi:10.5281/zenodo.18673949](https://doi.org/10.5281/zenodo.18673949)

---

## What's new in v1.1.9

### 🔔 In-app notification bell
Every scan now fires an in-app alert:
- 🔴 **FAWP active** — lists flagged tickers with regime info
- ✅ **All clear** — confirms no regimes detected
- Red badge on bell shows unread count
- "Mark all read" clears the queue
- Stacks up to 20 alerts per session

### 🎮 Demo mode — no account needed
New users can explore the full scanner without signing up:
- **"▶ Try demo — no account needed"** button on the login screen
- Loads synthetic data (SPY/QQQ/GLD/BTC/TLT — 600 bars each)
- Full dashboard access: Scanner, Curves, Heatmap, Compare, Validation, History
- Blocked from real ticker scans (yfinance) — shown upgrade prompt
- Green demo banner with dismiss button

### 📊 FAWP vs classic signals — Compare tab
New `fawp_index.compare.compare_signals()` function:
- Computes RSI-14, Realised Vol-21, Momentum-20, MA Slope-20
- Aligns to FAWP scan dates, measures forward-return lift at 1/5/20 bars
- Shows Pearson correlation with FAWP score
- Side-by-side bar chart: 20-bar return and hit rate
- CSV export

### 🐛 Bug fixes
- Restored `compare_fawp()` / `ComparisonResult` — accidentally removed in v0.22.0
- `import fawp_index` now works cleanly with both compare APIs
- All tests passing

---

## Install

```bash
pip install "fawp-index==1.1.9"              # core
pip install "fawp-index[dashboard]==1.1.9"   # + Streamlit dashboard
pip install "fawp-index[fast]==1.1.9"        # + Numba JIT (5-15x faster scans)
pip install "fawp-index[all]==1.1.9"         # everything
```

## Quick start

```python
from fawp_index.watchlist import scan_watchlist

result = scan_watchlist(["SPY", "QQQ", "GLD", "BTC-USD"], period="2y")
for asset in result.rank_by("score")[:3]:
    print(f"{asset.ticker}: score={asset.latest_score:.4f} "
          f"gap={asset.peak_gap_bits:.4f}b "
          f"{'🔴 FAWP' if asset.regime_active else '—'}")
```

## Full changelog

See [CHANGELOG.md](https://github.com/DrRalphClayton/fawp-index/blob/main/CHANGELOG.md)

## Papers

- E1–E7: [doi:10.5281/zenodo.18663547](https://doi.org/10.5281/zenodo.18663547)
- E8: [doi:10.5281/zenodo.18673949](https://doi.org/10.5281/zenodo.18673949)
