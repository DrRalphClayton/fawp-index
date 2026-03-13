"""
FAWP Dashboard — Streamlit app
================================

Run locally::

    pip install fawp-index[plot] streamlit
    cd dashboard
    streamlit run app.py

Deploy to Streamlit Cloud::

    # Push dashboard/ to your GitHub repo, then connect on share.streamlit.io

Ralph Clayton (2026) · https://doi.org/10.5281/zenodo.18673949
"""

import io
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title   = "FAWP Dashboard",
    page_icon    = "📡",
    layout       = "wide",
    initial_sidebar_state = "expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card {
    background: #0E2550; color: white; border-radius: 10px;
    padding: 1.2em 1.4em; text-align: center; margin-bottom: 0.5em;
  }
  .metric-card .val { font-size: 2em; font-weight: 700; }
  .metric-card .lbl { font-size: 0.85em; color: #aac; margin-top: 0.2em; }
  .fawp-red  { color: #C0111A; font-weight: 700; }
  .fawp-green{ color: #1a7a1a; font-weight: 700; }
  .section-header {
    font-size: 1.1em; font-weight: 700; color: #0E2550;
    border-bottom: 2px solid #D4AF37; padding-bottom: 4px;
    margin: 1.2em 0 0.6em;
  }
</style>
""", unsafe_allow_html=True)

# ── Imports (deferred so Streamlit can show errors nicely) ────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from fawp_index.market   import scan_fawp_market, MarketScanSeries
    from fawp_index.watchlist import WatchlistScanner, scan_watchlist
    from fawp_index.alerts   import AlertEngine
    from fawp_index.significance import fawp_significance
    HAS_FAWP = True
except ImportError as e:
    st.error(f"fawp-index not installed: {e}\n\n`pip install fawp-index[plot]`")
    st.stop()


# ═════════════════════════════════════════════════════════════════════════════
# Sidebar — configuration
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://img.shields.io/badge/fawp--index-v0.10.0-0E2550?style=flat-square",
             use_container_width=False)
    st.title("⚙️ Configuration")

    st.markdown("#### Data source")
    source = st.radio("", ["Upload CSV(s)", "Enter tickers (yfinance)", "Demo data"],
                      index=2, label_visibility="collapsed")

    st.markdown("#### Scanner settings")
    window    = st.slider("Rolling window (bars)",  60, 504, 252, step=10)
    step      = st.slider("Scan step (bars)",         1,  20,   5, step=1)
    tau_max   = st.slider("Max tau",                  5,  80,  40, step=5)
    n_null    = st.slider("Null permutations (0=fast)", 0, 200, 0, step=10)
    epsilon   = st.number_input("Epsilon (bits)", min_value=0.001, max_value=0.1,
                                value=0.01, step=0.001, format="%.3f")

    st.markdown("#### Timeframes")
    tfs_1d  = st.checkbox("Daily (1d)",   value=True)
    tfs_1wk = st.checkbox("Weekly (1wk)", value=False)
    timeframes = []
    if tfs_1d:  timeframes.append("1d")
    if tfs_1wk: timeframes.append("1wk")
    if not timeframes:
        timeframes = ["1d"]

    st.markdown("#### Alerts")
    alert_threshold = st.slider("Gap alert threshold (bits)", 0.0, 1.0, 0.05, 0.01)

    st.markdown("---")
    run_btn = st.button("▶ Run Scan", type="primary", use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# Data loading helpers
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading data…")
def _load_demo() -> dict:
    rng = np.random.default_rng(42)

    def _gbm(n, mu=0.0002, sigma=0.012, seed=0):
        r  = np.random.default_rng(seed)
        ret = r.normal(mu, sigma, n)
        p  = 100 * np.exp(np.cumsum(ret))
        return p

    n = 600
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    assets = {
        "SPY":   (0.0003, 0.010, 0),
        "QQQ":   (0.0004, 0.013, 1),
        "GLD":   (0.0001, 0.008, 2),
        "BTC":   (0.0008, 0.040, 3),
        "TLT":   (-0.0002, 0.009, 4),
    }
    dfs = {}
    for ticker, (mu, sigma, seed) in assets.items():
        prices = _gbm(n, mu, sigma, seed)
        vols   = np.random.default_rng(seed + 10).integers(
            500_000, 5_000_000, n
        ).astype(float)
        dfs[ticker] = pd.DataFrame(
            {"Close": prices, "Volume": vols}, index=dates
        )
    return dfs


def _load_uploaded(files) -> dict:
    dfs = {}
    for f in files:
        try:
            df = pd.read_csv(f)
            # Auto-detect date column
            for col in df.columns:
                if "date" in col.lower() or "time" in col.lower():
                    df[col] = pd.to_datetime(df[col])
                    df = df.set_index(col)
                    break
            # Auto-detect close column
            close_candidates = [c for c in df.columns
                                 if "close" in c.lower() or "adj" in c.lower() or "price" in c.lower()]
            if not close_candidates:
                st.warning(f"{f.name}: no Close-like column found — using first numeric column")
                close_candidates = [df.select_dtypes(include=np.number).columns[0]]
            df = df.rename(columns={close_candidates[0]: "Close"})
            ticker = Path(f.name).stem.upper()
            dfs[ticker] = df
        except Exception as e:
            st.error(f"Failed to load {f.name}: {e}")
    return dfs


@st.cache_data(show_spinner="Fetching from yfinance…")
def _load_yfinance(tickers_str: str, period: str) -> dict:
    try:
        import yfinance as yf
    except ImportError:
        st.error("yfinance not installed. `pip install yfinance`")
        return {}
    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
    dfs = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            if df.empty:
                st.warning(f"Empty data for {ticker}")
                continue
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            dfs[ticker] = df
        except Exception as e:
            st.warning(f"Failed to fetch {ticker}: {e}")
    return dfs


# ═════════════════════════════════════════════════════════════════════════════
# Run scan (cached by config hash)
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(
    show_spinner="Running FAWP scanner…",
    hash_funcs={dict: lambda d: str(sorted(d.keys()))},
)
def _run_scan(dfs, window, step, tau_max, n_null, epsilon, timeframes):
    from fawp_index.watchlist import WatchlistScanner
    from fawp_index.market import MarketScanConfig
    cfg = MarketScanConfig(
        window=window, step=step, tau_max=tau_max,
        n_null=n_null, epsilon=epsilon,
    )
    scanner = WatchlistScanner(config=cfg, timeframes=timeframes, verbose=False)
    return scanner.scan(dfs)


# ═════════════════════════════════════════════════════════════════════════════
# Main app
# ═════════════════════════════════════════════════════════════════════════════

st.title("📡 FAWP Dashboard")
st.caption(
    "Forecasting Without Power — "
    "detecting when you can still see where things are going "
    "but can no longer change them."
)

# ── Load data ─────────────────────────────────────────────────────────────────
dfs = {}

if source == "Demo data":
    dfs = _load_demo()
    st.info(f"Demo data: {', '.join(dfs.keys())}  ({len(next(iter(dfs.values())))} bars each)")

elif source == "Upload CSV(s)":
    uploaded = st.file_uploader(
        "Upload CSV files (one per asset). Must have a date column and a Close column.",
        type=["csv"], accept_multiple_files=True
    )
    if uploaded:
        dfs = _load_uploaded(uploaded)

elif source == "Enter tickers (yfinance)":
    col1, col2 = st.columns([3, 1])
    with col1:
        ticker_str = st.text_input(
            "Tickers (comma-separated)", "SPY, QQQ, GLD, BTC-USD"
        )
    with col2:
        period = st.selectbox("Period", ["1y", "2y", "5y", "max"], index=1)
    if st.button("Fetch data"):
        dfs = _load_yfinance(ticker_str, period)

if not dfs:
    st.warning("No data loaded. Choose a source above and press **Run Scan**.")
    st.stop()

# ── Run scan ──────────────────────────────────────────────────────────────────
if run_btn or "wl_result" not in st.session_state:
    with st.spinner("Scanning…"):
        st.session_state["wl_result"] = _run_scan(
            dfs, window, step, tau_max, n_null, epsilon, tuple(timeframes)
        )

wl = st.session_state["wl_result"]
ranked = wl.rank_by("score")


# ═════════════════════════════════════════════════════════════════════════════
# Tabs
# ═════════════════════════════════════════════════════════════════════════════

tab_scanner, tab_curves, tab_heatmap, tab_significance, tab_export = st.tabs([
    "🔍 Scanner", "📈 Curves", "🟥 Heatmap", "🔬 Significance", "💾 Export"
])


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Scanner
# ─────────────────────────────────────────────────────────────────────────────

with tab_scanner:
    # KPI cards
    n_active = wl.n_flagged
    n_total  = wl.n_assets
    peak     = wl.peak if hasattr(wl, "peak") else ranked[0] if ranked else None
    best_score = ranked[0].latest_score if ranked else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="val">{n_total}</div>
            <div class="lbl">Assets scanned</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        color = "#C0111A" if n_active else "#1a7a1a"
        st.markdown(f"""<div class="metric-card" style="background:{color}">
            <div class="val">{n_active}</div>
            <div class="lbl">FAWP active</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="val">{best_score:.4f}</div>
            <div class="lbl">Highest score</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        pct = int(n_active / n_total * 100) if n_total else 0
        st.markdown(f"""<div class="metric-card">
            <div class="val">{pct}%</div>
            <div class="lbl">Flagged</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Ranked by Current Score</div>',
                unsafe_allow_html=True)

    rank_col = st.selectbox(
        "Rank by", ["score", "gap", "persistence", "freshness"],
        key="rank_col"
    )
    sorted_assets = wl.rank_by(rank_col)

    table_rows = []
    for a in sorted_assets:
        if a.error:
            table_rows.append({
                "Ticker": a.ticker, "TF": a.timeframe,
                "Score": "ERROR", "Gap (bits)": "—",
                "FAWP": "❌", "Regime start": "—",
                "Days": 0, "Freshness": "—", "ODW": "—",
            })
            continue
        table_rows.append({
            "Ticker": a.ticker,
            "TF":     a.timeframe,
            "Score":  round(a.latest_score, 4),
            "Gap (bits)": round(a.peak_gap_bits, 4),
            "FAWP":   "🔴 YES" if a.regime_active else "—",
            "Regime start": str(a.regime_start.date()) if a.regime_start else "—",
            "Days in regime": a.days_in_regime,
            "Signal age":    a.signal_age_days,
            "ODW": (f"{a.peak_odw_start}–{a.peak_odw_end}"
                    if a.peak_odw_start is not None else "—"),
        })

    df_table = pd.DataFrame(table_rows)

    def _color_row(row):
        if row.get("FAWP", "") == "🔴 YES":
            return ["background-color: #fff5f5"] * len(row)
        return [""] * len(row)

    st.dataframe(
        df_table.style.apply(_color_row, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    # Alert check
    if alert_threshold > 0:
        st.markdown('<div class="section-header">⚡ Threshold Alerts</div>',
                    unsafe_allow_html=True)
        triggered = [a for a in sorted_assets
                     if a.regime_active and a.peak_gap_bits >= alert_threshold]
        if triggered:
            for a in triggered:
                st.warning(
                    f"**{a.ticker}** [{a.timeframe}] — gap={a.peak_gap_bits:.4f} bits "
                    f"≥ threshold={alert_threshold:.3f}  |  "
                    f"score={a.latest_score:.4f}  |  "
                    f"ODW {a.peak_odw_start}–{a.peak_odw_end}"
                )
        else:
            st.success(
                f"No assets with gap ≥ {alert_threshold:.3f} bits currently active."
            )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Curves
# ─────────────────────────────────────────────────────────────────────────────

with tab_curves:
    valid_assets = [a for a in ranked if not a.error and a.scan is not None]
    if not valid_assets:
        st.warning("No valid scans available.")
    else:
        asset_labels = [f"{a.ticker} ({a.timeframe})" for a in valid_assets]
        sel_label    = st.selectbox("Select asset", asset_labels, key="curve_asset")
        sel_asset    = valid_assets[asset_labels.index(sel_label)]
        scan         = sel_asset.scan

        # Which window to inspect
        n_windows = len(scan.windows)
        win_idx   = st.slider("Window (latest = right)", 0, n_windows - 1,
                              n_windows - 1, key="win_idx")
        win = scan.windows[win_idx]

        col_l, col_r = st.columns(2)

        # ── Regime score time series ─────────────────────────────────────
        with col_l:
            st.markdown(f"**Regime score over time — {sel_asset.ticker}**")
            if HAS_MPL:
                fig, ax = plt.subplots(figsize=(6, 3))
                dates_n  = np.arange(len(scan.dates))
                colors   = ["#C0111A" if f else "#1a7a1a" for f in scan.fawp_flags]
                ax.bar(dates_n, scan.regime_scores, color=colors, width=1.0,
                       alpha=0.75, edgecolor="none")
                ax.axvline(win_idx, color="gold", lw=2, ls="--", label="selected")
                n  = len(scan.dates)
                ticks = np.linspace(0, n - 1, min(6, n), dtype=int)
                ax.set_xticks(ticks)
                ax.set_xticklabels([str(scan.dates[i].date()) for i in ticks],
                                   rotation=25, ha="right", fontsize=7)
                ax.set_ylabel("Regime score", fontsize=8)
                ax.set_ylim(0, max(scan.regime_scores.max() * 1.15, 0.01))
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.2, axis="y")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        # ── MI curves for selected window ────────────────────────────────
        with col_r:
            st.markdown(
                f"**MI curves — {win.date.date()}  "
                f"({'🔴 FAWP' if win.fawp_found else '🟢 none'})**"
            )
            if HAS_MPL:
                fig, ax = plt.subplots(figsize=(6, 3))
                tau = win.tau
                ax.plot(tau, win.pred_mi,  color="darkorange", lw=2, marker="o",
                        ms=3, label="Pred MI (corrected)")
                ax.plot(tau, win.steer_mi, "b--", lw=2, marker="s", ms=3,
                        label="Steer MI (corrected)")
                ax.fill_between(tau, win.steer_mi, win.pred_mi,
                                where=(win.pred_mi > win.steer_mi),
                                alpha=0.15, color="orange", label="Leverage gap")
                # ODW shading
                odw = win.odw_result
                if odw.odw_start is not None:
                    ax.axvspan(odw.odw_start - 0.5, odw.odw_end + 0.5,
                               alpha=0.12, color="#C0111A", zorder=0,
                               label=f"ODW {odw.odw_start}–{odw.odw_end}")
                ax.axhline(epsilon, color="grey", ls=":", lw=0.8, label=f"ε={epsilon}")
                ax.set_xlabel("τ (steering lag)", fontsize=8)
                ax.set_ylabel("MI (bits)", fontsize=8)
                ax.legend(fontsize=7, loc="upper right")
                ax.grid(True, alpha=0.2)
                ax.set_ylim(bottom=0)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        # ── Leverage gap bar: τ-wise ─────────────────────────────────────
        st.markdown("**Leverage gap by τ (selected window)**")
        if HAS_MPL:
            gap = np.maximum(0, win.pred_mi - win.steer_mi)
            fig, ax = plt.subplots(figsize=(10, 2.5))
            bar_colors = [
                "#C0111A" if (win.odw_result.odw_start is not None
                              and win.odw_result.odw_start <= t <= win.odw_result.odw_end)
                else "#888"
                for t in win.tau
            ]
            ax.bar(win.tau, gap, color=bar_colors, alpha=0.8, edgecolor="none")
            ax.set_xlabel("τ", fontsize=8)
            ax.set_ylabel("Gap (bits)", fontsize=8)
            ax.set_title("Red = inside ODW", fontsize=8)
            ax.grid(True, alpha=0.2, axis="y")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        # ── Window stats ─────────────────────────────────────────────────
        with st.expander("Window statistics"):
            odw = win.odw_result
            st.json({
                "date":         str(win.date.date()),
                "fawp_found":   bool(win.fawp_found),
                "regime_score": round(float(win.regime_score), 6),
                "tau_h_plus":   odw.tau_h_plus,
                "tau_f":        odw.tau_f,
                "odw_start":    odw.odw_start,
                "odw_end":      odw.odw_end,
                "peak_gap_bits": round(float(odw.peak_gap_bits), 6),
                "n_obs":        win.n_obs,
            })


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Heatmap
# ─────────────────────────────────────────────────────────────────────────────

with tab_heatmap:
    st.markdown('<div class="section-header">Regime Score Heatmap</div>',
                unsafe_allow_html=True)
    st.caption("Rows = assets, columns = timeframes. Color = latest regime score.")

    if HAS_MPL:
        tickers  = sorted(set(a.ticker    for a in wl.assets if not a.error))
        tfs      = sorted(set(a.timeframe for a in wl.assets if not a.error))
        n_t, n_tf = len(tickers), len(tfs)

        mat = np.full((n_t, n_tf), np.nan)
        for a in wl.assets:
            if a.error: continue
            i = tickers.index(a.ticker)
            j = tfs.index(a.timeframe)
            mat[i, j] = a.latest_score

        fig, ax = plt.subplots(figsize=(max(4, n_tf * 1.8), max(3, n_t * 0.8)))
        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r",
                       vmin=0, vmax=max(0.01, np.nanmax(mat)))
        ax.set_xticks(range(n_tf)); ax.set_xticklabels(tfs, fontsize=9)
        ax.set_yticks(range(n_t));  ax.set_yticklabels(tickers, fontsize=9)
        plt.colorbar(im, ax=ax, label="Regime score")
        for i in range(n_t):
            for j in range(n_tf):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center",
                            fontsize=8, color="white" if mat[i,j] > 0.3 else "black")
        ax.set_title("FAWP Regime Score Heatmap", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # Gap heatmap
    st.markdown('<div class="section-header">Peak Leverage Gap Heatmap (bits)</div>',
                unsafe_allow_html=True)

    if HAS_MPL:
        mat_gap = np.full((n_t, n_tf), np.nan)
        for a in wl.assets:
            if a.error: continue
            mat_gap[tickers.index(a.ticker), tfs.index(a.timeframe)] = a.peak_gap_bits

        fig, ax = plt.subplots(figsize=(max(4, n_tf * 1.8), max(3, n_t * 0.8)))
        im2 = ax.imshow(mat_gap, aspect="auto", cmap="RdYlGn_r",
                        vmin=0, vmax=max(0.01, np.nanmax(mat_gap)))
        ax.set_xticks(range(n_tf)); ax.set_xticklabels(tfs, fontsize=9)
        ax.set_yticks(range(n_t));  ax.set_yticklabels(tickers, fontsize=9)
        plt.colorbar(im2, ax=ax, label="Peak gap (bits)")
        for i in range(n_t):
            for j in range(n_tf):
                if not np.isnan(mat_gap[i, j]):
                    ax.text(j, i, f"{mat_gap[i,j]:.3f}", ha="center", va="center",
                            fontsize=8, color="white" if mat_gap[i,j] > 0.3 else "black")
        ax.set_title("Peak Leverage Gap Heatmap", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 4 — Significance
# ─────────────────────────────────────────────────────────────────────────────

with tab_significance:
    st.markdown('<div class="section-header">Bootstrap Significance Test</div>',
                unsafe_allow_html=True)
    st.caption(
        "Runs seed-bootstrap significance test on the selected asset's "
        "most recent ODW result."
    )

    valid_assets_sig = [a for a in ranked if not a.error and a.scan is not None]
    if not valid_assets_sig:
        st.warning("No valid scans.")
    else:
        sig_labels = [f"{a.ticker} ({a.timeframe})" for a in valid_assets_sig]
        sel_sig    = st.selectbox("Asset", sig_labels, key="sig_asset")
        sel_a      = valid_assets_sig[sig_labels.index(sel_sig)]

        col_l2, col_r2 = st.columns(2)
        with col_l2:
            n_boot = st.slider("Bootstrap samples", 50, 500, 100, step=50)
            alpha  = st.slider("Alpha level", 0.01, 0.10, 0.05, step=0.01)
        with col_r2:
            run_sig = st.button("Run significance test", type="primary")

        if run_sig:
            with st.spinner("Running bootstrap…"):
                try:
                    odw    = sel_a.scan.latest.odw_result
                    sig    = fawp_significance(odw, n_bootstrap=n_boot, alpha=alpha)
                    st.success(
                        f"**p_fawp = {sig.p_value_fawp:.3f}**  |  "
                        f"p_null = {sig.p_value_null:.3f}  |  "
                        f"significant = {'✅ YES' if sig.significant else '❌ NO'}"
                    )

                    col_s1, col_s2, col_s3 = st.columns(3)
                    col_s1.metric("p(FAWP)", f"{sig.p_value_fawp:.3f}")
                    col_s2.metric("p(null)",  f"{sig.p_value_null:.3f}")
                    col_s3.metric("Significant", "YES" if sig.significant else "NO")

                    with st.expander("Full significance summary"):
                        st.text(sig.summary())

                    if HAS_MPL:
                        fig = sig.plot(show=False)
                        if fig is not None:
                            st.pyplot(fig, use_container_width=True)
                            plt.close(fig)

                except Exception as e:
                    st.error(f"Significance test failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 5 — Export
# ─────────────────────────────────────────────────────────────────────────────

with tab_export:
    st.markdown('<div class="section-header">Export Results</div>',
                unsafe_allow_html=True)

    col_e1, col_e2, col_e3 = st.columns(3)

    # ── CSV ──────────────────────────────────────────────────────────────────
    with col_e1:
        st.markdown("**📄 CSV — watchlist summary**")
        csv_bytes = wl.to_dataframe().to_csv(index=False).encode()
        st.download_button(
            "Download watchlist.csv",
            data=csv_bytes, file_name="fawp_watchlist.csv",
            mime="text/csv", use_container_width=True,
        )

    # ── JSON ─────────────────────────────────────────────────────────────────
    with col_e2:
        st.markdown("**📋 JSON — full scan result**")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            wl.to_json(tf.name)
            json_bytes = Path(tf.name).read_bytes()
        st.download_button(
            "Download watchlist.json",
            data=json_bytes, file_name="fawp_watchlist.json",
            mime="application/json", use_container_width=True,
        )

    # ── HTML ─────────────────────────────────────────────────────────────────
    with col_e3:
        st.markdown("**🌐 HTML — self-contained report**")
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tf:
            wl.to_html(tf.name)
            html_bytes = Path(tf.name).read_bytes()
        st.download_button(
            "Download watchlist.html",
            data=html_bytes, file_name="fawp_watchlist.html",
            mime="text/html", use_container_width=True,
        )

    # ── Per-asset CSVs ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**Per-asset scan CSVs**")
    for a in ranked:
        if a.error or a.scan is None:
            continue
        csv_data = a.scan.to_dataframe().to_csv(index=False).encode()
        st.download_button(
            f"📥 {a.ticker} ({a.timeframe})",
            data=csv_data,
            file_name=f"fawp_{a.ticker}_{a.timeframe}.csv",
            mime="text/csv",
            key=f"dl_{a.ticker}_{a.timeframe}",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    "fawp-index v0.10.0 · Ralph Clayton (2026) · "
    "[GitHub](https://github.com/DrRalphClayton/fawp-index) · "
    "[Paper (E1-E7)](https://doi.org/10.5281/zenodo.18663547) · "
    "[Paper (E8)](https://doi.org/10.5281/zenodo.18673949) · "
    "[Book](https://www.amazon.com/dp/B0GS1ZVNM7/)"
)
