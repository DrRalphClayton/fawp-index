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

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title   = "FAWP Scanner",
    page_icon    = "📡",
    layout       = "wide",
    initial_sidebar_state = "expanded",
)

# ── Design system ─────────────────────────────────────────────────────────────
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=JetBrains+Mono:ital,wght@0,400;0,600;1,400&family=DM+Sans:wght@400;500;600&display=swap');

:root {
  --bg-app:      #07101E;
  --bg-card:     #0D1729;
  --bg-card2:    #111E35;
  --accent:      #D4AF37;
  --accent-dim:  #6A5518;
  --crimson:     #C0111A;
  --crimson-glow:rgba(192,17,26,0.35);
  --green:       #1DB954;
  --text-1:      #EDF0F8;
  --text-2:      #7A90B8;
  --text-3:      #3A4E70;
  --border:      #182540;
  --border-2:    #243650;
}

/* ── Base ── */
html, body, [class*="css"],
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
  font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stAppViewContainer"] > .main {
  background: var(--bg-app) !important;
}
[data-testid="stHeader"] {
  background: transparent !important;
  border-bottom: 1px solid var(--border) !important;
}
/* Sidebar */
section[data-testid="stSidebar"] {
  background: #060D1A !important;
  border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] > div { padding-top: 0 !important; }

/* ── Typography ── */
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: transparent !important;
  border-bottom: 1px solid var(--border) !important;
  gap: 0 !important;
  padding: 0 !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  color: var(--text-2) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.78rem !important;
  font-weight: 500 !important;
  letter-spacing: 0.09em !important;
  text-transform: uppercase !important;
  border-bottom: 2px solid transparent !important;
  border-radius: 0 !important;
  padding: 0.65em 1.4em !important;
  transition: color 0.15s !important;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--text-1) !important; }
.stTabs [aria-selected="true"] {
  color: var(--accent) !important;
  border-bottom: 2px solid var(--accent) !important;
  font-weight: 600 !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.2em !important; }

/* ── Buttons ── */
.stButton > button {
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 600 !important;
  border-radius: 4px !important;
  letter-spacing: 0.04em !important;
  transition: all 0.15s !important;
}
.stButton > button[kind="primary"] {
  background: var(--accent) !important;
  color: #07101E !important;
  border: none !important;
}
.stButton > button[kind="primary"]:hover {
  background: #C09C28 !important;
  box-shadow: 0 0 18px rgba(212,175,55,0.3) !important;
}
.stButton > button:not([kind="primary"]) {
  background: var(--bg-card) !important;
  color: var(--text-2) !important;
  border: 1px solid var(--border-2) !important;
}
.stButton > button:not([kind="primary"]):hover {
  border-color: var(--accent-dim) !important;
  color: var(--text-1) !important;
}

/* ── Download buttons ── */
.stDownloadButton > button {
  background: var(--bg-card) !important;
  color: var(--text-2) !important;
  border: 1px solid var(--border-2) !important;
  font-family: 'DM Sans', sans-serif !important;
  border-radius: 4px !important;
}
.stDownloadButton > button:hover {
  border-color: var(--accent-dim) !important;
  color: var(--accent) !important;
}

/* ── Inputs ── */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input,
textarea {
  background: var(--bg-card) !important;
  border: 1px solid var(--border-2) !important;
  color: var(--text-1) !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.9em !important;
  border-radius: 4px !important;
}
.stSelectbox [data-baseweb="select"] > div {
  background: var(--bg-card) !important;
  border: 1px solid var(--border-2) !important;
  border-radius: 4px !important;
}

/* ── Sliders ── */
[data-testid="stSlider"] > div > div > div > div {
  background: var(--accent) !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] iframe { border: none !important; }
[data-testid="dataframe"] { background: var(--bg-card) !important; }

/* ── Expander ── */
details[data-testid="stExpander"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 6px !important;
}
details[data-testid="stExpander"] summary {
  color: var(--text-2) !important;
  font-size: 0.85em !important;
  letter-spacing: 0.04em !important;
}

/* ── Alerts ── */
div.stAlert { border-radius: 5px !important; }
div.stSuccess > div { background: rgba(29,185,84,0.10) !important; border-left: 3px solid var(--green) !important; }
div.stWarning > div { background: rgba(212,175,55,0.10) !important; border-left: 3px solid var(--accent) !important; }
div.stError > div   { background: rgba(192,17,26,0.10) !important; border-left: 3px solid var(--crimson) !important; }
div.stInfo > div    { background: rgba(120,160,220,0.10) !important; border-left: 3px solid #4A7FCC !important; }

/* ── Sidebar label fix ── */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] span {
  color: var(--text-2) !important;
}
section[data-testid="stSidebar"] .stRadio label { color: var(--text-1) !important; }

/* ──────────────────────────────────────────
   Custom component styles
   ────────────────────────────────────────── */

/* Page header */
.page-hdr {
  padding: 0.4em 0 1.2em;
  border-bottom: 1px solid var(--border);
  margin-bottom: 1.4em;
}
.page-hdr-eyebrow {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.68rem;
  color: var(--accent);
  text-transform: uppercase;
  letter-spacing: 0.14em;
  margin-bottom: 0.3em;
}
.page-hdr-title {
  font-family: 'Syne', sans-serif;
  font-size: 1.9em;
  font-weight: 800;
  color: var(--text-1);
  letter-spacing: -0.01em;
  line-height: 1.1;
}
.page-hdr-title em { color: var(--accent); font-style: normal; }
.page-hdr-sub {
  font-size: 0.85em;
  color: var(--text-3);
  margin-top: 0.4em;
  letter-spacing: 0.01em;
}

/* Sidebar brand block */
.sb-brand {
  padding: 1.2em 1em 0.6em;
  border-bottom: 1px solid var(--border);
  margin-bottom: 0.5em;
}
.sb-brand-name {
  font-family: 'Syne', sans-serif !important;
  font-size: 1.15em;
  font-weight: 800;
  color: var(--accent) !important;
  letter-spacing: 0.02em;
}
.sb-brand-ver {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.65em;
  color: var(--text-3) !important;
  letter-spacing: 0.08em;
}
.sb-section {
  font-family: 'DM Sans', sans-serif;
  font-size: 0.64rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--text-3) !important;
  padding: 0.8em 0 0.4em;
}

/* KPI cards */
.kpi-row { display: flex; gap: 12px; margin-bottom: 1.4em; }
.kpi-card {
  flex: 1;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-top: 2px solid var(--accent-dim);
  border-radius: 6px;
  padding: 1em 1.1em 0.9em;
  min-width: 0;
  transition: border-top-color 0.2s;
}
.kpi-card.alert {
  border-top-color: var(--crimson);
  animation: pulse-top 2s ease-in-out infinite;
}
@keyframes pulse-top {
  0%, 100% { box-shadow: none; }
  50%       { box-shadow: 0 -2px 16px var(--crimson-glow); }
}
.kpi-val {
  font-family: 'JetBrains Mono', monospace;
  font-size: 1.85em;
  font-weight: 600;
  color: var(--text-1);
  line-height: 1.05;
}
.kpi-card.alert .kpi-val { color: var(--crimson); }
.kpi-lbl {
  font-size: 0.68em;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--text-3);
  margin-top: 0.4em;
}

/* Section header */
.sec-hdr {
  font-size: 0.68rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--accent);
  padding-bottom: 5px;
  border-bottom: 1px solid var(--accent-dim);
  margin: 1.5em 0 0.75em;
}

/* FAWP status pill */
.pill {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 0.18em 0.65em;
  border-radius: 100px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.68em;
  font-weight: 600;
  letter-spacing: 0.07em;
  text-transform: uppercase;
}
.pill-fawp  { background: rgba(192,17,26,0.15); color: #FF3040; border: 1px solid rgba(192,17,26,0.35); }
.pill-clear { background: rgba(29,185,84,0.10); color: #1DB954; border: 1px solid rgba(29,185,84,0.25); }
.pill-fawp::before {
  content: '';
  display: inline-block;
  width: 6px; height: 6px;
  border-radius: 50%;
  background: currentColor;
  animation: blink 1.4s ease-in-out infinite;
}
@keyframes blink {
  0%,100% { opacity: 1; }
  50%      { opacity: 0.25; }
}

/* Info banner */
.info-bar {
  display: flex;
  align-items: center;
  gap: 10px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-left: 3px solid var(--accent);
  border-radius: 5px;
  padding: 0.65em 1em;
  font-size: 0.83em;
  color: var(--text-2);
  margin-bottom: 1.2em;
}
.info-bar .ib-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.8em;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--accent);
  white-space: nowrap;
}

/* Alert threshold row */
.alert-row {
  background: rgba(192,17,26,0.08);
  border: 1px solid rgba(192,17,26,0.3);
  border-radius: 5px;
  padding: 0.75em 1em;
  margin: 0.5em 0;
  font-size: 0.85em;
  color: var(--text-1);
}
.alert-row strong { color: #FF3040; font-family: 'JetBrains Mono', monospace; }

/* Footer */
.fawp-footer {
  margin-top: 2em;
  padding-top: 0.8em;
  border-top: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 1.2em;
  flex-wrap: wrap;
}
.fawp-footer a {
  font-size: 0.75em;
  color: var(--text-3) !important;
  text-decoration: none;
  letter-spacing: 0.03em;
  transition: color 0.15s;
}
.fawp-footer a:hover { color: var(--accent) !important; }
.fawp-footer .ft-ver {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.68em;
  color: var(--text-3);
  letter-spacing: 0.07em;
}
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

# ── Deferred imports ──────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from fawp_index import __version__ as _FAWP_VERSION
    from fawp_index.watchlist import WatchlistScanner, scan_watchlist  # noqa: F401
    from fawp_index.significance import fawp_significance
    HAS_FAWP = True
except ImportError as e:
    st.error(f"fawp-index not installed: {e}\n\n`pip install fawp-index[plot]`")
    st.stop()


# ── Matplotlib dark theme helper ──────────────────────────────────────────────
def _dark_fig(w=7, h=3.2):
    """Return a (fig, ax) pair with the dark FAWP theme applied."""
    fig, ax = plt.subplots(figsize=(w, h))
    for obj in [fig, ax]:
        obj.set_facecolor("#0D1729")
    ax.tick_params(colors="#7A90B8", labelsize=7.5)
    for spine in ax.spines.values():
        spine.set_edgecolor("#182540")
    ax.xaxis.label.set_color("#7A90B8")
    ax.yaxis.label.set_color("#7A90B8")
    ax.grid(True, color="#182540", linewidth=0.7, alpha=0.9)
    return fig, ax


def _dark_fig2(w=7, h=3.2):
    """Two-column figure with dark theme."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(w, h))
    for obj in [fig, ax1, ax2]:
        obj.set_facecolor("#0D1729")
    for ax in (ax1, ax2):
        ax.tick_params(colors="#7A90B8", labelsize=7.5)
        for spine in ax.spines.values():
            spine.set_edgecolor("#182540")
        ax.xaxis.label.set_color("#7A90B8")
        ax.yaxis.label.set_color("#7A90B8")
        ax.grid(True, color="#182540", linewidth=0.7, alpha=0.9)
    return fig, (ax1, ax2)


# ── HTML helpers ──────────────────────────────────────────────────────────────
def _kpi(val, label, alert=False):
    cls = "kpi-card alert" if alert else "kpi-card"
    return f'<div class="{cls}"><div class="kpi-val">{val}</div><div class="kpi-lbl">{label}</div></div>'


def _sec(label):
    return f'<div class="sec-hdr">{label}</div>'


def _pill(active: bool):
    if active:
        return '<span class="pill pill-fawp">FAWP</span>'
    return '<span class="pill pill-clear">Clear</span>'


# ═════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div class="sb-brand">
      <div class="sb-brand-name">FAWP Scanner</div>
      <div class="sb-brand-ver">fawp-index&nbsp;&nbsp;v{_FAWP_VERSION}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section">Data source</div>', unsafe_allow_html=True)
    source = st.radio(
        "", ["Upload CSV(s)", "Enter tickers (yfinance)", "Demo data"],
        index=2, label_visibility="collapsed",
    )

    st.markdown('<div class="sb-section">Scanner settings</div>', unsafe_allow_html=True)
    window  = st.slider("Rolling window (bars)",        60, 504, 252, step=10)
    step    = st.slider("Scan step (bars)",               1,  20,   5, step=1)
    tau_max = st.slider("Max tau",                        5,  80,  40, step=5)
    n_null  = st.slider("Null permutations (0=fast)",     0, 200,   0, step=10)
    epsilon = st.number_input(
        "Epsilon (bits)", min_value=0.001, max_value=0.1,
        value=0.01, step=0.001, format="%.3f",
    )

    st.markdown('<div class="sb-section">Timeframes</div>', unsafe_allow_html=True)
    tfs_1d  = st.checkbox("Daily (1d)",   value=True)
    tfs_1wk = st.checkbox("Weekly (1wk)", value=False)
    timeframes = []
    if tfs_1d:
        timeframes.append("1d")
    if tfs_1wk:
        timeframes.append("1wk")
    if not timeframes:
        timeframes = ["1d"]

    st.markdown('<div class="sb-section">Alerts</div>', unsafe_allow_html=True)
    alert_threshold = st.slider("Gap threshold (bits)", 0.0, 1.0, 0.05, 0.01)

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("▶  Run Scan", type="primary", use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# Data loading helpers
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading demo data…")
def _load_demo() -> dict:
    def _gbm(n, mu=0.0002, sigma=0.012, seed=0):
        r   = np.random.default_rng(seed)
        ret = r.normal(mu, sigma, n)
        p   = 100 * np.exp(np.cumsum(ret))
        return p

    n     = 600
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    assets = {
        "SPY": (0.0003, 0.010, 0),
        "QQQ": (0.0004, 0.013, 1),
        "GLD": (0.0001, 0.008, 2),
        "BTC": (0.0008, 0.040, 3),
        "TLT": (-0.0002, 0.009, 4),
    }
    dfs = {}
    for ticker, (mu, sigma, seed) in assets.items():
        prices = _gbm(n, mu, sigma, seed)
        vols   = np.random.default_rng(seed + 10).integers(500_000, 5_000_000, n).astype(float)
        dfs[ticker] = pd.DataFrame({"Close": prices, "Volume": vols}, index=dates)
    return dfs


def _load_uploaded(files) -> dict:
    dfs = {}
    for f in files:
        try:
            df = pd.read_csv(f)
            for col in df.columns:
                if "date" in col.lower() or "time" in col.lower():
                    df[col] = pd.to_datetime(df[col])
                    df = df.set_index(col)
                    break
            close_candidates = [c for c in df.columns
                                 if "close" in c.lower() or "adj" in c.lower() or "price" in c.lower()]
            if not close_candidates:
                st.warning(f"{f.name}: no Close-like column found — using first numeric column")
                close_candidates = [df.select_dtypes(include=np.number).columns[0]]
            df = df.rename(columns={close_candidates[0]: "Close"})
            dfs[Path(f.name).stem.upper()] = df
        except Exception as e:
            st.error(f"Failed to load {f.name}: {e}")
    return dfs


@st.cache_data(show_spinner="Fetching from yfinance…")
def _load_yfinance(tickers_str: str, period: str) -> dict:
    try:
        import yfinance as yf
    except ImportError:
        st.error("yfinance not installed — `pip install yfinance`")
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
# Run scan (cached)
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(
    show_spinner="Running FAWP scanner…",
    hash_funcs={dict: lambda d: str(sorted(d.keys()))},
)
def _run_scan(dfs, window, step, tau_max, n_null, epsilon, timeframes):
    from fawp_index.watchlist import WatchlistScanner
    from fawp_index.market import MarketScanConfig
    cfg = MarketScanConfig(window=window, step=step, tau_max=tau_max,
                           n_null=n_null, epsilon=epsilon)
    scanner = WatchlistScanner(config=cfg, timeframes=timeframes, verbose=False)
    return scanner.scan(dfs)


# ═════════════════════════════════════════════════════════════════════════════
# Page header
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="page-hdr">
  <div class="page-hdr-eyebrow">Information-Control Exclusion Principle</div>
  <div class="page-hdr-title"><em>FAWP</em> Dashboard</div>
  <div class="page-hdr-sub">
    Detecting when prediction persists after steering has collapsed
  </div>
</div>
""", unsafe_allow_html=True)


# ── Load data ─────────────────────────────────────────────────────────────────
dfs = {}

if source == "Demo data":
    dfs = _load_demo()
    tickers_str = ", ".join(dfs.keys())
    n_bars = len(next(iter(dfs.values())))
    st.markdown(
        f'<div class="info-bar">'
        f'<span class="ib-label">Demo</span>'
        f'{tickers_str} &nbsp;·&nbsp; {n_bars} bars each'
        f'</div>',
        unsafe_allow_html=True,
    )

elif source == "Upload CSV(s)":
    uploaded = st.file_uploader(
        "Upload CSV files — one per asset. Needs a date column and a Close column.",
        type=["csv"], accept_multiple_files=True,
    )
    if uploaded:
        dfs = _load_uploaded(uploaded)

elif source == "Enter tickers (yfinance)":
    col1, col2 = st.columns([3, 1])
    with col1:
        ticker_str = st.text_input("Tickers (comma-separated)", "SPY, QQQ, GLD, BTC-USD")
    with col2:
        period = st.selectbox("Period", ["1y", "2y", "5y", "max"], index=1)
    if st.button("Fetch data"):
        dfs = _load_yfinance(ticker_str, period)

if not dfs:
    st.warning("No data loaded. Choose a source and press **Run Scan**.")
    st.stop()

# ── Run scan ──────────────────────────────────────────────────────────────────
if run_btn or "wl_result" not in st.session_state:
    with st.spinner("Scanning…"):
        st.session_state["wl_result"] = _run_scan(
            dfs, window, step, tau_max, n_null, epsilon, tuple(timeframes)
        )

wl     = st.session_state["wl_result"]
ranked = wl.rank_by("score")


# ═════════════════════════════════════════════════════════════════════════════
# Tabs
# ═════════════════════════════════════════════════════════════════════════════

tab_scanner, tab_curves, tab_heatmap, tab_significance, tab_export = st.tabs([
    "Scanner", "Curves", "Heatmap", "Significance", "Export",
])


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Scanner
# ─────────────────────────────────────────────────────────────────────────────

with tab_scanner:
    n_active   = wl.n_flagged
    n_total    = wl.n_assets
    best_score = ranked[0].latest_score if ranked else 0.0
    pct        = int(n_active / n_total * 100) if n_total else 0

    st.markdown(
        '<div class="kpi-row">'
        + _kpi(n_total, "Assets scanned")
        + _kpi(n_active, "FAWP active", alert=bool(n_active))
        + _kpi(f"{best_score:.4f}", "Top score", alert=bool(n_active))
        + _kpi(f"{pct}%", "Flagged", alert=bool(n_active))
        + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(_sec("Ranked results"), unsafe_allow_html=True)

    rank_col      = st.selectbox("Sort by", ["score", "gap", "persistence", "freshness"], key="rank_col")
    sorted_assets = wl.rank_by(rank_col)

    table_rows = []
    for a in sorted_assets:
        if a.error:
            table_rows.append({
                "Status": "ERROR", "Ticker": a.ticker, "TF": a.timeframe,
                "Score": None, "Gap (bits)": None,
                "Regime start": "—", "Days": 0, "Signal age": 0, "ODW": "—",
            })
            continue
        table_rows.append({
            "Status":       "FAWP" if a.regime_active else "—",
            "Ticker":       a.ticker,
            "TF":           a.timeframe,
            "Score":        round(a.latest_score, 4),
            "Gap (bits)":   round(a.peak_gap_bits, 4),
            "Regime start": str(a.regime_start.date()) if a.regime_start else "—",
            "Days":         a.days_in_regime,
            "Signal age":   a.signal_age_days,
            "ODW": (f"{a.peak_odw_start}–{a.peak_odw_end}"
                    if a.peak_odw_start is not None else "—"),
        })

    df_table = pd.DataFrame(table_rows)

    def _color_row(row):
        if row.get("Status", "") == "FAWP":
            return ["background-color: rgba(192,17,26,0.08)"] * len(row)
        return [""] * len(row)

    st.dataframe(
        df_table.style.apply(_color_row, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    # Threshold alerts
    if alert_threshold > 0:
        st.markdown(_sec("Threshold alerts"), unsafe_allow_html=True)
        triggered = [a for a in sorted_assets
                     if a.regime_active and a.peak_gap_bits >= alert_threshold]
        if triggered:
            for a in triggered:
                st.markdown(
                    f'<div class="alert-row">'
                    f'<strong>{a.ticker}</strong> [{a.timeframe}] &nbsp;·&nbsp; '
                    f'gap = <strong>{a.peak_gap_bits:.4f} bits</strong> '
                    f'≥ threshold {alert_threshold:.3f} &nbsp;·&nbsp; '
                    f'score = {a.latest_score:.4f} &nbsp;·&nbsp; '
                    f'ODW {a.peak_odw_start}–{a.peak_odw_end}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.success(f"No assets with gap ≥ {alert_threshold:.3f} bits currently active.")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Curves
# ─────────────────────────────────────────────────────────────────────────────

with tab_curves:
    valid_assets = [a for a in ranked if not a.error and a.scan is not None]
    if not valid_assets:
        st.warning("No valid scans available.")
    else:
        asset_labels = [f"{a.ticker}  ({a.timeframe})" for a in valid_assets]
        sel_label    = st.selectbox("Asset", asset_labels, key="curve_asset")
        sel_asset    = valid_assets[asset_labels.index(sel_label)]
        scan         = sel_asset.scan

        n_windows = len(scan.windows)
        win_idx   = st.slider("Window (latest = right)", 0, n_windows - 1,
                              n_windows - 1, key="win_idx")
        win = scan.windows[win_idx]

        col_l, col_r = st.columns(2)

        # Regime score time series
        with col_l:
            st.markdown(f'<div class="sec-hdr">Regime score — {sel_asset.ticker}</div>',
                        unsafe_allow_html=True)
            if HAS_MPL:
                fig, ax = _dark_fig(6, 3)
                dates_n = np.arange(len(scan.dates))
                colors  = ["#C0111A" if f else "#1DB954" for f in scan.fawp_flags]
                ax.bar(dates_n, scan.regime_scores, color=colors,
                       width=1.0, alpha=0.85, edgecolor="none")
                ax.axvline(win_idx, color="#D4AF37", lw=1.5, ls="--", label="selected")
                n     = len(scan.dates)
                ticks = np.linspace(0, n - 1, min(6, n), dtype=int)
                ax.set_xticks(ticks)
                ax.set_xticklabels([str(scan.dates[i].date()) for i in ticks],
                                   rotation=25, ha="right", fontsize=7, color="#7A90B8")
                ax.set_ylabel("Score", fontsize=8, color="#7A90B8")
                ax.set_ylim(0, max(scan.regime_scores.max() * 1.15, 0.01))
                ax.legend(fontsize=7, facecolor="#0D1729", labelcolor="#7A90B8",
                          edgecolor="#182540")
                plt.tight_layout(pad=0.4)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        # MI curves for selected window
        with col_r:
            status_html = _pill(win.fawp_found)
            st.markdown(
                f'<div class="sec-hdr">MI curves — {win.date.date()} &nbsp; {status_html}</div>',
                unsafe_allow_html=True,
            )
            if HAS_MPL:
                fig, ax = _dark_fig(6, 3)
                tau = win.tau
                ax.plot(tau, win.pred_mi, color="#D4AF37", lw=1.8, marker="o",
                        ms=3, label="Pred MI", zorder=3)
                ax.plot(tau, win.steer_mi, color="#4A7FCC", lw=1.8, ls="--",
                        marker="s", ms=3, label="Steer MI", zorder=3)
                ax.fill_between(tau, win.steer_mi, win.pred_mi,
                                where=(win.pred_mi > win.steer_mi),
                                alpha=0.18, color="#D4AF37", label="Leverage gap")
                odw = win.odw_result
                if odw.odw_start is not None:
                    ax.axvspan(odw.odw_start - 0.5, odw.odw_end + 0.5,
                               alpha=0.14, color="#C0111A", zorder=0,
                               label=f"ODW {odw.odw_start}–{odw.odw_end}")
                ax.axhline(epsilon, color="#3A4E70", ls=":", lw=1.0, label=f"ε={epsilon}")
                ax.set_xlabel("τ (lag)", fontsize=8)
                ax.set_ylabel("MI (bits)", fontsize=8)
                ax.legend(fontsize=7, facecolor="#0D1729", labelcolor="#7A90B8",
                          edgecolor="#182540", loc="upper right")
                ax.set_ylim(bottom=0)
                plt.tight_layout(pad=0.4)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        # Leverage gap bar by τ
        st.markdown(_sec("Leverage gap by τ — selected window"), unsafe_allow_html=True)
        if HAS_MPL:
            gap = np.maximum(0, win.pred_mi - win.steer_mi)
            fig, ax = _dark_fig(10, 2.4)
            bar_colors = [
                "#C0111A" if (win.odw_result.odw_start is not None
                              and win.odw_result.odw_start <= t <= win.odw_result.odw_end)
                else "#2A4070"
                for t in win.tau
            ]
            ax.bar(win.tau, gap, color=bar_colors, alpha=0.9, edgecolor="none")
            ax.set_xlabel("τ", fontsize=8)
            ax.set_ylabel("Gap (bits)", fontsize=8)
            ax.set_title("Crimson = inside ODW", fontsize=7.5, color="#3A4E70", pad=4)
            plt.tight_layout(pad=0.4)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        # Window stats
        with st.expander("Window statistics"):
            odw = win.odw_result
            st.json({
                "date":          str(win.date.date()),
                "fawp_found":    bool(win.fawp_found),
                "regime_score":  round(float(win.regime_score), 6),
                "tau_h_plus":    odw.tau_h_plus,
                "tau_f":         odw.tau_f,
                "odw_start":     odw.odw_start,
                "odw_end":       odw.odw_end,
                "peak_gap_bits": round(float(odw.peak_gap_bits), 6),
                "n_obs":         win.n_obs,
            })


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Heatmap
# ─────────────────────────────────────────────────────────────────────────────

with tab_heatmap:
    if HAS_MPL:
        tickers_h = sorted(set(a.ticker    for a in wl.assets if not a.error))
        tfs_h     = sorted(set(a.timeframe for a in wl.assets if not a.error))
        n_t, n_tf = len(tickers_h), len(tfs_h)

        mat = np.full((n_t, n_tf), np.nan)
        for a in wl.assets:
            if a.error:
                continue
            mat[tickers_h.index(a.ticker), tfs_h.index(a.timeframe)] = a.latest_score

        # Score heatmap
        st.markdown(_sec("Regime score heatmap"), unsafe_allow_html=True)
        st.caption("Rows = assets · Columns = timeframes · Color = latest score")
        fig, ax = plt.subplots(figsize=(max(4, n_tf * 1.8), max(3, n_t * 0.8)))
        fig.patch.set_facecolor("#0D1729")
        ax.set_facecolor("#0D1729")
        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r",
                       vmin=0, vmax=max(0.01, np.nanmax(mat)))
        ax.set_xticks(range(n_tf))
        ax.set_xticklabels(tfs_h, fontsize=9, color="#7A90B8")
        ax.set_yticks(range(n_t))
        ax.set_yticklabels(tickers_h, fontsize=9, color="#7A90B8")
        cb = plt.colorbar(im, ax=ax, label="Score")
        cb.ax.yaxis.set_tick_params(color="#7A90B8", labelsize=7)
        cb.set_label("Score", color="#7A90B8", fontsize=8)
        for i in range(n_t):
            for j in range(n_tf):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center",
                            fontsize=8, color="white" if mat[i, j] > 0.3 else "#7A90B8",
                            fontfamily="monospace")
        for spine in ax.spines.values():
            spine.set_edgecolor("#182540")
        plt.tight_layout(pad=0.4)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Gap heatmap
        st.markdown(_sec("Peak leverage gap heatmap (bits)"), unsafe_allow_html=True)
        mat_gap = np.full((n_t, n_tf), np.nan)
        for a in wl.assets:
            if a.error:
                continue
            mat_gap[tickers_h.index(a.ticker), tfs_h.index(a.timeframe)] = a.peak_gap_bits

        fig, ax = plt.subplots(figsize=(max(4, n_tf * 1.8), max(3, n_t * 0.8)))
        fig.patch.set_facecolor("#0D1729")
        ax.set_facecolor("#0D1729")
        im2 = ax.imshow(mat_gap, aspect="auto", cmap="RdYlGn_r",
                        vmin=0, vmax=max(0.01, np.nanmax(mat_gap)))
        ax.set_xticks(range(n_tf))
        ax.set_xticklabels(tfs_h, fontsize=9, color="#7A90B8")
        ax.set_yticks(range(n_t))
        ax.set_yticklabels(tickers_h, fontsize=9, color="#7A90B8")
        cb2 = plt.colorbar(im2, ax=ax, label="bits")
        cb2.ax.yaxis.set_tick_params(color="#7A90B8", labelsize=7)
        cb2.set_label("bits", color="#7A90B8", fontsize=8)
        for i in range(n_t):
            for j in range(n_tf):
                if not np.isnan(mat_gap[i, j]):
                    ax.text(j, i, f"{mat_gap[i,j]:.3f}", ha="center", va="center",
                            fontsize=8, color="white" if mat_gap[i, j] > 0.3 else "#7A90B8",
                            fontfamily="monospace")
        for spine in ax.spines.values():
            spine.set_edgecolor("#182540")
        plt.tight_layout(pad=0.4)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.warning("matplotlib not installed — `pip install fawp-index[plot]`")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 4 — Significance
# ─────────────────────────────────────────────────────────────────────────────

with tab_significance:
    st.markdown(_sec("Bootstrap significance test"), unsafe_allow_html=True)
    st.caption(
        "Runs a seed-bootstrap significance test on the selected asset's "
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
                    odw = sel_a.scan.latest.odw_result
                    sig = fawp_significance(odw, n_bootstrap=n_boot, alpha=alpha)

                    sig_label = "✓ Significant" if sig.significant else "Not significant"
                    st.success(
                        f"**p_fawp = {sig.p_value_fawp:.3f}**  ·  "
                        f"p_null = {sig.p_value_null:.3f}  ·  {sig_label}"
                    )

                    c1, c2, c3 = st.columns(3)
                    c1.metric("p(FAWP)", f"{sig.p_value_fawp:.3f}")
                    c2.metric("p(null)",  f"{sig.p_value_null:.3f}")
                    c3.metric("Significant", "YES" if sig.significant else "NO")

                    with st.expander("Full significance summary"):
                        st.text(sig.summary())

                    if HAS_MPL:
                        fig = sig.plot(show=False)
                        if fig is not None:
                            fig.patch.set_facecolor("#0D1729")
                            for ax in fig.get_axes():
                                ax.set_facecolor("#0D1729")
                                for spine in ax.spines.values():
                                    spine.set_edgecolor("#182540")
                                ax.tick_params(colors="#7A90B8")
                            st.pyplot(fig, use_container_width=True)
                            plt.close(fig)

                except Exception as e:
                    st.error(f"Significance test failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 5 — Export
# ─────────────────────────────────────────────────────────────────────────────

with tab_export:
    st.markdown(_sec("Download results"), unsafe_allow_html=True)

    col_e1, col_e2, col_e3 = st.columns(3)

    with col_e1:
        st.markdown("**CSV** — watchlist summary")
        csv_bytes = wl.to_dataframe().to_csv(index=False).encode()
        st.download_button(
            "Download watchlist.csv",
            data=csv_bytes, file_name="fawp_watchlist.csv",
            mime="text/csv", use_container_width=True,
        )

    with col_e2:
        st.markdown("**JSON** — full scan result")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            wl.to_json(tf.name)
            json_bytes = Path(tf.name).read_bytes()
        st.download_button(
            "Download watchlist.json",
            data=json_bytes, file_name="fawp_watchlist.json",
            mime="application/json", use_container_width=True,
        )

    with col_e3:
        st.markdown("**HTML** — self-contained report")
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tf:
            wl.to_html(tf.name)
            html_bytes = Path(tf.name).read_bytes()
        st.download_button(
            "Download watchlist.html",
            data=html_bytes, file_name="fawp_watchlist.html",
            mime="text/html", use_container_width=True,
        )

    st.markdown(_sec("Per-asset CSVs"), unsafe_allow_html=True)
    for a in ranked:
        if a.error or a.scan is None:
            continue
        st.download_button(
            f"{a.ticker}  [{a.timeframe}]",
            data=a.scan.to_dataframe().to_csv(index=False).encode(),
            file_name=f"fawp_{a.ticker}_{a.timeframe}.csv",
            mime="text/csv",
            key=f"dl_{a.ticker}_{a.timeframe}",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    f'<div class="fawp-footer">'
    f'<span class="ft-ver">fawp-index&nbsp;&nbsp;v{_FAWP_VERSION}</span>'
    f'<a href="https://github.com/DrRalphClayton/fawp-index" target="_blank">GitHub</a>'
    f'<a href="https://doi.org/10.5281/zenodo.18663547" target="_blank">Paper E1–E7</a>'
    f'<a href="https://doi.org/10.5281/zenodo.18673949" target="_blank">Paper E8</a>'
    f'<a href="https://www.amazon.com/dp/B0GS1ZVNM7/" target="_blank">Book</a>'
    f'<span class="ft-ver">Ralph Clayton · 2026</span>'
    f'</div>',
    unsafe_allow_html=True,
)
