"""
FAWP Dashboard v0.15.0 — Streamlit app
========================================
Ralph Clayton (2026) · https://doi.org/10.5281/zenodo.18673949
"""

import time
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="FAWP Scanner",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=JetBrains+Mono:ital,wght@0,400;0,600;1,400&family=DM+Sans:wght@400;500;600&display=swap');

:root {
  --bg-app:       #07101E;
  --bg-card:      #0D1729;
  --bg-card2:     #111E35;
  --accent:       #D4AF37;
  --accent-dim:   #6A5518;
  --crimson:      #C0111A;
  --crimson-glow: rgba(192,17,26,0.35);
  --green:        #1DB954;
  --blue-mild:    #4A7FCC;
  --text-1:       #EDF0F8;
  --text-2:       #7A90B8;
  --text-3:       #3A4E70;
  --border:       #182540;
  --border-2:     #243650;
}

html, body, [class*="css"],
[data-testid="stAppViewContainer"],
[data-testid="stMain"] { font-family: 'DM Sans', sans-serif !important; }
[data-testid="stAppViewContainer"] > .main { background: var(--bg-app) !important; }
[data-testid="stHeader"] { background: transparent !important; border-bottom: 1px solid var(--border) !important; }
section[data-testid="stSidebar"] { background: #060D1A !important; border-right: 1px solid var(--border) !important; }
section[data-testid="stSidebar"] > div { padding-top: 0 !important; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

.stTabs [data-baseweb="tab-list"] { background: transparent !important; border-bottom: 1px solid var(--border) !important; gap: 0 !important; padding: 0 !important; }
.stTabs [data-baseweb="tab"] { background: transparent !important; color: var(--text-2) !important; font-family: 'DM Sans', sans-serif !important; font-size: 0.78rem !important; font-weight: 500 !important; letter-spacing: 0.09em !important; text-transform: uppercase !important; border-bottom: 2px solid transparent !important; border-radius: 0 !important; padding: 0.65em 1.4em !important; transition: color 0.15s !important; }
.stTabs [data-baseweb="tab"]:hover { color: var(--text-1) !important; }
.stTabs [aria-selected="true"] { color: var(--accent) !important; border-bottom: 2px solid var(--accent) !important; font-weight: 600 !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.2em !important; }

.stButton > button { font-family: 'DM Sans', sans-serif !important; font-weight: 600 !important; border-radius: 4px !important; letter-spacing: 0.04em !important; transition: all 0.15s !important; }
.stButton > button[kind="primary"] { background: var(--accent) !important; color: #07101E !important; border: none !important; }
.stButton > button[kind="primary"]:hover { background: #C09C28 !important; box-shadow: 0 0 18px rgba(212,175,55,0.3) !important; }
.stButton > button:not([kind="primary"]) { background: var(--bg-card) !important; color: var(--text-2) !important; border: 1px solid var(--border-2) !important; }
.stDownloadButton > button { background: var(--bg-card) !important; color: var(--text-2) !important; border: 1px solid var(--border-2) !important; font-family: 'DM Sans', sans-serif !important; border-radius: 4px !important; }
.stDownloadButton > button:hover { border-color: var(--accent-dim) !important; color: var(--accent) !important; }

[data-testid="stNumberInput"] input, [data-testid="stTextInput"] input, textarea { background: var(--bg-card) !important; border: 1px solid var(--border-2) !important; color: var(--text-1) !important; font-family: 'JetBrains Mono', monospace !important; font-size: 0.9em !important; border-radius: 4px !important; }
.stSelectbox [data-baseweb="select"] > div { background: var(--bg-card) !important; border: 1px solid var(--border-2) !important; border-radius: 4px !important; }
[data-testid="stSlider"] > div > div > div > div { background: var(--accent) !important; }
details[data-testid="stExpander"] { background: var(--bg-card) !important; border: 1px solid var(--border) !important; border-radius: 6px !important; }
details[data-testid="stExpander"] summary { color: var(--text-2) !important; font-size: 0.85em !important; letter-spacing: 0.04em !important; }
div.stAlert { border-radius: 5px !important; }
div.stSuccess > div { background: rgba(29,185,84,0.10) !important; border-left: 3px solid var(--green) !important; }
div.stWarning > div { background: rgba(212,175,55,0.10) !important; border-left: 3px solid var(--accent) !important; }
div.stError > div   { background: rgba(192,17,26,0.10) !important; border-left: 3px solid var(--crimson) !important; }
div.stInfo > div    { background: rgba(120,160,220,0.10) !important; border-left: 3px solid #4A7FCC !important; }
section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] small, section[data-testid="stSidebar"] span { color: var(--text-2) !important; }
section[data-testid="stSidebar"] .stRadio label { color: var(--text-1) !important; }

/* ── Brand block ── */
.sb-brand { padding: 1.2em 1em 0.6em; border-bottom: 1px solid var(--border); margin-bottom: 0.5em; }
.sb-brand-name { font-family: 'Syne', sans-serif !important; font-size: 1.15em; font-weight: 800; color: var(--accent) !important; letter-spacing: 0.02em; }
.sb-brand-ver  { font-family: 'JetBrains Mono', monospace; font-size: 0.65em; color: var(--text-3) !important; letter-spacing: 0.08em; }
.sb-section { font-family: 'DM Sans', sans-serif; font-size: 0.64rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.12em; color: var(--text-3) !important; padding: 0.8em 0 0.4em; }

/* ── Page header ── */
.page-hdr { padding: 0.4em 0 1.2em; border-bottom: 1px solid var(--border); margin-bottom: 1.4em; }
.page-hdr-eyebrow { font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: var(--accent); text-transform: uppercase; letter-spacing: 0.14em; margin-bottom: 0.3em; }
.page-hdr-title { font-family: 'Syne', sans-serif; font-size: 1.9em; font-weight: 800; color: var(--text-1); letter-spacing: -0.01em; line-height: 1.1; }
.page-hdr-title em { color: var(--accent); font-style: normal; }
.page-hdr-sub { font-size: 0.85em; color: var(--text-3); margin-top: 0.4em; }

/* ── KPI cards ── */
.kpi-row { display: flex; gap: 12px; margin-bottom: 1.4em; }
.kpi-card { flex: 1; background: var(--bg-card); border: 1px solid var(--border); border-top: 2px solid var(--accent-dim); border-radius: 6px; padding: 1em 1.1em 0.9em; min-width: 0; transition: border-top-color 0.2s; }
.kpi-card.alert { border-top-color: var(--crimson); animation: pulse-top 2s ease-in-out infinite; }
@keyframes pulse-top { 0%, 100% { box-shadow: none; } 50% { box-shadow: 0 -2px 16px var(--crimson-glow); } }
.kpi-val { font-family: 'JetBrains Mono', monospace; font-size: 1.85em; font-weight: 600; color: var(--text-1); line-height: 1.05; }
.kpi-card.alert .kpi-val { color: var(--crimson); }
.kpi-lbl { font-size: 0.68em; font-weight: 500; text-transform: uppercase; letter-spacing: 0.1em; color: var(--text-3); margin-top: 0.4em; }

/* ── Section header ── */
.sec-hdr { font-size: 0.68rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.12em; color: var(--accent); padding-bottom: 5px; border-bottom: 1px solid var(--accent-dim); margin: 1.5em 0 0.75em; }

/* ── Status pills ── */
.pill { display: inline-flex; align-items: center; gap: 5px; padding: 0.18em 0.65em; border-radius: 100px; font-family: 'JetBrains Mono', monospace; font-size: 0.68em; font-weight: 600; letter-spacing: 0.07em; text-transform: uppercase; }
.pill-fawp   { background: rgba(192,17,26,0.15);  color: #FF3040; border: 1px solid rgba(192,17,26,0.35); }
.pill-high   { background: rgba(212,175,55,0.12);  color: #D4AF37; border: 1px solid rgba(212,175,55,0.3); }
.pill-watch  { background: rgba(74,127,204,0.10);  color: #6A9FD8; border: 1px solid rgba(74,127,204,0.25); }
.pill-clear  { background: rgba(29,185,84,0.08);   color: #1DB954; border: 1px solid rgba(29,185,84,0.2); }
.pill-fawp::before { content: ''; display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: currentColor; animation: blink 1.4s ease-in-out infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.25} }

/* ── Sparkline ── */
.spark { display: inline-flex; align-items: flex-end; gap: 2px; height: 18px; vertical-align: middle; }
.spark-bar { width: 4px; border-radius: 1px 1px 0 0; }
.trend-up   { color: #C0111A; font-size: 0.8em; margin-left: 3px; }
.trend-down { color: #1DB954; font-size: 0.8em; margin-left: 3px; }
.trend-flat { color: #3A4E70; font-size: 0.8em; margin-left: 3px; }

/* ── ODW bar ── */
.odw-wrap { display: inline-flex; align-items: center; gap: 5px; }
.odw-track { width: 56px; height: 6px; background: #182540; border-radius: 3px; position: relative; overflow: hidden; display: inline-block; }
.odw-fill  { position: absolute; top: 0; height: 100%; border-radius: 3px; background: var(--crimson); opacity: 0.75; }
.odw-label { font-family: 'JetBrains Mono', monospace; font-size: 0.72em; color: var(--text-3); }

/* ── Score coloring ── */
.score-fawp  { font-family: 'JetBrains Mono', monospace; color: #FF3040; font-weight: 600; }
.score-high  { font-family: 'JetBrains Mono', monospace; color: #D4AF37; }
.score-watch { font-family: 'JetBrains Mono', monospace; color: #6A9FD8; }
.score-clear { font-family: 'JetBrains Mono', monospace; color: #3A4E70; }

/* ── Filter bar ── */
.filter-bar { display: flex; gap: 6px; margin-bottom: 0.9em; flex-wrap: wrap; }
.fbtn { padding: 0.22em 0.9em; border-radius: 100px; font-size: 0.72em; font-weight: 600; cursor: pointer; border: 1px solid var(--border-2); background: transparent; color: var(--text-2); letter-spacing: 0.04em; transition: all 0.15s; font-family: 'DM Sans', sans-serif; }
.fbtn:hover { border-color: var(--accent-dim); color: var(--accent); }
.fbtn.on { background: var(--accent); color: #07101E; border-color: var(--accent); }

/* ── Asset rows ── */
.asset-row { background: var(--bg-card); border: 1px solid var(--border); border-radius: 6px; padding: 0.7em 1em; margin-bottom: 5px; display: flex; align-items: center; gap: 14px; font-size: 0.85em; transition: border-color 0.15s; }
.asset-row:hover { border-color: var(--border-2); }
.asset-row.fawp-active { border-left: 3px solid var(--crimson); }
.asset-row.high-risk   { border-left: 3px solid var(--accent); }
.asset-ticker { font-family: 'JetBrains Mono', monospace; font-weight: 600; color: var(--text-1); min-width: 72px; }
.asset-tf     { font-family: 'JetBrains Mono', monospace; font-size: 0.82em; color: var(--text-3); min-width: 28px; }
.asset-gap    { font-family: 'JetBrains Mono', monospace; font-size: 0.82em; color: var(--text-2); min-width: 70px; }
.asset-days   { font-family: 'JetBrains Mono', monospace; font-size: 0.8em; color: var(--text-3); min-width: 45px; }
.asset-spacer { flex: 1; }

/* ── Explain card ── */
.explain-card { background: var(--bg-card2); border: 1px solid var(--accent-dim); border-left: 3px solid var(--accent); border-radius: 6px; padding: 0.9em 1.2em; margin: 5px 0 8px; font-size: 0.82em; }
.explain-card .why-title { color: var(--accent); font-size: 0.65em; text-transform: uppercase; letter-spacing: 0.12em; font-weight: 600; margin-bottom: 0.55em; font-family: 'DM Sans', sans-serif; }
.explain-card ul { margin: 0; padding-left: 1.1em; color: var(--text-2); line-height: 1.9; }
.explain-card li { margin-bottom: 0.1em; }
.explain-card .rec { margin-top: 0.65em; padding-top: 0.65em; border-top: 1px solid var(--border); color: var(--text-3); font-size: 0.92em; font-style: italic; }

/* ── Mini leaderboard ── */
.lb-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 0.6em; }
.lb-cat { background: var(--bg-card); border: 1px solid var(--border); border-radius: 6px; padding: 0.8em 1em; }
.lb-cat-title { font-size: 0.62em; text-transform: uppercase; letter-spacing: 0.11em; color: var(--text-3); font-weight: 600; margin-bottom: 0.55em; font-family: 'DM Sans', sans-serif; }
.lb-row { display: flex; justify-content: space-between; align-items: center; padding: 0.22em 0; border-bottom: 1px solid var(--border); }
.lb-row:last-child { border-bottom: none; }
.lb-ticker { font-family: 'JetBrains Mono', monospace; font-size: 0.8em; color: var(--text-1); font-weight: 600; }
.lb-val    { font-family: 'JetBrains Mono', monospace; font-size: 0.78em; color: var(--accent); }
.lb-val.up { color: var(--crimson); }
.lb-val.ok { color: #1DB954; }

/* ── Info bar ── */
.info-bar { display: flex; align-items: center; gap: 10px; background: var(--bg-card); border: 1px solid var(--border); border-left: 3px solid var(--accent); border-radius: 5px; padding: 0.65em 1em; font-size: 0.83em; color: var(--text-2); margin-bottom: 1.2em; }
.info-bar .ib-label { font-family: 'JetBrains Mono', monospace; font-size: 0.8em; text-transform: uppercase; letter-spacing: 0.1em; color: var(--accent); white-space: nowrap; }

/* ── Alert row ── */
.alert-row { background: rgba(192,17,26,0.08); border: 1px solid rgba(192,17,26,0.3); border-radius: 5px; padding: 0.75em 1em; margin: 0.5em 0; font-size: 0.85em; color: var(--text-1); }
.alert-row strong { color: #FF3040; font-family: 'JetBrains Mono', monospace; }

/* ── Scan meta ── */
.scan-meta { font-family: 'JetBrains Mono', monospace; font-size: 0.7em; color: var(--text-3); margin-bottom: 0.6em; display: flex; gap: 1.2em; }
.scan-meta span { display: flex; align-items: center; gap: 4px; }

/* ── Footer ── */
.fawp-footer { margin-top: 2em; padding-top: 0.8em; border-top: 1px solid var(--border); display: flex; align-items: center; gap: 1.2em; flex-wrap: wrap; }
.fawp-footer a { font-size: 0.75em; color: var(--text-3) !important; text-decoration: none; letter-spacing: 0.03em; transition: color 0.15s; }
.fawp-footer a:hover { color: var(--accent) !important; }
.fawp-footer .ft-ver { font-family: 'JetBrains Mono', monospace; font-size: 0.68em; color: var(--text-3); letter-spacing: 0.07em; }
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
    from fawp_index.leaderboard import Leaderboard
    from fawp_index.explain import explain_asset
    HAS_FAWP = True
except ImportError as e:
    st.error(f"fawp-index not installed: {e}\n\n`pip install fawp-index[plot]`")
    st.stop()


# ── Dark matplotlib helper ─────────────────────────────────────────────────
def _dark_fig(w=7, h=3.2):
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


# ── HTML component helpers ─────────────────────────────────────────────────
def _kpi(val, label, alert=False):
    cls = "kpi-card alert" if alert else "kpi-card"
    return (f'<div class="{cls}">'
            f'<div class="kpi-val">{val}</div>'
            f'<div class="kpi-lbl">{label}</div></div>')


def _sec(label):
    return f'<div class="sec-hdr">{label}</div>'


def _severity_pill(a) -> str:
    s = a.latest_score
    if a.regime_active:
        return '<span class="pill pill-fawp">FAWP</span>'
    if s >= 0.005:
        return '<span class="pill pill-high">HIGH</span>'
    if s >= 0.001:
        return '<span class="pill pill-watch">WATCH</span>'
    return '<span class="pill pill-clear">CLEAR</span>'


def _score_cls(a) -> str:
    s = a.latest_score
    if a.regime_active:       return "score-fawp"
    if s >= 0.005:            return "score-high"
    if s >= 0.001:            return "score-watch"
    return "score-clear"


def _sparkline(a) -> str:
    if a.scan is None or len(a.scan.windows) < 2:
        return ""
    recent = a.scan.windows[-6:]
    scores = [w.regime_score for w in recent]
    mx = max(scores) if max(scores) > 0 else 1.0
    bars = ""
    for i, sc in enumerate(scores):
        h = max(3, int(18 * sc / mx))
        is_fawp = recent[i].fawp_found
        color = "#C0111A" if is_fawp else ("#D4AF37" if sc > mx * 0.5 else "#3A4E70")
        bars += f'<span class="spark-bar" style="height:{h}px;background:{color}"></span>'
    # trend arrow
    if len(scores) >= 3:
        slope = scores[-1] - scores[-3]
        if slope > mx * 0.05:
            arrow = '<span class="trend-up">▲</span>'
        elif slope < -mx * 0.05:
            arrow = '<span class="trend-down">▼</span>'
        else:
            arrow = '<span class="trend-flat">—</span>'
    else:
        arrow = ""
    return f'<span class="spark">{bars}</span>{arrow}'


def _odw_bar(a, tau_max: int = 40) -> str:
    if a.peak_odw_start is None or a.peak_odw_end is None:
        return '<span class="odw-label">—</span>'
    s, e = int(a.peak_odw_start), int(a.peak_odw_end)
    left  = int(100 * s / tau_max)
    width = max(4, int(100 * (e - s + 1) / tau_max))
    return (
        f'<div class="odw-wrap">'
        f'<span class="odw-label">{s}</span>'
        f'<div class="odw-track"><div class="odw-fill" style="left:{left}%;width:{width}%"></div></div>'
        f'<span class="odw-label">{e}</span>'
        f'</div>'
    )


def _explain_html(a) -> str:
    try:
        text = explain_asset(a, verbose=False)
        lines = text.split("\n")
        bullets = [ln.strip().lstrip("•").strip() for ln in lines
                   if ln.strip().startswith("•") and len(ln.strip()) > 3]
        rec_lines = [ln for ln in lines if "Recommend" in ln or "Predict" in ln
                     or "Reduce" in ln or "Monitor" in ln or "Suspend" in ln
                     or "No active" in ln]
        rec = rec_lines[0].replace("Recommendation:", "").strip() if rec_lines else ""
        items = "".join(f"<li>{b}</li>" for b in bullets[:5])
        rec_html = f'<div class="rec">{rec}</div>' if rec else ""
        return (
            f'<div class="explain-card">'
            f'<div class="why-title">Why {a.ticker} is flagged</div>'
            f'<ul>{items}</ul>'
            f'{rec_html}'
            f'</div>'
        )
    except Exception:
        return ""


def _leaderboard_html(lb) -> str:
    def _rows(entries, val_key="score", up_class=""):
        if not entries:
            return "<div style='color:#3A4E70;font-size:.8em;font-style:italic'>none</div>"
        out = ""
        for e in entries[:4]:
            val_str = e.detail if e.detail else f"{e.score:.4f}"
            cls = f"lb-val {up_class}" if up_class else "lb-val"
            out += (f'<div class="lb-row">'
                    f'<span class="lb-ticker">{e.ticker}</span>'
                    f'<span class="{cls}">{val_str}</span>'
                    f'</div>')
        return out

    return f"""
<div class="lb-grid">
  <div class="lb-cat">
    <div class="lb-cat-title">Top FAWP</div>
    {_rows(lb.top_fawp, up_class="up")}
  </div>
  <div class="lb-cat">
    <div class="lb-cat-title">Rising risk</div>
    {_rows(lb.rising_risk, up_class="up")}
  </div>
  <div class="lb-cat">
    <div class="lb-cat-title">Collapsing control</div>
    {_rows(lb.collapsing_control)}
  </div>
  <div class="lb-cat">
    <div class="lb-cat-title">Strongest ODW</div>
    {_rows(lb.strongest_odw)}
  </div>
</div>"""


# ═══════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div class="sb-brand">
      <div class="sb-brand-name">FAWP Scanner</div>
      <div class="sb-brand-ver">fawp-index&nbsp;&nbsp;v{_FAWP_VERSION}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section">Data source</div>', unsafe_allow_html=True)
    source = st.radio("", ["Upload CSV(s)", "Enter tickers (yfinance)", "Demo data"],
                      index=2, label_visibility="collapsed")

    st.markdown('<div class="sb-section">Scanner settings</div>', unsafe_allow_html=True)
    window  = st.slider("Rolling window (bars)",      60, 504, 252, step=10)
    step    = st.slider("Scan step (bars)",             1,  20,   5, step=1)
    tau_max = st.slider("Max tau",                      5,  80,  40, step=5)
    n_null  = st.slider("Null permutations (0=fast)",   0, 200,   0, step=10)
    epsilon = st.number_input("Epsilon (bits)", min_value=0.001, max_value=0.1,
                               value=0.01, step=0.001, format="%.3f")

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


# ═══════════════════════════════════════════════════════════════════════════
# Data loaders
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Loading demo data…")
def _load_demo() -> dict:
    def _gbm(n, mu=0.0002, sigma=0.012, seed=0):
        r = np.random.default_rng(seed)
        p = 100 * np.exp(np.cumsum(r.normal(mu, sigma, n)))
        return p

    n     = 600
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    assets = {"SPY":(0.0003,0.010,0),"QQQ":(0.0004,0.013,1),
              "GLD":(0.0001,0.008,2),"BTC":(0.0008,0.040,3),"TLT":(-0.0002,0.009,4)}
    dfs = {}
    for ticker, (mu, sigma, seed) in assets.items():
        vols = np.random.default_rng(seed+10).integers(500_000, 5_000_000, n).astype(float)
        dfs[ticker] = pd.DataFrame({"Close":_gbm(n,mu,sigma,seed),"Volume":vols}, index=dates)
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
            cc = [c for c in df.columns if "close" in c.lower() or "adj" in c.lower() or "price" in c.lower()]
            if not cc:
                cc = [df.select_dtypes(include=np.number).columns[0]]
            df = df.rename(columns={cc[0]: "Close"})
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
    dfs = {}
    for ticker in [t.strip().upper() for t in tickers_str.split(",") if t.strip()]:
        try:
            df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            if not df.empty:
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                dfs[ticker] = df
        except Exception as e:
            st.warning(f"Failed to fetch {ticker}: {e}")
    return dfs


@st.cache_data(
    show_spinner="Running FAWP scanner…",
    hash_funcs={dict: lambda d: str(sorted(d.keys()))},
)
def _run_scan(dfs, window, step, tau_max, n_null, epsilon, timeframes):
    from fawp_index.watchlist import WatchlistScanner
    from fawp_index.market import MarketScanConfig
    cfg = MarketScanConfig(window=window, step=step, tau_max=tau_max,
                           n_null=n_null, epsilon=epsilon)
    return WatchlistScanner(config=cfg, timeframes=timeframes, verbose=False).scan(dfs)


# ═══════════════════════════════════════════════════════════════════════════
# Page header
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="page-hdr">
  <div class="page-hdr-eyebrow">Information-Control Exclusion Principle</div>
  <div class="page-hdr-title"><em>FAWP</em> Dashboard</div>
  <div class="page-hdr-sub">Detecting when prediction persists after steering has collapsed</div>
</div>
""", unsafe_allow_html=True)

# ── Auto-detect demo mode from env (set by fawp-demo CLI) ────────────────
import os as _os
_DEMO_MODE    = _os.environ.get("FAWP_DEMO",         "0") == "1"
_DEMO_TICKERS = _os.environ.get("FAWP_DEMO_TICKERS", "")

# ── Load data ──────────────────────────────────────────────────────────────
dfs = {}

if _DEMO_MODE and not _DEMO_TICKERS and source != "Upload CSV(s)":
    source = "Demo data"
elif _DEMO_MODE and _DEMO_TICKERS:
    source = "Enter tickers (yfinance)"

if source == "Demo data":
    dfs = _load_demo()
    tickers_str = ", ".join(dfs.keys())
    n_bars = len(next(iter(dfs.values())))
    st.markdown(
        f'<div class="info-bar"><span class="ib-label">Demo</span>'
        f'{tickers_str} &nbsp;·&nbsp; {n_bars} bars each</div>',
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
        _default_tickers = _DEMO_TICKERS.replace(",", ", ") if _DEMO_TICKERS else "SPY, QQQ, GLD, BTC-USD"
        ticker_str = st.text_input("Tickers (comma-separated)", _default_tickers)
    with col2:
        period = st.selectbox("Period", ["1y", "2y", "5y", "max"], index=1)
    if st.button("Fetch data"):
        dfs = _load_yfinance(ticker_str, period)

if not dfs:
    st.warning("No data loaded. Choose a source and press **Run Scan**.")
    st.stop()

# ── Run scan ───────────────────────────────────────────────────────────────
if run_btn or "wl_result" not in st.session_state:
    _t0 = time.time()
    with st.spinner("Scanning…"):
        st.session_state["wl_result"]      = _run_scan(dfs, window, step, tau_max, n_null, epsilon, tuple(timeframes))
        st.session_state["scan_duration"]  = round(time.time() - _t0, 1)
        st.session_state["scan_timestamp"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")

wl             = st.session_state["wl_result"]
ranked         = wl.rank_by("score")
scan_duration  = st.session_state.get("scan_duration", "—")
scan_timestamp = st.session_state.get("scan_timestamp", "—")

# ═══════════════════════════════════════════════════════════════════════════
# Tabs
# ═══════════════════════════════════════════════════════════════════════════
tab_scanner, tab_curves, tab_heatmap, tab_significance, tab_export = st.tabs([
    "Scanner", "Curves", "Heatmap", "Significance", "Export",
])


# ──────────────────────────────────────────────────────────────────────────
# Tab 1 — Scanner
# ──────────────────────────────────────────────────────────────────────────
with tab_scanner:

    # KPI row
    n_active   = wl.n_flagged
    n_total    = wl.n_assets
    best_score = ranked[0].latest_score if ranked else 0.0
    pct        = int(n_active / n_total * 100) if n_total else 0

    st.markdown(
        '<div class="kpi-row">'
        + _kpi(n_total,             "Assets scanned")
        + _kpi(n_active,            "FAWP active",  alert=bool(n_active))
        + _kpi(f"{best_score:.4f}", "Top score",    alert=bool(n_active))
        + _kpi(f"{pct}%",           "Flagged",      alert=bool(n_active))
        + "</div>",
        unsafe_allow_html=True,
    )

    # Scan meta line
    st.markdown(
        f'<div class="scan-meta">'
        f'<span>Scanned {scan_timestamp}</span>'
        f'<span>Duration {scan_duration}s</span>'
        f'<span>ε = {epsilon:.3f} bits</span>'
        f'<span>Window {window} bars · τmax {tau_max}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Sort + filter controls
    col_sort, col_filter = st.columns([1, 3])
    with col_sort:
        rank_col = st.selectbox("Sort by", ["score", "gap", "persistence", "freshness"], key="rank_col")
    with col_filter:
        flt = st.radio("Show", ["All", "FAWP only", "Watching", "Rising"],
                       horizontal=True, key="flt", label_visibility="collapsed")

    sorted_assets = wl.rank_by(rank_col)

    # Apply filter
    def _passes_filter(a, flt):
        if flt == "All":       return True
        if flt == "FAWP only": return a.regime_active
        if flt == "Watching":  return not a.regime_active and a.latest_score >= 0.001
        if flt == "Rising":
            if a.scan is None or len(a.scan.windows) < 3: return False
            w = a.scan.windows
            return w[-1].regime_score > w[-3].regime_score
        return True

    filtered = [a for a in sorted_assets if not a.error and _passes_filter(a, flt)]

    st.markdown(_sec(f"Results — {len(filtered)} asset(s)"), unsafe_allow_html=True)

    if not filtered:
        st.info("No assets match the current filter.")
    else:
        for a in filtered:
            pill_html  = _severity_pill(a)
            spark_html = _sparkline(a)
            odw_html   = _odw_bar(a, tau_max)
            score_cls  = _score_cls(a)
            row_cls    = "asset-row fawp-active" if a.regime_active else (
                         "asset-row high-risk" if a.latest_score >= 0.005 else "asset-row")
            days_str   = f"{a.days_in_regime}d" if a.days_in_regime else "—"
            age_str    = f"age {a.signal_age_days}d"

            st.markdown(
                f'<div class="{row_cls}">'
                f'<span class="asset-ticker">{a.ticker}</span>'
                f'<span class="asset-tf">{a.timeframe}</span>'
                f'{pill_html}'
                f'<span class="{score_cls}" style="min-width:64px">{a.latest_score:.4f}</span>'
                f'{spark_html}'
                f'<span class="asset-spacer"></span>'
                f'<span class="asset-gap">gap {a.peak_gap_bits:.4f}b</span>'
                f'{odw_html}'
                f'<span class="asset-days">{days_str}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Inline explain expander for flagged/high-risk assets
            if a.regime_active or a.latest_score >= 0.005:
                with st.expander(f"Why {a.ticker} is flagged", expanded=False):
                    st.markdown(_explain_html(a), unsafe_allow_html=True)

    # Mini leaderboard
    st.markdown(_sec("Leaderboard"), unsafe_allow_html=True)
    try:
        lb = Leaderboard.from_watchlist(wl)
        st.markdown(_leaderboard_html(lb), unsafe_allow_html=True)
        # Download leaderboard
        col_lb1, col_lb2 = st.columns([1, 4])
        with col_lb1:
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tf:
                lb.to_html(tf.name)
                lb_html = Path(tf.name).read_bytes()
            st.download_button(
                "Download leaderboard",
                data=lb_html,
                file_name="fawp_leaderboard.html",
                mime="text/html",
            )
    except Exception as e:
        st.caption(f"Leaderboard unavailable: {e}")

    # Threshold alerts
    if alert_threshold > 0:
        st.markdown(_sec("Threshold alerts"), unsafe_allow_html=True)
        triggered = [a for a in sorted_assets if a.regime_active and a.peak_gap_bits >= alert_threshold]
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


# ──────────────────────────────────────────────────────────────────────────
# Tab 2 — Curves
# ──────────────────────────────────────────────────────────────────────────
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

        with col_l:
            st.markdown(f'<div class="sec-hdr">Regime score — {sel_asset.ticker}</div>',
                        unsafe_allow_html=True)
            if HAS_MPL:
                fig, ax = _dark_fig(6, 3)
                dates_n = np.arange(len(scan.dates))
                colors  = ["#C0111A" if f else "#1DB954" for f in scan.fawp_flags]
                ax.bar(dates_n, scan.regime_scores, color=colors, width=1.0, alpha=0.85, edgecolor="none")
                ax.axvline(win_idx, color="#D4AF37", lw=1.5, ls="--", label="selected")
                n = len(scan.dates)
                ticks = np.linspace(0, n - 1, min(6, n), dtype=int)
                ax.set_xticks(ticks)
                ax.set_xticklabels([str(scan.dates[i].date()) for i in ticks],
                                   rotation=25, ha="right", fontsize=7, color="#7A90B8")
                ax.set_ylabel("Score", fontsize=8, color="#7A90B8")
                ax.set_ylim(0, max(scan.regime_scores.max() * 1.15, 0.01))
                ax.legend(fontsize=7, facecolor="#0D1729", labelcolor="#7A90B8", edgecolor="#182540")
                plt.tight_layout(pad=0.4)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        with col_r:
            fawp_pill = ('<span class="pill pill-fawp">FAWP</span>'
                         if win.fawp_found else '<span class="pill pill-clear">Clear</span>')
            st.markdown(f'<div class="sec-hdr">MI curves — {win.date.date()} &nbsp; {fawp_pill}</div>',
                        unsafe_allow_html=True)
            if HAS_MPL:
                fig, ax = _dark_fig(6, 3)
                tau = win.tau
                ax.plot(tau, win.pred_mi, color="#D4AF37", lw=1.8, marker="o", ms=3, label="Pred MI", zorder=3)
                ax.plot(tau, win.steer_mi, color="#4A7FCC", lw=1.8, ls="--", marker="s", ms=3, label="Steer MI", zorder=3)
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
                "peak_gap_bits":round(float(odw.peak_gap_bits), 6),
                "n_obs":        win.n_obs,
            })


# ──────────────────────────────────────────────────────────────────────────
# Tab 3 — Heatmap
# ──────────────────────────────────────────────────────────────────────────
with tab_heatmap:
    if HAS_MPL:
        tickers_h = sorted(set(a.ticker    for a in wl.assets if not a.error))
        tfs_h     = sorted(set(a.timeframe for a in wl.assets if not a.error))
        n_t, n_tf = len(tickers_h), len(tfs_h)

        mat = np.full((n_t, n_tf), np.nan)
        for a in wl.assets:
            if not a.error:
                mat[tickers_h.index(a.ticker), tfs_h.index(a.timeframe)] = a.latest_score

        def _heatmap(mat_data, title, cbar_label):
            fig, ax = plt.subplots(figsize=(max(4, n_tf * 1.8), max(3, n_t * 0.8)))
            fig.patch.set_facecolor("#0D1729")
            ax.set_facecolor("#0D1729")
            im = ax.imshow(mat_data, aspect="auto", cmap="RdYlGn_r",
                           vmin=0, vmax=max(0.01, float(np.nanmax(mat_data))))
            ax.set_xticks(range(n_tf)); ax.set_xticklabels(tfs_h, fontsize=9, color="#7A90B8")
            ax.set_yticks(range(n_t));  ax.set_yticklabels(tickers_h, fontsize=9, color="#7A90B8")
            cb = plt.colorbar(im, ax=ax, label=cbar_label)
            cb.ax.yaxis.set_tick_params(color="#7A90B8", labelsize=7)
            cb.set_label(cbar_label, color="#7A90B8", fontsize=8)
            for i in range(n_t):
                for j in range(n_tf):
                    if not np.isnan(mat_data[i, j]):
                        ax.text(j, i, f"{mat_data[i,j]:.3f}", ha="center", va="center",
                                fontsize=8, color="white" if mat_data[i,j] > 0.3 else "#7A90B8",
                                fontfamily="monospace")
            for spine in ax.spines.values():
                spine.set_edgecolor("#182540")
            plt.tight_layout(pad=0.4)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.markdown(_sec("Regime score heatmap"), unsafe_allow_html=True)
        st.caption("Rows = assets · Columns = timeframes · Color = latest score")
        _heatmap(mat, "Score", "Score")

        mat_gap = np.full((n_t, n_tf), np.nan)
        for a in wl.assets:
            if not a.error:
                mat_gap[tickers_h.index(a.ticker), tfs_h.index(a.timeframe)] = a.peak_gap_bits

        st.markdown(_sec("Peak leverage gap heatmap (bits)"), unsafe_allow_html=True)
        _heatmap(mat_gap, "bits", "bits")
    else:
        st.warning("matplotlib not installed — `pip install fawp-index[plot]`")


# ──────────────────────────────────────────────────────────────────────────
# Tab 4 — Significance
# ──────────────────────────────────────────────────────────────────────────
with tab_significance:
    st.markdown(_sec("Bootstrap significance test"), unsafe_allow_html=True)
    st.caption("Runs a seed-bootstrap significance test on the selected asset's most recent ODW result.")

    valid_sig = [a for a in ranked if not a.error and a.scan is not None]
    if not valid_sig:
        st.warning("No valid scans.")
    else:
        sig_labels = [f"{a.ticker} ({a.timeframe})" for a in valid_sig]
        sel_sig    = st.selectbox("Asset", sig_labels, key="sig_asset")
        sel_a      = valid_sig[sig_labels.index(sel_sig)]

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


# ──────────────────────────────────────────────────────────────────────────
# Tab 5 — Export
# ──────────────────────────────────────────────────────────────────────────
with tab_export:
    st.markdown(_sec("Download results"), unsafe_allow_html=True)

    col_e1, col_e2, col_e3 = st.columns(3)
    with col_e1:
        st.markdown("**CSV** — watchlist summary")
        st.download_button("Download watchlist.csv",
            data=wl.to_dataframe().to_csv(index=False).encode(),
            file_name="fawp_watchlist.csv", mime="text/csv", use_container_width=True)
    with col_e2:
        st.markdown("**JSON** — full scan result")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            wl.to_json(tf.name)
            json_bytes = Path(tf.name).read_bytes()
        st.download_button("Download watchlist.json",
            data=json_bytes, file_name="fawp_watchlist.json",
            mime="application/json", use_container_width=True)
    with col_e3:
        st.markdown("**HTML** — self-contained report")
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tf:
            wl.to_html(tf.name)
            html_bytes = Path(tf.name).read_bytes()
        st.download_button("Download watchlist.html",
            data=html_bytes, file_name="fawp_watchlist.html",
            mime="text/html", use_container_width=True)

    st.markdown(_sec("Leaderboard export"), unsafe_allow_html=True)
    try:
        lb = Leaderboard.from_watchlist(wl)
        col_l1, col_l2, col_l3 = st.columns(3)
        with col_l1:
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tf:
                lb.to_html(tf.name)
                lb_h = Path(tf.name).read_bytes()
            st.download_button("Download leaderboard.html", data=lb_h,
                file_name="fawp_leaderboard.html", mime="text/html", use_container_width=True)
        with col_l2:
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
                lb.to_csv(tf.name)
                lb_c = Path(tf.name).read_bytes()
            st.download_button("Download leaderboard.csv", data=lb_c,
                file_name="fawp_leaderboard.csv", mime="text/csv", use_container_width=True)
        with col_l3:
            lb_j = lb.to_dict()
            import json as _json
            st.download_button("Download leaderboard.json",
                data=_json.dumps(lb_j, indent=2).encode(),
                file_name="fawp_leaderboard.json", mime="application/json", use_container_width=True)
    except Exception as e:
        st.caption(f"Leaderboard export unavailable: {e}")

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


# ── Footer ─────────────────────────────────────────────────────────────────
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
