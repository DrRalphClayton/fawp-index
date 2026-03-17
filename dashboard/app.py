"""
FAWP Dashboard v1.1.6 — Streamlit app
========================================
Ralph Clayton (2026) · https://doi.org/10.5281/zenodo.18673949
"""

import sys
import time
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Make sibling imports work: streamlit run dashboard/app.py
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

try:
    from auth import (
        require_auth, get_user_email, sign_out,
        get_plan, get_limit, is_pro, is_admin,
    )
    _AUTH_ENABLED = True
except Exception:
    _AUTH_ENABLED = False
    def require_auth(): pass
    def get_user_email(): return None
    def sign_out(): pass
    def get_plan(): return "free"
    def get_limit(f): return None
    def is_pro(): return True
    def is_admin(): return False

st.set_page_config(
    page_title="FAWP — Information-Control Exclusion Principle",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded",
)

if _AUTH_ENABLED:
    require_auth()

try:
    from share import share_button as _share_button
    _SHARE_OK = True
except Exception:
    _SHARE_OK = False
    def _share_button(*a, **k): pass

# Demo mode state
import os as _os_early
_FAWP_ENV_DEMO = _os_early.getenv("FAWP_DEMO", "0") in ("1", "true", "yes")
_IS_DEMO = bool(st.session_state.get("_demo_mode", _FAWP_ENV_DEMO))

# ── App mode routing ───────────────────────────────────────────────────────────
# "finance" | "weather" | None (landing)
_APP_MODE = st.session_state.get("_app_mode", None)

if _APP_MODE is None:
    # ── Landing page ─────────────────────────────────────────────────────────
    st.markdown("""
<style>
.landing-wrap {
    display:flex; flex-direction:column; align-items:center;
    justify-content:center; min-height:75vh; padding:2em 1em;
}
.landing-title {
    font-family:'Syne',sans-serif; font-size:3em; font-weight:800;
    color:#D4AF37; letter-spacing:-.02em; text-align:center;
    margin-bottom:.2em;
}
.landing-sub {
    color:#3A4E70; font-size:.95em; text-align:center;
    margin-bottom:3em; font-family:'DM Sans',sans-serif;
}
.mode-card {
    background:#0D1729; border:1px solid #182540;
    border-radius:14px; padding:2.4em 2.8em;
    text-align:center; cursor:pointer;
    transition:border-color .2s, transform .15s;
    max-width:340px; width:100%;
}
.mode-card:hover { border-color:#D4AF37; transform:translateY(-3px); }
.mode-icon  { font-size:2.8em; margin-bottom:.5em; }
.mode-name  { font-family:'Syne',sans-serif; font-size:1.4em;
              font-weight:800; color:#EDF0F8; margin-bottom:.4em; }
.mode-desc  { color:#7A90B8; font-size:.85em; line-height:1.5; }
</style>
<div class="landing-wrap">
  <div class="landing-title">FAWP</div>
  <div class="landing-sub">
    Information-Control Exclusion Principle detector ·
    <a href="https://doi.org/10.5281/zenodo.18673949"
       style="color:#4A7FCC">doi:10.5281/zenodo.18673949</a>
  </div>
</div>
""", unsafe_allow_html=True)

    col_l, col_fin, col_gap, col_wx, col_r = st.columns([1, 3, 0.6, 3, 1])

    with col_fin:
        st.markdown("""
<div class="mode-card">
  <div class="mode-icon">📈</div>
  <div class="mode-name">FAWP Finance</div>
  <div class="mode-desc">
    Scan equities, crypto, ETFs and commodities.<br>
    Detect when forecast skill persists after market steering has collapsed.
    Real-time via yfinance or upload your own CSV.
  </div>
</div>
""", unsafe_allow_html=True)
        if st.button("▶  Open Finance Scanner",
                     use_container_width=True, type="primary",
                     key="goto_finance"):
            st.session_state["_app_mode"] = "finance"
            st.rerun()

    with col_wx:
        st.markdown("""
<div class="mode-card">
  <div class="mode-icon">🌦</div>
  <div class="mode-name">FAWP Weather</div>
  <div class="mode-desc">
    Scan ERA5 reanalysis for any location on Earth.<br>
    Detect when weather remains forecastable but intervention
    windows have already closed. Free · no API key needed.
  </div>
</div>
""", unsafe_allow_html=True)
        if st.button("▶  Open Weather Scanner",
                     use_container_width=True, type="primary",
                     key="goto_weather"):
            st.session_state["_app_mode"] = "weather"
            st.rerun()

    st.markdown("""
<div style="text-align:center;margin-top:3em;color:#1E2E4A;font-size:.78em">
  fawp-index v1.1.6 · Ralph Clayton · 2026 ·
  <a href="https://github.com/DrRalphClayton/fawp-index"
     style="color:#1E2E4A">GitHub</a> ·
  <a href="https://pypi.org/project/fawp-index/"
     style="color:#1E2E4A">PyPI</a>
</div>
""", unsafe_allow_html=True)
    st.stop()

# ── Mode header with back button ───────────────────────────────────────────────
_mode_label = "📈 Finance Scanner" if _APP_MODE == "finance" else "🌦 Weather Scanner"
_back_col, _title_col = st.columns([1, 9])
with _back_col:
    if st.button("← Back", key="back_to_landing"):
        st.session_state.pop("_app_mode", None)
        # Clear mode-specific state
        for _k in ["wl_result","input_dfs","_fetched_key","wx_result","wx_hazard"]:
            st.session_state.pop(_k, None)
        st.rerun()
with _title_col:
    st.markdown(
        f'<div style="padding:.4em 0;font-family:Syne,sans-serif;font-size:1.1em;'
        f'font-weight:700;color:#7A90B8">{_mode_label}</div>',
        unsafe_allow_html=True)

# ── Route to correct app ───────────────────────────────────────────────────────
if _APP_MODE == "weather":
    # Weather uses real ERA5 data — but keep demo bypass alive for demo users
    # (previously this cleared _demo_bypass which kicked demo users to login)
    # ── Inline weather dashboard ───────────────────────────────────────────
    import importlib.util as _ilu, os as _os2
    _wx_path = _os2.path.join(_THIS_DIR, "weather_app.py")
    _spec    = _ilu.spec_from_file_location("weather_app", _wx_path)
    _wxmod   = _ilu.module_from_spec(_spec)
    try:
        _spec.loader.exec_module(_wxmod)
    except SystemExit:
        pass
    st.stop()

# If _APP_MODE == "finance", fall through to the rest of app.py below


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
.conf-high   { font-family: 'JetBrains Mono', monospace; font-size: 0.65em; font-weight: 600; padding: 0.12em 0.55em; border-radius: 100px; background: rgba(29,185,84,0.12); color: #1DB954; border: 1px solid rgba(29,185,84,0.3); letter-spacing: 0.06em; }
.conf-medium { font-family: 'JetBrains Mono', monospace; font-size: 0.65em; font-weight: 600; padding: 0.12em 0.55em; border-radius: 100px; background: rgba(212,175,55,0.10); color: #D4AF37; border: 1px solid rgba(212,175,55,0.3); letter-spacing: 0.06em; }
.conf-low    { font-family: 'JetBrains Mono', monospace; font-size: 0.65em; font-weight: 600; padding: 0.12em 0.55em; border-radius: 100px; background: rgba(122,144,184,0.10); color: #7A90B8; border: 1px solid rgba(122,144,184,0.25); letter-spacing: 0.06em; }
.conf-insuf  { font-family: 'JetBrains Mono', monospace; font-size: 0.65em; color: #3A4E70; }
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

/* ── Validation / History ── */
.val-table { width:100%; border-collapse:collapse; font-size:0.84em; }
.val-table th { text-align:left; padding:.4em .8em; color:var(--text-3); font-size:.68em; text-transform:uppercase; letter-spacing:.08em; border-bottom:1px solid var(--border); }
.val-table td { padding:.4em .8em; border-bottom:1px solid var(--border); font-family:'JetBrains Mono', monospace; color:var(--text-2); }
.val-table tr:last-child td { border-bottom:none; }
.val-pos { color:#1DB954 !important; } .val-neg { color:#C0111A !important; }
.hist-row { display:flex; gap:12px; align-items:center; background:var(--bg-card); border:1px solid var(--border); border-radius:5px; padding:.5em 1em; margin-bottom:4px; font-size:.83em; }
.hist-date { font-family:'JetBrains Mono', monospace; font-size:.78em; color:var(--text-3); min-width:100px; }
.hist-score { font-family:'JetBrains Mono', monospace; color:var(--accent); min-width:60px; }
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
    from fawp_index.explain import explain_asset, confidence_badge
    from fawp_index.scan_history import ScanHistory
    from fawp_index.validation import validate_signals
    from fawp_index.compare import compare_signals
    from fawp_index.report_html import generate_html_report
    from fawp_index.constants import EPSILON_STEERING_RAW as _EPS_STEER
    from fawp_index.weather import fawp_from_open_meteo, scan_weather_grid, WeatherFAWPResult
    HAS_FAWP = True
except ImportError as e:
    st.error(f"fawp-index not installed: {e}\n\n`pip install fawp-index[plot]`")
    st.stop()

# Per-user storage (Supabase when logged in, local filesystem fallback)
try:
    from supabase_store import get_store, send_alert_email
    _HAS_SUPA_STORE = True
except ImportError:
    _HAS_SUPA_STORE = False
    def get_store():
        from fawp_index.scan_history import ScanHistory as _SH
        class _FallbackStore:
            def save_scan(self, r, label=''):
                try: _SH().save(r)
                except Exception: pass
            def asset_timeline(self, t, tf='1d', last_n=0):
                return _SH().asset_timeline(t, tf, last_n=last_n)
            def all_assets(self): return _SH().all_assets()
            def n_snapshots(self): return _SH().n_snapshots()
        return _FallbackStore()
    def send_alert_email(subject, body): return False


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



def _empty_state(icon, title, body, action=""):
    a = f'<div style="margin-top:1.2em;color:#D4AF37;font-size:.85em;font-weight:600">{action}</div>' if action else ""
    return (
        '<div style="text-align:center;padding:3em 2em;background:#0D1729;'
        'border:1px dashed #182540;border-radius:8px;max-width:480px;margin:2em auto">'
        f'<div style="font-size:2.2em;margin-bottom:.4em">{icon}</div>'
        f'<div style="font-family:sans-serif;font-size:1.05em;font-weight:700;'
        f'color:#EDF0F8;margin-bottom:.5em">{title}</div>'
        f'<div style="color:#7A90B8;font-size:.88em;line-height:1.5">{body}</div>'
        + a + '</div>'
    )


def _add_notification(title, body, kind="info"):
    """Queue an in-app notification."""
    import pandas as _pd
    notifs = st.session_state.get("_notifications", [])
    notifs.insert(0, {
        "title": title, "body": body, "kind": kind,
        "ts":    _pd.Timestamp.now().strftime("%H:%M"),
        "read":  False,
    })
    st.session_state["_notifications"] = notifs[:20]


def _notification_bell():
    """Render notification bell + dropdown in the sidebar."""
    notifs  = st.session_state.get("_notifications", [])
    n_unread = sum(1 for n in notifs if not n.get("read"))
    badge_html = (
        '<span class="notif-badge">' + str(n_unread) + "</span>"
        if n_unread else ""
    )
    st.sidebar.markdown(
        '<div class="notif-bell">🔔' + badge_html + "</div>",
        unsafe_allow_html=True,
    )
    label = f"Alerts {'🔴' if n_unread else '·'} ({len(notifs)})"
    with st.sidebar.expander(label, expanded=False):
        if not notifs:
            st.caption("No alerts yet.")
        else:
            if st.button("Mark all read", key="mark_all_read"):
                for n in notifs:
                    n["read"] = True
                st.rerun()
            for ntf in notifs[:10]:
                kls  = ntf.get("kind", "info")
                read = " style='opacity:.5'" if ntf.get("read") else ""
                st.markdown(
                    f'<div class="notif-item {kls}"{read}>'
                    f'<b style="color:#EDF0F8">{ntf["title"]}</b>'
                    f'<div style="color:#7A90B8;font-size:.88em">{ntf["body"]}</div>'
                    f'<div class="notif-ts">{ntf["ts"]}</div></div>',
                    unsafe_allow_html=True,
                )


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


def _confidence_html(a) -> str:
    """Render a null-calibrated confidence badge for an AssetResult."""
    try:
        badge = confidence_badge(a)
        tier  = badge["tier"]
        score = badge["score"]
        if tier == "HIGH":
            return f'<span class="conf-high" title="Confidence: {score:.2f} — high gap persistence + ODW concentration">HIGH conf</span>'
        if tier == "MEDIUM":
            return f'<span class="conf-medium" title="Confidence: {score:.2f} — moderate persistence">MED conf</span>'
        if tier == "LOW":
            return f'<span class="conf-low" title="Confidence: {score:.2f} — low persistence or diffuse gap">LOW conf</span>'
        return '<span class="conf-insuf" title="Insufficient scan windows">—</span>'
    except Exception:
        return ""


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
    # Default: Demo data for demo bypass users, Enter tickers for signed-in users
    _default_src_idx = 2 if st.session_state.get("_demo_bypass") else 1
    source = st.radio("", ["Upload CSV(s)", "Enter tickers (yfinance)", "Demo data"],
                      index=_default_src_idx, label_visibility="collapsed", key="data_source")
    # Clear stale data if source changed
    if st.session_state.get("_last_source") != source:
        st.session_state["_last_source"] = source
        st.session_state.pop("input_dfs", None)
        st.session_state.pop("wl_result", None)

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

    # ── User info + sign-out ──────────────────────────────────────────────
    if _AUTH_ENABLED:
        _email = get_user_email()
        if _email:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                f'<div style="font-family:monospace;font-size:0.65em;color:#3A4E70;"'
                f'>{_email}</div>',
                unsafe_allow_html=True,
            )
            _plan = get_plan()
            _plan_colors = {"free": "#3A4E70", "pro": "#D4AF37", "admin": "#C0111A"}
            _pcol = _plan_colors.get(_plan, "#3A4E70")
            st.markdown(
                f'<div style="font-family:monospace;font-size:0.62em;'
                f'color:{_pcol};font-weight:600;padding:.15em 0">'
                f'{_plan.upper()} PLAN</div>',
                unsafe_allow_html=True)
            if st.button("Sign out", use_container_width=True):
                sign_out()

    _notification_bell()


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
    """Returns {ticker: df} dict. Raises ImportError if yfinance missing."""
    import yfinance as yf
    dfs = {}
    errors = {}
    for ticker in [t.strip().upper() for t in tickers_str.split(",") if t.strip()]:
        try:
            df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            if not df.empty:
                # Flatten MultiIndex columns (yfinance 0.2.x returns MultiIndex)
                if hasattr(df.columns, "levels"):
                    df.columns = [c[0] if isinstance(c, tuple) else c
                                  for c in df.columns]
                else:
                    df.columns = [c[0] if isinstance(c, tuple) else c
                                  for c in df.columns]
                # Ensure Close column exists
                if "Close" not in df.columns and "close" in [c.lower() for c in df.columns]:
                    df = df.rename(columns={c: c.title() for c in df.columns})
                dfs[ticker] = df
            else:
                errors[ticker] = "empty response"
        except Exception as e:
            errors[ticker] = str(e)
    return dfs, errors


@st.cache_data(show_spinner="Running FAWP scanner…")
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
_DEMO_MODE    = _os.environ.get("FAWP_DEMO", "0") == "1" or st.session_state.get("_demo_mode", False)
_DEMO_TICKERS = _os.environ.get("FAWP_DEMO_TICKERS", "")

# ── Load data ──────────────────────────────────────────────────────────────
if "input_dfs" not in st.session_state:
    st.session_state["input_dfs"] = {}

if _DEMO_MODE and not _DEMO_TICKERS and source != "Upload CSV(s)":
    source = "Demo data"
elif _DEMO_MODE and _DEMO_TICKERS:
    source = "Enter tickers (yfinance)"
# Gate yfinance for demo bypass (not signed in)
if st.session_state.get("_demo_bypass") and source == "Enter tickers (yfinance)":
    source = "Demo data"
    st.sidebar.caption("Sign up to scan real tickers.")

if source == "Demo data":
    st.session_state["input_dfs"] = _load_demo()
    dfs = st.session_state["input_dfs"]
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
        st.session_state["input_dfs"] = _load_uploaded(uploaded)
    dfs = st.session_state.get("input_dfs", {})

elif source == "Enter tickers (yfinance)":
    col1, col2 = st.columns([3, 1])
    with col1:
        _default_tickers = _DEMO_TICKERS.replace(",", ", ") if _DEMO_TICKERS else "SPY, QQQ, GLD"
        _max_t = get_limit("max_tickers") or 999
        if not is_pro() and _AUTH_ENABLED:
            st.caption(f"Free plan: up to {_max_t} tickers")
        ticker_str = st.text_input("Tickers (comma-separated)", _default_tickers)
    with col2:
        period = st.selectbox("Period", ["1y", "2y", "5y", "max"], index=1)
    # Detect if inputs changed since last fetch
    _fetch_key = f"{ticker_str.strip()}|{period}"
    _last_key  = st.session_state.get("_fetched_key", "")
    _has_data  = bool(st.session_state.get("input_dfs"))
    if _has_data and _fetch_key != _last_key:
        st.warning(
            "⚠ Inputs changed — click **Fetch data** to refresh before scanning.",
            icon=None,
        )

    if st.button("Fetch data"):
        tickers = [t.strip().upper() for t in ticker_str.split(",") if t.strip()]
        if not is_pro() and _AUTH_ENABLED and len(tickers) > _max_t:
            st.error(f"Free plan supports up to {_max_t} tickers.")
        else:
            try:
                _fetched, _fetch_errors = _load_yfinance(",".join(tickers), period)
                for _t, _e in _fetch_errors.items():
                    st.warning(f"Failed to fetch {_t}: {_e}")
                if _fetched:
                    st.session_state["input_dfs"]   = _fetched
                    st.session_state["_fetched_key"] = _fetch_key
                    tickers_loaded = ", ".join(_fetched.keys())
                    n_bars = max(len(v) for v in _fetched.values())
                    first_date = min(str(v.index[0].date()) for v in _fetched.values())
                    last_date  = max(str(v.index[-1].date()) for v in _fetched.values())
                    st.success(f"✓ Loaded: {tickers_loaded} · {n_bars} bars · {first_date} → {last_date}")
                    st.info("Click ▶ Run Scan in the sidebar to analyse.")
                else:
                    st.error("No data returned — check ticker symbols (e.g. SPY, AAPL, BTC-USD) and try again.")
            except ImportError:
                st.error("yfinance not installed — `pip install yfinance`")
            except Exception as _fetch_ex:
                st.error(f"Fetch failed: {_fetch_ex}")
    dfs = st.session_state.get("input_dfs", {})

else:
    dfs = st.session_state.get("input_dfs", {})

if not dfs:
    st.markdown(_empty_state("📡","No data loaded","Choose a source in the sidebar and press <b>▶ Run Scan</b>.","← Select source in sidebar"), unsafe_allow_html=True)
    st.stop()

# ── Run scan ───────────────────────────────────────────────────────────────
if run_btn:
    _t0 = time.time()
    with st.spinner("Scanning…"):
        st.session_state["wl_result"]      = _run_scan(dfs, window, step, tau_max, n_null, epsilon, tuple(timeframes))
        st.session_state["scan_duration"]  = round(time.time() - _t0, 1)
        st.session_state["scan_timestamp"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    # Auto-save to per-user store
    try:
        get_store().save_scan(st.session_state["wl_result"])
    except Exception:
        pass
    # Fire in-app notifications
    _wl_snap = st.session_state["wl_result"]
    if _wl_snap.n_flagged > 0:
        _active = _wl_snap.active_regimes()[:3]
        _tickers = ", ".join(f"{a.ticker}[{a.timeframe}]" for a in _active)
        _add_notification(
            f"🔴 FAWP — {_wl_snap.n_flagged} regime(s) active",
            _tickers,
            kind="fawp",
        )
    else:
        _add_notification(
            f"✅ Scan complete — {_wl_snap.n_assets} assets clear",
            f"No FAWP regimes detected",
            kind="clear",
        )

wl = st.session_state.get("wl_result")
if wl is None:
    st.info("Load data and click ▶ Run Scan to start.")
    st.stop()

ranked = wl.rank_by("score")
scan_duration  = st.session_state.get("scan_duration", "—")
scan_timestamp = st.session_state.get("scan_timestamp", "—")
# Email alert if FAWP fired and user is logged in
if _AUTH_ENABLED and wl.n_flagged > 0 and run_btn:
    try:
        _flagged = ', '.join(
            f'{a.ticker}[{a.timeframe}]' for a in wl.active_regimes()[:5]
        )
        _subj = f'FAWP Alert — {wl.n_flagged} regime(s) active'
        _body = (
            f'FAWP Scanner detected {wl.n_flagged} active regime(s):\n\n'
            f'{_flagged}\n\n'
            f'Scanned: {scan_timestamp}\n'
            f'View at: https://fawp-scanner.info\n\n'
            f'fawp-index v{_FAWP_VERSION}'
        )
        send_alert_email(_subj, _body)
    except Exception:
        pass

# ═══════════════════════════════════════════════════════════════════════════
# Tabs
# ═══════════════════════════════════════════════════════════════════════════
# ── Demo mode banner ────────────────────────────────────────────────────
if _IS_DEMO and not st.session_state.get("_demo_banner_dismissed"):
    col_db1, col_db2 = st.columns([5, 1])
    with col_db1:
        st.markdown(
            '<div style="background:#1A2E10;border:1px solid #2A4A1A;'
            'border-radius:6px;padding:.6em 1em;font-size:.85em;color:#7ABF5E">'
            '🎮 <b>Demo mode</b> — synthetic data only. '
            '<a href="#" style="color:#D4AF37">Sign up free</a> for real scans.'
            '</div>',
            unsafe_allow_html=True)
    with col_db2:
        if st.button("✕", key="dismiss_demo_banner"):
            st.session_state["_demo_banner_dismissed"] = True
            st.rerun()
tab_scanner, tab_curves, tab_heatmap, tab_significance, tab_validation, tab_history, tab_compare, tab_weather, tab_admin, tab_export = st.tabs([
    "Scanner", "Curves", "Heatmap", "Significance",
    "Validation", "History", "Compare", "🌦 Weather", "⚙ Admin", "Export",
])


# ──────────────────────────────────────────────────────────────────────────
# Tab 1 — Scanner
# ──────────────────────────────────────────────────────────────────────────
with tab_scanner:
    # ── Onboarding — show on first use ───────────────────────────────────────
    if "wl_result" not in st.session_state:
        _plan = get_plan() if _AUTH_ENABLED else "free"
        _max_t = get_limit("max_tickers") or 3 if not is_pro() else 999
        st.markdown("""
<div style="background:#0D1729;border:1px solid #182540;border-top:3px solid #D4AF37;
border-radius:8px;padding:2em 2.2em 1.8em;max-width:640px;margin:2em auto">
<div style="font-family:'Syne',sans-serif;font-size:1.25em;font-weight:800;
color:#D4AF37;margin-bottom:.3em">Welcome to FAWP Scanner</div>
<div style="color:#7A90B8;font-size:.88em;margin-bottom:1.4em">
Detecting the Information-Control Exclusion Principle in real-time.<br>
Follow these steps to run your first scan:
</div>
<div style="display:flex;flex-direction:column;gap:.8em">
<div style="display:flex;align-items:flex-start;gap:.9em">
  <div style="background:#D4AF37;color:#07101E;font-weight:800;font-size:.8em;
  min-width:24px;height:24px;border-radius:50%;display:flex;align-items:center;
  justify-content:center">1</div>
  <div><b style="color:#EDF0F8">Enter tickers</b>
  <span style="color:#7A90B8;font-size:.88em"> — type comma-separated symbols in the sidebar
  (e.g. SPY, QQQ, GLD)</span></div>
</div>
<div style="display:flex;align-items:flex-start;gap:.9em">
  <div style="background:#D4AF37;color:#07101E;font-weight:800;font-size:.8em;
  min-width:24px;height:24px;border-radius:50%;display:flex;align-items:center;
  justify-content:center">2</div>
  <div><b style="color:#EDF0F8">Set period</b>
  <span style="color:#7A90B8;font-size:.88em"> — 2y is a good default for daily data</span></div>
</div>
<div style="display:flex;align-items:flex-start;gap:.9em">
  <div style="background:#D4AF37;color:#07101E;font-weight:800;font-size:.8em;
  min-width:24px;height:24px;border-radius:50%;display:flex;align-items:center;
  justify-content:center">3</div>
  <div><b style="color:#EDF0F8">Click ▶ Run Scan</b>
  <span style="color:#7A90B8;font-size:.88em"> — results appear here with severity tiers,
  sparklines, and confidence badges</span></div>
</div>
<div style="display:flex;align-items:flex-start;gap:.9em">
  <div style="background:#2A4070;color:#EDF0F8;font-weight:800;font-size:.8em;
  min-width:24px;height:24px;border-radius:50%;display:flex;align-items:center;
  justify-content:center">4</div>
  <div><b style="color:#EDF0F8">Explore flagged assets</b>
  <span style="color:#7A90B8;font-size:.88em"> — click any FAWP or HIGH asset to see
  the "Why flagged?" explanation card</span></div>
</div>
</div>
</div>
""", unsafe_allow_html=True)

        st.markdown(
            '<div style="text-align:center;margin-top:1em">'
            '<a href="https://github.com/DrRalphClayton/fawp-index" '
            'style="color:#3A4E70;font-size:.8em;text-decoration:none">'
            'Docs &nbsp;·&nbsp; GitHub &nbsp;·&nbsp; '
            'doi:10.5281/zenodo.18673949</a></div>',
            unsafe_allow_html=True,
        )
        st.stop()


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
        st.markdown(_empty_state("🔍","No assets match","Try switching to <b>All</b> in the filter, or run a scan with more tickers."), unsafe_allow_html=True)
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

            conf_html  = _confidence_html(a)
            st.markdown(
                f'<div class="{row_cls}">'
                f'<span class="asset-ticker">{a.ticker}</span>'
                f'<span class="asset-tf">{a.timeframe}</span>'
                f'{pill_html}'
                f'<span class="{score_cls}" style="min-width:64px">{a.latest_score:.4f}</span>'
                f'{spark_html}'
                f'{conf_html}'
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
        st.markdown(_empty_state("📈","No scan data","Run a scan first, then explore MI curves here.","→ Go to Scanner tab"), unsafe_allow_html=True)
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
# ──────────────────────────────────────────────────────────────────────────
# Tab 5 — Validation
# ──────────────────────────────────────────────────────────────────────────
with tab_validation:
    st.markdown(_sec("Forward-return validation"), unsafe_allow_html=True)
    st.caption(
        "After a FAWP signal fires, what happens to price? "
        "Select an asset and provide price data to compute forward returns "
        "at multiple horizons."
    )

    valid_v = [a for a in ranked if not a.error and a.scan is not None]
    if not valid_v:
        st.warning("No valid scans.")
    else:
        col_v1, col_v2 = st.columns([2, 1])
        with col_v1:
            val_labels = [f"{a.ticker} ({a.timeframe})" for a in valid_v]
            sel_val_lbl = st.selectbox("Asset", val_labels, key="val_asset")
            sel_val_a   = valid_v[val_labels.index(sel_val_lbl)]
        with col_v2:
            val_horizons = st.multiselect(
                "Horizons (bars)",
                [1, 5, 10, 20, 40, 60],
                default=[1, 5, 10, 20],
                key="val_horizons",
            )

        run_val = st.button("Run validation", type="primary", key="run_val")

        if run_val or "val_report" in st.session_state:
            if run_val:
                # Rebuild price series from the asset's scan dates + latest score
                # We derive approximate prices from the loaded dfs dict
                ticker_key = sel_val_a.ticker
                tf_key     = sel_val_a.timeframe
                if ticker_key in dfs:
                    df_raw = dfs[ticker_key]
                    close_col = [c for c in df_raw.columns
                                 if "close" in str(c).lower() or c == "Close"]
                    if close_col:
                        prices = df_raw[close_col[0]].squeeze()
                        with st.spinner("Computing forward returns…"):
                            report = validate_signals(
                                sel_val_a, prices,
                                horizons=sorted(val_horizons) if val_horizons else [1,5,10,20],
                            )
                        st.session_state["val_report"] = report
                    else:
                        st.warning("No Close column found in loaded data.")
                else:
                    st.warning(
                        f"Price data for {ticker_key} not found. "
                        "Load data via 'Enter tickers' or 'Upload CSV' first."
                    )

            if "val_report" in st.session_state:
                rpt = st.session_state["val_report"]
                if rpt.n_signals < 3:
                    st.info(
                        f"Only {rpt.n_signals} FAWP signals found — "
                        "need at least 3 for meaningful statistics. "
                        "Try a longer period or lower epsilon."
                    )
                else:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("FAWP signals", rpt.n_signals)
                    c2.metric("Price bars",   rpt.n_prices)
                    if rpt.horizons:
                        best = max(rpt.horizons, key=lambda h: h.hit_rate)
                        c3.metric("Best hit rate",
                                  f"{best.hit_rate*100:.1f}% @ {best.horizon}b")

                    if rpt.horizons:
                        st.markdown(_sec("Forward-return statistics"), unsafe_allow_html=True)
                        rows_html = ""
                        for h in rpt.horizons:
                            ret_cls  = "val-pos" if h.mean_return > 0 else "val-neg"
                            hit_cls  = "val-pos" if h.hit_rate >= 0.5 else "val-neg"
                            mae_cls  = "val-neg"
                            mfe_cls  = "val-pos"
                            rows_html += (
                                f"<tr>"
                                f"<td><b>{h.horizon}</b></td>"
                                f"<td>{h.n_signals}</td>"
                                f"<td class='{ret_cls}'>{h.mean_return*100:+.2f}%</td>"
                                f"<td>{h.median_return*100:+.2f}%</td>"
                                f"<td class='{hit_cls}'>{h.hit_rate*100:.1f}%</td>"
                                f"<td class='{mae_cls}'>{h.mae*100:.2f}%</td>"
                                f"<td class='{mfe_cls}'>{h.mfe*100:.2f}%</td>"
                                f"<td>{h.std_return*100:.2f}%</td>"
                                f"</tr>"
                            )
                        st.markdown(
                            "<table class='val-table'><thead><tr>"
                            "<th>Horizon</th><th>N</th><th>Mean ret</th>"
                            "<th>Median</th><th>Hit rate</th>"
                            "<th>Avg MAE</th><th>Avg MFE</th><th>Std dev</th>"
                            f"</tr></thead><tbody>{rows_html}</tbody></table>",
                            unsafe_allow_html=True,
                        )

                    if rpt.regime_mean_return and rpt.baseline_mean_return:
                        st.markdown(_sec("FAWP vs baseline"), unsafe_allow_html=True)
                        comp_rows = ""
                        for hz in sorted(rpt.regime_mean_return):
                            reg  = rpt.regime_mean_return.get(hz, 0.0)
                            base = rpt.baseline_mean_return.get(hz, 0.0)
                            diff = reg - base
                            diff_cls = "val-pos" if diff > 0 else "val-neg"
                            comp_rows += (
                                f"<tr>"
                                f"<td><b>{hz}</b></td>"
                                f"<td>{reg*100:+.2f}%</td>"
                                f"<td>{base*100:+.2f}%</td>"
                                f"<td class='{diff_cls}'>{diff*100:+.2f}%</td>"
                                f"</tr>"
                            )
                        st.markdown(
                            "<table class='val-table'><thead><tr>"
                            "<th>Horizon</th><th>FAWP mean</th>"
                            "<th>Baseline mean</th><th>Difference</th>"
                            f"</tr></thead><tbody>{comp_rows}</tbody></table>",
                            unsafe_allow_html=True,
                        )

                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        st.download_button(
                            "Download validation.csv",
                            data=rpt.to_csv(
                                __import__("tempfile").NamedTemporaryFile(
                                    suffix=".csv", delete=False
                                ).name
                            ).read_bytes() if False else
                            __import__("pandas").DataFrame(
                                [h.to_dict() for h in rpt.horizons]
                            ).to_csv(index=False).encode(),
                            file_name=f"fawp_validation_{rpt.ticker}.csv",
                            mime="text/csv",
                        )
                    with col_dl2:
                        import tempfile as _tf2
                        with _tf2.NamedTemporaryFile(suffix=".html", delete=False) as _tf3:
                            rpt.to_html(_tf3.name)
                            _val_html = __import__("pathlib").Path(_tf3.name).read_bytes()
                        st.download_button(
                            "Download validation.html",
                            data=_val_html,
                            file_name=f"fawp_validation_{rpt.ticker}.html",
                            mime="text/html",
                        )


# ──────────────────────────────────────────────────────────────────────────
# Tab 6 — History
# ──────────────────────────────────────────────────────────────────────────
with tab_history:
    st.markdown(_sec("Scan history"), unsafe_allow_html=True)
    st.caption(
        "Every scan is automatically saved. "
        "Select an asset to see how its score and regime state evolved."
    )
    try:
        hist = get_store()
        n_snaps = hist.n_snapshots()
        st.info(f"{n_snaps} snapshots stored · {hist._dir}")

        all_assets_hist = hist.all_assets() if hasattr(hist, 'all_assets') else []
        if not all_assets_hist:
            st.markdown(_empty_state("🕐","No scan history yet","Every scan is saved here automatically. Run your first scan to start.","→ Scanner tab"), unsafe_allow_html=True)
        else:
            asset_options = [f"{a['ticker']} ({a['timeframe']})"
                             for a in all_assets_hist]
            sel_hist = st.selectbox("Asset", asset_options, key="hist_asset")
            sel_parts = sel_hist.replace(")", "").split(" (")
            hticker, htf = sel_parts[0], sel_parts[1]

            tl = hist.asset_timeline(hticker, htf)
            if tl.empty:
                st.markdown(_empty_state("📊","No history for this asset","Include it in your next scan to start tracking."), unsafe_allow_html=True)
            else:
                onset = hist.first_onset(hticker, htf)
                last  = hist.last_seen_active(hticker, htf)

                c1, c2, c3 = st.columns(3)
                c1.metric("Snapshots", len(tl))
                c2.metric("First onset", onset or "never")
                c3.metric("Last active", last or "never")

                st.markdown(_sec("Score timeline"), unsafe_allow_html=True)

                if HAS_MPL:
                    import matplotlib.pyplot as plt
                    fig, ax = _dark_fig(10, 2.8)
                    dates  = tl["scanned_at"].dt.strftime("%m-%d %H:%M").tolist()
                    scores = tl["latest_score"].tolist()
                    active = tl["regime_active"].tolist()
                    colors = ["#C0111A" if a else "#2A4070" for a in active]
                    ax.bar(range(len(scores)), scores, color=colors, alpha=0.85,
                           edgecolor="none")
                    tick_step = max(1, len(dates) // 8)
                    ax.set_xticks(range(0, len(dates), tick_step))
                    ax.set_xticklabels(dates[::tick_step], rotation=25,
                                       ha="right", fontsize=7, color="#7A90B8")
                    ax.set_ylabel("Score", fontsize=8, color="#7A90B8")
                    ax.set_ylim(0, max(max(scores) * 1.15, 0.01))
                    plt.tight_layout(pad=0.4)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                st.markdown(_sec("Recent snapshots"), unsafe_allow_html=True)
                recent_tl = tl.tail(20).iloc[::-1]
                for _, row in recent_tl.iterrows():
                    active_str = (
                        '<span class="pill pill-fawp">FAWP</span>'
                        if row["regime_active"]
                        else '<span class="pill pill-clear">Clear</span>'
                    )
                    st.markdown(
                        f'<div class="hist-row">'
                        f'<span class="hist-date">{str(row["scanned_at"])[:16]}</span>'
                        f'{active_str}'
                        f'<span class="hist-score">{row["latest_score"]:.4f}</span>'
                        f'<span style="color:var(--text-3);font-size:.8em">'
                        f'gap {row["peak_gap_bits"]:.4f}b</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                st.download_button(
                    "Download timeline CSV",
                    data=tl.to_csv(index=False).encode(),
                    file_name=f"fawp_history_{hticker}_{htf}.csv",
                    mime="text/csv",
                )
    except Exception as e:
        st.error(f"History unavailable: {e}")


# ──────────────────────────────────────────────────────────────────────────
# Compare tab
# ──────────────────────────────────────────────────────────────────────────
with tab_compare:
    st.markdown(_sec("FAWP vs classic signals"), unsafe_allow_html=True)
    st.caption(
        "Compare FAWP regime score against RSI, realised volatility, "
        "momentum and MA slope. Each row shows forward-return lift "
        "when the signal is in its extreme zone (top 20%)."
    )
    valid_cmp = [a for a in ranked if not a.error and a.scan is not None]
    if not valid_cmp:
        st.markdown(_empty_state(
            "📊", "No scan data",
            "Run a scan first, then come back here to compare signals.",
            "Go to Scanner tab"
        ), unsafe_allow_html=True)
    else:
        col_c1, col_c2 = st.columns([2, 1])
        with col_c1:
            cmp_labels = [f"{a.ticker} ({a.timeframe})" for a in valid_cmp]
            sel_cmp    = st.selectbox("Asset", cmp_labels, key="cmp_asset")
            sel_cmp_a  = valid_cmp[cmp_labels.index(sel_cmp)]
        with col_c2:
            run_cmp = st.button("Run comparison", type="primary", key="run_cmp")

        if run_cmp:
            ticker_cmp = sel_cmp_a.ticker
            if ticker_cmp in dfs:
                df_cmp    = dfs[ticker_cmp]
                close_col = [c for c in df_cmp.columns if "close" in str(c).lower()]
                if close_col:
                    with st.spinner("Computing comparisons…"):
                        cmp_rpt = compare_signals(sel_cmp_a, df_cmp[close_col[0]].squeeze())
                    st.session_state["cmp_report"] = cmp_rpt
                else:
                    st.warning("No Close column found.")
            else:
                st.warning(f"Price data for {ticker_cmp} not loaded.")

        if "cmp_report" in st.session_state:
            crpt = st.session_state["cmp_report"]
            st.markdown(_sec("Forward-return lift at extreme signal"), unsafe_allow_html=True)

            def _ret_cls(v):
                return "val-pos" if v > 0 else "val-neg"

            rows_html = (
                f"<tr style='border-left:3px solid #D4AF37'>"
                f"<td><b>FAWP score</b></td><td style='color:#7A90B8'>—</td>"
                f"<td class='{_ret_cls(crpt.fawp_fwd_return_1)}'>{crpt.fawp_fwd_return_1*100:+.2f}%</td>"
                f"<td class='{_ret_cls(crpt.fawp_fwd_return_5)}'>{crpt.fawp_fwd_return_5*100:+.2f}%</td>"
                f"<td class='{_ret_cls(crpt.fawp_fwd_return_20)}'>{crpt.fawp_fwd_return_20*100:+.2f}%</td>"
                f"<td class='{_ret_cls(crpt.fawp_hit_rate_20 - 0.5)}'>{crpt.fawp_hit_rate_20*100:.1f}%</td>"
                f"</tr>"
            )
            for s in crpt.signals:
                rows_html += (
                    f"<tr><td>{s.name}</td>"
                    f"<td style='font-family:monospace;color:#7A90B8'>{s.correlation:+.3f}</td>"
                    f"<td class='{_ret_cls(s.fwd_return_1)}'>{s.fwd_return_1*100:+.2f}%</td>"
                    f"<td class='{_ret_cls(s.fwd_return_5)}'>{s.fwd_return_5*100:+.2f}%</td>"
                    f"<td class='{_ret_cls(s.fwd_return_20)}'>{s.fwd_return_20*100:+.2f}%</td>"
                    f"<td class='{_ret_cls(s.hit_rate_20 - 0.5)}'>{s.hit_rate_20*100:.1f}%</td>"
                    f"</tr>"
                )
            st.markdown(
                "<table class='val-table'><thead><tr>"
                "<th>Signal</th><th>Corr(FAWP)</th>"
                "<th>Ret@1</th><th>Ret@5</th><th>Ret@20</th><th>Hit%@20</th>"
                f"</tr></thead><tbody>{rows_html}</tbody></table>",
                unsafe_allow_html=True,
            )

            if HAS_MPL and crpt.signals:
                import matplotlib.pyplot as plt
                labels = ["FAWP"] + [s.name for s in crpt.signals]
                ret20  = [crpt.fawp_fwd_return_20*100] + [s.fwd_return_20*100 for s in crpt.signals]
                hit20  = [crpt.fawp_hit_rate_20*100]   + [s.hit_rate_20*100   for s in crpt.signals]
                colors = ["#D4AF37"] + ["#2A4070"] * len(crpt.signals)
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.4))
                fig.patch.set_facecolor("#07101E")
                for ax, vals, title in [
                    (ax1, ret20, "20-bar fwd return (%)"),
                    (ax2, hit20, "Hit rate at 20 bars (%)"),
                ]:
                    ax.set_facecolor("#0D1729")
                    ax.bar(labels, vals, color=colors, alpha=0.85, edgecolor="none")
                    ax.axhline(0, color="#182540", lw=1)
                    ax.set_title(title, fontsize=8, color="#7A90B8", pad=4)
                    ax.tick_params(colors="#7A90B8", labelsize=7)
                    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=7)
                    for sp in ax.spines.values():
                        sp.set_edgecolor("#182540")
                plt.tight_layout(pad=0.5)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            st.download_button(
                "Download comparison CSV",
                data=crpt.to_dataframe().to_csv(index=False).encode(),
                file_name=f"fawp_compare_{crpt.ticker}.csv",
                mime="text/csv",
            )


# ──────────────────────────────────────────────────────────────────────────
# Tab: Weather FAWP
# ──────────────────────────────────────────────────────────────────────────
with tab_weather:
    st.markdown(_sec("Weather & Climate FAWP Detection"), unsafe_allow_html=True)
    st.caption(
        "Detect the Information-Control Exclusion Principle in atmospheric data. "
        "Uses ERA5 reanalysis via Open-Meteo — free, no API key needed."
    )

    _W_VARS = {
        "temperature_2m":             "Temperature 2m (°C)",
        "precipitation_sum":          "Precipitation (mm/day)",
        "wind_speed_10m":             "Wind Speed 10m (m/s)",
        "surface_pressure":           "Surface Pressure (hPa)",
        "cloud_cover":                "Cloud Cover (%)",
        "shortwave_radiation":        "Shortwave Radiation (W/m²)",
    }

    # Initialise keys so presets can write to them before inputs render
    if "w_lat_v" not in st.session_state:
        st.session_state["w_lat_v"] = 51.5
    if "w_lon_v" not in st.session_state:
        st.session_state["w_lon_v"] = -0.1

    # Quick presets — written BEFORE the inputs so the keys exist
    st.caption("Presets:")
    preset_cols = st.columns(5)
    presets = [
        ("London",   51.5,  -0.1),
        ("New York", 40.7, -74.0),
        ("Tokyo",    35.7, 139.7),
        ("Sydney",  -33.9, 151.2),
        ("Paris",    48.9,   2.4),
    ]
    for i, (name, lat, lon) in enumerate(presets):
        with preset_cols[i]:
            if st.button(name, key=f"preset_{name}"):
                st.session_state["w_lat_v"] = lat
                st.session_state["w_lon_v"] = lon
                st.rerun()

    col_w1, col_w2, col_w3 = st.columns(3)
    with col_w1:
        # Bind inputs to session_state keys so preset buttons actually update them
        w_lat = st.number_input("Latitude",  min_value=-90.0,  max_value=90.0,
                                step=0.1, format="%.2f", key="w_lat_v")
        w_lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0,
                                step=0.1, format="%.2f", key="w_lon_v")
        w_var = st.selectbox("Variable", list(_W_VARS.keys()),
                             format_func=lambda k: _W_VARS[k])
    with col_w2:
        w_start = st.text_input("Start date", "2015-01-01")
        w_end   = st.text_input("End date",   "2024-12-31")
        w_horiz = st.slider("Forecast horizon (days)", 1, 30, 7)
    with col_w3:
        w_tau   = st.slider("Max tau", 5, 60, 30, step=5)
        w_null  = st.slider("Null permutations", 0, 200, 50, step=10)
        w_seas  = st.checkbox("Remove seasonality", key="w_seasonal",
                              help="Subtract 365-day trend. Recommended for temperature.")
        run_weather = st.button("🌦 Run Weather Scan", type="primary",
                                use_container_width=True, key="run_weather")

    if run_weather:
        _w_deps_ok = True
        try:
            import openmeteo_requests  # noqa: F401
        except ImportError:
            st.error(
                "Open-Meteo client not installed.\n\n"
                "`pip install openmeteo-requests requests-cache retry-requests`"
            )
            _w_deps_ok = False

        if _w_deps_ok:
            with st.spinner(f"Fetching ERA5 {w_var} @ ({w_lat:.1f}, {w_lon:.1f})…"):
                try:
                    w_result = fawp_from_open_meteo(
                        latitude     = w_lat,
                        longitude    = w_lon,
                        variable     = w_var,
                        start_date   = w_start,
                        end_date     = w_end,
                        horizon_days = w_horiz,
                        tau_max      = w_tau,
                        n_null       = w_null,
                    )
                    st.session_state["w_result"] = w_result
                except Exception as we:
                    st.error(f"Weather scan failed: {we}")

    if "w_result" in st.session_state:
        wr = st.session_state["w_result"]
        # KPI row
        kc1, kc2, kc3, kc4 = st.columns(4)
        kc1.metric("FAWP", "🔴 YES" if wr.fawp_found else "✅ NO")
        kc2.metric("Peak gap", f"{wr.peak_gap_bits:.4f} bits")
        kc3.metric("ODW",
                   f"τ {wr.odw_start}–{wr.odw_end}" if wr.fawp_found else "—")
        kc4.metric("Observations", f"{wr.n_obs:,}")

        if wr.fawp_found:
            st.success(
                f"**FAWP detected** — {_W_VARS.get(wr.variable, wr.variable)} "
                f"at {wr.location}. "
                f"Predictive coupling persists (τ⁺ₕ={wr.odw_result.tau_h_plus}) "
                f"while steering has collapsed. "
                f"ODW: τ = {wr.odw_start}–{wr.odw_end}."
            )
        else:
            st.info("No FAWP regime detected at this location/variable/period.")

        # MI curves chart
        if HAS_MPL and len(wr.tau) > 0:
            import matplotlib.pyplot as plt
            fig, ax = _dark_fig(10, 3.2)
            ax.plot(wr.tau, wr.pred_mi,  color="#D4AF37", lw=1.8, label="Pred MI (forecast skill)")
            ax.plot(wr.tau, wr.steer_mi, color="#4A7FCC", lw=1.5, ls="--", label="Steer MI (intervention)")
            ax.axhline(_EPS_STEER, color="#3A4E70", ls=":", lw=1, label=f"ε = {EPSILON_STEERING_RAW}")
            if wr.fawp_found and wr.odw_start:
                ax.axvspan(wr.odw_start, wr.odw_end, alpha=0.15, color="#C0111A", label="ODW")
            ax.set_xlabel("τ (delay)", fontsize=8, color="#7A90B8")
            ax.set_ylabel("MI (bits)", fontsize=8, color="#7A90B8")
            ax.legend(fontsize=7, facecolor="#0D1729", labelcolor="#EDF0F8",
                      edgecolor="#182540")
            plt.tight_layout(pad=0.4)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        # Download
        import json as _json
        st.download_button(
            "Download result JSON",
            data=_json.dumps(wr.to_dict(), indent=2).encode(),
            file_name=f"fawp_weather_{wr.variable}_{wr.location.replace(' ','_')}.json",
            mime="application/json",
        )
    elif not run_weather:
        st.markdown(_empty_state(
            "🌦", "No weather scan yet",
            "Enter coordinates, pick a variable and date range,<br>then click Run Weather Scan.",
            "Supports ERA5 data for any location on Earth"
        ), unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────
# Admin tab — manage users and plans
# ──────────────────────────────────────────────────────────────────────────
with tab_admin:
    if not _AUTH_ENABLED or not is_admin():
        st.info("Admin access required.")
    else:
        st.markdown(_sec("Admin Panel"), unsafe_allow_html=True)
        st.caption("Manage user plans. Only visible to admin accounts.")

        # ── Upgrade a user ────────────────────────────────────────────────
        st.markdown(_sec("Set user plan"), unsafe_allow_html=True)
        with st.form("admin_set_plan"):
            _admin_email = st.text_input("User email")
            _admin_plan  = st.selectbox("Plan", ["free", "pro", "admin"])
            _admin_sub   = st.form_submit_button("Update plan", type="primary")
        if _admin_sub and _admin_email:
            try:
                import os as _os
                from supabase import create_client as _cc
                _surl = _os.environ.get("SUPABASE_URL", "")
                _skey = _os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
                _adm  = _cc(_surl, _skey)
                # Find user by email via admin API
                _users = _adm.auth.admin.list_users()
                _uid   = None
                for _u in _users:
                    if getattr(_u, "email", "") == _admin_email.strip():
                        _uid = getattr(_u, "id", None)
                        break
                if _uid:
                    _adm.table("profiles").upsert({
                        "id":    _uid,
                        "email": _admin_email.strip(),
                        "plan":  _admin_plan,
                    }).execute()
                    st.success(f"Updated {_admin_email} → {_admin_plan}")
                else:
                    st.error(f"User not found: {_admin_email}")
            except Exception as _ae:
                st.error(f"Admin action failed: {_ae}")

        # ── User list ─────────────────────────────────────────────────────
        st.markdown(_sec("All users"), unsafe_allow_html=True)
        try:
            import os as _os2
            from supabase import create_client as _cc2
            _surl2 = _os2.environ.get("SUPABASE_URL", "")
            _skey2 = _os2.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
            if _surl2 and _skey2:
                _adm2 = _cc2(_surl2, _skey2)
                _pres = _adm2.table("profiles").select("email, plan, created_at").execute()
                _rows = _pres.data or []
                if _rows:
                    _rows_html = "".join(
                        f"<tr><td>{r.get('email','?')}</td>"
                        f"<td style='color:#D4AF37;font-weight:600'>{r.get('plan','free').upper()}</td>"
                        f"<td style='color:#3A4E70'>{str(r.get('created_at',''))[:10]}</td></tr>"
                        for r in _rows
                    )
                    st.markdown(
                        "<table class='val-table'><thead><tr>"
                        "<th>Email</th><th>Plan</th><th>Joined</th>"
                        f"</tr></thead><tbody>{_rows_html}</tbody></table>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption("No users yet.")
        except Exception as _ue:
            st.caption(f"Could not load users: {_ue}")


with tab_export:
    # ── HTML report ──────────────────────────────────────────────────────
    st.markdown(_sec("Download report"), unsafe_allow_html=True)
    if wl:
        _scan_ts = st.session_state.get("scan_timestamp", "scan")
        if st.button("Generate HTML report", key="gen_html_report"):
            with st.spinner("Building report…"):
                _html = generate_html_report(
                    wl, title=f"FAWP Finance Scan — {_scan_ts}")
            st.download_button(
                "📄 Download HTML report",
                data=_html.encode(),
                file_name=f"fawp_scan_{_scan_ts.replace(' ','_')}.html",
                mime="text/html",
                key="dl_html_report",
            )
    st.markdown("---")
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
