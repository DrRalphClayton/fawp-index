"""
FAWP Dashboard v3.7.27 — Streamlit app
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
# "finance" | "weather" | "seismic" | None (landing)
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

    # ── Live stats ─────────────────────────────────────────────
    @st.cache_data(ttl=7200)
    def _check_update():
        try:
            import urllib.request, json as _jv, fawp_index as _fiv
            with urllib.request.urlopen(
                "https://pypi.org/pypi/fawp-index/json", timeout=2
            ) as r:
                _lat = _jv.loads(r.read())["info"]["version"]
            _cv  = tuple(int(x) for x in _fiv.__version__.split(".")[:3])
            _ltv = tuple(int(x) for x in _lat.split(".")[:3])
            return _lat if _ltv > _cv else None
        except Exception:
            return None
    _upd = _check_update()
    if _upd:
        st.sidebar.info(f"\u2b06 v{_upd} available  \u2014  `pip install --upgrade fawp-index`")

    @st.cache_data(ttl=3600)
    def _github_stars():
        try:
            import urllib.request, json as _j
            with urllib.request.urlopen(
                "https://api.github.com/repos/DrRalphClayton/fawp-index", timeout=3) as r:
                return _j.loads(r.read()).get("stargazers_count", 0)
        except Exception:
            return None
    _stars = _github_stars()
    _stars_str = str(_stars) if _stars else "★"

    @st.cache_data(ttl=3600)
    def _pypi_downloads():
        try:
            import urllib.request, json as _j
            url = "https://pypistats.org/api/packages/fawp-index/recent"
            with urllib.request.urlopen(url, timeout=3) as r:
                d = _j.loads(r.read())
            return d.get("data", {}).get("last_month", 0)
        except Exception:
            return None
    _dl = _pypi_downloads()
    _dl_str = f"{_dl:,}/month" if _dl else "1.5k+/month"
    @st.cache_data(ttl=300)
    def _today_scan_count():
        try:
            _store_sc = get_store()
            if hasattr(_store_sc, "_db") and _store_sc._db is not None:
                import pandas as _pd_sc
                _today_sc = _pd_sc.Timestamp.now().strftime("%Y-%m-%d")
                _res_sc = (_store_sc._db.table("fawp_scan_history")
                           .select("id", count="exact")
                           .gte("scanned_at", _today_sc)
                           .execute())
                return _res_sc.count or 0
        except Exception:
            pass
        return None
    _sc = _today_scan_count()
    _scan_count_str = f"{_sc:,}" if _sc is not None else "—"
    st.markdown(
        f'<div style="display:flex;justify-content:center;gap:2.5em;margin:-1em 0 2em;flex-wrap:wrap">'  
        f'<div style="text-align:center"><div style="font-size:1.5em;font-weight:800;color:#D4AF37">{_dl_str}</div>'
        f'<div style="font-size:.72em;color:#3A4E70;text-transform:uppercase">PyPI downloads</div></div>'
        f'<div style="text-align:center"><div style="font-size:1.5em;font-weight:800;color:#D4AF37">4</div>'
        f'<div style="font-size:.72em;color:#3A4E70;text-transform:uppercase">Live scanners</div></div>'
        f'<div style="text-align:center"><div style="font-size:1.5em;font-weight:800;color:#D4AF37">{_stars_str}</div>'
        f'<div style="font-size:.72em;color:#3A4E70;text-transform:uppercase">GitHub stars</div></div>'
        f'<div style="text-align:center"><div style="font-size:1.5em;font-weight:800;color:#D4AF37">{_scan_count_str}</div>'
        f'<div style="font-size:.72em;color:#3A4E70;text-transform:uppercase">Scans run today</div></div>'
        f'<div style="text-align:center"><div style="font-size:1.5em;font-weight:800;color:#D4AF37">4,244</div>'
        f'<div style="font-size:.72em;color:#3A4E70;text-transform:uppercase">E9 validation runs</div></div>'
        f'<div style="text-align:center"><div style="font-size:1.5em;font-weight:800;color:#D4AF37">2.234 bits</div>'
        f'<div style="font-size:.72em;color:#3A4E70;text-transform:uppercase">SPHERE-16 peak MI</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    col_l, col_fin, col_gap, col_wx, col_gap2, col_seis, col_gap3, col_ctrl, col_r = st.columns([0.2, 3, 0.25, 3, 0.25, 3, 0.25, 3, 0.2])

    with col_fin:
        st.markdown("""
<div class="mode-card">
  <div class="mode-icon">📈</div>
  <div class="mode-name">FAWP Finance</div>
  <svg viewBox="0 0 120 40" style="width:100%;height:40px;margin:.4em 0"><path d="M0,35 C15,30 20,5 35,4 C50,3 55,20 70,22 C85,24 95,32 120,35" fill="none" stroke="#D4AF37" stroke-width="2"/><path d="M0,38 C20,36 40,35 60,37 C80,39 100,38 120,38" fill="none" stroke="#4A7FCC" stroke-width="1.2" stroke-dasharray="3,2"/><rect x="28" y="0" width="50" height="40" fill="#C0111A" opacity="0.12"/></svg>
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
  <svg viewBox="0 0 120 40" style="width:100%;height:40px;margin:.4em 0"><path d="M0,36 C10,35 20,10 35,8 C50,6 55,18 70,20 C85,22 100,30 120,36" fill="none" stroke="#D4AF37" stroke-width="2"/><path d="M0,38 C15,34 30,33 50,35 C70,37 100,37 120,38" fill="none" stroke="#4A7FCC" stroke-width="1.2" stroke-dasharray="3,2"/><rect x="32" y="0" width="42" height="40" fill="#C0111A" opacity="0.12"/></svg>
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

    with col_seis:
        st.markdown("""
<div class="mode-card">
  <div class="mode-icon">🌍</div>
  <div class="mode-name">FAWP Seismic</div>
  <svg viewBox="0 0 120 40" style="width:100%;height:40px;margin:.4em 0"><path d="M0,30 C8,28 12,38 16,20 C20,5 24,32 32,28 C40,24 46,12 56,10 C66,8 72,22 82,20 C92,18 102,30 120,36" fill="none" stroke="#D4AF37" stroke-width="2"/><path d="M0,38 C20,36 45,35 65,37 C85,39 105,38 120,38" fill="none" stroke="#4A7FCC" stroke-width="1.2" stroke-dasharray="3,2"/><rect x="42" y="0" width="38" height="40" fill="#C0111A" opacity="0.12"/></svg>
  <div class="mode-desc">
    Scan USGS earthquake catalogs for any region.<br>
    Detect when seismic activity is forecastable but the
    intervention window has collapsed. Free · no API key needed.
  </div>
</div>
""", unsafe_allow_html=True)
        if st.button("▶  Open Seismic Scanner",
                     use_container_width=True, type="primary",
                     key="goto_seismic"):
            st.session_state["_app_mode"] = "seismic"
            st.rerun()

    with col_ctrl:
        st.markdown("""
<div class="mode-card">
  <div class="mode-icon">⚙️</div>
  <div class="mode-name">FAWP Dynamic Systems</div>
  <svg viewBox="0 0 120 40" style="width:100%;height:40px;margin:.4em 0"><path d="M0,38 C8,38 12,38 16,38 C20,38 20,8 24,8 C28,8 28,28 32,28 C36,28 36,18 44,18 C52,18 52,32 58,32 C64,32 64,14 70,14 C76,14 76,24 82,24 C88,24 88,20 94,20 C100,20 106,38 120,38" fill="none" stroke="#D4AF37" stroke-width="2"/><path d="M0,38 C20,36 40,35 60,37 C80,39 100,38 120,38" fill="none" stroke="#4A7FCC" stroke-width="1.2" stroke-dasharray="3,2"/><rect x="42" y="0" width="38" height="40" fill="#C0111A" opacity="0.12"/></svg>
  <div class="mode-desc">
    Upload any state + action CSV.<br>
    Control systems, ML training logs, Kalman filters —
    detect when prediction persists but steering has collapsed.
  </div>
</div>
""", unsafe_allow_html=True)
        if st.button("▶  Open Dynamo Scanner",
                     use_container_width=True, type="primary",
                     key="goto_control"):
            st.session_state["_app_mode"] = "control"
            st.rerun()

    # Footer — resolve version before embedding in markdown
    try:
        import fawp_index as _fi_ver
        _APP_VER = _fi_ver.__version__
    except Exception:
        _APP_VER = "?"
    st.markdown(
        f'<div style="text-align:center;margin-top:3em;'
        f'color:#1E2E4A;font-size:.78em">'
        f'fawp-index v{_APP_VER} · Ralph Clayton · 2026 · '
        f'<a href="https://github.com/DrRalphClayton/fawp-index"'
        f'   style="color:#1E2E4A">GitHub</a> · '
        f'<a href="https://pypi.org/project/fawp-index/"'
        f'   style="color:#1E2E4A">PyPI</a></div>',
        unsafe_allow_html=True
    )
    st.stop()

# ── Top nav bar — Home + direct app switcher ───────────────────────────────────
_NAV_CSS = """
<style>
[data-theme="light"] {
    --bg-app: #F5F7FA; --bg-card: #FFFFFF; --bg-card2: #EEF1F7;
    --accent: #B8922A; --text-1: #0D1729; --text-3: #5A6E8A;
    --crimson: #C0111A; --green: #1A8C42; --blue-mild: #2A5FA8;
    --muted: #8A9AB8;
}

.fawp-nav {
    display:flex; align-items:center; gap:.6em;
    padding:.5em 0 .8em; border-bottom:1px solid #182540;
    margin-bottom:1em;
}
.nav-home {
    font-family:'Syne',sans-serif; font-weight:800; font-size:1.05em;
    color:#D4AF37; text-decoration:none; margin-right:.4em;
    letter-spacing:-.01em;
}
.nav-sep { color:#182540; font-size:1.1em; }
.nav-btn {
    background:none; border:1px solid #182540; border-radius:7px;
    padding:.28em .85em; font-size:.8em; font-weight:600;
    cursor:pointer; transition:border-color .15s, color .15s;
    font-family:'DM Sans',sans-serif; letter-spacing:.01em;
}
.nav-btn:hover { border-color:#D4AF37; }
.nav-btn-active {
    border-color:#D4AF37 !important; color:#D4AF37 !important;
    background:#0D1729 !important;
}
</style>
"""
st.markdown(_NAV_CSS, unsafe_allow_html=True)

def _switch_mode(new_mode):
    """Clear current mode state and switch to new_mode."""
    for _k in ["wl_result","input_dfs","_fetched_key","wx_result","wx_hazard",
               "seis_result","seis_raw","seis_daily","dynamo_result"]:
        st.session_state.pop(_k, None)
    if new_mode is None:
        st.session_state.pop("_app_mode", None)
    else:
        st.session_state["_app_mode"] = new_mode
    st.rerun()

# Render nav row: FAWP home | 📈 Finance | 🌦 Weather | 🌍 Seismic
_nav_home, _nav_fin, _nav_wx, _nav_seis, _nav_ctrl = st.columns([2, 2, 2, 2, 2])
with _nav_home:
    if st.button("⚡ FAWP", key="nav_home",
                 help="Back to launcher",
                 use_container_width=True):
        _switch_mode(None)
with _nav_fin:
    _fin_type = "primary" if _APP_MODE == "finance" else "secondary"
    if st.button("📈 Finance", key="nav_finance",
                 type=_fin_type, use_container_width=True,
                 disabled=(_APP_MODE == "finance")):
        _switch_mode("finance")
with _nav_wx:
    _wx_type = "primary" if _APP_MODE == "weather" else "secondary"
    if st.button("🌦 Weather", key="nav_weather",
                 type=_wx_type, use_container_width=True,
                 disabled=(_APP_MODE == "weather")):
        _switch_mode("weather")
with _nav_seis:
    _seis_type = "primary" if _APP_MODE == "seismic" else "secondary"
    if st.button("🌍 Seismic", key="nav_seismic",
                 type=_seis_type, use_container_width=True,
                 disabled=(_APP_MODE == "seismic")):
        _switch_mode("seismic")
with _nav_ctrl:
    _ctrl_type = "primary" if _APP_MODE == "control" else "secondary"
    if st.button("⚙️ Dynamo", key="nav_control",
                 type=_ctrl_type, use_container_width=True,
                 disabled=(_APP_MODE == "control")):
        _switch_mode("control")

# Dark/light mode toggle (persists in session)
# Dark/light — persist via URL query param so it survives page reload
_qp = st.query_params
# Persist dark/light preference: on load, read theme from URL param
if "theme" not in st.session_state:
    _saved_theme = _qp.get("theme", "dark")
    if _saved_theme not in ("dark", "light"): _saved_theme = "dark"
    st.session_state["theme"] = _saved_theme
else:
    # Write session theme back to URL so F5 preserves it
    _qp["theme"] = st.session_state["theme"]
_theme_icon = "☀️" if st.session_state["theme"] == "dark" else "🌙"
if st.sidebar.button(f"{_theme_icon} Toggle theme", key="theme_toggle",
                     use_container_width=True):
    _new_theme = "light" if st.session_state["theme"] == "dark" else "dark"
    st.session_state["theme"] = _new_theme
    st.query_params["theme"] = _new_theme
    st.rerun()
if st.session_state.get("theme", "dark") != _qp.get("theme", "dark"):
    st.query_params["theme"] = st.session_state.get("theme", "dark")
if st.session_state["theme"] == "light":
    st.markdown(
        "<script>document.documentElement.setAttribute('data-theme','light')</script>",
        unsafe_allow_html=True,
    )

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

if _APP_MODE == "control":
    import importlib.util as _ilu3, os as _os4
    _ctrl_path = _os4.path.join(_THIS_DIR, "control_app.py")
    _spec3     = _ilu3.spec_from_file_location("control_app", _ctrl_path)
    _ctrlmod   = _ilu3.module_from_spec(_spec3)
    try:
        _spec3.loader.exec_module(_ctrlmod)
    except SystemExit:
        pass
    st.stop()

if _APP_MODE == "seismic":
    import importlib.util as _ilu2, os as _os3
    _seis_path = _os3.path.join(_THIS_DIR, "seismic_app.py")
    _spec2     = _ilu2.spec_from_file_location("seismic_app", _seis_path)
    _seismod   = _ilu2.module_from_spec(_spec2)
    try:
        _spec2.loader.exec_module(_seismod)
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
.pill-wall   { background: rgba(255,107,43,0.15); color: #FF6B2B; border: 1px solid rgba(255,107,43,0.35); }
.pill-resid  { background: rgba(74,127,204,0.15); color: #4A7FCC; border: 1px solid rgba(74,127,204,0.35); }
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
.asset-row { background: var(--bg-card); border: 1px solid var(--border); border-radius: 6px; padding: 0.7em 1em; margin-bottom: 5px; display: flex; align-items: center; gap: 14px; font-size: 0.85em; transition: border-color 0.15s; flex-wrap: wrap; }
@media (max-width: 768px) {
  .asset-row { flex-direction: column; align-items: flex-start; gap: 6px; }
  .kpi-card  { min-width: 100px; }
  .mode-card { padding: 1em .8em; }
  .mode-name { font-size: 1em; }
}
.asset-row:hover { border-color: var(--border-2); }
.asset-row.fawp-active { border-left: 3px solid var(--crimson); }
.asset-row.high-risk   { border-left: 3px solid var(--accent); }
.asset-row.sev-critical{ border-left: 4px solid #8B0000; background: rgba(139,0,0,0.12); }
.asset-row.sev-high    { border-left: 4px solid #C0111A; background: rgba(192,17,26,0.09); }
.asset-row.sev-medium  { border-left: 4px solid #FF6B2B; background: rgba(255,107,43,0.07); }
.asset-row.sev-low     { border-left: 3px solid #D4AF37; background: rgba(212,175,55,0.05); }
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

# Mobile sidebar: auto-collapse on narrow viewports
st.components.v1.html(
    "<script>(function(){"
    "function _cb(){if(window.innerWidth<=768){"
    "var b=window.parent.document.querySelector('[data-testid=\"collapsedControl\"]');"
    "if(b&&b.getAttribute('aria-expanded')!=='false')b.click();}}"
    "if(document.readyState==='complete')_cb();"
    "else window.addEventListener('load',_cb);"
    "})();</script>",
    height=0, scrolling=False
)

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
        try:
            import fawp_index as _fi_p
            _sm = float(getattr(a, 'steer_mi_mean', getattr(getattr(a,'odw_result',None),'steer_mi_at_h',0) or 0))
            if _sm <= _fi_p.ALPHA_A_SQ:
                return '<span class="pill pill-fawp">FAWP · null</span>'
            elif _sm <= _fi_p.ALPHA_A:
                return '<span class="pill pill-resid">FAWP · residual</span>'
        except Exception:
            pass
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
    alert_threshold = st.slider("Default gap threshold (bits)", 0.0, 1.0, 0.05, 0.01,
                               key="alert_threshold_global")
    # Per-ticker thresholds — expand to customise
    with st.sidebar.expander("⚙ Per-ticker thresholds", expanded=False):
        st.caption("Override threshold per ticker. Leave blank = use default.")
        if "ticker_thresholds" not in st.session_state:
            st.session_state["ticker_thresholds"] = {}
        _tt_input = st.text_area(
            "ticker=threshold (one per line, e.g. SPY=0.05)",
            value="\n".join(f"{k}={v}" for k,v in
                           st.session_state["ticker_thresholds"].items()),
            height=100, key="tt_input")
        if _tt_input:
            _tt_parsed = {}
            for _ttl in _tt_input.strip().splitlines():
                _tts = _ttl.strip().split("=")
                if len(_tts) == 2:
                    try: _tt_parsed[_tts[0].strip().upper()] = float(_tts[1])
                    except ValueError: pass
            st.session_state["ticker_thresholds"] = _tt_parsed

    st.markdown("<br>", unsafe_allow_html=True)
    # Auto-scheduler
    import time as _time_sched
    _auto_on   = st.sidebar.toggle("⏱ Auto-rescan", key="auto_rescan_on",
                                   help="Automatically re-run scan on a timer")
    if _auto_on:
        _auto_mins = st.sidebar.slider("Rescan every (min)", 1, 60, 5, key="auto_rescan_mins")
        _last_auto = st.session_state.get("_last_auto_scan_ts", 0)
        _secs_left = max(0, int(_auto_mins * 60 - (_time_sched.time() - _last_auto)))
        if _secs_left > 0:
            st.sidebar.caption(f"Next rescan in {_secs_left//60}m {_secs_left%60}s")
        else:
            st.session_state["_trigger_auto_scan"] = True

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

    # Webhook alerts config (Slack / Discord)
    with st.sidebar.expander("🔔 Webhook alerts", expanded=False):
        _wh_slack   = st.text_input("Slack webhook URL",
                                    value=st.session_state.get("_slack_url",""),
                                    type="password", key="slack_url_input",
                                    placeholder="https://hooks.slack.com/services/…")
        _wh_discord = st.text_input("Discord webhook URL",
                                    value=st.session_state.get("_discord_url",""),
                                    type="password", key="discord_url_input",
                                    placeholder="https://discord.com/api/webhooks/…")
        if st.button("Save webhook settings", key="save_webhooks",
                     use_container_width=True):
            st.session_state["_slack_url"]   = _wh_slack
            st.session_state["_discord_url"] = _wh_discord
            st.success("Saved — alerts will fire on next FAWP detection.")
        _wh_test = st.button("Send test alert", key="test_webhook",
                             use_container_width=True)
        if _wh_test:
            try:
                from fawp_index.alerts import AlertEngine, FAWPAlert
                _eng = AlertEngine()
                if st.session_state.get("_slack_url"):
                    _eng.add_slack(st.session_state["_slack_url"])
                if st.session_state.get("_discord_url"):
                    _eng.add_discord(st.session_state["_discord_url"])
                if not _eng._backends:
                    st.warning("No webhook URLs saved yet.")
                else:
                    _eng.fire(FAWPAlert(
                        ticker="TEST", timeframe="1d", score=0.042,
                        gap_bits=0.042, odw_start=28, odw_end=34,
                        alert_type="fawp_detected",
                    ))
                    st.success("Test alert sent!")
            except Exception as _we:
                st.error(f"Webhook test failed: {_we}")
    # Auto-fire webhooks when scan completes with FAWP active
    if (st.session_state.get("_slack_url") or st.session_state.get("_discord_url")):
        if "wl_result" in st.session_state:
            _wl = st.session_state["wl_result"]
            _wh_fired_key = f"wh_fired_{id(_wl)}"
            if not st.session_state.get(_wh_fired_key) and _wl.n_flagged > 0:
                try:
                    from fawp_index.alerts import AlertEngine, FAWPAlert
                    _eng2 = AlertEngine()
                    if st.session_state.get("_slack_url"):
                        _eng2.add_slack(st.session_state["_slack_url"])
                    if st.session_state.get("_discord_url"):
                        _eng2.add_discord(st.session_state["_discord_url"])
                    for _a in _wl.active_regimes()[:5]:
                        _eng2.fire(FAWPAlert(
                            ticker=_a.ticker, timeframe=_a.timeframe,
                            score=_a.latest_score, gap_bits=_a.peak_gap_bits,
                            odw_start=_a.peak_odw_start, odw_end=_a.peak_odw_end,
                            alert_type="fawp_detected",
                        ))
                    st.session_state[_wh_fired_key] = True
                except Exception:
                    pass


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
    st.checkbox("Auto re-scan when file changes", key="csv_auto_refresh")
    if uploaded:
        _nh = hash(tuple(f.getvalue() for f in uploaded))
        _oh = st.session_state.get("_csv_hash")
        if _nh != _oh:
            st.session_state["input_dfs"] = _load_uploaded(uploaded)
            st.session_state["_csv_hash"] = _nh
            if st.session_state.get("csv_auto_refresh") and _oh is not None:
                st.toast("📂 File changed — re-scanning…", icon="📂")
                st.session_state.pop("wl_result", None)
                st.rerun()
    dfs = st.session_state.get("input_dfs", {})

elif source == "Enter tickers (yfinance)":
    col1, col2 = st.columns([3, 1])
    with col1:
        _default_tickers = _DEMO_TICKERS.replace(",", ", ") if _DEMO_TICKERS else "SPY, QQQ, GLD"
        _max_t = get_limit("max_tickers") or 999
        if not is_pro() and _AUTH_ENABLED:
            st.caption(f"Free plan: up to {_max_t} tickers")
        # Preset portfolio buttons
        _PRESETS = {
            "📈 S&P core":   "SPY, QQQ, IWM, DIA",
            "₿ Crypto":      "BTC-USD, ETH-USD, SOL-USD, BNB-USD",
            "🥇 Commodities":"GLD, SLV, USO, CORN",
            "🌍 Global ETF": "EFA, EEM, VEU, ACWI",
            "🏦 Sectors":    "XLF, XLK, XLE, XLV, XLI",
        }
        _p_cols = st.columns(len(_PRESETS))
        for _pi, (_plbl, _ptickers) in enumerate(_PRESETS.items()):
            with _p_cols[_pi]:
                if st.button(_plbl, key=f"preset_{_pi}", use_container_width=True,
                             help=_ptickers):
                    st.session_state["_preset_tickers"] = _ptickers
                    st.rerun()
        if "preset_tickers" in st.session_state.get("_preset_tickers", ""):
            _default_tickers = st.session_state.pop("_preset_tickers", _default_tickers)
        elif "_preset_tickers" in st.session_state:
            _default_tickers = st.session_state.pop("_preset_tickers")
        # CSV import for watchlist
    _wl_csv = st.file_uploader("Or import tickers from CSV (one ticker per row)",
                               type=["csv"], key="wl_csv_import")
    if _wl_csv is not None:
        import io as _io_wl
        _csv_lines = _wl_csv.getvalue().decode().strip().splitlines()
        _csv_tickers = [l.split(",")[0].strip().upper() for l in _csv_lines
                        if l.strip() and not l.startswith("#")]
        _default_tickers = ",".join(_csv_tickers[:50])
        st.success(f"Imported {len(_csv_tickers)} tickers from CSV")
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
_auto_trigger = st.session_state.pop("_trigger_auto_scan", False)
if run_btn or _auto_trigger:
    if _auto_trigger: st.session_state["_last_auto_scan_ts"] = _time_sched.time()
    _t0 = time.time()
    _n_tickers = len(dfs)
    _prog_bar = st.progress(0.0, f"Scanning 0/{_n_tickers} assets…")
    import threading as _thr
    _done_flag = [False]
    def _fake_progress():
        import time as _tp
        _step = 0
        while not _done_flag[0] and _step < 95:
            _frac = min(0.95, _step / 100)
            _prog_bar.progress(_frac, f"Scanning {max(1,int(_frac*_n_tickers))}/{_n_tickers}…")
            _tp.sleep(0.3); _step += 3
    _thr.Thread(target=_fake_progress, daemon=True).start()
    st.session_state["wl_result"]      = _run_scan(dfs, window, step, tau_max, n_null, epsilon, tuple(timeframes))
    _done_flag[0] = True
    _prog_bar.progress(1.0, f"✅ {_n_tickers}/{_n_tickers} assets scanned")
    import time as _tc; _tc.sleep(0.35); _prog_bar.empty()
    st.session_state["scan_duration"]  = round(time.time() - _t0, 1)
    st.session_state["scan_timestamp"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    _n_f = st.session_state["wl_result"].n_flagged
    if _n_f >= 3:
        _ct_t = [a.ticker for a in st.session_state["wl_result"].assets if a.regime_active][:5]
        st.toast(f"⚠️ Systemic: {_n_f} assets in FAWP ({chr(44).join(_ct_t)}) — contagion?", icon="⚠️")
        st.session_state["_contagion_assets"] = _ct_t
    else:
        st.session_state.pop("_contagion_assets", None)
    # Auto diff-alert: detect FAWP set changes and email if subscribed
    _prev_fawp = set(st.session_state.get("_prev_fawp_tickers", []))
    _curr_fawp = {a.ticker for a in
                  getattr(st.session_state.get("wl_result"), "assets", [])
                  if a.regime_active}
    if _prev_fawp != _curr_fawp and _prev_fawp:  # skip first scan
        _ent = sorted(_curr_fawp - _prev_fawp)
        _ext = sorted(_prev_fawp - _curr_fawp)
        if _ent or _ext:
            _chg = []
            if _ent: _chg.append("Entered: " + ", ".join(_ent))
            if _ext: _chg.append("Exited: "  + ", ".join(_ext))
            st.toast("🔔 FAWP change: " + " | ".join(_chg), icon="🔔")
            _uemail = (st.session_state.get("auth_user") or {}).get("email", "")
            _rkey   = st.session_state.get("resend_api_key", "")
            if _uemail and _rkey and st.session_state.get("digest_toggle"):
                try:
                    from email_digest import send_digest
                    send_digest(_uemail, api_key=_rkey,
                                finance_results=st.session_state.get("wl_result"),
                                seismic_result=st.session_state.get("seis_result"),
                                dynamo_result=st.session_state.get("dynamo_result"))
                except Exception:
                    pass
    st.session_state["_prev_fawp_tickers"] = list(_curr_fawp)
    st.toast(f"🔴 {_n_f} FAWP regime(s) active" if _n_f else "✅ Scan complete — no FAWP active",
             icon="🔴" if _n_f else "✅")
    # Encode key scan params into URL for sharing
    try:
        import base64 as _b64, json as _jsurl
        _url_data = _b64.urlsafe_b64encode(_jsurl.dumps({
            "t": ",".join(list(dfs.keys())[:5]),
            "w": window, "e": epsilon, "n": _n_f
        }).encode()).decode()[:80]
        st.query_params["scan"] = _url_data
    except Exception:
        pass
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
if st.session_state.get("_contagion_assets"):
    _ct = st.session_state["_contagion_assets"]
    st.warning(f"⚠️ **Systemic FAWP event** — {len(_ct)} assets entered regime simultaneously: "
               f"{", ".join(_ct)}. Possible market contagion. Check 🌐 All Scanners.")
if "scan" in st.query_params:
    _share_url = f"https://fawp-scanner.info/?theme={st.session_state.get('theme','dark')}&scan={st.query_params['scan']}"
    # Permalink: if arriving via shared link with no active scan, show decoded summary
    if "wl_result" not in st.session_state:
        try:
            import base64 as _b64d, json as _jsd
            _decoded = _jsd.loads(_b64d.urlsafe_b64decode(st.query_params["scan"] + "=="))
            st.info(f"📎 Shared scan · Tickers: {_decoded.get('t','?')} · ε={_decoded.get('e','?')} · {_decoded.get('n',0)} FAWP active")
        except Exception:
            pass
    st.sidebar.text_input("🔗 Share this scan", value=_share_url,
                          key="share_url_box", help="Copy link to share this scan result")
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
tab_scanner, tab_curves, tab_heatmap, tab_significance, tab_validation, tab_history, tab_xdomain, tab_compare, tab_weather, tab_admin, tab_export = st.tabs([
    "📊 Scanner", "📈 Curves", "🔥 Heatmap", "📐 Significance",
    "✅ Validation", "📜 History", "🌐 All Scanners", "⚖ Compare",
    "🌦 Weather", "⚙ Admin", "📤 Export",
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
<div style="color:#7A90B8;font-size:.88em;margin-bottom:.6em">
Detecting the Information-Control Exclusion Principle in real-time.<br>
Follow these steps to run your first scan:
</div>
<div style="background:#0A1520;border:1px solid #1E3050;border-radius:6px;padding:.6em .9em;
margin-bottom:1em;font-size:.82em;color:#5A8ABA">
💡 <b style="color:#D4AF37">Quick start:</b>
Use the sidebar presets (S&P core, Crypto, etc.) to fill tickers instantly,
then click <b style="color:#EDF0F8">▶ Run Scan</b>.
The <b style="color:#EDF0F8">MI chart</b> shows forecast vs steering coupling —
when gold rises and blue collapses, that's FAWP.
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
    # E9.7 calibration callout
    import fawp_index as _fi_cal
    st.markdown(
        f'<div style="font-size:.72em;color:#3A4E70;padding:.3em .6em;'
        f'background:#0D1729;border-radius:6px;border:1px solid #182540;'
        f'display:inline-block;margin-bottom:.6em">'
        f'📐 gap2 peak leads cliff by <b style="color:#D4AF37">+{_fi_cal.E97_MEAN_LEAD_GAP2_TO_CLIFF_U:.3f} delays</b> · '
        f'ODW localisation error <b style="color:#D4AF37">~{_fi_cal.E97_MEAN_ABS_ERR_GAP2_VS_ODW_START:.1f} delays</b> · '
        f'<a href="https://doi.org/10.5281/zenodo.19065421" style="color:#4A7FCC">E9.7</a>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Regime confidence interval — estimate ±σ from scan windows
    _ci_note = ""
    if wl and ranked:
        _top = ranked[0]
        if _top.scan is not None and len(_top.scan.windows) >= 5:
            import numpy as _np_ci
            _gap_vals = [w.odw_result.peak_gap_bits for w in _top.scan.windows
                         if hasattr(w, "odw_result") and w.odw_result is not None]
            if len(_gap_vals) >= 5:
                _gap_arr = _np_ci.array(_gap_vals, dtype=float)
                _ci_lo = max(0.0, float(_gap_arr.mean() - 1.96*_gap_arr.std()))
                _ci_hi = float(_gap_arr.mean() + 1.96*_gap_arr.std())
                _ci_note = (f"Top asset 95% CI: [{_ci_lo:.4f}, {_ci_hi:.4f}]b  "
                            f"(n={len(_gap_vals)} windows)")
                st.caption(f"📊 {_ci_note}")

    # Portfolio mode: aggregate FAWP exposure score
    with st.expander("💼 Portfolio FAWP exposure", expanded=False):
        st.caption("Enter position weights to compute aggregate FAWP exposure score.")
        if wl and ranked:
            _port_weights = {}
            _pw_cols = st.columns(min(5, len(ranked[:10])))
            for _pi, _pa in enumerate(ranked[:10]):
                with _pw_cols[_pi % len(_pw_cols)]:
                    _pw = st.number_input(
                        f"{_pa.ticker}", 0.0, 100.0, 10.0, 5.0,
                        key=f"pw_{_pa.ticker}_{_pa.timeframe}",
                        help=f"{_pa.timeframe} · gap {_pa.peak_gap_bits:.4f}b"
                    )
                _port_weights[_pa] = _pw / 100.0
            _total_w = sum(_port_weights.values())
            if _total_w > 0:
                _fawp_exp = sum(w for a, w in _port_weights.items() if a.regime_active) / _total_w
                _gap_exp  = sum(float(a.peak_gap_bits or 0) * w
                                for a, w in _port_weights.items()) / _total_w
                _col = "#C0111A" if _fawp_exp > 0.4 else "#D4AF37" if _fawp_exp > 0.1 else "#1DB954"
                st.markdown(
                    f'<div style="background:#0D1729;border:1px solid #1E3050;border-radius:6px;'
                    f'padding:.6em 1em;margin-top:.4em">'
                    f'<span style="font-size:1.25em;font-weight:800;color:{_col}">'
                    f'{_fawp_exp*100:.1f}% of portfolio in FAWP</span>'
                    f'<span style="color:#7A90B8;font-size:.82em;margin-left:.8em">'
                    f'· weighted avg gap {_gap_exp:.4f} bits</span></div>',
                    unsafe_allow_html=True
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
            # 5-level severity colour scale
            _gap_v = float(a.peak_gap_bits or 0)
            if a.regime_active and _gap_v >= 1.5:     row_cls = "asset-row sev-critical"
            elif a.regime_active and _gap_v >= 0.5:   row_cls = "asset-row sev-high"
            elif a.regime_active:                      row_cls = "asset-row sev-medium"
            elif a.latest_score >= 0.005:             row_cls = "asset-row sev-low"
            else:                                       row_cls = "asset-row"
            threshold_badge = (
                '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;'
                'background:#FF3040;animation:blink 1.4s ease-in-out infinite;'
                'margin-right:.35em;vertical-align:middle"'
                ' title="Gap ≥ threshold {alert_threshold:.3f}b"></span>'
                if (st.session_state.get("ticker_thresholds",{}).get(a.ticker, alert_threshold) > 0
                and a.peak_gap_bits >= st.session_state.get("ticker_thresholds",{}).get(
                    a.ticker, alert_threshold) and a.regime_active)
                else ""
            )
            days_str   = f"{a.days_in_regime}d" if a.days_in_regime else "—"
            decay_str  = (f"−{abs(a.steer_decay_rate):.4f} b/τ"
                          if hasattr(a, "steer_decay_rate") and a.steer_decay_rate else "")
            age_str    = f"age {a.signal_age_days}d"
            # Streak: consecutive scans in FAWP from history
            _streak = 0
            try:
                _hist_store = get_store()
                if hasattr(_hist_store, "asset_timeline"):
                    _tl_s = _hist_store.asset_timeline(a.ticker, a.timeframe)
                    if not _tl_s.empty and "regime_active" in _tl_s.columns:
                        _acts = _tl_s["regime_active"].values[::-1]
                        for _v in _acts:
                            if _v: _streak += 1
                            else: break
            except Exception:
                pass
            streak_html = (
                f'<span style="font-size:.72em;background:#4A1A1A;color:#FF6B6B;'
                f'border-radius:4px;padding:.1em .4em;margin-left:.3em">🔥 {_streak} scans</span>'
                if _streak >= 2 else ""
            )

            conf_html  = _confidence_html(a)
            st.markdown(
                f'<div class="{row_cls}">'
                f'{threshold_badge}<span class="asset-ticker">{a.ticker}</span>'
                f'<span class="asset-tf">{a.timeframe}</span>'
                f'{streak_html}'
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

            # Inline explain + MI curve expander for flagged/high-risk assets
            if a.regime_active or a.latest_score >= 0.005:
                with st.expander(f"Why {a.ticker} is flagged", expanded=False):
                    st.markdown(_explain_html(a), unsafe_allow_html=True)
                    # Full MI curve chart on expand
                    if HAS_MPL and a.scan is not None:
                        try:
                            _pred_mi  = [w.pred_mi  for w in a.scan.windows if hasattr(w, "pred_mi")]
                            _steer_mi = [w.steer_mi for w in a.scan.windows if hasattr(w, "steer_mi")]
                            if not _pred_mi:
                                raise ValueError("no MI data")
                            import matplotlib.pyplot as _plt_mi2, numpy as _np_mi2
                            _tau2 = _np_mi2.arange(1, len(_pred_mi[-1]) + 1)
                            _fig_mi2, _ax_mi2 = _plt_mi2.subplots(figsize=(8, 3), facecolor="#0D1729")
                            _ax_mi2.set_facecolor("#07101E")
                            _ax_mi2.plot(_tau2, _pred_mi[-1], color="#D4AF37", lw=2, label="Pred MI (latest)")
                            _ax_mi2.plot(_tau2, _steer_mi[-1], color="#4A7FCC", lw=1.5, ls="--", label="Steer MI")
                            _ax_mi2.axhline(epsilon, color="#3A4E70", ls=":", lw=1, label=f"ε={epsilon}")
                            _ax_mi2.set_xlabel("τ (delay)", fontsize=8, color="#7A90B8")
                            _ax_mi2.set_ylabel("MI (bits)", fontsize=8, color="#7A90B8")
                            _ax_mi2.set_title(f"{a.ticker} [{a.timeframe}] MI curves",
                                             color="#D4AF37", fontsize=9)
                            _ax_mi2.legend(fontsize=7, framealpha=0.2)
                            for _sp2 in _ax_mi2.spines.values(): _sp2.set_edgecolor("#3A4E70")
                            _ax_mi2.tick_params(colors="#7A90B8")
                            _fig_mi2.tight_layout()
                            st.pyplot(_fig_mi2, use_container_width=True)
                            _plt_mi2.close(_fig_mi2)
                        except Exception:
                            pass

    # Multi-ticker comparison chart
    if len(filtered) > 1 and HAS_MPL:
        import matplotlib.pyplot as _plt_cmp, matplotlib.patches as _mp_cmp
        st.markdown(_sec("Score comparison"), unsafe_allow_html=True)
        _tks  = [f"{a.ticker}\n{a.timeframe}" for a in filtered]
        _scrs = [a.latest_score for a in filtered]
        _gps  = [a.peak_gap_bits for a in filtered]
        _cols = ["#C0111A" if a.regime_active else "#2A4070" for a in filtered]
        _fig_c, (_ax_s, _ax_g) = _plt_cmp.subplots(1, 2, figsize=(10, max(2.5, len(filtered)*0.4+1)))
        _fig_c.patch.set_facecolor("#07101E")
        for _ax in (_ax_s, _ax_g):
            _ax.set_facecolor("#0D1729")
            for _sp in _ax.spines.values(): _sp.set_edgecolor("#3A4E70")
            _ax.tick_params(colors="#7A90B8", labelsize=8)
        _y = range(len(_tks))
        _ax_s.barh(list(_y), _scrs, color=_cols, alpha=0.85, height=0.6)
        _ax_s.set_yticks(list(_y)); _ax_s.set_yticklabels(_tks, fontsize=8, color="#EDF0F8")
        _ax_s.set_xlabel("FAWP score", fontsize=8, color="#7A90B8")
        _ax_s.set_title("Score", color="#D4AF37", fontsize=9, fontweight="bold")
        _ax_s.invert_yaxis()
        _ax_g.barh(list(_y), _gps, color=_cols, alpha=0.85, height=0.6)
        _ax_g.set_yticks(list(_y)); _ax_g.set_yticklabels([], fontsize=8)
        _ax_g.set_xlabel("Peak gap (bits)", fontsize=8, color="#7A90B8")
        _ax_g.set_title("Gap (bits)", color="#D4AF37", fontsize=9, fontweight="bold")
        _ax_g.axvline(epsilon, color="#3A4E70", ls=":", lw=1)
        _ax_g.invert_yaxis()
        _fig_c.legend(handles=[_mp_cmp.Patch(color="#C0111A",label="FAWP"),_mp_cmp.Patch(color="#2A4070",label="Clear")],
                      loc="lower right", fontsize=7, framealpha=0.2)
        _fig_c.tight_layout(pad=0.6)
        st.pyplot(_fig_c, use_container_width=True)
        import io as _io_c; _cb = _io_c.BytesIO()
        _fig_c.savefig(_cb, format="png", dpi=150, bbox_inches="tight")
        _plt_cmp.close(_fig_c); _cb.seek(0)
        st.download_button("⬇ Download comparison PNG", data=_cb,
                           file_name="fawp_comparison.png", mime="image/png", key="cmp_dl")

    # FAWP age chart — how old is the current signal?
    fawp_assets = [a for a in filtered if a.regime_active and a.odw_result is not None]
    if fawp_assets and HAS_MPL:
        import matplotlib.pyplot as _plt_age
        st.markdown(_sec("Signal age — time inside detection window"), unsafe_allow_html=True)
        st.caption("Bar = position within ODW. Full bar = at cliff edge. Dashed = τ⁺ₕ horizon.")
        _fig_age, _ax_age = _plt_age.subplots(figsize=(8, max(1.5, len(fawp_assets)*0.4)))
        _fig_age.patch.set_facecolor("#07101E"); _ax_age.set_facecolor("#0D1729")
        for _sp in _ax_age.spines.values(): _sp.set_edgecolor("#3A4E70")
        _ax_age.tick_params(colors="#7A90B8", labelsize=8)
        for _ai, _aa in enumerate(fawp_assets):
            _odw = _aa.odw_result
            _win_len = max(1, (_odw.odw_end or 35) - (_odw.odw_start or 30))
            _age_bars = _aa.days_in_regime or 1
            _pct = min(1.0, _age_bars / _win_len)
            _col = "#C0111A" if _pct > 0.7 else "#D4AF37" if _pct > 0.4 else "#1DB954"
            _ax_age.barh(_ai, _pct, color=_col, alpha=0.8, height=0.6)
            _ax_age.text(_pct+0.02, _ai, f"{_age_bars}d / {_win_len}d ({_pct*100:.0f}%)",
                        va="center", fontsize=7, color="#EDF0F8")
            if _odw.tau_h_plus and _odw.odw_end:
                _h_pct = min(1.0, (_odw.tau_h_plus - (_odw.odw_start or 30)) / _win_len)
                _ax_age.axvline(_h_pct, color="#4A7FCC", ls="--", lw=1, alpha=0.7)
        _ax_age.set_yticks(range(len(fawp_assets)))
        _ax_age.set_yticklabels([f"{a.ticker} [{a.timeframe}]" for a in fawp_assets],
                                fontsize=8, color="#EDF0F8")
        _ax_age.set_xlim(0, 1.35); _ax_age.set_xlabel("ODW progress", fontsize=8, color="#7A90B8")
        _ax_age.axvline(1.0, color="#C0111A", lw=1.5, ls=":", alpha=0.5)
        _fig_age.tight_layout(); st.pyplot(_fig_age, use_container_width=True)
        _plt_age.close(_fig_age)

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
        # 📸 Quick screenshot export of KPI + results
        if HAS_MPL and col_lb2:
            with col_lb2:
                if st.button("📸 Screenshot", key="screenshot_btn", use_container_width=True,
                             help="Export KPI summary + top results as a PNG"):
                    try:
                        import matplotlib.pyplot as _plt_ss, io as _io_ss, numpy as _np_ss
                        _fig_ss, _ax_ss = _plt_ss.subplots(figsize=(10, 4), facecolor="#07101E")
                        _ax_ss.set_facecolor("#0D1729")
                        _ax_ss.axis("off")
                        _ss_rows = [["Ticker","TF","FAWP","Score","Gap (bits)"]]
                        for _ssa in filtered[:8]:
                            _ss_rows.append([
                                _ssa.ticker, _ssa.timeframe,
                                "🔴 YES" if _ssa.regime_active else "—",
                                f"{_ssa.latest_score:.4f}", f"{_ssa.peak_gap_bits:.4f}",
                            ])
                        _tbl = _ax_ss.table(_ss_rows, cellLoc="center", loc="center",
                                            bbox=[0,0,1,1])
                        _tbl.auto_set_font_size(False); _tbl.set_fontsize(9)
                        for (_r,_c), _cell in _tbl.get_celld().items():
                            _cell.set_facecolor("#182540" if _r==0 else "#0D1729")
                            _cell.set_edgecolor("#3A4E70")
                            _cell.set_text_props(color="#D4AF37" if _r==0 else "#EDF0F8")
                        _ax_ss.set_title(f"FAWP Finance Scan — {scan_timestamp}  ·  ε={epsilon:.3f}",
                                        color="#D4AF37", fontsize=10, pad=12)
                        _fig_ss.tight_layout()
                        _ss_buf = _io_ss.BytesIO()
                        _fig_ss.savefig(_ss_buf, format="png", dpi=150, bbox_inches="tight",
                                        facecolor="#07101E")
                        _plt_ss.close(_fig_ss); _ss_buf.seek(0)
                        st.download_button("⬇ Download screenshot", data=_ss_buf,
                                           file_name=f"fawp_scan_{scan_timestamp.replace(' ','_')}.png",
                                           mime="image/png", key="ss_dl")
                    except Exception as _sse:
                        st.caption(f"Screenshot: {_sse}")
    except Exception as e:
        st.caption(f"Leaderboard unavailable: {e}")

    # Global community leaderboard (anonymised, shared via Supabase)
    st.markdown(_sec("Global leaderboard — top flagged assets today"), unsafe_allow_html=True)
    try:
        _store_gl = get_store()
        if hasattr(_store_gl, "_db") and _store_gl._db is not None:
            # Write today's FAWP-active assets to shared table
            if wl and wl.n_flagged > 0:
                import json as _jgl
                _today = pd.Timestamp.now().strftime("%Y-%m-%d")
                for _ga in wl.active_regimes():
                    _store_gl._db.table("fawp_global_lb").upsert({
                        "ticker":     _ga.ticker,
                        "timeframe":  _ga.timeframe,
                        "score":      round(_ga.latest_score, 4),
                        "gap_bits":   round(_ga.peak_gap_bits, 4),
                        "scan_date":  _today,
                        "count":      1,
                    }, on_conflict="ticker,timeframe,scan_date").execute()
            # Read global top 10
            _gl_res = (_store_gl._db.table("fawp_global_lb")
                       .select("ticker,timeframe,score,gap_bits,count")
                       .order("gap_bits", desc=True)
                       .limit(10).execute())
            _gl_rows = _gl_res.data or []
            if _gl_rows:
                import pandas as _pd_gl
                _gl_df = _pd_gl.DataFrame(_gl_rows)
                st.dataframe(_gl_df, use_container_width=True, hide_index=True)
                st.caption("Anonymised — shows asset symbols only, no user data.")
            else:
                st.caption("No global data yet — run scans to contribute.")
        else:
            st.caption("Sign in to contribute to and view the global leaderboard.")
    except Exception as _gle:
        st.caption(f"Global leaderboard unavailable: {_gle}")

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

        st.markdown(_sec("FAWP regime heatmap"), unsafe_allow_html=True)
        st.caption("🔴 = FAWP active · Color intensity = score magnitude · Numbers = peak gap (bits)")

        # Richer heatmap: FAWP active cells highlighted with red border
        mat_active = np.full((n_t, n_tf), False)
        for a in wl.assets:
            if not a.error:
                mat_active[tickers_h.index(a.ticker), tfs_h.index(a.timeframe)] = a.regime_active

        fig_h2, ax_h2 = plt.subplots(figsize=(max(4, n_tf * 1.8), max(3, n_t * 0.8)))
        fig_h2.patch.set_facecolor("#0D1729")
        ax_h2.set_facecolor("#0D1729")
        im_h2 = ax_h2.imshow(mat, aspect="auto", cmap="RdYlGn_r",
                              vmin=0, vmax=max(0.01, float(np.nanmax(mat))))
        ax_h2.set_xticks(range(n_tf)); ax_h2.set_xticklabels(tfs_h, fontsize=9, color="#7A90B8")
        ax_h2.set_yticks(range(n_t));  ax_h2.set_yticklabels(tickers_h, fontsize=9, color="#7A90B8")
        cb_h2 = plt.colorbar(im_h2, ax=ax_h2)
        cb_h2.ax.yaxis.set_tick_params(color="#7A90B8", labelsize=7)
        cb_h2.set_label("Score", color="#7A90B8", fontsize=8)
        for i in range(n_t):
            for j in range(n_tf):
                if not np.isnan(mat[i, j]):
                    # Show gap bits in cell
                    _gap = mat_gap[i, j] if not np.isnan(mat_gap[i, j]) else 0
                    ax_h2.text(j, i, f"{_gap:.3f}", ha="center", va="center",
                               fontsize=7, color="white", fontfamily="monospace")
                    # Red border for FAWP active cells
                    if mat_active[i, j]:
                        from matplotlib.patches import Rectangle
                        ax_h2.add_patch(Rectangle((j - 0.48, i - 0.48), 0.96, 0.96,
                                                   fill=False, edgecolor="#C0111A", lw=2.5))
        for spine in ax_h2.spines.values(): spine.set_edgecolor("#182540")
        plt.tight_layout(pad=0.4)
        st.pyplot(fig_h2, use_container_width=True)
        plt.close(fig_h2)

        st.markdown(_sec("Regime score heatmap (numeric)"), unsafe_allow_html=True)
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
    # Scan history timeline chart
    st.markdown("### 📈 Scan History Timeline")
    st.caption("Peak gap bits over time for each asset from scan history.")
    try:
        _tl_assets = hist.all_assets() if hasattr(hist,"all_assets") else []
        if _tl_assets and HAS_MPL:
            import matplotlib.pyplot as _plt_tl
            import pandas as _pd_tl
            _fig_tl, _ax_tl = _plt_tl.subplots(figsize=(10, 4), facecolor="#0D1729")
            _ax_tl.set_facecolor("#07101E")
            for _ta in _tl_assets[:8]:
                _tl = hist.asset_timeline(_ta["ticker"], _ta.get("timeframe","1d"))
                if _tl.empty or "peak_gap_bits" not in _tl.columns: continue
                _tl = _tl.dropna(subset=["peak_gap_bits"]).sort_values("scanned_at")
                _dates = _pd_tl.to_datetime(_tl["scanned_at"])
                _gaps  = _tl["peak_gap_bits"].astype(float)
                _col   = "#C0111A" if _tl["regime_active"].any() else "#3A4E70"
                _ax_tl.plot(_dates, _gaps, lw=1.4, alpha=0.85, color=_col,
                           label=_ta["ticker"])
                # Mark FAWP entries
                _fawp_rows = _tl[_tl["regime_active"] == True]
                if not _fawp_rows.empty:
                    _ax_tl.scatter(_pd_tl.to_datetime(_fawp_rows["scanned_at"]),
                                  _fawp_rows["peak_gap_bits"].astype(float),
                                  color=_col, s=22, zorder=5)
            for _sp in _ax_tl.spines.values(): _sp.set_edgecolor("#3A4E70")
            _ax_tl.tick_params(colors="#7A90B8", labelsize=7)
            _ax_tl.set_ylabel("Peak gap (bits)", fontsize=8, color="#7A90B8")
            _ax_tl.set_title("FAWP intensity over time  (dots = FAWP active)",
                            color="#D4AF37", fontsize=9)
            _ax_tl.legend(fontsize=7, framealpha=0.2, facecolor="#0D1729",
                         ncol=4, loc="upper left")
            _fig_tl.autofmt_xdate(rotation=30, ha="right")
            _fig_tl.tight_layout()
            st.pyplot(_fig_tl, use_container_width=True)
            _plt_tl.close(_fig_tl)
        elif not _tl_assets:
            st.info("No scan history yet. Run a scan first.")
    except Exception as _tle:
        st.caption(f"Timeline unavailable: {_tle}")
    st.markdown("---")

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

                # Rolling time×ticker heatmap
                # Multi-asset FAWP correlation matrix
                if HAS_MPL and hasattr(hist, "all_assets"):
                    try:
                        import matplotlib.pyplot as _plt_corr, numpy as _np_corr
                        _corr_assets = hist.all_assets()[:12]
                        _corr_vecs = []
                        _corr_labels = []
                        for _ca in _corr_assets:
                            _tl = hist.asset_timeline(_ca["ticker"], _ca.get("timeframe","1d"), last_n=30)
                            if not _tl.empty and "regime_active" in _tl.columns:
                                _corr_vecs.append(_tl["regime_active"].astype(float).values)
                                _corr_labels.append(_ca["ticker"])
                        if len(_corr_vecs) >= 2:
                            _min_len = min(len(v) for v in _corr_vecs)
                            _corr_mtx = _np_corr.corrcoef([v[:_min_len] for v in _corr_vecs])
                            _fig_corr, _ax_corr = _plt_corr.subplots(
                                figsize=(max(4, len(_corr_labels)*0.6),
                                         max(4, len(_corr_labels)*0.6)), facecolor="#0D1729")
                            _ax_corr.set_facecolor("#07101E")
                            _im_c = _ax_corr.imshow(_corr_mtx, cmap="RdYlGn", vmin=-1, vmax=1)
                            _plt_corr.colorbar(_im_c, ax=_ax_corr, shrink=0.8,
                                               label="correlation").ax.tick_params(colors="#7A90B8", labelsize=7)
                            _ax_corr.set_xticks(range(len(_corr_labels)))
                            _ax_corr.set_yticks(range(len(_corr_labels)))
                            _ax_corr.set_xticklabels(_corr_labels, rotation=45, ha="right", fontsize=7, color="#EDF0F8")
                            _ax_corr.set_yticklabels(_corr_labels, fontsize=7, color="#EDF0F8")
                            for _sp in _ax_corr.spines.values(): _sp.set_edgecolor("#3A4E70")
                            _ax_corr.set_title("FAWP onset correlation", color="#D4AF37", fontsize=9)
                            _fig_corr.tight_layout()
                            st.markdown(_sec("Multi-asset FAWP correlation"), unsafe_allow_html=True)
                            st.caption("Pairwise correlation of FAWP regime timing.")
                            st.pyplot(_fig_corr, use_container_width=True)
                            _plt_corr.close(_fig_corr)
                    except Exception as _ce:
                        st.caption(f"Correlation matrix: {_ce}")

                st.markdown(_sec("Time × ticker FAWP heatmap"), unsafe_allow_html=True)
                st.caption("Rows = assets from last scan · Columns = scan history · Color = FAWP score")
                try:
                    if HAS_MPL and hasattr(hist, "all_assets") and hasattr(hist, "asset_timeline"):
                        _all_a = hist.all_assets()
                        _hm_tickers = [f"{a['ticker']}|{a['timeframe']}" for a in _all_a[:15]]
                        _hm_data = {}
                        for _hma in _all_a[:15]:
                            _hm_tl = hist.asset_timeline(_hma["ticker"], _hma["timeframe"], last_n=20)
                            if not _hm_tl.empty:
                                _hm_data[f"{_hma['ticker']}|{_hma['timeframe']}"] = (
                                    _hm_tl["latest_score"].values,
                                    _hm_tl["scanned_at"].dt.strftime("%m-%d").values,
                                    _hm_tl["regime_active"].values,
                                )
                        if _hm_data:
                            import matplotlib.pyplot as _plt_hm, numpy as _np_hm
                            _rows = list(_hm_data.keys())
                            _max_cols = max(len(v[0]) for v in _hm_data.values())
                            _mat_hm = _np_hm.full((len(_rows), _max_cols), _np_hm.nan)
                            _act_hm = _np_hm.zeros((len(_rows), _max_cols), dtype=bool)
                            _date_lbls = []
                            for _ri, _rk in enumerate(_rows):
                                _scores, _dates, _active = _hm_data[_rk]
                                _mat_hm[_ri, :len(_scores)] = _scores
                                _act_hm[_ri, :len(_active)] = _active
                                if len(_dates) > len(_date_lbls):
                                    _date_lbls = list(_dates)
                            _fig_hm, _ax_hm = _plt_hm.subplots(
                                figsize=(max(6, _max_cols * 0.5), max(3, len(_rows) * 0.5)))
                            _fig_hm.patch.set_facecolor("#07101E")
                            _ax_hm.set_facecolor("#0D1729")
                            _im_hm = _ax_hm.imshow(_mat_hm, aspect="auto", cmap="RdYlGn_r",
                                                    vmin=0, vmax=max(0.01, float(_np_hm.nanmax(_mat_hm))))
                            # Red border on active cells
                            for _ri in range(len(_rows)):
                                for _ci in range(_max_cols):
                                    if _act_hm[_ri, _ci]:
                                        from matplotlib.patches import Rectangle as _Rect
                                        _ax_hm.add_patch(_Rect((_ci-0.48,_ri-0.48),0.96,0.96,
                                                               fill=False,edgecolor="#C0111A",lw=2))
                            _plt_hm.colorbar(_im_hm, ax=_ax_hm, label="Score",
                                             shrink=0.7).ax.tick_params(colors="#7A90B8",labelsize=7)
                            _ax_hm.set_yticks(range(len(_rows)))
                            _ax_hm.set_yticklabels(_rows, fontsize=7, color="#EDF0F8")
                            _ax_hm.set_xticks(range(len(_date_lbls)))
                            _ax_hm.set_xticklabels(_date_lbls, fontsize=7, color="#7A90B8", rotation=30, ha="right")
                            for _sp in _ax_hm.spines.values(): _sp.set_edgecolor("#3A4E70")
                            _fig_hm.tight_layout()
                            st.pyplot(_fig_hm, use_container_width=True)
                            _plt_hm.close(_fig_hm)
                except Exception as _hme:
                    st.caption(f"Heatmap unavailable: {_hme}")

                st.markdown(_sec("Score & gap timeline"), unsafe_allow_html=True)

                if HAS_MPL:
                    import matplotlib.pyplot as plt
                    dates  = tl["scanned_at"].dt.strftime("%m-%d %H:%M").tolist()
                    scores = tl["latest_score"].tolist()
                    gaps   = tl["peak_gap_bits"].tolist()
                    active = tl["regime_active"].tolist()
                    tick_step = max(1, len(dates) // 8)
                    xticks = range(0, len(dates), tick_step)

                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4.5),
                                                    facecolor="#07101E", sharex=True)
                    for ax in (ax1, ax2):
                        ax.set_facecolor("#0D1729")
                        for sp in ax.spines.values(): sp.set_edgecolor("#3A4E70")
                        ax.tick_params(colors="#3A4E70")

                    # Top: FAWP score bars coloured by regime state
                    colors = ["#C0111A" if a else "#2A4070" for a in active]
                    ax1.bar(range(len(scores)), scores, color=colors, alpha=0.85, edgecolor="none")
                    ax1.set_ylabel("Score", fontsize=8, color="#7A90B8")
                    ax1.set_ylim(0, max(max(scores) * 1.15, 0.01))
                    # Legend patches
                    import matplotlib.patches as _mp
                    ax1.legend(handles=[
                        _mp.Patch(color="#C0111A", label="FAWP active"),
                        _mp.Patch(color="#2A4070", label="Clear"),
                    ], fontsize=7, framealpha=0.2, loc="upper left")

                    # Bottom: peak gap bits line
                    ax2.plot(range(len(gaps)), gaps, color="#D4AF37", lw=1.8)
                    ax2.fill_between(range(len(gaps)), gaps, alpha=0.15, color="#D4AF37")
                    ax2.axhline(0.01, color="#3A4E70", ls=":", lw=1, label="ε=0.01")
                    ax2.set_ylabel("Peak gap (bits)", fontsize=8, color="#7A90B8")
                    ax2.set_xticks(list(xticks))
                    ax2.set_xticklabels([dates[i] for i in xticks],
                                        rotation=25, ha="right", fontsize=7, color="#7A90B8")
                    ax2.legend(fontsize=7, framealpha=0.2)

                    fig.tight_layout(pad=0.4)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                    # PNG export
                    import io as _io
                    buf = _io.BytesIO()
                    fig2, (ax1b, ax2b) = plt.subplots(2, 1, figsize=(10, 4.5), facecolor="#07101E", sharex=True)
                    for ax in (ax1b, ax2b):
                        ax.set_facecolor("#0D1729")
                        for sp in ax.spines.values(): sp.set_edgecolor("#3A4E70")
                        ax.tick_params(colors="#3A4E70")
                    ax1b.bar(range(len(scores)), scores, color=colors, alpha=0.85, edgecolor="none")
                    ax1b.set_ylabel("Score", fontsize=8, color="#7A90B8")
                    ax2b.plot(range(len(gaps)), gaps, color="#D4AF37", lw=1.8)
                    ax2b.fill_between(range(len(gaps)), gaps, alpha=0.15, color="#D4AF37")
                    ax2b.set_xticks(list(xticks))
                    ax2b.set_xticklabels([dates[i] for i in xticks], rotation=25, ha="right", fontsize=7, color="#7A90B8")
                    fig2.tight_layout(pad=0.4)
                    fig2.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                    plt.close(fig2)
                    buf.seek(0)
                    st.download_button("⬇ Download chart PNG", data=buf,
                                       file_name=f"fawp_history_{hticker}_{htf}.png",
                                       mime="image/png")

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

                # Regime duration histogram
                if HAS_MPL:
                    try:
                        import matplotlib.pyplot as _plt_dur, numpy as _np_dur
                        # Calculate regime run lengths from regime_active column
                        _acts = tl["regime_active"].values
                        _durations = []
                        _run = 0
                        for _v in _acts:
                            if _v:
                                _run += 1
                            elif _run > 0:
                                _durations.append(_run)
                                _run = 0
                        if _run > 0:
                            _durations.append(_run)

                        if _durations:
                            _fig_dur, _ax_dur = _plt_dur.subplots(figsize=(6, 2.5),
                                                                    facecolor="#0D1729")
                            _ax_dur.set_facecolor("#07101E")
                            for _sp in _ax_dur.spines.values(): _sp.set_edgecolor("#3A4E70")
                            _ax_dur.tick_params(colors="#7A90B8", labelsize=8)
                            _ax_dur.hist(_durations, bins=max(5, len(_durations)//2),
                                        color="#C0111A", alpha=0.75, edgecolor="#3A4E70")
                            _ax_dur.axvline(_np_dur.mean(_durations), color="#D4AF37",
                                           ls="--", lw=1.5, label=f"Mean: {_np_dur.mean(_durations):.1f}")
                            _ax_dur.set_xlabel("Regime duration (scans)", fontsize=8, color="#7A90B8")
                            _ax_dur.set_ylabel("Count", fontsize=8, color="#7A90B8")
                            _ax_dur.set_title(f"FAWP regime durations — {hticker}",
                                             color="#D4AF37", fontsize=9)
                            _ax_dur.legend(fontsize=7, framealpha=0.2)
                            _fig_dur.tight_layout()
                            st.markdown(_sec("Regime duration distribution"), unsafe_allow_html=True)
                            st.pyplot(_fig_dur, use_container_width=True)
                            _plt_dur.close(_fig_dur)
                            st.caption(f"{len(_durations)} regime episode(s) · "
                                      f"mean duration {_np_dur.mean(_durations):.1f} scans · "
                                      f"longest {max(_durations)} scans")
                    except Exception as _due:
                        st.caption(f"Duration histogram: {_due}")

                st.download_button(
                    "Download timeline CSV",
                    data=tl.to_csv(index=False).encode(),
                    file_name=f"fawp_history_{hticker}_{htf}.csv",
                    mime="text/csv",
                )
    except Exception as e:
        st.error(f"History unavailable: {e}")

    # Email digest subscription toggle
    st.markdown(_sec("Weekly email digest"), unsafe_allow_html=True)
    _user_email = get_user_email() if _AUTH_ENABLED else None
    if _user_email:
        _digest_key = f"digest_sub_{_user_email}"
        _subscribed = st.session_state.get(_digest_key, False)
        _tog = st.toggle("Subscribe to weekly FAWP digest",
                         value=_subscribed, key="digest_toggle")
        if _tog != _subscribed:
            st.session_state[_digest_key] = _tog
            if _tog:
                st.success(f"✅ Digest enabled — weekly summary will be sent to {_user_email}")
            else:
                st.info("Digest unsubscribed.")
        if _subscribed:
            st.caption(f"Weekly digest → {_user_email} · Sent every Monday")
            if st.button("Send test digest now", key="test_digest"):
                try:
                    from email_digest import send_digest
                    _wl_data = get_store().all_assets() if hasattr(get_store(), "all_assets") else []
                    _resend_key = st.session_state.get("resend_api_key", "")
                    if not _resend_key:
                        st.warning("Add your Resend API key in Admin tab to enable email delivery.")
                        ok = False
                    else:
                        ok = send_digest(
                            _user_email, api_key=_resend_key,
                            finance_results=_wl_data,
                            seismic_result=st.session_state.get("seis_result"),
                            dynamo_result=st.session_state.get("dynamo_result"),
                        )
                    st.success("Test digest sent!" if ok else "Send failed — check RESEND_API_KEY.")
                except Exception as _de:
                    st.error(f"Digest error: {_de}")
    else:
        st.caption("Sign in to subscribe to weekly digest emails.")


# ──────────────────────────────────────────────────────────────────────────
# Compare tab
# ──────────────────────────────────────────────────────────────────────────
with tab_xdomain:
    st.markdown("### 🌐 Cross-Scanner FAWP Leaderboard")
    st.caption("All active FAWP signals across Finance · Weather · Seismic · Dynamic Systems, ranked by peak gap.")
    _xd_rows = []
    if "wl_result" in st.session_state and st.session_state["wl_result"] is not None:
        for _xa in getattr(st.session_state["wl_result"], "assets", []):
            if getattr(_xa, "regime_active", False):
                _xd_rows.append({
                    "Scanner": "📈 Finance", "Asset": getattr(_xa,"ticker","—"),
                    "TF": getattr(_xa,"timeframe","—"),
                    "Peak Gap (bits)": round(float(getattr(_xa,"peak_gap_bits",0) or 0), 4),
                    "τ⁺ₕ": getattr(getattr(_xa,"odw_result",None),"tau_h_plus","—") or "—",
                })
    if "wx_result" in st.session_state:
        _wxr = st.session_state["wx_result"]
        if getattr(_wxr, "fawp_found", False):
            _xd_rows.append({
                "Scanner": "🌦 Weather",
                "Asset": f"{getattr(_wxr,'location','?')} / {getattr(_wxr,'variable','?')}",
                "TF": "daily",
                "Peak Gap (bits)": round(float(getattr(_wxr,"peak_gap_bits",0) or 0), 4),
                "τ⁺ₕ": getattr(getattr(_wxr,"odw_result",None),"tau_h_plus","—") or "—",
            })
    if "seis_result" in st.session_state:
        _sr = st.session_state["seis_result"]
        if getattr(_sr, "fawp_found", False):
            _xd_rows.append({
                "Scanner": "🌍 Seismic",
                "Asset": st.session_state.get("seis_region","Region"),
                "TF": "daily",
                "Peak Gap (bits)": round(float(getattr(_sr,"peak_gap_bits",0) or 0), 4),
                "τ⁺ₕ": getattr(_sr,"tau_h_plus","—") or "—",
            })
    if "dynamo_result" in st.session_state:
        _dr = st.session_state["dynamo_result"]
        _dodw = _dr.get("odw")
        if _dodw and getattr(_dodw,"fawp_found",False):
            _xd_rows.append({
                "Scanner": "⚙️ Dynamic Systems",
                "Asset": _dr.get("domain","Custom"),
                "TF": "steps",
                "Peak Gap (bits)": round(float(getattr(_dodw,"peak_gap_bits",0) or 0), 4),
                "τ⁺ₕ": getattr(_dodw,"tau_h_plus","—") or "—",
            })
    if _xd_rows:
        import pandas as _pd_xd
        _xd_df = _pd_xd.DataFrame(_xd_rows).sort_values("Peak Gap (bits)", ascending=False)
        st.dataframe(_xd_df, use_container_width=True, hide_index=True)
        st.success(f"**{len(_xd_rows)} active FAWP signal(s)** — "
                   f"peak: **{_xd_df['Peak Gap (bits)'].max():.4f} bits** "
                   f"({_xd_df.iloc[0]['Scanner']} · {_xd_df.iloc[0]['Asset']})")
    else:
        st.info("No active FAWP signals in this session. Run scans in any of the 4 scanners.")

    # First-warning timestamps from history
    st.markdown("---")
    st.markdown("### 🕐 First-Warning Timestamps")
    st.caption("Date each asset first entered FAWP, from scan history.")
    try:
        _fw_rows = []
        for _fwa in (hist.all_assets() if hasattr(hist,"all_assets") else [])[:20]:
            _tl = hist.asset_timeline(_fwa["ticker"], _fwa.get("timeframe","1d"))
            if not _tl.empty and "regime_active" in _tl.columns:
                _first = _tl[_tl["regime_active"] == True]
                if not _first.empty:
                    _fw_rows.append({
                        "Asset": _fwa["ticker"], "TF": _fwa.get("timeframe","1d"),
                        "First FAWP": str(_first["scanned_at"].min())[:16],
                        "Last scan": str(_tl["scanned_at"].max())[:16],
                        "Active scans": int(_first["regime_active"].sum()),
                    })
        if _fw_rows:
            import pandas as _pd_fw
            st.dataframe(_pd_fw.DataFrame(_fw_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No FAWP history yet.")
    except Exception as _fwe:
        st.caption(f"History unavailable: {_fwe}")

with tab_compare:
    # Scan result diff
    st.markdown("### 📊 Scan Diff")
    st.caption("Save current scan as baseline, run again, see what changed.")
    _d1, _d2 = st.columns(2)
    with _d1:
        if st.button("💾 Save as baseline", key="diff_save",
                     disabled="wl_result" not in st.session_state):
            import copy as _cp_d
            st.session_state["diff_baseline"] = _cp_d.deepcopy(st.session_state["wl_result"])
            st.success("Baseline saved.")
    with _d2:
        if st.button("🔄 Compare", key="diff_compare",
                     disabled=("wl_result" not in st.session_state or
                               "diff_baseline" not in st.session_state)):
            import pandas as _pd_d
            _base = st.session_state["diff_baseline"]
            _curr = st.session_state["wl_result"]
            _ba   = {a.ticker for a in getattr(_base,"assets",[]) if a.regime_active}
            _ca   = {a.ticker for a in getattr(_curr,"assets",[]) if a.regime_active}
            _bg   = {a.ticker: float(a.peak_gap_bits or 0) for a in getattr(_base,"assets",[])}
            _cg   = {a.ticker: float(a.peak_gap_bits or 0) for a in getattr(_curr,"assets",[])}
            _rows = ([{"Ticker":t,"Change":"🆕 Entered FAWP","Gap Δ":f"+{_cg.get(t,0):.4f}"} for t in _ca-_ba]
                    +[{"Ticker":t,"Change":"✅ Exited FAWP","Gap Δ":f"-{_bg.get(t,0):.4f}"} for t in _ba-_ca]
                    +[{"Ticker":t,"Change":"🔴 Still in FAWP","Gap Δ":f"{_cg.get(t,0)-_bg.get(t,0):+.4f}"} for t in _ba&_ca])
            st.session_state["diff_result"] = _rows
    if "diff_result" in st.session_state:
        import pandas as _pd_d2
        _dr = st.session_state["diff_result"]
        if _dr:
            st.dataframe(_pd_d2.DataFrame(_dr), use_container_width=True, hide_index=True)
        else:
            st.info("No changes.")
    st.markdown("---")
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
                        tau_max            = w_tau,
                        n_null             = w_null,
                        remove_seasonality = w_seas,
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
            ax.axhline(_EPS_STEER, color="#3A4E70", ls=":", lw=1, label=f"ε = {_EPS_STEER}")
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
    # RSS feed export
    try:
        from rss_feed import generate_rss_feed, scan_results_to_rss_items
        _rss_items = scan_results_to_rss_items(
            wl_result=st.session_state.get("wl_result"),
            wx_result=st.session_state.get("wx_result"),
            seis_result=st.session_state.get("seis_result"),
            dynamo_result=st.session_state.get("dynamo_result"),
        )
        _rss_xml = generate_rss_feed(_rss_items)
        st.markdown("#### 📡 RSS Feed")
        st.caption(f"{len(_rss_items)} active FAWP signal(s) in feed · Subscribe in any RSS reader")
        st.download_button("⬇ Download RSS feed (feed.xml)",
                           data=_rss_xml.encode(),
                           file_name="fawp_feed.xml",
                           mime="application/rss+xml",
                           key="rss_dl")
        with st.expander("Preview feed XML"):
            st.code(_rss_xml[:2000], language="xml")
    except Exception as _rsse:
        st.caption(f"RSS: {_rsse}")
    st.markdown("---")

    # ── HTML report ──────────────────────────────────────────────────────
    st.markdown(_sec("Download report"), unsafe_allow_html=True)
    if wl:
        _scan_ts = st.session_state.get("scan_timestamp", "scan")
        _exp_col1, _exp_col2 = st.columns(2)
        with _exp_col1:
            if st.button("Generate HTML report", key="gen_html_report", use_container_width=True):
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
        with _exp_col2:
            if st.button("Generate PDF report", key="gen_pdf_report", use_container_width=True):
                with st.spinner("Building PDF…"):
                    try:
                        import tempfile as _tmpf2, json as _json_pdf
                        from fawp_index.report import generate_report as _gen_report
                        _wl_dict = json.loads(wl.to_json_str()) if hasattr(wl, "to_json_str") else wl.to_dataframe().to_dict("records")
                        with _tmpf2.NamedTemporaryFile(suffix=".pdf", delete=False) as _tf_pdf:
                            _pdf_path = _gen_report(
                                result=_wl_dict,
                                output_path=_tf_pdf.name,
                                title=f"FAWP Finance Scan — {_scan_ts}",
                                mode="report",
                            )
                        _pdf_bytes = open(str(_pdf_path), "rb").read()
                        st.download_button(
                            "📥 Download PDF report",
                            data=_pdf_bytes,
                            file_name=f"fawp_scan_{_scan_ts.replace(' ','_')}.pdf",
                            mime="application/pdf",
                            key="dl_pdf_report",
                        )
                    except Exception as _pdf_err:
                        st.error(f"PDF failed: {_pdf_err}. Install: pip install reportlab")
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
