"""
FAWP Weather Scanner — Standalone Streamlit app.
Ralph Clayton (2026) · doi:10.5281/zenodo.18673949
"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

try:
    from share import share_button as _share_button
    _SHARE_OK = True
except Exception:
    _SHARE_OK = False
    def _share_button(*a, **k): pass

try:
    from weather_watchlist import (
        save_weather_location, render_weather_watchlist_panel,
        update_last_result,
    )
    _WL_OK = True
except Exception:
    _WL_OK = False

try:
    st.set_page_config(
        page_title="FAWP Weather Scanner",
        page_icon="🌦",
        layout="wide",
        initial_sidebar_state="expanded",
    )
except Exception:
    pass

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800;900&family=Space+Grotesk:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
  --bg:      #030810;
  --bg2:     #060D18;
  --card:    #091220;
  --card2:   #0C1828;
  --border:  #111E32;
  --border2: #1A2A42;
  --gold:    #F2C440;
  --gold2:   #C8A020;
  --gold-dim: rgba(242,196,64,.12);
  --red:     #E83030;
  --red-dim: rgba(232,48,48,.1);
  --green:   #22C468;
  --green-dim: rgba(34,196,104,.08);
  --blue:    #3888F8;
  --text:    #E8EDF8;
  --muted:   #5070A0;
  --dim:     #1E3050;
}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"], .main, .block-container {
  background: var(--bg) !important;
  font-family: 'Space Grotesk', sans-serif !important;
  color: var(--text) !important;
}
[data-testid="stMain"] .block-container {
  padding-top: 1.5rem !important;
  max-width: 1100px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--bg2) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { font-family: 'Space Grotesk', sans-serif !important; }
[data-testid="stSidebar"] label {
  color: var(--muted) !important;
  font-size: .75em !important;
  font-weight: 600 !important;
  letter-spacing: .05em !important;
  text-transform: uppercase !important;
}

/* Sidebar inputs */
[data-testid="stSidebar"] [data-testid="stTextInput"] input,
[data-testid="stSidebar"] [data-testid="stNumberInput"] input {
  background: var(--card) !important;
  border: 1px solid var(--border2) !important;
  border-radius: 8px !important;
  color: var(--text) !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: .88em !important;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
  background: var(--card) !important;
  border: 1px solid var(--border2) !important;
  border-radius: 8px !important;
  color: var(--text) !important;
}

/* Run Scan button — gold fill */
[data-testid="stSidebar"] [data-testid="stButton"] > button,
[data-testid="stButton"][data-key="run_btn"] > button {
  background: linear-gradient(135deg, #F2C440, #C8A020) !important;
  color: #030810 !important;
  font-family: 'Outfit', sans-serif !important;
  font-weight: 800 !important;
  font-size: .92em !important;
  letter-spacing: .05em !important;
  border: none !important;
  border-radius: 10px !important;
  padding: .65em 0 !important;
  width: 100% !important;
  box-shadow: 0 4px 24px rgba(242,196,64,.2) !important;
  transition: all .2s !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 32px rgba(242,196,64,.3) !important;
}

/* Mode radio */
[data-testid="stRadio"] label {
  color: var(--text) !important;
  font-size: .88em !important;
  font-weight: 500 !important;
}

/* Slider track */
[data-baseweb="slider"] [data-testid="stSliderTrack"] > div:first-child {
  background: var(--gold) !important;
}
[data-baseweb="slider"] div[role="slider"] {
  background: var(--gold) !important;
  border-color: var(--gold) !important;
  box-shadow: 0 0 0 3px rgba(242,196,64,.2) !important;
}

/* ── Metric cards ── */
.kpi-row { display:flex; gap:.75em; flex-wrap:wrap; margin:1.2em 0; }
.kpi-card {
  background: var(--card);
  border: 1px solid var(--border2);
  border-radius: 12px;
  padding: 1em 1.4em;
  flex: 1;
  min-width: 110px;
  text-align: center;
  position: relative;
  overflow: hidden;
}
.kpi-card::after {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, var(--gold), transparent);
}
.kpi-val {
  font-family: 'JetBrains Mono', monospace;
  font-size: 1.4em;
  font-weight: 600;
  color: var(--gold);
  letter-spacing: -.02em;
  line-height: 1.1;
}
.kpi-lbl {
  font-size: .65em;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: .1em;
  margin-top: .35em;
  font-weight: 600;
}

/* ── FAWP result banner ── */
.result-banner {
  border-radius: 14px;
  padding: 1.2em 1.6em;
  margin: 1em 0;
  display: flex;
  align-items: center;
  gap: 1em;
}
.result-banner.fawp-yes {
  background: var(--red-dim);
  border: 1px solid rgba(232,48,48,.25);
}
.result-banner.fawp-no {
  background: var(--green-dim);
  border: 1px solid rgba(34,196,104,.2);
}
.result-icon { font-size: 1.8em; }
.result-label {
  font-family: 'Outfit', sans-serif;
  font-weight: 800;
  font-size: 1.1em;
  letter-spacing: .02em;
}
.result-sub { color: var(--muted); font-size: .82em; margin-top: .2em; }

/* ── Hazard pill ── */
.hazard-pill {
  display: inline-flex; align-items: center; gap: .4em;
  padding: .3em .9em;
  border-radius: 8px;
  font-size: .8em; font-weight: 700;
  background: var(--gold-dim);
  color: var(--gold);
  border: 1px solid rgba(242,196,64,.2);
  letter-spacing: .02em;
  margin-right: .5em;
}

/* ── Action window bars ── */
.aw-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:.75em; margin:.8em 0; }
.aw-card {
  background: var(--card2);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: .8em 1em;
}
.aw-name { font-size:.7em; color:var(--muted); text-transform:uppercase; letter-spacing:.07em; font-weight:600; }
.aw-time { font-family:'JetBrains Mono',monospace; font-size:1.1em; color:var(--text); font-weight:600; margin:.2em 0; }
.aw-bar  { background:var(--border); border-radius:3px; height:5px; margin:.3em 0; }
.aw-fill { border-radius:3px; height:5px; }
.aw-pct  { font-size:.65em; color:var(--dim); }

/* ── Section header ── */
.wx-sec {
  font-family: 'Outfit', sans-serif;
  font-size: .72em;
  font-weight: 700;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: .14em;
  margin: 1.8em 0 .8em;
  padding-bottom: .4em;
  border-bottom: 1px solid var(--border);
}

/* ── Explanation box ── */
.exp-box {
  background: linear-gradient(135deg, var(--bg2), var(--card));
  border: 1px solid var(--border2);
  border-left: 3px solid var(--gold);
  border-radius: 12px;
  padding: 1.2em 1.6em;
  margin: .8em 0;
  font-size: .9em;
  color: #7090B8;
  line-height: 1.8;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
  border: 1px solid var(--border2) !important;
  border-radius: 12px !important;
  background: var(--card) !important;
  margin: .5em 0 !important;
}
details summary {
  color: var(--text) !important;
  font-weight: 600 !important;
  font-size: .9em !important;
}

/* ── Sidebar logo ── */
.wx-logo {
  padding: .8em 0 1.4em;
  border-bottom: 1px solid var(--border);
  margin-bottom: 1em;
}

/* ── Soft error / warning ── */
[data-testid="stAlert"] {
  border-radius: 10px !important;
  border-left-width: 3px !important;
}
div[data-baseweb="notification"] {
  background: rgba(240,160,32,.08) !important;
  border: 1px solid rgba(240,160,32,.3) !important;
  border-left: 3px solid #F0A020 !important;
  border-radius: 10px !important;
}

/* ── Pill tab toggle (Hazard/Variable) ── */
.pill-tabs { display:flex; gap:4px; background:var(--card); border:1px solid var(--border2);
             border-radius:10px; padding:3px; margin:.4em 0; }
.pill-tab  { flex:1; padding:.4em .8em; border-radius:8px; font-size:.8em; font-weight:700;
             text-align:center; cursor:pointer; color:var(--muted);
             transition:all .18s; letter-spacing:.03em; font-family:'Outfit',sans-serif; }
.pill-tab.active { background:var(--gold); color:#030810; }

/* ── Hazard-coloured pills ── */
.hz-heat  { --hz: #F04020; } .hz-rain { --hz: #2090E8; }
.hz-wind  { --hz: #A060F0; } .hz-ice  { --hz: #40C8F0; }
.hz-fire  { --hz: #F08020; } .hz-fog  { --hz: #8098B8; }
.hazard-pill { border-color: rgba(var(--hz-rgb),.3) !important; }

/* ── Degree suffix on inputs ── */
.deg-wrap { position:relative; display:inline-block; width:100%; }
.deg-sfx  { position:absolute; right:10px; top:50%; transform:translateY(-50%);
            color:var(--muted); font-size:.8em; pointer-events:none; }

/* ── City search combobox ── */
.city-chip {
  display:inline-flex; align-items:center; gap:.35em;
  padding:.22em .7em; border-radius:6px; font-size:.78em; font-weight:600;
  cursor:pointer; transition:all .15s;
  background:var(--card2); border:1px solid var(--border2); color:var(--muted);
  margin:.15em;
}
.city-chip:hover { border-color:var(--gold); color:var(--gold); }
.city-chip.active { background:var(--gold-dim); border-color:var(--gold); color:var(--gold); }

/* ── Mode selector ── */
.mode-pills { display:grid; grid-template-columns:1fr 1fr; gap:4px; margin-bottom:1em; }
.mode-pill  { padding:.45em .5em; border-radius:8px; font-size:.76em; font-weight:700;
              text-align:center; cursor:pointer; transition:all .18s;
              background:var(--card); border:1px solid var(--border); color:var(--muted);
              letter-spacing:.02em; }
.mode-pill.active { background:var(--gold-dim); border-color:var(--gold); color:var(--gold); }

.wx-logo-title {
  font-family: 'Outfit', sans-serif;
  font-size: 1.6em;
  font-weight: 900;
  color: var(--gold);
  letter-spacing: -.03em;
  line-height: 1;
}
.wx-logo-sub {
  font-size: .68em;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: .1em;
  margin-top: .4em;
}

/* ── Main header ── */
.wx-header {
  margin-bottom: 1.4em;
  padding-bottom: 1em;
  border-bottom: 1px solid var(--border);
}
.wx-header-title {
  font-family: 'Outfit', sans-serif;
  font-size: 2.2em;
  font-weight: 900;
  color: var(--gold);
  letter-spacing: -.04em;
  line-height: 1;
}
.wx-header-meta {
  color: var(--muted);
  font-size: .8em;
  margin-top: .4em;
}
.wx-header-meta a { color: var(--blue); text-decoration: none; }

/* ── Empty state ── */
.wx-empty {
  text-align: center;
  padding: 5em 2em;
  max-width: 500px;
  margin: 0 auto;
}
.wx-empty-icon { font-size: 3.5em; margin-bottom: .5em; opacity: .6; }
.wx-empty-title {
  font-family: 'Outfit', sans-serif;
  font-size: 1.3em;
  font-weight: 800;
  color: var(--text);
  margin-bottom: .5em;
}
.wx-empty-sub { color: var(--muted); font-size: .88em; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────────────────────
HAZARDS = {
    "🌡 Heat":         {"var": "temperature_2m",     "desc": "Extreme heat window",       "color": "#F04020", "bg": "rgba(240,64,32,.1)"},
    "🌧 Heavy Rain":   {"var": "precipitation_sum",   "desc": "Flood/disruption window",   "color": "#2090E8", "bg": "rgba(32,144,232,.1)"},
    "💨 Wind":         {"var": "wind_speed_10m",      "desc": "Storm/grid disruption",     "color": "#A060F0", "bg": "rgba(160,96,240,.1)"},
    "❄ Winter Ice":   {"var": "temperature_2m",      "desc": "Ice/road hazard window",    "color": "#40C8F0", "bg": "rgba(64,200,240,.1)"},
    "☀ Fire Weather": {"var": "shortwave_radiation", "desc": "High fire risk window",     "color": "#F08020", "bg": "rgba(240,128,32,.1)"},
    "🌫 Cloud/Fog":   {"var": "cloud_cover",         "desc": "Visibility disruption",     "color": "#8098B8", "bg": "rgba(128,152,184,.1)"},
}
ACTION_WINDOWS = {
    "🌡 Heat":         {"Evacuation": 72, "Grid prep": 48, "Staffing": 24, "Cooling": 12},
    "🌧 Heavy Rain":   {"Evacuation": 48, "Drainage": 24, "Road closure": 12, "Pumps": 6},
    "💨 Wind":         {"Grid prep": 36, "Road clear": 24, "Staffing": 12, "Shelter": 6},
    "❄ Winter Ice":   {"Road treat": 24, "Gritting": 12, "Travel ban": 6, "Emergency": 3},
    "☀ Fire Weather": {"Evacuation": 48, "Firebreak": 24, "Resources": 12, "Contain": 6},
    "🌫 Cloud/Fog":   {"Aviation": 12, "Road warn": 6, "Marine": 6, "Advisory": 3},
}
VARIABLES = {
    "temperature_2m":      "🌡 Temperature 2m (°C)",
    "precipitation_sum":   "🌧 Precipitation (mm/day)",
    "wind_speed_10m":      "💨 Wind Speed 10m (m/s)",
    "surface_pressure":    "⬇ Surface Pressure (hPa)",
    "cloud_cover":         "☁ Cloud Cover (%)",
    "shortwave_radiation": "☀ Shortwave Radiation (W/m²)",
}
PRESETS = {
    "London":   (51.50, -0.10), "New York": (40.71,-74.01),
    "Tokyo":    (35.69,139.69), "Sydney":  (-33.87,151.21),
    "Paris":    (48.86,  2.35), "Dubai":    (25.20, 55.27),
    "Chicago":  (41.88,-87.63), "Mumbai":   (19.08, 72.88),
}

def _fmt_loc(lat, lon):
    return f"({abs(lat):.2f}{'N' if lat>=0 else 'S'}, {abs(lon):.2f}{'E' if lon>=0 else 'W'})"

def _kpi_html(val, lbl, color="var(--gold)"):
    return (f'<div class="kpi-card">'
            f'<div class="kpi-val" style="color:{color}">{val}</div>'
            f'<div class="kpi-lbl">{lbl}</div></div>')

def _kpi(col, val, lbl, color="var(--gold)"):
    col.markdown(_kpi_html(val, lbl, color), unsafe_allow_html=True)

def _action_window_panel(hazard, odw_start, odw_end, hazard_color="#F2C440"):
    windows = ACTION_WINDOWS.get(hazard, {})
    if not windows: return
    st.markdown('<div class="wx-sec">⏱ Action Windows</div>', unsafe_allow_html=True)
    odw_days = (odw_end - odw_start + 1) if (odw_start and odw_end) else None
    cards = ""
    for action, hours in windows.items():
        days = hours / 24
        pct  = min(1.0, odw_days / days) if odw_days else 0
        color = hazard_color if pct > 0.6 else ("#F2C440" if pct > 0.3 else "#E83030")
        fill  = int(pct * 100)
        cards += (f'<div class="aw-card">'
                  f'<div class="aw-name">{action}</div>'
                  f'<div class="aw-time" style="color:{color}">{hours}h</div>'
                  f'<div class="aw-bar"><div class="aw-fill" style="width:{fill}%;background:{color}"></div></div>'
                  f'<div class="aw-pct">{fill}% usable</div>'
                  f'</div>')
    st.markdown(f'<div class="aw-grid">{cards}</div>', unsafe_allow_html=True)

def _explanation_panel(r, hazard):
    h_info = HAZARDS.get(hazard, {})
    if r.fawp_found:
        text = (
            f"<b style='color:var(--text)'>Forecast skill is present — the window to act is closing.</b><br><br>"
            f"The ERA5 signal for <b>{VARIABLES.get(r.variable, r.variable)}</b> at {r.location} shows "
            f"that the atmosphere remains predictable at lags beyond τ={r.odw_result.tau_h_plus}. "
            f"However, the ability to influence outcomes through intervention has already collapsed at the same lag.<br><br>"
            f"<b style='color:var(--gold)'>Leverage gap: {r.peak_gap_bits:.4f} bits.</b> &nbsp;"
            f"<b>ODW: τ = {r.odw_start}–{r.odw_end}.</b><br><br>"
            f"In plain terms: you can see it coming better than you can still change it. "
            f"{'<br><b>' + h_info.get('desc','') + '</b> — response capacity is limited.' if hazard else ''}"
        )
    else:
        text = (
            f"<b style='color:var(--text)'>No FAWP regime detected.</b><br><br>"
            f"Predictive and steering coupling collapse together — no persistent window "
            f"where forecasts remain useful while interventions have failed.<br><br>"
            f"Peak gap: {r.peak_gap_bits:.4f} bits. Try a longer date range, different variable, or lower ε."
        )
    st.markdown(f'<div class="exp-box">{text}</div>', unsafe_allow_html=True)

def _mi_chart(r, epsilon):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if len(r.tau) == 0: return
        fig, ax = plt.subplots(figsize=(11, 3.2))
        fig.patch.set_facecolor("#030810")
        ax.set_facecolor("#091220")
        ax.plot(r.tau, r.pred_mi,  color="#F2C440", lw=2.2, label="Prediction MI", zorder=3)
        ax.plot(r.tau, r.steer_mi, color="#3888F8", lw=1.6, ls="--",
                label="Steering MI", alpha=.85, zorder=2)
        ax.axhline(epsilon, color="#1E3050", ls=":", lw=1.2, label=f"ε={epsilon}")
        if r.fawp_found and r.odw_start is not None:
            ax.axvspan(r.odw_start, r.odw_end, alpha=.15, color="#E83030",
                       label=f"ODW τ={r.odw_start}–{r.odw_end}", zorder=1)
            if r.odw_result.tau_f is not None:
                ax.axvline(r.odw_result.tau_f, color="#E83030",
                           lw=1.2, ls=":", alpha=.6)
        ax.set_xlabel("τ (delay, days)", fontsize=8, color="#5070A0")
        ax.set_ylabel("MI (bits)", fontsize=8, color="#5070A0")
        ax.tick_params(colors="#5070A0", labelsize=7)
        ax.legend(fontsize=7.5, facecolor="#091220", labelcolor="#E8EDF8",
                  edgecolor="#1A2A42", framealpha=.9)
        for sp in ax.spines.values(): sp.set_edgecolor("#111E32")
        ax.grid(axis="y", color="#111E32", alpha=.6, lw=.5)
        plt.tight_layout(pad=.4)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    except Exception:
        pass

def _multi_scan_panel():
    st.markdown('<div class="wx-sec">🗺 Multi-Location Scan</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([3, 1])
    with c1:
        cities_input = st.text_area(
            "Locations (name, lat, lon — one per line)",
            "London, 51.5, -0.1\nParis, 48.9, 2.4\nNew York, 40.7, -74.0\nTokyo, 35.7, 139.7\nSydney, -33.9, 151.2",
            height=130,
        )
    with c2:
        ms_var   = st.selectbox("Variable", list(VARIABLES.keys()),
                                format_func=lambda k: VARIABLES[k], key="ms_var")
        ms_start = st.text_input("Start", "2015-01-01", key="ms_start")
        ms_end   = st.text_input("End",   "2024-12-31", key="ms_end")
        ms_null  = st.slider("Null perms", 0, 100, 20, key="ms_null")
        run_multi = st.button("▶ Scan All", type="primary",
                              use_container_width=True, key="run_multi")
    if run_multi:
        locs = []
        for line in cities_input.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                try:
                    locs.append({"name": parts[0],
                                 "lat": float(parts[1]),
                                 "lon": float(parts[2])})
                except ValueError:
                    st.warning(f"Skipping: {line}")
        if not locs:
            st.error("No valid locations.")
        else:
            try:
                from fawp_index.weather import scan_weather_grid
            except ImportError as e:
                st.error(f"fawp-index not available: {e}")
                return
            with st.spinner(f"Scanning {len(locs)} locations…"):
                try:
                    results = scan_weather_grid(
                        locations=locs, variable=ms_var,
                        start_date=ms_start, end_date=ms_end,
                        horizon_days=7, tau_max=30, n_null=ms_null,
                    )
                    st.session_state["ms_results"] = results
                except Exception as e:
                    st.error(f"Scan failed: {e}")

    if "ms_results" in st.session_state:
        results = st.session_state["ms_results"]
        rows = [{"Location": r.location,
                 "FAWP": "🔴 YES" if r.fawp_found else "—",
                 "Gap (bits)": f"{r.peak_gap_bits:.4f}",
                 "ODW": f"τ {r.odw_start}–{r.odw_end}" if r.fawp_found else "—",
                 "τ⁺ₕ": str(r.odw_result.tau_h_plus) if r.odw_result.tau_h_plus is not None else "—"}
                for r in results]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        try:
            from fawp_index.weather import plot_weather_map
            fig_map = plot_weather_map(results, title=f"FAWP Scan — {VARIABLES.get(ms_var, ms_var)}")
            st.plotly_chart(fig_map, use_container_width=True)
        except ImportError:
            st.caption("Install plotly for map view.")
        except Exception:
            pass
        import json as _j
        st.download_button("Download JSON",
            data=_j.dumps([r.to_dict() for r in results], indent=2).encode(),
            file_name=f"fawp_multiscan_{ms_var}.json", mime="application/json")

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
<div class="wx-logo">
  <div class="wx-logo-title">🌦 FAWP Weather</div>
  <div class="wx-logo-sub">Information-Control Exclusion Principle</div>
</div>
""", unsafe_allow_html=True)

    mode = st.radio("Mode", ["Single location", "Compare two locations",
                              "Upload NWP data", "Multi-location scan"],
                    label_visibility="collapsed")

    if mode == "Single location":
        st.markdown("**Location**")
        if "wx_lat" not in st.session_state: st.session_state["wx_lat"] = 51.5
        if "wx_lon" not in st.session_state: st.session_state["wx_lon"] = -0.1
        # City chips
        st.markdown('<div style="margin:.3em 0 .5em;line-height:2.2">' +
            "".join(f'<span style="display:inline-flex;align-items:center;padding:.2em .65em;'
                   f'border-radius:6px;font-size:.75em;font-weight:600;cursor:pointer;margin:.1em;'
                   f'background:var(--card2);border:1px solid var(--border2);color:var(--muted)">'
                   f'{city}</span>'
                   for city in PRESETS.keys()) +
            '</div>', unsafe_allow_html=True)
        # City preset buttons — one click fills lat/lon AND triggers scan
        _pcols = st.columns(len(PRESETS))
        for _pi, (_pcity, (_plat, _plon)) in enumerate(PRESETS.items()):
            with _pcols[_pi]:
                if st.button(_pcity, key=f"cpbtn_{_pi}", use_container_width=True,
                             help=f"{_plat:.2f}°, {_plon:.2f}°"):
                    st.session_state["wx_lat"] = _plat
                    st.session_state["wx_lon"] = _plon
                    st.session_state["_city_auto_run"] = True
                    st.rerun()
        # City name autocomplete (Nominatim / OpenStreetMap — no API key)
        _city_q = st.text_input("🔍 City search (or enter coordinates below)",
                                placeholder="e.g. Tokyo, Buenos Aires, Mumbai…",
                                key="wx_city_q")
        if _city_q and len(_city_q) >= 3:
            if st.button("Search", key="wx_city_search", use_container_width=True):
                try:
                    import urllib.request, json as _j, urllib.parse
                    _url = ("https://nominatim.openstreetmap.org/search"
                            f"?q={urllib.parse.quote(_city_q)}&format=json&limit=5")
                    _req = urllib.request.Request(_url,
                           headers={"User-Agent": "fawp-scanner/" + __import__("fawp_index").__version__})
                    with urllib.request.urlopen(_req, timeout=4) as _r:
                        _hits = _j.loads(_r.read())
                    if _hits:
                        st.session_state["wx_geocode_results"] = _hits
                    else:
                        st.warning("No results — try a different city name.")
                except Exception as _ge:
                    st.warning(f"Geocode unavailable: {_ge}")

        if "wx_geocode_results" in st.session_state:
            _hits = st.session_state["wx_geocode_results"]
            _labels = [f"{h.get('display_name','')[:60]}" for h in _hits]
            _sel_idx = st.selectbox("Select location", range(len(_labels)),
                                    format_func=lambda i: _labels[i],
                                    key="wx_geocode_sel")
            if st.button("Use this location", key="wx_geocode_use", use_container_width=True):
                _chosen = _hits[_sel_idx]
                st.session_state["wx_lat"] = float(_chosen["lat"])
                st.session_state["wx_lon"] = float(_chosen["lon"])
                del st.session_state["wx_geocode_results"]
                st.rerun()

        lat = st.number_input("Latitude °N/S",  min_value=-90.0,  max_value=90.0,  step=0.1, format="%.2f", key="wx_lat")
        lon = st.number_input("Longitude °E/W", min_value=-180.0, max_value=180.0, step=0.1, format="%.2f", key="wx_lon")

        # Use my location button
        st.components.v1.html("""
<button onclick="
  navigator.geolocation.getCurrentPosition(function(p){
    window.parent.postMessage({type:'streamlit:setComponentValue',
      value:{lat:p.coords.latitude,lon:p.coords.longitude}}, '*');
    document.getElementById('geo-btn').textContent='✓ Location set — adjust above if needed';
  }, function(){ document.getElementById('geo-btn').textContent='Location unavailable'; });
" id='geo-btn' style='
  background:transparent;border:1px solid #1A2A42;color:#5070A0;
  padding:.3em .8em;border-radius:6px;font-size:.75em;cursor:pointer;
  font-family:Space Grotesk,sans-serif;width:100%;margin-bottom:.4em;
  transition:all .2s;
' onmouseover="this.style.borderColor='#F2C440';this.style.color='#F2C440'"
  onmouseout="this.style.borderColor='#1A2A42';this.style.color='#5070A0'">
  📍 Use my location
</button>""", height=40)
        st.markdown("**Hazard / Variable**")
        scan_mode = st.radio("", ["Hazard", "Variable"], horizontal=True, key="scan_mode")
        if scan_mode == "Hazard":
            hazard   = st.selectbox("Hazard", list(HAZARDS.keys()))
            variable = HAZARDS[hazard]["var"]
            st.caption(f"→ {VARIABLES.get(variable, variable)}")
        else:
            hazard   = None
            variable = st.selectbox("Variable", list(VARIABLES.keys()),
                                    format_func=lambda k: VARIABLES[k])

        st.markdown("**Period & settings**")
        start_date   = st.text_input("Start date", "2010-01-01")
        end_date     = st.text_input("End date",   "2024-12-31")
        horizon_days = st.slider("Forecast horizon (days)", 1, 30, 7)
        tau_max      = st.slider("Max tau", 5, 60, 30, step=5)
        n_null       = st.slider("Null permutations", 0, 200, 50, step=10)
        st.toggle("Scan ALL variables", key="wx_multi_mode",
                  help="Runs FAWP on all ERA5 variables in sequence.")
        estimator    = st.selectbox("MI estimator", ["pearson", "knn"],
                         help="pearson: fast, Gaussian. knn: non-parametric, better for non-Gaussian data (see E9 methods). Requires scikit-learn.")
        remove_anomaly = st.toggle("Anomaly mode",
                          help="Subtract climatological mean before detection — separates "
                               "FAWP-in-trend from FAWP-in-variability. Uses 365-day "
                               "rolling mean as baseline.", key="wx_anomaly_mode")
        epsilon      = st.number_input("Epsilon (bits)", value=0.01,
                                       min_value=0.001, max_value=0.1,
                                       step=0.001, format="%.3f")
        run_btn = st.button("▶ Run Scan", type="primary",
                            use_container_width=True, key="run_btn")

    st.sidebar.markdown("---")
    with st.sidebar.expander("📅 Compare two time periods"):
        st.caption("Detect how FAWP changed between two eras — useful for climate change.")
        _pc_start_a = st.text_input("Period A start", "1990-01-01", key="pc_start_a")
        _pc_end_a   = st.text_input("Period A end",   "2004-12-31", key="pc_end_a")
        _pc_start_b = st.text_input("Period B start", "2010-01-01", key="pc_start_b")
        _pc_end_b   = st.text_input("Period B end",   "2024-12-31", key="pc_end_b")
        _pc_run     = st.button("▶ Compare periods", key="pc_run", use_container_width=True)
        if _pc_run:
            try:
                from fawp_index.weather import fawp_from_open_meteo as _fom_pc
                with st.spinner("Scanning Period A…"):
                    _res_a = _fom_pc(latitude=lat, longitude=lon, variable=variable,
                                     start_date=_pc_start_a, end_date=_pc_end_a,
                                     horizon_days=horizon_days, tau_max=tau_max,
                                     epsilon=epsilon, n_null=n_null)
                with st.spinner("Scanning Period B…"):
                    _res_b = _fom_pc(latitude=lat, longitude=lon, variable=variable,
                                     start_date=_pc_start_b, end_date=_pc_end_b,
                                     horizon_days=horizon_days, tau_max=tau_max,
                                     epsilon=epsilon, n_null=n_null)
                st.session_state["wx_period_a"] = _res_a
                st.session_state["wx_period_b"] = _res_b
            except Exception as _pce:
                st.error(f"Period comparison failed: {_pce}")
    if "wx_period_a" in st.session_state and "wx_period_b" in st.session_state:
        _ra = st.session_state["wx_period_a"]
        _rb = st.session_state["wx_period_b"]
        st.markdown('<div class="wx-sec">Period comparison</div>', unsafe_allow_html=True)
        _pc1, _pc2 = st.columns(2)
        for _col, _r, _lbl in [(_pc1, _ra, "Period A"), (_pc2, _rb, "Period B")]:
            with _col:
                _bc = "fawp-yes" if _r.fawp_found else "fawp-no"
                st.markdown(
                    f'<div class="result-banner {_bc}" style="padding:.5em .8em">'
                    f'<b>{_lbl}</b>: {"🔴 FAWP" if _r.fawp_found else "✅ Clear"} · '
                    f'gap {_r.peak_gap_bits:.4f}b</div>', unsafe_allow_html=True)
        import matplotlib.pyplot as _plt_pc, io as _io_pc
        _fig_pc, _ax_pc = _plt_pc.subplots(figsize=(9, 3.5), facecolor="#0D1729")
        _ax_pc.set_facecolor("#07101E")
        _ax_pc.plot(_ra.tau, _ra.pred_mi, color="#D4AF37", lw=2, label="Period A")
        _ax_pc.plot(_rb.tau, _rb.pred_mi, color="#4A7FCC", lw=2, ls="--", label="Period B")
        _ax_pc.axhline(epsilon, color="#3A4E70", ls=":", lw=1)
        _ax_pc.set_xlabel("τ (delay, days)", fontsize=8, color="#7A90B8")
        _ax_pc.set_ylabel("MI (bits)", fontsize=8, color="#7A90B8")
        _ax_pc.set_title(f"Period comparison — {variable}", color="#D4AF37", fontsize=9)
        _ax_pc.legend(fontsize=8, framealpha=0.2)
        _fig_pc.tight_layout()
        st.pyplot(_fig_pc, use_container_width=True)
        _plt_pc.close(_fig_pc)
        st.markdown("---")
        st.caption("ERA5 reanalysis · Open-Meteo · free · no API key")

    elif mode == "Compare two locations":
        st.markdown("**Location A**")
        if "cmp_lat_a" not in st.session_state: st.session_state["cmp_lat_a"] = 51.5
        if "cmp_lon_a" not in st.session_state: st.session_state["cmp_lon_a"] = -0.1
        if "cmp_lat_b" not in st.session_state: st.session_state["cmp_lat_b"] = 48.9
        if "cmp_lon_b" not in st.session_state: st.session_state["cmp_lon_b"] = 2.4
        cmp_name_a = st.text_input("Name A", "London", key="cmp_name_a")
        cmp_lat_a  = st.number_input("Lat A", min_value=-90.0,  max_value=90.0,  step=0.1, format="%.2f", key="cmp_lat_a")
        cmp_lon_a  = st.number_input("Lon A", min_value=-180.0, max_value=180.0, step=0.1, format="%.2f", key="cmp_lon_a")
        st.markdown("**Location B**")
        cmp_name_b = st.text_input("Name B", "Paris", key="cmp_name_b")
        cmp_lat_b  = st.number_input("Lat B", min_value=-90.0,  max_value=90.0,  step=0.1, format="%.2f", key="cmp_lat_b")
        cmp_lon_b  = st.number_input("Lon B", min_value=-180.0, max_value=180.0, step=0.1, format="%.2f", key="cmp_lon_b")
        st.markdown("**Settings**")
        cmp_var   = st.selectbox("Variable", list(VARIABLES.keys()), format_func=lambda k: VARIABLES[k], key="cmp_var")
        cmp_start = st.text_input("Start", "2010-01-01", key="cmp_start")
        cmp_end   = st.text_input("End",   "2024-12-31", key="cmp_end")
        cmp_horiz = st.slider("Horizon (days)", 1, 30, 7, key="cmp_horiz")
        cmp_null  = st.slider("Null perms", 0, 100, 30, key="cmp_null")
        run_cmp   = st.button("▶ Compare", type="primary", use_container_width=True, key="run_cmp")

    elif mode == "Upload NWP data":
        st.markdown("**Forecast CSV**")
        nwp_fc_file  = st.file_uploader("Forecast (.csv)", type=["csv"], key="nwp_fc")
        nwp_fc_col   = st.text_input("Forecast column", "forecast", key="nwp_fc_col")
        st.markdown("**Observation CSV**")
        nwp_obs_file = st.file_uploader("Observation (.csv)", type=["csv"], key="nwp_obs")
        nwp_obs_col  = st.text_input("Observed column", "observed", key="nwp_obs_col")
        nwp_date_col = st.text_input("Date column", "date", key="nwp_date_col")
        nwp_intv_col = st.text_input("Intervention col (optional)", "", key="nwp_intv_col")
        nwp_var      = st.text_input("Variable label", "temperature_2m", key="nwp_var")
        nwp_loc      = st.text_input("Location label", "uploaded", key="nwp_loc")
        nwp_null     = st.slider("Null perms", 0, 200, 50, key="nwp_null")
        run_nwp      = st.button("▶ Run FAWP on uploaded data", type="primary",
                                  use_container_width=True, key="run_nwp")

# ── Main header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="wx-header">
  <div class="wx-header-title">🌦 FAWP Weather Scanner</div>
  <div class="wx-header-meta">
    Real ERA5 reanalysis ·
    <a href="https://open-meteo.com">Open-Meteo</a> ·
    <a href="https://fawp-scanner.info">fawp-scanner.info</a>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Multi-location mode ───────────────────────────────────────────────────────
if mode == "Multi-location scan":
    _multi_scan_panel()
    st.stop()

# ── Compare mode ──────────────────────────────────────────────────────────────
if mode == "Compare two locations":
    if run_cmp:
        try:
            from fawp_index.weather import compare_locations
            with st.spinner(f"Scanning {cmp_name_a} and {cmp_name_b}…"):
                r_a, r_b = compare_locations(
                    {"lat": cmp_lat_a, "lon": cmp_lon_a, "name": cmp_name_a},
                    {"lat": cmp_lat_b, "lon": cmp_lon_b, "name": cmp_name_b},
                    variable=cmp_var, start_date=cmp_start, end_date=cmp_end,
                    horizon_days=cmp_horiz, n_null=cmp_null,
                )
                st.session_state["cmp_r_a"] = r_a
                st.session_state["cmp_r_b"] = r_b
        except Exception as e:
            st.error(f"Compare failed: {e}")

    if "cmp_r_a" in st.session_state:
        r_a = st.session_state["cmp_r_a"]
        r_b = st.session_state["cmp_r_b"]
        col_a, col_b = st.columns(2)
        for col, r in [(col_a, r_a), (col_b, r_b)]:
            with col:
                badge_cls = "fawp-yes" if r.fawp_found else "fawp-no"
                badge_ico = "🔴 FAWP" if r.fawp_found else "✅ Clear"
                st.markdown(
                    f'<div class="result-banner {badge_cls}">'
                    f'<div class="result-label">{r.location}</div>'
                    f'<div class="result-label" style="margin-left:auto">{badge_ico}</div>'
                    f'</div>', unsafe_allow_html=True)
                kc = st.columns(3)
                _kpi(kc[0], f"{r.peak_gap_bits:.4f}", "Gap (bits)")
                _kpi(kc[1], str(r.odw_result.tau_h_plus) if r.odw_result.tau_h_plus is not None else "—", "τ⁺ₕ")
                _kpi(kc[2], f"τ{r.odw_start}–{r.odw_end}" if r.fawp_found else "—", "ODW")
                _mi_chart(r, 0.01)
    elif not run_cmp:
        st.markdown("""<div class="wx-empty">
  <div class="wx-empty-icon">⚖</div>
  <div class="wx-empty-title">Compare two locations</div>
  <div class="wx-empty-sub">Fill in both locations in the sidebar and click ▶ Compare.</div>
</div>""", unsafe_allow_html=True)
    st.stop()

# ── NWP upload mode ───────────────────────────────────────────────────────────
if mode == "Upload NWP data":
    if run_nwp:
        if not nwp_fc_file or not nwp_obs_file:
            st.error("Upload both forecast and observation CSV files.")
        else:
            import tempfile, os
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
                f.write(nwp_fc_file.read()); fc_tmp = f.name
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
                f.write(nwp_obs_file.read()); obs_tmp = f.name
            try:
                from fawp_index.weather import fawp_from_nwp_csvs
                with st.spinner("Running FAWP on uploaded data…"):
                    nr = fawp_from_nwp_csvs(
                        forecast_path=fc_tmp, observed_path=obs_tmp,
                        forecast_col=nwp_fc_col, observed_col=nwp_obs_col,
                        date_col=nwp_date_col,
                        intervention_col=nwp_intv_col if nwp_intv_col.strip() else None,
                        variable=nwp_var, location=nwp_loc, n_null=nwp_null,
                    )
                    st.session_state["nwp_result"] = nr
            except Exception as e:
                st.error(f"Failed: {e}")
            finally:
                os.unlink(fc_tmp); os.unlink(obs_tmp)

    if "nwp_result" in st.session_state:
        nr = st.session_state["nwp_result"]
        bc = "fawp-yes" if nr.fawp_found else "fawp-no"
        bi = "🔴 FAWP DETECTED" if nr.fawp_found else "✅ No FAWP"
        st.markdown(f'<div class="result-banner {bc}"><div class="result-label">{bi}</div></div>',
                    unsafe_allow_html=True)
        kc = st.columns(4)
        _kpi(kc[0], f"{nr.peak_gap_bits:.4f}", "Peak gap (bits)")
        _kpi(kc[1], str(nr.odw_result.tau_h_plus) if nr.odw_result.tau_h_plus is not None else "—", "τ⁺ₕ")
        _kpi(kc[2], f"τ{nr.odw_start}–{nr.odw_end}" if nr.fawp_found else "—", "ODW")
        _kpi(kc[3], f"{nr.n_obs:,}", "Observations")
        st.markdown('<div class="wx-sec">MI Curves</div>', unsafe_allow_html=True)
        _mi_chart(nr, 0.01)
        import json as _j
        st.download_button("Download JSON", data=_j.dumps(nr.to_dict(), indent=2).encode(),
                           file_name=f"fawp_nwp_{nr.variable}.json", mime="application/json")
    else:
        st.markdown("""<div class="wx-empty">
  <div class="wx-empty-icon">📂</div>
  <div class="wx-empty-title">Upload NWP forecast data</div>
  <div class="wx-empty-sub">Upload your forecast and observation CSVs in the sidebar.
  FAWP is computed on real model skill — no ERA5 proxy.</div>
</div>""", unsafe_allow_html=True)
        with st.expander("📋 Expected CSV format"):
            st.markdown("""**Forecast CSV:**
```
date,forecast,ensemble_spread
2020-01-01,12.3,0.8
2020-01-02,13.1,0.9
```
**Observation CSV:**
```
date,observed
2020-01-01,12.8
2020-01-02,13.5
```""")
    st.stop()

# ── Single location scan ──────────────────────────────────────────────────────
if run_btn or st.session_state.pop("_city_auto_run", False):
    if st.session_state.get("wx_multi_mode"):
        _MV_VARS = ["temperature_2m","precipitation","wind_speed_10m",
                    "surface_pressure","cloud_cover"]
        from fawp_index.weather import fawp_from_open_meteo as _fom_mv
        _mv_res, _mv_pg = {}, st.progress(0.0)
        for _mvi, _mvv in enumerate(_MV_VARS):
            _mv_pg.progress((_mvi+1)/len(_MV_VARS), f"{_mvv}…")
            try: _mv_res[_mvv] = _fom_mv(latitude=lat, longitude=lon, variable=_mvv,
                    start_date=start_date, end_date=end_date, horizon_days=horizon_days,
                    tau_max=tau_max, epsilon=epsilon, n_null=n_null)
            except Exception: pass
        _mv_pg.empty()
        st.session_state["wx_multi_results"] = _mv_res
        st.rerun()
    try:
        import openmeteo_requests
    except ImportError:
        st.error("Install: `pip install openmeteo-requests requests-cache retry-requests`")
        st.stop()
    try:
        from fawp_index.weather import fawp_from_open_meteo
    except ImportError as e:
        st.error(f"fawp-index not installed: {e}")
        st.stop()
    with st.spinner(f"Fetching ERA5 {variable} @ {_fmt_loc(lat, lon)}…"):
        try:
            # Anomaly mode: pre-subtract climatological mean
            if st.session_state.get("wx_anomaly_mode", False):
                try:
                    from fawp_index.weather import (
                        _fetch_openmeteo_daily_series, _compute_weather_mi_curves,
                        WeatherFAWPResult, _deseasonalise,
                    )
                    import requests_cache, openmeteo_requests
                    from retry_requests import retry
                    _cache = requests_cache.CachedSession(".fawp_wx_cache", expire_after=3600)
                    _session = retry(_cache, retries=5, backoff_factor=0.2)
                    _om = openmeteo_requests.Client(session=_session)
                    _times, _vals = _fetch_openmeteo_daily_series(
                        _om, lat, lon, start_date, end_date, variable)
                    import numpy as _np_wx
                    _anom = _deseasonalise(_vals, period=365)
                    _n = len(_anom) - horizon_days
                    _pred   = _anom[:_n]
                    _future = _anom[horizon_days:horizon_days + _n]
                    _steer  = _np_wx.diff(_anom)[:_n]
                    _odw, _tau, _pred_mi, _steer_mi = _compute_weather_mi_curves(
                        _pred, _future, _steer, tau_max=tau_max, epsilon=epsilon, n_null=n_null)
                    result = WeatherFAWPResult(
                        variable=variable + " (anomaly)", location=f"({lat:.2f},{lon:.2f})",
                        odw_result=_odw, tau=_tau, pred_mi=_pred_mi, steer_mi=_steer_mi,
                        skill_metric="MI", n_obs=_n, horizon_days=horizon_days,
                        date_range=(start_date, end_date), metadata={"mode": "anomaly"})
                except Exception as _ae:
                    st.error(f"Anomaly mode failed: {_ae} — falling back to raw scan")
                    result = fawp_from_open_meteo(
                        latitude=lat, longitude=lon, variable=variable,
                        start_date=start_date, end_date=end_date,
                        horizon_days=horizon_days, tau_max=tau_max,
                        epsilon=epsilon, n_null=n_null, estimator=estimator,
                    )
            else:
                result = fawp_from_open_meteo(
                    latitude=lat, longitude=lon, variable=variable,
                    start_date=start_date, end_date=end_date,
                    horizon_days=horizon_days, tau_max=tau_max,
                    epsilon=epsilon, n_null=n_null,
                    estimator=estimator,
                )
            st.session_state["wx_result"]  = result
            st.session_state["wx_hazard"]  = hazard
            st.session_state["wx_epsilon"] = epsilon
        except Exception as e:
            st.error(f"Scan failed: {e}")
            st.stop()

if st.session_state.get("wx_multi_mode") and "wx_multi_results" in st.session_state:
    st.markdown('<div class="wx-sec">Multi-variable scan results</div>', unsafe_allow_html=True)
    _mvr = st.session_state["wx_multi_results"]
    import pandas as _pd_mv
    _mv_rows = [{"Variable": _vn,
                 "FAWP": "🔴 YES" if _vr.fawp_found else "✅ No",
                 "Peak gap": f"{_vr.peak_gap_bits:.4f}b",
                 "τ⁺ₕ": str(_vr.odw_result.tau_h_plus if odw_result.tau_h_plus is not None else "—"),
                 "n obs": _vr.n_obs}
                for _vn, _vr in _mvr.items()]
    st.dataframe(_pd_mv.DataFrame(_mv_rows).sort_values("Peak gap", ascending=False),
                 use_container_width=True, hide_index=True)
    _n_fawp_mv = sum(1 for _vr in _mvr.values() if _vr.fawp_found)
    st.error(f"🔴 {_n_fawp_mv}/{len(_mvr)} variables in FAWP") if _n_fawp_mv else     st.success(f"✅ No FAWP across {len(_mvr)} variables")
    # ERA5 variable comparison chart
    try:
        import matplotlib.pyplot as _plt_mv
        _fig_mv, (_ax_p, _ax_s) = _plt_mv.subplots(1, 2, figsize=(11, 3.5), facecolor="#0D1729")
        _MVC = ["#D4AF37","#4A7FCC","#1DB954","#C0111A","#FF8C00"]
        for _ax in (_ax_p, _ax_s):
            _ax.set_facecolor("#07101E")
            for _sp in _ax.spines.values(): _sp.set_edgecolor("#3A4E70")
            _ax.tick_params(colors="#7A90B8", labelsize=8)
        for _mvi, (_mvv, _mvr2) in enumerate(_mvr.items()):
            _c = _MVC[_mvi % len(_MVC)]
            _ax_p.plot(_mvr2.tau, _mvr2.pred_mi,  color=_c, lw=1.5, alpha=0.85, label=_mvv.replace("_"," "))
            _ax_s.plot(_mvr2.tau, _mvr2.steer_mi, color=_c, lw=1.2, ls="--", alpha=0.7)
        for _ax in (_ax_p, _ax_s):
            _ax.axhline(epsilon, color="#3A4E70", ls=":", lw=1)
            _ax.set_xlabel("τ (delay, days)", fontsize=8, color="#7A90B8")
            _ax.set_ylabel("MI (bits)", fontsize=8, color="#7A90B8")
        _ax_p.set_title("Prediction MI — all variables", color="#D4AF37", fontsize=9, fontweight="bold")
        _ax_s.set_title("Steering MI — all variables", color="#4A7FCC", fontsize=9, fontweight="bold")
        _ax_p.legend(fontsize=7, framealpha=0.2)
        _fig_mv.tight_layout(); st.pyplot(_fig_mv, use_container_width=True)
        _plt_mv.close(_fig_mv)
    except Exception as _mve:
        st.caption(f"Comparison chart: {_mve}")
    st.markdown("---")

# ── FAWP global weather map ───────────────────────────────────────────────────
with st.expander("🌍 FAWP Global Weather Map (coarse grid scan)", expanded=False):
    st.caption("Scan a lat/lon grid and map FAWP detections. Runs quickly at coarse resolution.")
    _gm_c1, _gm_c2, _gm_c3 = st.columns(3)
    with _gm_c1:
        _gm_var  = st.selectbox("Variable", list(VARIABLES.keys()),
                                format_func=lambda k: VARIABLES[k], key="gm_var")
        _gm_step = st.slider("Grid step (°)", 5, 20, 10, key="gm_step")
    with _gm_c2:
        _gm_eps  = st.number_input("ε threshold", 0.001, 0.1, 0.01,
                                   format="%.3f", key="gm_eps")
        _gm_yrs  = st.slider("Years of data", 1, 10, 3, key="gm_yrs")
    with _gm_c3:
        _gm_n_null = st.slider("Null perms", 0, 50, 10, key="gm_n_null")

    if st.button("▶ Run global map scan", key="gm_run", type="primary"):
        from fawp_index.weather import fetch_openmeteo, fawp_from_forecast
        import numpy as _np_gm, pandas as _pd_gm
        _lats = list(range(-60, 75, _gm_step))
        _lons = list(range(-180, 180, _gm_step))
        _gm_rows = []
        _gm_prog = st.progress(0.0, "Scanning global grid…")
        _gm_total = len(_lats) * len(_lons)
        _gm_done = 0
        for _glat in _lats:
            for _glon in _lons:
                try:
                    _gdf = fetch_openmeteo(str(_glat), days=365*_gm_yrs,
                                           var=_gm_var, lat=float(_glat),
                                           lon=float(_glon))
                    if _gdf is not None and len(_gdf) > 100:
                        _gr = fawp_from_forecast(_gdf[_gm_var].values,
                                                  epsilon=_gm_eps, n_null=_gm_n_null)
                        _gm_rows.append({"lat": _glat, "lon": _glon,
                                         "fawp": _gr.fawp_found,
                                         "gap":  float(_gr.peak_gap_bits or 0)})
                except Exception:
                    pass
                _gm_done += 1
                _gm_prog.progress(_gm_done/_gm_total,
                                  f"Scanned {_gm_done}/{_gm_total} grid points…")
        _gm_prog.empty()
        st.session_state["gm_results"] = _gm_rows
        st.rerun()

    if "gm_results" in st.session_state and st.session_state["gm_results"]:
        _gmr = st.session_state["gm_results"]
        _gm_fawp = [r for r in _gmr if r["fawp"]]
        st.success(f"🔴 {len(_gm_fawp)}/{len(_gmr)} grid points in FAWP")
        try:
            import folium as _fol
            from streamlit_folium import st_folium as _stf
            _gm_map = _fol.Map(location=[20, 0], zoom_start=2,
                               tiles="CartoDB dark_matter")
            for _gr in _gmr:
                _col = "#C0111A" if _gr["fawp"] else "#3A4E70"
                _fol.CircleMarker(
                    location=[_gr["lat"], _gr["lon"]],
                    radius=max(3, _gr["gap"]*15),
                    color=_col, fill=True, fill_color=_col,
                    fill_opacity=0.7,
                    popup=f"({_gr['lat']}°,{_gr['lon']}°) gap={_gr['gap']:.4f}b"
                ).add_to(_gm_map)
            _stf(_gm_map, height=400, use_container_width=True)
        except ImportError:
            import matplotlib.pyplot as _plt_gm
            import pandas as _pd_gm
            _gdf2 = _pd_gm.DataFrame(_gmr)
            _fig_gm, _ax_gm = _plt_gm.subplots(figsize=(10,4),facecolor="#0D1729")
            _ax_gm.set_facecolor("#07101E")
            _ax_gm.scatter(_gdf2["lon"], _gdf2["lat"],
                          c=["#C0111A" if f else "#3A4E70" for f in _gdf2["fawp"]],
                          s=_gdf2["gap"]*500+10, alpha=0.75)
            _ax_gm.set_xlabel("Longitude",fontsize=8,color="#7A90B8")
            _ax_gm.set_ylabel("Latitude",fontsize=8,color="#7A90B8")
            _ax_gm.set_title("FAWP Weather Global Map",color="#D4AF37",fontsize=9)
            _fig_gm.tight_layout(); st.pyplot(_fig_gm,use_container_width=True)
            _plt_gm.close(_fig_gm)

if "wx_result" in st.session_state:
    r       = st.session_state["wx_result"]
    hazard  = st.session_state.get("wx_hazard")
    epsilon = st.session_state.get("wx_epsilon", 0.01)

    # Result banner
    bc  = "fawp-yes" if r.fawp_found else "fawp-no"
    bi  = "🔴 FAWP DETECTED" if r.fawp_found else "✅ No FAWP Detected"
    bsub = f"{VARIABLES.get(r.variable, r.variable)} · {r.location} · {r.date_range[0]} → {r.date_range[1]}"
    hpill = ""
    if hazard:
        hc = HAZARDS.get(hazard, {}).get("color", "var(--gold)")
        hb = HAZARDS.get(hazard, {}).get("bg", "var(--gold-dim)")
        hpill = (f'<span style="display:inline-flex;align-items:center;gap:.4em;'
                 f'padding:.3em .9em;border-radius:8px;font-size:.8em;font-weight:700;'
                 f'background:{hb};color:{hc};border:1px solid {hc}33;letter-spacing:.02em">'
                 f'{hazard}</span>')
        st.markdown(
            f'<div class="result-banner {bc}">'
            f'<div>'
            f'<div class="result-label">{bi}</div>'
            f'<div class="result-sub">{bsub}</div>'
            f'</div>'
            f'<div style="margin-left:auto">{hpill}</div>'
            f'</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="result-banner {bc}">'
            f'<div class="result-label">{bi}</div>'
            f'<div class="result-sub">{bsub}</div>'
            f'</div>', unsafe_allow_html=True)

    # KPI row
    c1,c2,c3,c4,c5 = st.columns(5)
    _kpi(c1, f"{r.peak_gap_bits:.4f}", "Peak gap (bits)")
    _kpi(c2, str(r.odw_result.tau_h_plus) if r.odw_result.tau_h_plus is not None else "—", "τ⁺ₕ horizon")
    _kpi(c3, str(r.odw_result.tau_f)      if r.odw_result.tau_f      is not None else "—", "τf cliff")
    _kpi(c4, f"τ{r.odw_start}–{r.odw_end}" if r.fawp_found else "—", "ODW",
         "var(--red)" if r.fawp_found else "var(--muted)")
    _kpi(c5, f"{r.n_obs:,}", "Observations")

    # E9.7 timing badge
    # Weather anomaly flag — check if FAWP window coincides with extreme values
    try:
        import numpy as _np_anm
        _pred_series = getattr(r, "_pred_series", None) or (
            r.pred_mi if hasattr(r, "pred_mi") else None
        )
        # Use the raw MI curve peak as proxy for extreme event timing
        if r.fawp_found and r.odw_start and r.odw_end:
            _odw_pred = r.pred_mi[r.odw_start-1:r.odw_end] if hasattr(r, 'pred_mi') else []
            _all_pred = r.pred_mi if hasattr(r, 'pred_mi') else []
            if len(_all_pred) > 0 and len(_odw_pred) > 0:
                _p95 = _np_anm.percentile(_all_pred, 95)
                _p05 = _np_anm.percentile(_all_pred, 5)
                _odw_max = float(_np_anm.max(_odw_pred))
                if _odw_max >= _p95:
                    st.warning(f"⚡ **Extreme forecast signal in ODW** — peak MI ({_odw_max:.4f}b) "
                               f"is in the top 5% of all τ values. "
                               f"The detection window coincides with a strong predictability spike.")
                elif _odw_max <= _p05:
                    st.info(f"ℹ️ ODW shows low MI ({_odw_max:.4f}b) — marginal FAWP signal.")
    except Exception:
        pass

    import fawp_index as _fi
    _lead = _fi.E97_MEAN_LEAD_GAP2_TO_CLIFF_U
    _err  = _fi.E97_MEAN_ABS_ERR_GAP2_VS_ODW_START
    st.markdown(
        f'<div style="font-size:.72em;color:#3A4E70;margin:.3em 0 .8em;padding:.3em .6em;'
        f'background:#0D1729;border-radius:6px;border:1px solid #182540;display:inline-block">'
        f'📐 gap2 peak leads cliff by <b style="color:#D4AF37">+{_lead:.3f} delays</b> · '
        f'ODW localisation error <b style="color:#D4AF37">~{_err:.1f} delays</b> · '
        f'<a href="https://doi.org/10.5281/zenodo.19065421" style="color:#4A7FCC">E9.7</a>'
        f'</div>',
        unsafe_allow_html=True
    )

    # Action windows
    if hazard and r.fawp_found:
        _hc = HAZARDS.get(hazard, {}).get('color', '#F2C440')
        _action_window_panel(hazard, r.odw_start, r.odw_end, hazard_color=_hc)

    # MI chart
    st.markdown('<div class="wx-sec">Prediction vs Steering MI</div>', unsafe_allow_html=True)
    _mi_chart(r, epsilon)
    # PNG export of MI chart
    import io as _io_wx
    import matplotlib.pyplot as _plt_wx
    _wx_fig2, _wx_ax = _plt_wx.subplots(figsize=(9, 3.5), facecolor="#0D1729")
    _wx_ax.set_facecolor("#07101E")
    _wx_ax.plot(r.tau, r.pred_mi,  color="#D4AF37", lw=2,   label="Prediction MI")
    _wx_ax.plot(r.tau, r.steer_mi, color="#4A7FCC", lw=1.5, ls="--", label="Steering MI")
    _wx_ax.axhline(epsilon, color="#3A4E70", ls=":", lw=1)
    if r.fawp_found and r.odw_start:
        _wx_ax.axvspan(r.odw_start, r.odw_end, color="#C0111A", alpha=0.15)
    _wx_ax.legend(fontsize=8, framealpha=0.2)
    _wx_ax.set_xlabel("τ (delay steps)", color="#7A90B8", fontsize=9)
    _wx_ax.set_ylabel("MI (bits)", color="#7A90B8", fontsize=9)
    _wx_ax.set_title(f"FAWP Weather — {r.variable} · {r.location}", color="#D4AF37", fontsize=9)
    _wx_fig2.tight_layout()
    _wx_buf = _io_wx.BytesIO()
    _wx_fig2.savefig(_wx_buf, format="png", dpi=150, bbox_inches="tight")
    _plt_wx.close(_wx_fig2)
    _wx_buf.seek(0)
    st.download_button("⬇ Download MI chart PNG", data=_wx_buf,
                       file_name=f"fawp_weather_{r.variable}_{r.location}.png",
                       mime="image/png", key="wx_png_dl")

    # Explanation
    st.markdown('<div class="wx-sec">Interpretation</div>', unsafe_allow_html=True)
    _explanation_panel(r, hazard)

    # Rolling timeline
    with st.expander("📈 FAWP timeline — how has this changed over time?"):
        tl_c1, tl_c2, tl_c3 = st.columns(3)
        with tl_c1: tl_win = st.slider("Window (years)", 1, 5, 2, key="tl_win")
        with tl_c2: tl_step = st.slider("Step (months)", 3, 12, 6, key="tl_step")
        with tl_c3:
            tl_null = st.slider("Null perms", 0, 50, 10, key="tl_null")
            run_tl  = st.button("▶ Run Timeline", key="run_tl", use_container_width=True)
        if run_tl:
            try:
                from fawp_index.weather import fawp_rolling_timeline
                with st.spinner("Computing timeline…"):
                    tl_df = fawp_rolling_timeline(
                        lat, lon, variable=variable,
                        start_date=start_date, end_date=end_date,
                        window_years=tl_win, step_months=tl_step,
                        horizon_days=horizon_days, tau_max=tau_max,
                        epsilon=epsilon, n_null=tl_null,
                    )
                    st.session_state["wx_timeline"] = tl_df
            except Exception as e:
                st.error(f"Timeline failed: {e}")
        if "wx_timeline" in st.session_state:
            tl = st.session_state["wx_timeline"].dropna(subset=["peak_gap_bits"])
            if len(tl):
                try:
                    import matplotlib; matplotlib.use("Agg")
                    import matplotlib.pyplot as plt
                    import matplotlib.patches as mpatches
                    fig, ax = plt.subplots(figsize=(11, 3))
                    fig.patch.set_facecolor("#030810")
                    ax.set_facecolor("#091220")
                    xs = range(len(tl))
                    labels = [str(r.window_start)[:7] for r in tl.itertuples()]
                    step_l = max(1, len(labels)//8)

                    # Dual-axis: gap bits (left) + raw variable mean (right)
                    ax2 = ax.twinx() if "raw_mean" in tl.columns else None

                    ax.plot(list(xs), tl["peak_gap_bits"].values,
                            color="#F2C440", lw=2, zorder=3, label="Peak gap (bits)")
                    for i, row in enumerate(tl.itertuples()):
                        if row.fawp_found:
                            ax.axvspan(i-.4, i+.4, alpha=.2, color="#E83030", zorder=1)
                    ax.axhline(epsilon, color="#1E3050", ls=":", lw=1)

                    # Overlay raw variable if available
                    if ax2 is not None and "raw_mean" in tl.columns:
                        ax2.plot(list(xs), tl["raw_mean"].values,
                                 color="#4A7FCC", lw=1.2, ls="--", alpha=0.7,
                                 label=f"{variable} (mean)")
                        ax2.set_ylabel(f"{variable}", fontsize=7, color="#4A7FCC")
                        ax2.tick_params(colors="#4A7FCC", labelsize=7)
                        ax2.spines["right"].set_edgecolor("#4A7FCC")
                        # Combined legend
                        lines1, labs1 = ax.get_legend_handles_labels()
                        lines2, labs2 = ax2.get_legend_handles_labels()
                        ax.legend(lines1 + lines2, labs1 + labs2,
                                  fontsize=7, framealpha=0.2, loc="upper left")

                    ax.set_xticks(list(xs)[::step_l])
                    ax.set_xticklabels(labels[::step_l], rotation=30, ha="right",
                                       fontsize=7, color="#5070A0")
                    ax.set_ylabel("Peak gap (bits)", fontsize=8, color="#5070A0")
                    ax.tick_params(colors="#5070A0", labelsize=7)
                    for sp in ax.spines.values(): sp.set_edgecolor("#111E32")
                    ax.grid(axis="y", color="#111E32", alpha=.5, lw=.5)
                    plt.tight_layout(pad=.4)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                    n_f = int(tl["fawp_found"].sum())
                    pct = n_f/len(tl)*100
                    st.markdown(f"**{n_f}/{len(tl)} windows** show FAWP ({pct:.0f}%)")
                    st.download_button("Download CSV", data=tl.to_csv(index=False).encode(),
                                       file_name=f"fawp_timeline_{variable}.csv", mime="text/csv")
                except Exception:
                    st.dataframe(tl, use_container_width=True, hide_index=True)

    # Data quality + downloads
    with st.expander("📊 Data quality"):
        q1,q2,q3,q4 = st.columns(4)
        q1.metric("Sample size", f"{r.n_obs:,}")
        q2.metric("Tau grid", f"{len(r.tau)} points")
        q3.metric("Max pred MI", f"{r.pred_mi.max():.4f} b" if len(r.pred_mi) else "—")
        q4.metric("Null floor β", "0.99")
        st.caption("ERA5 reanalysis from Open-Meteo (CC BY 4.0). Prediction channel: I(value_t ; value_{t+Δ}). Steering: day-over-day change as intervention proxy.")

    import json as _j
    dc1, dc2, dc3 = st.columns(3)
    with dc1:
        st.download_button("Download JSON", data=_j.dumps(r.to_dict(), indent=2).encode(),
                           file_name=f"fawp_weather_{r.variable}.json", mime="application/json")
    with dc2:
        df_dl = pd.DataFrame({"tau": r.tau, "pred_mi": r.pred_mi, "steer_mi": r.steer_mi})
        st.download_button("Download MI CSV", data=df_dl.to_csv(index=False).encode(),
                           file_name=f"fawp_mi_{r.variable}.csv", mime="text/csv")
    with dc3:
        try:
            from fawp_index.report_html import generate_html_report
            rpt = generate_html_report(r, title=f"FAWP Weather — {r.variable} {r.location}")
            st.download_button("📄 HTML Report", data=rpt.encode(),
                               file_name=f"fawp_weather_{r.variable}.html", mime="text/html")
        except Exception:
            pass

    # ── Share + Save ──────────────────────────────────────────────────────
    sc1, sc2 = st.columns(2)
    with sc1:
        if _SHARE_OK:
            _share_button("weather", f"FAWP Weather — {r.variable} {r.location}", r.to_dict())
    with sc2:
        if _WL_OK and st.button("📍 Save to My Locations", key="save_wx_loc"):
            try:
                from supabase_store import _current_user_id
                uid = _current_user_id()
            except Exception:
                uid = None
            if uid:
                _loc_name = f"{r.location} · {r.variable}"
                ok = save_weather_location(
                    uid, _loc_name, lat, lon, variable,
                    hazard=hazard,
                )
                if ok:
                    update_last_result(uid, _loc_name, r.to_dict())
                    st.success(f"Saved: {_loc_name}")
                else:
                    st.warning("Could not save — sign in to use watchlist.")
            else:
                st.info("Sign in to save locations.")

else:
    st.markdown("""
<div class="wx-empty">
  <div class="wx-empty-icon">🌦</div>
  <div class="wx-empty-title">FAWP Weather Scanner</div>
  <div class="wx-empty-sub">
    Select a location and hazard in the sidebar,<br>
    then click <b>▶ Run Scan</b>.<br><br>
    ERA5 reanalysis · 1940–present · any location on Earth<br>
    Free · no API key needed
  </div>
</div>
""", unsafe_allow_html=True)
