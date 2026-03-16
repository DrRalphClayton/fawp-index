"""
FAWP Weather Scanner — Standalone Streamlit app.

Detect the Information-Control Exclusion Principle in ERA5 reanalysis.
Uses Open-Meteo — free ERA5 data, no API key needed.

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

# set_page_config only when run standalone (not imported by app.py)
if __name__ == "__main__" or not st.session_state.get("_app_mode"):
    try:
        st.set_page_config(
            page_title="FAWP Weather Scanner",
            page_icon="🌦",
            layout="wide",
            initial_sidebar_state="expanded",
        )
    except Exception:
        pass  # Already set by parent app.py

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=JetBrains+Mono:wght@400;600&family=DM+Sans:wght@400;500;600&display=swap');
html, body, [data-testid="stAppViewContainer"] { background: #07101E !important; }
[data-testid="stSidebar"] { background: #090F1B !important; border-right: 1px solid #182540; }
.metric-card { background:#0D1729; border:1px solid #182540; border-radius:8px; padding:1em 1.2em; text-align:center; }
.metric-val  { font-family:'JetBrains Mono',monospace; font-size:1.6em; font-weight:600; color:#D4AF37; }
.metric-lbl  { font-size:.75em; color:#7A90B8; text-transform:uppercase; letter-spacing:.08em; margin-top:.2em; }
.fawp-badge  { display:inline-block; padding:.3em 1em; border-radius:100px; font-weight:700; font-size:.9em; }
.fawp-yes    { background:rgba(192,17,26,.12); color:#C0111A; border:1px solid rgba(192,17,26,.3); }
.fawp-no     { background:rgba(29,185,84,.08); color:#1DB954; border:1px solid rgba(29,185,84,.2); }
.hazard-pill { display:inline-block; padding:.25em .8em; border-radius:6px; font-size:.82em; font-weight:600; margin:.2em; }
.aw-bar      { background:#182540; border-radius:4px; height:8px; margin:.3em 0; }
.aw-fill     { background:#D4AF37; border-radius:4px; height:8px; transition:width .3s; }
.exp-box     { background:#0A1523; border:1px solid #182540; border-left:3px solid #D4AF37;
               border-radius:6px; padding:1em 1.2em; margin:.8em 0; font-size:.88em; color:#7A90B8; line-height:1.6; }
</style>
""", unsafe_allow_html=True)

# ── Hazard definitions ────────────────────────────────────────────────────────
HAZARDS = {
    "🌡 Heat":          {"var": "temperature_2m",          "thresh": 35.0,  "unit": "°C",  "desc": "Extreme heat window"},
    "🌧 Heavy Rain":    {"var": "precipitation_sum",        "thresh": 20.0,  "unit": "mm",  "desc": "Flood/disruption window"},
    "💨 Wind":          {"var": "wind_speed_10m",           "thresh": 15.0,  "unit": "m/s", "desc": "Storm/grid disruption"},
    "❄ Winter Ice":    {"var": "temperature_2m",           "thresh": 0.0,   "unit": "°C",  "desc": "Ice/road hazard window"},
    "☀ Fire Weather":  {"var": "shortwave_radiation",      "thresh": 500.0, "unit": "W/m²","desc": "High fire risk window"},
    "🌫 Cloud/Fog":    {"var": "cloud_cover",              "thresh": 85.0,  "unit": "%",   "desc": "Visibility disruption"},
}

# Action windows by hazard (hours available for response)
ACTION_WINDOWS = {
    "🌡 Heat":          {"evacuation": 72, "grid_prep": 48, "staffing": 24, "cooling": 12},
    "🌧 Heavy Rain":    {"evacuation": 48, "drainage_prep": 24, "road_close": 12, "pump_deploy": 6},
    "💨 Wind":          {"grid_prep": 36,  "road_clear": 24,  "staffing": 12,  "shelter": 6},
    "❄ Winter Ice":    {"road_treat": 24,  "gritting": 12,    "travel_ban": 6,  "emergency": 3},
    "☀ Fire Weather":  {"evacuation": 48,  "firebreak": 24,   "resource_stage": 12, "contain": 6},
    "🌫 Cloud/Fog":    {"aviation": 12,    "road_warn": 6,    "marine": 6,      "advisory": 3},
}

VARIABLES = {
    "temperature_2m":          "🌡 Temperature 2m (°C)",
    "precipitation_sum":       "🌧 Precipitation (mm/day)",
    "wind_speed_10m":          "💨 Wind Speed 10m (m/s)",
    "surface_pressure":        "⬇ Surface Pressure (hPa)",
    "cloud_cover":             "☁ Cloud Cover (%)",
    "shortwave_radiation":     "☀ Shortwave Radiation (W/m²)",
}

PRESETS = {
    "London":    (51.50,  -0.10),
    "New York":  (40.71, -74.01),
    "Tokyo":     (35.69, 139.69),
    "Sydney":   (-33.87, 151.21),
    "Paris":     (48.86,   2.35),
    "Dubai":     (25.20,  55.27),
    "Chicago":   (41.88, -87.63),
    "Mumbai":    (19.08,  72.88),
}

def _fmt_loc(lat, lon):
    return f"({abs(lat):.2f}{'N' if lat>=0 else 'S'}, {abs(lon):.2f}{'E' if lon>=0 else 'W'})"

def _kpi(col, val, lbl, color="#D4AF37"):
    col.markdown(
        f'<div class="metric-card"><div class="metric-val" style="color:{color}">{val}</div>'
        f'<div class="metric-lbl">{lbl}</div></div>',
        unsafe_allow_html=True)

def _action_window_panel(hazard, odw_start, odw_end, tau_f):
    """Render action-window bars showing how much response time remains."""
    windows = ACTION_WINDOWS.get(hazard, {})
    if not windows:
        return
    st.markdown("#### ⏱ Action Windows")
    st.caption(
        "If FAWP is detected, these are the typical response windows available "
        "before outcomes become fixed. The ODW represents the usable forecast-lead window."
    )
    odw_days = (odw_end - odw_start + 1) if (odw_start and odw_end) else None
    cols = st.columns(len(windows))
    for i, (action, hours) in enumerate(windows.items()):
        days_avail = hours / 24
        if odw_days:
            pct = min(1.0, odw_days / days_avail)
            color = "#1DB954" if pct > 0.6 else ("#D4AF37" if pct > 0.3 else "#C0111A")
            fill = int(pct * 100)
        else:
            pct, color, fill = 0, "#3A4E70", 0
        cols[i].markdown(
            f'<div class="metric-card">'
            f'<div style="font-size:.75em;color:#7A90B8;text-transform:uppercase;letter-spacing:.06em">{action.replace("_"," ")}</div>'
            f'<div class="metric-val" style="font-size:1.1em;color:{color}">{hours}h</div>'
            f'<div class="aw-bar"><div class="aw-fill" style="width:{fill}%;background:{color}"></div></div>'
            f'<div style="font-size:.7em;color:#3A4E70">{int(fill)}% of window usable</div>'
            f'</div>',
            unsafe_allow_html=True)

def _explanation_panel(r, hazard):
    """Plain-English explanation of results."""
    h_info = HAZARDS.get(hazard, {})
    gap = r.peak_gap_bits
    odw = f"τ={r.odw_start}–{r.odw_end}" if r.fawp_found else "none detected"
    st.markdown("#### 📖 What this means")
    if r.fawp_found:
        st.markdown(f"""
<div class="exp-box">
<b style="color:#EDF0F8">Forecast skill is present — but the window to act is closing.</b><br><br>
The ERA5 signal for <b>{VARIABLES.get(r.variable, r.variable)}</b> at {r.location} shows
that the atmosphere remains predictable at lags beyond τ={r.odw_result.tau_h_plus}.
However, the ability to influence outcomes through intervention has already collapsed
at the same lag.<br><br>
<b style="color:#D4AF37">Leverage gap: {gap:.4f} bits.</b> &nbsp;
<b>Operational Detection Window: {odw}.</b><br><br>
In plain terms: you can see what is likely to happen better than you can still change it.
This is the Information-Control Exclusion Principle — FAWP.
{f'<br><br><b>For {hazard}:</b> {h_info.get("desc", "")} — response capacity is limited.' if hazard else ''}
</div>
""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
<div class="exp-box">
<b style="color:#EDF0F8">No FAWP regime detected at this location and period.</b><br><br>
Predictive and steering coupling collapse together — there is no persistent window
where forecasts remain useful while interventions have already failed.<br><br>
This could mean the system is still controllable, the period is too short, or the
variable does not exhibit the delayed-collapse pattern. Try a longer date range,
a different variable, or a lower epsilon threshold.
</div>
""", unsafe_allow_html=True)

def _multi_scan_panel():
    """Multi-location scan panel."""
    st.markdown("#### 🗺 Multi-Location Scan")
    st.caption("Scan multiple locations and rank by FAWP severity.")

    col_m1, col_m2 = st.columns([3, 1])
    with col_m1:
        cities_input = st.text_area(
            "Locations (one per line: name, lat, lon)",
            "London, 51.5, -0.1\nParis, 48.9, 2.4\nNew York, 40.7, -74.0\nTokyo, 35.7, 139.7\nSydney, -33.9, 151.2",
            height=120,
        )
    with col_m2:
        ms_var    = st.selectbox("Variable", list(VARIABLES.keys()),
                                 format_func=lambda k: VARIABLES[k], key="ms_var")
        ms_start  = st.text_input("Start", "2015-01-01", key="ms_start")
        ms_end    = st.text_input("End",   "2024-12-31", key="ms_end")
        ms_null   = st.slider("Null perms", 0, 100, 20, key="ms_null")
        run_multi = st.button("▶ Scan All", type="primary", use_container_width=True,
                              key="run_multi")

    if run_multi:
        locations = []
        for line in cities_input.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                try:
                    locations.append({"name": parts[0],
                                      "lat": float(parts[1]),
                                      "lon": float(parts[2])})
                except ValueError:
                    st.warning(f"Skipping bad line: {line}")

        if not locations:
            st.error("No valid locations parsed.")
        else:
            try:
                from fawp_index.weather import scan_weather_grid
            except ImportError as e:
                st.error(f"fawp-index not available: {e}")
                return

            with st.spinner(f"Scanning {len(locations)} location(s)…"):
                results = scan_weather_grid(
                    locations    = locations,
                    variable     = ms_var,
                    start_date   = ms_start,
                    end_date     = ms_end,
                    horizon_days = 7,
                    tau_max      = 30,
                    n_null       = ms_null,
                    verbose      = False,
                )
                st.markdown("**Results — ranked by leverage gap:**")
                rows = []
                for r in results:
                    rows.append({
                        "Location":   r.location,
                        "FAWP":       "🔴 YES" if r.fawp_found else "—",
                        "Gap (bits)": f"{r.peak_gap_bits:.4f}",
                        "ODW":        f"τ {r.odw_start}–{r.odw_end}" if r.fawp_found else "—",
                        "τ⁺ₕ":        str(r.odw_result.tau_h_plus) if r.odw_result.tau_h_plus else "—",
                        "τf":         str(r.odw_result.tau_f) if r.odw_result.tau_f else "—",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                # ── Map ──────────────────────────────────────────────────
                try:
                    from fawp_index.weather import plot_weather_map
                    fig_map = plot_weather_map(
                        results,
                        title=f"FAWP Scan — {VARIABLES.get(ms_var, ms_var)}",
                    )
                    st.plotly_chart(fig_map, use_container_width=True)
                except ImportError:
                    st.caption("Install plotly for map: `pip install plotly`")
                except Exception as _map_err:
                    st.caption(f"Map unavailable: {_map_err}")

                import json as _j
                st.download_button(
                    "Download multi-scan JSON",
                    data=_j.dumps([r.to_dict() for r in results], indent=2).encode(),
                    file_name=f"fawp_weather_multiscan_{ms_var}.json",
                    mime="application/json",
                )

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div style='padding:.8em 0 1.2em'>
  <div style='font-family:Syne,sans-serif;font-size:1.4em;font-weight:800;color:#D4AF37'>
    🌦 FAWP Weather
  </div>
  <div style='color:#3A4E70;font-size:.78em'>Information-Control Exclusion Principle</div>
</div>
""", unsafe_allow_html=True)

    mode = st.radio("Mode",
                    ["Single location", "Compare two locations",
                     "Upload NWP data", "Multi-location scan"],
                    label_visibility="collapsed")

    if mode == "Single location":
        st.markdown("**Location**")
        preset = st.selectbox("Quick select", ["Custom"] + list(PRESETS.keys()))

        # Initialise keys before inputs render
        if "wx_lat" not in st.session_state:
            st.session_state["wx_lat"] = 51.5
        if "wx_lon" not in st.session_state:
            st.session_state["wx_lon"] = -0.1

        # Preset overwrites session state so inputs update on next render
        if preset != "Custom":
            st.session_state["wx_lat"], st.session_state["wx_lon"] = PRESETS[preset]

        lat = st.number_input("Latitude",  min_value=-90.0,  max_value=90.0,  step=0.1, format="%.2f", key="wx_lat")
        lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, step=0.1, format="%.2f", key="wx_lon")

        st.markdown("**Hazard / Variable**")
        scan_mode = st.radio("Input mode", ["Hazard", "Variable"],
                             horizontal=True, label_visibility="collapsed")
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
        epsilon      = st.number_input("Epsilon (bits)", value=0.01,
                                       min_value=0.001, max_value=0.1, step=0.001, format="%.3f")
        remove_seas  = st.checkbox("Remove seasonality",
                                   help="Subtract 365-day rolling mean. "
                                        "Recommended for temperature to detect "
                                        "anomaly-level FAWP, not the annual cycle.")
        run_btn = st.button("▶ Run Scan", type="primary", use_container_width=True)

        st.markdown("---")
        st.caption("ERA5 reanalysis · Open-Meteo · free · no API key")

    elif mode == "Compare two locations":
        st.markdown("**Location A**")
        if "cmp_lat_a" not in st.session_state: st.session_state["cmp_lat_a"] = 51.5
        if "cmp_lon_a" not in st.session_state: st.session_state["cmp_lon_a"] = -0.1
        if "cmp_lat_b" not in st.session_state: st.session_state["cmp_lat_b"] = 48.9
        if "cmp_lon_b" not in st.session_state: st.session_state["cmp_lon_b"] = 2.4
        cmp_name_a = st.text_input("Name A", "London",  key="cmp_name_a")
        cmp_lat_a  = st.number_input("Lat A", min_value=-90.0, max_value=90.0,  step=0.1, format="%.2f", key="cmp_lat_a")
        cmp_lon_a  = st.number_input("Lon A", min_value=-180.0, max_value=180.0, step=0.1, format="%.2f", key="cmp_lon_a")
        st.markdown("**Location B**")
        cmp_name_b = st.text_input("Name B", "Paris",  key="cmp_name_b")
        cmp_lat_b  = st.number_input("Lat B", min_value=-90.0, max_value=90.0,  step=0.1, format="%.2f", key="cmp_lat_b")
        cmp_lon_b  = st.number_input("Lon B", min_value=-180.0, max_value=180.0, step=0.1, format="%.2f", key="cmp_lon_b")
        st.markdown("**Settings**")
        cmp_var    = st.selectbox("Variable", list(VARIABLES.keys()),
                                  format_func=lambda k: VARIABLES[k], key="cmp_var")
        cmp_start  = st.text_input("Start", "2010-01-01", key="cmp_start")
        cmp_end    = st.text_input("End",   "2024-12-31", key="cmp_end")
        cmp_horiz  = st.slider("Horizon (days)", 1, 30, 7, key="cmp_horiz")
        cmp_null   = st.slider("Null perms", 0, 100, 30, key="cmp_null")
        cmp_seas   = st.checkbox("Remove seasonality", key="cmp_seas")
        run_cmp    = st.button("▶ Compare", type="primary",
                               use_container_width=True, key="run_cmp")

    elif mode == "Upload NWP data":
        st.markdown("**Forecast CSV**")
        nwp_fc_file  = st.file_uploader("Forecast file (.csv)", type=["csv"], key="nwp_fc")
        nwp_fc_col   = st.text_input("Forecast column", "forecast", key="nwp_fc_col")
        st.markdown("**Observation CSV**")
        nwp_obs_file = st.file_uploader("Observation file (.csv)", type=["csv"], key="nwp_obs")
        nwp_obs_col  = st.text_input("Observed column", "observed", key="nwp_obs_col")
        nwp_date_col = st.text_input("Date column",     "date",     key="nwp_date_col")
        nwp_intv_col = st.text_input("Intervention col (optional, e.g. spread)", "",
                                     key="nwp_intv_col")
        nwp_var      = st.text_input("Variable label", "temperature_2m", key="nwp_var")
        nwp_loc      = st.text_input("Location label", "uploaded",       key="nwp_loc")
        nwp_null     = st.slider("Null perms", 0, 200, 50, key="nwp_null")
        nwp_seas     = st.checkbox("Remove seasonality", key="nwp_seas")
        run_nwp      = st.button("▶ Run FAWP on uploaded data", type="primary",
                                 use_container_width=True, key="run_nwp")


# ── Main header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding:.8em 0 .4em'>
  <span style='font-family:Syne,sans-serif;font-size:1.9em;font-weight:800;color:#D4AF37'>
    FAWP Weather Scanner
  </span>
  <span style='color:#3A4E70;font-size:.82em;margin-left:1em'>
    Real ERA5 reanalysis · <a href='https://open-meteo.com' style='color:#4A7FCC'>Open-Meteo</a> · <a href='https://fawp-scanner.info' style='color:#7A90B8'>fawp-scanner.info</a>
  </span>
</div>
""", unsafe_allow_html=True)

# ── Compare two locations mode ────────────────────────────────────────────────
if mode == "Compare two locations":
    st.markdown('<div style="font-family:Syne,sans-serif;font-size:1.1em;font-weight:700;color:#D4AF37;margin-bottom:.8em">Compare two locations</div>', unsafe_allow_html=True)
    if run_cmp:
        try:
            from fawp_index.weather import compare_locations
        except ImportError as e:
            st.error(f"fawp-index not available: {e}")
        else:
            with st.spinner(f"Scanning {cmp_name_a} and {cmp_name_b}…"):
                try:
                    r_a, r_b = compare_locations(
                        {"lat": cmp_lat_a, "lon": cmp_lon_a, "name": cmp_name_a},
                        {"lat": cmp_lat_b, "lon": cmp_lon_b, "name": cmp_name_b},
                        variable=cmp_var, start_date=cmp_start, end_date=cmp_end,
                        horizon_days=cmp_horiz, n_null=cmp_null,
                        remove_seasonality=cmp_seas,
                    )
                    st.session_state["cmp_r_a"] = r_a
                    st.session_state["cmp_r_b"] = r_b
                except Exception as e:
                    st.error(f"Compare failed: {e}")
    if "cmp_r_a" in st.session_state and "cmp_r_b" in st.session_state:
        r_a = st.session_state["cmp_r_a"]
        r_b = st.session_state["cmp_r_b"]
        col_a, col_b = st.columns(2)
        for col, r in [(col_a, r_a), (col_b, r_b)]:
            with col:
                badge = "🔴 FAWP" if r.fawp_found else "✅ Clear"
                st.markdown(f"#### {r.location}  {badge}")
                _kpi_cols = st.columns(3)
                _kpi(_kpi_cols[0], f"{r.peak_gap_bits:.4f}", "Gap (bits)")
                _kpi(_kpi_cols[1], str(r.odw_result.tau_h_plus) if r.odw_result.tau_h_plus else "—", "τ⁺ₕ")
                _kpi(_kpi_cols[2], f"τ {r.odw_start}–{r.odw_end}" if r.fawp_found else "—", "ODW")
                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(5, 2.5))
                    fig.patch.set_facecolor("#07101E")
                    ax.set_facecolor("#0D1729")
                    ax.plot(r.tau, r.pred_mi,  color="#D4AF37", lw=1.8, label="Pred MI")
                    ax.plot(r.tau, r.steer_mi, color="#4A7FCC", lw=1.4, ls="--", label="Steer MI")
                    if r.fawp_found and r.odw_start:
                        ax.axvspan(r.odw_start, r.odw_end, alpha=0.2, color="#C0111A")
                    ax.tick_params(colors="#7A90B8", labelsize=7)
                    ax.set_xlabel("τ", fontsize=7, color="#7A90B8")
                    for sp in ax.spines.values(): sp.set_edgecolor("#182540")
                    ax.legend(fontsize=6, facecolor="#0D1729", labelcolor="#EDF0F8",
                              edgecolor="#182540")
                    plt.tight_layout(pad=0.3)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                except Exception:
                    pass
                st.caption(r.explain() if hasattr(r, "explain") else "")
    elif not run_cmp:
        st.info("Fill in both locations in the sidebar and click ▶ Compare.")
    st.stop()

# ── NWP upload mode ───────────────────────────────────────────────────────────
if mode == "Upload NWP data":
    st.markdown('<div style="font-family:Syne,sans-serif;font-size:1.1em;font-weight:700;color:#D4AF37;margin-bottom:.8em">NWP Forecast vs Observation</div>', unsafe_allow_html=True)
    st.caption("Upload your own forecast and observation CSVs. FAWP is computed on real model skill — no ERA5 proxy.")
    if run_nwp:
        if not nwp_fc_file or not nwp_obs_file:
            st.error("Upload both forecast and observation CSV files.")
        else:
            import tempfile, os
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f_fc:
                f_fc.write(nwp_fc_file.read()); fc_tmp = f_fc.name
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f_obs:
                f_obs.write(nwp_obs_file.read()); obs_tmp = f_obs.name
            try:
                from fawp_index.weather import fawp_from_nwp_csvs
                with st.spinner("Running FAWP on uploaded NWP data…"):
                    nwp_result = fawp_from_nwp_csvs(
                        forecast_path     = fc_tmp,
                        observed_path     = obs_tmp,
                        forecast_col      = nwp_fc_col,
                        observed_col      = nwp_obs_col,
                        date_col          = nwp_date_col,
                        intervention_col  = nwp_intv_col if nwp_intv_col.strip() else None,
                        variable          = nwp_var,
                        location          = nwp_loc,
                        n_null            = nwp_null,
                        remove_seasonality= nwp_seas,
                    )
                st.session_state["nwp_result"] = nwp_result
            except Exception as e:
                st.error(f"Failed: {e}")
            finally:
                os.unlink(fc_tmp); os.unlink(obs_tmp)
    if "nwp_result" in st.session_state:
        nr = st.session_state["nwp_result"]
        badge = "🔴 FAWP DETECTED" if nr.fawp_found else "✅ No FAWP"
        st.markdown(f"### {badge}")
        kc = st.columns(4)
        _kpi(kc[0], f"{nr.peak_gap_bits:.4f}", "Peak gap (bits)")
        _kpi(kc[1], str(nr.odw_result.tau_h_plus) if nr.odw_result.tau_h_plus else "—", "τ⁺ₕ")
        _kpi(kc[2], f"τ {nr.odw_start}–{nr.odw_end}" if nr.fawp_found else "—", "ODW")
        _kpi(kc[3], f"{nr.n_obs:,}", "Observations")
        if nr.fawp_found:
            st.success(nr.explain())
        else:
            st.info(nr.explain())
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 3))
            fig.patch.set_facecolor("#07101E")
            ax.set_facecolor("#0D1729")
            ax.plot(nr.tau, nr.pred_mi,  color="#D4AF37", lw=2, label="Prediction MI (forecast skill)")
            ax.plot(nr.tau, nr.steer_mi, color="#4A7FCC", lw=1.5, ls="--", label="Steering MI (intervention)")
            if nr.fawp_found and nr.odw_start:
                ax.axvspan(nr.odw_start, nr.odw_end, alpha=0.18, color="#C0111A", label="ODW")
            ax.set_xlabel("τ (delay)", fontsize=8, color="#7A90B8")
            ax.set_ylabel("MI (bits)", fontsize=8, color="#7A90B8")
            ax.tick_params(colors="#7A90B8", labelsize=7)
            ax.legend(fontsize=7, facecolor="#0D1729", labelcolor="#EDF0F8", edgecolor="#182540")
            for sp in ax.spines.values(): sp.set_edgecolor("#182540")
            plt.tight_layout(pad=0.4)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except Exception:
            pass
        import json as _j
        st.download_button("Download result JSON",
            data=_j.dumps(nr.to_dict(), indent=2).encode(),
            file_name=f"fawp_nwp_{nr.variable}.json", mime="application/json")
    elif not run_nwp:
        st.info("Upload a forecast CSV and observation CSV in the sidebar, then click ▶ Run.")
        with st.expander("📋 Expected CSV format"):
            st.markdown("""**Forecast CSV** — one row per date:
```
date,forecast,ensemble_spread
2020-01-01,12.3,0.8
2020-01-02,13.1,0.9
```
**Observation CSV** — same dates:
```
date,observed
2020-01-01,12.8
2020-01-02,13.5
```
Both files must share the same date column name.""")
    st.stop()

# ── Multi-location mode ───────────────────────────────────────────────────────
if mode == "Multi-location scan":
    _multi_scan_panel()
    st.stop()

# ── Single location mode ──────────────────────────────────────────────────────
if run_btn:
    try:
        import openmeteo_requests  # noqa
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
            result = fawp_from_open_meteo(
                latitude=lat, longitude=lon, variable=variable,
                start_date=start_date, end_date=end_date,
                horizon_days=horizon_days, tau_max=tau_max,
                epsilon=epsilon, n_null=n_null,
                remove_seasonality=remove_seas,
            )
            st.session_state["wx_result"]   = result
            st.session_state["wx_hazard"]   = hazard
            st.session_state["wx_epsilon"]  = epsilon
            st.session_state["wx_seasonal"] = remove_seas
        except Exception as e:
            st.error(f"Scan failed: {e}")
            st.stop()

if "wx_result" in st.session_state:
    r       = st.session_state["wx_result"]
    hazard  = st.session_state.get("wx_hazard")
    epsilon = st.session_state.get("wx_epsilon", 0.01)

    # ── Hazard pill + FAWP badge ──────────────────────────────────────────
    hazard_html = ""
    if hazard:
        hinfo = HAZARDS[hazard]
        hazard_html = (
            f'<span class="hazard-pill" style="background:rgba(212,175,55,.1);'
            f'color:#D4AF37;border:1px solid rgba(212,175,55,.3)">{hazard}</span> '
        )
    badge_cls = "fawp-yes" if r.fawp_found else "fawp-no"
    badge_txt = "🔴 FAWP DETECTED" if r.fawp_found else "✅ No FAWP"
    st.markdown(
        f'<div style="margin:.4em 0 1em">{hazard_html}'
        f'<span class="fawp-badge {badge_cls}">{badge_txt}</span>'
        f'<span style="color:#3A4E70;font-size:.8em;margin-left:1em">'
        f'{VARIABLES.get(r.variable, r.variable)} · {r.location} · '
        f'{r.date_range[0]} → {r.date_range[1]}</span></div>',
        unsafe_allow_html=True)

    # ── KPI row ───────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    _kpi(c1, f"{r.peak_gap_bits:.4f}", "Peak gap (bits)")
    _kpi(c2, str(r.odw_result.tau_h_plus) if r.odw_result.tau_h_plus else "—", "τ⁺ₕ horizon")
    _kpi(c3, str(r.odw_result.tau_f)      if r.odw_result.tau_f      else "—", "τf cliff")
    _kpi(c4, f"τ {r.odw_start}–{r.odw_end}" if r.fawp_found else "—", "ODW",
         "#C0111A" if r.fawp_found else "#7A90B8")
    _kpi(c5, f"{r.n_obs:,}", "Observations")

    # ── Action windows (hazard mode only) ────────────────────────────────
    if hazard and r.fawp_found:
        _action_window_panel(hazard, r.odw_start, r.odw_end, r.odw_result.tau_f)

    # ── MI curves ─────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if len(r.tau) > 0:
            st.markdown("#### Prediction vs Steering MI curves")
            fig, ax = plt.subplots(figsize=(11, 3.4))
            fig.patch.set_facecolor("#07101E")
            ax.set_facecolor("#0D1729")
            ax.plot(r.tau, r.pred_mi,  color="#D4AF37", lw=2.0, label="Prediction MI")
            ax.plot(r.tau, r.steer_mi, color="#4A7FCC", lw=1.6, ls="--", label="Steering MI")
            ax.axhline(epsilon, color="#3A4E70", ls=":", lw=1.2, label=f"ε={epsilon}")
            if r.fawp_found and r.odw_start is not None:
                ax.axvspan(r.odw_start, r.odw_end, alpha=0.18, color="#C0111A",
                           label=f"ODW τ={r.odw_start}–{r.odw_end}")
                if r.odw_result.tau_f:
                    ax.axvline(r.odw_result.tau_f, color="#C0111A", lw=1, ls=":",
                               label=f"cliff τ={r.odw_result.tau_f}")
            ax.set_xlabel("τ (delay, days)", fontsize=9, color="#7A90B8")
            ax.set_ylabel("MI (bits)", fontsize=9, color="#7A90B8")
            ax.tick_params(colors="#7A90B8", labelsize=8)
            ax.legend(fontsize=8, facecolor="#0D1729", labelcolor="#EDF0F8",
                      edgecolor="#182540")
            for sp in ax.spines.values(): sp.set_edgecolor("#182540")
            ax.grid(axis="y", color="#182540", alpha=0.5, lw=0.5)
            plt.tight_layout(pad=0.4)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
    except Exception:
        pass

    # ── Explanation panel ─────────────────────────────────────────────────
    _explanation_panel(r, hazard)


    # ── Rolling window timeline ───────────────────────────────────────────────
    with st.expander("📈 FAWP timeline — how has this changed over time?", expanded=False):
        st.caption(
            "Slides a rolling window across the full date range and computes FAWP "
            "for each window. Shows whether the leverage gap is growing, shrinking, "
            "or stable at this location."
        )
        tl_col1, tl_col2, tl_col3 = st.columns(3)
        with tl_col1:
            tl_window = st.slider("Window size (years)", 1, 5, 2, key="tl_window")
        with tl_col2:
            tl_step = st.slider("Step (months)", 3, 12, 6, key="tl_step")
        with tl_col3:
            tl_null = st.slider("Null perms (timeline)", 0, 50, 10, key="tl_null")
            run_timeline = st.button("▶ Run Timeline", key="run_timeline",
                                     use_container_width=True)

        if run_timeline:
            try:
                from fawp_index.weather import fawp_rolling_timeline
            except ImportError as e:
                st.error(f"fawp-index not available: {e}")
            else:
                with st.spinner(f"Computing timeline ({tl_window}yr windows, {tl_step}mo steps)…"):
                    try:
                        tl_df = fawp_rolling_timeline(
                            latitude    = lat,
                            longitude   = lon,
                            variable    = variable,
                            start_date  = start_date,
                            end_date    = end_date,
                            window_years  = tl_window,
                            step_months   = tl_step,
                            horizon_days  = horizon_days,
                            tau_max       = tau_max,
                            epsilon       = epsilon,
                            n_null        = tl_null,
                            remove_seasonality = remove_seas,
                        )
                        st.session_state["wx_timeline"] = tl_df
                    except Exception as e:
                        st.error(f"Timeline failed: {e}")

        if "wx_timeline" in st.session_state:
            tl = st.session_state["wx_timeline"]
            tl_valid = tl.dropna(subset=["peak_gap_bits"])
            if len(tl_valid) > 0:
                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt
                    import matplotlib.patches as mpatches
                    import numpy as np

                    fig, ax = plt.subplots(figsize=(11, 3.2))
                    fig.patch.set_facecolor("#07101E")
                    ax.set_facecolor("#0D1729")

                    # Plot gap line
                    xs = range(len(tl_valid))
                    gaps = tl_valid["peak_gap_bits"].values
                    ax.plot(list(xs), gaps, color="#D4AF37", lw=2, label="Peak gap (bits)")

                    # Shade FAWP windows red
                    for i, row in enumerate(tl_valid.itertuples()):
                        if row.fawp_found:
                            ax.axvspan(i - 0.4, i + 0.4, alpha=0.25,
                                       color="#C0111A", zorder=0)

                    ax.axhline(epsilon, color="#3A4E70", ls=":", lw=1,
                               label=f"ε = {epsilon}")

                    # X labels: window start dates (every other)
                    labels = [str(r.window_start)[:7] for r in tl_valid.itertuples()]
                    step_label = max(1, len(labels) // 8)
                    ax.set_xticks(list(xs)[::step_label])
                    ax.set_xticklabels(labels[::step_label], rotation=30,
                                       ha="right", fontsize=7, color="#7A90B8")
                    ax.set_ylabel("Peak gap (bits)", fontsize=8, color="#7A90B8")
                    ax.tick_params(colors="#7A90B8", labelsize=7)
                    for sp in ax.spines.values(): sp.set_edgecolor("#182540")
                    ax.grid(axis="y", color="#182540", alpha=0.4, lw=0.5)

                    fawp_patch = mpatches.Patch(color="#C0111A", alpha=0.4,
                                                label="FAWP detected")
                    ax.legend(handles=[ax.lines[0], ax.lines[1], fawp_patch],
                              fontsize=7, facecolor="#0D1729",
                              labelcolor="#EDF0F8", edgecolor="#182540")
                    plt.tight_layout(pad=0.4)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                    # Summary table
                    n_fawp = int(tl_valid["fawp_found"].sum())
                    pct    = n_fawp / len(tl_valid) * 100
                    max_gap = float(tl_valid["peak_gap_bits"].max())
                    max_win = tl_valid.loc[tl_valid["peak_gap_bits"].idxmax(), "window_start"]
                    st.markdown(
                        f"**{n_fawp}/{len(tl_valid)} windows** show FAWP ({pct:.0f}%) · "
                        f"Peak gap **{max_gap:.4f} bits** at {str(max_win)[:7]}"
                    )
                    st.download_button(
                        "Download timeline CSV",
                        data=tl_valid.to_csv(index=False).encode(),
                        file_name=f"fawp_timeline_{variable}_{lat}_{lon}.csv",
                        mime="text/csv",
                    )
                except Exception as e:
                    st.error(f"Chart failed: {e}")
                    st.dataframe(tl_valid, use_container_width=True, hide_index=True)

    # ── Confidence / data quality ─────────────────────────────────────────
    with st.expander("📊 Data quality & confidence"):
        q1, q2, q3, q4 = st.columns(4)
        seasonal_lbl = "Yes" if st.session_state.get("wx_seasonal") else "No"
        q1.metric("Sample size", f"{r.n_obs:,}")
        q2.metric("Tau grid", f"{len(r.tau)} points")
        q3.metric("Max pred MI", f"{r.pred_mi.max():.4f} b" if len(r.pred_mi) else "—")
        q4.metric("Null floor β", "0.99")
        q5, q6 = st.columns(2)
        q5.metric("Seasonality removed", seasonal_lbl)
        st.caption(
            "ERA5 reanalysis from Open-Meteo (free, CC BY 4.0). "
            "Prediction channel: I(value_t ; value_{t+Δ}). "
            "Steering channel: day-over-day change as intervention proxy. "
            "Null correction β=0.99 (conservative). "
            "Results reflect structural MI patterns in the ERA5 signal."
        )

    # ── Download ──────────────────────────────────────────────────────────
    import json as _j
    c_d1, c_d2, c_d3 = st.columns(3)
    with c_d1:
        st.download_button("Download JSON", data=_j.dumps(r.to_dict(), indent=2).encode(),
                           file_name=f"fawp_weather_{r.variable}.json", mime="application/json")
    with c_d2:
        df_dl = pd.DataFrame({"tau": r.tau, "pred_mi": r.pred_mi, "steer_mi": r.steer_mi})
        st.download_button("Download MI CSV", data=df_dl.to_csv(index=False).encode(),
                           file_name=f"fawp_weather_{r.variable}_mi.csv", mime="text/csv")
    with c_d3:
        try:
            from fawp_index.report_html import generate_html_report
            _wrpt = generate_html_report(r, title=f"FAWP Weather — {r.variable} {r.location}")
            st.download_button("📄 HTML Report", data=_wrpt.encode(),
                               file_name=f"fawp_weather_{r.variable}.html", mime="text/html")
        except Exception:
            pass

else:
    st.markdown("""
<div style="text-align:center;padding:4em 2em;max-width:600px;margin:0 auto">
  <div style="font-size:3em;margin-bottom:.5em">🌦</div>
  <div style="font-family:Syne,sans-serif;font-size:1.3em;font-weight:700;color:#EDF0F8;margin-bottom:.6em">
    Weather FAWP Scanner
  </div>
  <div style="color:#7A90B8;font-size:.9em;line-height:1.7">
    Select a location and hazard in the sidebar,<br>
    then click <b>▶ Run Scan</b>.<br><br>
    ERA5 reanalysis · 1940–present · any location on Earth<br>
    Free · no API key needed<br><br>
    <span style="color:#3A4E70;font-size:.85em">ERA5 reanalysis · any location · 1940–present</span>
  </div>
</div>
""", unsafe_allow_html=True)
