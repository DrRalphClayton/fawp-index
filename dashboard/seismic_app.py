"""
FAWP Seismic Scanner — Standalone Streamlit module.
Ralph Clayton (2026) · doi:10.5281/zenodo.18673949

Fetches earthquake data from the USGS Earthquake Catalog API (free, no API key)
and runs FAWP detection to identify when seismic activity becomes forecastable
but the ability to intervene / evacuate / act has already collapsed.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from fawp_index.weather import (
    _compute_weather_mi_curves,
    WeatherFAWPResult,
)

# ── Styling helpers (match app.py palette) ─────────────────────────────────────
_CSS = """
<style>
.seis-header {
    font-family:'Syne',sans-serif; font-size:1.9em; font-weight:800;
    color:#D4AF37; margin-bottom:.1em;
}
.seis-sub {
    color:#3A4E70; font-size:.82em; margin-bottom:1.8em;
}
.seis-sec {
    font-family:'Syne',sans-serif; font-size:.85em; font-weight:700;
    letter-spacing:.1em; text-transform:uppercase;
    color:#3A4E70; margin:1.8em 0 .6em;
    border-bottom:1px solid #182540; padding-bottom:.3em;
}
.kpi-box {
    background:#0D1729; border:1px solid #182540;
    border-radius:10px; padding:1em 1.2em; text-align:center;
}
.kpi-val { font-size:1.6em; font-weight:700; color:#D4AF37; }
.kpi-lbl { font-size:.72em; color:#3A4E70; letter-spacing:.05em;
           text-transform:uppercase; margin-top:.2em; }
.result-banner {
    padding:1em 1.4em; border-radius:10px;
    display:flex; align-items:center; gap:1em;
    margin:1em 0; font-family:'Syne',sans-serif; font-weight:700;
}
.fawp-yes { background:#1A0610; border:1.5px solid #C0111A; color:#E03040; }
.fawp-no  { background:#071810; border:1.5px solid #1DB954; color:#1DB954; }
.result-label { font-size:1.15em; }
.result-sub   { font-size:.78em; font-weight:400; color:#7A90B8; }
</style>
"""

# ── USGS region presets ────────────────────────────────────────────────────────
REGIONS = {
    "California (San Andreas)":    {"minlat":  32.5, "maxlat":  42.0, "minlon": -124.5, "maxlon": -114.0},
    "Japan":                       {"minlat":  30.0, "maxlat":  46.0, "minlon":  128.0, "maxlon":  146.0},
    "Turkey (Anatolian Fault)":    {"minlat":  36.0, "maxlat":  42.5, "minlon":   25.0, "maxlon":   45.0},
    "Chile":                       {"minlat": -56.0, "maxlat": -17.0, "minlon":  -76.0, "maxlon":  -65.0},
    "New Zealand":                 {"minlat": -48.0, "maxlat": -34.0, "minlon":  165.0, "maxlon":  179.0},
    "Indonesia / Sumatra":         {"minlat": -11.0, "maxlat":   6.0, "minlon":   94.0, "maxlon":  141.0},
    "Alaska":                      {"minlat":  51.0, "maxlat":  72.0, "minlon": -180.0, "maxlon": -130.0},
    "Greece / Aegean":             {"minlat":  35.0, "maxlat":  42.0, "minlon":   19.0, "maxlon":   29.0},
    "Italy (Apennines)":           {"minlat":  36.0, "maxlat":  47.0, "minlon":    6.0, "maxlon":   19.0},
    "Mexico (Pacific coast)":      {"minlat":  14.0, "maxlat":  23.0, "minlon": -105.0, "maxlon":  -87.0},
    "Custom bounding box":         {},
}

VARIABLES = {
    "daily_count":       "Event count (M≥2 per day)",
    "max_magnitude":     "Max magnitude per day",
    "mean_magnitude":    "Mean magnitude per day",
    "seismic_energy":    "Seismic energy release (J/day)",
    "depth_mean":        "Mean focal depth (km/day)",
}


# ── USGS fetch ─────────────────────────────────────────────────────────────────
def _fetch_usgs(minlat, maxlat, minlon, maxlon, start_date, end_date, min_magnitude=2.0):
    """
    Fetch earthquake events from USGS FDSN web service.
    Returns a DataFrame with columns: time, magnitude, depth, latitude, longitude.
    """
    import urllib.request, json, time

    base = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = (
        f"?format=geojson"
        f"&starttime={start_date}"
        f"&endtime={end_date}"
        f"&minlatitude={minlat}"
        f"&maxlatitude={maxlat}"
        f"&minlongitude={minlon}"
        f"&maxlongitude={maxlon}"
        f"&minmagnitude={min_magnitude}"
        f"&orderby=time-asc"
        f"&limit=20000"
    )
    url = base + params

    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        raise RuntimeError(f"USGS fetch failed: {e}") from e

    features = data.get("features", [])
    if not features:
        raise ValueError(
            f"No earthquakes M≥{min_magnitude} found in this region / date range. "
            "Try widening the region, lowering the minimum magnitude, or extending the date range."
        )

    rows = []
    for f in features:
        props = f["properties"]
        coords = f["geometry"]["coordinates"]
        t = pd.to_datetime(props["time"], unit="ms", utc=True).tz_localize(None)
        mag = props.get("mag")
        dep = coords[2] if len(coords) > 2 else None
        if mag is not None:
            rows.append({"time": t, "magnitude": float(mag),
                         "depth": float(dep) if dep is not None else np.nan,
                         "lon": coords[0], "lat": coords[1]})

    df = pd.DataFrame(rows)
    df["date"] = df["time"].dt.normalize()
    return df


def _build_daily_series(raw_df, variable):
    """Convert raw event DataFrame to a daily time series for the chosen variable."""
    grp = raw_df.groupby("date")

    if variable == "daily_count":
        series = grp["magnitude"].count().rename("value")
    elif variable == "max_magnitude":
        series = grp["magnitude"].max().rename("value")
    elif variable == "mean_magnitude":
        series = grp["magnitude"].mean().rename("value")
    elif variable == "seismic_energy":
        # E = 10^(1.5*M + 4.8)  (Gutenberg–Richter energy relation, Joules)
        def energy(mags):
            return np.sum(10 ** (1.5 * mags + 4.8))
        series = grp["magnitude"].apply(energy).rename("value")
    elif variable == "depth_mean":
        series = grp["depth"].mean().rename("value")
    else:
        series = grp["magnitude"].count().rename("value")

    # Reindex to fill missing days with 0 (no events = 0 count / 0 energy)
    date_range = pd.date_range(raw_df["date"].min(), raw_df["date"].max(), freq="D")
    fill_val = 0.0 if variable in ("daily_count", "seismic_energy") else np.nan
    series = series.reindex(date_range, fill_value=fill_val)
    series = series.interpolate(method="linear", limit=7)
    return series


def _run_seismic_fawp(daily_series, horizon_days, tau_max, n_null, epsilon):
    """Run FAWP detection on a daily seismic time series."""
    values = daily_series.values.astype(float)
    n = len(values) - horizon_days
    if n < 60:
        raise ValueError(
            f"Not enough data: need at least {60 + horizon_days} days, got {len(values)}."
        )

    pred_series   = values[:n]
    future_series = values[horizon_days:horizon_days + n]
    steer_proxy   = np.diff(values)[:n]   # day-over-day change = intervention proxy

    odw, tau, pred_mi, steer_mi = _compute_weather_mi_curves(
        pred_series   = pred_series,
        future_series = future_series,
        steer_series  = steer_proxy,
        tau_max       = tau_max,
        epsilon       = epsilon,
        n_null        = n_null,
    )

    result = WeatherFAWPResult(
        variable     = "seismic",
        location     = "selected region",
        odw_result   = odw,
        tau          = tau,
        pred_mi      = pred_mi,
        steer_mi     = steer_mi,
        skill_metric = "MI",
        n_obs        = n,
        horizon_days = horizon_days,
        date_range   = (
            daily_series.index[0].strftime("%Y-%m-%d"),
            daily_series.index[-1].strftime("%Y-%m-%d"),
        ),
        metadata     = {},
    )
    return result


# ── Plot helpers ───────────────────────────────────────────────────────────────
_DARK_BG   = "#07101E"
_CARD_BG   = "#0D1729"
_GOLD      = "#D4AF37"
_CRIMSON   = "#C0111A"
_BLUE      = "#4A7FCC"
_MUTED     = "#3A4E70"


def _mi_chart(r, epsilon):
    fig, ax = plt.subplots(figsize=(9, 3.5), facecolor=_CARD_BG)
    ax.set_facecolor(_DARK_BG)
    ax.plot(r.tau, r.pred_mi,  color=_GOLD,    lw=2,   label="Prediction MI (seismic pattern)")
    ax.plot(r.tau, r.steer_mi, color=_BLUE,    lw=1.5, ls="--", label="Steering MI (intervention proxy)")
    ax.axhline(epsilon, color=_MUTED, ls=":", lw=1.2, label=f"ε={epsilon}")
    if r.fawp_found and r.odw_start and r.odw_end:
        ax.axvspan(r.odw_start, r.odw_end,
                   color=_CRIMSON, alpha=0.15, label=f"ODW τ={r.odw_start}–{r.odw_end}")
    for spine in ax.spines.values():
        spine.set_edgecolor(_MUTED)
    ax.tick_params(colors=_MUTED)
    ax.set_xlabel("τ (delay steps / days)", color=_MUTED, fontsize=9)
    ax.set_ylabel("MI (bits)", color=_MUTED, fontsize=9)
    ax.set_title("FAWP Seismic MI Curves", color=_GOLD, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.2, labelcolor=_MUTED)
    fig.tight_layout()
    return fig


def _event_chart(raw_df, daily_series, variable_label):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 4.5),
                                   facecolor=_CARD_BG, sharex=False)
    for ax in (ax1, ax2):
        ax.set_facecolor(_DARK_BG)
        for spine in ax.spines.values():
            spine.set_edgecolor(_MUTED)
        ax.tick_params(colors=_MUTED)

    # Top: scatter of events by magnitude
    sc = ax1.scatter(raw_df["time"], raw_df["magnitude"],
                     c=raw_df["magnitude"], cmap="YlOrRd",
                     s=raw_df["magnitude"].apply(lambda m: max(2, (m - 1.5) ** 2.5)),
                     alpha=0.6, linewidths=0)
    ax1.set_ylabel("Magnitude", color=_MUTED, fontsize=8)
    ax1.set_title("Earthquake events", color=_GOLD, fontsize=9, fontweight="bold")

    # Bottom: daily series
    ax2.fill_between(daily_series.index, daily_series.values, alpha=0.4, color=_GOLD)
    ax2.plot(daily_series.index, daily_series.values, color=_GOLD, lw=1.2)
    ax2.set_ylabel(variable_label, color=_MUTED, fontsize=8)
    ax2.set_xlabel("Date", color=_MUTED, fontsize=8)

    fig.tight_layout()
    return fig


def _map_chart(raw_df):
    """Simple scatter map of events."""
    fig, ax = plt.subplots(figsize=(9, 4), facecolor=_CARD_BG)
    ax.set_facecolor(_DARK_BG)
    sc = ax.scatter(
        raw_df["lon"], raw_df["lat"],
        c=raw_df["magnitude"], cmap="YlOrRd",
        s=raw_df["magnitude"].apply(lambda m: max(3, (m - 1) ** 2.8)),
        alpha=0.55, linewidths=0,
    )
    plt.colorbar(sc, ax=ax, label="Magnitude", shrink=0.7)
    ax.set_xlabel("Longitude", color=_MUTED, fontsize=8)
    ax.set_ylabel("Latitude",  color=_MUTED, fontsize=8)
    ax.set_title("Event locations", color=_GOLD, fontsize=9, fontweight="bold")
    for spine in ax.spines.values(): spine.set_edgecolor(_MUTED)
    ax.tick_params(colors=_MUTED)
    fig.tight_layout()
    return fig


def _kpi(col, val, label, color=None):
    color = color or _GOLD
    col.markdown(
        f'<div class="kpi-box">'
        f'<div class="kpi-val" style="color:{color}">{val}</div>'
        f'<div class="kpi-lbl">{label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Main render ────────────────────────────────────────────────────────────────
st.markdown(_CSS, unsafe_allow_html=True)

# ── Nav bar ──────────────────────────────────────────────────────────────────
def _seis_nav_switch(mode):
    for k in ["wl_result","input_dfs","wx_result","wx_hazard","seis_result","seis_raw","seis_daily"]:
        st.session_state.pop(k, None)
    if mode is None:
        st.session_state.pop("_app_mode", None)
    else:
        st.session_state["_app_mode"] = mode
    st.rerun()

_sn1, _sn2, _sn3, _sn4 = st.columns([2, 2, 2, 2])
with _sn1:
    if st.button("⚡ FAWP", key="sn_h", use_container_width=True): _seis_nav_switch(None)
with _sn2:
    if st.button("📈 Finance", key="sn_f", use_container_width=True): _seis_nav_switch("finance")
with _sn3:
    if st.button("🌦 Weather", key="sn_w", use_container_width=True): _seis_nav_switch("weather")
with _sn4:
    st.button("🌍 Seismic", key="sn_s", use_container_width=True, disabled=True, type="primary")
st.markdown("<hr style='border-color:#182540;margin:.2em 0 .8em'>", unsafe_allow_html=True)

st.markdown(
    '<div class="seis-header">🌍 FAWP Seismic Scanner</div>'
    '<div class="seis-sub">USGS Earthquake Catalog · Free · No API key needed · '
    '<a href="https://earthquake.usgs.gov/fdsnws/event/1/" '
    'style="color:#4A7FCC">USGS FDSN API</a></div>',
    unsafe_allow_html=True,
)

# ── Sidebar controls ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌍 Region")
    region_name = st.selectbox("Region preset", list(REGIONS.keys()))

    region = REGIONS[region_name]
    if region_name == "Custom bounding box":
        col_a, col_b = st.columns(2)
        with col_a:
            minlat = st.number_input("Min lat", value=30.0, min_value=-90.0, max_value=90.0, step=0.5)
            minlon = st.number_input("Min lon", value=-125.0, min_value=-180.0, max_value=180.0, step=0.5)
        with col_b:
            maxlat = st.number_input("Max lat", value=42.0, min_value=-90.0, max_value=90.0, step=0.5)
            maxlon = st.number_input("Max lon", value=-114.0, min_value=-180.0, max_value=180.0, step=0.5)
    else:
        minlat = region["minlat"]; maxlat = region["maxlat"]
        minlon = region["minlon"]; maxlon = region["maxlon"]
        st.caption(f"Lat {minlat}–{maxlat} · Lon {minlon}–{maxlon}")

    st.markdown("### 📅 Date range")
    col_s, col_e = st.columns(2)
    with col_s:
        start_date = st.text_input("Start", value="2015-01-01")
    with col_e:
        end_date = st.text_input("End",   value="2024-12-31")

    st.markdown("### 📊 Variable")
    variable = st.selectbox("Seismic variable", list(VARIABLES.keys()),
                             format_func=lambda k: VARIABLES[k])

    st.markdown("### ⚙️ FAWP settings")
    min_mag      = st.slider("Min magnitude", 1.0, 5.0, 2.0, 0.5)
    horizon_days = st.slider("Forecast horizon (days)", 1, 30, 7)
    tau_max      = st.slider("Max tau", 10, 60, 30)
    n_null       = st.slider("Null permutations", 0, 100, 30)
    epsilon      = st.number_input("Epsilon (bits)", value=0.01, min_value=0.001,
                                   max_value=0.5, format="%.3f")

    run_btn = st.button("🌍 Run Seismic Scan", type="primary", use_container_width=True)

# ── Run scan ───────────────────────────────────────────────────────────────────
if run_btn:
    # Validate dates
    try:
        pd.Timestamp(start_date); pd.Timestamp(end_date)
    except Exception:
        st.error("Invalid date format — use YYYY-MM-DD.")
        st.stop()

    if minlat >= maxlat or minlon >= maxlon:
        st.error("Invalid bounding box — check lat/lon min/max.")
        st.stop()

    with st.spinner(f"Fetching earthquakes from USGS… ({region_name})"):
        try:
            raw_df = _fetch_usgs(minlat, maxlat, minlon, maxlon,
                                 start_date, end_date, min_magnitude=min_mag)
            st.session_state["seis_raw"]    = raw_df
            st.session_state["seis_region"] = region_name
            st.session_state["seis_var"]    = variable
            st.session_state["seis_start"]  = start_date
            st.session_state["seis_end"]    = end_date
        except Exception as e:
            st.error(f"Fetch failed: {e}")
            st.stop()

    with st.spinner("Building daily series and running FAWP detection…"):
        try:
            raw_df = st.session_state["seis_raw"]
            daily  = _build_daily_series(raw_df, variable)
            result = _run_seismic_fawp(daily, horizon_days, tau_max, n_null, epsilon)
            st.session_state["seis_result"]  = result
            st.session_state["seis_daily"]   = daily
            st.session_state["seis_epsilon"] = epsilon
            st.session_state["seis_horiz"]   = horizon_days
        except Exception as e:
            st.error(f"FAWP detection failed: {e}")
            st.stop()

# ── Results ────────────────────────────────────────────────────────────────────
if "seis_result" in st.session_state:
    r       = st.session_state["seis_result"]
    raw_df  = st.session_state["seis_raw"]
    daily   = st.session_state["seis_daily"]
    epsilon = st.session_state["seis_epsilon"]
    var_lbl = VARIABLES.get(st.session_state["seis_var"], st.session_state["seis_var"])
    region_lbl = st.session_state["seis_region"]

    # Banner
    bc  = "fawp-yes" if r.fawp_found else "fawp-no"
    bi  = "🔴 SEISMIC FAWP DETECTED" if r.fawp_found else "✅ No FAWP Detected"
    bsub = f"{var_lbl} · {region_lbl} · {r.date_range[0]} → {r.date_range[1]}"
    st.markdown(
        f'<div class="result-banner {bc}">'
        f'<div><div class="result-label">{bi}</div>'
        f'<div class="result-sub">{bsub}</div></div>'
        f'</div>', unsafe_allow_html=True
    )

    # KPIs
    st.markdown('<div class="seis-sec">Key metrics</div>', unsafe_allow_html=True)
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    _kpi(k1, f"{r.peak_gap_bits:.4f}", "Peak gap (bits)")
    _kpi(k2, str(r.odw_result.tau_h_plus) if r.odw_result.tau_h_plus else "—", "τ⁺ₕ horizon")
    _kpi(k3, str(r.odw_result.tau_f) if r.odw_result.tau_f else "—", "τf cliff")
    _kpi(k4, f"τ{r.odw_start}–{r.odw_end}" if r.fawp_found else "—", "ODW",
         "#C0111A" if r.fawp_found else "#3A4E70")
    _kpi(k5, f"{len(raw_df):,}", "Total events")
    _kpi(k6, f"{r.n_obs:,}", "Days analysed")

    # MI chart
    st.markdown('<div class="seis-sec">Prediction vs Steering MI</div>', unsafe_allow_html=True)
    st.pyplot(_mi_chart(r, epsilon), use_container_width=True)

    # Interpretation
    st.markdown('<div class="seis-sec">Interpretation</div>', unsafe_allow_html=True)
    if r.fawp_found:
        st.markdown(
            f"**FAWP detected** — seismic activity in **{region_lbl}** shows the "
            f"Information-Control Exclusion Principle signature: the {var_lbl.lower()} "
            f"remains statistically forecastable (Prediction MI peaks at {r.peak_gap_bits:.3f} bits) "
            f"but the steering/intervention channel has collapsed toward ε={epsilon}. "
            f"\n\nThe **Optimal Decision Window** spans τ={r.odw_start}–{r.odw_end} delay steps. "
            f"The τ⁺ₕ agency horizon is **{r.odw_result.tau_h_plus}** and the failure cliff "
            f"τf is **{r.odw_result.tau_f}** — after which meaningful response is excluded by "
            f"information theory, not just logistics."
        )
    else:
        st.markdown(
            f"No FAWP detected in **{region_lbl}** for this period. "
            f"The steering MI does not fall below ε={epsilon} while prediction MI remains elevated — "
            f"the Information-Control Exclusion Principle is not active. "
            f"Peak gap is {r.peak_gap_bits:.4f} bits."
        )

    # Event charts
    st.markdown('<div class="seis-sec">Seismic data overview</div>', unsafe_allow_html=True)
    tab_events, tab_map = st.tabs(["📈 Time series", "🗺️ Event map"])
    with tab_events:
        st.pyplot(_event_chart(raw_df, daily, var_lbl), use_container_width=True)
    with tab_map:
        st.pyplot(_map_chart(raw_df), use_container_width=True)

    # Stats
    st.markdown('<div class="seis-sec">Statistics</div>', unsafe_allow_html=True)
    sc1, sc2, sc3, sc4 = st.columns(4)
    _kpi(sc1, f"{raw_df['magnitude'].max():.1f}", "Max magnitude")
    _kpi(sc2, f"{raw_df['magnitude'].mean():.2f}", "Mean magnitude")
    _kpi(sc3, f"{raw_df['depth'].mean():.1f} km", "Mean depth")
    _kpi(sc4, f"{len(raw_df) / max(1,(pd.Timestamp(end_date)-pd.Timestamp(start_date)).days):.1f}/day",
         "Avg events/day")

    # Download
    st.markdown('<div class="seis-sec">Export</div>', unsafe_allow_html=True)
    import json as _json
    export = {
        "region": region_lbl,
        "variable": st.session_state["seis_var"],
        "date_range": list(r.date_range),
        "fawp_found": r.fawp_found,
        "peak_gap_bits": round(r.peak_gap_bits, 6),
        "odw_start": r.odw_start,
        "odw_end": r.odw_end,
        "tau_h_plus": r.odw_result.tau_h_plus,
        "tau_f": r.odw_result.tau_f,
        "n_events": len(raw_df),
        "n_obs": r.n_obs,
    }
    st.download_button(
        "⬇ Download result JSON",
        data=_json.dumps(export, indent=2).encode(),
        file_name=f"fawp_seismic_{region_lbl.replace(' ','_').replace('(','').replace(')','')}.json",
        mime="application/json",
    )
