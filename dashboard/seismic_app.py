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

try:
    from share import share_button as _seis_share_button
except Exception:
    def _seis_share_button(*a, **k): pass

try:
    from seismic_watchlist import render_seismic_watchlist as _render_seis_wl
except Exception:
    def _render_seis_wl(): pass

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
    "energy_flux":       "Cumulative energy flux (J/day, 30d MA)",
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
    elif variable == "energy_flux":
        # 30-day rolling cumulative seismic energy flux
        df["_e"] = 10 ** (1.5 * df["magnitude"] + 4.8)
        _raw_ef = df.groupby("date")["_e"].sum().reindex(date_range, fill_value=0.0)
        daily = _raw_ef.rolling(30, min_periods=1).mean()
        return daily

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
    fill_val = 0.0 if variable in ("daily_count", "seismic_energy", "energy_flux") else np.nan
    series = series.reindex(date_range, fill_value=fill_val)
    series = series.interpolate(method="linear", limit=7)
    return series


def _run_seismic_fawp(daily_series, horizon_days, tau_max, n_null, epsilon, estimator="pearson"):
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
        estimator     = estimator,
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


def _map_chart_folium(raw_df):
    """Interactive Folium map of earthquake events."""
    try:
        import folium
        from folium.plugins import HeatMap
        import streamlit.components.v1 as components

        center_lat = float(raw_df["lat"].mean())
        center_lon = float(raw_df["lon"].mean())
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=5,
            tiles="CartoDB dark_matter",
        )

        # Heatmap layer
        heat_data = raw_df[["lat", "lon", "magnitude"]].values.tolist()
        HeatMap(heat_data, radius=8, blur=10, max_zoom=8,
                gradient={"0.3": "#D4AF37", "0.6": "#C0111A", "1.0": "#FFFFFF"}).add_to(m)

        # Circle markers for M≥4
        big = raw_df[raw_df["magnitude"] >= 4.0]
        for _, row in big.iterrows():
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=max(3, (row["magnitude"] - 2) * 3),
                color="#C0111A", fill=True, fill_color="#C0111A",
                fill_opacity=0.7, weight=1,
                tooltip=f"M{row['magnitude']:.1f} · {row['time'].strftime('%Y-%m-%d')}",
            ).add_to(m)

        html_str = m._repr_html_()
        components.html(html_str, height=450)
        return None  # rendered directly
    except ImportError:
        return _map_chart_matplotlib(raw_df)


def _map_chart_matplotlib(raw_df):
    """Fallback static scatter map (used if folium not installed)."""
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
    # Custom region presets — save/recall bounding boxes
    _saved_presets = st.session_state.get("seis_saved_presets", {})
    _all_region_options = list(REGIONS.keys()) + list(_saved_presets.keys())
    region_name = st.selectbox("Region preset", _all_region_options,
                               help="Built-in or saved custom regions")
    _sp_col1, _sp_col2 = st.columns([1,1])
    with _sp_col1:
        _sp_name = st.text_input("Save current box as", placeholder="My Region", key="sp_name")
    with _sp_col2:
        if st.button("💾 Save", key="sp_save", use_container_width=True):
            if _sp_name.strip():
                _saved_presets[_sp_name.strip()] = {
                    "minlat": float(st.session_state.get("seis_minlat", -90)),
                    "maxlat": float(st.session_state.get("seis_maxlat", 90)),
                    "minlon": float(st.session_state.get("seis_minlon", -180)),
                    "maxlon": float(st.session_state.get("seis_maxlon", 180)),
                }
                st.session_state["seis_saved_presets"] = _saved_presets
                st.success(f"Saved: {_sp_name.strip()}")
            else:
                st.warning("Enter a name first.")
    if _saved_presets:
        _del_name = st.selectbox("Delete saved preset", ["—"] + list(_saved_presets.keys()),
                                 key="sp_del")
        if _del_name != "—" and st.button("🗑 Delete", key="sp_del_btn"):
            _saved_presets.pop(_del_name, None)
            st.session_state["seis_saved_presets"] = _saved_presets
            st.rerun()

    region = REGIONS.get(region_name) or st.session_state.get("seis_saved_presets", {}).get(region_name)
    if region is None:
        region = list(REGIONS.values())[0]  # fallback
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
    estimator    = st.selectbox("MI estimator", ["pearson", "knn"],
                     key="seis_estimator",
                     help="pearson: fast (default). knn: better for non-Gaussian seismic data.")
    epsilon      = st.number_input("Epsilon (bits)", value=0.01, min_value=0.001,
                                   max_value=0.5, format="%.3f")

    run_btn = st.button("🌍 Run Seismic Scan", type="primary", use_container_width=True)
    st.sidebar.markdown("---")
    _rt_on = st.sidebar.toggle("⚡ Live 30-day feed", key="seis_realtime",
                               help="USGS real-time, refreshes every 5 min")
    if _rt_on:
        st.sidebar.slider("🔔 Alert threshold (M≥)", 3.0, 8.0, 5.0, 0.5,
                          key="seis_alert_mag",
                          help="Toast alert when a new event ≥ this magnitude appears.")
        import time as _trt
        _rt_age = _trt.time() - st.session_state.get("seis_rt_ts", 0)
        if _rt_age > 300 or "seis_rt_raw" not in st.session_state:
            with st.spinner("Fetching real-time data…"):
                try:
                    import urllib.request as _urt2, json as _jrt2
                    _now = pd.Timestamp.now()
                    _rt_url = (f"https://earthquake.usgs.gov/fdsnws/event/1/query"
                               f"?format=geojson&starttime={(_now-pd.Timedelta(days=30)):%Y-%m-%d}"
                               f"&endtime={_now:%Y-%m-%d}"
                               f"&minlatitude={minlat}&maxlatitude={maxlat}"
                               f"&minlongitude={minlon}&maxlongitude={maxlon}"
                               f"&minmagnitude={min_mag}&orderby=time-asc&limit=5000")
                    with _urt2.urlopen(_rt_url, timeout=15) as _r2:
                        _rtd = _jrt2.loads(_r2.read())
                    _rtrows = []
                    for _f2 in _rtd.get("features", []):
                        _p2 = _f2["properties"]; _c2 = _f2["geometry"]["coordinates"]
                        _t2 = pd.to_datetime(_p2["time"], unit="ms", utc=True).tz_localize(None)
                        if _p2.get("mag"):
                            _rtrows.append({"time": _t2, "magnitude": float(_p2["mag"]),
                                            "depth": float(_c2[2]) if len(_c2) > 2 else float("nan"),
                                            "lon": _c2[0], "lat": _c2[1], "date": _t2.normalize()})
                    if _rtrows:
                        st.session_state["seis_rt_raw"] = pd.DataFrame(_rtrows)
                        st.session_state["seis_rt_ts"]  = _trt.time()
                        st.sidebar.success(f"✅ {len(_rtrows)} live events")

                        # ── Real-time magnitude alerts ──────────────────────────────────
                        _alert_thresh = st.session_state.get("seis_alert_mag", 5.0)
                        _prev_max_t   = st.session_state.get("seis_rt_prev_max_t", pd.Timestamp("1970-01-01"))
                        _new_df       = pd.DataFrame(_rtrows)
                        # Find events NEW since last fetch AND above threshold
                        _new_events   = _new_df[
                            (_new_df["time"] > _prev_max_t) &
                            (_new_df["magnitude"] >= _alert_thresh)
                        ].sort_values("magnitude", ascending=False)
                        if len(_new_events) > 0:
                            for _, _ev in _new_events.head(3).iterrows():
                                _icon = "🔴" if _ev["magnitude"] >= 6.0 else "🟠" if _ev["magnitude"] >= 5.0 else "🟡"
                                st.toast(
                                    f"{_icon} M{_ev['magnitude']:.1f} — {region_name}\n"
                                    f"Depth {_ev['depth']:.0f}km · {str(_ev['time'])[:16]}",
                                    icon=_icon,
                                )
                        # Update last-seen timestamp
                        if _rtrows:
                            st.session_state["seis_rt_prev_max_t"] = _new_df["time"].max()
                except Exception as _rte2:
                    st.sidebar.error(f"Real-time fetch failed: {_rte2}")
        else:
            _secs = max(0, 300 - int(_rt_age))
            st.sidebar.caption(f"Cached · next update in {_secs//60}m {_secs%60}s")
        if "seis_rt_raw" in st.session_state:
            if st.sidebar.button("▶ FAWP on live data", key="rt_fawp", use_container_width=True):
                _rtf = st.session_state["seis_rt_raw"]
                _rtd2 = _build_daily_series(_rtf, variable)
                with st.spinner("Running FAWP on live data…"):
                    _rtr = _run_seismic_fawp(_rtd2, horizon_days, tau_max, n_null, epsilon)
                    st.session_state.update({
                        "seis_result": _rtr, "seis_raw": _rtf, "seis_daily": _rtd2,
                        "seis_epsilon": epsilon, "seis_region": f"Live ({region_name})",
                        "seis_var": variable, "seis_start": str(_rtf["date"].min())[:10],
                        "seis_end": str(_rtf["date"].max())[:10]})
                st.rerun()
    _render_seis_wl()

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
            result = _run_seismic_fawp(daily, horizon_days, tau_max, n_null, epsilon, estimator)
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

    # E9.7 timing badge
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

    # MI chart
    st.markdown('<div class="seis-sec">Prediction vs Steering MI</div>', unsafe_allow_html=True)
    _mi_fig = _mi_chart(r, epsilon)
    st.pyplot(_mi_fig, use_container_width=True)
    import io as _seis_io
    _seis_buf = _seis_io.BytesIO()
    _mi_fig.savefig(_seis_buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(_mi_fig)
    _seis_buf.seek(0)
    st.download_button("⬇ Download MI chart PNG", data=_seis_buf,
                       file_name=f"fawp_seismic_{region_lbl.replace(' ','_')}_mi.png",
                       mime="image/png", key="seis_mi_png")

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

    # Post-scan magnitude filter (re-slices without re-fetching)
    st.markdown('<div class="seis-sec">Seismic data overview</div>', unsafe_allow_html=True)
    _filt_col1, _filt_col2, _filt_col3 = st.columns([2, 2, 3])
    with _filt_col1:
        _mag_filter = st.slider("Filter: min magnitude", 0.0, 7.0,
                                float(st.session_state.get("seis_mag_filter", 0.0)),
                                0.5, key="seis_mag_filter",
                                help="Re-slice event map and stats without re-fetching USGS data")
    with _filt_col2:
        _depth_max = st.slider("Max depth (km)", 10, 700,
                               int(st.session_state.get("seis_depth_max", 700)),
                               10, key="seis_depth_max",
                               help="Filter events by focal depth")
    with _filt_col3:
        st.caption(f"Showing events M≥{_mag_filter:.1f} · depth ≤{_depth_max}km")

    # Apply filter to a view (does not re-run FAWP)
    _raw_view = raw_df.copy()
    if _mag_filter > 0:
        _raw_view = _raw_view[_raw_view["magnitude"] >= _mag_filter]
    if _depth_max < 700:
        _raw_view = _raw_view[_raw_view["depth"].fillna(0) <= _depth_max]
    _daily_view = _build_daily_series(_raw_view, st.session_state["seis_var"]) if len(_raw_view) else daily
    st.caption(f"{len(_raw_view):,} events shown (of {len(raw_df):,} total)")

    tab_events, tab_map = st.tabs(["📈 Time series", "🗺️ Event map"])
    with tab_events:
        st.pyplot(_event_chart(_raw_view, _daily_view, var_lbl), use_container_width=True)
    with tab_map:
        result = _map_chart_folium(_raw_view)
        if result is not None:
            st.pyplot(result, use_container_width=True)

    # Stats
    st.markdown('<div class="seis-sec">Statistics</div>', unsafe_allow_html=True)
    sc1, sc2, sc3, sc4 = st.columns(4)
    _kpi(sc1, f"{_raw_view['magnitude'].max():.1f}" if len(_raw_view) else "—", "Max magnitude (filtered)")
    _kpi(sc2, f"{_raw_view['magnitude'].mean():.2f}" if len(_raw_view) else "—", "Mean magnitude (filtered)")
    _kpi(sc3, f"{_raw_view['depth'].mean():.1f} km" if len(_raw_view) else "—", "Mean depth (filtered)")
    _kpi(sc4, f"{len(_raw_view) / max(1,(pd.Timestamp(end_date)-pd.Timestamp(start_date)).days):.1f}/day",
         "Avg events/day (filtered)")

    # Focal depth profile
    st.markdown('<div class="seis-sec">Focal depth profile</div>', unsafe_allow_html=True)
    if len(_raw_view) >= 10 and _raw_view["depth"].notna().sum() >= 10:
        try:
            import matplotlib.pyplot as _plt_dp, numpy as _np_dp, io as _io_dp
            _depths = _raw_view["depth"].dropna().values
            _mags_dp = _raw_view.loc[_raw_view["depth"].notna(), "magnitude"].values
            _fig_dp, (_ax_d1, _ax_d2) = _plt_dp.subplots(1, 2, figsize=(9, 3.5),
                                                            facecolor="#0D1729")
            for _axd in (_ax_d1, _ax_d2):
                _axd.set_facecolor("#07101E")
                for _sp in _axd.spines.values(): _sp.set_edgecolor("#3A4E70")
                _axd.tick_params(colors="#7A90B8", labelsize=8)
            _bins_d = _np_dp.linspace(0, min(_depths.max(), 700), 30)
            _ax_d1.hist(_depths, bins=_bins_d, color="#4A7FCC", alpha=0.7, edgecolor="none")
            _ax_d1.axvline(70,  color="#D4AF37", ls="--", lw=1, label="Crust/mantle (70km)")
            _ax_d1.axvline(300, color="#C0111A", ls="--", lw=1, label="Deep slab (300km)")
            _ax_d1.set_xlabel("Depth (km)", fontsize=8, color="#7A90B8")
            _ax_d1.set_ylabel("Event count", fontsize=8, color="#7A90B8")
            _ax_d1.set_title("Depth histogram", color="#D4AF37", fontsize=9, fontweight="bold")
            _ax_d1.legend(fontsize=7, framealpha=0.2)
            _sc_d = _ax_d2.scatter(_depths, _mags_dp, c=_mags_dp, cmap="YlOrRd",
                                    s=15, alpha=0.6, linewidths=0)
            _plt_dp.colorbar(_sc_d, ax=_ax_d2, label="Magnitude").ax.tick_params(
                colors="#7A90B8", labelsize=7)
            _ax_d2.set_xlabel("Depth (km)", fontsize=8, color="#7A90B8")
            _ax_d2.set_ylabel("Magnitude", fontsize=8, color="#7A90B8")
            _shallow_d  = int((_depths < 70).sum())
            _inter_d    = int(((_depths >= 70) & (_depths < 300)).sum())
            _deep_d     = int((_depths >= 300).sum())
            _dom_d      = max([("shallow", _shallow_d), ("intermediate", _inter_d),
                               ("deep", _deep_d)], key=lambda x: x[1])[0]
            _ax_d2.set_title(f"Depth vs Magnitude (dominant: {_dom_d})",
                             color="#D4AF37", fontsize=8, fontweight="bold")
            _fig_dp.tight_layout()
            st.pyplot(_fig_dp, use_container_width=True)
            _dp_buf = _io_dp.BytesIO()
            _fig_dp.savefig(_dp_buf, format="png", dpi=150, bbox_inches="tight")
            _plt_dp.close(_fig_dp); _dp_buf.seek(0)
            st.download_button("⬇ Depth profile PNG", data=_dp_buf,
                               file_name=f"depth_{region_lbl.replace(' ','_')}.png",
                               mime="image/png", key="dp_dl")
        except Exception as _dpe:
            st.caption(f"Depth profile unavailable: {_dpe}")
    else:
        st.caption("Need ≥10 events with depth data.")

    # Gutenberg-Richter b-value analysis
    st.markdown('<div class="seis-sec">Gutenberg-Richter Analysis</div>', unsafe_allow_html=True)
    if len(_raw_view) >= 20:
        try:
            import numpy as _np_gr, matplotlib.pyplot as _plt_gr, io as _io_gr
            _mags = _raw_view["magnitude"].dropna().values
            _mc   = float(_mags.min())
            _b    = 1.0 / (_np_gr.log(10) * (_np_gr.mean(_mags) - _mc + 0.05))
            _a    = _np_gr.log10(len(_mags)) + _b * _mc
            _gc1, _gc2, _gc3 = st.columns(3)
            _kpi(_gc1, f"{_b:.3f}", "b-value (Aki MLE)",
                 "#1DB954" if 0.6 <= _b <= 1.4 else "#C0111A")
            _kpi(_gc2, f"{_a:.2f}", "a-value")
            _kpi(_gc3, f"M≥{_mc:.1f}", "Completeness Mc")
            if _b < 0.6:
                st.warning(f"⚠️ b={_b:.3f} — low, may indicate stress accumulation.")
            elif _b > 1.4:
                st.info(f"ℹ️ b={_b:.3f} — high, typical of volcanic/aftershock regions.")
            else:
                st.success(f"✅ b={_b:.3f} within typical tectonic range (0.6–1.4).")
            _mb = _np_gr.arange(_mc, _mags.max() + 0.5, 0.5)
            _cc = [_np_gr.sum(_mags >= m) for m in _mb]
            _fg, _ax = _plt_gr.subplots(figsize=(7, 3), facecolor="#0D1729")
            _ax.set_facecolor("#07101E")
            _ax.scatter(_mb, _np_gr.log10(_np_gr.maximum(_cc, 1)),
                        color="#D4AF37", s=25, zorder=3, label="Observed")
            _ax.plot(_mb, _a - _b * _mb, color="#4A7FCC", lw=1.5, ls="--",
                     label=f"G-R fit (b={_b:.2f})")
            for _sp in _ax.spines.values(): _sp.set_edgecolor("#3A4E70")
            _ax.tick_params(colors="#7A90B8", labelsize=8)
            _ax.set_xlabel("Magnitude", fontsize=8, color="#7A90B8")
            _ax.set_ylabel("log₁₀(N ≥ M)", fontsize=8, color="#7A90B8")
            _ax.set_title("Gutenberg-Richter", color="#D4AF37", fontsize=9, fontweight="bold")
            _ax.legend(fontsize=8, framealpha=0.2)
            _fg.tight_layout()
            st.pyplot(_fg, use_container_width=True)
            _grbuf = _io_gr.BytesIO()
            _fg.savefig(_grbuf, format="png", dpi=150, bbox_inches="tight")
            _plt_gr.close(_fg); _grbuf.seek(0)
            st.download_button("⬇ Download G-R PNG", data=_grbuf,
                               file_name=f"gr_{region_lbl.replace(' ','_')}.png",
                               mime="image/png", key="gr_dl")
        except Exception as _gre:
            st.caption(f"G-R analysis unavailable: {_gre}")
    else:
        st.caption("Need ≥20 events for G-R analysis.")

    # Download
    with st.expander("🔬 Aftershock sequence analysis"):
        st.caption("Detects mainshock-aftershock patterns via inter-event time distribution (Omori-Utsu proxy).")
        if len(_raw_view) >= 20:
            try:
                import numpy as _np_as, matplotlib.pyplot as _plt_as, io as _io_as
                _t_as = _raw_view.sort_values("time")["time"].values
                _t_s  = _np_as.array([_np_as.datetime64(t, "s").astype(float) for t in _t_as])
                _iet  = _np_as.diff(_t_s) / 3600
                _iet  = _iet[_iet > 0]
                if len(_iet) >= 10:
                    _log_iet = _np_as.log10(_iet)
                    _hv, _hb = _np_as.histogram(_log_iet, bins=20)
                    _bc = (_hb[:-1] + _hb[1:]) / 2
                    _mk = _hv > 0
                    if _mk.sum() >= 4:
                        _slope_as = _np_as.polyfit(_bc[_mk], _np_as.log10(_hv[_mk]+1), 1)[0]
                        _is_as = _slope_as < -0.5
                        _lbl_as = "🔴 Aftershock sequence (Omori decay)" if _is_as else "✅ Background seismicity"
                        st.markdown(f"**{_lbl_as}**")
                        st.caption(f"Log-log IET slope: {_slope_as:.3f} (threshold: −0.5)")
                        _fig_as, _ax_as = _plt_as.subplots(figsize=(7, 3), facecolor="#0D1729")
                        _ax_as.set_facecolor("#07101E")
                        for _sp in _ax_as.spines.values(): _sp.set_edgecolor("#3A4E70")
                        _ax_as.tick_params(colors="#7A90B8", labelsize=8)
                        _bw = _bc[1]-_bc[0] if len(_bc)>1 else 0.1
                        _ax_as.bar(_bc, _hv, width=_bw,
                                   color="#C0111A" if _is_as else "#1DB954", alpha=0.7)
                        _xf = _np_as.linspace(_bc[_mk].min(), _bc[_mk].max(), 40)
                        _c0 = _np_as.polyfit(_bc[_mk], _np_as.log10(_hv[_mk]+1), 1)[1]
                        _yf = _np_as.maximum(10**(_c0 + _slope_as*_xf) - 1, 0)
                        _ax_as.plot(_xf, _yf, color="#D4AF37", lw=1.5, ls="--",
                                    label=f"slope={_slope_as:.2f}")
                        _ax_as.set_xlabel("log₁₀(inter-event time, hours)", fontsize=8, color="#7A90B8")
                        _ax_as.set_ylabel("Count", fontsize=8, color="#7A90B8")
                        _ax_as.set_title("Inter-event time distribution (Omori-Utsu proxy)",
                                         color="#D4AF37", fontsize=9)
                        _ax_as.legend(fontsize=7, framealpha=0.2)
                        _fig_as.tight_layout()
                        st.pyplot(_fig_as, use_container_width=True)
                        _plt_as.close(_fig_as)
            except Exception as _ase:
                st.caption(f"Aftershock analysis failed: {_ase}")
        else:
            st.caption("Need ≥20 events for aftershock analysis.")

    with st.expander("🌐 Global batch scan — all regions"):
        st.caption("Scans all 10 presets and shows a summary table.")
        _bvar  = st.selectbox("Variable", list(VARIABLES.keys()),
                              format_func=lambda k: VARIABLES[k], key="batch_var")
        _bst   = st.text_input("Start", "2015-01-01", key="batch_st")
        _ben   = st.text_input("End",   "2024-12-31", key="batch_en")
        _bmag  = st.slider("Min magnitude", 1.0, 5.0, 2.0, 0.5, key="batch_mag")
        if st.button("🌐 Run batch scan", key="run_batch", type="primary",
                     use_container_width=True):
            _brows = []
            _bprog = st.progress(0.0)
            _brl   = [(n, c) for n, c in REGIONS.items() if c]
            for _bi, (_bn, _bc) in enumerate(_brl):
                _bprog.progress((_bi + 1) / len(_brl), f"{_bn}…")
                try:
                    _brd  = _fetch_usgs(_bc["minlat"], _bc["maxlat"],
                                        _bc["minlon"], _bc["maxlon"],
                                        _bst, _ben, _bmag)
                    _bdd  = _build_daily_series(_brd, _bvar)
                    _brr  = _run_seismic_fawp(_bdd, 7, 30, 20, 0.01)
                    _brows.append({
                        "Region":   _bn,
                        "FAWP":     "🔴 YES" if _brr.fawp_found else "—",
                        "Peak gap": f"{_brr.peak_gap_bits:.4f}",
                        "ODW":      f"τ{_brr.odw_start}–{_brr.odw_end}" if _brr.fawp_found else "—",
                        "Events":   len(_brd),
                    })
                except Exception as _berr:
                    _brows.append({"Region": _bn, "FAWP": "ERR",
                                   "Peak gap": str(_berr)[:40], "ODW": "—", "Events": 0})
            _bprog.empty()
            import pandas as _pd_b
            _bdf = _pd_b.DataFrame(_brows).sort_values("Peak gap", ascending=False)
            st.dataframe(_bdf, use_container_width=True, hide_index=True)
            st.download_button("⬇ Download batch CSV", data=_bdf.to_csv(index=False).encode(),
                               file_name="fawp_batch.csv", mime="text/csv", key="batch_dl")

            # World map overlay of batch results
            try:
                import folium, streamlit.components.v1 as _comp
                _wm = folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB dark_matter")
                _REGION_CENTERS = {
                    "California (San Andreas)":  [37.0, -119.0],
                    "Japan":                     [38.0, 137.0],
                    "Turkey (Anatolian Fault)":  [39.0, 35.0],
                    "Chile":                     [-36.0, -71.0],
                    "New Zealand":               [-41.0, 172.0],
                    "Indonesia / Sumatra":       [-2.0, 118.0],
                    "Alaska":                    [62.0, -155.0],
                    "Greece / Aegean":           [38.5, 24.0],
                    "Italy (Apennines)":         [42.0, 13.0],
                    "Mexico (Pacific coast)":    [18.0, -96.0],
                }
                for _, _row in _bdf.iterrows():
                    _ctr = _REGION_CENTERS.get(_row["Region"])
                    if not _ctr: continue
                    _is_fawp = "🔴" in str(_row["FAWP"])
                    _gap_f = 0.0
                    try: _gap_f = float(_row["Peak gap"])
                    except Exception: pass
                    folium.CircleMarker(
                        location=_ctr,
                        radius=max(8, min(25, _gap_f * 60)),
                        color="#C0111A" if _is_fawp else "#1DB954",
                        fill=True, fill_color="#C0111A" if _is_fawp else "#1DB954",
                        fill_opacity=0.7, weight=2,
                        tooltip=f"{_row['Region']}: FAWP={'YES' if _is_fawp else 'NO'} · gap={_gap_f:.4f}b",
                    ).add_to(_wm)
                _comp.html(_wm._repr_html_(), height=420)
            except ImportError:
                st.caption("Install folium for world map: pip install folium")

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
    col_dl, col_sh = st.columns([2, 1])
    with col_dl:
        st.download_button(
            "⬇ Download result JSON",
            data=_json.dumps(export, indent=2).encode(),
            file_name=f"fawp_seismic_{region_lbl.replace(' ','_').replace('(','').replace(')','')}.json",
            mime="application/json",
        )
    with col_sh:
        _seis_share_button("seismic", f"FAWP Seismic — {var_lbl} {region_lbl}", export)
