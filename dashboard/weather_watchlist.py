"""
dashboard/weather_watchlist.py — Weather watchlist for FAWP Scanner.

Lets signed-in users save locations + hazards for daily monitoring.
Stored in fawp_weather_watchlist Supabase table.
"""
from __future__ import annotations
import os, json
from typing import Optional
import streamlit as st


def _sb():
    from supabase import create_client
    return create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_ANON_KEY"],
    )


def save_weather_location(
    user_id:     str,
    name:        str,
    latitude:    float,
    longitude:   float,
    variable:    str = "temperature_2m",
    hazard:      Optional[str] = None,
    horizon_days: int = 7,
    alert_enabled: bool = True,
) -> bool:
    """Save a location to the user's weather watchlist."""
    try:
        _sb().table("fawp_weather_watchlist").upsert({
            "user_id":      user_id,
            "name":         name[:80],
            "latitude":     latitude,
            "longitude":    longitude,
            "variable":     variable,
            "hazard":       hazard,
            "horizon_days": horizon_days,
            "alert_enabled": alert_enabled,
        }, on_conflict="user_id,name").execute()
        return True
    except Exception:
        return False


def get_weather_watchlist(user_id: str) -> list:
    """Return the user's saved weather locations."""
    try:
        resp = _sb().table("fawp_weather_watchlist") \
                    .select("*") \
                    .eq("user_id", user_id) \
                    .order("created_at", desc=True) \
                    .execute()
        return resp.data or []
    except Exception:
        return []


def delete_weather_location(user_id: str, name: str) -> bool:
    """Delete a saved location."""
    try:
        _sb().table("fawp_weather_watchlist") \
             .delete() \
             .eq("user_id", user_id) \
             .eq("name", name) \
             .execute()
        return True
    except Exception:
        return False


def update_last_result(user_id: str, name: str, result_dict: dict) -> bool:
    """Update the last scan result for a saved location."""
    try:
        import datetime
        _sb().table("fawp_weather_watchlist") \
             .update({
                 "last_scanned": datetime.datetime.utcnow().isoformat(),
                 "last_result":  result_dict,
             }) \
             .eq("user_id", user_id) \
             .eq("name", name) \
             .execute()
        return True
    except Exception:
        return False


def render_weather_watchlist_panel(user_id: str, on_load=None):
    """
    Render the weather watchlist panel.
    
    Parameters
    ----------
    user_id : str
    on_load : callable(lat, lon, variable, hazard)  Called when user clicks Load.
    """
    st.markdown("#### 📍 My Weather Locations")
    locations = get_weather_watchlist(user_id)

    if not locations:
        st.caption("No saved locations yet. Run a scan and click Save Location.")
        return

    for loc in locations:
        col_name, col_result, col_load, col_del = st.columns([3, 2, 1, 1])
        with col_name:
            last_r = loc.get("last_result") or {}
            fawp = last_r.get("fawp_found")
            badge = "🔴" if fawp else ("✅" if fawp is False else "—")
            st.markdown(f"**{badge} {loc['name']}**  \n"
                        f"<span style='color:#5070A0;font-size:.78em'>"
                        f"{loc['variable']} · {loc['latitude']:.1f},{loc['longitude']:.1f}</span>",
                        unsafe_allow_html=True)
        with col_result:
            if fawp is not None:
                gap = last_r.get("peak_gap_bits", 0)
                st.caption(f"Gap: {gap:.4f} bits")
        with col_load:
            if st.button("Load", key=f"wl_load_{loc['id']}"):
                if on_load:
                    on_load(loc["latitude"], loc["longitude"],
                            loc["variable"], loc.get("hazard"))
        with col_del:
            if st.button("✕", key=f"wl_del_{loc['id']}"):
                delete_weather_location(user_id, loc["name"])
                st.rerun()
