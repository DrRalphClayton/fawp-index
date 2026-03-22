"""
FAWP Seismic Watchlist — save regions + alert on FAWP detection.
Mirrors weather_watchlist.py but for the USGS seismic scanner.
"""
from __future__ import annotations
import os, json
from typing import Optional
import streamlit as st

_TABLE = "fawp_seismic_watchlist"


def _db():
    try:
        from supabase_store import _get_supabase_client, _current_user_id
        return _get_supabase_client(), _current_user_id()
    except Exception:
        return None, None


def render_seismic_watchlist():
    """Render the seismic watchlist UI inside the seismic app sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📌 Watchlist")

    db, uid = _db()
    if db is None or uid is None:
        st.sidebar.caption("Sign in to save regions to your watchlist.")
        return

    # Load existing watchlist entries
    try:
        res = db.table(_TABLE).select("*").eq("user_id", uid).order("created_at").execute()
        entries = res.data or []
    except Exception:
        entries = []

    # Show existing entries
    for entry in entries:
        cfg = entry.get("config", {})
        label = entry.get("name", cfg.get("region", "Unknown"))
        col_lbl, col_del = st.sidebar.columns([4, 1])
        col_lbl.caption(f"📍 {label}")
        if col_del.button("✕", key=f"del_seis_{entry['id']}", help="Remove"):
            try:
                db.table(_TABLE).delete().eq("id", entry["id"]).execute()
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Delete failed: {e}")

    # Save current region
    if st.sidebar.button("💾 Save current region", use_container_width=True, key="save_seis_region"):
        region = st.session_state.get("seis_region", "")
        variable = st.session_state.get("seis_var", "daily_count")
        start = st.session_state.get("seis_start", "2015-01-01")
        end   = st.session_state.get("seis_end",   "2024-12-31")
        if not region:
            st.sidebar.warning("Run a scan first to save the region.")
        else:
            try:
                db.table(_TABLE).insert({
                    "user_id": uid,
                    "name": region,
                    "config": json.dumps({
                        "region": region,
                        "variable": variable,
                        "start_date": start,
                        "end_date": end,
                    }),
                }).execute()
                st.sidebar.success(f"Saved: {region}")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Save failed: {e}")
