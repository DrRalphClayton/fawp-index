"""
dashboard/share.py — Shared result links for FAWP Scanner.

Generates short shareable URLs for finance and weather scan results.
Results stored in Supabase fawp_shared_results table (public read).
"""
from __future__ import annotations
import hashlib, json, time, os
from typing import Optional, Any
import streamlit as st

_BASE_URL = os.environ.get("FAWP_BASE_URL", "https://fawp-scanner.info")


def _short_id(payload: str) -> str:
    """Generate 8-char hex ID from content hash + timestamp."""
    h = hashlib.sha256(f"{payload}{time.time()}".encode()).hexdigest()
    return h[:8]


def create_share(
    result_type: str,
    title:       str,
    payload:     Any,
    user_id:     Optional[str] = None,
    expires_days: Optional[int] = None,
) -> Optional[str]:
    """
    Store a result and return a shareable URL.

    Parameters
    ----------
    result_type : "finance" | "weather"
    title : str   Short label shown in shared view.
    payload : any JSON-serialisable result dict.
    user_id : str  Supabase user UUID.
    expires_days : int  Days until link expires. None = never.

    Returns
    -------
    str | None  Full URL e.g. https://fawp-scanner.info/?share=ab12cd34
    """
    try:
        from supabase import create_client
        url  = os.environ.get("SUPABASE_URL", "")
        key  = os.environ.get("SUPABASE_ANON_KEY", "")
        if not url or not key:
            return None
        sb = create_client(url, key)

        payload_str = json.dumps(payload, default=str)
        share_id    = _short_id(payload_str)

        row = {
            "id":          share_id,
            "user_id":     user_id,
            "result_type": result_type,
            "title":       title[:120],
            "payload":     json.loads(payload_str),
        }
        if expires_days:
            import datetime
            row["expires_at"] = (
                datetime.datetime.utcnow() +
                datetime.timedelta(days=expires_days)
            ).isoformat()

        sb.table("fawp_shared_results").insert(row).execute()
        return f"{_BASE_URL}/?share={share_id}"

    except Exception:
        return None


def load_share(share_id: str) -> Optional[dict]:
    """
    Load a shared result by ID and increment view count.

    Returns
    -------
    dict | None   {"result_type", "title", "payload", "created_at"}
    """
    try:
        from supabase import create_client
        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_ANON_KEY", "")
        sb  = create_client(url, key)

        resp = sb.table("fawp_shared_results") \
                 .select("*") \
                 .eq("id", share_id) \
                 .single() \
                 .execute()
        if not resp.data:
            return None

        # Increment view count (best-effort)
        try:
            sb.table("fawp_shared_results") \
              .update({"view_count": resp.data["view_count"] + 1}) \
              .eq("id", share_id).execute()
        except Exception:
            pass

        return resp.data
    except Exception:
        return None


def share_button(result_type: str, title: str, payload: Any, user_id: Optional[str] = None):
    """
    Render a Share button. On click, creates a share and shows the URL.
    """
    if st.button("🔗 Share result", key=f"share_{result_type}_{hash(title)}"):
        with st.spinner("Creating shareable link…"):
            url = create_share(result_type, title, payload, user_id)
        if url:
            st.success(f"Shareable link created!")
            st.code(url)
            st.caption("Anyone with this link can view your result — no login needed.")
        else:
            # Fallback: show JSON
            st.warning("Could not create link (Supabase unavailable). Copy the JSON below:")
            st.code(json.dumps(payload, default=str, indent=2)[:2000])
