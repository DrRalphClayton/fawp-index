"""
dashboard/supabase_store.py — Per-user Supabase-backed storage.

Replaces the shared filesystem for scan history and watchlists.
Falls back to local filesystem if Supabase is not configured.

Two Supabase tables required (SQL in SUPABASE_SETUP.md):
    fawp_scan_history  (id, user_id, scanned_at, label, payload)
    fawp_watchlists    (id, user_id, name, config, created_at, last_scanned)

Usage::

    from supabase_store import get_store
    store = get_store()               # auto-detects user + backend
    store.save_scan(wl_result)
    tl = store.asset_timeline("SPY", "1d")
    store.save_watchlist("tech", ["AAPL", "MSFT"], period="2y")
    wls = store.list_watchlists()
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd


def _supabase_client():
    """Return Supabase client or None if not configured."""
    try:
        import streamlit as st
        from supabase import create_client
        try:
            url = st.secrets.get("SUPABASE_URL") or os.environ.get("SUPABASE_URL", "")
            key = (st.secrets.get("SUPABASE_KEY")
                   or st.secrets.get("SUPABASE_ANON_KEY")
                   or os.environ.get("SUPABASE_KEY", "")
                   or os.environ.get("SUPABASE_ANON_KEY", ""))
        except Exception:
            url = os.environ.get("SUPABASE_URL", "")
            key = (os.environ.get("SUPABASE_KEY", "")
                   or os.environ.get("SUPABASE_ANON_KEY", ""))
        if not url or not key:
            return None
        return create_client(url, key)
    except Exception:
        return None


def _current_user_id() -> Optional[str]:
    """Return current Supabase user ID from session_state, or None."""
    try:
        import streamlit as st
        session = st.session_state.get("supabase_session")
        if session:
            return session.get("user", {}).get("id")
    except Exception:
        pass
    return None


def _current_user_email() -> Optional[str]:
    """Return current user email or None."""
    try:
        import streamlit as st
        session = st.session_state.get("supabase_session")
        if session:
            return session.get("user", {}).get("email")
    except Exception:
        pass
    return None


# ── LocalFallback ─────────────────────────────────────────────────────────────

class _LocalStore:
    """Local filesystem fallback (single-user)."""

    def __init__(self):
        from fawp_index.scan_history import ScanHistory
        from fawp_index.watchlist_store import WatchlistStore
        self._hist = ScanHistory()
        self._wl   = WatchlistStore()

    def save_scan(self, wl_result, label: str = ""):
        try:
            self._hist.save(wl_result, label=label)
        except Exception:
            pass

    def asset_timeline(self, ticker: str, tf: str = "1d",
                       last_n: int = 0) -> pd.DataFrame:
        return self._hist.asset_timeline(ticker, tf, last_n=last_n)

    def all_assets(self) -> List[dict]:
        return self._hist.all_assets()

    def recent_scans(self, n: int = 10) -> List[dict]:
        return self._hist.recent(n=n)

    def first_onset(self, ticker: str, tf: str = "1d") -> Optional[str]:
        return self._hist.first_onset(ticker, tf)

    def n_snapshots(self) -> int:
        return self._hist.n_snapshots()

    def save_watchlist(self, name: str, tickers: List[str],
                       period: str = "2y",
                       timeframes: Optional[List[str]] = None,
                       overwrite: bool = True):
        self._wl.create(name, tickers, period=period,
                        timeframes=timeframes, overwrite=overwrite)

    def list_watchlists(self) -> List[dict]:
        out = []
        for n in self._wl.list():
            info = self._wl.show(n)
            out.append({"name": n, **info})
        return out

    def get_watchlist(self, name: str) -> Optional[dict]:
        try:
            return self._wl.show(name)
        except KeyError:
            return None

    def delete_watchlist(self, name: str):
        self._wl.delete(name)

    def scan_watchlist(self, name: str, **kwargs):
        return self._wl.scan(name, **kwargs)


# ── SupabaseStore ─────────────────────────────────────────────────────────────

class _SupabaseStore:
    """Per-user Supabase-backed store."""

    def __init__(self, client, user_id: str, user_email: str):
        self._db       = client
        self._user_id  = user_id
        self._email    = user_email

    # ── Scan history ──────────────────────────────────────────────────────────

    def save_scan(self, wl_result, label: str = ""):
        try:
            assets = [
                {
                    "ticker":          a.ticker,
                    "timeframe":       a.timeframe,
                    "latest_score":    round(float(a.latest_score),  6),
                    "peak_gap_bits":   round(float(a.peak_gap_bits), 6),
                    "regime_active":   bool(a.regime_active),
                    "days_in_regime":  int(a.days_in_regime),
                    "signal_age_days": int(a.signal_age_days),
                    "odw_start":       a.peak_odw_start,
                    "odw_end":         a.peak_odw_end,
                }
                for a in wl_result.assets if not a.error
            ]
            self._db.table("fawp_scan_history").insert({
                "user_id":    self._user_id,
                "scanned_at": datetime.now().isoformat(),
                "label":      label,
                "n_assets":   wl_result.n_assets,
                "n_flagged":  wl_result.n_flagged,
                "payload":    json.dumps(assets),
            }).execute()
        except Exception:
            pass

    def _load_history_rows(self, limit: int = 500) -> List[dict]:
        try:
            res = (self._db.table("fawp_scan_history")
                   .select("*")
                   .eq("user_id", self._user_id)
                   .order("scanned_at", desc=False)
                   .limit(limit)
                   .execute())
            return res.data or []
        except Exception:
            return []

    def asset_timeline(self, ticker: str, tf: str = "1d",
                       last_n: int = 0) -> pd.DataFrame:
        rows_out = []
        for row in self._load_history_rows():
            try:
                assets = json.loads(row.get("payload", "[]"))
            except Exception:
                continue
            for a in assets:
                if a.get("ticker") == ticker and a.get("timeframe") == tf:
                    rows_out.append({
                        "scanned_at":     row["scanned_at"],
                        "latest_score":   a["latest_score"],
                        "peak_gap_bits":  a["peak_gap_bits"],
                        "regime_active":  a["regime_active"],
                        "days_in_regime": a["days_in_regime"],
                        "signal_age_days":a["signal_age_days"],
                        "odw_start":      a.get("odw_start"),
                        "odw_end":        a.get("odw_end"),
                    })
        if not rows_out:
            return pd.DataFrame()
        df = pd.DataFrame(rows_out)
        df["scanned_at"] = pd.to_datetime(df["scanned_at"])
        df = df.sort_values("scanned_at").reset_index(drop=True)
        if last_n > 0:
            df = df.tail(last_n).reset_index(drop=True)
        return df

    def all_assets(self) -> List[dict]:
        latest: Dict[str, dict] = {}
        for row in self._load_history_rows():
            try:
                assets = json.loads(row.get("payload", "[]"))
            except Exception:
                continue
            for a in assets:
                key = f"{a.get('ticker')}|{a.get('timeframe')}"
                latest[key] = {
                    "ticker":        a.get("ticker"),
                    "timeframe":     a.get("timeframe"),
                    "latest_score":  a.get("latest_score", 0.0),
                    "regime_active": a.get("regime_active", False),
                    "last_seen":     row.get("scanned_at", ""),
                }
        return sorted(latest.values(), key=lambda x: x["latest_score"], reverse=True)

    def recent_scans(self, n: int = 10) -> List[dict]:
        rows = self._load_history_rows(limit=n * 2)[-n:][::-1]
        return [
            {
                "scanned_at": r.get("scanned_at", ""),
                "n_assets":   r.get("n_assets",  0),
                "n_flagged":  r.get("n_flagged", 0),
                "label":      r.get("label", ""),
            }
            for r in rows
        ]

    def first_onset(self, ticker: str, tf: str = "1d") -> Optional[str]:
        tl = self.asset_timeline(ticker, tf)
        if tl.empty:
            return None
        active = tl[tl["regime_active"]]
        if active.empty:
            return None
        return str(active.iloc[0]["scanned_at"].date())

    def n_snapshots(self) -> int:
        try:
            res = (self._db.table("fawp_scan_history")
                   .select("id", count="exact")
                   .eq("user_id", self._user_id)
                   .execute())
            return res.count or 0
        except Exception:
            return 0

    # ── Watchlists ────────────────────────────────────────────────────────────

    def save_watchlist(self, name: str, tickers: List[str],
                       period: str = "2y",
                       timeframes: Optional[List[str]] = None,
                       overwrite: bool = True):
        cfg = json.dumps({
            "tickers":    tickers,
            "period":     period,
            "timeframes": timeframes or ["1d"],
        })
        try:
            existing = (self._db.table("fawp_watchlists")
                        .select("id")
                        .eq("user_id", self._user_id)
                        .eq("name", name)
                        .execute())
            if existing.data and overwrite:
                (self._db.table("fawp_watchlists")
                 .update({"config": cfg})
                 .eq("user_id", self._user_id)
                 .eq("name", name)
                 .execute())
            elif not existing.data:
                (self._db.table("fawp_watchlists")
                 .insert({
                     "user_id":    self._user_id,
                     "name":       name,
                     "config":     cfg,
                     "created_at": datetime.now().isoformat(),
                 }).execute())
        except Exception:
            pass

    def list_watchlists(self) -> List[dict]:
        try:
            res = (self._db.table("fawp_watchlists")
                   .select("*")
                   .eq("user_id", self._user_id)
                   .order("name")
                   .execute())
            out = []
            for row in (res.data or []):
                try:
                    cfg = json.loads(row.get("config", "{}"))
                    out.append({
                        "name":        row["name"],
                        "tickers":     cfg.get("tickers", []),
                        "period":      cfg.get("period", "2y"),
                        "timeframes":  cfg.get("timeframes", ["1d"]),
                        "created_at":  row.get("created_at", ""),
                        "last_scanned":row.get("last_scanned"),
                    })
                except Exception:
                    continue
            return out
        except Exception:
            return []

    def get_watchlist(self, name: str) -> Optional[dict]:
        wls = self.list_watchlists()
        for w in wls:
            if w["name"] == name:
                return w
        return None

    def delete_watchlist(self, name: str):
        try:
            (self._db.table("fawp_watchlists")
             .delete()
             .eq("user_id", self._user_id)
             .eq("name", name)
             .execute())
        except Exception:
            pass

    def scan_watchlist(self, name: str, **kwargs):
        from fawp_index.watchlist import scan_watchlist
        wl = self.get_watchlist(name)
        if wl is None:
            raise KeyError(f"Watchlist '{name}' not found")
        kw = dict(period=wl["period"], timeframes=wl["timeframes"])
        kw.update(kwargs)
        result = scan_watchlist(wl["tickers"], **kw)
        try:
            (self._db.table("fawp_watchlists")
             .update({"last_scanned": datetime.now().isoformat()})
             .eq("user_id", self._user_id)
             .eq("name", name)
             .execute())
        except Exception:
            pass
        return result


# ── Public factory ────────────────────────────────────────────────────────────

def get_store() -> "_LocalStore | _SupabaseStore":
    """
    Return the appropriate store for the current user.

    Returns a _SupabaseStore if Supabase is configured and user is logged in,
    otherwise returns a _LocalStore (filesystem fallback).
    """
    client  = _supabase_client()
    user_id = _current_user_id()
    email   = _current_user_email()
    if client and user_id:
        return _SupabaseStore(client, user_id, email or "")
    return _LocalStore()


# ── Email alert helper ────────────────────────────────────────────────────────

def send_alert_email(subject: str, body: str) -> bool:
    """
    Send an alert email to the current user via Supabase.

    Uses Supabase's built-in email service (requires SMTP configured
    in Supabase → Project Settings → Auth → SMTP Settings).

    Parameters
    ----------
    subject : str
    body    : str  — plain text body

    Returns
    -------
    bool — True if sent successfully
    """
    client = _supabase_client()
    email  = _current_user_email()
    if not client or not email:
        return False

    # Use service role key for admin email operations
    service_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not service_key:
        return False

    try:
        from supabase import create_client
        import streamlit as st
        try:
            url = st.secrets.get("SUPABASE_URL") or os.environ.get("SUPABASE_URL", "")
        except Exception:
            url = os.environ.get("SUPABASE_URL", "")

        admin = create_client(url, service_key)
        # Use Supabase Edge Functions or direct SMTP
        # For now use a webhook to a simple email service
        import urllib.request
        import json as _json
        payload = _json.dumps({
            "to":      email,
            "subject": subject,
            "body":    body,
        }).encode()
        # Trigger via Supabase edge function (user must deploy this separately)
        fn_url = f"{url}/functions/v1/send-alert-email"
        req = urllib.request.Request(
            fn_url, data=payload,
            headers={
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {service_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status in (200, 201, 204)
    except Exception:
        return False
