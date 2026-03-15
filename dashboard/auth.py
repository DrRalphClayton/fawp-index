"""
dashboard/auth.py — Supabase email authentication for FAWP Scanner.

Required env / Streamlit secrets:
    SUPABASE_URL      — https://xxxx.supabase.co
    SUPABASE_KEY      — anon/public key  (or SUPABASE_ANON_KEY)

Install: pip install supabase
"""

from __future__ import annotations

import os
from typing import Optional

import streamlit as st

_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client

    try:
        from supabase import create_client
    except ImportError:
        st.error("supabase-py not installed. Run: `pip install supabase`")
        st.stop()

    try:
        url = (st.secrets.get("SUPABASE_URL")
               or os.environ.get("SUPABASE_URL", ""))
        key = (st.secrets.get("SUPABASE_KEY")
               or st.secrets.get("SUPABASE_ANON_KEY")
               or os.environ.get("SUPABASE_KEY", "")
               or os.environ.get("SUPABASE_ANON_KEY", ""))
    except Exception:
        url = os.environ.get("SUPABASE_URL", "")
        key = (os.environ.get("SUPABASE_KEY", "")
               or os.environ.get("SUPABASE_ANON_KEY", ""))

    if not url or not key:
        st.error(
            "Missing Supabase credentials.\n\n"
            "Add to `.streamlit/secrets.toml`:\n"
            "```\nSUPABASE_URL = \"https://xxxx.supabase.co\"\n"
            "SUPABASE_KEY = \"eyJhbGci...\"\n```"
        )
        st.stop()

    _client = create_client(url, key)

    # Restore session into the client on every Streamlit rerun
    sess = st.session_state.get("supabase_session")
    if sess and sess.get("access_token") and sess.get("refresh_token"):
        try:
            _client.auth.set_session(
                sess["access_token"], sess["refresh_token"]
            )
        except Exception:
            st.session_state.pop("supabase_session", None)

    return _client


def _store_session(auth_response):
    if auth_response is None:
        st.session_state.pop("supabase_session", None)
        return
    session_obj = getattr(auth_response, "session", auth_response)
    if session_obj is None:
        st.session_state.pop("supabase_session", None)
        return
    st.session_state["supabase_session"] = {
        "access_token":  getattr(session_obj, "access_token",  None),
        "refresh_token": getattr(session_obj, "refresh_token", None),
    }


def get_user_email() -> Optional[str]:
    try:
        user_resp = _get_client().auth.get_user()
        user = getattr(user_resp, "user", None) or user_resp
        return getattr(user, "email", None)
    except Exception:
        return None


def is_authenticated() -> bool:
    sess = st.session_state.get("supabase_session")
    return bool(sess and sess.get("access_token") and sess.get("refresh_token"))


def sign_in(email: str, password: str) -> tuple[bool, str]:
    try:
        res = _get_client().auth.sign_in_with_password(
            {"email": email, "password": password}
        )
        _store_session(res)
        return True, ""
    except Exception as e:
        err = str(e)
        if "Invalid login" in err or "invalid_credentials" in err:
            return False, "Invalid email or password."
        if "Email not confirmed" in err:
            return False, "Please confirm your email before signing in."
        return False, f"Sign-in failed: {err}"


def sign_up(email: str, password: str) -> tuple[bool, str]:
    try:
        res = _get_client().auth.sign_up(
            {"email": email, "password": password}
        )
        user    = getattr(res, "user",    None)
        session = getattr(res, "session", None)
        if session:
            _store_session(res)
            return True, "Account created and signed in."
        if user:
            return True, "Account created. Check your email and confirm your address."
        return False, "Sign-up failed."
    except Exception as e:
        err = str(e)
        if "already registered" in err or "User already registered" in err:
            return False, "This email is already registered. Try signing in."
        if "Password should be" in err:
            return False, "Password must be at least 6 characters."
        return False, f"Sign-up failed: {err}"


def sign_out():
    try:
        _get_client().auth.sign_out()
    except Exception:
        pass
    st.session_state.pop("supabase_session", None)
    st.rerun()


def reset_password(email: str) -> tuple[bool, str]:
    try:
        redirect = os.environ.get(
            "SUPABASE_RESET_REDIRECT", "https://fawp-scanner.info"
        )
        _get_client().auth.reset_password_for_email(
            email, {"redirect_to": redirect}
        )
        return True, "Password reset email sent. Check your inbox."
    except Exception as e:
        return False, f"Reset failed: {e}"



# ── Plan / entitlement helpers ────────────────────────────────────────────────

_PLAN_LIMITS = {
    "free": {
        "max_tickers":    3,
        "max_history":    10,
        "alerts":         False,
        "schedule":       False,
        "leaderboard":    False,
        "export_pdf":     False,
    },
    "pro": {
        "max_tickers":    999,
        "max_history":    500,
        "alerts":         True,
        "schedule":       True,
        "leaderboard":    True,
        "export_pdf":     True,
    },
    "admin": {
        "max_tickers":    999,
        "max_history":    9999,
        "alerts":         True,
        "schedule":       True,
        "leaderboard":    True,
        "export_pdf":     True,
        "admin_panel":    True,
    },
}


def get_plan() -> str:
    """
    Return the current user's plan: 'free', 'pro', or 'admin'.
    Falls back to 'free' if not authenticated or on any error.
    """
    if not is_authenticated():
        return "free"
    # Cache in session_state to avoid repeated DB hits
    cached = st.session_state.get("_user_plan")
    if cached:
        return cached
    try:
        client = _get_client()
        res = client.table("profiles").select("plan").eq(
            "id", _get_user_id()
        ).single().execute()
        plan = (res.data or {}).get("plan", "free")
    except Exception:
        plan = "free"
    st.session_state["_user_plan"] = plan
    return plan


def get_limit(feature: str):
    """Return the limit for a feature given the current plan."""
    plan = get_plan()
    limits = _PLAN_LIMITS.get(plan, _PLAN_LIMITS["free"])
    return limits.get(feature)


def is_pro() -> bool:
    return get_plan() in ("pro", "admin")


def is_admin() -> bool:
    return get_plan() == "admin"


def _get_user_id() -> Optional[str]:
    """Return current user's UUID from session_state."""
    sess = st.session_state.get("supabase_session")
    if not sess:
        return None
    try:
        client = _get_client()
        user_resp = client.auth.get_user()
        user = getattr(user_resp, "user", None) or user_resp
        return getattr(user, "id", None)
    except Exception:
        return None


def clear_plan_cache():
    """Call after plan upgrade so the next get_plan() re-fetches."""
    st.session_state.pop("_user_plan", None)

# ── Login wall UI ─────────────────────────────────────────────────────────────

def require_auth():
    """
    Render the login wall if the user is not authenticated.
    Calls st.stop() so nothing below renders until signed in.
    """
    if is_authenticated():
        return
    # Allow demo bypass
    if st.session_state.get("_demo_bypass"):
        return

    # Hide Streamlit chrome and sidebar completely on login page
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500;600&display=swap');

    /* Hide all Streamlit UI chrome */
    #MainMenu, header, footer, [data-testid="stSidebar"],
    [data-testid="collapsedControl"], [data-testid="stToolbar"] {
        display: none !important;
    }

    /* Full-page dark background */
    html, body, [data-testid="stAppViewContainer"],
    [data-testid="stMain"], .main, .block-container {
        background: #07101E !important;
        padding: 0 !important;
        margin: 0 !important;
        max-width: 100% !important;
    }

    /* Centre the content */
    [data-testid="stVerticalBlock"] {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-start;
        min-height: 100vh;
        padding: 0 !important;
    }

    /* ── Card ── */
    .login-card {
        width: 100%;
        max-width: 420px;
        margin: 7vh auto 0;
        background: #0D1729;
        border: 1px solid #1E2E4A;
        border-top: 3px solid #D4AF37;
        border-radius: 10px;
        padding: 2.6em 2.4em 2.2em;
        box-shadow: 0 8px 40px rgba(0,0,0,0.5);
    }

    /* ── Logo ── */
    .login-logo {
        font-family: 'Syne', sans-serif;
        font-size: 1.9em;
        font-weight: 800;
        color: #D4AF37;
        letter-spacing: -0.01em;
        margin-bottom: 0.15em;
    }
    .login-sub {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.8em;
        color: #3A4E70;
        margin-bottom: 2em;
        letter-spacing: 0.01em;
    }
    .login-divider {
        border: none;
        border-top: 1px solid #1E2E4A;
        margin: 1.4em 0;
    }

    /* ── Inputs ── */
    [data-testid="stTextInput"] > div > div {
        background: #07101E !important;
        border: 1px solid #1E2E4A !important;
        border-radius: 6px !important;
        color: #EDF0F8 !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    [data-testid="stTextInput"] > div > div:focus-within {
        border-color: #D4AF37 !important;
        box-shadow: 0 0 0 2px rgba(212,175,55,0.15) !important;
    }
    [data-testid="stTextInput"] label {
        color: #7A90B8 !important;
        font-size: 0.78em !important;
        font-weight: 600 !important;
        letter-spacing: 0.07em !important;
        text-transform: uppercase !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    /* ── Button ── */
    [data-testid="stFormSubmitButton"] > button {
        background: #D4AF37 !important;
        color: #07101E !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.9em !important;
        letter-spacing: 0.04em !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.65em 0 !important;
        width: 100% !important;
        cursor: pointer !important;
        transition: opacity 0.15s !important;
        margin-top: 0.5em !important;
    }
    [data-testid="stFormSubmitButton"] > button:hover {
        opacity: 0.88 !important;
    }

    /* ── Tabs ── */
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        background: transparent !important;
        border-bottom: 1px solid #1E2E4A !important;
        gap: 0 !important;
        margin-bottom: 1.4em !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab"] {
        background: transparent !important;
        color: #3A4E70 !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.82em !important;
        font-weight: 600 !important;
        padding: 0.5em 1.2em !important;
        border-bottom: 2px solid transparent !important;
    }
    [data-testid="stTabs"] [aria-selected="true"] {
        color: #D4AF37 !important;
        border-bottom-color: #D4AF37 !important;
    }

    /* ── Messages ── */
    .auth-ok  { color: #1DB954; font-size: 0.82em; padding: .45em 0; font-family: 'DM Sans', sans-serif; }
    .auth-err { color: #FF3040; font-size: 0.82em; padding: .45em 0; font-family: 'DM Sans', sans-serif; }

    /* ── Footer note ── */
    .login-footer {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.72em;
        color: #1E2E4A;
        text-align: center;
        margin-top: 1.6em;
    }
    </style>
    """, unsafe_allow_html=True)

    # Card wrapper
    st.markdown('<div class="login-card">', unsafe_allow_html=True)
    st.markdown('<div class="login-logo">FAWP Scanner</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="login-sub">Information-Control Exclusion Principle</div>',
        unsafe_allow_html=True,
    )

    tab_in, tab_up, tab_reset = st.tabs(["Sign in", "Create account", "Reset password"])

    # ── Sign in ───────────────────────────────────────────────────────────────
    with tab_in:
        with st.form("signin_form", clear_on_submit=False):
            email_in = st.text_input("Email",    placeholder="you@example.com",
                                     label_visibility="visible")
            pass_in  = st.text_input("Password", placeholder="••••••••",
                                     type="password", label_visibility="visible")
            sub_in   = st.form_submit_button("Sign in →",
                                             use_container_width=True,
                                             type="primary")
        if sub_in:
            if not email_in or not pass_in:
                st.markdown('<div class="auth-err">Email and password required.</div>',
                            unsafe_allow_html=True)
            else:
                with st.spinner(""):
                    ok, msg = sign_in(email_in.strip(), pass_in)
                if ok:
                    st.rerun()
                else:
                    st.markdown(f'<div class="auth-err">{msg}</div>',
                                unsafe_allow_html=True)

    # ── Sign up ───────────────────────────────────────────────────────────────
    with tab_up:
        with st.form("signup_form", clear_on_submit=True):
            email_up = st.text_input("Email",            placeholder="you@example.com")
            pass_up  = st.text_input("Password",         placeholder="Min 6 characters",
                                     type="password")
            pass_up2 = st.text_input("Confirm password", placeholder="Repeat password",
                                     type="password")
            sub_up   = st.form_submit_button("Create account →",
                                             use_container_width=True,
                                             type="primary")
        if sub_up:
            if not email_up or not pass_up:
                st.markdown('<div class="auth-err">All fields required.</div>',
                            unsafe_allow_html=True)
            elif pass_up != pass_up2:
                st.markdown('<div class="auth-err">Passwords do not match.</div>',
                            unsafe_allow_html=True)
            elif len(pass_up) < 6:
                st.markdown('<div class="auth-err">Password must be 6+ characters.</div>',
                            unsafe_allow_html=True)
            else:
                with st.spinner(""):
                    ok2, msg2 = sign_up(email_up.strip(), pass_up)
                cls = "auth-ok" if ok2 else "auth-err"
                st.markdown(f'<div class="{cls}">{msg2}</div>',
                            unsafe_allow_html=True)
                if ok2 and "signed in" in msg2:
                    st.rerun()

    # ── Reset password ────────────────────────────────────────────────────────
    with tab_reset:
        with st.form("reset_form", clear_on_submit=True):
            email_r = st.text_input("Email", placeholder="you@example.com")
            sub_r   = st.form_submit_button("Send reset email →",
                                            use_container_width=True)
        if sub_r:
            if not email_r:
                st.markdown('<div class="auth-err">Enter your email.</div>',
                            unsafe_allow_html=True)
            else:
                with st.spinner(""):
                    ok3, msg3 = reset_password(email_r.strip())
                cls = "auth-ok" if ok3 else "auth-err"
                st.markdown(f'<div class="{cls}">{msg3}</div>',
                            unsafe_allow_html=True)

    st.markdown(
        '<div class="login-footer">fawp-scanner.info · '
        'doi:10.5281/zenodo.18673949</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        '<div style="text-align:center;margin:1.2em 0 .4em">'
        '<span style="color:#3A4E70;font-size:.8em">or</span></div>',
        unsafe_allow_html=True)
    if st.button(
        '▶  Try demo — no account needed',
        use_container_width=True,
        key='demo_bypass_btn',
    ):
        st.session_state['_demo_bypass'] = True
        st.session_state['_demo_mode']   = True
        st.rerun()
    st.markdown(
        '<div style="text-align:center;padding:.5em 0 1em">'
        '<span style="color:#3A4E70;font-size:.75em">'
        'Demo uses synthetic data · Sign up for real scans</span></div>',
        unsafe_allow_html=True)
    st.stop()
