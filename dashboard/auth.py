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
            # Session expired — clear it so user sees login again
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
    """Return verified email for the current user, or None."""
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
    """Send a password-reset email via Supabase."""
    try:
        redirect = os.environ.get(
            "SUPABASE_RESET_REDIRECT", "http://localhost:8501"
        )
        _get_client().auth.reset_password_for_email(
            email, {"redirect_to": redirect}
        )
        return True, "Password reset email sent. Check your inbox."
    except Exception as e:
        return False, f"Reset failed: {e}"


# ── Login wall UI ─────────────────────────────────────────────────────────────

def require_auth():
    """
    Render the login wall if the user is not authenticated.
    Calls st.stop() so nothing below renders until signed in.
    """
    if is_authenticated():
        return

    st.markdown("""
    <style>
    .auth-wrap {
        max-width: 400px; margin: 6vh auto 0;
        background: #0D1729; border: 1px solid #182540;
        border-top: 3px solid #D4AF37; border-radius: 8px;
        padding: 2.4em 2.2em 2em;
    }
    .auth-logo { font-family: 'Syne', sans-serif; font-size: 1.6em;
        font-weight: 800; color: #D4AF37; margin-bottom: 0.1em; }
    .auth-sub  { font-size: 0.82em; color: #3A4E70; margin-bottom: 1.8em; }
    .auth-ok   { color: #1DB954; font-size: 0.85em; padding: .5em 0; }
    .auth-err  { color: #FF3040; font-size: 0.85em; padding: .5em 0; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="auth-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="auth-logo">FAWP Scanner</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="auth-sub">Information-Control Exclusion Principle detector</div>',
        unsafe_allow_html=True,
    )

    tab_in, tab_up, tab_reset = st.tabs(["Sign in", "Create account", "Reset password"])

    with tab_in:
        with st.form("signin_form", clear_on_submit=False):
            email_in = st.text_input("Email",    placeholder="you@example.com")
            pass_in  = st.text_input("Password", placeholder="••••••••", type="password")
            sub_in   = st.form_submit_button("Sign in", use_container_width=True, type="primary")
        if sub_in:
            if not email_in or not pass_in:
                st.markdown('<div class="auth-err">Email and password required.</div>',
                            unsafe_allow_html=True)
            else:
                with st.spinner("Signing in…"):
                    ok, msg = sign_in(email_in.strip(), pass_in)
                if ok:
                    st.rerun()
                else:
                    st.markdown(f'<div class="auth-err">{msg}</div>', unsafe_allow_html=True)

    with tab_up:
        with st.form("signup_form", clear_on_submit=True):
            email_up = st.text_input("Email",            placeholder="you@example.com")
            pass_up  = st.text_input("Password",         placeholder="Min 6 characters", type="password")
            pass_up2 = st.text_input("Confirm password", placeholder="Repeat password",  type="password")
            sub_up   = st.form_submit_button("Create account", use_container_width=True, type="primary")
        if sub_up:
            if not email_up or not pass_up:
                st.markdown('<div class="auth-err">All fields required.</div>', unsafe_allow_html=True)
            elif pass_up != pass_up2:
                st.markdown('<div class="auth-err">Passwords do not match.</div>', unsafe_allow_html=True)
            elif len(pass_up) < 6:
                st.markdown('<div class="auth-err">Password must be 6+ characters.</div>', unsafe_allow_html=True)
            else:
                with st.spinner("Creating account…"):
                    ok2, msg2 = sign_up(email_up.strip(), pass_up)
                cls = "auth-ok" if ok2 else "auth-err"
                st.markdown(f'<div class="{cls}">{msg2}</div>', unsafe_allow_html=True)
                if ok2 and "signed in" in msg2:
                    st.rerun()

    with tab_reset:
        with st.form("reset_form", clear_on_submit=True):
            email_r = st.text_input("Email", placeholder="you@example.com")
            sub_r   = st.form_submit_button("Send reset email", use_container_width=True)
        if sub_r:
            if not email_r:
                st.markdown('<div class="auth-err">Enter your email.</div>', unsafe_allow_html=True)
            else:
                with st.spinner("Sending…"):
                    ok3, msg3 = reset_password(email_r.strip())
                cls = "auth-ok" if ok3 else "auth-err"
                st.markdown(f'<div class="{cls}">{msg3}</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()
