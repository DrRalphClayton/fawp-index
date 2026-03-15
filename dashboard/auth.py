"""
dashboard/auth.py — Supabase email authentication for FAWP Scanner.

Wraps supabase-py to provide sign-in, sign-up, sign-out, and
session persistence via Streamlit session_state.

Required environment / Streamlit secrets:
    SUPABASE_URL  — your project URL  (https://xxxx.supabase.co)
    SUPABASE_KEY  — your anon/public key

Set them in dashboard/.streamlit/secrets.toml:
    SUPABASE_URL = "https://xxxx.supabase.co"
    SUPABASE_KEY = "eyJhbGci..."

Or as environment variables (for Render / Docker):
    SUPABASE_URL=https://xxxx.supabase.co
    SUPABASE_KEY=eyJhbGci...

Install:
    pip install supabase
"""

from __future__ import annotations

import os
from typing import Optional

import streamlit as st


# ── Supabase client (lazy singleton) ─────────────────────────────────────────

_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client

    try:
        from supabase import create_client
    except ImportError:
        st.error(
            "supabase-py is not installed.\n\n"
            "Run: `pip install supabase`"
        )
        st.stop()

    # Try Streamlit secrets first, then env vars
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
            "Supabase credentials not configured.\n\n"
            "Add to `.streamlit/secrets.toml`:\n"
            "```\nSUPABASE_URL = \"https://xxxx.supabase.co\"\n"
            "SUPABASE_KEY = \"eyJhbGci...\"\n```\n\n"
            "Or set `SUPABASE_URL` and `SUPABASE_KEY` environment variables."
        )
        st.stop()

    _client = create_client(url, key)
    return _client


# ── Session helpers ───────────────────────────────────────────────────────────

def get_session() -> Optional[dict]:
    """Return the current Supabase session dict, or None if not logged in."""
    return st.session_state.get("supabase_session")


def get_user() -> Optional[dict]:
    """Return the current user dict, or None if not logged in."""
    session = get_session()
    if session is None:
        return None
    return session.get("user")


def get_user_email() -> Optional[str]:
    """Return the logged-in user's email, or None."""
    user = get_user()
    if user is None:
        return None
    return user.get("email")


def is_authenticated() -> bool:
    """True if a valid session exists."""
    return get_session() is not None


def _store_session(session_obj):
    """Store a supabase session object in Streamlit session_state."""
    if session_obj is None:
        st.session_state.pop("supabase_session", None)
        return
    # Handle both dict and object responses from supabase-py v1/v2
    if hasattr(session_obj, "session"):
        session_obj = session_obj.session
    if hasattr(session_obj, "__dict__"):
        session_dict = {
            "access_token":  getattr(session_obj, "access_token",  None),
            "refresh_token": getattr(session_obj, "refresh_token", None),
            "user": {
                "id":    getattr(getattr(session_obj, "user", None), "id",    ""),
                "email": getattr(getattr(session_obj, "user", None), "email", ""),
            },
        }
    else:
        session_dict = dict(session_obj)
    st.session_state["supabase_session"] = session_dict


# ── Auth actions ──────────────────────────────────────────────────────────────

def sign_in(email: str, password: str) -> tuple[bool, str]:
    """
    Sign in with email + password.

    Returns
    -------
    (success, error_message)
    """
    try:
        client = _get_client()
        res = client.auth.sign_in_with_password({"email": email, "password": password})
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
    """
    Register a new account.

    Returns
    -------
    (success, message)
    """
    try:
        client = _get_client()
        res = client.auth.sign_up({"email": email, "password": password})
        user = getattr(res, "user", None)
        if user:
            return True, (
                "Account created! Check your email to confirm your address, "
                "then sign in."
            )
        return False, "Sign-up failed — no user returned."
    except Exception as e:
        err = str(e)
        if "already registered" in err or "User already registered" in err:
            return False, "This email is already registered. Try signing in."
        if "Password should be" in err:
            return False, "Password must be at least 6 characters."
        return False, f"Sign-up failed: {err}"


def sign_out():
    """Sign out and clear the session."""
    try:
        client = _get_client()
        client.auth.sign_out()
    except Exception:
        pass
    st.session_state.pop("supabase_session", None)
    st.rerun()


def reset_password(email: str) -> tuple[bool, str]:
    """Send a password-reset email."""
    try:
        client = _get_client()
        client.auth.reset_password_email(email)
        return True, "Password reset email sent. Check your inbox."
    except Exception as e:
        return False, f"Reset failed: {e}"


# ── Auth gate UI ──────────────────────────────────────────────────────────────

def require_auth():
    """
    Render the login / sign-up wall if the user is not authenticated.

    Call this at the top of app.py — it calls st.stop() if not logged in
    so nothing below it renders until the user signs in.

    Example
    -------
        from auth import require_auth, get_user_email, sign_out
        require_auth()                    # blocks here until signed in
        st.write(f"Welcome {get_user_email()}")
    """
    if is_authenticated():
        return   # already logged in — let the rest of app render

    # ── Login wall CSS ────────────────────────────────────────────────────
    st.markdown("""
    <style>
    .auth-wrap {
        max-width: 400px;
        margin: 6vh auto 0;
        background: #0D1729;
        border: 1px solid #182540;
        border-top: 3px solid #D4AF37;
        border-radius: 8px;
        padding: 2.4em 2.2em 2em;
    }
    .auth-logo {
        font-family: 'Syne', sans-serif;
        font-size: 1.6em;
        font-weight: 800;
        color: #D4AF37;
        margin-bottom: 0.1em;
    }
    .auth-sub {
        font-size: 0.82em;
        color: #3A4E70;
        margin-bottom: 1.8em;
    }
    .auth-msg-ok  { color: #1DB954; font-size: 0.85em; padding: .5em 0; }
    .auth-msg-err { color: #FF3040; font-size: 0.85em; padding: .5em 0; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="auth-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="auth-logo">FAWP Scanner</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="auth-sub">Information-Control Exclusion Principle detector</div>',
        unsafe_allow_html=True,
    )

    tab_in, tab_up, tab_reset = st.tabs(["Sign in", "Create account", "Reset password"])

    # ── Sign in ───────────────────────────────────────────────────────────
    with tab_in:
        with st.form("signin_form", clear_on_submit=False):
            email_in    = st.text_input("Email",    key="si_email",
                                        placeholder="you@example.com")
            password_in = st.text_input("Password", key="si_pass",
                                        type="password",
                                        placeholder="••••••••")
            submitted   = st.form_submit_button(
                "Sign in", use_container_width=True, type="primary"
            )

        if submitted:
            if not email_in or not password_in:
                st.markdown(
                    '<div class="auth-msg-err">Email and password are required.</div>',
                    unsafe_allow_html=True,
                )
            else:
                with st.spinner("Signing in…"):
                    ok, msg = sign_in(email_in.strip(), password_in)
                if ok:
                    st.rerun()
                else:
                    st.markdown(
                        f'<div class="auth-msg-err">{msg}</div>',
                        unsafe_allow_html=True,
                    )

    # ── Sign up ───────────────────────────────────────────────────────────
    with tab_up:
        with st.form("signup_form", clear_on_submit=True):
            email_up  = st.text_input("Email",           key="su_email",
                                      placeholder="you@example.com")
            pass_up   = st.text_input("Password",        key="su_pass",
                                      type="password",
                                      placeholder="Min 6 characters")
            pass_up2  = st.text_input("Confirm password",key="su_pass2",
                                      type="password",
                                      placeholder="Repeat password")
            submitted2 = st.form_submit_button(
                "Create account", use_container_width=True, type="primary"
            )

        if submitted2:
            if not email_up or not pass_up:
                st.markdown(
                    '<div class="auth-msg-err">All fields are required.</div>',
                    unsafe_allow_html=True,
                )
            elif pass_up != pass_up2:
                st.markdown(
                    '<div class="auth-msg-err">Passwords do not match.</div>',
                    unsafe_allow_html=True,
                )
            elif len(pass_up) < 6:
                st.markdown(
                    '<div class="auth-msg-err">Password must be at least 6 characters.</div>',
                    unsafe_allow_html=True,
                )
            else:
                with st.spinner("Creating account…"):
                    ok2, msg2 = sign_up(email_up.strip(), pass_up)
                cls = "auth-msg-ok" if ok2 else "auth-msg-err"
                st.markdown(
                    f'<div class="{cls}">{msg2}</div>',
                    unsafe_allow_html=True,
                )

    # ── Reset password ────────────────────────────────────────────────────
    with tab_reset:
        with st.form("reset_form", clear_on_submit=True):
            email_r  = st.text_input("Email", key="r_email",
                                     placeholder="you@example.com")
            submitted3 = st.form_submit_button(
                "Send reset email", use_container_width=True
            )

        if submitted3:
            if not email_r:
                st.markdown(
                    '<div class="auth-msg-err">Enter your email address.</div>',
                    unsafe_allow_html=True,
                )
            else:
                with st.spinner("Sending…"):
                    ok3, msg3 = reset_password(email_r.strip())
                cls = "auth-msg-ok" if ok3 else "auth-msg-err"
                st.markdown(
                    f'<div class="{cls}">{msg3}</div>',
                    unsafe_allow_html=True,
                )

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()   # nothing below renders until signed in
