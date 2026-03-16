"""
dashboard/email_digest.py — Weekly FAWP alert digest via Supabase email.

Sends a weekly summary of watchlist FAWP detections to signed-in users.
Uses Supabase's built-in email (SMTP must be configured in Supabase dashboard)
or Resend (set RESEND_API_KEY env var for production).

Usage:
    # In cron_scan.py or Supabase Edge Function:
    from email_digest import send_digest
    send_digest(user_id, user_email, finance_results, weather_results)
"""
from __future__ import annotations
import os
from typing import Optional
from datetime import datetime


_FAWP_URL = os.environ.get("FAWP_BASE_URL", "https://fawp-scanner.info")


def _build_digest_html(
    user_email:      str,
    finance_results: list,
    weather_results: list,
    scan_date:       str,
) -> str:
    """Build HTML email body for the weekly digest."""

    finance_rows = ""
    n_finance_flagged = 0
    for r in finance_results[:20]:
        fawp = getattr(r, "regime_active", False) or (isinstance(r, dict) and r.get("fawp_found"))
        if fawp:
            n_finance_flagged += 1
        ticker = getattr(r, "ticker", r.get("ticker", "?") if isinstance(r, dict) else "?")
        gap    = getattr(r, "peak_gap_bits", r.get("peak_gap_bits", 0) if isinstance(r, dict) else 0)
        status = "🔴 FAWP" if fawp else "✅ Clear"
        color  = "#E83030" if fawp else "#22C468"
        finance_rows += (
            f'<tr><td style="padding:.5em .8em;border-bottom:1px solid #1A2A42">{ticker}</td>'
            f'<td style="padding:.5em .8em;border-bottom:1px solid #1A2A42;color:{color};font-weight:700">{status}</td>'
            f'<td style="padding:.5em .8em;border-bottom:1px solid #1A2A42;font-family:monospace">{gap:.4f}</td></tr>'
        )

    weather_rows = ""
    n_weather_flagged = 0
    for r in weather_results[:20]:
        fawp = r.get("fawp_found", False) if isinstance(r, dict) else getattr(r, "fawp_found", False)
        if fawp:
            n_weather_flagged += 1
        loc  = r.get("location", "?") if isinstance(r, dict) else getattr(r, "location", "?")
        var  = r.get("variable", "?") if isinstance(r, dict) else getattr(r, "variable", "?")
        gap  = r.get("peak_gap_bits", 0) if isinstance(r, dict) else getattr(r, "peak_gap_bits", 0)
        status = "🔴 FAWP" if fawp else "✅ Clear"
        color  = "#E83030" if fawp else "#22C468"
        weather_rows += (
            f'<tr><td style="padding:.5em .8em;border-bottom:1px solid #1A2A42">{loc}</td>'
            f'<td style="padding:.5em .8em;border-bottom:1px solid #1A2A42">{var}</td>'
            f'<td style="padding:.5em .8em;border-bottom:1px solid #1A2A42;color:{color};font-weight:700">{status}</td>'
            f'<td style="padding:.5em .8em;border-bottom:1px solid #1A2A42;font-family:monospace">{gap:.4f}</td></tr>'
        )

    total_flagged = n_finance_flagged + n_weather_flagged
    headline = (
        f"🔴 {total_flagged} FAWP alert{'s' if total_flagged != 1 else ''} this week"
        if total_flagged > 0 else "✅ No FAWP alerts this week"
    )

    finance_section = ""
    if finance_results:
        finance_section = f"""
        <h2 style="font-family:sans-serif;font-size:1em;color:#F2C440;
                   text-transform:uppercase;letter-spacing:.08em;margin:1.5em 0 .5em">
          Finance Scanner
        </h2>
        <table style="width:100%;border-collapse:collapse;font-size:.88em">
          <tr style="color:#5070A0;font-size:.8em;text-transform:uppercase;letter-spacing:.06em">
            <th style="text-align:left;padding:.4em .8em">Ticker</th>
            <th style="text-align:left;padding:.4em .8em">Status</th>
            <th style="text-align:left;padding:.4em .8em">Gap (bits)</th>
          </tr>
          {finance_rows}
        </table>"""

    weather_section = ""
    if weather_results:
        weather_section = f"""
        <h2 style="font-family:sans-serif;font-size:1em;color:#F2C440;
                   text-transform:uppercase;letter-spacing:.08em;margin:1.5em 0 .5em">
          Weather Scanner
        </h2>
        <table style="width:100%;border-collapse:collapse;font-size:.88em">
          <tr style="color:#5070A0;font-size:.8em;text-transform:uppercase;letter-spacing:.06em">
            <th style="text-align:left;padding:.4em .8em">Location</th>
            <th style="text-align:left;padding:.4em .8em">Variable</th>
            <th style="text-align:left;padding:.4em .8em">Status</th>
            <th style="text-align:left;padding:.4em .8em">Gap (bits)</th>
          </tr>
          {weather_rows}
        </table>"""

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body style="background:#030810;color:#E8EDF8;font-family:'Helvetica Neue',Arial,sans-serif;
             max-width:600px;margin:0 auto;padding:2em 1.5em">

  <div style="border-bottom:1px solid #1A2A42;padding-bottom:1em;margin-bottom:1.5em">
    <div style="font-size:1.5em;font-weight:800;color:#F2C440;letter-spacing:-.02em">
      🔴 FAWP Scanner
    </div>
    <div style="color:#5070A0;font-size:.8em;margin-top:.3em">
      Weekly digest · {scan_date}
    </div>
  </div>

  <div style="background:#091220;border:1px solid #1A2A42;border-left:3px solid #F2C440;
              border-radius:10px;padding:1em 1.4em;margin-bottom:1.5em">
    <div style="font-size:1.15em;font-weight:700">{headline}</div>
    <div style="color:#5070A0;font-size:.82em;margin-top:.3em">
      Information-Control Exclusion Principle · doi:10.5281/zenodo.18673949
    </div>
  </div>

  {finance_section}
  {weather_section}

  <div style="margin-top:2em;padding-top:1em;border-top:1px solid #1A2A42;
              font-size:.75em;color:#283850;text-align:center">
    <a href="{_FAWP_URL}" style="color:#3888F8">Open FAWP Scanner</a> ·
    fawp-scanner.info · Ralph Clayton 2026
  </div>
</body>
</html>"""


def send_digest(
    user_email:      str,
    finance_results: list = None,
    weather_results: list = None,
    scan_date:       Optional[str] = None,
) -> bool:
    """
    Send the weekly digest email to a user.

    Tries Resend first (RESEND_API_KEY env var), falls back to Supabase SMTP.

    Parameters
    ----------
    user_email : str
    finance_results : list of AssetResult or dict
    weather_results : list of WeatherFAWPResult.to_dict()
    scan_date : str  e.g. "2026-03-16"

    Returns
    -------
    bool  True if sent successfully.
    """
    finance_results = finance_results or []
    weather_results = weather_results or []
    scan_date       = scan_date or datetime.utcnow().strftime("%Y-%m-%d")

    html = _build_digest_html(user_email, finance_results, weather_results, scan_date)
    subject = f"FAWP Weekly Digest — {scan_date}"

    # ── Try Resend first ──────────────────────────────────────────────────────
    resend_key = os.environ.get("RESEND_API_KEY", "")
    if resend_key:
        try:
            import urllib.request, json as _j
            payload = _j.dumps({
                "from":    "FAWP Scanner <alerts@fawp-scanner.info>",
                "to":      [user_email],
                "subject": subject,
                "html":    html,
            }).encode()
            req = urllib.request.Request(
                "https://api.resend.com/emails",
                data=payload,
                headers={
                    "Authorization": f"Bearer {resend_key}",
                    "Content-Type":  "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status == 200
        except Exception:
            pass

    # ── Fallback: Supabase SMTP ───────────────────────────────────────────────
    try:
        from supabase import create_client
        sb = create_client(
            os.environ["SUPABASE_URL"],
            os.environ.get("SUPABASE_SERVICE_ROLE_KEY", os.environ["SUPABASE_ANON_KEY"]),
        )
        sb.auth.admin.send_email(user_email, subject=subject, html=html)
        return True
    except Exception:
        return False


def schedule_digest_check():
    """
    Call from cron_scan.py — sends digests to users whose last digest
    was more than 7 days ago.
    """
    try:
        import os
        from supabase import create_client
        sb = create_client(
            os.environ["SUPABASE_URL"],
            os.environ["SUPABASE_SERVICE_ROLE_KEY"],
        )
        # Get users with enabled alerts and old or missing digest timestamp
        resp = sb.table("profiles") \
                 .select("user_id, last_digest_at, plan") \
                 .or_("last_digest_at.is.null,last_digest_at.lt." +
                      __import__("datetime").datetime.utcnow()
                      .replace(hour=0,minute=0,second=0)
                      .isoformat()) \
                 .execute()
        return resp.data or []
    except Exception:
        return []
