"""
dashboard/cron_scan.py — Cron job for scheduled scans.

Runs on a schedule (e.g. daily at 9am UTC) via Render Cron Jobs.
Loads all user schedules from Supabase, runs each watchlist scan,
saves results to scan history, and sends email alerts if FAWP fires.

Add to Render:
  Command  : python /app/dashboard/cron_scan.py
  Schedule : 0 9 * * 1-5   (9am UTC Mon-Fri)

Requires environment variables:
  SUPABASE_URL
  SUPABASE_ANON_KEY (or SUPABASE_KEY)
  SUPABASE_SERVICE_ROLE_KEY

Ralph Clayton (2026) · doi:10.5281/zenodo.18673949
"""

import json
import os
import sys
import traceback
from datetime import datetime

# Add dashboard dir to path for supabase_store
sys.path.insert(0, os.path.dirname(__file__))


def _client(service: bool = False):
    from supabase import create_client
    url = os.environ.get("SUPABASE_URL", "")
    if service:
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    else:
        key = (os.environ.get("SUPABASE_ANON_KEY", "")
               or os.environ.get("SUPABASE_KEY", ""))
    return create_client(url, key)


def run_scheduled_scans():
    print(f"[cron] Starting scheduled scan run — {datetime.now().isoformat()}")

    admin = _client(service=True)

    # Load all active schedules
    try:
        res = (admin.table("fawp_schedules")
               .select("*")
               .eq("active", True)
               .execute())
        schedules = res.data or []
    except Exception as e:
        print(f"[cron] Failed to load schedules: {e}")
        return

    print(f"[cron] Found {len(schedules)} active schedules")

    from fawp_index.watchlist import scan_watchlist

    for sched in schedules:
        user_id   = sched.get("user_id")
        wl_name   = sched.get("watchlist", "")
        do_email  = sched.get("email_alerts", True)

        print(f"[cron] Scanning '{wl_name}' for user {user_id[:8]}…")

        try:
            # Load watchlist config from Supabase
            wl_res = (admin.table("fawp_watchlists")
                      .select("*")
                      .eq("user_id", user_id)
                      .eq("name", wl_name)
                      .execute())
            if not wl_res.data:
                print(f"[cron]   Watchlist '{wl_name}' not found for user")
                continue

            wl_cfg  = json.loads(wl_res.data[0].get("config", "{}"))
            tickers = wl_cfg.get("tickers", [])
            period  = wl_cfg.get("period", "2y")
            tfs     = wl_cfg.get("timeframes", ["1d"])

            if not tickers:
                continue

            # Run scan
            result = scan_watchlist(tickers, period=period, timeframes=tfs,
                                    n_null=0, verbose=False)

            # Save to history
            assets_payload = [
                {
                    "ticker":         a.ticker,
                    "timeframe":      a.timeframe,
                    "latest_score":   round(float(a.latest_score),  6),
                    "peak_gap_bits":  round(float(a.peak_gap_bits), 6),
                    "regime_active":  bool(a.regime_active),
                    "days_in_regime": int(a.days_in_regime),
                    "signal_age_days":int(a.signal_age_days),
                    "odw_start":      a.peak_odw_start,
                    "odw_end":        a.peak_odw_end,
                }
                for a in result.assets if not a.error
            ]
            admin.table("fawp_scan_history").insert({
                "user_id":    user_id,
                "scanned_at": datetime.now().isoformat(),
                "label":      f"scheduled:{wl_name}",
                "n_assets":   result.n_assets,
                "n_flagged":  result.n_flagged,
                "payload":    json.dumps(assets_payload),
            }).execute()

            # Update last_run
            admin.table("fawp_schedules").update(
                {"last_run": datetime.now().isoformat()}
            ).eq("user_id", user_id).eq("watchlist", wl_name).execute()

            print(f"[cron]   {result.n_flagged}/{result.n_assets} flagged")

            # Email alert
            if do_email and result.n_flagged > 0:
                try:
                    # Get user email
                    user_res = admin.auth.admin.get_user_by_id(user_id)
                    email = getattr(getattr(user_res, "user", None), "email", None)
                    if email:
                        flagged = ", ".join(
                            f"{a.ticker}[{a.timeframe}]"
                            for a in result.active_regimes()[:5]
                        )
                        _send_email(admin, email,
                                    f"FAWP Alert — {result.n_flagged} regime(s) in {wl_name}",
                                    f"Scheduled scan detected FAWP in {wl_name}:\n\n"
                                    f"{flagged}\n\n"
                                    f"View at: https://fawp-scanner.info\n\n"
                                    f"fawp-index")
                        print(f"[cron]   Alert sent to {email}")
                except Exception as email_err:
                    print(f"[cron]   Email failed: {email_err}")

        except Exception as e:
            print(f"[cron]   ERROR on '{wl_name}': {e}")
            traceback.print_exc()

    print(f"[cron] Done — {datetime.now().isoformat()}")


def _send_email(admin_client, to: str, subject: str, body: str):
    """Send email via Supabase Edge Function."""
    import urllib.request
    url  = os.environ.get("SUPABASE_URL", "")
    key  = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    data = json.dumps({"to": to, "subject": subject, "body": body}).encode()
    req  = urllib.request.Request(
        f"{url}/functions/v1/send-alert-email",
        data=data,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15):
        pass


if __name__ == "__main__":
    run_scheduled_scans()
