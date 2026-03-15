# Supabase Auth Setup

## 1. Create a Supabase project

1. Go to [supabase.com](https://supabase.com) → New project
2. Note your **Project URL** and **anon/public API key**
   (Settings → API → Project URL + anon key)

## 2. Enable Email auth

Supabase → Authentication → Providers → Email → Enable

Recommended settings:
- **Confirm email**: enabled (users must verify email)
- **Minimum password length**: 6

## 3. Add credentials

### Local development

Copy the example file and fill in your credentials:

```bash
cp dashboard/.streamlit/secrets.toml.example dashboard/.streamlit/secrets.toml
```

Edit `secrets.toml`:
```toml
SUPABASE_URL = "https://xxxxxxxxxxxxxxxxxxxx.supabase.co"
SUPABASE_KEY = "eyJhbGci..."
```

**Never commit `secrets.toml` to git** — it is in `.gitignore`.

### Render deployment

In your Render service → Environment → Add environment variables:

| Key | Value |
|-----|-------|
| `SUPABASE_URL` | `https://xxxx.supabase.co` |
| `SUPABASE_KEY` | `eyJhbGci...` |

### Streamlit Community Cloud

App → Settings → Secrets → paste:
```toml
SUPABASE_URL = "https://xxxx.supabase.co"
SUPABASE_KEY = "eyJhbGci..."
```

## 4. Install and run

```bash
pip install supabase
streamlit run dashboard/app.py
```

## 5. Test

1. Open the dashboard — you should see the login screen
2. Click "Create account", register with any email
3. Check your email for the confirmation link
4. Sign in — the full scanner loads

## Disable auth (development)

If `supabase-py` is not installed, auth is silently skipped and the
dashboard runs without a login wall. To explicitly disable auth,
don't install supabase:

```bash
pip install fawp-index[dashboard]   # no supabase
```

## User data isolation

Currently all users share the same scan history (`~/.fawp/history/`).
For per-user isolation, extend `ScanHistory` to use
`get_user_email()` as a subdirectory prefix.

---

## 6. Run database migrations

In Supabase → SQL Editor → New query, paste and run the contents of:

```
dashboard/supabase/migrations.sql
```

This creates three tables with Row Level Security:
- `fawp_scan_history` — per-user scan snapshots
- `fawp_watchlists` — per-user named watchlists
- `fawp_schedules` — scheduled scan configuration

## 7. Set up scheduled scans (optional)

In Render → your service → **Cron Jobs** → New Cron Job:

| Field | Value |
|-------|-------|
| Name | FAWP Daily Scan |
| Command | `python /app/dashboard/cron_scan.py` |
| Schedule | `0 9 * * 1-5` (9am UTC Mon-Fri) |

Requires `SUPABASE_SERVICE_ROLE_KEY` in environment variables.

## 8. Email alerts (optional)

Deploy a Supabase Edge Function named `send-alert-email`.
Template: https://supabase.com/docs/guides/functions

The function receives `{to, subject, body}` and sends via your configured SMTP.
