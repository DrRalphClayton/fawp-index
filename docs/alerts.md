# fawp_index.alerts — Alert Engine

Fire alerts when FAWP regimes change, leverage gaps cross thresholds,
or agency horizons collapse. Supports Telegram, Discord, email, webhooks,
and terminal output.

## Quick start

```python
from fawp_index.watchlist import scan_watchlist
from fawp_index.alerts import AlertEngine

engine = AlertEngine(gap_threshold=0.05, state_path="fawp_state.json")
engine.add_terminal()
engine.add_telegram(token="BOT_TOKEN", chat_id="CHAT_ID")
engine.add_discord(webhook_url="https://discord.com/api/webhooks/...")
engine.add_email(
    smtp_host="smtp.gmail.com",
    username="you@gmail.com",
    password="app_password",
    to_addrs=["you@gmail.com"],
)
engine.add_webhook("https://hooks.slack.com/services/...")

result = scan_watchlist(dfs)
alerts = engine.check(result)      # fires only on state transitions

# Daily digest
engine.daily_summary(result)
```

## Alert types

| Type | Fires when |
|------|-----------|
| `NEW_FAWP` | Asset just entered FAWP regime (not seen in previous check) |
| `REGIME_END` | Asset just left FAWP regime |
| `GAP_THRESHOLD` | `peak_gap_bits >= gap_threshold` while regime active |
| `HORIZON_COLLAPSE` | `tau_h_plus < horizon_warn_tau` |
| `DAILY_SUMMARY` | On every `engine.daily_summary()` call |

## State awareness

The engine tracks which regimes were active in the **previous** check.
`NEW_FAWP` only fires once when a regime is first detected, not on every
subsequent scan. Set `state_path` to persist state across Python processes:

```python
engine = AlertEngine(state_path="fawp_state.json")
# State is loaded on init, saved after every check.
```

## Scheduling (run every day)

```python
import schedule, time
from fawp_index.watchlist import scan_watchlist

engine = AlertEngine(state_path="state.json")
engine.add_telegram(token="...", chat_id="...")

def daily_scan():
    result = scan_watchlist(["SPY", "QQQ", "BTC-USD"], period="1y")
    engine.check(result)
    engine.daily_summary(result)

schedule.every().day.at("07:00").do(daily_scan)
while True:
    schedule.run_pending()
    time.sleep(60)
```

## Error handling

```python
# suppress_errors=True (default): backend errors are printed, not raised
# suppress_errors=False: backend exceptions propagate
engine = AlertEngine(suppress_errors=False)
```

## Custom callback

```python
def my_handler(alert):
    print(f"Custom: {alert.ticker} {alert.alert_type} score={alert.score:.4f}")

engine.add_callback(my_handler)
```
