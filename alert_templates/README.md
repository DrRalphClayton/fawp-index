# Alert Templates

Drop-in message templates for `AlertEngine.set_template()`.

Each file is a Python snippet that configures a full template set
for a specific use-case. Copy the lines you need into your alert setup.

## Available presets

| File | Style | Best for |
|------|-------|---------|
| `trading_desk.py` | Concise · emoji-heavy | Terminal / Telegram / Discord |
| `research.py`     | Verbose · math notation | Email / Slack research channels |
| `minimal.py`      | Plain text · no emoji | Webhook / custom downstream |

## Usage

```python
from fawp_index.alerts import AlertEngine
from fawp_index.alert_template_presets import TRADING_DESK, RESEARCH, MINIMAL

engine = AlertEngine(gap_threshold=0.05, state_path="state.json")
engine.add_terminal()
engine.add_slack("https://hooks.slack.com/services/...")

# Apply a preset
for alert_type, template in TRADING_DESK.items():
    engine.set_template(alert_type, template)
```

## Template fields

| Field | Example |
|-------|---------|
| `{ticker}` | SPY |
| `{timeframe}` | 1d |
| `{score}` | 0.0142 |
| `{gap}` | 0.0181 |
| `{odw}` | τ 1–12 |
| `{severity}` | HIGH |
| `{alert_type}` | NEW_FAWP |
| `{timestamp}` | 2026-03-14 09:31 |
| `{version}` | 2.5.0 |
