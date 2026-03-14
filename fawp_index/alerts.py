"""
fawp_index.alerts — FAWP alert engine
======================================

Fire alerts when FAWP regimes change, leverage gaps cross thresholds,
or agency horizons collapse.  Supports terminal, Telegram, Discord,
email, and generic webhooks.

Quick start
-----------
::

    from fawp_index.watchlist import scan_watchlist
    from fawp_index.alerts import AlertEngine

    engine = AlertEngine(gap_threshold=0.05)
    engine.add_terminal()
    engine.add_telegram(token="BOT_TOKEN", chat_id="CHAT_ID")
    engine.add_discord(webhook_url="https://discord.com/api/webhooks/...")

    result = scan_watchlist({"SPY": spy_df, "QQQ": qqq_df})
    alerts = engine.check(result)

    for a in alerts:
        print(a.message)

State-aware (new vs repeat alerts)
------------------------------------
The engine tracks which regimes were active in the previous check.
Set ``state_path`` to persist state across runs::

    engine = AlertEngine(state_path="fawp_state.json")
    engine.add_terminal()
    alerts = engine.check(result)     # fires NEW_FAWP / REGIME_END diffs
    engine.check(result)              # no duplicate alerts if nothing changed

Daily summary
-------------
::

    engine.daily_summary(result)      # prints / sends condensed digest

Alert types
-----------
- ``NEW_FAWP``          — ticker just entered FAWP regime
- ``REGIME_END``        — ticker just left FAWP regime
- ``GAP_THRESHOLD``     — leverage gap crossed ``gap_threshold`` bits
- ``HORIZON_COLLAPSE``  — agency horizon < ``horizon_warn_tau``
- ``DAILY_SUMMARY``     — once-per-run digest of all active regimes

Ralph Clayton (2026) · https://doi.org/10.5281/zenodo.18673949
"""

from __future__ import annotations

import json
import smtplib
import urllib.request
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime
from email.mime.text import MIMEText
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from fawp_index import __version__ as _VERSION


# ─────────────────────────────────────────────────────────────────────────────
# Alert types and dataclass
# ─────────────────────────────────────────────────────────────────────────────

class AlertType(str, Enum):
    NEW_FAWP          = "NEW_FAWP"
    REGIME_END        = "REGIME_END"
    GAP_THRESHOLD     = "GAP_THRESHOLD"
    HORIZON_COLLAPSE  = "HORIZON_COLLAPSE"
    DAILY_SUMMARY     = "DAILY_SUMMARY"


class AlertSeverity(str, Enum):
    """Severity tier for a FAWP alert, based on regime score."""
    LOW      = "LOW"       # score 0.05–0.25
    MEDIUM   = "MEDIUM"    # score 0.25–0.50
    HIGH     = "HIGH"      # score 0.50–0.75
    CRITICAL = "CRITICAL"  # score > 0.75


def _score_to_severity(score: float) -> AlertSeverity:
    if score >= 0.75:
        return AlertSeverity.CRITICAL
    if score >= 0.50:
        return AlertSeverity.HIGH
    if score >= 0.25:
        return AlertSeverity.MEDIUM
    return AlertSeverity.LOW


@dataclass
class FAWPAlert:
    """
    A single FAWP alert.

    Attributes
    ----------
    ticker : str
    timeframe : str
    alert_type : AlertType
    severity : AlertSeverity
        LOW / MEDIUM / HIGH / CRITICAL based on regime score.
    score : float
    gap_bits : float
    odw_start : int or None
    odw_end : int or None
    timestamp : datetime
    message : str
        Human-readable message, ready to send to any channel.
    """
    ticker:     str
    timeframe:  str
    alert_type: AlertType
    score:      float
    gap_bits:   float
    odw_start:  Optional[int]
    odw_end:    Optional[int]
    timestamp:  datetime
    message:    str
    severity:   AlertSeverity = AlertSeverity.LOW

    def to_dict(self) -> dict:
        return {
            "ticker":     self.ticker,
            "timeframe":  self.timeframe,
            "alert_type": self.alert_type.value,
            "score":      round(self.score, 6),
            "gap_bits":   round(self.gap_bits, 6),
            "odw_start":  self.odw_start,
            "odw_end":    self.odw_end,
            "timestamp":  self.timestamp.isoformat(),
            "severity":   self.severity.value,
            "message":    self.message,
        }


def _fmt_alert(ticker, tf, alert_type, score, gap, odw_start, odw_end) -> str:
    """Build the human-readable alert message."""
    odw = f"τ {odw_start}–{odw_end}" if odw_start is not None else "—"
    prefix = {
        AlertType.NEW_FAWP:         "🔴 NEW FAWP",
        AlertType.REGIME_END:       "🟢 REGIME END",
        AlertType.GAP_THRESHOLD:    "⚡ GAP ALERT",
        AlertType.HORIZON_COLLAPSE: "⚠️  HORIZON",
        AlertType.DAILY_SUMMARY:    "📋 SUMMARY",
    }[alert_type]
    return (
        f"{prefix} | {ticker} [{tf}] | "
        f"score={score:.4f} gap={gap:.4f}b | ODW {odw} | "
        f"fawp-index v{_VERSION}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Alert backends (pluggable)
# ─────────────────────────────────────────────────────────────────────────────

class _TerminalBackend:
    name = "terminal"
    def send(self, alert: FAWPAlert):
        ts = alert.timestamp.strftime("%H:%M:%S")
        print(f"[FAWP ALERT {ts}] {alert.message}")


class _TelegramBackend:
    name = "telegram"
    def __init__(self, token: str, chat_id: str):
        self.token   = token
        self.chat_id = str(chat_id)

    def send(self, alert: FAWPAlert):
        text = urllib.parse.quote(alert.message)
        url  = (f"https://api.telegram.org/bot{self.token}"
                f"/sendMessage?chat_id={self.chat_id}&text={text}")
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                if r.status not in (200, 201):
                    raise RuntimeError(f"Telegram HTTP {r.status}")
        except Exception as e:
            raise RuntimeError(f"Telegram send failed: {e}") from e


class _DiscordBackend:
    name = "discord"
    def __init__(self, webhook_url: str):
        self.url = webhook_url

    def send(self, alert: FAWPAlert):
        payload = json.dumps({"content": alert.message}).encode()
        req = urllib.request.Request(
            self.url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                if r.status not in (200, 204):
                    raise RuntimeError(f"Discord HTTP {r.status}")
        except Exception as e:
            raise RuntimeError(f"Discord send failed: {e}") from e


class _EmailBackend:
    """
    SMTP email backend.

    Parameters
    ----------
    smtp_host : str
    smtp_port : int
        Default 587 (STARTTLS). Use 465 for SSL.
    username : str
    password : str
    from_addr : str
    to_addrs : list of str
    use_tls : bool
        Default True (STARTTLS). Set False for SSL on port 465.
    """
    name = "email"
    def __init__(self, smtp_host, smtp_port=587, username="", password="",
                 from_addr="", to_addrs=None, use_tls=True):
        self.host      = smtp_host
        self.port      = smtp_port
        self.username  = username
        self.password  = password
        self.from_addr = from_addr or username
        self.to_addrs  = to_addrs or [username]
        self.use_tls   = use_tls

    def send(self, alert: FAWPAlert):
        msg = MIMEText(alert.message)
        msg["Subject"] = f"[FAWP] {alert.alert_type.value} — {alert.ticker}"
        msg["From"]    = self.from_addr
        msg["To"]      = ", ".join(self.to_addrs)
        try:
            smtp_cls = smtplib.SMTP if self.use_tls else smtplib.SMTP_SSL
            with smtp_cls(self.host, self.port, timeout=15) as s:
                if self.use_tls:
                    s.starttls()
                if self.username:
                    s.login(self.username, self.password)
                s.sendmail(self.from_addr, self.to_addrs, msg.as_string())
        except Exception as e:
            raise RuntimeError(f"Email send failed: {e}") from e


class _WebhookBackend:
    """Generic JSON POST webhook (e.g. Slack incoming webhooks, custom APIs)."""
    name = "webhook"
    def __init__(self, url: str, headers: Optional[Dict[str, str]] = None):
        self.url     = url
        self.headers = headers or {"Content-Type": "application/json"}

    def send(self, alert: FAWPAlert):
        payload = json.dumps(alert.to_dict()).encode()
        req = urllib.request.Request(
            self.url, data=payload, headers=self.headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                if r.status not in (200, 201, 204):
                    raise RuntimeError(f"Webhook HTTP {r.status}")
        except Exception as e:
            raise RuntimeError(f"Webhook send failed: {e}") from e


class _CallbackBackend:
    """Call a Python function with the alert. Useful for custom integrations."""
    name = "callback"
    def __init__(self, fn: Callable[[FAWPAlert], None]):
        self.fn = fn

    def send(self, alert: FAWPAlert):
        self.fn(alert)


# ─────────────────────────────────────────────────────────────────────────────
# Alert engine
# ─────────────────────────────────────────────────────────────────────────────

class AlertEngine:
    """
    Multi-channel FAWP alert engine.

    Compares a WatchlistResult against the previous known state and fires
    alerts for transitions, threshold crossings, and daily summaries.

    Parameters
    ----------
    gap_threshold : float
        Fire ``GAP_THRESHOLD`` when peak_gap_bits crosses this value. Default 0.05.
    horizon_warn_tau : int or None
        Fire ``HORIZON_COLLAPSE`` when tau_h_plus < this value. Default None (off).
    state_path : str or Path, optional
        JSON file to persist previous state across runs. Enables NEW_FAWP /
        REGIME_END diff detection. If None, all active regimes are treated as new.
    suppress_errors : bool
        If True, backend send errors are caught and printed rather than raised.
        Default True.

    Examples
    --------
    ::

        engine = AlertEngine(gap_threshold=0.05)
        engine.add_terminal()
        engine.add_telegram(token="...", chat_id="...")
        engine.add_discord(webhook_url="https://discord.com/api/webhooks/...")
        engine.add_email(
            smtp_host="smtp.gmail.com",
            username="you@gmail.com",
            password="app_password",
            to_addrs=["you@gmail.com"],
        )
        engine.add_webhook("https://hooks.slack.com/services/...")

        alerts = engine.check(watchlist_result)
        engine.daily_summary(watchlist_result)
    """

    def __init__(
        self,
        gap_threshold:           float = 0.05,
        horizon_warn_tau:        Optional[int] = None,
        state_path:              Optional[Union[str, Path]] = None,
        suppress_errors:         bool = True,
        # ── New in v0.11.0 ───────────────────────────────────────────────
        cooldown_hours:          float = 0.0,
        min_consecutive_windows: int   = 1,
        score_change_threshold:  float = 0.0,
        min_severity:            Optional[AlertSeverity] = None,
    ):
        """
        Parameters
        ----------
        gap_threshold : float
            Fire GAP_THRESHOLD when peak_gap_bits crosses this value.
        horizon_warn_tau : int or None
            Fire HORIZON_COLLAPSE when tau_h_plus < this value.
        state_path : str or Path, optional
            JSON file for persistent state across runs.
        suppress_errors : bool
            Catch and print backend send errors rather than raising.
        cooldown_hours : float
            Suppress repeat alerts for the same (ticker, timeframe) within
            this many hours after the last alert. 0 = no cooldown.
        min_consecutive_windows : int
            Only fire NEW_FAWP after this many consecutive flagged windows.
            1 = fire immediately (default, original behaviour).
        score_change_threshold : float
            Only fire GAP_THRESHOLD if the score has changed by at least
            this amount since the last check. 0 = always fire (default).
        min_severity : AlertSeverity or None
            Suppress any alert below this severity tier.
            None = no suppression (default).
        """
        self.gap_threshold            = gap_threshold
        self.horizon_warn_tau         = horizon_warn_tau
        self.state_path               = Path(state_path) if state_path else None
        self.suppress_errors          = suppress_errors
        self.cooldown_hours           = cooldown_hours
        self.min_consecutive_windows  = max(1, int(min_consecutive_windows))
        self.score_change_threshold   = score_change_threshold
        self.min_severity             = min_severity
        self._backends: list          = []
        self._prev_state: Dict[str, dict] = self._load_state()

    # ── Backend registration ─────────────────────────────────────────────────

    def add_terminal(self) -> "AlertEngine":
        """Print alerts to the terminal. Always useful."""
        self._backends.append(_TerminalBackend())
        return self

    def add_telegram(self, token: str, chat_id: str) -> "AlertEngine":
        """Send to a Telegram bot."""
        self._backends.append(_TelegramBackend(token, chat_id))
        return self

    def add_discord(self, webhook_url: str) -> "AlertEngine":
        """Post to a Discord channel via webhook."""
        self._backends.append(_DiscordBackend(webhook_url))
        return self

    def add_email(
        self,
        smtp_host: str,
        smtp_port: int = 587,
        username:  str = "",
        password:  str = "",
        from_addr: str = "",
        to_addrs:  Optional[List[str]] = None,
        use_tls:   bool = True,
    ) -> "AlertEngine":
        """Send via SMTP email."""
        self._backends.append(_EmailBackend(
            smtp_host, smtp_port, username, password, from_addr, to_addrs, use_tls
        ))
        return self

    def add_webhook(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
    ) -> "AlertEngine":
        """POST JSON payload to any webhook URL (Slack, custom, etc.)."""
        self._backends.append(_WebhookBackend(url, headers))
        return self

    def add_callback(self, fn: Callable[[FAWPAlert], None]) -> "AlertEngine":
        """Call a Python function with each alert."""
        self._backends.append(_CallbackBackend(fn))
        return self

    # ── Core check ───────────────────────────────────────────────────────────

    def check(self, watchlist_result) -> List[FAWPAlert]:
        """
        Compare a WatchlistResult against the previous state and fire alerts.

        Applies cooldown, consecutive-window, score-change, and severity filters.

        Parameters
        ----------
        watchlist_result : WatchlistResult

        Returns
        -------
        list of FAWPAlert — all alerts that were generated and sent.
        """
        alerts: List[FAWPAlert] = []
        now    = datetime.now()

        for asset in watchlist_result.assets:
            if asset.error:
                continue

            key      = f"{asset.ticker}|{asset.timeframe}"
            prev     = self._prev_state.get(key, {})
            was_active    = prev.get("active", False)
            prev_score    = prev.get("score", 0.0)
            last_alert_ts = prev.get("last_alert_ts")

            # ── Cooldown check ────────────────────────────────────────────
            in_cooldown = False
            if self.cooldown_hours > 0 and last_alert_ts:
                try:
                    last_dt  = datetime.fromisoformat(last_alert_ts)
                    elapsed  = (now - last_dt).total_seconds() / 3600
                    in_cooldown = elapsed < self.cooldown_hours
                except Exception:
                    pass

            # ── Consecutive-window check ──────────────────────────────────
            meets_consecutive = True
            if self.min_consecutive_windows > 1 and asset.scan is not None:
                recent = asset.scan.windows[-self.min_consecutive_windows:]
                meets_consecutive = (
                    len(recent) >= self.min_consecutive_windows
                    and all(w.fawp_found for w in recent)
                )

            # ── NEW_FAWP ──────────────────────────────────────────────────
            if (asset.regime_active and not was_active
                    and meets_consecutive and not in_cooldown):
                a = self._make_alert(asset, AlertType.NEW_FAWP, now)
                if self._passes_severity(a):
                    alerts.append(a)

            # ── REGIME_END ────────────────────────────────────────────────
            elif not asset.regime_active and was_active and not in_cooldown:
                a = self._make_alert(asset, AlertType.REGIME_END, now)
                if self._passes_severity(a):
                    alerts.append(a)

            # ── GAP_THRESHOLD ─────────────────────────────────────────────
            score_delta = abs(asset.latest_score - prev_score)
            if (asset.regime_active
                    and asset.peak_gap_bits >= self.gap_threshold
                    and score_delta >= self.score_change_threshold
                    and not in_cooldown):
                a = self._make_alert(asset, AlertType.GAP_THRESHOLD, now)
                if self._passes_severity(a):
                    alerts.append(a)

            # ── HORIZON_COLLAPSE ──────────────────────────────────────────
            if (self.horizon_warn_tau is not None
                    and asset.scan is not None
                    and not in_cooldown):
                try:
                    tau_h = asset.scan.latest.odw_result.tau_h_plus
                    if tau_h is not None and tau_h < self.horizon_warn_tau:
                        a = self._make_alert(asset, AlertType.HORIZON_COLLAPSE, now)
                        if self._passes_severity(a):
                            alerts.append(a)
                except Exception:
                    pass

        # Send all queued alerts
        last_alert_time = now.isoformat() if alerts else None
        for alert in alerts:
            self._dispatch(alert)

        # Update state
        new_state: Dict[str, dict] = {}
        for asset in watchlist_result.assets:
            if asset.error:
                continue
            key = f"{asset.ticker}|{asset.timeframe}"
            prev = self._prev_state.get(key, {})
            fired_now = any(
                a.ticker == asset.ticker and a.timeframe == asset.timeframe
                for a in alerts
            )
            new_state[key] = {
                "active":       asset.regime_active,
                "score":        float(asset.latest_score),
                "last_alert_ts": now.isoformat() if fired_now else prev.get("last_alert_ts"),
            }
        self._prev_state = new_state
        self._save_state()
        return alerts

    # ── Severity filter ───────────────────────────────────────────────────────

    def _passes_severity(self, alert: FAWPAlert) -> bool:
        if self.min_severity is None:
            return True
        order = [AlertSeverity.LOW, AlertSeverity.MEDIUM,
                 AlertSeverity.HIGH, AlertSeverity.CRITICAL]
        return order.index(alert.severity) >= order.index(self.min_severity)

    def daily_summary(self, watchlist_result) -> FAWPAlert:
        """
        Build and send a daily digest of all active regimes.

        Returns the summary FAWPAlert.
        """
        now    = datetime.now()
        active = [a for a in watchlist_result.assets if a.regime_active and not a.error]
        total  = len([a for a in watchlist_result.assets if not a.error])

        if active:
            top = sorted(active, key=lambda a: a.latest_score, reverse=True)[:5]
            top_str = "  |  ".join(
                f"{a.ticker}[{a.timeframe}] score={a.latest_score:.3f}" for a in top
            )
            msg = (
                f"📋 DAILY SUMMARY | {len(active)}/{total} assets in FAWP | "
                f"Top: {top_str} | fawp-index v{_VERSION}"
            )
        else:
            msg = (
                f"📋 DAILY SUMMARY | 0/{total} assets in FAWP — "
                f"all clear | fawp-index v{_VERSION}"
            )

        # Use the top asset or a dummy for the dataclass
        ref  = active[0] if active else watchlist_result.assets[0]
        alert = FAWPAlert(
            ticker     = "WATCHLIST",
            timeframe  = "all",
            alert_type = AlertType.DAILY_SUMMARY,
            score      = float(ref.latest_score) if active else 0.0,
            gap_bits   = float(ref.peak_gap_bits) if active else 0.0,
            odw_start  = None,
            odw_end    = None,
            timestamp  = now,
            message    = msg,
        )
        self._dispatch(alert)
        return alert

    # ── Internals ────────────────────────────────────────────────────────────

    def _make_alert(self, asset, alert_type: AlertType, now: datetime) -> FAWPAlert:
        msg = _fmt_alert(
            asset.ticker, asset.timeframe, alert_type,
            asset.latest_score, asset.peak_gap_bits,
            asset.peak_odw_start, asset.peak_odw_end,
        )
        return FAWPAlert(
            ticker     = asset.ticker,
            timeframe  = asset.timeframe,
            alert_type = alert_type,
            score      = float(asset.latest_score),
            gap_bits   = float(asset.peak_gap_bits),
            odw_start  = asset.peak_odw_start,
            odw_end    = asset.peak_odw_end,
            timestamp  = now,
            message    = msg,
            severity   = _score_to_severity(float(asset.latest_score)),
        )

    def _dispatch(self, alert: FAWPAlert):
        for backend in self._backends:
            try:
                backend.send(alert)
            except Exception as e:
                if self.suppress_errors:
                    print(f"[FAWP alert] {backend.name} send error: {e}")
                else:
                    raise

    def _load_state(self) -> Dict[str, dict]:
        if self.state_path and self.state_path.exists():
            try:
                raw = json.loads(self.state_path.read_text())
                # Migrate old format (Dict[str, bool]) → new format (Dict[str, dict])
                migrated: Dict[str, dict] = {}
                for k, v in raw.items():
                    if isinstance(v, bool):
                        migrated[k] = {"active": v, "score": 0.0, "last_alert_ts": None}
                    elif isinstance(v, dict):
                        migrated[k] = v
                return migrated
            except Exception:
                pass
        return {}

    def _save_state(self):
        if self.state_path:
            try:
                self.state_path.write_text(json.dumps(self._prev_state, indent=2))
            except Exception as e:
                print(f"[FAWP alert] Could not save state: {e}")


    @property
    def backends(self) -> List[str]:
        """Names of registered backends."""
        return [b.name for b in self._backends]
