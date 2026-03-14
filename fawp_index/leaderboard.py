"""
fawp_index.leaderboard — Market-wide ranked leaderboard from a WatchlistResult.

Produces four ranked categories from any WatchlistResult:

  ``top_fawp``          — assets with an active FAWP regime, ranked by score
  ``rising_risk``       — assets with rapidly increasing regime score (momentum)
  ``collapsing_control`` — high pred MI / low steer MI ratio; not yet FAWP but close
  ``strongest_odw``     — widest Operational Detection Windows detected

Usage::

    from fawp_index.watchlist import scan_watchlist
    from fawp_index.leaderboard import Leaderboard

    result = scan_watchlist(["SPY", "QQQ", "GLD", "BTC-USD"], period="2y")
    lb = Leaderboard.from_watchlist(result)

    print(lb.summary())
    lb.to_html("leaderboard.html")
    lb.to_csv("leaderboard.csv")

Or directly from the CLI::

    fawp-scan --preset equities --leaderboard

Ralph Clayton (2026) · doi:10.5281/zenodo.18673949
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from fawp_index import __version__ as _VERSION


# ── Score-trend helper ────────────────────────────────────────────────────────

def _score_slope(asset) -> float:
    """Slope of regime score over the last 10 windows (OLS). Positive = rising."""
    if asset.scan is None or len(asset.scan.windows) < 3:
        return 0.0
    scores = np.array([w.regime_score for w in asset.scan.windows[-10:]], dtype=float)
    n = len(scores)
    x = np.arange(n, dtype=float)
    if n < 2:
        return 0.0
    slope = float(np.polyfit(x, scores, 1)[0])
    return slope


def _control_gap_ratio(asset) -> float:
    """
    pred_mi / (steer_mi + 1e-6) for the latest window.
    High ratio = prediction survives while control is near zero.
    """
    if asset.scan is None:
        return 0.0
    latest = asset.scan.latest
    pred_arr  = np.asarray(latest.pred_mi,  dtype=float)
    steer_arr = np.asarray(latest.steer_mi, dtype=float)
    if pred_arr.size == 0:
        return 0.0
    mean_pred  = float(np.mean(pred_arr))
    mean_steer = float(np.mean(steer_arr))
    return mean_pred / (mean_steer + 1e-6)


def _odw_width(asset) -> int:
    """Width of the peak ODW in tau steps (0 if no ODW)."""
    if asset.peak_odw_start is None or asset.peak_odw_end is None:
        return 0
    return max(0, int(asset.peak_odw_end) - int(asset.peak_odw_start) + 1)


# ── LeaderboardEntry ─────────────────────────────────────────────────────────

@dataclass
class LeaderboardEntry:
    """One row in a leaderboard category."""
    rank:      int
    ticker:    str
    timeframe: str
    score:     float
    gap_bits:  float
    odw_start: Optional[int]
    odw_end:   Optional[int]
    days_active:  int
    signal_age:   int
    detail:    str = ""          # category-specific metric string

    def to_dict(self) -> dict:
        return {
            "rank":       self.rank,
            "ticker":     self.ticker,
            "timeframe":  self.timeframe,
            "score":      round(self.score, 4),
            "gap_bits":   round(self.gap_bits, 4),
            "odw_start":  self.odw_start,
            "odw_end":    self.odw_end,
            "days_active": self.days_active,
            "signal_age":  self.signal_age,
            "detail":     self.detail,
        }


# ── Leaderboard ───────────────────────────────────────────────────────────────

@dataclass
class Leaderboard:
    """
    Four ranked leaderboard categories derived from a WatchlistResult.

    Attributes
    ----------
    top_fawp : list of LeaderboardEntry
        Assets with an active FAWP regime, ranked by regime score.
    rising_risk : list of LeaderboardEntry
        Assets with the fastest-increasing regime score (positive slope),
        whether or not they are currently flagged.
    collapsing_control : list of LeaderboardEntry
        Assets where prediction MI is high relative to steering MI,
        indicating an emerging or deepening control collapse.
    strongest_odw : list of LeaderboardEntry
        Assets with the widest Operational Detection Windows.
    generated_date : str
    n_assets : int
    n_flagged : int
    """
    top_fawp:           List[LeaderboardEntry]
    rising_risk:        List[LeaderboardEntry]
    collapsing_control: List[LeaderboardEntry]
    strongest_odw:      List[LeaderboardEntry]
    generated_date:     str = field(default_factory=lambda: date.today().isoformat())
    n_assets:           int = 0
    n_flagged:          int = 0

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def from_watchlist(
        cls,
        result,
        top_n: int = 10,
    ) -> "Leaderboard":
        """
        Build a Leaderboard from a WatchlistResult.

        Parameters
        ----------
        result : WatchlistResult
        top_n : int
            Maximum entries per category.
        """
        valid = [a for a in result.assets if not a.error]

        def _entry(rank, asset, detail="") -> LeaderboardEntry:
            return LeaderboardEntry(
                rank       = rank,
                ticker     = asset.ticker,
                timeframe  = asset.timeframe,
                score      = round(asset.latest_score, 4),
                gap_bits   = round(asset.peak_gap_bits, 4),
                odw_start  = asset.peak_odw_start,
                odw_end    = asset.peak_odw_end,
                days_active  = asset.days_in_regime,
                signal_age   = asset.signal_age_days,
                detail     = detail,
            )

        # ── Top FAWP ─────────────────────────────────────────────────────────
        active = sorted(
            [a for a in valid if a.regime_active],
            key=lambda a: a.latest_score, reverse=True,
        )
        top_fawp = [
            _entry(i + 1, a, detail=f"active {a.days_in_regime}d")
            for i, a in enumerate(active[:top_n])
        ]

        # ── Rising risk ───────────────────────────────────────────────────────
        slopes = [(a, _score_slope(a)) for a in valid]
        rising = sorted(slopes, key=lambda t: t[1], reverse=True)
        rising_risk = [
            _entry(i + 1, a, detail=f"slope +{sl:.4f}/win")
            for i, (a, sl) in enumerate(rising[:top_n])
            if sl > 0
        ]

        # ── Collapsing control ────────────────────────────────────────────────
        ratios = [(a, _control_gap_ratio(a)) for a in valid]
        collapsing = sorted(ratios, key=lambda t: t[1], reverse=True)
        collapsing_control = [
            _entry(i + 1, a, detail=f"pred/steer={r:.1f}x")
            for i, (a, r) in enumerate(collapsing[:top_n])
            if r > 2.0
        ]

        # ── Strongest ODW ─────────────────────────────────────────────────────
        widths = [(a, _odw_width(a)) for a in valid]
        odw_sorted = sorted(widths, key=lambda t: t[1], reverse=True)
        strongest_odw = [
            _entry(i + 1, a, detail=f"ODW {a.peak_odw_start}–{a.peak_odw_end} ({w}τ)")
            for i, (a, w) in enumerate(odw_sorted[:top_n])
            if w > 0
        ]

        return cls(
            top_fawp           = top_fawp,
            rising_risk        = rising_risk,
            collapsing_control = collapsing_control,
            strongest_odw      = strongest_odw,
            n_assets           = len(valid),
            n_flagged          = len(active),
        )

    # ── Output helpers ────────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = [
            "=" * 68,
            f"  FAWP LEADERBOARD  —  {self.generated_date}  "
            f"({self.n_flagged}/{self.n_assets} flagged)",
            "=" * 68,
        ]

        def _fmt_section(title: str, entries: List[LeaderboardEntry]) -> list:
            out = ["", f"── {title} ──"]
            if not entries:
                out.append("  (none)")
                return out
            hdr = f"  {'#':>3}  {'Ticker':<10} {'TF':<5} {'Score':>6} {'Gap(b)':>7}  {'Detail'}"
            out.append(hdr)
            out.append("  " + "-" * 58)
            for e in entries:
                out.append(
                    f"  {e.rank:>3}  {e.ticker:<10} {e.timeframe:<5} "
                    f"{e.score:>6.4f} {e.gap_bits:>7.4f}  {e.detail}"
                )
            return out

        lines += _fmt_section("Top FAWP",           self.top_fawp)
        lines += _fmt_section("Rising Risk",         self.rising_risk)
        lines += _fmt_section("Collapsing Control",  self.collapsing_control)
        lines += _fmt_section("Strongest ODW",       self.strongest_odw)
        lines += ["", "=" * 68]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "meta": {
                "generated_date":     self.generated_date,
                "fawp_index_version": _VERSION,
                "n_assets":           self.n_assets,
                "n_flagged":          self.n_flagged,
            },
            "top_fawp":           [e.to_dict() for e in self.top_fawp],
            "rising_risk":        [e.to_dict() for e in self.rising_risk],
            "collapsing_control": [e.to_dict() for e in self.collapsing_control],
            "strongest_odw":      [e.to_dict() for e in self.strongest_odw],
        }

    def to_json(self, path: Union[str, Path], indent: int = 2) -> Path:
        p = Path(path)
        p.write_text(json.dumps(self.to_dict(), indent=indent))
        return p

    def to_csv(self, path: Union[str, Path]) -> Path:
        import pandas as pd
        rows = []
        for cat, entries in [
            ("top_fawp", self.top_fawp),
            ("rising_risk", self.rising_risk),
            ("collapsing_control", self.collapsing_control),
            ("strongest_odw", self.strongest_odw),
        ]:
            for e in entries:
                row = e.to_dict()
                row["category"] = cat
                rows.append(row)
        p = Path(path)
        pd.DataFrame(rows).to_csv(p, index=False)
        return p

    def to_html(self, path: Union[str, Path]) -> Path:
        p = Path(path)
        p.write_text(self._render_html())
        return p

    def _render_html(self) -> str:
        hdr_col = "#0E2550"
        categories = [
            ("Top FAWP",           "🔴", self.top_fawp),
            ("Rising Risk",        "📈", self.rising_risk),
            ("Collapsing Control", "⚠️",  self.collapsing_control),
            ("Strongest ODW",      "🎯", self.strongest_odw),
        ]

        def _table(entries: List[LeaderboardEntry]) -> str:
            if not entries:
                return "<p style='color:#888;font-style:italic'>None detected.</p>"
            rows_html = ""
            for e in entries:
                odw = f"{e.odw_start}–{e.odw_end}" if e.odw_start is not None else "—"
                rows_html += (
                    f"<tr>"
                    f"<td>{e.rank}</td>"
                    f"<td><b>{e.ticker}</b></td>"
                    f"<td>{e.timeframe}</td>"
                    f"<td>{e.score:.4f}</td>"
                    f"<td>{e.gap_bits:.4f}</td>"
                    f"<td>{odw}</td>"
                    f"<td>{e.days_active}d</td>"
                    f"<td style='color:#888;font-size:0.85em'>{e.detail}</td>"
                    f"</tr>"
                )
            return (
                "<table><thead><tr>"
                "<th>#</th><th>Ticker</th><th>TF</th>"
                "<th>Score</th><th>Gap (bits)</th><th>ODW</th>"
                "<th>Days active</th><th>Detail</th>"
                "</tr></thead>"
                f"<tbody>{rows_html}</tbody></table>"
            )

        sections = ""
        for title, icon, entries in categories:
            sections += (
                f"<div class='section'>"
                f"<h2>{icon} {title}</h2>"
                f"{_table(entries)}"
                f"</div>"
            )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>FAWP Leaderboard</title>
<style>
  body {{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
        background:#07101E;color:#EDF0F8;margin:0;padding:0}}
  header {{background:{hdr_col};color:white;padding:1.8em 2em 1.4em}}
  header h1 {{margin:0 0 0.3em;font-size:1.45em}}
  header p  {{margin:0.2em 0;font-size:0.88em;color:#aac}}
  .badge {{display:inline-block;padding:0.4em 1.2em;border-radius:20px;
           background:rgba(255,255,255,0.12);font-size:0.78em;
           font-weight:700;letter-spacing:0.05em;margin-right:0.5em}}
  .container {{padding:1.5em 2em}}
  .section {{margin-bottom:2em;background:#0D1729;border-radius:8px;
             border:1px solid #182540;padding:1.2em 1.4em}}
  h2 {{margin:0 0 0.8em;font-size:1.05em;color:#D4AF37;
       border-bottom:1px solid #243650;padding-bottom:0.4em}}
  table {{width:100%;border-collapse:collapse;font-size:0.88em}}
  th {{text-align:left;padding:0.45em 0.8em;color:#7A90B8;
       font-size:0.75em;text-transform:uppercase;letter-spacing:0.07em;
       border-bottom:1px solid #182540}}
  td {{padding:0.4em 0.8em;border-bottom:1px solid #111E35}}
  tr:last-child td {{border-bottom:none}}
  tr:hover td {{background:rgba(255,255,255,0.03)}}
  footer {{text-align:center;padding:1.5em;color:#3A4E70;font-size:0.8em}}
  footer a {{color:#5577AA}}
</style>
</head>
<body>
<header>
  <h1>FAWP Leaderboard</h1>
  <p>
    <span class="badge">{self.n_flagged} / {self.n_assets} flagged</span>
    <span class="badge">Generated {self.generated_date}</span>
    <span class="badge">fawp-index v{_VERSION}</span>
  </p>
</header>
<div class="container">
{sections}
</div>
<footer>
  fawp-index v{_VERSION} &middot; Ralph Clayton (2026) &middot;
  <a href="https://doi.org/10.5281/zenodo.18673949">doi:10.5281/zenodo.18673949</a>
</footer>
</body>
</html>"""
