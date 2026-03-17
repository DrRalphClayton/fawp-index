"""
fawp_index.report_html — One-click HTML report from a WatchlistResult or WeatherFAWPResult.

Usage::

    from fawp_index.report_html import generate_html_report
    html = generate_html_report(wl_result, title="SPY QQQ Scan")
    with open("report.html", "w") as f: f.write(html)
"""

from __future__ import annotations
import base64, io
from datetime import datetime
from typing import Optional, Union

_VERSION = "1.1.8"


def _b64_chart(fig) -> str:
    """Convert matplotlib figure to base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="#07101E", bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def generate_html_report(
    result,
    title: str = "FAWP Scan Report",
    include_charts: bool = True,
) -> str:
    """
    Generate a self-contained HTML report from a WatchlistResult or WeatherFAWPResult.

    Parameters
    ----------
    result : WatchlistResult or WeatherFAWPResult
    title : str
    include_charts : bool   Embed MI curve charts as base64 PNG. Default True.

    Returns
    -------
    str — complete HTML document, ready to save as .html
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Detect result type
    from fawp_index.weather import WeatherFAWPResult
    is_weather = isinstance(result, WeatherFAWPResult)

    # ── Build content ──────────────────────────────────────────────────────
    if is_weather:
        content = _weather_report_body(result, include_charts)
        subtitle = f"{result.variable} · {result.location}"
    else:
        content = _finance_report_body(result, include_charts)
        n = getattr(result, 'n_assets', '?')
        flagged = getattr(result, 'n_flagged', '?')
        subtitle = f"{n} assets scanned · {flagged} flagged"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #07101E; color: #EDF0F8; font-family: 'DM Sans', sans-serif;
          line-height: 1.6; padding: 2.5em; max-width: 900px; margin: 0 auto; }}
  h1   {{ font-family: 'Syne', sans-serif; font-size: 1.8em; font-weight: 800;
          color: #D4AF37; margin-bottom: .2em; }}
  h2   {{ font-family: 'Syne', sans-serif; font-size: 1.1em; font-weight: 700;
          color: #EDF0F8; margin: 1.8em 0 .6em; border-bottom: 1px solid #182540;
          padding-bottom: .3em; }}
  .sub {{ color: #3A4E70; font-size: .85em; margin-bottom: .4em; }}
  .ts  {{ color: #3A4E70; font-size: .75em; margin-bottom: 1.8em; }}
  .kpi-row {{ display: flex; gap: 1em; flex-wrap: wrap; margin: 1em 0; }}
  .kpi {{ background: #0D1729; border: 1px solid #182540; border-radius: 8px;
          padding: .8em 1.2em; min-width: 120px; text-align: center; }}
  .kpi-val {{ font-family: 'JetBrains Mono', monospace; font-size: 1.3em;
              font-weight: 600; color: #D4AF37; }}
  .kpi-lbl {{ font-size: .7em; color: #7A90B8; text-transform: uppercase;
              letter-spacing: .07em; margin-top: .2em; }}
  table {{ width: 100%; border-collapse: collapse; margin: .8em 0; font-size: .88em; }}
  th    {{ color: #7A90B8; text-align: left; padding: .4em .8em;
           border-bottom: 1px solid #182540; font-size: .75em;
           text-transform: uppercase; letter-spacing: .06em; }}
  td    {{ padding: .45em .8em; border-bottom: 1px solid #111E35; }}
  .fawp {{ color: #C0111A; font-weight: 700; }}
  .clear{{ color: #1DB954; }}
  .exp-box {{ background: #0A1523; border: 1px solid #182540; border-left: 3px solid #D4AF37;
              border-radius: 6px; padding: 1em 1.2em; margin: 1em 0;
              font-size: .88em; color: #7A90B8; line-height: 1.6; }}
  img   {{ max-width: 100%; border-radius: 6px; margin: .8em 0; }}
  .footer {{ margin-top: 3em; padding-top: 1em; border-top: 1px solid #182540;
             font-size: .75em; color: #3A4E70; text-align: center; }}
</style>
</head>
<body>
<h1>🔴 {title}</h1>
<div class="sub">{subtitle}</div>
<div class="ts">Generated {ts} · fawp-index v{_VERSION} · doi:10.5281/zenodo.18673949</div>
{content}
<div class="footer">
  fawp-index v{_VERSION} · Ralph Clayton 2026 ·
  <a href="https://fawp-scanner.info" style="color:#4A7FCC">fawp-scanner.info</a> ·
  <a href="https://doi.org/10.5281/zenodo.18673949" style="color:#4A7FCC">doi:10.5281/zenodo.18673949</a>
</div>
</body>
</html>"""


def _kpi(val, lbl):
    return (f'<div class="kpi"><div class="kpi-val">{val}</div>'
            f'<div class="kpi-lbl">{lbl}</div></div>')


def _weather_report_body(r, include_charts: bool) -> str:
    fawp_badge = '<span class="fawp">🔴 FAWP DETECTED</span>' if r.fawp_found \
                 else '<span class="clear">✅ No FAWP</span>'
    odw = f"τ {r.odw_start}–{r.odw_end}" if r.fawp_found else "—"
    tauh = str(r.odw_result.tau_h_plus) if r.odw_result.tau_h_plus else "—"
    tauf = str(r.odw_result.tau_f)      if r.odw_result.tau_f      else "—"

    kpis = (
        _kpi(fawp_badge, "Detection") +
        _kpi(f"{r.peak_gap_bits:.4f} b", "Peak gap") +
        _kpi(odw, "ODW") +
        _kpi(tauh, "τ⁺ₕ horizon") +
        _kpi(tauf, "τf cliff") +
        _kpi(f"{r.n_obs:,}", "Observations")
    )

    chart_html = ""
    if include_charts and len(r.tau) > 0:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
            fig, ax = plt.subplots(figsize=(9, 3))
            fig.patch.set_facecolor("#07101E")
            ax.set_facecolor("#0D1729")
            ax.plot(r.tau, r.pred_mi,  color="#D4AF37", lw=2,   label="Prediction MI")
            ax.plot(r.tau, r.steer_mi, color="#4A7FCC", lw=1.5, ls="--", label="Steering MI")
            ax.axhline(0.01, color="#3A4E70", ls=":", lw=1)
            if r.fawp_found and r.odw_start:
                ax.axvspan(r.odw_start, r.odw_end, alpha=0.15, color="#C0111A", label="ODW")
            ax.set_xlabel("τ (delay, days)", fontsize=8, color="#7A90B8")
            ax.set_ylabel("MI (bits)", fontsize=8, color="#7A90B8")
            ax.tick_params(colors="#7A90B8", labelsize=7)
            ax.legend(fontsize=7, facecolor="#0D1729", labelcolor="#EDF0F8",
                      edgecolor="#182540")
            for sp in ax.spines.values(): sp.set_edgecolor("#182540")
            plt.tight_layout(pad=0.4)
            chart_html = f'<img src="data:image/png;base64,{_b64_chart(fig)}" alt="MI curves">'
            plt.close(fig)
        except Exception:
            pass

    explanation = r.explain() if hasattr(r, "explain") else ""
    exp_box = f'<div class="exp-box">{explanation.replace(chr(10), "<br>")}</div>' if explanation else ""

    return f"""
<h2>Result</h2>
<div class="kpi-row">{kpis}</div>
<table>
  <tr><th>Field</th><th>Value</th></tr>
  <tr><td>Variable</td><td>{r.variable}</td></tr>
  <tr><td>Location</td><td>{r.location}</td></tr>
  <tr><td>Period</td><td>{r.date_range[0]} → {r.date_range[1]}</td></tr>
  <tr><td>Forecast horizon</td><td>{r.horizon_days} day(s)</td></tr>
  <tr><td>Skill metric</td><td>{r.skill_metric}</td></tr>
</table>
<h2>MI Curves</h2>
{chart_html if chart_html else "<p style='color:#3A4E70'>Install matplotlib to include charts.</p>"}
<h2>Interpretation</h2>
{exp_box}"""


def _finance_report_body(r, include_charts: bool) -> str:
    assets = r.rank_by("score") if hasattr(r, "rank_by") else []
    flagged = [a for a in assets if getattr(a, "regime_active", False) and not getattr(a, "error", False)]
    n_total  = getattr(r, "n_assets",  len(assets))
    n_flagged = getattr(r, "n_flagged", len(flagged))

    kpis = (
        _kpi(n_flagged, "FAWP active") +
        _kpi(n_total, "Assets scanned") +
        _kpi(f"{n_flagged/max(n_total,1)*100:.0f}%", "Flag rate")
    )

    rows = ""
    for a in assets[:20]:
        if getattr(a, "error", False):
            continue
        fawp_cls = "fawp" if getattr(a, "regime_active", False) else "clear"
        fawp_txt = "🔴 FAWP" if getattr(a, "regime_active", False) else "—"
        score    = f"{getattr(a, 'latest_score', 0):.4f}"
        gap      = f"{getattr(a, 'peak_gap_bits', 0):.4f}"
        odw_s    = getattr(a, "peak_odw_start", None)
        odw_e    = getattr(a, "peak_odw_end",   None)
        odw      = f"τ {odw_s}–{odw_e}" if odw_s is not None else "—"
        rows += (f"<tr><td>{a.ticker}</td><td>{a.timeframe}</td>"
                 f"<td class='{fawp_cls}'>{fawp_txt}</td>"
                 f"<td>{score}</td><td>{gap}</td><td>{odw}</td></tr>")

    return f"""
<h2>Summary</h2>
<div class="kpi-row">{kpis}</div>
<h2>Asset Results</h2>
<table>
  <tr><th>Ticker</th><th>Timeframe</th><th>FAWP</th>
      <th>Score</th><th>Gap (bits)</th><th>ODW</th></tr>
  {rows}
</table>"""
