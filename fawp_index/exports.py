"""
fawp_index.exports — One-shot export methods for FAWP result objects.

Adds .to_json(), .to_markdown(), .to_html() to ODWResult, AlphaV2Result,
and FAWPResult.  Each method writes the file and returns the path.

Usage
-----
    from fawp_index import ODWDetector, FAWPAlphaIndexV2

    odw   = ODWDetector.from_e9_2_data()
    alpha = FAWPAlphaIndexV2.from_e9_2_data()

    odw.to_json("odw.json")
    odw.to_markdown("odw.md")
    odw.to_html("odw.html")

    alpha.to_json("alpha.json")
    alpha.to_markdown("alpha.md")
    alpha.to_html("alpha.html")

All three formats include:
  - every numeric result field
  - plain-English diagnosis (explain() output)
  - run metadata (date, fawp-index version, doi)

HTML is self-contained (no external dependencies), styled for readability,
and suitable for sharing as a standalone file or embedding in a notebook.

This module is internal.  The methods are injected onto the result classes
by fawp_index/__init__.py at import time.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

_VERSION = "0.7.0"
_DOI     = "https://doi.org/10.5281/zenodo.18673949"
_GITHUB  = "https://github.com/DrRalphClayton/fawp-index"


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe(v) -> Any:
    """Make a value JSON-serialisable."""
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, np.ndarray):
        return [_safe(x) for x in v.tolist()]
    if isinstance(v, dict):
        return {k: _safe(vv) for k, vv in v.items()}
    if isinstance(v, (list, tuple)):
        return [_safe(x) for x in v]
    return v


def _meta() -> dict:
    return {
        "generated_date": date.today().isoformat(),
        "fawp_index_version": _VERSION,
        "doi": _DOI,
    }


def _diagnosis(result) -> str:
    try:
        from fawp_index.explain import explain
        return explain(result)
    except Exception:
        pass
    if hasattr(result, "summary"):
        return result.summary()
    return ""


def _write(path: Union[str, Path], text: str, encoding: str = "utf-8") -> Path:
    p = Path(path)
    p.write_text(text, encoding=encoding)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# ODWResult exports
# ─────────────────────────────────────────────────────────────────────────────

def _odw_to_dict(self) -> dict:
    return {
        "result_type": "ODWResult",
        "meta": _meta(),
        "results": {
            "fawp_found":          _safe(self.fawp_found),
            "tau_h_plus":          _safe(self.tau_h_plus),
            "tau_f":               _safe(self.tau_f),
            "odw_start":           _safe(self.odw_start),
            "odw_end":             _safe(self.odw_end),
            "odw_size":            _safe(self.odw_size),
            "mean_lead_to_cliff":  _safe(self.mean_lead_to_cliff),
            "peak_gap_tau":        _safe(self.peak_gap_tau),
            "peak_gap_bits":       _safe(self.peak_gap_bits),
        },
        "diagnosis": _diagnosis(self),
    }


def _odw_to_json(self, path: Union[str, Path], indent: int = 2) -> Path:
    """
    Write result to JSON.

    Parameters
    ----------
    path : str or Path
    indent : int — JSON indentation. Default 2.

    Returns
    -------
    Path

    Example
    -------
        odw = ODWDetector.from_e9_2_data()
        odw.to_json("odw_result.json")
    """
    return _write(path, json.dumps(_odw_to_dict(self), indent=indent))


def _odw_to_markdown(self, path: Union[str, Path]) -> Path:
    """
    Write result to Markdown.

    Returns
    -------
    Path

    Example
    -------
        odw.to_markdown("odw_result.md")
    """
    d = _odw_to_dict(self)
    r = d["results"]
    m = d["meta"]
    status = "✅ FAWP DETECTED" if r["fawp_found"] else "❌ FAWP NOT DETECTED"
    odw_str = (f"τ = {r['odw_start']} — {r['odw_end']}  ({r['odw_size']} steps)"
               if r["odw_start"] is not None else "none")
    lead_str = f"{r['mean_lead_to_cliff']:.1f} steps" if r["mean_lead_to_cliff"] else "—"

    lines = [
        "# FAWP Analysis — Operational Detection Window",
        "",
        f"**{status}**",
        "",
        f"*Generated {m['generated_date']} · fawp-index v{m['fawp_index_version']}*  ",
        f"*{m['doi']}*",
        "",
        "## Key Numbers",
        "",
        "| Quantity | Value |",
        "|---|---|",
        f"| FAWP found | {'YES' if r['fawp_found'] else 'NO'} |",
        f"| Post-zero horizon (tau_h+) | {r['tau_h_plus'] if r['tau_h_plus'] is not None else '—'} |",
        f"| Failure cliff (tau_f) | {r['tau_f'] if r['tau_f'] is not None else '—'} |",
        f"| Operational Detection Window | {odw_str} |",
        f"| Mean lead to cliff | {lead_str} |",
        f"| Peak leverage gap | {r['peak_gap_bits']:.4f} bits |",
        f"| Peak gap tau | {r['peak_gap_tau'] if r['peak_gap_tau'] is not None else '—'} |",
        "",
        "## Plain-English Diagnosis",
        "",
    ]
    for line in d["diagnosis"].split("\n"):
        lines.append(line)
    lines += [
        "",
        "---",
        f"*[fawp-index]({_GITHUB}) · [paper]({_DOI})*",
    ]
    return _write(path, "\n".join(lines))


def _odw_to_html(self, path: Union[str, Path]) -> Path:
    """
    Write result to a self-contained HTML file.

    Returns
    -------
    Path

    Example
    -------
        odw.to_html("odw_result.html")
    """
    d = _odw_to_dict(self)
    r = d["results"]
    m = d["meta"]
    status_colour = "#1a7a1a" if r["fawp_found"] else "#aa1111"
    status_label  = "FAWP DETECTED" if r["fawp_found"] else "FAWP NOT DETECTED"
    odw_str = (f"&tau; = {r['odw_start']} &mdash; {r['odw_end']}  ({r['odw_size']} steps)"
               if r["odw_start"] is not None else "none")
    lead_str = f"{r['mean_lead_to_cliff']:.1f} steps" if r["mean_lead_to_cliff"] else "&mdash;"

    rows = [
        ("FAWP found",               "YES" if r["fawp_found"] else "NO"),
        ("Post-zero horizon &tau;<sub>h</sub><sup>+</sup>",
                                     str(r["tau_h_plus"]) if r["tau_h_plus"] is not None else "&mdash;"),
        ("Failure cliff &tau;<sub>f</sub>",
                                     str(r["tau_f"]) if r["tau_f"] is not None else "&mdash;"),
        ("Operational Detection Window", odw_str),
        ("Mean lead to cliff",       lead_str),
        ("Peak leverage gap",        f"{r['peak_gap_bits']:.4f} bits"),
        ("Peak gap &tau;",           str(r["peak_gap_tau"]) if r["peak_gap_tau"] is not None else "&mdash;"),
    ]

    diag_html = "<br>".join(
        line.replace("&", "&amp;").replace("<br>", "<br>")
        if not line.startswith("=") and not line.startswith("-")
        else ""
        for line in d["diagnosis"].split("\n")
    )

    html = _render_html(
        title="ODW Analysis",
        result_type="Operational Detection Window",
        status_label=status_label,
        status_colour=status_colour,
        rows=rows,
        diag_html=diag_html,
        meta=m,
    )
    return _write(path, html)


# ─────────────────────────────────────────────────────────────────────────────
# AlphaV2Result exports
# ─────────────────────────────────────────────────────────────────────────────

def _alpha2_to_dict(self) -> dict:
    return {
        "result_type": "AlphaV2Result",
        "meta": _meta(),
        "results": {
            "fawp_detected":  _safe(self.fawp_detected),
            "odw_start":      _safe(self.odw_start),
            "odw_end":        _safe(self.odw_end),
            "peak_alpha2":    _safe(self.peak_alpha2),
            "peak_tau2":      _safe(self.peak_tau2),
            "peak_pred_bits": _safe(float(self.pred_mi_corr.max())),
            "params":         _safe(self.params),
        },
        "curves": {
            "tau":          _safe(self.tau_grid),
            "pred_corr":    _safe(self.pred_mi_corr),
            "steer_corr":   _safe(self.steer_mi_corr),
            "S_m":          _safe(self.S_m),
            "R_log":        _safe(self.R_log),
            "alpha2":       _safe(self.alpha2),
            "gate":         _safe(self.gate),
        },
        "diagnosis": _diagnosis(self),
    }


def _alpha2_to_json(self, path: Union[str, Path], indent: int = 2,
                    include_curves: bool = True) -> Path:
    """
    Write result to JSON.

    Parameters
    ----------
    path : str or Path
    indent : int
    include_curves : bool
        Include full tau-wise curve arrays. Default True.

    Returns
    -------
    Path

    Example
    -------
        alpha.to_json("alpha_result.json")
        alpha.to_json("alpha_summary.json", include_curves=False)
    """
    d = _alpha2_to_dict(self)
    if not include_curves:
        d.pop("curves", None)
    return _write(path, json.dumps(d, indent=indent))


def _alpha2_to_markdown(self, path: Union[str, Path]) -> Path:
    """
    Write result to Markdown.

    Example
    -------
        alpha.to_markdown("alpha_result.md")
    """
    d = _alpha2_to_dict(self)
    r = d["results"]
    m = d["meta"]
    p = r["params"]
    status = "✅ FAWP DETECTED" if r["fawp_detected"] else "❌ FAWP NOT DETECTED"
    odw_str = (f"τ = {r['odw_start']} — {r['odw_end']}"
               if r["odw_start"] is not None else "none")

    lines = [
        "# FAWP Analysis — Alpha Index v2.1",
        "",
        f"**{status}**",
        "",
        f"*Generated {m['generated_date']} · fawp-index v{m['fawp_index_version']}*  ",
        f"*{m['doi']}*",
        "",
        "## Key Numbers",
        "",
        "| Quantity | Value |",
        "|---|---|",
        f"| FAWP detected | {'YES' if r['fawp_detected'] else 'NO'} |",
        f"| ODW | {odw_str} |",
        f"| Peak α₂ | {r['peak_alpha2']:.4f} |",
        f"| Peak α₂ tau | {r['peak_tau2'] if r['peak_tau2'] is not None else '—'} |",
        f"| Peak I_pred | {r['peak_pred_bits']:.4f} bits |",
        "",
        "## Parameters",
        "",
        "| Parameter | Value |",
        "|---|---|",
        f"| m (stability window) | {p.get('m', '—')} |",
        f"| η (pred buffer) | {p.get('eta', '—')} |",
        f"| ε (steer threshold) | {p.get('epsilon', '—')} |",
        f"| κ (resonance scaling) | {p.get('kappa', '—')} |",
        "",
        "## Plain-English Diagnosis",
        "",
    ]
    for line in d["diagnosis"].split("\n"):
        lines.append(line)
    lines += [
        "",
        "---",
        f"*[fawp-index]({_GITHUB}) · [paper]({_DOI})*",
    ]
    return _write(path, "\n".join(lines))


def _alpha2_to_html(self, path: Union[str, Path]) -> Path:
    """
    Write result to a self-contained HTML file.

    Example
    -------
        alpha.to_html("alpha_result.html")
    """
    d = _alpha2_to_dict(self)
    r = d["results"]
    m = d["meta"]
    p = r["params"]
    status_colour = "#1a7a1a" if r["fawp_detected"] else "#aa1111"
    status_label  = "FAWP DETECTED" if r["fawp_detected"] else "FAWP NOT DETECTED"
    odw_str = (f"&tau; = {r['odw_start']} &mdash; {r['odw_end']}"
               if r["odw_start"] is not None else "none")

    rows = [
        ("FAWP detected",           "YES" if r["fawp_detected"] else "NO"),
        ("ODW",                     odw_str),
        ("Peak &alpha;<sub>2</sub>", f"{r['peak_alpha2']:.4f}"),
        ("Peak &alpha;<sub>2</sub> &tau;",
                                    str(r["peak_tau2"]) if r["peak_tau2"] is not None else "&mdash;"),
        ("Peak I&#771;<sub>pred</sub>", f"{r['peak_pred_bits']:.4f} bits"),
        ("m (stability window)",    str(p.get("m", "—"))),
        ("&eta; (pred buffer)",     str(p.get("eta", "—"))),
        ("&epsilon; (steer threshold)", str(p.get("epsilon", "—"))),
        ("&kappa; (resonance)",     str(p.get("kappa", "—"))),
    ]

    diag_html = "<br>".join(
        line if not line.startswith("=") and not line.startswith("-") else ""
        for line in d["diagnosis"].split("\n")
    )

    # Embed curve chart if matplotlib available
    chart_html = _alpha2_chart_html(self)

    html = _render_html(
        title="Alpha Index v2.1 Analysis",
        result_type="Upgraded FAWP Alpha Index v2.1",
        status_label=status_label,
        status_colour=status_colour,
        rows=rows,
        diag_html=diag_html,
        meta=m,
        extra_body=chart_html,
    )
    return _write(path, html)


def _alpha2_chart_html(result) -> str:
    """Embed alpha_2 curve as a base64 PNG in an <img> tag."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import base64, io

        tau    = result.tau_grid
        alpha2 = result.alpha2

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(tau, alpha2, lw=2.2, color="#C0111A", label="α₂(τ)")
        ax.fill_between(tau, alpha2, alpha=0.13, color="#C0111A")
        ax.plot(tau, result.pred_mi_corr, lw=1.6, ls="--",
                color="#0E2550", alpha=0.7, label="I_pred (corr.)")
        ax.axhline(0, lw=0.8, color="black")
        if result.odw_start is not None:
            ax.axvspan(result.odw_start, result.odw_end,
                       alpha=0.15, color="green", label="ODW")
        ax.set_xlabel("Latency τ"); ax.set_ylabel("bits / index")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
        ax.set_title("α₂(τ) — Upgraded FAWP Alpha Index", fontsize=10)
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return (
            '<h2 style="color:#0E2550;margin-top:2em">Chart</h2>'
            f'<img src="data:image/png;base64,{b64}" '
            'style="max-width:100%;border:1px solid #ddd;border-radius:4px">'
            '<p style="font-size:0.8em;color:#888;text-align:center">'
            'Figure: α₂(τ) with null-corrected I_pred. '
            'Green shading = Operational Detection Window.</p>'
        )
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Shared HTML renderer
# ─────────────────────────────────────────────────────────────────────────────

def _render_html(
    title: str,
    result_type: str,
    status_label: str,
    status_colour: str,
    rows: list,
    diag_html: str,
    meta: dict,
    extra_body: str = "",
) -> str:
    table_rows = ""
    for i, (k, v) in enumerate(rows):
        bg = "#f8f8f8" if i % 2 == 0 else "#ffffff"
        table_rows += (
            f'<tr style="background:{bg}">'
            f'<td style="padding:7px 12px;color:#444;font-weight:500">{k}</td>'
            f'<td style="padding:7px 12px;font-weight:700;color:#0E2550">{v}</td>'
            f"</tr>\n"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} — fawp-index</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    max-width: 860px; margin: 0 auto; padding: 2em 1.5em;
    color: #222; background: #fafafa; line-height: 1.6;
  }}
  header {{
    background: #0E2550; color: white; padding: 2em 2em 1.5em;
    border-radius: 8px; margin-bottom: 1.5em;
  }}
  header h1 {{ margin: 0 0 0.3em; font-size: 1.5em; }}
  header p  {{ margin: 0.2em 0; font-size: 0.88em; color: #aac; }}
  .badge {{
    display: inline-block; padding: 0.4em 1.1em;
    border-radius: 20px; font-weight: 700; font-size: 1em;
    color: white; background: {status_colour}; margin: 0.8em 0;
  }}
  h2 {{ color: #0E2550; border-bottom: 2px solid #D4AF37;
        padding-bottom: 4px; margin-top: 1.8em; }}
  table {{
    width: 100%; border-collapse: collapse; margin: 1em 0;
    border-radius: 6px; overflow: hidden;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
  }}
  thead th {{
    background: #0E2550; color: white; padding: 9px 12px;
    text-align: left; font-size: 0.9em;
  }}
  .diag {{
    background: #f4f4f4; border-left: 4px solid #D4AF37;
    padding: 1em 1.2em; border-radius: 4px;
    font-size: 0.92em; white-space: pre-wrap; font-family: monospace;
    color: #333;
  }}
  footer {{
    margin-top: 3em; padding-top: 1em;
    border-top: 1px solid #ddd; font-size: 0.8em; color: #888;
  }}
  a {{ color: #0E2550; }}
</style>
</head>
<body>

<header>
  <h1>FAWP Analysis &mdash; {result_type}</h1>
  <p>Generated {meta["generated_date"]} &bull; fawp-index v{meta["fawp_index_version"]}</p>
  <p><a href="{meta["doi"]}" style="color:#D4AF37">{meta["doi"]}</a></p>
</header>

<div class="badge">{status_label}</div>

<h2>Key Numbers</h2>
<table>
  <thead><tr><th>Quantity</th><th>Value</th></tr></thead>
  <tbody>
{table_rows}
  </tbody>
</table>

<h2>Plain-English Diagnosis</h2>
<div class="diag">{diag_html}</div>

{extra_body}

<footer>
  <a href="https://github.com/DrRalphClayton/fawp-index">fawp-index</a> &bull;
  Ralph Clayton (2026) &bull;
  <a href="{meta["doi"]}">{meta["doi"]}</a>
</footer>

</body>
</html>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Injection — called once by fawp_index/__init__.py
# ─────────────────────────────────────────────────────────────────────────────

def _inject_exports():
    """Attach export methods to all result classes."""
    from fawp_index.detection.odw import ODWResult
    from fawp_index.core.alpha_v2 import AlphaV2Result

    # ODWResult
    ODWResult.to_json     = _odw_to_json
    ODWResult.to_markdown = _odw_to_markdown
    ODWResult.to_html     = _odw_to_html
    ODWResult.to_dict     = _odw_to_dict

    # AlphaV2Result
    AlphaV2Result.to_json     = _alpha2_to_json
    AlphaV2Result.to_markdown = _alpha2_to_markdown
    AlphaV2Result.to_html     = _alpha2_to_html
    AlphaV2Result.to_dict     = _alpha2_to_dict
