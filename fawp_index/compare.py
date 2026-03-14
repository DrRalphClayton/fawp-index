"""
fawp_index.compare — Side-by-side FAWP result comparison
=========================================================

Compare two FAWP detection results: two assets, two time windows,
two parameter sets, or two experiments.

Quick start
-----------
    from fawp_index import ODWDetector, compare_fawp

    odw_a = ODWDetector.from_e9_2_data(steering='u')
    odw_b = ODWDetector.from_e9_2_data(steering='xi')

    cmp = compare_fawp(odw_a, odw_b, label_a="u-steering", label_b="xi-steering")
    print(cmp.summary())
    cmp.to_html("comparison.html")
    cmp.to_json("comparison.json")
    cmp.plot()

Also works with AlphaV2Result, BenchmarkResult, or any mix::

    from fawp_index import FAWPAlphaIndexV2
    alpha_a = FAWPAlphaIndexV2.from_e9_2_data(steering='u')
    alpha_b = FAWPAlphaIndexV2.from_e9_2_data(steering='xi')
    cmp = compare_fawp(alpha_a, alpha_b, label_a="u", label_b="xi")

Ralph Clayton (2026) · https://doi.org/10.5281/zenodo.18673949
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np

from fawp_index import __version__ as _VERSION
_DOI     = "https://doi.org/10.5281/zenodo.18673949"
_GITHUB  = "https://github.com/DrRalphClayton/fawp-index"


# ─────────────────────────────────────────────────────────────────────────────
# Row dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ComparisonRow:
    """One row in a comparison table."""
    field:     str          # e.g. "tau_h+"
    val_a:     Any          # value for result A
    val_b:     Any          # value for result B
    delta:     Any          # B - A (or None if not numeric)
    winner:    str          # "A", "B", "tie", or "—"
    direction: str          # "lower_better", "higher_better", "—"
    note:      str = ""     # optional annotation


# ─────────────────────────────────────────────────────────────────────────────
# ComparisonResult
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ComparisonResult:
    """
    Output of compare_fawp().

    Attributes
    ----------
    label_a, label_b : str
        Names for the two results.
    result_a, result_b : ODWResult or AlphaV2Result
        The two result objects.
    result_type : str
        'ODW' or 'AlphaV2'.
    rows : list of ComparisonRow
        One row per comparable field.
    winner_overall : str
        'A', 'B', or 'tie'.
    score_a, score_b : int
        Number of fields won by each side.
    """

    label_a: str
    label_b: str
    result_a: object
    result_b: object
    result_type: str
    rows: List[ComparisonRow]
    winner_overall: str
    score_a: int
    score_b: int

    # ── summary ──────────────────────────────────────────────────────────────

    def summary(self) -> str:
        w = self.winner_overall
        win_label = (
            f"{self.label_a}" if w == "A"
            else f"{self.label_b}" if w == "B"
            else "tie"
        )

        col_a = max(len(self.label_a), 10)
        col_b = max(len(self.label_b), 10)
        col_f = 26

        header = (
            f"  {'Field':<{col_f}} {self.label_a:>{col_a}} "
            f"{self.label_b:>{col_b}}  {'Delta':>10}  {'Winner':>6}"
        )
        sep = "  " + "-" * (col_f + col_a + col_b + 28)

        lines = [
            "=" * (col_f + col_a + col_b + 32),
            f"  FAWP Comparison: {self.label_a!r} vs {self.label_b!r}",
            "=" * (col_f + col_a + col_b + 32),
            header, sep,
        ]

        for row in self.rows:
            def _fmt(v):
                if v is None:
                    return "—"
                if isinstance(v, bool):
                    return "YES" if v else "NO"
                if isinstance(v, float):
                    return f"{v:.4f}"
                return str(v)

            delta_str = ""
            if row.delta is not None and isinstance(row.delta, (int, float)):
                sign = "+" if row.delta > 0 else ""
                delta_str = f"{sign}{row.delta:.3f}"
            elif row.delta is not None:
                delta_str = str(row.delta)

            arrow = {"A": "← A", "B": "B →", "tie": "=", "—": ""}
            lines.append(
                f"  {row.field:<{col_f}} {_fmt(row.val_a):>{col_a}} "
                f"{_fmt(row.val_b):>{col_b}}  {delta_str:>10}  "
                f"{arrow.get(row.winner, ''):>6}"
            )

        lines += [
            sep,
            f"  Score: {self.label_a}={self.score_a}  {self.label_b}={self.score_b}",
            f"  Overall winner: {win_label}",
            "=" * (col_f + col_a + col_b + 32),
        ]
        return "\n".join(lines)

    # ── plot ─────────────────────────────────────────────────────────────────

    def plot(self, show: bool = True, save_path: Optional[str] = None):
        """
        Radar/bar chart comparison of key metrics.

        Returns matplotlib Figure.
        """
        try:
            import matplotlib
            matplotlib.use("Agg" if not show else matplotlib.get_backend())
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("pip install fawp-index[plot]")

        # Pick numeric rows for bar chart
        numeric_rows = [
            r for r in self.rows
            if isinstance(r.val_a, (int, float)) and isinstance(r.val_b, (int, float))
            and r.val_a is not None and r.val_b is not None
        ]
        if not numeric_rows:
            return None

        labels    = [r.field for r in numeric_rows]
        vals_a    = [float(r.val_a) for r in numeric_rows]
        vals_b    = [float(r.val_b) for r in numeric_rows]
        n = len(labels)
        x = np.arange(n)
        w = 0.35

        fig, ax = plt.subplots(figsize=(max(7, n * 1.4), 5))
        ax.bar(x - w/2, vals_a, w, label=self.label_a,
               color="#0E2550", alpha=0.85)
        ax.bar(x + w/2, vals_b, w, label=self.label_b,
               color="#C0111A", alpha=0.85)

        # Winner highlights
        for i, row in enumerate(numeric_rows):
            if row.winner == "A":
                ax.bar(x[i] - w/2, vals_a[i], w, color="#D4AF37", alpha=0.35)
            elif row.winner == "B":
                ax.bar(x[i] + w/2, vals_b[i], w, color="#D4AF37", alpha=0.35)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("Value")
        ax.set_title(
            f"FAWP Comparison: {self.label_a!r} vs {self.label_b!r}\n"
            f"Score: {self.label_a}={self.score_a}  {self.label_b}={self.score_b}  "
            f"Overall: {self.winner_overall}",
            fontsize=9,
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, axis="y")
        fig.text(0.99, 0.01, "fawp-index | Clayton (2026)",
                 ha="right", fontsize=7, color="grey", style="italic")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            try:
                plt.show()
            except Exception:
                pass
        return fig

    # ── exports ──────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        def _ser(v):
            if v is None:
                return None
            if isinstance(v, (bool, np.bool_)):
                return bool(v)
            if isinstance(v, (int, np.integer)):
                return int(v)
            if isinstance(v, (float, np.floating)):
                return float(v) if np.isfinite(v) else None
            return v

        return {
            "meta": {
                "generated_date": date.today().isoformat(),
                "fawp_index_version": _VERSION,
                "doi": _DOI,
            },
            "label_a": self.label_a,
            "label_b": self.label_b,
            "result_type": self.result_type,
            "winner_overall": self.winner_overall,
            "score_a": self.score_a,
            "score_b": self.score_b,
            "rows": [
                {
                    "field":     row.field,
                    "val_a":     _ser(row.val_a),
                    "val_b":     _ser(row.val_b),
                    "delta":     _ser(row.delta),
                    "winner":    row.winner,
                    "direction": row.direction,
                    "note":      row.note,
                }
                for row in self.rows
            ],
        }

    def to_json(self, path: Union[str, Path], indent: int = 2) -> Path:
        """Write comparison to JSON."""
        p = Path(path)
        p.write_text(json.dumps(self.to_dict(), indent=indent))
        return p

    def to_html(self, path: Union[str, Path]) -> Path:
        """Write comparison to a self-contained HTML file."""
        p = Path(path)
        p.write_text(_cmp_html(self))
        return p

    def to_pdf(self, path: Union[str, Path], **kwargs) -> Path:
        """Write comparison as a PDF via fawp_index.report."""
        from fawp_index.report import generate_report
        return generate_report(
            {self.label_a: self.result_a, self.label_b: self.result_b},
            path,
            title=f"Comparison: {self.label_a} vs {self.label_b}",
            **kwargs,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Result extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_odw_fields(r) -> dict:
    """Pull comparable fields from an ODWResult."""
    return {
        "FAWP detected":      bool(r.fawp_found),
        "tau_h+":             r.tau_h_plus,
        "tau_f":              r.tau_f,
        "ODW start":          r.odw_start,
        "ODW end":            r.odw_end,
        "ODW size (steps)":   r.odw_size,
        "Peak gap (bits)":    float(r.peak_gap_bits),
        "Peak gap tau":       r.peak_gap_tau,
        "Mean lead to cliff": (float(r.mean_lead_to_cliff)
                               if r.mean_lead_to_cliff is not None else None),
    }


def _extract_alpha2_fields(r) -> dict:
    """Pull comparable fields from an AlphaV2Result."""
    return {
        "FAWP detected":  bool(r.fawp_detected),
        "ODW start":      r.odw_start,
        "ODW end":        r.odw_end,
        "Peak alpha_2":   float(r.peak_alpha2),
        "Peak alpha_2 tau": r.peak_tau2,
        "Peak I_pred":    float(r.pred_mi_corr.max()),
        "Param m":        r.params.get("m"),
        "Param eta":      r.params.get("eta"),
        "Param epsilon":  r.params.get("epsilon"),
        "Param kappa":    r.params.get("kappa"),
    }


def _extract_any(r) -> Tuple[dict, str]:
    """Auto-detect result type and extract fields."""
    if hasattr(r, "odw_result"):
        # BenchmarkResult — use its ODWResult
        return _extract_odw_fields(r.odw_result), "ODW"
    if hasattr(r, "tau_h_plus") and hasattr(r, "odw_start"):
        return _extract_odw_fields(r), "ODW"
    if hasattr(r, "alpha2") and hasattr(r, "S_m"):
        return _extract_alpha2_fields(r), "AlphaV2"
    raise TypeError(
        f"Cannot compare object of type {type(r).__name__}. "
        "Expected ODWResult, AlphaV2Result, or BenchmarkResult."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Winner logic
# ─────────────────────────────────────────────────────────────────────────────

# For each field: is lower or higher better?
_FIELD_DIRECTION = {
    "FAWP detected":      None,             # boolean — no direction
    "tau_h+":             "lower_better",   # earlier horizon is more sensitive
    "tau_f":              "higher_better",  # later cliff → more time
    "ODW start":          "lower_better",   # earlier detection
    "ODW end":            "higher_better",  # longer window
    "ODW size (steps)":   "higher_better",  # larger window
    "Peak gap (bits)":    "higher_better",  # bigger gap = clearer signal
    "Peak gap tau":       None,             # informational
    "Mean lead to cliff": "higher_better",  # more lead time
    "Peak alpha_2":       "higher_better",
    "Peak alpha_2 tau":   None,
    "Peak I_pred":        "higher_better",
    "Param m":            None,
    "Param eta":          None,
    "Param epsilon":      None,
    "Param kappa":        None,
}


def _determine_winner(
    field: str,
    val_a: Any,
    val_b: Any,
) -> Tuple[str, str]:
    """Return (winner, direction) for one field."""
    direction = _FIELD_DIRECTION.get(field, None)

    # Both None
    if val_a is None and val_b is None:
        return "tie", direction or "—"
    if val_a is None:
        return "B", direction or "—"
    if val_b is None:
        return "A", direction or "—"

    # Boolean
    if isinstance(val_a, bool) and isinstance(val_b, bool):
        if val_a == val_b:
            return "tie", "—"
        return "—", "—"

    # Numeric
    if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
        if direction == "lower_better":
            if val_a < val_b:
                return "A", direction
            if val_b < val_a:
                return "B", direction
            return "tie", direction
        if direction == "higher_better":
            if val_a > val_b:
                return "A", direction
            if val_b > val_a:
                return "B", direction
            return "tie", direction

    return "—", direction or "—"


# ─────────────────────────────────────────────────────────────────────────────
# Main comparison builder
# ─────────────────────────────────────────────────────────────────────────────

def compare_fawp(
    result_a,
    result_b,
    label_a: str = "A",
    label_b: str = "B",
) -> ComparisonResult:
    """
    Compare two FAWP detection results side by side.

    Accepts ODWResult, AlphaV2Result, or BenchmarkResult (or any mix).
    Auto-detects result type.

    Parameters
    ----------
    result_a : ODWResult | AlphaV2Result | BenchmarkResult
    result_b : same type as result_a
    label_a : str — display name for result A (default 'A')
    label_b : str — display name for result B (default 'B')

    Returns
    -------
    ComparisonResult

    Examples
    --------
    Compare two steering definitions::

        from fawp_index import ODWDetector, compare_fawp

        odw_u  = ODWDetector.from_e9_2_data(steering='u')
        odw_xi = ODWDetector.from_e9_2_data(steering='xi')

        cmp = compare_fawp(odw_u, odw_xi, label_a="u-steering", label_b="xi-steering")
        print(cmp.summary())
        cmp.to_html("comparison.html")

    Compare two parameter sets::

        from fawp_index import FAWPAlphaIndexV2

        a1 = FAWPAlphaIndexV2(m=3).compute(tau, pred, steer, fail)
        a2 = FAWPAlphaIndexV2(m=7).compute(tau, pred, steer, fail)
        cmp = compare_fawp(a1, a2, label_a="m=3", label_b="m=7")

    Compare benchmark cases::

        from fawp_index.benchmarks import clean_control, delayed_collapse
        cmp = compare_fawp(clean_control(), delayed_collapse(),
                           label_a="clean", label_b="delayed")
    """
    fields_a, type_a = _extract_any(result_a)
    fields_b, type_b = _extract_any(result_b)

    if type_a != type_b:
        raise TypeError(
            f"Cannot directly compare {type_a} and {type_b} results. "
            "Both must be the same result type."
        )

    rows: List[ComparisonRow] = []
    score_a = score_b = 0

    # Use field order from type A
    for field_name, val_a in fields_a.items():
        val_b = fields_b.get(field_name)

        # Delta
        delta = None
        if (isinstance(val_a, (int, float)) and val_a is not None and
                isinstance(val_b, (int, float)) and val_b is not None):
            try:
                delta = float(val_b) - float(val_a)
            except Exception:
                delta = None

        winner, direction = _determine_winner(field_name, val_a, val_b)
        if winner == "A":
            score_a += 1
        elif winner == "B":
            score_b += 1

        rows.append(ComparisonRow(
            field=field_name, val_a=val_a, val_b=val_b,
            delta=delta, winner=winner, direction=direction,
        ))

    if score_a > score_b:
        overall = "A"
    elif score_b > score_a:
        overall = "B"
    else:
        overall = "tie"

    # Unwrap BenchmarkResult → ODWResult for storage
    ra = result_a.odw_result if hasattr(result_a, "odw_result") else result_a
    rb = result_b.odw_result if hasattr(result_b, "odw_result") else result_b

    return ComparisonResult(
        label_a=label_a, label_b=label_b,
        result_a=ra, result_b=rb,
        result_type=type_a,
        rows=rows,
        winner_overall=overall,
        score_a=score_a, score_b=score_b,
    )


# ─────────────────────────────────────────────────────────────────────────────
# HTML renderer
# ─────────────────────────────────────────────────────────────────────────────

def _cmp_html(cmp: ComparisonResult) -> str:
    w = cmp.winner_overall
    win_name  = cmp.label_a if w == "A" else cmp.label_b if w == "B" else "Tie"
    hdr_col   = "#0E2550"

    def _fmt(v):
        if v is None:
            return "&mdash;"
        if isinstance(v, bool):
            return "YES" if v else "NO"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    def _delta_fmt(row):
        if row.delta is None:
            return "&mdash;"
        sign = "+" if row.delta > 0 else ""
        return f"{sign}{row.delta:.3f}"

    winner_col = {"A": "#1a7a1a", "B": "#aa1111", "tie": "#888", "—": "#888"}

    table_rows = ""
    for i, row in enumerate(cmp.rows):
        bg = "#f8f8f8" if i % 2 == 0 else "#fff"
        wc = winner_col.get(row.winner, "#888")
        win_label = (
            f'<span style="color:{wc};font-weight:700">'
            f'{"← " + cmp.label_a if row.winner == "A" else cmp.label_b + " →" if row.winner == "B" else row.winner}'
            f"</span>"
        )
        # Highlight winning cell
        style_a = f'style="padding:7px 12px;font-weight:{"700" if row.winner=="A" else "400"};color:{"#0E2550" if row.winner=="A" else "#333"}"'
        style_b = f'style="padding:7px 12px;font-weight:{"700" if row.winner=="B" else "400"};color:{"#0E2550" if row.winner=="B" else "#333"}"'
        table_rows += (
            f'<tr style="background:{bg}">'
            f'<td style="padding:7px 12px;font-weight:500">{row.field}</td>'
            f'<td {style_a}>{_fmt(row.val_a)}</td>'
            f'<td {style_b}>{_fmt(row.val_b)}</td>'
            f'<td style="padding:7px 12px;color:#666;font-family:monospace">{_delta_fmt(row)}</td>'
            f'<td style="padding:7px 12px">{win_label}</td>'
            f'</tr>\n'
        )

    # Try to embed comparison bar chart
    chart_html = ""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import base64
        import io

        numeric = [
            r for r in cmp.rows
            if isinstance(r.val_a, (int, float)) and r.val_a is not None
            and isinstance(r.val_b, (int, float)) and r.val_b is not None
        ]
        if numeric:
            labels = [r.field for r in numeric]
            vals_a = [float(r.val_a) for r in numeric]
            vals_b = [float(r.val_b) for r in numeric]
            n = len(labels)
            x = np.arange(n)
            ww = 0.35

            fig, ax = plt.subplots(figsize=(max(6, n * 1.3), 4))
            ax.bar(x - ww/2, vals_a, ww, label=cmp.label_a,
                   color="#0E2550", alpha=0.85)
            ax.bar(x + ww/2, vals_b, ww, label=cmp.label_b,
                   color="#C0111A", alpha=0.85)
            for i, row in enumerate(numeric):
                if row.winner == "A":
                    ax.bar(x[i] - ww/2, vals_a[i], ww, color="#D4AF37", alpha=0.35)
                elif row.winner == "B":
                    ax.bar(x[i] + ww/2, vals_b[i], ww, color="#D4AF37", alpha=0.35)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=8)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.2, axis="y")
            ax.set_title(f"{cmp.label_a} vs {cmp.label_b}", fontsize=9)
            plt.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
            plt.close(fig)
            b64 = base64.b64encode(buf.getvalue()).decode()
            chart_html = (
                '<h2 style="color:#0E2550;margin-top:2em">Chart</h2>'
                f'<img src="data:image/png;base64,{b64}" '
                'style="max-width:100%;border:1px solid #ddd;border-radius:4px">'
                f'<p style="font-size:0.8em;color:#888;text-align:center">'
                f'Gold highlight = winner per field.</p>'
            )
    except Exception:
        pass

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>FAWP Comparison: {cmp.label_a} vs {cmp.label_b}</title>
<style>
  body {{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
         max-width:900px;margin:0 auto;padding:2em 1.5em;
         background:#fafafa;color:#222;line-height:1.6}}
  header {{background:{hdr_col};color:white;padding:1.8em 2em 1.4em;
           border-radius:8px;margin-bottom:1.5em}}
  header h1 {{margin:0 0 0.3em;font-size:1.45em}}
  header p {{margin:0.2em 0;font-size:0.88em;color:#aac}}
  .badge {{display:inline-block;padding:0.35em 1.1em;border-radius:16px;
           font-weight:700;color:#0E2550;background:#D4AF37;margin:0.6em 0}}
  .score {{display:flex;gap:2em;margin:0.8em 0}}
  .score-box {{background:#fff;border:2px solid #ddd;border-radius:8px;
               padding:0.6em 1.2em;text-align:center;min-width:120px}}
  .score-box.winner {{border-color:#D4AF37}}
  .score-box h3 {{margin:0;font-size:1em;color:#0E2550}}
  .score-box span {{font-size:1.8em;font-weight:700;color:#0E2550}}
  h2 {{color:#0E2550;border-bottom:2px solid #D4AF37;padding-bottom:4px}}
  table {{width:100%;border-collapse:collapse;margin:1em 0;
          box-shadow:0 1px 4px rgba(0,0,0,0.07);border-radius:6px;overflow:hidden}}
  thead th {{background:{hdr_col};color:white;padding:9px 12px;text-align:left}}
  footer {{margin-top:3em;padding-top:1em;border-top:1px solid #ddd;
           font-size:0.8em;color:#888}}
  a {{color:#0E2550}}
</style>
</head>
<body>
<header>
  <h1>FAWP Comparison: {cmp.label_a} vs {cmp.label_b}</h1>
  <p>Generated {date.today().isoformat()} &bull
  fawp-index v{_VERSION}</p>
  <p><a href="{_DOI}" style="color:#D4AF37">{_DOI}</a></p>
</header>

<div class="badge">Overall winner: {win_name}</div>

<div class="score">
  <div class="score-box {"winner" if w == "A" else ""}">
    <h3>{cmp.label_a}</h3>
    <span>{cmp.score_a}</span><br>
    <small>fields won</small>
  </div>
  <div class="score-box {"winner" if w == "B" else ""}">
    <h3>{cmp.label_b}</h3>
    <span>{cmp.score_b}</span><br>
    <small>fields won</small>
  </div>
</div>

<h2>Field Comparison ({cmp.result_type})</h2>
<table>
  <thead>
    <tr>
      <th>Field</th>
      <th>{cmp.label_a}</th>
      <th>{cmp.label_b}</th>
      <th>Delta (B&minus;A)</th>
      <th>Winner</th>
    </tr>
  </thead>
  <tbody>{table_rows}</tbody>
</table>

{chart_html}

<footer>
  <a href="{_GITHUB}">fawp-index</a> &bull;
  Ralph Clayton (2026) &bull;
  <a href="{_DOI}">{_DOI}</a>
</footer>
</body>
</html>
"""
