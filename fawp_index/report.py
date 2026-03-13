"""
fawp_index.report — PDF report generator for FAWP results.

Produces a structured, citable PDF from any FAWP result object.
Designed to be handed to a journal reviewer or collaborator without
requiring them to run Python.

Usage
-----
    from fawp_index import ODWDetector, FAWPAlphaIndexV2
    from fawp_index.report import generate_report

    odw    = ODWDetector.from_e9_2_data()
    alpha  = FAWPAlphaIndexV2.from_e9_2_data()

    # Polished citable report (default)
    generate_report(
        odw,
        "e9_2_report.pdf",
        title="E9.2 Steering Definition Comparison",
        doi="10.5281/zenodo.18673949",
    )

    # Both result types together
    generate_report(
        {"odw": odw, "alpha": alpha},
        "e9_2_full.pdf",
        title="E9.2 Full Analysis",
    )

    # Personal lab-notebook style
    generate_report(odw, "lab_notes.pdf", mode="lab")

Requires
--------
    pip install fawp-index[report]
    (adds reportlab)

Note on Greek / math notation
------------------------------
ReportLab's built-in fonts (Helvetica, Times-Roman) do not contain Greek
glyphs.  All math notation in body text uses ASCII: tau_h, tau_f, I_pred,
I_steer, alpha_2.  Subscripts are rendered with ReportLab's <sub> tag.
Figures (matplotlib) carry the proper symbols.
"""

from __future__ import annotations

import io
import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

# ── reportlab imports (optional dep) ─────────────────────────────────────────
def _require_reportlab():
    try:
        import reportlab  # noqa: F401
    except ImportError:
        raise ImportError(
            "reportlab is required for PDF reports.\n"
            "Install with:  pip install fawp-index[report]\n"
            "or:            pip install reportlab"
        )

# ── matplotlib import (optional dep) ─────────────────────────────────────────
def _require_matplotlib():
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        raise ImportError(
            "matplotlib is required to embed figures in PDF reports.\n"
            "Install with:  pip install fawp-index[plot]"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette (matches the rest of the package)
# ─────────────────────────────────────────────────────────────────────────────
_NAVY   = (0.07, 0.15, 0.31)   # cover background
_GOLD   = (0.84, 0.68, 0.22)   # accent
_RED    = (0.72, 0.11, 0.11)   # severity high
_GREEN  = (0.13, 0.55, 0.13)   # severity ok
_LGREY  = (0.94, 0.94, 0.94)   # table stripe
_MGREY  = (0.60, 0.60, 0.60)   # subtitle text


# ─────────────────────────────────────────────────────────────────────────────
# Figure helpers — matplotlib → PNG bytes (no temp files)
# ─────────────────────────────────────────────────────────────────────────────

def _fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def _figure_odw(odw_result, title: str = "Leverage Gap & Failure Cliff") -> bytes:
    """MI curves + leverage gap + ODW shading for an ODWResult."""
    _require_matplotlib()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from fawp_index.data import E9_2_AGGREGATE_CURVES
    import pandas as pd

    try:
        df = pd.read_csv(E9_2_AGGREGATE_CURVES).sort_values("tau")
    except Exception:
        df = None

    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    fig.suptitle(title, fontsize=11, y=1.01)

    ax1, ax2 = axes

    if df is not None:
        tau = df["tau"].values
        ax1.plot(tau, df["pred_strat_corr"].values, lw=2.2, label="I_pred (corr.)")
        ax1.plot(tau, df["steer_u_corr"].values, lw=2.0, ls="--", label="I_steer u (corr.)")
        ax1.plot(tau, df["steer_xi_corr"].values, lw=1.6, ls="-.", alpha=0.8,
                 label="I_steer xi (corr.)")
        ax2.plot(tau, df["gap_u_corr"].values, lw=2.2, label="Gap (u)")
        ax2.plot(tau, df["gap_xi_corr"].values, lw=1.8, ls="--", label="Gap (xi)")

        fr = df["fail_rate"].values
        ax2r = ax2.twinx()
        ax2r.plot(tau, fr, lw=1.4, ls=":", color="grey", alpha=0.7, label="Fail rate")
        ax2r.set_ylabel("Failure rate", fontsize=8, color="grey")
        ax2r.set_ylim(-0.05, 1.05)
        ax2r.tick_params(axis="y", labelcolor="grey", labelsize=7)

    if odw_result.odw_start is not None:
        for ax in (ax1, ax2):
            ax.axvspan(odw_result.odw_start, odw_result.odw_end,
                       alpha=0.15, color="green", label="ODW")
    if odw_result.tau_h_plus is not None:
        ax1.axvline(odw_result.tau_h_plus, ls=":", lw=1.3, color="steelblue",
                    label=f"tau_h+ = {odw_result.tau_h_plus}")
    if odw_result.tau_f is not None:
        for ax in (ax1, ax2):
            ax.axvline(odw_result.tau_f, ls=":", lw=1.3, color="firebrick",
                       label=f"tau_f = {odw_result.tau_f}")

    ax1.set_ylabel("MI (bits)"); ax1.legend(fontsize=7); ax1.grid(True, alpha=0.25)
    ax2.set_ylabel("Leverage gap (bits)"); ax2.legend(fontsize=7); ax2.grid(True, alpha=0.25)
    ax2.axhline(0, lw=0.8, color="black")
    ax2.set_xlabel("Latency tau")

    fig.text(0.99, 0.01, "fawp-index | Clayton (2026)", ha="right",
             fontsize=6, color="grey", style="italic")
    plt.tight_layout()
    data = _fig_to_bytes(fig)
    plt.close(fig)
    return data


def _figure_alpha2(alpha_result, title: str = "Upgraded FAWP Alpha Index v2.1") -> bytes:
    """alpha_2(tau) and S_m curves for an AlphaV2Result."""
    _require_matplotlib()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tau    = alpha_result.tau_grid
    alpha2 = alpha_result.alpha2
    S_m    = alpha_result.S_m
    R_log  = alpha_result.R_log

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    fig.suptitle(title, fontsize=11, y=1.01)

    ax1.plot(tau, alpha_result.pred_mi_corr, lw=2.0, label="I_pred (corr.)")
    ax1.plot(tau, alpha_result.steer_mi_corr, lw=1.8, ls="--", label="I_steer (corr.)")
    ax1.plot(tau, S_m, lw=1.4, ls=":", alpha=0.8,
             label=f"S_m (m={alpha_result.params.get('m',5)})")
    ax1.axhline(alpha_result.params.get("eta", 1e-4), ls="--", lw=0.8,
                color="grey", alpha=0.6, label="eta threshold")
    if alpha_result.odw_start is not None:
        ax1.axvspan(alpha_result.odw_start, alpha_result.odw_end,
                    alpha=0.12, color="green", label="ODW")
    ax1.set_ylabel("MI (bits)"); ax1.legend(fontsize=7); ax1.grid(True, alpha=0.25)

    ax2.plot(tau, alpha2, lw=2.4, color="crimson", label="alpha_2(tau)")
    ax2.plot(tau, R_log, lw=1.2, ls=":", color="orange", alpha=0.8, label="R_log")
    ax2.fill_between(tau, alpha2, alpha=0.12, color="crimson")
    ax2.axhline(0, lw=0.8, color="black")
    if alpha_result.odw_start is not None:
        ax2.axvspan(alpha_result.odw_start, alpha_result.odw_end,
                    alpha=0.12, color="green")
    ax2.set_xlabel("Latency tau")
    ax2.set_ylabel("alpha_2(tau)"); ax2.legend(fontsize=7); ax2.grid(True, alpha=0.25)

    fig.text(0.99, 0.01, "fawp-index | Clayton (2026)", ha="right",
             fontsize=6, color="grey", style="italic")
    plt.tight_layout()
    data = _fig_to_bytes(fig)
    plt.close(fig)
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Result introspection helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_odw(obj) -> bool:
    return hasattr(obj, "tau_h_plus") and hasattr(obj, "odw_start")

def _is_alpha2(obj) -> bool:
    return hasattr(obj, "alpha2") and hasattr(obj, "S_m")

def _is_fawp(obj) -> bool:
    return hasattr(obj, "fawp_index") and hasattr(obj, "leverage_gap")


def _odw_key_numbers(r) -> list[tuple[str, str]]:
    rows = [
        ("FAWP detected",         "YES" if r.fawp_found else "NO"),
        ("Post-zero horizon",     str(r.tau_h_plus) if r.tau_h_plus is not None else "—"),
        ("Failure cliff (tau_f)", str(r.tau_f) if r.tau_f is not None else "—"),
        ("ODW start",             str(r.odw_start) if r.odw_start is not None else "—"),
        ("ODW end",               str(r.odw_end) if r.odw_end is not None else "—"),
        ("ODW size (steps)",      str(r.odw_size)),
        ("Mean lead to cliff",    f"{r.mean_lead_to_cliff:.1f}" if r.mean_lead_to_cliff else "—"),
        ("Peak leverage gap",     f"{r.peak_gap_bits:.4f} bits"),
        ("Peak gap tau",          str(r.peak_gap_tau) if r.peak_gap_tau is not None else "—"),
    ]
    return rows


def _alpha2_key_numbers(r) -> list[tuple[str, str]]:
    rows = [
        ("FAWP detected",     "YES" if r.fawp_detected else "NO"),
        ("ODW start",         str(r.odw_start) if r.odw_start is not None else "—"),
        ("ODW end",           str(r.odw_end) if r.odw_end is not None else "—"),
        ("Peak alpha_2",      f"{r.peak_alpha2:.4f}"),
        ("Peak alpha_2 tau",  str(r.peak_tau2) if r.peak_tau2 is not None else "—"),
        ("Peak I_pred",       f"{r.pred_mi_corr.max():.4f} bits"),
        ("Param m",           str(r.params.get("m", "—"))),
        ("Param eta",         str(r.params.get("eta", "—"))),
        ("Param epsilon",     str(r.params.get("epsilon", "—"))),
        ("Param kappa",       str(r.params.get("kappa", "—"))),
    ]
    return rows


def _diagnosis_text(result) -> str:
    """Plain-English diagnosis — uses explain() if available, falls back to summary()."""
    try:
        from fawp_index.explain import explain
        return explain(result)
    except Exception:
        pass
    if hasattr(result, "summary"):
        return result.summary()
    return "No diagnosis available."


# ─────────────────────────────────────────────────────────────────────────────
# ReportLab style helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_styles():
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib import colors

    base = getSampleStyleSheet()

    styles = {
        "title_cover": ParagraphStyle(
            "title_cover",
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=28,
            textColor=colors.HexColor("#FFFFFF"),
            alignment=TA_CENTER,
            spaceAfter=12,
        ),
        "subtitle_cover": ParagraphStyle(
            "subtitle_cover",
            fontName="Helvetica",
            fontSize=12,
            leading=16,
            textColor=colors.HexColor("#D4AF37"),
            alignment=TA_CENTER,
            spaceAfter=6,
        ),
        "meta_cover": ParagraphStyle(
            "meta_cover",
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#BBBBBB"),
            alignment=TA_CENTER,
            spaceAfter=4,
        ),
        "section_heading": ParagraphStyle(
            "section_heading",
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=18,
            textColor=colors.HexColor("#0E2550"),
            spaceBefore=18,
            spaceAfter=6,
            borderPad=2,
        ),
        "body": ParagraphStyle(
            "body",
            fontName="Helvetica",
            fontSize=9,
            leading=14,
            spaceBefore=4,
            spaceAfter=4,
            textColor=colors.HexColor("#222222"),
        ),
        "mono": ParagraphStyle(
            "mono",
            fontName="Courier",
            fontSize=8,
            leading=12,
            spaceBefore=2,
            spaceAfter=2,
            textColor=colors.HexColor("#333333"),
            backColor=colors.HexColor("#F4F4F4"),
            leftIndent=8,
        ),
        "caption": ParagraphStyle(
            "caption",
            fontName="Helvetica-Oblique",
            fontSize=7.5,
            leading=11,
            textColor=colors.HexColor("#666666"),
            alignment=TA_CENTER,
            spaceAfter=8,
        ),
        "footer": ParagraphStyle(
            "footer",
            fontName="Helvetica-Oblique",
            fontSize=7,
            leading=10,
            textColor=colors.HexColor("#AAAAAA"),
            alignment=TA_CENTER,
        ),
        "table_header": ParagraphStyle(
            "table_header",
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor("#FFFFFF"),
        ),
        "table_cell": ParagraphStyle(
            "table_cell",
            fontName="Helvetica",
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor("#222222"),
        ),
        "table_cell_bold": ParagraphStyle(
            "table_cell_bold",
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor("#222222"),
        ),
    }
    return styles


def _rl_img(png_bytes: bytes, width_pts: float):
    """PNG bytes → ReportLab Image at given width, height auto."""
    from reportlab.platypus import Image as RLImage
    buf = io.BytesIO(png_bytes)
    img = RLImage(buf)
    aspect = img.imageHeight / img.imageWidth
    img.drawWidth  = width_pts
    img.drawHeight = width_pts * aspect
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Section builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_cover(story, styles, title, author, doi, mode, today_str):
    from reportlab.platypus import Spacer, Paragraph, HRFlowable
    from reportlab.lib import colors
    from reportlab.lib.units import inch

    # Coloured background rectangle drawn via a canvas callback
    class _CoverBackground:
        def __init__(self):
            self._width = None
            self._height = None
        def draw(self): pass
        def wrap(self, aw, ah):
            self._width, self._height = aw, ah
            return aw, 0
        def drawOn(self, canvas, x, y, _sW=0):
            pass

    story.append(Spacer(1, 1.8 * inch))
    story.append(Paragraph(title, styles["title_cover"]))
    story.append(Spacer(1, 0.12 * inch))

    mode_label = "Laboratory Notebook" if mode == "lab" else "Research Report"
    story.append(Paragraph(mode_label, styles["subtitle_cover"]))
    story.append(Spacer(1, 0.3 * inch))

    story.append(Paragraph(author, styles["meta_cover"]))
    story.append(Paragraph(today_str, styles["meta_cover"]))
    if doi:
        story.append(Paragraph(f"doi: {doi}", styles["meta_cover"]))
    story.append(Spacer(1, 0.2 * inch))
    story.append(HRFlowable(width="80%", thickness=1,
                             color=colors.HexColor("#D4AF37"),
                             spaceAfter=10))
    story.append(Paragraph(
        "Generated by fawp-index — Information-Control Exclusion Principle detector",
        styles["footer"],
    ))
    story.append(Paragraph(
        "Clayton (2026) · https://doi.org/10.5281/zenodo.18673949",
        styles["footer"],
    ))
    from reportlab.platypus import PageBreak
    story.append(PageBreak())


def _build_key_numbers(story, styles, rows: list[tuple[str, str]], section_title: str):
    from reportlab.platypus import Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch

    story.append(Paragraph(section_title, styles["section_heading"]))

    fawp_row = next((r for r in rows if r[0] == "FAWP detected"), None)
    if fawp_row:
        detected = fawp_row[1] == "YES"
        colour = colors.HexColor("#1B7D1B") if detected else colors.HexColor("#AA1111")
        label  = "FAWP DETECTED" if detected else "FAWP NOT DETECTED"
        story.append(Paragraph(
            f'<font color="{"#1B7D1B" if detected else "#AA1111"}"><b>{label}</b></font>',
            styles["body"],
        ))
        story.append(Spacer(1, 6))

    table_data = [
        [
            Paragraph("Quantity", styles["table_header"]),
            Paragraph("Value", styles["table_header"]),
        ]
    ]
    for k, v in rows:
        table_data.append([
            Paragraph(k, styles["table_cell"]),
            Paragraph(v, styles["table_cell_bold"]),
        ])

    col_widths = [3.2 * inch, 2.4 * inch]
    tbl = Table(table_data, colWidths=col_widths, repeatRows=1)

    row_styles = []
    for i in range(1, len(table_data)):
        bg = colors.HexColor("#F4F4F4") if i % 2 == 0 else colors.white
        row_styles.append(("BACKGROUND", (0, i), (-1, i), bg))

    tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), colors.HexColor("#0E2550")),
        ("ROWBACKGROUND",(0, 1), (-1, -1), colors.white),
        ("GRID",         (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ] + row_styles))

    story.append(tbl)
    story.append(Spacer(1, 10))


def _build_diagnosis(story, styles, text: str, mode: str):
    from reportlab.platypus import Paragraph, Spacer

    story.append(Paragraph("Plain-English Diagnosis", styles["section_heading"]))

    # Split on newlines, render each line (keep mono for the box-drawing lines)
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            story.append(Spacer(1, 4))
            continue
        if stripped.startswith("=") or stripped.startswith("-"):
            continue  # skip pure separator lines
        # Severity markers stay in body style
        style = styles["mono"] if stripped.startswith("|") else styles["body"]
        # Escape XML-special characters for ReportLab
        safe = stripped.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        story.append(Paragraph(safe, style))

    story.append(Spacer(1, 8))


def _build_figures(story, styles, figures: list[tuple[str, bytes]]):
    from reportlab.platypus import Paragraph, Spacer, PageBreak
    from reportlab.lib.units import inch

    story.append(Paragraph("Figures", styles["section_heading"]))
    for caption, png_bytes in figures:
        img = _rl_img(png_bytes, width_pts=5.8 * inch)
        story.append(img)
        story.append(Paragraph(caption, styles["caption"]))
        story.append(Spacer(1, 14))


def _build_methods(story, styles):
    from reportlab.platypus import Paragraph, Spacer

    story.append(Paragraph("Methods", styles["section_heading"]))

    paras = [
        (
            "Mutual Information Estimation",
            "Predictive and steering mutual information curves are estimated along a delay "
            "grid tau using correlation-based estimators (Gaussian MI approximation). "
            "Prediction is measured in both pooled and stratified forms; stratified "
            "prediction is used as the primary detector input to control time-confounding."
        ),
        (
            "Conservative Null Correction",
            "Raw MI is almost never exactly zero in finite samples. Each curve is "
            "corrected by subtracting a conservative null floor: the maximum of the "
            "q_beta quantile of shuffle-null and autocorrelation-preserving shift-null "
            "distributions, with beta = 0.99. Any residual MI that does not exceed this "
            "floor is treated as zero."
        ),
        (
            "Operational Detection Window",
            "The post-zero agency horizon tau_h is the first delay at or after tau = 1 "
            "at which corrected steering MI falls to or below the threshold epsilon. "
            "The functional failure cliff tau_f is the first delay at which the failure "
            "rate exceeds the cliff criterion (default 0.99). The Operational Detection "
            "Window (ODW) is the persistent pre-cliff interval in which corrected "
            "predictive MI exceeds epsilon while corrected steering MI does not, "
            "subject to a m-of-n persistence rule (default 3-of-4)."
        ),
        (
            "Upgraded Alpha Index v2.1",
            "The alpha_2(tau) index (Clayton 2026) strengthens detection via: "
            "(i) null-quantile floor subtraction (shuffle + shift, q_0.99); "
            "(ii) post-correction buffers eta and epsilon near 10^-4 bits; "
            "(iii) a robust stability window S_m(tau) = min over k=0..m of I_pred(tau-k) "
            "(default m = 5) to suppress single-step spikes; "
            "(iv) a scale-invariant log-slope resonance term R_log(tau) that measures "
            "relative growth of predictability without finite-difference noise amplification."
        ),
        (
            "Baseline Dynamical Setting",
            "The E9 confirmation suite uses an unstable delayed-feedback controller with "
            "gain a = 1.02, controller gain K = 0.8, and prediction horizon delta = 20. "
            "The unstable gain ensures that delayed intervention becomes increasingly costly. "
            "The detection question is whether intervention remains operationally effective "
            "as latency increases, not whether the system is globally controllable in principle."
        ),
    ]

    for heading, body in paras:
        story.append(Paragraph(f"<b>{heading}</b>", styles["body"]))
        story.append(Paragraph(body, styles["body"]))
        story.append(Spacer(1, 6))


def _build_citation(story, styles, doi: Optional[str], title: str, author: str, today_str: str):
    from reportlab.platypus import Paragraph, Spacer

    story.append(Paragraph("Citation", styles["section_heading"]))

    doi_line = doi or "10.5281/zenodo.18673949"
    year = today_str[:4]

    bibtex = (
        "@software{fawp_index,\n"
        f"  author  = {{{author}}},\n"
        f"  title   = {{fawp-index: {title}}},\n"
        f"  year    = {{{year}}},\n"
        f"  doi     = {{{doi_line}}},\n"
        "  url     = {https://github.com/DrRalphClayton/fawp-index},\n"
        "  note    = {Python package, version 0.9.0}\n"
        "}"
    )

    story.append(Paragraph("BibTeX:", styles["body"]))
    for line in bibtex.split("\n"):
        safe = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        story.append(Paragraph(safe, styles["mono"]))

    story.append(Spacer(1, 10))
    story.append(Paragraph(
        f"DOI: {doi_line}", styles["body"]
    ))
    story.append(Paragraph(
        "Book: Forecasting Without Power (Deluxe 2026) — "
        "https://www.amazon.com/dp/B0GS1ZVNM7/",
        styles["body"],
    ))
    story.append(Paragraph(
        "GitHub: https://github.com/DrRalphClayton/fawp-index",
        styles["body"],
    ))


# ─────────────────────────────────────────────────────────────────────────────
# Cover page with coloured background (canvas callback)
# ─────────────────────────────────────────────────────────────────────────────

def _make_cover_canvas(title, author, doi, mode, today_str):
    """Return an onFirstPage callback that draws the navy cover background."""
    def _on_cover(canvas, doc):
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        w, h = doc.pagesize

        # Navy background
        canvas.saveState()
        canvas.setFillColor(colors.HexColor("#0E2550"))
        canvas.rect(0, 0, w, h, fill=1, stroke=0)

        # Gold accent bar
        canvas.setFillColor(colors.HexColor("#D4AF37"))
        canvas.rect(0, h * 0.42, w, 3, fill=1, stroke=0)
        canvas.rect(0, h * 0.42 - 6, w, 1, fill=1, stroke=0)

        # FAWP watermark text (very faint)
        canvas.setFont("Helvetica-Bold", 80)
        canvas.setFillColor(colors.HexColor("#FFFFFF"))
        canvas.setFillAlpha(0.04)
        canvas.drawCentredString(w / 2, h * 0.15, "FAWP")
        canvas.setFillAlpha(1.0)
        canvas.restoreState()

    return _on_cover


def _make_later_canvas():
    """Return an onLaterPages callback for header/footer on non-cover pages."""
    def _on_later(canvas, doc):
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        w, h = doc.pagesize

        # Top rule
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#0E2550"))
        canvas.setLineWidth(1)
        canvas.line(doc.leftMargin, h - doc.topMargin + 8,
                    w - doc.rightMargin, h - doc.topMargin + 8)

        # Footer rule + text
        canvas.setStrokeColor(colors.HexColor("#CCCCCC"))
        canvas.setLineWidth(0.5)
        canvas.line(doc.leftMargin, doc.bottomMargin - 8,
                    w - doc.rightMargin, doc.bottomMargin - 8)
        canvas.setFont("Helvetica-Oblique", 7)
        canvas.setFillColor(colors.HexColor("#AAAAAA"))
        canvas.drawString(doc.leftMargin, doc.bottomMargin - 18,
                          "fawp-index | Clayton (2026) | doi:10.5281/zenodo.18673949")
        canvas.drawRightString(w - doc.rightMargin, doc.bottomMargin - 18,
                               f"Page {doc.page}")
        canvas.restoreState()

    return _on_later


# ─────────────────────────────────────────────────────────────────────────────
# Main builder
# ─────────────────────────────────────────────────────────────────────────────

class FAWPReport:
    """
    Builder for FAWP PDF reports.

    Parameters
    ----------
    mode : 'report' or 'lab'
        'report' — polished, citable, with methods + citation pages.
        'lab'    — personal notebook style, more verbose, no citation.
    title : str, optional
        Report title. Auto-generated if not provided.
    author : str, optional
        Author name. Default 'Ralph Clayton'.
    doi : str, optional
        DOI to print on the cover and citation page.
    include_figures : bool
        Whether to render and embed matplotlib figures. Default True.
    include_methods : bool
        Whether to include the methods section. Default True (report mode).
    """

    def __init__(
        self,
        mode: str = "report",
        title: Optional[str] = None,
        author: str = "Ralph Clayton",
        doi: Optional[str] = "10.5281/zenodo.18673949",
        include_figures: bool = True,
        include_methods: bool = True,
    ):
        if mode not in ("report", "lab"):
            raise ValueError("mode must be 'report' or 'lab'")
        self.mode = mode
        self._title = title
        self.author = author
        self.doi = doi
        self.include_figures = include_figures
        self.include_methods = include_methods

    def build(
        self,
        result: Any,
        output_path: Union[str, Path],
    ) -> Path:
        """
        Build and write the PDF.

        Parameters
        ----------
        result : ODWResult, AlphaV2Result, FAWPResult, or dict
            One result or a dict of named results.
        output_path : str or Path
            Where to write the PDF.

        Returns
        -------
        Path — the written file path.
        """
        _require_reportlab()

        from reportlab.platypus import SimpleDocTemplate, PageBreak, Spacer
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import inch

        output_path = Path(output_path)
        today_str = date.today().isoformat()

        # Normalise to dict
        if isinstance(result, dict):
            results = result
        else:
            key = ("odw" if _is_odw(result)
                   else "alpha" if _is_alpha2(result)
                   else "fawp")
            results = {key: result}

        title = self._title or self._auto_title(results)

        # Build story
        styles = _make_styles()
        story = []

        # ── 1. Cover ────────────────────────────────────────────────────────
        _build_cover(story, styles, title, self.author, self.doi, self.mode, today_str)

        # ── 2. Key numbers (one section per result) ─────────────────────────
        for name, res in results.items():
            if _is_odw(res):
                rows = _odw_key_numbers(res)
                _build_key_numbers(story, styles, rows,
                                   f"Key Numbers — ODW ({name})")
            elif _is_alpha2(res):
                rows = _alpha2_key_numbers(res)
                _build_key_numbers(story, styles, rows,
                                   f"Key Numbers — Alpha Index v2.1 ({name})")

        story.append(PageBreak())

        # ── 3. Diagnosis ────────────────────────────────────────────────────
        for name, res in results.items():
            diag = _diagnosis_text(res)
            if diag:
                _build_diagnosis(story, styles, diag, self.mode)

        story.append(PageBreak())

        # ── 4. Figures ──────────────────────────────────────────────────────
        if self.include_figures:
            try:
                _require_matplotlib()
                figures = []
                for name, res in results.items():
                    if _is_odw(res):
                        png = _figure_odw(res, f"Leverage gap and failure cliff — {name}")
                        figures.append((
                            f"Figure: MI curves, leverage gap, and ODW for {name}. "
                            "Green shading = Operational Detection Window. "
                            "Dotted lines = tau_h+ (blue) and tau_f (red).",
                            png,
                        ))
                    elif _is_alpha2(res):
                        png = _figure_alpha2(res, f"Alpha Index v2.1 — {name}")
                        figures.append((
                            f"Figure: alpha_2(tau) index for {name}. "
                            "Top panel: null-corrected MI curves and S_m stability window. "
                            "Bottom panel: alpha_2(tau) with log-slope resonance R_log.",
                            png,
                        ))
                if figures:
                    _build_figures(story, styles, figures)
                    story.append(PageBreak())
            except ImportError:
                story.append(Spacer(1, 0.2 * inch))

        # ── 5. Methods ──────────────────────────────────────────────────────
        if self.include_methods and self.mode == "report":
            _build_methods(story, styles)
            story.append(PageBreak())

        # ── 6. Citation ─────────────────────────────────────────────────────
        if self.mode == "report":
            _build_citation(story, styles, self.doi, title, self.author, today_str)

        # ── Assemble ────────────────────────────────────────────────────────
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            leftMargin=0.9 * inch,
            rightMargin=0.9 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
            title=title,
            author=self.author,
            subject="FAWP Analysis Report",
            creator="fawp-index",
        )

        on_cover = _make_cover_canvas(title, self.author, self.doi, self.mode, today_str)
        on_later = _make_later_canvas()

        doc.build(story, onFirstPage=on_cover, onLaterPages=on_later)
        return output_path

    @staticmethod
    def _auto_title(results: dict) -> str:
        parts = []
        for name, res in results.items():
            if _is_odw(res):
                parts.append(f"ODW Analysis ({name})")
            elif _is_alpha2(res):
                parts.append(f"Alpha Index v2.1 ({name})")
            else:
                parts.append(f"FAWP Analysis ({name})")
        return " | ".join(parts) if parts else "FAWP Analysis Report"


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(
    result: Any,
    output_path: Union[str, Path],
    *,
    title: Optional[str] = None,
    mode: str = "report",
    author: str = "Ralph Clayton",
    doi: Optional[str] = "10.5281/zenodo.18673949",
    include_figures: bool = True,
    include_methods: bool = True,
) -> Path:
    """
    Generate a PDF report from any FAWP result object.

    Parameters
    ----------
    result : ODWResult, AlphaV2Result, FAWPResult, or dict
        A single result or a dict of named results.
    output_path : str or Path
        Where to write the PDF.
    title : str, optional
        Report title. Auto-generated if not provided.
    mode : 'report' or 'lab'
        'report' — polished, citable, with methods + citation (default).
        'lab'    — personal notebook style, no citation page.
    author : str
        Author name printed on the cover. Default 'Ralph Clayton'.
    doi : str, optional
        DOI printed on the cover and citation page.
    include_figures : bool
        Embed matplotlib figures. Requires matplotlib. Default True.
    include_methods : bool
        Include the methods section. Default True (report mode only).

    Returns
    -------
    Path — the written file.

    Examples
    --------
    Quick start from bundled E9.2 data::

        from fawp_index import ODWDetector, FAWPAlphaIndexV2
        from fawp_index.report import generate_report

        odw   = ODWDetector.from_e9_2_data()
        alpha = FAWPAlphaIndexV2.from_e9_2_data()

        # Polished citable report
        generate_report(
            {"odw": odw, "alpha": alpha},
            "e9_2_report.pdf",
            title="E9.2 Steering Definition Comparison",
        )

        # Lab notebook
        generate_report(odw, "lab.pdf", mode="lab")

    Raises
    ------
    ImportError
        If reportlab is not installed.
        Install with: pip install fawp-index[report]
    """
    builder = FAWPReport(
        mode=mode,
        title=title,
        author=author,
        doi=doi,
        include_figures=include_figures,
        include_methods=include_methods,
    )
    return builder.build(result, output_path)
