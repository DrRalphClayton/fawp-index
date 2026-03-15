"""
fawp_index.viz.plotly_plots — Interactive Plotly figures.

Requires: pip install "fawp-index[plotly]"

Functions
---------
plot_mi_curves(result_or_window)
    Interactive pred/steer MI curves with ODW shading and epsilon line.

plot_regime_score(scan)
    Bar chart of regime score over time, coloured by FAWP flag.

plot_leverage_gap_bar(window)
    Per-tau leverage gap bar chart with ODW highlighted.

plot_leaderboard(lb)
    Grouped bar chart of the four leaderboard categories.

plot_heatmap(watchlist_result, metric="score")
    Assets × timeframes heatmap (score or gap).

All functions return a plotly Figure object. Call .show() or pass to
Streamlit with st.plotly_chart(fig, use_container_width=True).

Ralph Clayton (2026) · doi:10.5281/zenodo.18673949
"""

from typing import Optional

_DARK = dict(
    paper_bgcolor="#07101E",
    plot_bgcolor="#0D1729",
    font_color="#7A90B8",
    font_family="JetBrains Mono, monospace",
)
_AMBER  = "#D4AF37"
_CRIMSON = "#C0111A"
_BLUE   = "#4A7FCC"
_GREEN  = "#1DB954"
_GRID   = "#182540"


def _base_layout(**kwargs) -> dict:
    layout = dict(
        paper_bgcolor=_DARK["paper_bgcolor"],
        plot_bgcolor=_DARK["plot_bgcolor"],
        font=dict(color=_DARK["font_color"], family=_DARK["font_family"], size=11),
        xaxis=dict(gridcolor=_GRID, zerolinecolor=_GRID, linecolor=_GRID),
        yaxis=dict(gridcolor=_GRID, zerolinecolor=_GRID, linecolor=_GRID),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=_GRID, borderwidth=1),
        margin=dict(l=50, r=30, t=50, b=40),
        hovermode="x unified",
    )
    layout.update(kwargs)
    return layout


def plot_mi_curves(obj, epsilon: float = 0.01, title: Optional[str] = None):
    """
    Interactive MI curves: pred MI, steer MI, leverage gap fill, ODW shading.

    Parameters
    ----------
    obj : MarketWindowResult or any object with .tau, .pred_mi, .steer_mi, .odw_result
    epsilon : float
        Epsilon threshold line value.
    title : str, optional

    Returns
    -------
    plotly.graph_objects.Figure
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError('pip install "fawp-index[plotly]"')

    import numpy as np

    tau       = obj.tau
    pred_mi   = obj.pred_mi
    steer_mi  = obj.steer_mi
    odw       = obj.odw_result

    fig = go.Figure()

    # Leverage gap fill
    fig.add_trace(go.Scatter(
        x=list(tau) + list(tau[::-1]),
        y=list(np.maximum(pred_mi, steer_mi)) + list(np.minimum(pred_mi, steer_mi)[::-1]),
        fill="toself",
        fillcolor="rgba(212,175,55,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Leverage gap",
        hoverinfo="skip",
    ))

    # Pred MI
    fig.add_trace(go.Scatter(
        x=tau, y=pred_mi,
        mode="lines+markers",
        line=dict(color=_AMBER, width=2),
        marker=dict(size=4),
        name="Pred MI",
    ))

    # Steer MI
    fig.add_trace(go.Scatter(
        x=tau, y=steer_mi,
        mode="lines+markers",
        line=dict(color=_BLUE, width=2, dash="dash"),
        marker=dict(size=4, symbol="square"),
        name="Steer MI",
    ))

    # Epsilon line
    fig.add_hline(
        y=epsilon, line_dash="dot", line_color="#3A4E70",
        annotation_text=f"ε={epsilon}", annotation_position="bottom right",
        annotation_font_color="#3A4E70",
    )

    # ODW shading
    if odw.odw_start is not None:
        fig.add_vrect(
            x0=odw.odw_start - 0.5, x1=odw.odw_end + 0.5,
            fillcolor="rgba(192,17,26,0.12)", layer="below",
            line_width=0,
            annotation_text=f"ODW {odw.odw_start}–{odw.odw_end}",
            annotation_position="top left",
            annotation_font_color=_CRIMSON,
        )

    # tau_h+ vertical line
    if odw.tau_h_plus is not None:
        fig.add_vline(
            x=odw.tau_h_plus, line_dash="dot", line_color=_BLUE,
            annotation_text=f"τ⁺ₕ={odw.tau_h_plus}",
            annotation_font_color=_BLUE,
        )

    fig.update_layout(
        title=title or "MI curves — Pred vs Steer",
        xaxis_title="τ (steering lag)",
        yaxis_title="MI (bits)",
        yaxis_rangemode="tozero",
        **_base_layout(),
    )
    return fig


def plot_regime_score(scan, title: Optional[str] = None):
    """
    Regime score bar chart over time, crimson for FAWP windows, green otherwise.

    Parameters
    ----------
    scan : MarketScanSeries
    title : str, optional

    Returns
    -------
    plotly.graph_objects.Figure
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError('pip install "fawp-index[plotly]"')

    dates  = [str(d.date()) for d in scan.dates]
    scores = scan.regime_scores
    colors = [_CRIMSON if f else _GREEN for f in scan.fawp_flags]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dates, y=scores,
        marker_color=colors,
        marker_line_width=0,
        name="Regime score",
        hovertemplate="%{x}<br>Score: %{y:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title=title or "Regime score over time",
        xaxis_title="Date",
        yaxis_title="Score",
        bargap=0,
        **_base_layout(),
    )
    return fig


def plot_leverage_gap_bar(window, tau_max: int = 40, title: Optional[str] = None):
    """
    Per-tau leverage gap bars with ODW highlighted in crimson.

    Parameters
    ----------
    window : MarketWindowResult
    tau_max : int
    title : str, optional

    Returns
    -------
    plotly.graph_objects.Figure
    """
    try:
        import plotly.graph_objects as go
        import numpy as np
    except ImportError:
        raise ImportError('pip install "fawp-index[plotly]"')

    gap    = list(map(float, __import__("numpy").maximum(0, window.pred_mi - window.steer_mi)))
    odw    = window.odw_result
    colors = []
    for t in window.tau:
        if odw.odw_start is not None and odw.odw_start <= t <= odw.odw_end:
            colors.append(_CRIMSON)
        else:
            colors.append("#2A4070")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(window.tau), y=gap,
        marker_color=colors, marker_line_width=0,
        name="Leverage gap",
        hovertemplate="τ=%{x}<br>Gap: %{y:.4f} bits<extra></extra>",
    ))

    fig.update_layout(
        title=title or f"Leverage gap by τ — {window.date.date()}",
        xaxis_title="τ",
        yaxis_title="Gap (bits)",
        bargap=0.1,
        **_base_layout(),
    )
    return fig


def plot_heatmap(watchlist_result, metric: str = "score", title: Optional[str] = None):
    """
    Assets × timeframes heatmap.

    Parameters
    ----------
    watchlist_result : WatchlistResult
    metric : 'score' or 'gap'
    title : str, optional

    Returns
    -------
    plotly.graph_objects.Figure
    """
    try:
        import plotly.graph_objects as go
        import numpy as np
    except ImportError:
        raise ImportError('pip install "fawp-index[plotly]"')

    valid    = [a for a in watchlist_result.assets if not a.error]
    tickers  = sorted(set(a.ticker    for a in valid))
    tfs      = sorted(set(a.timeframe for a in valid))

    mat = [[None] * len(tfs) for _ in tickers]
    for a in valid:
        i = tickers.index(a.ticker)
        j = tfs.index(a.timeframe)
        mat[i][j] = round(a.latest_score if metric == "score" else a.peak_gap_bits, 4)

    text = [[f"{v:.4f}" if v is not None else "—" for v in row] for row in mat]

    fig = go.Figure(go.Heatmap(
        z=mat, x=tfs, y=tickers,
        text=text, texttemplate="%{text}",
        colorscale="RdYlGn_r",
        reversescale=False,
        colorbar=dict(title=metric, tickfont=dict(color="#7A90B8")),
        hovertemplate="%{y} [%{x}]<br>" + metric + ": %{z:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title=title or f"FAWP {metric} heatmap",
        xaxis_title="Timeframe",
        yaxis_title="Asset",
        **_base_layout(),
    )
    return fig


def plot_leaderboard(lb, title: Optional[str] = None):
    """
    Grouped bar chart of leaderboard categories.

    Parameters
    ----------
    lb : Leaderboard
    title : str, optional

    Returns
    -------
    plotly.graph_objects.Figure
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError('pip install "fawp-index[plotly]"')

    categories = [
        ("Top FAWP",          lb.top_fawp,           _CRIMSON),
        ("Rising Risk",       lb.rising_risk,         _AMBER),
        ("Collapsing Control",lb.collapsing_control,  "#4A7FCC"),
        ("Strongest ODW",     lb.strongest_odw,       "#8A50C8"),
    ]

    fig = go.Figure()
    for cat_name, entries, color in categories:
        if not entries:
            continue
        labels = [f"{e.ticker} [{e.timeframe}]" for e in entries[:6]]
        scores = [e.score for e in entries[:6]]
        fig.add_trace(go.Bar(
            name=cat_name, x=labels, y=scores,
            marker_color=color, marker_line_width=0,
            hovertemplate="%{x}<br>Score: %{y:.4f}<extra></extra>",
        ))

    fig.update_layout(
        title=title or "FAWP leaderboard",
        barmode="group",
        xaxis_title="Asset",
        yaxis_title="Score",
        **_base_layout(),
    )
    return fig
