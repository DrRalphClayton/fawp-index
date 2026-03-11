"""
fawp-index: Visualization
Auto-generates publication-ready leverage gap plots.
Ralph Clayton (2026) — DOI: https://doi.org/10.5281/zenodo.18673949
"""

import numpy as np

def plot_leverage_gap(
    result,
    title: str = None,
    figsize: tuple = (12, 6),
    save_path: str = None,
    show: bool = True,
    failure_rate: np.ndarray = None,
):
    """
    Generate the leverage gap plot from a FAWPResult.

    The signature plot of FAWP research:
    - Orange: predictive coupling (persists)
    - Blue dashed: steering coupling (collapses)
    - Shaded: FAWP regime (leverage gap)
    - Gray dotted: failure rate (optional)

    Parameters
    ----------
    result : FAWPResult
        Output from FAWPAlphaIndex.compute()
    title : str, optional
        Plot title. Auto-generated if None.
    figsize : tuple
        Figure size (width, height).
    save_path : str, optional
        If provided, saves figure to this path.
    show : bool
        Whether to call plt.show().
    failure_rate : array, optional
        Optional failure rate to plot on right axis.

    Returns
    -------
    fig, axes
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting.\n"
            "Install it with: pip install matplotlib"
        )

    tau = result.tau
    pred = result.pred_mi_raw
    steer = result.steer_mi_raw
    alpha = result.alpha_index
    in_fawp = result.in_fawp

    if title is None:
        title = (
            f"FAWP Leverage Gap — Prediction Persists After Control Collapses\n"
            f"Agency Horizon τ_h={result.tau_h}  |  "
            f"Peak Alpha={result.peak_alpha:.4f} at τ={result.peak_tau}"
        )

    has_failure = failure_rate is not None and len(failure_rate) == len(tau)

    if has_failure:
        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = ax1.twinx()
    else:
        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = None

    # ── Shaded FAWP region ────────────────────────────────────────────────────
    for i, t in enumerate(tau):
        if in_fawp[i]:
            ax1.axvspan(t - 0.5, t + 0.5, alpha=0.12, color='steelblue', zorder=0)

    # ── Leverage gap fill ─────────────────────────────────────────────────────
    ax1.fill_between(
        tau,
        steer,
        pred,
        where=(pred > steer),
        alpha=0.15,
        color='orange',
        label='Leverage Gap (FAWP Region)',
        zorder=1,
    )

    # ── Steering coupling ─────────────────────────────────────────────────────
    ax1.plot(
        tau, steer,
        'b--', linewidth=2, marker='o', markersize=4,
        label=r'Steering Coupling $I(A_t; O_{t+\tau+1})$',
        zorder=3,
    )

    # ── Predictive coupling ───────────────────────────────────────────────────
    ax1.plot(
        tau, pred,
        color='darkorange', linewidth=2.5, marker='s', markersize=4,
        label=r'Predictive Coupling $I(D_t; X_{t+\Delta})$',
        zorder=4,
    )

    # ── Agency horizon ────────────────────────────────────────────────────────
    if result.tau_h is not None:
        ax1.axvline(
            result.tau_h, color='gray', linestyle='--', linewidth=1.5,
            label=f'Agency Horizon τ_h={result.tau_h}', zorder=2,
        )

    # ── Alpha index (secondary line) ──────────────────────────────────────────
    if alpha.max() > 0:
        ax1.plot(
            tau, alpha,
            'g:', linewidth=1.5,
            label=f'Alpha Index v2.1 (peak={result.peak_alpha:.3f})',
            zorder=3,
        )

    # ── Failure rate ──────────────────────────────────────────────────────────
    if has_failure and ax2 is not None:
        ax2.plot(
            tau, failure_rate,
            'gray', linestyle=':', linewidth=1.5,
            label='Failure Rate', zorder=2,
        )
        ax2.set_ylabel('Failure Probability', color='gray', fontsize=11)
        ax2.set_ylim(0, 1.1)
        ax2.tick_params(axis='y', labelcolor='gray')

    # ── Formatting ────────────────────────────────────────────────────────────
    ax1.set_xlabel('Latency τ (time steps)', fontsize=12)
    ax1.set_ylabel('Mutual Information (bits)', fontsize=12)
    ax1.set_title(title, fontsize=11, pad=12)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(tau[0] - 0.5, tau[-1] + 0.5)
    ax1.set_ylim(bottom=0)

    # ── Annotation ───────────────────────────────────────────────────────────
    if result.in_fawp.any() and result.peak_tau is not None:
        peak_pred = pred[tau == result.peak_tau]
        if len(peak_pred) > 0:
            ax1.annotate(
                f'FAWP\nα={result.peak_alpha:.3f}',
                xy=(result.peak_tau, peak_pred[0]),
                xytext=(result.peak_tau + 0.8, peak_pred[0] + 0.05),
                fontsize=8,
                color='darkorange',
                arrowprops=dict(arrowstyle='->', color='darkorange', lw=1.2),
            )

    # ── Citation watermark ────────────────────────────────────────────────────
    fig.text(
        0.99, 0.01,
        'fawp-index | Clayton (2026) doi:10.5281/zenodo.18673949',
        ha='right', va='bottom', fontsize=7, color='gray', style='italic',
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    if show:
        plt.show()

    return fig, ax1


def plot_alpha_index(
    result,
    title: str = None,
    figsize: tuple = (10, 4),
    save_path: str = None,
    show: bool = True,
):
    """
    Plot the FAWP Alpha Index v2.1 as a bar chart.
    Highlights FAWP-detected tau values in orange.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("pip install matplotlib")

    tau = result.tau
    alpha = result.alpha_index
    colors = ['darkorange' if f else 'steelblue' for f in result.in_fawp]

    if title is None:
        title = f"FAWP Alpha Index v2.1 | Peak={result.peak_alpha:.4f} at τ={result.peak_tau}"

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(tau, alpha, color=colors, edgecolor='white', linewidth=0.5)

    if result.tau_h is not None:
        ax.axvline(result.tau_h, color='gray', linestyle='--',
                   linewidth=1.5, label=f'τ_h={result.tau_h}')

    ax.set_xlabel('Latency τ', fontsize=11)
    ax.set_ylabel('Alpha Index', fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='darkorange', label='FAWP regime'),
        Patch(facecolor='steelblue', label='No FAWP'),
    ]
    ax.legend(handles=legend_elements, fontsize=9)

    fig.text(
        0.99, 0.01,
        'fawp-index | Clayton (2026)',
        ha='right', va='bottom', fontsize=7, color='gray', style='italic',
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()

    return fig, ax
