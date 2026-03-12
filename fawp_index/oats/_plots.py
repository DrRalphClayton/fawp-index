"""fawp_index.oats._plots — Internal plotting for OATS module."""

import numpy as np


def plot_horizon(result, save_path=None, show=True):
    """Plot analytic MI curve with horizon marker."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("pip install matplotlib")

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(result.tau_grid, result.mi, color='darkorange', linewidth=2.5,
            label=r"$I(\tau) = \frac{1}{2}\log_2\left(1 + \frac{P}{\sigma^2(\tau)}\right)$")
    ax.axhline(result.epsilon, color='gray', linestyle='--', linewidth=1.5,
               label=f"Threshold ε = {result.epsilon}")

    if np.isfinite(result.tau_h) and result.tau_h > 0:
        ax.axvline(result.tau_h, color='steelblue', linestyle='--', linewidth=1.8,
                   label=f"τ_h = {result.tau_h:.1f}")
        ax.annotate(f"τ_h = {result.tau_h:.1f}",
                    xy=(result.tau_h, result.epsilon),
                    xytext=(result.tau_h + result.tau_grid[-1] * 0.03, result.epsilon * 1.5),
                    fontsize=9, color='steelblue')

    ax.fill_between(result.tau_grid, 0, result.mi,
                    where=(result.mi >= result.epsilon),
                    alpha=0.12, color='darkorange', label="Agency region")

    ax.set_xlabel("Latency τ", fontsize=11)
    ax.set_ylabel("Mutual Information I(τ) (bits)", fontsize=11)
    ax.set_title(
        f"Agency Horizon — Analytic Model\n"
        f"P={result.P}, σ0²={result.sigma0_sq}, α={result.alpha}, ε={result.epsilon}",
        fontsize=10
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.text(0.99, 0.01, 'fawp-index | Clayton (2026) doi:10.5281/zenodo.18663547',
             ha='right', va='bottom', fontsize=7, color='gray', style='italic')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    return fig, ax


def plot_sweep_scaling(sweep_result, save_path=None, show=True):
    """Log-log scatter: MC vs theory horizons for E1 COVERED points."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("pip install matplotlib")

    df = sweep_result.df
    has_mc = sweep_result.has_mc and 'tau_h_mc' in df.columns

    fig, axes = plt.subplots(1, 2 if has_mc else 1, figsize=(12 if has_mc else 7, 5))
    if not has_mc:
        axes = [axes]

    # Panel 1: Horizon vs P for different alpha
    ax = axes[0]
    if 'P' in df.columns and 'alpha' in df.columns:
        alphas = df['alpha'].unique()
        cmap = plt.cm.viridis(np.linspace(0, 1, len(alphas)))
        for i, a in enumerate(sorted(alphas)):
            sub = df[df['alpha'] == a].sort_values('P')
            finite = sub[np.isfinite(sub['tau_h_theory'])]
            if len(finite):
                ax.loglog(finite['P'], finite['tau_h_theory'], 'o-',
                          color=cmap[i], linewidth=1.8, markersize=5,
                          label=f"α={a:.4g}")
    ax.set_xlabel("Signal power P", fontsize=10)
    ax.set_ylabel("Agency horizon τ_h", fontsize=10)
    ax.set_title("E1: Horizon scaling with P", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    # Panel 2: MC vs theory scatter (E1 COVERED)
    if has_mc:
        ax2 = axes[1]
        covered = df[df.get('status', df.get('status', 'COVERED')) == 'COVERED'].dropna(
            subset=['tau_h_theory', 'tau_h_mc']
        )
        if len(covered):
            ax2.loglog(covered['tau_h_theory'], covered['tau_h_mc'], 'o',
                       color='darkorange', alpha=0.6, markersize=6)
            lims = [min(covered['tau_h_theory'].min(), covered['tau_h_mc'].min()),
                    max(covered['tau_h_theory'].max(), covered['tau_h_mc'].max())]
            ax2.loglog(lims, lims, 'k--', linewidth=1.5, label='y = x')
            ax2.set_xlabel("Analytic τ_h", fontsize=10)
            ax2.set_ylabel("Monte Carlo τ_h", fontsize=10)
            ax2.set_title(
                f"E1: MC vs analytic ({len(covered)} COVERED runs)\n"
                f"Mean rel. error = {covered['error_relative'].mean():.4f}",
                fontsize=10
            )
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3, which='both')

    fig.text(0.99, 0.01, 'fawp-index | Clayton (2026) doi:10.5281/zenodo.18663547',
             ha='right', va='bottom', fontsize=7, color='gray', style='italic')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    return fig, axes


def plot_robustness(result, save_path=None, show=True):
    """Plot E4 distributional robustness: MI vs tau for all three scenarios."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("pip install matplotlib")

    colors = {'Baseline (Gaussian/Gaussian)': 'steelblue',
              'Bounded Agent (Uniform/Gaussian)': 'darkorange',
              'Heavy Tail Noise (Gaussian/Student-t)': 'green'}
    labels = {'Baseline (Gaussian/Gaussian)': 'Baseline (Gaussian/Gaussian)',
              'Bounded Agent (Uniform/Gaussian)': 'Bounded Agent (Uniform/Gaussian)',
              'Heavy Tail Noise (Gaussian/Student-t)': 'Heavy Tail (Gaussian/Student-t)'}

    fig, ax = plt.subplots(figsize=(10, 5))
    tau = result.tau_grid

    for name, d in result.scenarios.items():
        color = colors.get(name, 'gray')
        label = labels.get(name, name)
        ax.plot(tau, d['mean'], linewidth=2.2, color=color, label=label)
        ax.fill_between(tau, d['ci_lo'], d['ci_hi'], alpha=0.15, color=color)
        if result.tau_h.get(name) is not None:
            ax.axvline(result.tau_h[name], color=color, linestyle=':', linewidth=1.2, alpha=0.7)

    ax.set_xlabel("Latency τ", fontsize=11)
    ax.set_ylabel("Mutual Information I(τ) (bits)", fontsize=11)
    ax.set_title("E4: Distributional Robustness of the Agency Horizon\n"
                 "Mean ± 95% CI across seeds", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.text(0.99, 0.01, 'fawp-index | Clayton (2026) doi:10.5281/zenodo.18663547',
             ha='right', va='bottom', fontsize=7, color='gray', style='italic')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    return fig, ax
