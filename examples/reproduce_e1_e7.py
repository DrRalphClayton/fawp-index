"""
reproduce_e1_e7.py — Reproduce all E1-E7 published figures from bundled data.

Usage:
    python examples/reproduce_e1_e7.py          # all figures, interactive
    python examples/reproduce_e1_e7.py --save   # save all PNGs

Ralph Clayton (2026) — doi:10.5281/zenodo.18663547
"""
import sys, os as _os
# Allow running from repo root OR from examples/ directory
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse
import numpy as np


def reproduce_e1(save=False):
    """E1: Parameter sweep — analytic vs MC horizon scaling."""
    print("\n--- E1: Parameter Sweep (Horizon Scaling) ---")
    from fawp_index.oats import AgencyHorizon
    from fawp_index.data import E1_HORIZONS_SWEEP
    import pandas as pd

    ah = AgencyHorizon()
    sweep = ah.compare_e1()
    print(sweep.summary())

    save_path = "E1_horizon_scaling.png" if save else None
    sweep.plot_scaling(save_path=save_path, show=not save)
    if save:
        print(f"Saved: {save_path}")


def reproduce_e2(save=False):
    """E2: Convergence — MI estimator obeys 1/sqrt(N) CLT scaling."""
    print("\n--- E2: Convergence Test (CLT Scaling) ---")
    from fawp_index.data import E2_CONVERGENCE
    import pandas as pd
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  skipped (requires matplotlib)")
        return

    df = pd.read_csv(E2_CONVERGENCE)
    print(f"  {len(df)} sample sizes | N={df['N'].min():,}..{df['N'].max():,}")
    print(f"  Final CI width: {df['ci_width'].iloc[-1]:.6f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(df['N'], df['ci_width'], 'bo-', linewidth=2, markersize=5, label='95% CI width')
    # Reference 1/sqrt(N) line
    N_ref = df['N'].values
    scale = df['ci_width'].iloc[0] * np.sqrt(df['N'].iloc[0])
    ax.loglog(N_ref, scale / np.sqrt(N_ref), 'k--', linewidth=1.5, label='1/√N reference')
    ax.set_xlabel("Sample size N", fontsize=11)
    ax.set_ylabel("95% CI width (bits)", fontsize=11)
    ax.set_title("E2: MI Estimator Convergence\nCI width follows 1/√N (CLT scaling)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    fig.text(0.99, 0.01, 'fawp-index | Clayton (2026) doi:10.5281/zenodo.18663547',
             ha='right', va='bottom', fontsize=7, color='gray', style='italic')
    plt.tight_layout()

    if save:
        plt.savefig("E2_convergence.png", dpi=150, bbox_inches='tight')
        print("Saved: E2_convergence.png")
    else:
        plt.show()


def reproduce_e3(save=False):
    """E3: Noise-law discrimination — linear vs quadratic vs saturating."""
    print("\n--- E3: Noise-Law Discrimination ---")
    from fawp_index.data import E3_CURVES_LINEAR, E3_CURVES_QUADRATIC, E3_CURVES_SATURATING, E3_HORIZONS_SUMMARY
    import pandas as pd
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  skipped (requires matplotlib)")
        return

    lin  = pd.read_csv(E3_CURVES_LINEAR)
    quad = pd.read_csv(E3_CURVES_QUADRATIC)
    sat  = pd.read_csv(E3_CURVES_SATURATING)
    summ = pd.read_csv(E3_HORIZONS_SUMMARY)
    print(f"  Horizons summary:\n{summ.to_string(index=False)}")

    fig, ax = plt.subplots(figsize=(10, 5))
    for df, label, color in [
        (lin,  'Linear noise',     'darkorange'),
        (quad, 'Quadratic noise',  'steelblue'),
        (sat,  'Saturating noise', 'green'),
    ]:
        if 'mi_mc_mean' in df.columns:
            ax.plot(df['tau_s'], df['mi_mc_mean'], linewidth=2.2, color=color, label=label)
            ax.fill_between(df['tau_s'], df['mi_mc_ci95_low'], df['mi_mc_ci95_high'],
                            alpha=0.15, color=color)
            if 'mi_analytic' in df.columns:
                ax.plot(df['tau_s'], df['mi_analytic'], '--', linewidth=1.2,
                        color=color, alpha=0.6)

    ax.set_xlabel("Latency τ", fontsize=11)
    ax.set_ylabel("Mutual Information (bits)", fontsize=11)
    ax.set_title("E3: Noise-Law Discrimination\nMC mean ± 95% CI + analytic overlay", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.text(0.99, 0.01, 'fawp-index | Clayton (2026) doi:10.5281/zenodo.18663547',
             ha='right', va='bottom', fontsize=7, color='gray', style='italic')
    plt.tight_layout()

    if save:
        plt.savefig("E3_noise_laws.png", dpi=150, bbox_inches='tight')
        print("Saved: E3_noise_laws.png")
    else:
        plt.show()


def reproduce_e4(save=False):
    """E4: Distributional robustness."""
    print("\n--- E4: Distributional Robustness ---")
    from fawp_index.oats import DistributionalRobustness

    result = DistributionalRobustness.from_e4_data()
    print(result.summary())

    save_path = "E4_robustness.png" if save else None
    result.plot(save_path=save_path, show=not save)
    if save:
        print(f"Saved: {save_path}")


def reproduce_e5(save=False):
    """E5: Control cliff."""
    print("\n--- E5: Control Cliff ---")
    from fawp_index.simulate import ControlCliff

    result = ControlCliff.from_e5_data()
    print(result.summary())

    save_path = "E5_control_cliff.png" if save else None
    result.plot(save_path=save_path, show=not save)
    if save:
        print(f"Saved: {save_path}")


def reproduce_e6(save=False):
    """E6: Agency capacity surfaces (quantization cliff + erasure slope)."""
    print("\n--- E6: Agency Capacity Surfaces ---")
    from fawp_index.capacity import CapacitySurface

    result = CapacitySurface.from_e6_data()
    print(result.summary())

    save_path = "E6_capacity_surfaces.png" if save else None
    result.plot(surface='both', save_path=save_path, show=not save)
    if save:
        print(f"Saved: {save_path}")


def reproduce_e7(save=False):
    """E7: Quantum checks — Bell/CHSH + no-signaling."""
    print("\n--- E7: Quantum Consistency Checks ---")
    from fawp_index.data import E7_QUANTUM
    import pandas as pd
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("  skipped (requires matplotlib)")
        return

    df = pd.read_csv(E7_QUANTUM)
    print(f"  {len(df)} visibility values | v=0..{df['v'].max():.2f}")
    print(f"  Max CHSH S: {df['CHSH'].max():.4f} (Tsirelson bound: {2*np.sqrt(2):.4f})")
    print(f"  Max NS deviation: {df['NS_Dev'].max():.2e} (should be ≈ machine precision)")

    fig = plt.figure(figsize=(11, 5))
    gs = gridspec.GridSpec(1, 2)

    # CHSH S vs visibility
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df['v'], df['CHSH'], 'bo-', linewidth=2, markersize=4, label='Simulated S')
    ax1.plot(df['v'], df['CHSH_theory'], 'k--', linewidth=1.5, label='Theory: 2√2·v')
    ax1.axhline(2, color='red', linestyle=':', linewidth=1.5, alpha=0.7, label='Classical bound (S=2)')
    ax1.axhline(2*np.sqrt(2), color='green', linestyle=':', linewidth=1.5, alpha=0.7,
                label='Tsirelson bound (2√2)')
    ax1.axvline(1/np.sqrt(2), color='gray', linestyle='--', linewidth=1.2, alpha=0.6,
                label=f'v=1/√2 (Bell threshold)')
    ax1.set_xlabel("Visibility v", fontsize=11)
    ax1.set_ylabel("CHSH parameter S", fontsize=11)
    ax1.set_title("E7a: CHSH Bell Test\nSimulation vs theory (Werner state)", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # No-signaling deviation
    ax2 = fig.add_subplot(gs[1])
    ns_vals = df['NS_Dev'].replace(0, 1e-20)
    ax2.semilogy(df['v'], ns_vals, 'r^-', linewidth=2, markersize=5, label='No-signaling deviation Δ_NS')
    ax2.axhline(1e-15, color='gray', linestyle='--', linewidth=1.2, label='Machine precision (~1e-15)')
    ax2.set_xlabel("Visibility v", fontsize=11)
    ax2.set_ylabel("No-signaling deviation Δ_NS", fontsize=11)
    ax2.set_title("E7b: No-Signaling Check\nΔ_NS ≈ machine precision (correct)", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, which='both')

    fig.text(0.99, 0.01, 'fawp-index | Clayton (2026) doi:10.5281/zenodo.18663547',
             ha='right', va='bottom', fontsize=7, color='gray', style='italic')
    plt.tight_layout()

    if save:
        plt.savefig("E7_quantum_checks.png", dpi=150, bbox_inches='tight')
        print("Saved: E7_quantum_checks.png")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce E1-E7 published figures from fawp-index bundled data."
    )
    parser.add_argument('--save', action='store_true',
                        help='Save figures as PNGs instead of showing interactively')
    parser.add_argument('--experiments', nargs='+', type=int,
                        choices=[1, 2, 3, 4, 5, 6, 7],
                        default=[1, 2, 3, 4, 5, 6, 7],
                        help='Which experiments to reproduce (default: all)')
    args = parser.parse_args()

    print("fawp-index — Reproducing E1-E7 published figures")
    print("Ralph Clayton (2026) | doi:10.5281/zenodo.18663547")

    dispatch = {
        1: reproduce_e1,
        2: reproduce_e2,
        3: reproduce_e3,
        4: reproduce_e4,
        5: reproduce_e5,
        6: reproduce_e6,
        7: reproduce_e7,
    }

    for n in args.experiments:
        try:
            dispatch[n](save=args.save)
        except Exception as ex:
            print(f"  E{n} failed: {ex}")

    print("\nDone.")


if __name__ == "__main__":
    main()
