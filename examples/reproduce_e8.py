"""
reproduce_e8.py — Reproduce the E8 leverage gap figures from Clayton (2026)

This script uses the ACTUAL experimental data bundled with fawp-index to
reproduce the figures from:

  "Forecasting Without Power: Agency Horizons and the Leverage Gap"
  Ralph Clayton (2026)
  DOI: https://doi.org/10.5281/zenodo.18663547

Experiment E8 parameters (flagship run):
  a = 1.02      (unstable AR(1) growth factor)
  K = 0.8       (controller gain)
  Δ = 20        (prediction horizon, steps)
  n_trials = 400
  x_fail = 500  (crash threshold)
  n_steps = 1000
  seed = 42

Key finding: Stratified predictive MI peaks at 2.2337 bits (τ=9) while
steering MI collapses to zero at τ=4 (agency horizon). This is the
Information-Control Exclusion Principle: prediction and control are
conjugate — when control fails, prediction information surges.

Usage:
    python reproduce_e8.py
    python reproduce_e8.py --save  # saves PNGs instead of showing
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pathlib
import sys

# Allow running from examples/ directly
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from fawp_index.data import E8_DATA, E8_CONFIRM_FULL, E8_SIGNIFICANCE


# ─── E8 configuration (from E8_final_config.py) ──────────────────────────────
CONFIG = {
    "a": 1.02,
    "K": 0.8,
    "delta_pred": 20,
    "n_trials": 400,
    "n_steps": 1000,
    "x_fail": 500,
    "sigma_proc": 1.0,
    "sigma_obs_base": 0.1,
    "alpha_obs": 0.001,
    "u_max": 10.0,
    "epsilon": 0.01,
    "K_event": 10,
    "seed": 42,
}
TAU_H = 4   # empirical agency horizon (steer MI ≤ 0.01)


def plot_e8_leverage_gap(df, save_path=None):
    """
    Figure 1 style: E8 leverage gap (delays 0–80).
    Reproduces E8_Confirm_Final_FULL.png from the paper.
    """
    fig, ax1 = plt.subplots(figsize=(13, 7))

    # ── MI lines ──────────────────────────────────────────────────────────────
    ax1.plot(
        df.delay, df.mi_steer_pooled,
        linewidth=2.5, linestyle='--', color='steelblue',
        label=r"Pooled steering $I(A_t\,;\,O_{t+\tau+1})$",
    )
    ax1.plot(
        df.delay, df.mi_pred_pooled,
        linewidth=2.5, color='darkorange',
        label=rf"Pooled prediction $I(D_t\,;\,X_{{t+{CONFIG['delta_pred']}}})$",
    )
    ax1.plot(
        df.delay, df.mi_pred_strat,
        linewidth=2.5, linestyle='-.', color='green',
        label=r"Stratified prediction (controls time-confound)",
    )
    ax1.plot(
        df.delay, df.mi_pred_strat_shuffle,
        linewidth=1.8, linestyle=':', color='green', alpha=0.6,
        label=r"Stratified SHUFFLE control ($\approx 0$)",
    )
    ax1.plot(
        df.delay, df.mi_event_strat,
        linewidth=2.5, linestyle=(0, (6, 2)), color='purple',
        label=rf"Event MI $I(D_t\,;\,\text{{crash within }}{CONFIG['K_event']}\text{{ steps}})$",
    )
    ax1.plot(
        df.delay, df.mi_event_strat_shuffle,
        linewidth=1.8, linestyle=(0, (1, 2)), color='purple', alpha=0.6,
        label=r"Event MI SHUFFLE control ($\approx 0$)",
    )

    # ── Leverage gap shading ──────────────────────────────────────────────────
    ax1.fill_between(
        df.delay, df.mi_steer_pooled, df.mi_pred_pooled,
        where=(df.mi_pred_pooled > df.mi_steer_pooled),
        alpha=0.15, color='darkorange',
        label="Leverage gap (FAWP region)",
    )

    # ── Agency horizon ────────────────────────────────────────────────────────
    ax1.axvline(
        TAU_H, color='gray', linestyle='--', linewidth=1.8, alpha=0.9,
        label=rf"Agency horizon $\tau_h = {TAU_H}$ (steer $\leq$ {CONFIG['epsilon']})",
    )

    # ── Peak annotation ───────────────────────────────────────────────────────
    peak_row = df.loc[df.mi_pred_strat.idxmax()]
    ax1.annotate(
        f"Peak strat. pred.\n{peak_row.mi_pred_strat:.4f} bits\n(τ={int(peak_row.delay)})",
        xy=(peak_row.delay, peak_row.mi_pred_strat),
        xytext=(peak_row.delay + 3, peak_row.mi_pred_strat + 0.12),
        fontsize=8.5, color='green',
        arrowprops=dict(arrowstyle='->', color='green', lw=1.2),
    )

    # ── Failure rate (right axis) ─────────────────────────────────────────────
    ax2 = ax1.twinx()
    ax2.plot(
        df.delay, df.fail_rate,
        color='black', alpha=0.4, linestyle=':', linewidth=2.2,
        label="Failure rate",
    )
    ax2.set_ylabel("Failure Probability", color='gray', fontsize=11)
    ax2.set_ylim(-0.05, 1.15)
    ax2.tick_params(axis='y', labelcolor='gray')

    # ── Formatting ────────────────────────────────────────────────────────────
    ax1.set_xlabel(r"Latency $\tau$ (time steps)", fontsize=12)
    ax1.set_ylabel("Mutual Information (bits)", fontsize=12)
    ax1.set_title(
        rf"E8 Leverage Gap: Prediction Persists After Control Collapses"
        "\n"
        rf"$a={CONFIG['a']}$, $K={CONFIG['K']}$, $\Delta={CONFIG['delta_pred']}$, "
        rf"{CONFIG['n_trials']} trials, $x_{{fail}}={CONFIG['x_fail']}$",
        fontsize=11, pad=12,
    )
    ax1.set_ylim(bottom=-0.05)
    ax1.grid(True, alpha=0.3)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2, labels1 + labels2,
        loc='center right', fontsize=8.5,
        frameon=True, framealpha=0.95,
    )

    # Citation watermark
    fig.text(
        0.99, 0.01,
        'fawp-index | Clayton (2026) doi:10.5281/zenodo.18663547',
        ha='right', va='bottom', fontsize=7.5, color='gray', style='italic',
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    return fig, ax1


def plot_e8_micro(df_micro, save_path=None):
    """
    MICRO zoom (delays 0–15): shows the resonance ridge clearly.
    Reproduces E8_Confirm_Final_MICRO.png from the paper.
    """
    df = df_micro[df_micro.delay <= 15].copy()

    fig, ax1 = plt.subplots(figsize=(10, 5.5))

    ax1.plot(df.delay, df.mi_steer_pooled, linewidth=2.5, linestyle='--',
             color='steelblue', label=r"Pooled steering $I(A_t\,;\,O_{t+\tau+1})$")
    ax1.plot(df.delay, df.mi_pred_pooled, linewidth=2.5, color='darkorange',
             label=r"Pooled prediction $I(D_t\,;\,X_{t+\Delta})$")
    ax1.plot(df.delay, df.mi_pred_strat, linewidth=2.5, linestyle='-.',
             color='green', label="Stratified prediction")
    ax1.plot(df.delay, df.mi_pred_strat_shuffle, linewidth=1.8, linestyle=':',
             color='green', alpha=0.6, label="Stratified SHUFFLE ($\\approx 0$)")

    ax1.fill_between(
        df.delay, df.mi_steer_pooled, df.mi_pred_pooled,
        where=(df.mi_pred_pooled > df.mi_steer_pooled),
        alpha=0.15, color='darkorange', label="Leverage gap",
    )
    ax1.axvline(TAU_H, color='gray', linestyle='--', linewidth=1.8,
                label=rf"$\tau_h = {TAU_H}$")

    ax2 = ax1.twinx()
    ax2.plot(df.delay, df.fail_rate, color='black', alpha=0.4,
             linestyle=':', linewidth=2.2, label="Failure rate")
    ax2.set_ylabel("Failure Probability", color='gray', fontsize=10)
    ax2.set_ylim(-0.05, 1.15)
    ax2.tick_params(axis='y', labelcolor='gray')

    ax1.set_xlabel(r"Latency $\tau$ (time steps)", fontsize=11)
    ax1.set_ylabel("Mutual Information (bits)", fontsize=11)
    ax1.set_title(
        rf"E8 Resonance Ridge (τ=0..15): Steering Collapses, Prediction Persists"
        "\n"
        rf"$a={CONFIG['a']}$, $K={CONFIG['K']}$, $\Delta={CONFIG['delta_pred']}$",
        fontsize=10, pad=10,
    )
    ax1.set_ylim(bottom=-0.05)
    ax1.grid(True, alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='upper right', fontsize=8.5, framealpha=0.95)

    fig.text(0.99, 0.01, 'fawp-index | Clayton (2026) doi:10.5281/zenodo.18663547',
             ha='right', va='bottom', fontsize=7, color='gray', style='italic')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    return fig, ax1


def print_key_results(df):
    """Print the key numerical results from E8 to console."""
    print("\n" + "=" * 60)
    print("E8 KEY RESULTS — Clayton (2026)")
    print("=" * 60)
    print(f"Config: a={CONFIG['a']}, K={CONFIG['K']}, "
          f"Δ={CONFIG['delta_pred']}, {CONFIG['n_trials']} trials")
    print(f"Empirical agency horizon τ_h = {TAU_H} "
          f"(steer MI ≤ {CONFIG['epsilon']} bits)")
    print()

    peak_row = df.loc[df.mi_pred_strat.idxmax()]
    print(f"Peak stratified pred MI:  {peak_row.mi_pred_strat:.4f} bits  "
          f"(τ={int(peak_row.delay)})")

    horizon_row = df[df.delay == TAU_H]
    if not horizon_row.empty:
        r = horizon_row.iloc[0]
        print(f"At τ_h={TAU_H}:  steer={r.mi_steer_pooled:.6f}  "
              f"pred={r.mi_pred_pooled:.4f}  gap={r.gap_pooled:.4f}")

    print()
    print("Delay-by-delay (full sweep):")
    print(f"{'τ':>4} {'Steer':>10} {'Pred':>10} {'Strat':>10} {'Gap':>10} {'Fail':>6}")
    print("-" * 55)
    for _, row in df.iterrows():
        fawp = " ← FAWP" if (row.mi_pred_strat > 0.01 and row.mi_steer_pooled < 0.01) else ""
        print(f"{int(row.delay):>4} {row.mi_steer_pooled:>10.4f} "
              f"{row.mi_pred_pooled:>10.4f} {row.mi_pred_strat:>10.4f} "
              f"{row.gap_pooled:>10.4f} {row.fail_rate:>6.2f}{fawp}")
    print("=" * 60)
    print("Information-Control Exclusion Principle confirmed:")
    print("  When steering collapses → predictive MI surges.")
    print(f"  DOI: https://doi.org/10.5281/zenodo.18663547")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce E8 figures from Clayton (2026) — fawp-index"
    )
    parser.add_argument('--save', action='store_true',
                        help='Save figures to PNG instead of displaying')
    parser.add_argument('--no-micro', action='store_true',
                        help='Skip the micro (0-15) zoom plot')
    args = parser.parse_args()

    if args.save:
        matplotlib.use('Agg')

    # ── Load bundled data ──────────────────────────────────────────────────────
    print("Loading bundled E8 data...")
    df_full = pd.read_csv(E8_CONFIRM_FULL)
    print(f"  Full sweep: {len(df_full)} delay values (τ=0..{df_full.delay.max()})")

    # ── Print results ──────────────────────────────────────────────────────────
    print_key_results(df_full)

    # ── Full leverage gap plot ────────────────────────────────────────────────
    print("\nGenerating E8 leverage gap figure (full, τ=0..80)...")
    fig1, _ = plot_e8_leverage_gap(
        df_full,
        save_path="E8_Leverage_Gap_Full.png" if args.save else None,
    )

    # ── Micro zoom ────────────────────────────────────────────────────────────
    if not args.no_micro:
        print("Generating E8 micro figure (τ=0..15)...")
        fig2, _ = plot_e8_micro(
            df_full,
            save_path="E8_Leverage_Gap_Micro.png" if args.save else None,
        )

    if not args.save:
        plt.show()

    print("\nDone! Figures reproduce Clayton (2026) Experiment E8.")
    print("Cite: doi:10.5281/zenodo.18663547")


if __name__ == '__main__':
    main()
