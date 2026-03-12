"""
fawp-index: Command Line Interface
Run FAWP Alpha Index on any CSV from the terminal.

Usage:
    fawp-index mydata.csv --state price --action trade
    fawp-index mydata.csv --pred D --future X --action A --obs O
    fawp-index mydata.csv --state price --action trade --plot --save result.png
"""

import argparse
import sys
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        prog='fawp-index',
        description=(
            'FAWP Alpha Index v2.1 — Information-Control Exclusion Principle detector\n'
            'Ralph Clayton (2026) | doi:10.5281/zenodo.18673949'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple mode (state + action columns only):
  fawp-index data.csv --state price --action trade_size

  # Full mode (all four columns):
  fawp-index data.csv --pred state --future future_state --action action --obs observation

  # With plot saved to file:
  fawp-index data.csv --state price --action trade --plot --save leverage_gap.png

  # Custom tau grid and forecast horizon:
  fawp-index data.csv --state price --action trade --tau-max 20 --delta 30
        """
    )

    # Input
    parser.add_argument('csv', help='Path to CSV file')

    # Column selection — simple mode
    parser.add_argument('--state', help='State column (simple mode: auto-builds future)')
    parser.add_argument('--action', help='Action/control column')

    # Column selection — full mode
    parser.add_argument('--pred', help='Predictor column D_t')
    parser.add_argument('--future', help='Future target column X_{t+delta}')
    parser.add_argument('--obs', help='Observation column O_{t+tau+1}')

    # Parameters
    parser.add_argument('--delta', type=int, default=20, help='Forecast horizon (default: 20)')
    parser.add_argument('--tau-max', type=int, default=15, help='Max delay to sweep (default: 15)')
    parser.add_argument('--tau-min', type=int, default=1, help='Min delay (default: 1)')
    parser.add_argument('--eta', type=float, default=1e-4, help='Prediction threshold (default: 1e-4)')
    parser.add_argument('--epsilon', type=float, default=1e-4, help='Steering threshold (default: 1e-4)')
    parser.add_argument('--n-null', type=int, default=200, help='Null samples (default: 200)')
    parser.add_argument('--persist', type=int, default=5, help='Persistence window (default: 5)')

    # Output
    parser.add_argument('--plot', action='store_true', help='Show leverage gap plot')
    parser.add_argument('--save', help='Save plot to file (e.g. result.png)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--version', action='version', version='fawp-index 0.5.1')

    args = parser.parse_args()

    # ── Validate args ─────────────────────────────────────────────────────────
    simple_mode = args.state and args.action
    full_mode = args.pred and args.future and args.action and args.obs

    if not simple_mode and not full_mode:
        print("ERROR: Provide either --state + --action (simple) or --pred + --future + --action + --obs (full)")
        sys.exit(1)

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\nfawp-index v0.5.1 | Clayton (2026)")
    print(f"{'='*50}")
    print(f"Loading: {args.csv}")

    try:
        if simple_mode:
            from fawp_index.io.csv_loader import load_csv_simple
            data = load_csv_simple(
                args.csv,
                state_col=args.state,
                action_col=args.action,
                delta_pred=args.delta,
            )
        else:
            from fawp_index.io.csv_loader import load_csv
            data = load_csv(
                args.csv,
                pred_col=args.pred,
                future_col=args.future,
                action_col=args.action,
                obs_col=args.obs,
                delta_pred=args.delta,
            )
    except Exception as e:
        print(f"ERROR loading CSV: {e}")
        sys.exit(1)

    print(f"Rows loaded: {len(data.pred_series):,}")
    print(f"Forecast horizon Δ: {args.delta}")
    print(f"Delay sweep: τ={args.tau_min}..{args.tau_max}")

    # ── Run detector ──────────────────────────────────────────────────────────
    from fawp_index.core.alpha_index import FAWPAlphaIndex

    detector = FAWPAlphaIndex(
        eta=args.eta,
        epsilon=args.epsilon,
        m_persist=args.persist,
        n_null=args.n_null,
    )

    print(f"\nComputing FAWP Alpha Index...")
    result = detector.compute(
        pred_series=data.pred_series,
        future_series=data.future_series,
        action_series=data.action_series,
        obs_series=data.obs_series,
        tau_grid=list(range(args.tau_min, args.tau_max + 1)),
        verbose=args.verbose,
    )

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + result.summary())

    print("\nDelay-by-delay breakdown:")
    print(f"{'τ':>4} {'Pred MI':>10} {'Steer MI':>10} {'Gap':>10} {'FAWP':>6}")
    print("-" * 45)
    for i, tau in enumerate(result.tau):
        gap = result.pred_mi_raw[i] - result.steer_mi_raw[i]
        fawp_flag = "← ✓" if result.in_fawp[i] else ""
        print(f"{tau:>4} {result.pred_mi_raw[i]:>10.4f} {result.steer_mi_raw[i]:>10.4f} {gap:>10.4f} {fawp_flag}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    if args.plot or args.save:
        from fawp_index.viz.plots import plot_leverage_gap
        plot_leverage_gap(
            result,
            save_path=args.save,
            show=args.plot,
        )

    # ── Exit code ─────────────────────────────────────────────────────────────
    if result.in_fawp.any():
        print(f"\n⚠️  FAWP REGIME DETECTED — prediction survives beyond operational leverage")
        sys.exit(0)
    else:
        print(f"\n✓ No FAWP regime detected at current thresholds")
        sys.exit(0)


if __name__ == '__main__':
    main()
