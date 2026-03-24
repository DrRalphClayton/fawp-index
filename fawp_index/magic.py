"""
fawp_index.magic — Jupyter magic command for inline FAWP detection.

Usage
-----
Load the extension in a Jupyter cell:

    %load_ext fawp_index.magic

Then use the %%fawp cell magic:

    %%fawp --tau-max 30 --n-null 50
    df  # last expression must be a DataFrame or numpy array

Or the %fawp line magic:

    %fawp df --tau-max 30 --epsilon 0.01

The magic runs FAWP detection on the DataFrame's Close/close column
(or the first numeric column) and displays an inline result summary
with the MI curve plot.
"""
from __future__ import annotations
import argparse
import sys


def _parse_fawp_args(line: str):
    p = argparse.ArgumentParser(prog="%%fawp", add_help=False)
    p.add_argument("--pred-col",    default=None,  dest="pred_col")
    p.add_argument("--future-col",  default=None,  dest="future_col")
    p.add_argument("--action-col",  default=None,  dest="action_col")
    p.add_argument("--obs-col",     default=None,  dest="obs_col")
    p.add_argument("--tau-max",     type=int,   default=30,   dest="tau_max")
    p.add_argument("--delta",       type=int,   default=20)
    p.add_argument("--epsilon",     type=float, default=0.01)
    p.add_argument("--n-null",      type=int,   default=50,   dest="n_null")
    p.add_argument("--plot",        action="store_true", default=True)
    p.add_argument("--no-plot",     action="store_false", dest="plot")
    p.add_argument("var_name",      nargs="?",  default=None)
    try:
        args, _ = p.parse_known_args(line.split())
    except SystemExit:
        args = p.parse_args([])
    return args


def _run_fawp_on_obj(obj, args, ip):
    """Run FAWP on a DataFrame or ndarray and display results inline."""
    import numpy as np
    import pandas as pd

    # Resolve DataFrame or array
    if isinstance(obj, pd.DataFrame):
        df = obj
        close_candidates = ["Close", "close", "price", "Price", "value"]
        col = args.pred_col or next((c for c in close_candidates if c in df.columns),
                                    df.select_dtypes("number").columns[0])
        series = df[col].dropna().values.astype(float)
    elif isinstance(obj, np.ndarray):
        series = obj.astype(float)
    elif isinstance(obj, (list, pd.Series)):
        series = np.array(obj, dtype=float)
    else:
        print(f"%%fawp: expected DataFrame, Series, or ndarray, got {type(obj).__name__}")
        return

    # Build channels
    delta = args.delta
    n = len(series) - delta
    if n < 50:
        print(f"%%fawp: series too short ({len(series)} < {delta + 50})")
        return

    pred_s   = series[:n]
    future_s = series[delta:delta + n]
    steer_s  = np.diff(series)[:n]

    # Run FAWP
    from fawp_index.weather import _compute_weather_mi_curves
    odw, tau, pred_mi, steer_mi = _compute_weather_mi_curves(
        pred_s, future_s, steer_s,
        tau_max=args.tau_max, epsilon=args.epsilon, n_null=args.n_null,
    )

    # Print summary
    status = "🔴 FAWP DETECTED" if odw.fawp_found else "✅ No FAWP"
    print(f"\n{status}")
    print(f"  Peak gap   : {odw.peak_gap_bits:.4f} bits")
    print(f"  τ⁺ₕ horizon: {odw.tau_h_plus or '—'}")
    print(f"  τf cliff   : {odw.tau_f or '—'}")
    print(f"  ODW        : τ{odw.odw_start}–{odw.odw_end}" if odw.fawp_found else "  ODW        : —")
    print(f"  n obs      : {n}\n")

    # Inline plot
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from IPython.display import display

            fig, ax = plt.subplots(figsize=(9, 3.5))
            fig.patch.set_facecolor("#07101E")
            ax.set_facecolor("#0D1729")
            ax.plot(tau, pred_mi,  color="#D4AF37", lw=2,   label="Prediction MI")
            ax.plot(tau, steer_mi, color="#4A7FCC", lw=1.5, ls="--", label="Steering MI")
            ax.axhline(args.epsilon, color="#3A4E70", ls=":", lw=1, label=f"ε={args.epsilon}")
            if odw.fawp_found and odw.odw_start and odw.odw_end:
                ax.axvspan(odw.odw_start, odw.odw_end, color="#C0111A", alpha=0.15,
                           label=f"ODW τ{odw.odw_start}–{odw.odw_end}")
            for sp in ax.spines.values(): sp.set_edgecolor("#3A4E70")
            ax.tick_params(colors="#7A90B8")
            ax.set_xlabel("τ (delay)", fontsize=8, color="#7A90B8")
            ax.set_ylabel("MI (bits)", fontsize=8, color="#7A90B8")
            ax.set_title(status, color="#D4AF37", fontsize=10, fontweight="bold")
            ax.legend(fontsize=8, framealpha=0.2)
            fig.tight_layout()
            display(fig)
            plt.close(fig)
        except ImportError:
            print("  (install matplotlib for inline plot)")

    # Return result object for further use
    from fawp_index.weather import WeatherFAWPResult
    return WeatherFAWPResult(
        variable="series", location="inline", odw_result=odw, tau=tau,
        pred_mi=pred_mi, steer_mi=steer_mi, skill_metric="MI", n_obs=n,
        horizon_days=delta, date_range=("", ""), metadata={},
    )


def load_ipython_extension(ip):
    """Register %%fawp and %fawp magics with IPython/Jupyter."""

    from IPython.core.magic import register_line_cell_magic

    @register_line_cell_magic
    def fawp(line, cell=None):
        """
        %%fawp [options]
        df_expression

        Run FAWP detection inline on a DataFrame, Series, or ndarray.

        Options
        -------
        --tau-max INT       Maximum delay (default: 30)
        --epsilon FLOAT     Steering threshold (default: 0.01)
        --n-null INT        Null permutations (default: 50)
        --delta INT         Forecast horizon (default: 20)
        --no-plot           Suppress plot output

        Examples
        --------
        %%fawp --tau-max 40
        my_df

        %fawp my_series --epsilon 0.005
        """
        args = _parse_fawp_args(line)

        if cell is not None:
            # Cell magic: evaluate cell content as expression
            cell = cell.strip()
            try:
                obj = ip.ev(cell)
            except Exception as e:
                print(f"%%fawp: could not evaluate cell: {e}")
                return
        elif args.var_name:
            try:
                obj = ip.ev(args.var_name)
            except Exception as e:
                print(f"%fawp: variable '{args.var_name}' not found: {e}")
                return
        else:
            print("%fawp: provide a variable name or use %%fawp cell magic")
            return

        return _run_fawp_on_obj(obj, args, ip)

    print("✅ fawp_index magic loaded — use %%fawp or %fawp")
