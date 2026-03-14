"""
fawp-index — Command Line Interface
====================================

Subcommands
-----------
  detect       Run FAWP detector on a CSV file
  market       Rolling FAWP scan on a price CSV
  watchlist    Scan multiple CSVs and rank by FAWP signal
  significance Bootstrap significance test on a detector result
  benchmarks   Run the built-in benchmark suite
  version      Print version and exit

Quick start
-----------
  fawp-index detect   data.csv --state price --action trade
  fawp-index market   prices.csv --close Close --volume Volume
  fawp-index watchlist spy.csv qqq.csv gld.csv --labels SPY QQQ GLD
  fawp-index significance data.csv --state price --action trade
  fawp-index benchmarks
  fawp-index version
"""

import argparse
import sys
from pathlib import Path

from fawp_index import __version__ as _VERSION


def _header():
    print(f"\nfawp-index v{_VERSION} | Clayton (2026) | doi:10.5281/zenodo.18673949")
    print("=" * 60)


def _load_price_df(csv_path, close_col, date_col=None):
    import pandas as pd
    df = pd.read_csv(csv_path)
    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
    else:
        for col in df.columns:
            if "date" in col.lower() or "time" in col.lower():
                df[col] = pd.to_datetime(df[col])
                df = df.set_index(col)
                break
        else:
            df.index = pd.to_datetime(df.index)
    if close_col not in df.columns:
        matches = [c for c in df.columns if c.lower() == close_col.lower()]
        if not matches:
            print(f"ERROR: column '{close_col}' not found. Available: {list(df.columns)}")
            sys.exit(1)
        df = df.rename(columns={matches[0]: close_col})
    return df


def _write_detect_output(result, path):
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".json":
        import json
        d = {
            "fawp_found": bool(result.in_fawp.any()),
            "tau_h": int(result.tau_h) if result.tau_h is not None else None,
            "peak_alpha": float(result.peak_alpha),
            "peak_tau": int(result.peak_tau) if result.peak_tau is not None else None,
        }
        p.write_text(json.dumps(d, indent=2))
    elif suffix == ".csv":
        import pandas as pd
        pd.DataFrame({
            "tau": result.tau,
            "pred_mi": result.pred_mi_raw,
            "steer_mi": result.steer_mi_raw,
            "gap": result.pred_mi_raw - result.steer_mi_raw,
            "fawp": result.in_fawp,
        }).to_csv(p, index=False)
    else:
        print(f"WARNING: unsupported format '{suffix}' — use .json or .csv")
        return
    print(f"Saved → {p}")


def cmd_detect(args):
    _header()
    print(f"Command : detect  |  {args.csv}")
    simple = bool(args.state and args.action)
    full   = bool(args.pred and args.future and args.action and args.obs)
    if not simple and not full:
        print("ERROR: --state+--action or --pred+--future+--action+--obs required")
        sys.exit(1)
    try:
        if simple:
            from fawp_index.io.csv_loader import load_csv_simple
            data = load_csv_simple(args.csv, state_col=args.state, action_col=args.action, delta_pred=args.delta)
        else:
            from fawp_index.io.csv_loader import load_csv
            data = load_csv(args.csv, pred_col=args.pred, future_col=args.future, action_col=args.action, obs_col=args.obs, delta_pred=args.delta)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    print(f"Rows {len(data.pred_series):,}  |  Delta={args.delta}  |  tau={args.tau_min}-{args.tau_max}")
    from fawp_index.core.alpha_index import FAWPAlphaIndex
    result = FAWPAlphaIndex(eta=args.eta, epsilon=args.epsilon, m_persist=args.persist, n_null=args.n_null).compute(
        pred_series=data.pred_series, future_series=data.future_series,
        action_series=data.action_series, obs_series=data.obs_series,
        tau_grid=list(range(args.tau_min, args.tau_max + 1)), verbose=args.verbose,
    )
    print("\n" + result.summary())
    print(f"\n{'tau':>4} {'Pred MI':>10} {'Steer MI':>10} {'Gap':>10} {'FAWP':>6}")
    print("-" * 46)
    for i, tau in enumerate(result.tau):
        gap = result.pred_mi_raw[i] - result.steer_mi_raw[i]
        flag = "<- ok" if result.in_fawp[i] else ""
        print(f"{tau:>4} {result.pred_mi_raw[i]:>10.4f} {result.steer_mi_raw[i]:>10.4f} {gap:>10.4f}  {flag}")
    if args.out:
        _write_detect_output(result, args.out)
    if args.plot or args.save:
        try:
            from fawp_index.viz.plots import plot_leverage_gap
            plot_leverage_gap(result, save_path=args.save, show=args.plot)
        except ImportError:
            print("WARNING: matplotlib not installed")
    status = "FAWP DETECTED" if result.in_fawp.any() else "No FAWP detected"
    print(f"\n{status}")


def cmd_market(args):
    _header()
    print(f"Command : market  |  {args.csv}")
    df = _load_price_df(args.csv, args.close, args.date_col)
    vol = args.volume if (args.volume and args.volume in df.columns) else None
    print(f"{len(df):,} rows  {df.index[0].date()} -> {df.index[-1].date()}  |  window={args.window} step={args.step} tau_max={args.tau_max}")
    from fawp_index.market import scan_fawp_market
    scan = scan_fawp_market(df, ticker=Path(args.csv).stem.upper(),
        close_col=args.close, volume_col=vol, window=args.window, step=args.step,
        tau_max=args.tau_max, n_null=args.n_null, verbose=True)
    print("\n" + scan.summary())
    if args.out:
        p = Path(args.out)
        fmt = p.suffix.lower().lstrip(".")
        writer = {"html": scan.to_html, "json": scan.to_json, "csv": scan.to_csv}.get(fmt)
        if writer:
            writer(p)
        else:
            print("WARNING: unsupported format")
        print(f"Saved -> {p}")
    if args.plot:
        try:
            scan.plot(prices=df[args.close])
        except ImportError:
            print("WARNING: matplotlib not installed")


def cmd_watchlist(args):
    import pandas as pd  # noqa: F401 — ensures pandas is available for _load_price_df
    _header()
    print("Command : watchlist")
    labels = args.labels or [Path(f).stem.upper() for f in args.csvs]
    if len(labels) != len(args.csvs):
        print("ERROR: --labels count must match CSV count")
        sys.exit(1)
    dfs = {label: _load_price_df(f, args.close, args.date_col) for label, f in zip(labels, args.csvs)}
    for label, df in dfs.items():
        print(f"  {label}: {len(df):,} rows")
    tf_list = [tf.strip() for tf in args.timeframes.split(",")]
    from fawp_index.watchlist import scan_watchlist
    result = scan_watchlist(dfs, timeframes=tf_list, window=args.window, step=args.step,
        tau_max=args.tau_max, n_null=args.n_null, verbose=True)
    print("\n" + result.summary(n=30))
    if args.out:
        p = Path(args.out)
        fmt = p.suffix.lower().lstrip(".")
        writer = {"html": result.to_html, "json": result.to_json, "csv": result.to_csv}.get(fmt)
        if writer:
            writer(p)
        else:
            print("WARNING: unsupported format")
        print(f"Saved -> {p}")
    ranked = result.rank_by(args.rank_by)
    print(f"\nTop {min(args.top, len(ranked))} by {args.rank_by}:")
    for a in ranked[:args.top]:
        flag = "FAWP" if a.regime_active else "    "
        print(f"  [{flag}]  {a.ticker:<10} [{a.timeframe}]  score={a.latest_score:.4f}  gap={a.peak_gap_bits:.4f}b  age={a.signal_age_days}d")


def cmd_significance(args):
    _header()
    print(f"Command : significance  |  {args.csv}")
    simple = bool(args.state and args.action)
    full   = bool(args.pred and args.future and args.action and args.obs)
    if not simple and not full:
        print("ERROR: --state+--action or --pred+--future+--action+--obs required")
        sys.exit(1)
    try:
        if simple:
            from fawp_index.io.csv_loader import load_csv_simple
            data = load_csv_simple(args.csv, state_col=args.state, action_col=args.action, delta_pred=args.delta)
        else:
            from fawp_index.io.csv_loader import load_csv
            data = load_csv(args.csv, pred_col=args.pred, future_col=args.future, action_col=args.action, obs_col=args.obs, delta_pred=args.delta)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    from fawp_index.core.alpha_index import FAWPAlphaIndex
    from fawp_index.detection.odw import ODWDetector
    from fawp_index import fawp_significance
    result = FAWPAlphaIndex(n_null=50).compute(
        pred_series=data.pred_series, future_series=data.future_series,
        action_series=data.action_series, obs_series=data.obs_series,
        tau_grid=list(range(1, args.tau_max + 1)),
    )
    odw = ODWDetector().detect(tau=result.tau, pred_corr=result.pred_mi_raw,
        steer_corr=result.steer_mi_raw,
        fail_rate=result.failure_rate if hasattr(result, "failure_rate") else None)
    sig = fawp_significance(odw, n_bootstrap=args.n_bootstrap, alpha=args.alpha)
    print(sig.summary())
    if args.out:
        p = Path(args.out)
        fmt = p.suffix.lower().lstrip(".")
        writer = {"html": sig.to_html, "json": sig.to_json}.get(fmt)
        if writer:
            writer(p)
        else:
            print("WARNING: use .html or .json")
        print(f"Saved -> {p}")


def cmd_benchmarks(args):
    _header()
    print("Command : benchmarks\n")
    from fawp_index import run_benchmarks
    suite = run_benchmarks()
    print(suite.summary())
    if args.verify:
        try:
            suite.verify_all()
            print("\nAll benchmark assertions pass.")
        except Exception as e:
            print(f"\nFailure: {e}")
            sys.exit(1)
    if args.out:
        p = Path(args.out)
        fmt = p.suffix.lower().lstrip(".")
        writer = {"html": suite.to_html, "json": suite.to_json}.get(fmt)
        if writer:
            writer(p)
        else:
            print("WARNING: use .html or .json")
        print(f"Saved -> {p}")


def cmd_version(_args):
    print(f"fawp-index {_VERSION}")
    print("Ralph Clayton (2026)")
    print("doi:10.5281/zenodo.18673949")
    print("https://github.com/DrRalphClayton/fawp-index")


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="fawp-index",
        description="FAWP Alpha Index — detect when prediction persists after steering has collapsed.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  fawp-index detect   data.csv --state price --action trade\n"
            "  fawp-index market   prices.csv --close Close --out report.html\n"
            "  fawp-index watchlist spy.csv qqq.csv --labels SPY QQQ --out wl.html\n"
            "  fawp-index significance data.csv --state price --action trade\n"
            "  fawp-index benchmarks --verify\n"
            "  fawp-index version\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"fawp-index {_VERSION}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    # detect
    p = sub.add_parser("detect", help="Run FAWP detector on a CSV")
    p.add_argument("csv")
    p.add_argument("--state")
    p.add_argument("--action")
    p.add_argument("--pred")
    p.add_argument("--future")
    p.add_argument("--obs")
    p.add_argument("--delta", type=int, default=20)
    p.add_argument("--tau-max", type=int, default=40)
    p.add_argument("--tau-min", type=int, default=1)
    p.add_argument("--eta", type=float, default=1e-4)
    p.add_argument("--epsilon", type=float, default=1e-4)
    p.add_argument("--n-null", type=int, default=200)
    p.add_argument("--persist", type=int, default=5)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--save")
    p.add_argument("--out")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_detect)

    # market
    p = sub.add_parser("market", help="Rolling FAWP market scan on a price CSV")
    p.add_argument("csv")
    p.add_argument("--close", default="Close")
    p.add_argument("--volume", default="Volume")
    p.add_argument("--date-col", default=None)
    p.add_argument("--window", type=int, default=252)
    p.add_argument("--step", type=int, default=5)
    p.add_argument("--tau-max", type=int, default=40)
    p.add_argument("--n-null", type=int, default=0)
    p.add_argument("--out")
    p.add_argument("--plot", action="store_true")
    p.set_defaults(func=cmd_market)

    # watchlist
    p = sub.add_parser("watchlist", help="Scan multiple CSVs, rank by FAWP signal")
    p.add_argument("csvs", nargs="+")
    p.add_argument("--labels", nargs="+")
    p.add_argument("--close", default="Close")
    p.add_argument("--volume", default="Volume")
    p.add_argument("--date-col", default=None)
    p.add_argument("--timeframes", default="1d")
    p.add_argument("--window", type=int, default=252)
    p.add_argument("--step", type=int, default=5)
    p.add_argument("--tau-max", type=int, default=40)
    p.add_argument("--n-null", type=int, default=0)
    p.add_argument("--rank-by", default="score", choices=["score", "gap", "entry", "persistence", "freshness"])
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--out")
    p.set_defaults(func=cmd_watchlist)

    # significance
    p = sub.add_parser("significance", help="Bootstrap significance test")
    p.add_argument("csv")
    p.add_argument("--state")
    p.add_argument("--action")
    p.add_argument("--pred")
    p.add_argument("--future")
    p.add_argument("--obs")
    p.add_argument("--delta", type=int, default=20)
    p.add_argument("--tau-max", type=int, default=40)
    p.add_argument("--n-bootstrap", type=int, default=200)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--out")
    p.set_defaults(func=cmd_significance)

    # benchmarks
    p = sub.add_parser("benchmarks", help="Run built-in benchmark suite")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--out")
    p.set_defaults(func=cmd_benchmarks)

    # version
    p = sub.add_parser("version", help="Print version and exit")
    p.set_defaults(func=cmd_version)

    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
