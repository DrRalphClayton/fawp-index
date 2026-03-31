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
    include_weather = getattr(args, "weather", False)
    label = "finance + climate" if include_weather else "finance"
    print(f"Command : benchmarks ({label})\n")
    from fawp_index import run_benchmarks
    suite = run_benchmarks(include_weather=include_weather)
    print(suite.summary())
    if include_weather:
        print("\nClimate benchmarks (E9-calibrated):")
        for c in suite.cases:
            if getattr(c, "domain", "finance") == "weather":
                status = "PASS ✓" if c.passed else "FAIL ✗"
                fawp   = "FAWP=True " if c.odw_result.fawp_found else "FAWP=False"
                exp    = "expected" if c.odw_result.fawp_found == c.expected_fawp else "UNEXPECTED"
                print(f"  {status}  {c.name:<28} {fawp}  ({exp})")
                print(f"         {c.description[:80]}...")
                print()
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
            "  fawp-index cite\n"
            "  fawp-index watch --tickers SPY,QQQ --email you@email.com\n"
            "  fawp-index report --data scan.json --out report.pdf\n"
            "  fawp-index forecast --forecast fc.csv --obs obs.csv\n"
            "  fawp-index timing\n"
            "  fawp-index verify\n"
            "  fawp-index grid\n"
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
    p.add_argument("--verify",  action="store_true",
                   help="Assert all expected_fawp values match")
    p.add_argument("--weather", action="store_true",
                   help="Include climate benchmarks (hurricane, drought, precip)")
    p.add_argument("--out", help="Save results to .html or .json")
    p.set_defaults(func=cmd_benchmarks)

    # version
    p = sub.add_parser("version", help="Print version and exit")
    p.set_defaults(func=cmd_version)

    # cite
    p = sub.add_parser("cite", help="Print BibTeX citations for fawp-index and papers")
    p.set_defaults(func=cmd_cite)

    # timing
    p = sub.add_parser("timing", help="Print E9.7 detector timing results from the paper")
    p.set_defaults(func=cmd_timing)

    # steering profile
    p = sub.add_parser("steering", help="Steering decay profile with SPHERE_23 αA/α²A thresholds")
    p.add_argument("--ticker",  required=True)
    p.add_argument("--period",  default="2y")
    p.add_argument("--epsilon", type=float, default=0.01)
    p.add_argument("--n-null",  type=int,   default=50,   dest="n_null")
    p.add_argument("--plot",    action="store_true")
    p.set_defaults(func=cmd_steering)

    # scan one-liner
    p = sub.add_parser("scan", help="One-liner FAWP scan for a single ticker")
    p.add_argument("--ticker",  required=True,              help="Ticker symbol (e.g. SPY)")
    p.add_argument("--period",  default="2y",               help="yfinance period (1y, 2y, …)")
    p.add_argument("--epsilon", type=float, default=0.01)
    p.add_argument("--n-null",  type=int,   default=50,     dest="n_null")
    p.add_argument("--out",     default=None,               help="Save result JSON to file")
    p.set_defaults(func=cmd_scan)

    # verify
    p = sub.add_parser("verify", help="Run calibration self-checks against SPHERE/E9 constants")
    p.add_argument("--sphere23", action="store_true",
                   help="Run live E11-1 simulation vs SPHERE_23 reference values")
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--tolerance", type=int, default=5,
                   help="Allowed horizon deviation in delay steps")
    p.set_defaults(func=cmd_verify)

    # grid
    # calibrate
    p = sub.add_parser("calibrate", help="Re-derive SPHERE-16 constants from simulation")
    p.add_argument("--a",        type=float, default=1.02,  help="AR(1) gain")
    p.add_argument("--K",        type=float, default=0.8,   help="Controller gain")
    p.add_argument("--delta",    type=int,   default=20,    help="Observation delay Δ")
    p.add_argument("--n-trials", type=int,   default=200,   dest="n_trials")
    p.add_argument("--n-null",   type=int,   default=50,    dest="n_null")
    p.add_argument("--tau-max",  type=int,   default=40,    dest="tau_max")
    p.add_argument("--seed",     type=int,   default=42)
    p.add_argument("--out",      default=None)
    p.set_defaults(func=cmd_calibrate)

    # agency sweep
    p = sub.add_parser("agency", help="Sweep agency horizon over parameter space (VTM Eq. 15-16)")
    p.add_argument("--mode",    default="alpha", choices=["alpha","P","surface"],
                   help="Sweep variable: alpha (noise growth) or P (signal power)")
    p.add_argument("--P",       type=float, default=1.0,  help="Action variance P (fixed in alpha mode)")
    p.add_argument("--alpha",   type=float, default=0.01, help="Noise growth α (fixed in P mode)")
    p.add_argument("--sigma0",  type=float, default=0.0001, help="Baseline noise σ²₀")
    p.add_argument("--epsilon", type=float, default=0.01,   help="Detectability threshold ε (bits)")
    p.add_argument("--a-min",   type=float, default=1e-3,   dest="a_min")
    p.add_argument("--a-max",   type=float, default=1e-1,   dest="a_max")
    p.add_argument("--P-min",   type=float, default=0.01,   dest="P_min")
    p.add_argument("--P-max",   type=float, default=100.0,  dest="P_max")
    p.add_argument("--n-steps", type=int,   default=20,     dest="n_steps")
    p.add_argument("--no-plot", action="store_true",        dest="no_plot")
    p.add_argument("--out",     default=None)
    p.set_defaults(func=cmd_agency)

    # leaderboard push/list
    lb_p = sub.add_parser("leaderboard", help="Push or list global FAWP leaderboard entries")
    lb_sub = lb_p.add_subparsers(dest="subcommand")
    # push
    lb_push = lb_sub.add_parser("push", help="Push a scan JSON to the global leaderboard")
    lb_push.add_argument("--data",  required=True, help="Path to FAWP result .json")
    lb_push.add_argument("--url",   default=None,  help="Supabase URL (or FAWP_SUPABASE_URL env)")
    lb_push.add_argument("--token", default=None,  help="Supabase token (or FAWP_SUPABASE_TOKEN env)")
    # list
    lb_list = lb_sub.add_parser("list", help="Show global leaderboard top entries")
    lb_list.add_argument("--top",   type=int, default=20)
    lb_list.add_argument("--url",   default=None)
    lb_list.add_argument("--token", default=None)
    lb_p.set_defaults(func=cmd_leaderboard)

    # changelog
    p = sub.add_parser("changelog", help="Print CHANGELOG entry for the installed version")
    p.add_argument("--version", default=None, help="Show entry for this version (default: installed)")
    p.set_defaults(func=cmd_changelog)

    # triple-horizon sweep
    p = sub.add_parser("triple-horizon-sweep",
                       help="E11-2 portability sweep over (a,K) grid")
    p.add_argument("--a-grid",            default="1.01,1.02,1.03",  dest="a_grid")
    p.add_argument("--K-grid",            default="0.4,0.8,1.2",     dest="K_grid")
    p.add_argument("--delta",             type=int,   default=20)
    p.add_argument("--readout-tau-scale", type=float, default=25.0,  dest="readout_tau_scale")
    p.add_argument("--epsilon",           type=float, default=0.01)
    p.add_argument("--tau-max",           type=int,   default=50,    dest="tau_max")
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--out",               default=None)
    p.set_defaults(func=cmd_triple_horizon_sweep)

    # triple-horizon
    p = sub.add_parser("triple-horizon", help="E11-1 Triple Horizon benchmark (SPHERE_23)")
    p.add_argument("--a",                type=float, default=1.02)
    p.add_argument("--K",                type=float, default=0.8)
    p.add_argument("--delta",            type=int,   default=20)
    p.add_argument("--readout-tau-scale",type=float, default=25.0, dest="readout_tau_scale")
    p.add_argument("--shield",           type=float, default=0.0)
    p.add_argument("--x-fail",           type=float, default=500.0, dest="x_fail")
    p.add_argument("--epsilon",          type=float, default=0.01)
    p.add_argument("--tau-max",          type=int,   default=50,  dest="tau_max")
    p.add_argument("--n-null",           type=int,   default=30,  dest="n_null")
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--out",              default=None)
    p.set_defaults(func=cmd_triple_horizon)

    # latent (LERI)
    p = sub.add_parser("latent", help="LERI: Latent Environmental Residual Inference access horizon")
    p.add_argument("--P",        type=float, default=1.0,   help="Signal power (action variance)")
    p.add_argument("--sigma0",   type=float, default=1.0,   help="Baseline noise σ²₀")
    p.add_argument("--alpha",    type=float, default=0.25,  help="Noise growth rate α")
    p.add_argument("--epsilon",  type=float, default=0.01,  help="Detectability threshold ε (bits)")
    p.add_argument("--beta",     type=float, default=0.99,  help="Null quantile β")
    p.add_argument("--persist",  type=int,   default=5,     help="Persistence gate m steps")
    p.add_argument("--tau-max",  type=int,   default=400,   dest="tau_max")
    p.add_argument("--n-null",   type=int,   default=50,    dest="n_null")
    p.add_argument("--seed",     type=int,   default=42)
    p.add_argument("--out",      default=None)
    p.set_defaults(func=cmd_latent)

    # diagnose
    p = sub.add_parser("diagnose", help="Plain-English diagnosis of a FAWP result JSON")
    p.add_argument("--data", required=True, help="Path to result .json file")
    p.set_defaults(func=cmd_diagnose)

    # sweep
    p = sub.add_parser("sweep", help="Sweep parameters and show FAWP detection sensitivity")
    p.add_argument("--data",    required=True, help="Path to scan JSON with pred_mi/steer_mi")
    p.add_argument("--epsilon", default="0.005,0.01,0.02",
                   help="Comma-separated epsilon values (default: 0.005,0.01,0.02)")
    p.add_argument("--tau-max", default="20,30,40", dest="tau_max",
                   help="Comma-separated tau_max values (default: 20,30,40)")
    p.add_argument("--n-null",  default="0,50,100", dest="n_null",
                   help="Comma-separated n_null values (default: 0,50,100)")
    p.add_argument("--out",     default=None)
    p.set_defaults(func=cmd_sweep)

    # export
    p = sub.add_parser("export", help="Export a FAWP result JSON to csv/html/pdf")
    p.add_argument("--data",   required=True, help="Path to result .json file")
    p.add_argument("--out",    default=None,  help="Output path (auto-detected from format)")
    p.add_argument("--format", default=None,  choices=["csv","html","pdf","parquet"],
                   help="Output format (default: inferred from --out extension)")
    p.add_argument("--title",  default=None)
    p.add_argument("--mode",   default="report", choices=["report","lab"])
    p.set_defaults(func=cmd_export)

    # compare
    p = sub.add_parser("compare", help="Compare FAWP signals for two assets or weather locations")
    p.add_argument("--mode",    default="finance", choices=["finance","weather"])
    # finance mode
    p.add_argument("--a",       default="SPY",  help="First ticker")
    p.add_argument("--b",       default="QQQ",  help="Second ticker")
    p.add_argument("--period",  default="2y")
    p.add_argument("--window",  type=int,   default=252)
    p.add_argument("--step",    type=int,   default=21)
    # weather mode
    p.add_argument("--lat-a",   type=float, default=51.5,  dest="lat_a")
    p.add_argument("--lon-a",   type=float, default=-0.1,  dest="lon_a")
    p.add_argument("--lat-b",   type=float, default=48.9,  dest="lat_b")
    p.add_argument("--lon-b",   type=float, default=2.4,   dest="lon_b")
    p.add_argument("--variable",default="temperature_2m")
    p.add_argument("--start",   default="2010-01-01")
    p.add_argument("--end",     default="2024-12-31")
    p.add_argument("--horizon", type=int,   default=7)
    # shared
    p.add_argument("--epsilon", type=float, default=0.01)
    p.add_argument("--n-null",  type=int,   default=50,   dest="n_null")
    p.add_argument("--out",     default=None)
    p.set_defaults(func=cmd_compare)

    # notebook
    p = sub.add_parser("notebook", help="Open the E9 replication notebook in Jupyter")
    p.add_argument("--lab", action="store_true", help="Open in JupyterLab instead of Jupyter Notebook")
    p.set_defaults(func=cmd_notebook)

    # status
    p = sub.add_parser("status", help="Print live project status and calibration checks")
    p.set_defaults(func=cmd_status)

    # backtest
    p = sub.add_parser("backtest", help="Run rolling FAWP scan and export CSV")
    p.add_argument("--ticker",  required=True)
    p.add_argument("--period",  default="5y")
    p.add_argument("--window",  type=int,   default=252)
    p.add_argument("--step",    type=int,   default=21)
    p.add_argument("--delta",   type=int,   default=20)
    p.add_argument("--tau-max", type=int,   default=40,   dest="tau_max")
    p.add_argument("--epsilon", type=float, default=0.01)
    p.add_argument("--n-null",  type=int,   default=50,   dest="n_null")
    p.add_argument("--out",     default=None)
    p.set_defaults(func=cmd_backtest)

    # watch
    p = sub.add_parser("watch", help="Run repeated scans and alert when FAWP fires")
    p.add_argument("--tickers",  required=True, help="Comma-separated tickers, e.g. SPY,QQQ,GLD")
    p.add_argument("--period",   default="2y",  help="Data period (default: 2y)")
    p.add_argument("--interval", type=int, default=3600, help="Seconds between scans (default: 3600)")
    p.add_argument("--email",    default=None, help="Email address for alerts")
    p.add_argument("--epsilon",  type=float, default=0.01)
    p.add_argument("--n-null",   type=int,   default=50,  dest="n_null")
    p.add_argument("--window",   type=int,   default=252)
    p.add_argument("--step",     type=int,   default=21)
    p.add_argument("--once",     action="store_true", help="Run once and exit (no loop)")
    p.set_defaults(func=cmd_watch)

    # report
    p = sub.add_parser("report", help="Generate a PDF report from a FAWP result JSON")
    p.add_argument("--ticker",  default=None, help="Ticker to scan inline (skips --data requirement)")
    p.add_argument("--period",  default="2y")
    p.add_argument("--epsilon", type=float, default=0.01)
    p.add_argument("--n-null",  type=int, default=50, dest="n_null")
    p.add_argument("--data",   required=True, help="Path to result .json file")
    p.add_argument("--out",    default=None,  help="Output PDF path (default: <data>.pdf)")
    p.add_argument("--title",  default=None,  help="Report title")
    p.add_argument("--mode",   default="report", choices=["report","lab"])
    p.add_argument("--author", default="Ralph Clayton")
    p.add_argument("--doi",    default="10.5281/zenodo.18673949")
    p.set_defaults(func=cmd_report)

    # forecast
    p = sub.add_parser("forecast", help="Run FAWP detection on NWP forecast vs observation CSVs")
    p.add_argument("--forecast",     required=True, help="Path to forecast CSV")
    p.add_argument("--obs",          required=True, help="Path to observed CSV")
    p.add_argument("--variable",     default="temperature", help="Variable name (for labelling)")
    p.add_argument("--location",     default="uploaded",    help="Location label")
    p.add_argument("--forecast-col", default="forecast",    dest="forecast_col")
    p.add_argument("--obs-col",      default="observed",    dest="obs_col")
    p.add_argument("--date-col",     default="date",        dest="date_col")
    p.add_argument("--horizon",      type=int,   default=1)
    p.add_argument("--tau-max",      type=int,   default=40,  dest="tau_max")
    p.add_argument("--epsilon",      type=float, default=0.01)
    p.add_argument("--n-null",       type=int,   default=100, dest="n_null")
    p.add_argument("--out",          default=None, help="Save to .json or .csv")
    p.set_defaults(func=cmd_forecast)

    p = sub.add_parser("grid", help="Generate FAWP detection basin heatmap over (a, K) space")
    p.add_argument("--a-min",  type=float, default=1.00)
    p.add_argument("--a-max",  type=float, default=1.10)
    p.add_argument("--k-min",  type=float, default=0.60)
    p.add_argument("--k-max",  type=float, default=0.95)
    p.add_argument("--n",      type=int,   default=10, help="Grid points per axis")
    p.add_argument("--seeds",  type=int,   default=3,  help="Seeds per cell")
    p.add_argument("--out",    default="fawp_basin.csv")
    p.add_argument("--plot",   action="store_true", help="Save heatmap PNG alongside CSV")
    p.set_defaults(func=cmd_grid)

    return parser



def cmd_cite(args):
    """Print BibTeX citations for fawp-index and its papers."""
    import fawp_index as _fi
    ver = _fi.__version__

    bib = f"""
% ─────────────────────────────────────────────────────────────
%  fawp-index — citation guide
%  Run:  fawp-index cite
% ─────────────────────────────────────────────────────────────

% 1. Software (PyPI package + this repo)
@software{{fawp_index_software,
  author    = {{Clayton, Ralph}},
  title     = {{fawp-index: FAWP Alpha Index — Information-Control Exclusion Principle detector}},
  year      = {{2026}},
  version   = {{{ver}}},
  doi       = {{10.5281/zenodo.18673949}},
  url       = {{https://github.com/DrRalphClayton/fawp-index}},
  license   = {{MIT}}
}}

% 2. E9 Suite (comparative timing, gap2, robustness)
@article{{fawp_e9_suite,
  author    = {{Clayton, Ralph}},
  title     = {{FAWP Alpha Index — E9 Suite: Comparative timing, gap2 detector, robustness}},
  year      = {{2026}},
  doi       = {{10.5281/zenodo.19065421}},
  url       = {{https://doi.org/10.5281/zenodo.19065421}}
}}

% 3. E8 / SPHERE-16 flagship calibration
@article{{fawp_sphere16,
  author    = {{Clayton, Ralph}},
  title     = {{FAWP Alpha Index — SPHERE-16: E8 Flagship Calibration}},
  year      = {{2026}},
  doi       = {{10.5281/zenodo.18673949}},
  url       = {{https://doi.org/10.5281/zenodo.18673949}}
}}

% 4. E1–E7 foundational theory (Volumetric Time Model)
@article{{fawp_e1_e7,
  author    = {{Clayton, Ralph}},
  title     = {{FAWP Alpha Index — E1–E7: Volumetric Time Model}},
  year      = {{2026}},
  doi       = {{10.5281/zenodo.18663547}},
  url       = {{https://doi.org/10.5281/zenodo.18663547}}
}}

% 5. Research data (Figshare)
@misc{{fawp_figshare,
  author    = {{Clayton, Ralph}},
  title     = {{fawp-index: research data and supplementary materials}},
  year      = {{2026}},
  doi       = {{10.6084/m9.figshare.31799104}},
  url       = {{https://doi.org/10.6084/m9.figshare.31799104}}
}}

% 6. OSF project
@misc{{fawp_osf,
  author    = {{Clayton, Ralph}},
  title     = {{FAWP Alpha Index — OSF Project}},
  year      = {{2026}},
  url       = {{https://osf.io/hzwgp/}}
}}
"""
    print(bib.strip())



def cmd_timing(_args):
    """Print E9.7 detector timing results."""
    import fawp_index as fi
    print()
    print("E9.7 Detector Timing — 4,244-run sweep (a=1.02, K=0.8)")
    print("=" * 58)
    print(f"  {'Detector':<28} {'Leads cliff (u)':<16} {'Leads cliff (ξ)':<16} {'Err vs ODW'}")
    print(f"  {'-'*28} {'-'*16} {'-'*16} {'-'*12}")
    _g_u  = f"+{fi.E97_MEAN_LEAD_GAP2_TO_CLIFF_U:.4f}"
    _g_xi = f"+{fi.E97_MEAN_LEAD_GAP2_TO_CLIFF_XI:.4f}"
    _a2_u = f"+{fi.E97_MEAN_LEAD_ALPHA2_TO_CLIFF_U:.4f}"
    _a2_xi= f"+{fi.E97_MEAN_LEAD_ALPHA2_TO_CLIFF_XI:.4f}"
    _a_u  = f"{fi.E97_MEAN_LEAD_ALPHA_TO_CLIFF_U:.4f}"
    _a_xi = f"{fi.E97_MEAN_LEAD_ALPHA_TO_CLIFF_XI:.4f}"
    print(f"  {'gap2 peak (raw lever. gap)':<28} "
          f"{_g_u:<16} {_g_xi:<16} "
          f"~{fi.E97_MEAN_ABS_ERR_GAP2_VS_ODW_START:.1f} delays  ✅ BEST")
    print(f"  {'α₂  (SPHERE-16)':<28} "
          f"{_a2_u:<16} {_a2_xi:<16} "
          f"~{fi.E97_MEAN_ABS_ERR_ALPHA2_VS_ODW_START:.1f} delays  ✅ Good")
    print(f"  {'α   (baseline, old)':<28} "
          f"{_a_u:<16} {_a_xi:<16} "
          f"~{fi.E97_MEAN_ABS_ERR_ALPHA_VS_ODW_START:.1f} delays  ❌ Lags")
    print()
    print(f"  Total runs : {fi.E97_N_RUNS}")
    print(f"  Mean τf    : {fi.E97_MEAN_TAU_F}")
    print(f"  Source     : doi:10.5281/zenodo.19065421")
    print()


def cmd_verify(args):
    """Run calibration self-checks against published SPHERE/E9 constants."""
    if getattr(args, "sphere23", False):
        return cmd_verify_sphere23(args)

    import fawp_index as fi
    from fawp_index.data import E9_2_SUMMARY_JSON
    import json

    checks = []
    def chk(name, got, expected, tol=0.01):
        ok = abs(float(got) - float(expected)) <= tol
        checks.append((name, got, expected, ok))

    # SPHERE-16 calibration (E8 flagship)
    chk("PEAK_PRED_BITS",      fi.PEAK_PRED_BITS,      2.233669,  1e-4)
    chk("ETA_PRED_CORRECTED",  fi.ETA_PRED_CORRECTED,  0.0,       1e-6)
    chk("PRED_AT_CLIFF",       fi.PRED_AT_CLIFF,       1.01,      0.01)
    chk("NULL_MAX_SHUFFLE_E8", fi.NULL_MAX_SHUFFLE_E8, 0.00216,   1e-4)
    chk("NULL_MAX_SHIFT_E8",   fi.NULL_MAX_SHIFT_E8,   0.00421,   1e-4)

    # E9 / flagship ODW constants (correct exported names)
    chk("TAU_PLUS_H_FLAGSHIP", fi.TAU_PLUS_H_FLAGSHIP, 4,   0)
    chk("TAU_F_FLAGSHIP",      fi.TAU_F_FLAGSHIP,      35,  0)
    chk("TAU_PLUS_H_E9",       fi.TAU_PLUS_H_E9,       31,  0)
    chk("TAU_F_E9",            fi.TAU_F_E9,            36,  0)
    chk("ODW_START_E9",        fi.ODW_START_E9,        31,  0)
    chk("ODW_END_E9",          fi.ODW_END_E9,          33,  0)

    # E9.7 timing
    chk("E97_MEAN_LEAD_GAP2_TO_CLIFF_U",  fi.E97_MEAN_LEAD_GAP2_TO_CLIFF_U,  0.7552, 1e-3)
    chk("E97_MEAN_LEAD_ALPHA_TO_CLIFF_U", fi.E97_MEAN_LEAD_ALPHA_TO_CLIFF_U, -2.004, 1e-3)
    chk("E97_MEAN_ABS_ERR_GAP2_VS_ODW_START", fi.E97_MEAN_ABS_ERR_GAP2_VS_ODW_START, 2.108, 0.01)

    # SPHERE_23 Triple Horizon (Experiment 11, March 2026)
    chk("ALPHA_A",            fi.ALPHA_A,             0.007297,           1e-5)
    chk("ALPHA_A_SQ",         fi.ALPHA_A_SQ,          5.325135447834e-5,  1e-10)
    chk("E11_TAU_ALPHA",       fi.E11_TAU_ALPHA,       10,                 0)
    chk("E11_TAU_PLUS_H",      fi.E11_TAU_PLUS_H,      12,                 0)
    chk("E11_TAU_F",           fi.E11_TAU_F,           29,                 0)
    chk("E11_TAU_ALPHA2",      fi.E11_TAU_ALPHA2,      32,                 0)
    chk("E11_TAU_READOUT",     fi.E11_TAU_READOUT,     38,                 0)

    # Verify bundled E9.2 data is loadable
    try:
        with open(E9_2_SUMMARY_JSON) as f:
            s = json.load(f)
        assert s["aggregate_summary"]["fawp_found_u"] is True
        checks.append(("E9.2 data loadable + FAWP found", True, True, True))
    except Exception as e:
        checks.append((f"E9.2 data error: {e}", None, None, False))

    passed = sum(1 for *_, ok in checks if ok)
    print()
    print(f"fawp-index calibration verification — {len(checks)} checks")
    print("=" * 55)
    for name, got, exp, ok in checks:
        icon = "✅" if ok else "❌"
        val = f"{got}" if got is True or got is None else f"{got}"
        print(f"  {icon}  {name:<42} {val}")
    print()
    print(f"  {passed}/{len(checks)} checks passed")
    if passed < len(checks):
        print("  ⚠️  Some checks failed — reinstall fawp-index or check your version")
    else:
        print("  ✅ All calibration checks pass")
    print()


def cmd_grid(args):
    """Generate a FAWP detection basin heatmap over (a, K) parameter space."""
    import numpy as np
    import pandas as pd

    a_vals = np.linspace(args.a_min, args.a_max, args.n)
    K_vals = np.linspace(args.k_min, args.k_max, args.n)

    print(f"FAWP basin scan: a∈[{args.a_min},{args.a_max}] × K∈[{args.k_min},{args.k_max}] n={args.n}×{args.n}")
    print(f"Seeds per cell: {args.seeds}  |  Output: {args.out}")
    print()

    # Inline synthetic delayed plant — no external imports needed
    from fawp_index.weather import _compute_weather_mi_curves

    rows = []
    total = len(a_vals) * len(K_vals)
    done = 0

    for a in a_vals:
        for K in K_vals:
            fawp_hits = 0
            peak_gaps = []
            for seed in range(args.seeds):
                try:
                    rng = np.random.default_rng(seed + 42000)
                    n = 500
                    x = np.zeros(n)
                    u = np.zeros(n)
                    # Delayed plant: x_{t} = a*x_{t-1} + u_t + noise
                    #                u_t   = -K*x_{t-1} + control_noise
                    for t in range(1, n):
                        u[t] = -K * x[t-1] + rng.normal(0, 0.5)
                        x[t] = a * x[t-1] + u[t] + rng.normal(0, 1.0)

                    delta = 20
                    nn    = n - delta
                    pred   = x[:nn]
                    future = x[delta:delta + nn]
                    steer  = np.diff(x)[:nn]

                    odw, _, _, _ = _compute_weather_mi_curves(
                        pred, future, steer, tau_max=40, n_null=20, epsilon=0.01
                    )
                    if odw.fawp_found:
                        fawp_hits += 1
                    peak_gaps.append(odw.peak_gap_bits)
                except Exception:
                    pass

            detect_rate = fawp_hits / max(1, args.seeds)
            mean_gap    = float(np.mean(peak_gaps)) if peak_gaps else 0.0
            rows.append({"a": round(a, 4), "K": round(K, 4),
                         "fawp_rate": round(detect_rate, 3),
                         "mean_peak_gap_bits": round(mean_gap, 4)})
            done += 1
            if done % 5 == 0 or done == total:
                print(f"  [{done:3d}/{total}] a={a:.3f} K={K:.3f} → rate={detect_rate:.2f} gap={mean_gap:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nSaved: {args.out}  ({len(df)} grid points)")

    # Optionally save heatmap
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            pivot = df.pivot(index="K", columns="a", values="fawp_rate")
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="YlOrRd",
                           extent=[pivot.columns.min(), pivot.columns.max(),
                                   pivot.index.min(), pivot.index.max()])
            plt.colorbar(im, ax=ax, label="FAWP detection rate")
            ax.set_xlabel("a (system gain)")
            ax.set_ylabel("K (control gain)")
            ax.set_title(f"FAWP Basin — {args.n}×{args.n} grid, {args.seeds} seeds/cell")
            png = args.out.replace(".csv", ".png")
            fig.savefig(png, dpi=150, bbox_inches="tight")
            print(f"Heatmap: {png}")
        except Exception as e:
            print(f"Plot skipped: {e}")
    print()


def cmd_forecast(args):
    """Run FAWP detection on NWP forecast vs observation CSVs."""
    import pandas as pd
    from fawp_index.weather import fawp_from_nwp_csvs

    print(f"FAWP NWP Forecast Verification")
    print(f"  Forecast : {args.forecast}")
    print(f"  Observed : {args.obs}")
    print(f"  Variable : {args.variable}")
    print()

    result = fawp_from_nwp_csvs(
        forecast_path    = args.forecast,
        observed_path    = args.obs,
        forecast_col     = args.forecast_col,
        observed_col     = args.obs_col,
        date_col         = args.date_col,
        variable         = args.variable,
        location         = args.location,
        horizon_days     = args.horizon,
        tau_max          = args.tau_max,
        epsilon          = args.epsilon,
        n_null           = args.n_null,
    )

    print(result.summary())

    if args.out:
        import json as _j
        if args.out.endswith(".json"):
            with open(args.out, "w") as f:
                _j.dump(result.to_dict(), f, indent=2)
            print(f"Saved → {args.out}")
        elif args.out.endswith(".csv"):
            df = pd.DataFrame({
                "tau": result.tau,
                "pred_mi": result.pred_mi,
                "steer_mi": result.steer_mi,
            })
            df.to_csv(args.out, index=False)
            print(f"Saved → {args.out}")


def cmd_report(args):
    """Generate a PDF report from a FAWP result JSON file, or run inline scan first."""
    # --ticker shortcut: scan inline then report
    if getattr(args, "ticker", None):
        print(f"Running inline scan for {args.ticker} [{args.period}]…")
        try:
            import yfinance as _yf_r, json as _j_r, tempfile as _tmp_r
            from fawp_index.market import scan_fawp_market as _sfm
            _df_r = _yf_r.download(args.ticker, period=args.period,
                                   auto_adjust=True, progress=False)
            if _df_r.empty:
                print(f"No data for {args.ticker}"); return
            _epsilon_r = getattr(args, "epsilon", 0.01)
            _n_null_r  = getattr(args, "n_null",  50)
            _r_scan = _sfm(_df_r, ticker=args.ticker,
                           epsilon=_epsilon_r, n_null=_n_null_r, verbose=False)
            _tf = _tmp_r.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
            _j_r.dump(getattr(_r_scan, "__dict__", {}), _tf, default=str)
            _tf.close()
            args.data = _tf.name
            print("Scan complete — generating report…")
        except ImportError:
            print("yfinance required: pip install yfinance"); return
    import json
    from fawp_index.report import generate_report

    with open(args.data) as f:
        result = json.load(f)

    out = args.out or args.data.replace(".json", ".pdf")
    path = generate_report(
        result      = result,
        output_path = out,
        title       = args.title,
        mode        = args.mode,
        author      = args.author,
        doi         = args.doi,
    )
    print(f"Report saved → {path}")


def cmd_watch(args):
    """Run repeated FAWP scans on a schedule and alert when regimes fire."""
    import time, json
    from datetime import datetime

    print(f"FAWP Watch daemon")
    print(f"  Tickers  : {args.tickers}")
    print(f"  Period   : {args.period}")
    print(f"  Interval : every {args.interval} seconds ({args.interval/3600:.1f} hours)")
    print(f"  Email    : {args.email or 'disabled'}")
    print()

    def _run_scan():
        try:
            import yfinance as yf
            import pandas as pd
            from fawp_index.market import MarketScanConfig, scan_market

            tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
            dfs = {}
            for t in tickers:
                try:
                    df = yf.download(t, period=args.period, auto_adjust=True, progress=False)
                    if not df.empty and "Close" in df.columns:
                        dfs[t] = df[["Close"]].rename(columns={"Close": "close"})
                except Exception:
                    pass

            if not dfs:
                print(f"[{datetime.now():%H:%M:%S}] No data fetched")
                return

            cfg = MarketScanConfig(epsilon=args.epsilon, n_null=args.n_null,
                                   window=args.window, step=args.step)
            result = scan_market(dfs, config=cfg)

            flagged = [a for a in result.active_regimes() if a.regime_active]
            ts = datetime.now().strftime("%Y-%m-%d %H:%M")
            print(f"[{ts}] {len(dfs)} tickers scanned · {len(flagged)} FAWP active", flush=True)

            for a in flagged:
                print(f"  🔴 {a.ticker} — score {a.latest_score:.4f} · gap {a.peak_gap_bits:.4f}b",
                      flush=True)

            if flagged and args.email:
                try:
                    import sys, os
                    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dashboard"))
                    from email_digest import send_digest
                    send_digest(args.email, finance_results=[a.__dict__ for a in flagged],
                                scan_date=ts)
                    print(f"  📧 Alert email sent to {args.email}", flush=True)
                except Exception as e:
                    print(f"  Email failed: {e}", flush=True)

        except Exception as e:
            print(f"[{datetime.now():%H:%M:%S}] Scan error: {e}", flush=True)

    while True:
        _run_scan()
        if args.once:
            break
        print(f"  Next scan in {args.interval}s…", flush=True)
        time.sleep(args.interval)


def cmd_status(_args):
    """Print live project status: PyPI version, CI, calibration checks."""
    import urllib.request, json as _j
    import fawp_index as fi

    print()
    print("fawp-index project status")
    print("=" * 48)

    # Installed version
    print(f"  Installed version : {fi.__version__}")

    # Latest PyPI version
    try:
        with urllib.request.urlopen(
            "https://pypi.org/pypi/fawp-index/json", timeout=4
        ) as r:
            _pypi = _j.loads(r.read())
        latest = _pypi["info"]["version"]
        up_to_date = fi.__version__ == latest
        icon = "✅" if up_to_date else "⬆️ "
        print(f"  PyPI latest       : {latest}  {icon}")
    except Exception:
        print("  PyPI latest       : (unavailable)")

    # PyPI downloads
    try:
        with urllib.request.urlopen(
            "https://pypistats.org/api/packages/fawp-index/recent", timeout=4
        ) as r:
            _dl = _j.loads(r.read())
        dm = _dl.get("data", {}).get("last_month", 0)
        print(f"  Downloads/month   : {dm:,}")
    except Exception:
        print("  Downloads/month   : (unavailable)")

    # Calibration self-check
    print()
    print("  Calibration checks:")
    checks = [
        ("PEAK_PRED_BITS",             fi.PEAK_PRED_BITS,             2.233669, 1e-4),
        ("TAU_PLUS_H_FLAGSHIP",        fi.TAU_PLUS_H_FLAGSHIP,        4,        0),
        ("TAU_F_FLAGSHIP",             fi.TAU_F_FLAGSHIP,             35,       0),
        ("E97_MEAN_LEAD_GAP2_U",       fi.E97_MEAN_LEAD_GAP2_TO_CLIFF_U, 0.7552, 1e-3),
        ("E97_MEAN_ABS_ERR_GAP2",      fi.E97_MEAN_ABS_ERR_GAP2_VS_ODW_START, 2.108, 0.01),
    ]
    all_ok = True
    for name, val, expected, tol in checks:
        ok = abs(float(val) - float(expected)) <= tol
        all_ok = all_ok and ok
        print(f"    {'✅' if ok else '❌'} {name} = {val}")

    print()
    print(f"  Calibration : {'✅ PASS' if all_ok else '❌ FAIL'}")
    print()
    print("  E9.7 detector timing (4,244-run sweep):")
    print(f"    gap2 peak  leads cliff by  +{fi.E97_MEAN_LEAD_GAP2_TO_CLIFF_U:.4f} delays  ✅ BEST")
    print(f"    α₂         leads cliff by  +{fi.E97_MEAN_LEAD_ALPHA2_TO_CLIFF_U:.4f} delays  ✅ Good")
    print(f"    α baseline lags  cliff by   {fi.E97_MEAN_LEAD_ALPHA_TO_CLIFF_U:.4f} delays  ❌ Lags")
    print(f"    ODW localisation error: ~{fi.E97_MEAN_ABS_ERR_GAP2_VS_ODW_START:.1f} delays")
    print()
    print(f"  Docs        : https://fawp-index.readthedocs.io")
    print(f"  GitHub      : https://github.com/DrRalphClayton/fawp-index")
    print(f"  Live demo   : https://fawp-scanner.info")
    print()


def cmd_backtest(args):
    """Run rolling FAWP scan over historical data and export results."""
    import pandas as pd
    from fawp_index.market import scan_fawp_market

    print(f"FAWP Backtest")
    print(f"  Ticker  : {args.ticker}")
    print(f"  Period  : {args.period}")
    print(f"  Window  : {args.window} bars  Step: {args.step}")
    print()

    try:
        import yfinance as yf
        df = yf.download(args.ticker, period=args.period,
                         auto_adjust=True, progress=False)
        if df.empty:
            print(f"No data for {args.ticker}")
            return
    except ImportError:
        print("yfinance required: pip install yfinance")
        return

    result = scan_fawp_market(
        df, ticker=args.ticker,
        window=args.window, step=args.step,
        delta_pred=args.delta, tau_max=args.tau_max,
        epsilon=args.epsilon, n_null=args.n_null,
        verbose=True,
    )

    # Build output DataFrame
    rows = []
    for w in result.windows:
        rows.append({
            "date":        str(w.date)[:10] if hasattr(w, "date") else "",
            "fawp_active": w.regime_active,
            "score":       round(w.regime_score, 6),
            "peak_gap_bits": round(w.peak_gap_bits, 6),
            "odw_start":   w.odw_start,
            "odw_end":     w.odw_end,
            "tau_h_plus":  w.tau_h_plus,
        })

    out_df = pd.DataFrame(rows)
    n_fawp = int(out_df["fawp_active"].sum())
    print(f"\n  {len(out_df)} windows · {n_fawp} FAWP active ({n_fawp/max(1,len(out_df))*100:.1f}%)")

    out = args.out or f"fawp_backtest_{args.ticker}.csv"
    out_df.to_csv(out, index=False)
    print(f"  Saved → {out}")


def cmd_notebook(args):
    """Open the E9 replication notebook in Jupyter."""
    import subprocess, sys, os
    from pathlib import Path

    # Find the notebook — prefer repo clone, fall back to installed data path
    candidates = [
        Path(__file__).parent.parent / "notebooks" / "E9_full_replication.ipynb",
        Path(sys.prefix) / "share" / "fawp-index" / "notebooks" / "E9_full_replication.ipynb",
    ]
    nb_path = next((p for p in candidates if p.exists()), None)

    if nb_path is None:
        print("Notebook not found. Clone the repo:")
        print("  git clone https://github.com/DrRalphClayton/fawp-index")
        print("  cd fawp-index && fawp-index notebook")
        return

    print(f"Opening: {nb_path}")

    if args.lab:
        cmd = [sys.executable, "-m", "jupyter", "lab", str(nb_path)]
        label = "JupyterLab"
    else:
        cmd = [sys.executable, "-m", "jupyter", "notebook", str(nb_path)]
        label = "Jupyter Notebook"

    print(f"Starting {label}…")
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print(f"Jupyter not installed. Install it:")
        print("  pip install jupyter")
    except KeyboardInterrupt:
        print("\nNotebook server stopped.")


def cmd_compare(args):
    """Compare FAWP signals for two assets or two weather locations side by side."""
    import pandas as pd

    if args.mode == "finance":
        try:
            import yfinance as yf
            print(f"FAWP Comparison: {args.a} vs {args.b}  [{args.period}]")
            print()
            from fawp_index.market import scan_fawp_market
            results = {}
            for ticker in [args.a, args.b]:
                df = yf.download(ticker, period=args.period, auto_adjust=True, progress=False)
                if df.empty:
                    print(f"  {ticker}: no data")
                    continue
                r = scan_fawp_market(df, ticker=ticker, window=args.window,
                                     step=args.step, epsilon=args.epsilon,
                                     n_null=args.n_null, verbose=False)
                results[ticker] = r
        except ImportError:
            print("yfinance required: pip install yfinance")
            return
    elif args.mode == "weather":
        from fawp_index.weather import fawp_from_open_meteo
        print(f"FAWP Comparison: ({args.lat_a},{args.lon_a}) vs ({args.lat_b},{args.lon_b})")
        print(f"  Variable : {args.variable}  Period: {args.start} → {args.end}")
        print()
        results = {}
        for name, lat, lon in [("A", args.lat_a, args.lon_a), ("B", args.lat_b, args.lon_b)]:
            r = fawp_from_open_meteo(latitude=lat, longitude=lon, variable=args.variable,
                                     start_date=args.start, end_date=args.end,
                                     horizon_days=args.horizon, epsilon=args.epsilon,
                                     n_null=args.n_null)
            results[f"Location {name} ({lat:.2f},{lon:.2f})"] = r

    # Print comparison table
    print(f"{'Label':<28} {'FAWP':>6} {'Peak gap':>10} {'τ⁺ₕ':>5} {'τf':>5} {'n obs':>7}")
    print("-" * 62)
    for label, r in results.items():
        fawp = "🔴 YES" if getattr(r, "fawp_found", False) or getattr(r, "in_fawp", [False])[-1] else "—"
        gap  = getattr(r, "peak_gap_bits", 0) or 0
        tau_h = getattr(r.odw_result if hasattr(r,"odw_result") else r, "tau_h_plus", None)
        tau_h = str(tau_h) if tau_h is not None else "—"
        tau_f = getattr(r.odw_result if hasattr(r,"odw_result") else r, "tau_f",     None)
        tau_f = str(tau_f) if tau_f is not None else "—"
        n_obs = getattr(r, "n_obs", getattr(r, "n_windows", "—"))
        print(f"{label:<28} {fawp:>6} {gap:>10.4f} {str(tau_h):>5} {str(tau_f):>5} {str(n_obs):>7}")
    print()

    if args.out:
        rows = []
        for label, r in results.items():
            rows.append({"label": label,
                         "fawp": bool(getattr(r,"fawp_found",False)),
                         "peak_gap_bits": getattr(r,"peak_gap_bits",0) or 0})
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"Saved → {args.out}")


def cmd_export(args):
    """Export a FAWP result JSON to CSV, HTML, or PDF."""
    import json
    from pathlib import Path

    with open(args.data) as f:
        result = json.load(f)

    fmt = args.format or Path(args.out).suffix.lstrip('.') if args.out else "csv"
    out = args.out or args.data.replace(".json", f".{fmt}")

    print(f"Exporting {args.data} → {out} [{fmt}]")

    if fmt == "csv":
        import pandas as pd
        if isinstance(result, list):
            df = pd.DataFrame(result)
        elif "assets" in result:
            df = pd.DataFrame(result["assets"])
        else:
            df = pd.DataFrame([result])
        df.to_csv(out, index=False)

    elif fmt == "html":
        try:
            from fawp_index.exports import odw_to_html
            odw_to_html(result, out)
        except Exception:
            with open(out, "w") as fh:
                fh.write(f"<pre>{json.dumps(result, indent=2)}</pre>")

    elif fmt == "pdf":
        from fawp_index.report import generate_report
        generate_report(result=result, output_path=out,
                        title=args.title or f"FAWP Export — {Path(args.data).stem}",
                        mode=args.mode)
    else:
        print(f"Unknown format: {fmt}. Use csv, html, pdf, or parquet.")
        return

    print(f"Saved → {out}")


def cmd_sweep(args):
    """Sweep (epsilon, tau_max, n_null) and show FAWP detection sensitivity table."""
    import json, itertools
    import pandas as pd
    from fawp_index.weather import _compute_weather_mi_curves
    import numpy as np

    print(f"FAWP parameter sweep on: {args.data}")
    with open(args.data) as f:
        d = json.load(f)

    # Extract series — support weather or finance JSON
    if "pred_mi" in d and "steer_mi" in d:
        pred_mi  = np.array(d["pred_mi"],  dtype=float)
        steer_mi = np.array(d["steer_mi"], dtype=float)
        tau      = np.array(d.get("tau", list(range(1, len(pred_mi)+1))), dtype=int)
        _has_raw = False
    else:
        print("JSON must contain pred_mi + steer_mi arrays (from a weather or finance scan).")
        return

    epsilons  = [float(e) for e in args.epsilon.split(",")]
    tau_maxes = [int(t)   for t in args.tau_max.split(",")]
    n_nulls   = [int(n)   for n in args.n_null.split(",")]

    rows = []
    from fawp_index.detection.odw import ODWDetector
    from fawp_index.constants import PERSISTENCE_RULE_M, PERSISTENCE_RULE_N

    for eps, tm, nn in itertools.product(epsilons, tau_maxes, n_nulls):
        # Trim curves to tau_max
        mask = tau <= tm
        pm = pred_mi[mask]; sm = steer_mi[mask]; ta = tau[mask]
        fail = np.zeros(len(ta))
        det = ODWDetector(epsilon=eps, persistence_m=PERSISTENCE_RULE_M,
                          persistence_n=PERSISTENCE_RULE_N)
        try:
            odw = det.detect(tau=ta, pred_corr=pm, steer_corr=sm, fail_rate=fail)
            rows.append({
                "epsilon": eps, "tau_max": tm, "n_null": nn,
                "fawp":         odw.fawp_found,
                "peak_gap":     round(odw.peak_gap_bits, 4),
                "odw_start":    odw.odw_start,
                "odw_end":      odw.odw_end,
                "tau_h_plus":   odw.tau_h_plus,
            })
        except Exception as e:
            rows.append({"epsilon": eps, "tau_max": tm, "n_null": nn,
                         "fawp": None, "peak_gap": None, "error": str(e)})

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    out = args.out or args.data.replace(".json", "_sweep.csv")
    df.to_csv(out, index=False)
    n_fawp = int(df["fawp"].sum()) if "fawp" in df else 0
    print(f"\n{n_fawp}/{len(df)} configurations detect FAWP")
    print(f"Saved → {out}")

def _check_version_update():
    """Check if a newer version is available on PyPI and print a notice."""
    try:
        import urllib.request, json as _j
        import fawp_index as fi
        with urllib.request.urlopen(
            "https://pypi.org/pypi/fawp-index/json", timeout=2
        ) as r:
            latest = _j.loads(r.read())["info"]["version"]
        cur = fi.__version__
        # Simple string compare — works for CalVer/SemVer alike
        cur_t = tuple(int(x) for x in cur.split(".")[:3])
        lat_t = tuple(int(x) for x in latest.split(".")[:3])
        if lat_t > cur_t:
            print(f"\u2b06  Update available: {cur} \u2192 {latest}")
            print(f"   pip install --upgrade fawp-index\n")
    except Exception:
        pass



def cmd_diagnose(args):
    """Load a FAWP result JSON and print a plain-English diagnosis."""
    import json
    from fawp_index.explain import explain_fawp
    from fawp_index.detection.odw import ODWResult

    with open(args.data) as f:
        d = json.load(f)

    # Reconstruct ODWResult-like namespace for explain
    class _R:
        pass

    r = _R()
    r.fawp_found    = d.get("fawp_found", False)
    r.peak_gap_bits = d.get("peak_gap_bits", 0.0)
    r.tau_h_plus    = d.get("tau_h_plus")
    r.tau_f         = d.get("tau_f")
    r.odw_start     = d.get("odw_start")
    r.odw_end       = d.get("odw_end")
    r.epsilon       = d.get("epsilon", 0.01)
    r.n_obs         = d.get("n_obs", 0)
    r.domain        = d.get("domain", "unknown")

    import numpy as np
    r.pred_mi  = np.array(d.get("pred_mi",  []))
    r.steer_mi = np.array(d.get("steer_mi", []))
    r.tau      = np.array(d.get("tau",      []))

    # Build ODW-like subobject
    r.odw_result = _R()
    r.odw_result.fawp_found    = r.fawp_found
    r.odw_result.peak_gap_bits = r.peak_gap_bits
    r.odw_result.tau_h_plus    = r.tau_h_plus
    r.odw_result.tau_f         = r.tau_f
    r.odw_result.odw_start     = r.odw_start
    r.odw_result.odw_end       = r.odw_end

    print(f"FAWP Diagnosis — {args.data}")
    print(f"Domain        : {r.domain}")
    print(f"Observations  : {r.n_obs}")
    print()

    status = "🔴 FAWP DETECTED" if r.fawp_found else "✅ No FAWP"
    print(f"Status        : {status}")
    print(f"Peak gap      : {r.peak_gap_bits:.4f} bits")
    print(f"Agency horizon: τ⁺ₕ = {r.tau_h_plus if r.tau_h_plus is not None else 'not reached'}")
    print(f"Failure cliff : τf  = {r.tau_f if r.tau_f is not None else 'not reached'}")
    print(f"Detection window: τ = {r.odw_start or '?'}–{r.odw_end or '?'}")
    print()

    if r.fawp_found:
        lead = (r.odw_end or 0) - (r.odw_start or 0)
        print("Plain-English diagnosis:")
        print(f"  Your system entered FAWP at τ = {r.odw_start}.")
        if r.peak_gap_bits:
            print(f"  Prediction peaked at {r.peak_gap_bits:.4f} bits while steering collapsed.")
        if r.tau_h_plus is not None and r.tau_f is not None:
            print(f"  Steering authority was lost at τ = {r.tau_h_plus}.")
            print(f"  Full control failure occurred at τ = {r.tau_f}.")
            print(f"  The operational detection window spans {lead} delay steps (τ = {r.odw_start}–{r.odw_end}).")
            print(f"  You had {lead} delay steps of warning before the cliff.")
    else:
        print("  No FAWP regime detected. Prediction and steering remain coupled.")
        print("  Forecast skill has not diverged from control authority.")

    # E9.7 calibration note
    try:
        import fawp_index as fi
        print()
        print(f"E9.7 calibration: gap2 peak leads cliff by +{fi.E97_MEAN_LEAD_GAP2_TO_CLIFF_U:.4f} delays")
        print(f"  ODW localisation error: ~{fi.E97_MEAN_ABS_ERR_GAP2_VS_ODW_START:.1f} steps")
    except Exception:
        pass


def cmd_latent(args):
    """
    Latent Environmental Residual Inference (LERI) CLI.

    Simulates the E-LERI record chain X → R → Y_τ → D_τ and computes
    the operational access horizon from your LERI paper.

    Reference: Clayton (2026) "Latent Environmental Residual Inference
    in the Volumetric Time Model", doi:10.5281/zenodo.18663547
    """
    import numpy as np

    # ── Parameters ────────────────────────────────────────────────────────
    P      = args.P
    sigma0 = args.sigma0
    alpha  = args.alpha
    eps    = args.epsilon
    beta   = args.beta
    m      = args.persist
    tau_max = args.tau_max
    n_null  = args.n_null
    seed    = args.seed

    # ── Analytic horizon (Eq. 15–16 from LERI paper) ──────────────────────
    denom = 2 ** (2 * eps) - 1
    if denom <= 0 or alpha <= 0:
        tau_analytic = float("inf")
    else:
        tau_analytic = max(0.0, (P / denom - sigma0) / alpha)

    print(f"LERI — Latent Environmental Residual Inference")
    print(f"  Parameters: P={P}  σ²₀={sigma0}  α={alpha}  ε={eps}b")
    print(f"  Analytic access horizon τ⁺ₕ ≈ {tau_analytic:.3f}")
    print()

    # ── Simulate record chain X → R → Y_τ → D_τ ──────────────────────────
    rng = np.random.default_rng(seed)
    n   = max(500, tau_max * 4)
    tau_arr = np.arange(1, tau_max + 1)

    X = rng.normal(0, 1, n)
    R = X + rng.normal(0, np.sqrt(sigma0), n)   # record with baseline noise

    from fawp_index.core.estimators import mi_from_arrays, conservative_null_floor

    mi_vals   = np.zeros(len(tau_arr))
    null_vals = np.zeros(len(tau_arr))
    scores    = np.zeros(len(tau_arr))

    print(f"  Simulating record chain (n={n}, τ_max={tau_max}, n_null={n_null})…")
    for ti, tau in enumerate(tau_arr):
        noise_tau = rng.normal(0, np.sqrt(sigma0 + alpha * tau), n)
        D_tau = R + noise_tau    # delayed readout with latency-dependent noise

        x_v = X[:-tau]; d_v = D_tau[tau:]
        raw  = mi_from_arrays(x_v, d_v)
        floor = conservative_null_floor(x_v, d_v, n_null, quantile=beta)
        mi_vals[ti]   = max(0.0, raw)
        null_vals[ti] = floor
        scores[ti]    = max(0.0, raw - floor)

    # ── Persistence-gated horizon ─────────────────────────────────────────
    tau_numeric = None
    run_below = 0
    for ti, tau in enumerate(tau_arr):
        if scores[ti] < 1e-4:
            run_below += 1
            if run_below >= m:
                tau_numeric = int(tau_arr[max(0, ti - m + 1)])
                break
        else:
            run_below = 0

    print(f"  Numeric operational horizon τₕ = {tau_numeric or 'not reached in τ_max'}")
    print()
    print(f"  {'tau':>5}  {'MI (bits)':>12}  {'null floor':>12}  {'score':>10}")
    print(f"  {'---':>5}  {'----------':>12}  {'----------':>12}  {'-----':>10}")
    step = max(1, len(tau_arr) // 20)
    for ti, tau in enumerate(tau_arr[::step]):
        idx = ti * step
        print(f"  {tau:>5}  {mi_vals[idx]:>12.6f}  {null_vals[idx]:>12.6f}  {scores[idx]:>10.6f}")
    print()

    # ── Save if requested ─────────────────────────────────────────────────
    if args.out:
        import pandas as pd
        df = pd.DataFrame({"tau": tau_arr, "mi": mi_vals,
                           "null_floor": null_vals, "score": scores})
        df.to_csv(args.out, index=False)
        print(f"Saved → {args.out}")

    print(f"LERI horizon: analytic = {tau_analytic:.2f}  numeric = {tau_numeric or '∞'}")
    print(f"LERI photon constants: {14005.0:.1f} m horizon · {4.6683e-5:.4e} s")


def cmd_leaderboard(args):
    """Push a local scan result to the global FAWP leaderboard on Supabase."""
    import json, os
    from pathlib import Path

    if args.subcommand == "push":
        with open(args.data) as f:
            d = json.load(f)

        url   = args.url   or os.environ.get("FAWP_SUPABASE_URL",   "")
        token = args.token or os.environ.get("FAWP_SUPABASE_TOKEN", "")

        if not url or not token:
            print("Supabase URL and token required.")
            print("  Set --url / --token, or env vars FAWP_SUPABASE_URL / FAWP_SUPABASE_TOKEN")
            return

        try:
            from supabase import create_client
        except ImportError:
            print("supabase package required: pip install supabase")
            return

        db = create_client(url, token)
        row = {
            "ticker":         d.get("ticker") or d.get("asset") or Path(args.data).stem,
            "timeframe":      d.get("timeframe", "custom"),
            "peak_gap_bits":  round(float(d.get("peak_gap_bits", 0) or 0), 6),
            "fawp_found":     bool(d.get("fawp_found", False)),
            "tau_h_plus":     d.get("tau_h_plus"),
            "tau_f":          d.get("tau_f"),
            "odw_start":      d.get("odw_start"),
            "odw_end":        d.get("odw_end"),
            "epsilon":        float(d.get("epsilon", 0.01)),
            "n_obs":          int(d.get("n_obs", 0)),
            "domain":         d.get("domain", "custom"),
            "source":         "cli",
        }
        result = db.table("fawp_global_lb").upsert(row).execute()
        print(f"✅ Pushed to leaderboard: {row['ticker']} — gap {row['peak_gap_bits']:.4f} bits")
        print(f"   fawp_found={row['fawp_found']}  τ⁺ₕ={row['tau_h_plus']}  domain={row['domain']}")

    elif args.subcommand == "list":
        url   = args.url   or os.environ.get("FAWP_SUPABASE_URL",   "")
        token = args.token or os.environ.get("FAWP_SUPABASE_TOKEN", "")
        if not url or not token:
            print("Supabase credentials required.")
            return
        try:
            from supabase import create_client
        except ImportError:
            print("pip install supabase"); return

        db  = create_client(url, token)
        res = (db.table("fawp_global_lb")
               .select("*")
               .order("peak_gap_bits", desc=True)
               .limit(args.top)
               .execute())
        rows = res.data or []
        print(f"Global FAWP Leaderboard (top {args.top})")
        print(f"{'Rank':>4}  {'Ticker':<20} {'Gap (bits)':>12} {'FAWP':>6} {'Domain':<16}")
        print("-" * 65)
        for i, r in enumerate(rows, 1):
            print(f"  {i:>2}.  {r.get('ticker','?'):<20} "
                  f"{float(r.get('peak_gap_bits',0)):>12.4f} "
                  f"{'🔴' if r.get('fawp_found') else '—':>6}  "
                  f"{r.get('domain','?'):<16}")


def cmd_agency(args):
    """Sweep agency horizon over (P, alpha, sigma0) and print/plot a surface.

    From the VTM paper (Eq. 15-16):
      τₕ = max(0, (P / (2^(2ε) - 1) - σ²₀) / α)
    """
    import numpy as np
    import pandas as pd

    eps   = args.epsilon
    denom = 2 ** (2 * eps) - 1

    if args.mode == "alpha":
        # Sweep alpha (noise growth rate)
        alpha_vals = np.logspace(
            np.log10(args.a_min), np.log10(args.a_max), args.n_steps)
        tau_h_vals = np.maximum(0, (args.P / denom - args.sigma0) / alpha_vals)
        print(f"Agency horizon sweep: P={args.P}  σ²₀={args.sigma0}  ε={eps}b")
        print(f"{'alpha':>12}  {'tau_h':>10}")
        print("-" * 26)
        for a, t in zip(alpha_vals, tau_h_vals):
            print(f"{a:>12.5f}  {t:>10.2f}")
        if not args.no_plot:
            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(7, 3))
                ax.loglog(alpha_vals, tau_h_vals, color="#D4AF37", lw=2)
                ax.set_xlabel("α (noise growth rate)")
                ax.set_ylabel("τₕ (agency horizon)")
                ax.set_title(f"Agency Horizon — Inverse Scaling Law  (P={args.P}, ε={eps}b)")
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"Plot skipped: {e}")

    elif args.mode == "surface":
        # 2D heatmap: P × alpha → tau_h
        import numpy as np
        P_vals     = np.logspace(np.log10(args.P_min),   np.log10(args.P_max),   args.n_steps)
        alpha_vals = np.logspace(np.log10(args.a_min),   np.log10(args.a_max),   args.n_steps)
        PP, AA     = np.meshgrid(P_vals, alpha_vals)
        TH         = np.maximum(0, (PP / denom - args.sigma0) / AA)
        print(f"VTM Agency Horizon Surface  (ε={eps}b  σ²₀={args.sigma0})")
        print(f"  P range: [{P_vals[0]:.3f}, {P_vals[-1]:.3f}]")
        print(f"  α range: [{alpha_vals[0]:.4f}, {alpha_vals[-1]:.4f}]")
        print(f"  τₕ min={TH.min():.1f}  max={TH.max():.1f}  mean={TH.mean():.1f}")
        if not args.no_plot:
            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(7, 5))
                im = ax.contourf(np.log10(PP), np.log10(AA), TH, levels=20, cmap="RdYlGn")
                plt.colorbar(im, ax=ax, label="Agency horizon τₕ (steps)")
                ax.set_xlabel("log₁₀ P (signal power)")
                ax.set_ylabel("log₁₀ α (noise growth rate)")
                ax.set_title(f"VTM Agency Horizon Surface  (ε={eps}b)")
                plt.tight_layout(); plt.show()
            except Exception as e:
                print(f"Plot skipped: {e}")
        if args.out:
            import pandas as pd
            rows = [{"P": P_vals[j], "alpha": alpha_vals[i], "tau_h": float(TH[i,j])}
                    for i in range(len(alpha_vals)) for j in range(len(P_vals))]
            pd.DataFrame(rows).to_csv(args.out, index=False)
            print(f"Saved → {args.out}")

    elif args.mode == "P":
        # Sweep signal power P
        P_vals = np.logspace(np.log10(args.P_min), np.log10(args.P_max), args.n_steps)
        tau_h_vals = np.maximum(0, (P_vals / denom - args.sigma0) / args.alpha)
        print(f"Agency horizon sweep: α={args.alpha}  σ²₀={args.sigma0}  ε={eps}b")
        print(f"{'P':>12}  {'tau_h':>10}")
        print("-" * 26)
        for P, t in zip(P_vals, tau_h_vals):
            print(f"{P:>12.4f}  {t:>10.2f}")

    if args.out:
        if args.mode == "alpha":
            df = pd.DataFrame({"alpha": alpha_vals, "tau_h": tau_h_vals})
        else:
            df = pd.DataFrame({"P": P_vals, "tau_h": tau_h_vals})
        df.to_csv(args.out, index=False)
        print(f"Saved → {args.out}")


def cmd_calibrate(args):
    """Re-derive SPHERE-16 calibration constants from fresh simulation.

    Runs the E8 flagship simulation and checks whether PEAK_PRED_BITS
    still agrees with the stored constant (2.233669 bits).
    """
    import numpy as np
    from fawp_index.constants import PEAK_PRED_BITS, TAU_PLUS_H_FLAGSHIP, TAU_F_FLAGSHIP
    from fawp_index.core.estimators import mi_from_arrays, conservative_null_floor

    a     = args.a
    K     = args.K
    delta = args.delta
    n_trials = args.n_trials
    seed  = args.seed
    tau_max = args.tau_max

    print(f"FAWP Calibration Check — SPHERE-16 baseline")
    print(f"  a={a}  K={K}  Δ={delta}  n_trials={n_trials}  seed={seed}")
    print(f"  Stored PEAK_PRED_BITS = {PEAK_PRED_BITS}")
    print()

    rng = np.random.default_rng(seed)
    n_steps = 1000
    tau_arr = np.arange(1, tau_max + 1)
    pred_mi = np.zeros(len(tau_arr))

    print(f"  Simulating {n_trials} trials × {n_steps} steps…")
    all_x = []
    all_u = []
    for trial in range(n_trials):
        x = np.zeros(n_steps); u = np.zeros(n_steps)
        for t in range(1, n_steps):
            obs = x[max(0, t - delta)]
            u[t] = -K * obs
            x[t] = a * x[t-1] + u[t] + rng.normal(0, 0.1)
            if abs(x[t]) > 500: x[t] = np.sign(x[t]) * 500
        all_x.append(x); all_u.append(u)

    X = np.concatenate(all_x)
    U = np.concatenate(all_u)

    for ti, tau in enumerate(tau_arr):
        xp = X[:-tau]; yp = X[tau:]
        raw   = mi_from_arrays(xp, yp)
        floor = conservative_null_floor(xp, yp, args.n_null, 0.99)
        pred_mi[ti] = max(0.0, raw - floor)

    peak_idx  = int(np.argmax(pred_mi))
    peak_tau  = tau_arr[peak_idx]
    peak_bits = float(pred_mi[peak_idx])
    delta_pct = abs(peak_bits - PEAK_PRED_BITS) / PEAK_PRED_BITS * 100

    print(f"  Simulated peak pred MI = {peak_bits:.6f} bits at τ = {peak_tau}")
    print(f"  Stored   PEAK_PRED_BITS = {PEAK_PRED_BITS} bits")
    print(f"  Δ = {delta_pct:.2f}%")
    print()

    if delta_pct < 5.0:
        print(f"✅ CALIBRATION PASS — within 5% of stored constant")
    else:
        print(f"⚠️  CALIBRATION DRIFT — {delta_pct:.1f}% deviation from stored constant")
        print(f"   Consider updating PEAK_PRED_BITS = {peak_bits:.6f}")

    if args.out:
        import pandas as pd
        pd.DataFrame({"tau": tau_arr, "pred_mi": pred_mi}).to_csv(args.out, index=False)
        print(f"Saved → {args.out}")


def cmd_triple_horizon(args):
    """Run the E11-1 Triple Horizon benchmark: readout, steering, and functional horizons.

    Simulates the unstable latent process + degradable readout chain + viability cliff
    from the SPHERE_23 paper and reports all five horizon boundaries.
    """
    import numpy as np
    from fawp_index.core.estimators import mi_from_arrays, conservative_null_floor
    from fawp_index.constants import ALPHA_A, ALPHA_A_SQ

    a = args.a; K = args.K; delta = args.delta
    rts  = args.readout_tau_scale   # readout decay scale
    shield = args.shield             # shielding factor [0,1]
    n = max(600, args.tau_max * 5)
    tau_arr = np.arange(1, args.tau_max + 1)
    rng = np.random.default_rng(args.seed)
    eps = args.epsilon

    print(f"Triple Horizon Benchmark — SPHERE_23 E11-1")
    print(f"  a={a}  K={K}  Δ={delta}  readout_scale={rts}  shield={shield}")
    print(f"  ε={eps}  τ_max={args.tau_max}  n={n}  seed={args.seed}")
    print()

    # Simulate latent process x with delayed controller
    x = np.zeros(n); u = np.zeros(n); r = np.zeros(n)
    for t in range(1, n):
        obs = x[max(0, t - delta)]
        u[t] = np.clip(-K * obs, -10, 10)
        x[t] = a * x[t-1] + u[t] + rng.normal(0, 0.1)
        if abs(x[t]) > 500: x[t] = np.sign(x[t]) * 500
        # Readout chain: r_t = x_t + noise, with shielding
        r[t] = (1.0 - shield) * x[t] + rng.normal(0, 0.5)

    # Compute steer MI, readout MI, pred MI per tau
    steer_mi   = np.zeros(len(tau_arr))
    readout_mi = np.zeros(len(tau_arr))
    pred_mi    = np.zeros(len(tau_arr))

    print(f"  Computing MI curves ({len(tau_arr)} delays)…")
    for ti, tau in enumerate(tau_arr):
        # Steering: u[t] → x[t+tau]
        xs = u[:-tau]; ys = x[tau:]
        steer_mi[ti]   = max(0.0, mi_from_arrays(xs, ys) -
                              conservative_null_floor(xs, ys, args.n_null, 0.99))
        # Readout: r[t] → x[t+tau] (degraded by tau-dependent noise)
        noisy_r = r[:n-tau] + rng.normal(0, 0.1 * tau / rts, n - tau)
        readout_mi[ti] = max(0.0, mi_from_arrays(noisy_r, x[tau:]) -
                              conservative_null_floor(noisy_r, x[tau:], args.n_null, 0.99))
        # Prediction: x[t] → x[t+tau]
        xp = x[:-tau]; yp = x[tau:]
        pred_mi[ti]    = max(0.0, mi_from_arrays(xp, yp) -
                              conservative_null_floor(xp, yp, args.n_null, 0.99))

    # Extract horizon boundaries
    def first_below(arr, thresh):
        for i, v in enumerate(arr):
            if v <= thresh: return int(tau_arr[i])
        return None

    tau_alpha    = first_below(steer_mi, ALPHA_A)
    tau_h_plus   = first_below(steer_mi, eps)
    tau_alpha2   = first_below(readout_mi, ALPHA_A_SQ)
    tau_readout  = first_below(readout_mi, eps)

    # Functional horizon: viability cliff (crash probability)
    x_fail = args.x_fail
    fail_rates = np.array([np.mean(np.abs(x[t:]) > x_fail) for t in tau_arr])
    tau_f_idx  = np.argmax(fail_rates >= 0.99)
    tau_f      = int(tau_arr[tau_f_idx]) if fail_rates[tau_f_idx] >= 0.99 else None

    print(f"  τα (steering wall)    : {tau_alpha  or 'not reached'}")
    print(f"  τ⁺ₕ (steering horizon): {tau_h_plus if tau_h_plus is not None else 'not reached'}")
    print(f"  τf  (functional)      : {tau_f if tau_f is not None else 'not reached'}")
    print(f"  τα² (residual floor)  : {tau_alpha2 or 'not reached'}")
    print(f"  τread (readout)       : {tau_readout or 'not reached'}")
    print()
    print(f"  SPHERE_23 E11-1 reference: τα=10  τ⁺ₕ=12  τf=29  τα²=32  τread=38")
    print()

    # Ordering
    vals = [(v, n) for v, n in [(tau_alpha,"τα"),(tau_h_plus,"τ⁺ₕ"),
                                  (tau_f,"τf"),(tau_alpha2,"τα²"),(tau_readout,"τread")]
            if v is not None]
    vals.sort()
    ordering = " < ".join(n for _, n in vals)
    print(f"  Detected ordering: {ordering}")
    e11_order = "τα < τ⁺ₕ < τf < τα² < τread"
    dominant = ordering == e11_order
    print(f"  Dominant E11 ordering: {'✅ YES' if dominant else '❌ NO (variant detected)'}")

    if args.out:
        import pandas as pd
        pd.DataFrame({"tau": tau_arr, "steer_mi": steer_mi,
                      "readout_mi": readout_mi, "pred_mi": pred_mi,
                      "fail_rate": fail_rates}).to_csv(args.out, index=False)
        print(f"Saved → {args.out}")


def cmd_changelog(args):
    """Print the CHANGELOG entry for the current or specified version."""
    import re
    from pathlib import Path

    # Find CHANGELOG.md: repo root or installed data path
    candidates = [
        Path(__file__).parent.parent / "CHANGELOG.md",
        Path(__file__).parent.parent.parent / "CHANGELOG.md",
    ]
    cl_path = next((p for p in candidates if p.exists()), None)
    if cl_path is None:
        print("CHANGELOG.md not found. Clone the repo: git clone https://github.com/DrRalphClayton/fawp-index")
        return

    import fawp_index as fi
    version = args.version or fi.__version__
    src_cl = cl_path.read_text()

    # Find the entry for this version
    pattern = rf"(## v?{re.escape(version)}.*?)(?=\n## |\Z)"
    m = re.search(pattern, src_cl, re.DOTALL | re.IGNORECASE)
    if m:
        print(m.group(1).strip())
    else:
        print(f"No CHANGELOG entry found for v{version}")
        print(f"(CHANGELOG at: {cl_path})")

        # Show the most recent entry instead
        first = re.search(r"(## v?\d+\.\d+.*?)(?=\n## |\Z)", src_cl, re.DOTALL)
        if first:
            print("\nMost recent entry:")
            print(first.group(1).strip()[:800])


def cmd_scan(args):
    """One-liner FAWP scan: fetch data and print a plain-text result."""
    print(f"FAWP scan: {args.ticker} [{args.period}]")
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance required: pip install yfinance"); return

    df = yf.download(args.ticker, period=args.period,
                     auto_adjust=True, progress=False)
    if df.empty:
        print(f"No data returned for {args.ticker}"); return

    from fawp_index.market import scan_fawp_market
    r = scan_fawp_market(df, ticker=args.ticker, epsilon=args.epsilon,
                         n_null=args.n_null, verbose=False)

    odw = getattr(r, "odw_result", r)
    fawp   = getattr(odw, "fawp_found",    False)
    gap    = float(getattr(odw, "peak_gap_bits", 0) or 0)
    tauh   = getattr(odw, "tau_h_plus", None)
    tauf   = getattr(odw, "tau_f",      None)
    start  = getattr(odw, "odw_start",  None)
    end    = getattr(odw, "odw_end",    None)
    n_bars = len(df)

    status = "🔴 FAWP DETECTED" if fawp else "✅ No FAWP"
    print(f"Status        : {status}")
    print(f"Ticker        : {args.ticker}  ({n_bars} bars, {args.period})")
    print(f"Peak gap      : {gap:.4f} bits")
    print(f"Agency horizon: τ⁺ₕ = {tauh if tauh is not None else 'not reached'}")
    print(f"Failure cliff : τf  = {tauf if tauf is not None else 'not reached'}")
    if start is not None and end is not None:
        print(f"ODW           : τ = {start}–{end}  (width {end - start} steps)")

    import fawp_index as fi
    sm = getattr(odw, "steer_mi_at_h", None)
    if sm is not None:
        if sm <= fi.ALPHA_A_SQ:
            tier = "null (below α²A)"
        elif sm <= fi.ALPHA_A:
            tier = "residual (αA ↔ α²A)"
        else:
            tier = "active steering"
        print(f"Steering tier : {tier}  (steer MI = {sm:.5f})")

    if args.out:
        import json
        with open(args.out, "w") as f:
            json.dump({"ticker": args.ticker, "period": args.period,
                       "fawp_found": bool(fawp), "peak_gap_bits": gap,
                       "tau_h_plus": tauh, "tau_f": tauf,
                       "odw_start": start, "odw_end": end,
                       "n_obs": n_bars}, f, indent=2)
        print(f"Saved → {args.out}")


def cmd_triple_horizon_sweep(args):
    """Sweep (a, K) grid — reproduces E11-2 portability sweep (SPHERE_23)."""
    import numpy as np
    from fawp_index.constants import ALPHA_A, ALPHA_A_SQ
    from fawp_index.core.estimators import mi_from_arrays, conservative_null_floor

    a_vals = [float(x) for x in args.a_grid.split(",")]
    K_vals = [float(x) for x in args.K_grid.split(",")]
    n = max(500, args.tau_max * 5)
    tau_arr = np.arange(1, args.tau_max + 1)
    eps = args.epsilon; rts = args.readout_tau_scale

    def _run(a, K):
        rng = np.random.default_rng(args.seed)
        x = np.zeros(n); u = np.zeros(n); r = np.zeros(n)
        for t in range(1, n):
            obs = x[max(0, t - args.delta)]
            u[t] = np.clip(-K * obs, -10, 10)
            x[t] = a * x[t-1] + u[t] + rng.normal(0, 0.1)
            if abs(x[t]) > 500: x[t] = np.sign(x[t]) * 500
            r[t] = x[t] + rng.normal(0, 0.5)
        sm = np.zeros(len(tau_arr)); rm = np.zeros(len(tau_arr))
        for ti, tau in enumerate(tau_arr):
            xs = u[:-tau]; ys = x[tau:]
            sm[ti] = max(0.0, mi_from_arrays(xs, ys) - conservative_null_floor(xs, ys, 20, 0.99))
            nr = r[:n-tau] + rng.normal(0, 0.1*tau/rts, n-tau)
            rm[ti] = max(0.0, mi_from_arrays(nr, x[tau:]) - conservative_null_floor(nr, x[tau:], 20, 0.99))
        def _fb(arr, th):
            for i, v in enumerate(arr):
                if v <= th: return int(tau_arr[i])
            return None
        fr = np.array([np.mean(np.abs(x[t:]) > 500) for t in tau_arr])
        fi = np.argmax(fr >= 0.99)
        return {"ta": _fb(sm, ALPHA_A), "th": _fb(sm, eps),
                "tf": int(tau_arr[fi]) if fr[fi] >= 0.99 else None,
                "ta2": _fb(rm, ALPHA_A_SQ), "tr": _fb(rm, eps)}

    n_configs = len(a_vals) * len(K_vals)
    print("Triple Horizon sweep — " + str(n_configs) + " configs")
    hdr = "{:>6} {:>5} {:>5} {:>5} {:>5} {:>5} {:>7}  {}".format(
          "a","K","tau_a","tau_h","tau_f","tau_a2","tau_rd","order")
    print(hdr)
    print("-" * 60)
    rows = []
    for a in a_vals:
        for K in K_vals:
            res = _run(a, K)
            ta, th, tf, ta2, tr = res["ta"], res["th"], res["tf"], res["ta2"], res["tr"]
            pairs = [(v, n) for v, n in [(ta,"ta"),(th,"th"),(tf,"tf"),(ta2,"ta2"),(tr,"tr")] if v is not None]
            pairs.sort()
            order = "<".join(nm for _, nm in pairs)
            dom = order == "ta<th<tf<ta2<tr"
            flag = "OK" if dom else "!!"
            row = "{:>6.3f} {:>5.2f} {:>5} {:>5} {:>5} {:>5} {:>7}  {} {}".format(
                  a, K, str(ta), str(th), str(tf), str(ta2), str(tr), flag, order)
            print(row)
            rows.append({"a":a,"K":K,"tau_alpha":ta,"tau_h_plus":th,
                         "tau_f":tf,"tau_alpha2":ta2,"tau_readout":tr,
                         "dominant":dom})
    dom_rate = sum(1 for r in rows if r["dominant"]) / len(rows)
    print("")
    print("Dominant ordering rate: " + str(round(dom_rate*100,1)) + "% (E11-2 ref: 94.4%)")
    if args.out:
        import pandas as pd
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print("Saved to " + args.out)


def cmd_steering(args):
    """Compute steering decay profile for a ticker vs SPHERE_23 αA / α²A thresholds."""
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance required: pip install yfinance"); return

    import numpy as np
    from fawp_index.constants import ALPHA_A, ALPHA_A_SQ
    from fawp_index.core.estimators import mi_from_arrays, conservative_null_floor
    from fawp_index.market import scan_fawp_market

    print(f"Steering profile: {args.ticker} [{args.period}]")
    df = yf.download(args.ticker, period=args.period, auto_adjust=True, progress=False)
    if df.empty:
        print(f"No data for {args.ticker}"); return

    r = scan_fawp_market(df, ticker=args.ticker, epsilon=args.epsilon,
                         n_null=args.n_null, verbose=False)
    odw = getattr(r, "odw_result", None)
    steer = getattr(odw, "steer_mi", np.array([])) if odw else np.array([])
    tau   = getattr(odw, "tau",      np.array([])) if odw else np.array([])

    if len(steer) == 0:
        print("No steering MI curve available — run a fresh scan first"); return

    decay = float(np.polyfit(tau, steer, 1)[0]) if len(steer) >= 3 else 0.0

    print(f"Decay rate     : {decay:+.6f} bits/τ")
    print(f"Peak steer MI  : {steer.max():.4f} bits  at τ={tau[steer.argmax()]}")
    print(f"Final steer MI : {steer[-1]:.4f} bits  at τ={tau[-1]}")
    print()

    # Tier classification at each delay
    wall_cross   = next((int(tau[i]) for i,v in enumerate(steer) if v <= ALPHA_A),   None)
    floor_cross  = next((int(tau[i]) for i,v in enumerate(steer) if v <= ALPHA_A_SQ), None)
    eps_cross    = next((int(tau[i]) for i,v in enumerate(steer) if v <= args.epsilon), None)

    print(f"αA  steering wall   (={ALPHA_A:.6f}) crossed at: τ = {wall_cross  or 'not reached'}")
    print(f"α²A residual floor  (={ALPHA_A_SQ:.2e}) crossed at: τ = {floor_cross or 'not reached'}")
    print(f"ε   op. horizon     (={args.epsilon:.4f})       crossed at: τ = {eps_cross   or 'not reached'}")
    print()

    # Per-tau table (sampled)
    step = max(1, len(tau) // 20)
    print(f"  {'τ':>5}  {'steer MI':>10}  {'tier'}")
    print(f"  {'—'*35}")
    for i in range(0, len(tau), step):
        v = steer[i]
        tier = ("null      " if v <= ALPHA_A_SQ else
                "residual  " if v <= ALPHA_A    else "active    ")
        print(f"  {tau[i]:>5.0f}  {v:>10.5f}  {tier}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(tau, steer, color="#4A7FCC", lw=1.5, label="Steering MI")
            ax.axhline(ALPHA_A,    color="#FF6B2B", ls="--", lw=1, label=f"αA={ALPHA_A:.5f}")
            ax.axhline(ALPHA_A_SQ, color="#D4AF37", ls=":",  lw=1, label=f"α²A={ALPHA_A_SQ:.2e}")
            ax.axhline(args.epsilon, color="#C0111A", ls="-.", lw=1, label=f"ε={args.epsilon}")
            if wall_cross:  ax.axvline(wall_cross,  color="#FF6B2B", ls="--", alpha=.5)
            if floor_cross: ax.axvline(floor_cross, color="#D4AF37", ls=":", alpha=.5)
            ax.set_xlabel("τ (delay steps)"); ax.set_ylabel("Steering MI (bits)")
            ax.set_title(f"{args.ticker} — Steering decay profile (SPHERE_23 thresholds)")
            ax.legend(fontsize=8); plt.tight_layout(); plt.show()
        except Exception as e:
            print(f"Plot skipped: {e}")

def main():
    _check_version_update()
    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
