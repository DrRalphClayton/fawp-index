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

    # verify
    p = sub.add_parser("verify", help="Run calibration self-checks against SPHERE/E9 constants")
    p.set_defaults(func=cmd_verify)

    # grid
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
    print(f"  {'gap2 peak (raw lever. gap)':<28} "
          f"{'+{:.4f}'.format(fi.E97_MEAN_LEAD_GAP2_TO_CLIFF_U):<16} "
          f"{'+{:.4f}'.format(fi.E97_MEAN_LEAD_GAP2_TO_CLIFF_XI):<16} "
          f"~{fi.E97_MEAN_ABS_ERR_GAP2_VS_ODW_START:.1f} delays  ✅ BEST")
    print(f"  {'α₂  (SPHERE-16)':<28} "
          f"{'+{:.4f}'.format(fi.E97_MEAN_LEAD_ALPHA2_TO_CLIFF_U):<16} "
          f"{'+{:.4f}'.format(fi.E97_MEAN_LEAD_ALPHA2_TO_CLIFF_XI):<16} "
          f"~{fi.E97_MEAN_ABS_ERR_ALPHA2_VS_ODW_START:.1f} delays  ✅ Good")
    print(f"  {'α   (baseline, old)':<28} "
          f"{'{:.4f}'.format(fi.E97_MEAN_LEAD_ALPHA_TO_CLIFF_U):<16} "
          f"{'{:.4f}'.format(fi.E97_MEAN_LEAD_ALPHA_TO_CLIFF_XI):<16} "
          f"~{fi.E97_MEAN_ABS_ERR_ALPHA_VS_ODW_START:.1f} delays  ❌ Lags")
    print()
    print(f"  Total runs : {fi.E97_N_RUNS}")
    print(f"  Mean τf    : {fi.E97_MEAN_TAU_F}")
    print(f"  Source     : doi:10.5281/zenodo.19065421")
    print()


def cmd_verify(_args):
    """Run calibration self-checks against published SPHERE/E9 constants."""
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

def main():
    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
