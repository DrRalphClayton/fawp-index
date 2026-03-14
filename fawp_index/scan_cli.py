"""
fawp-scan — Quick FAWP watchlist scanner
=========================================

Scan any set of tickers or a named preset, print a ranked table,
and optionally save an HTML/JSON/CSV report.

Requires: pip install yfinance

Usage
-----
  fawp-scan BTC-USD
  fawp-scan SPY QQQ GLD TLT
  fawp-scan BTC-USD ETH-USD SOL-USD --out crypto_scan.html

  fawp-scan --preset crypto
  fawp-scan --preset equities
  fawp-scan --preset sectors  --timeframe 1wk
  fawp-scan --preset etfs     --rank-by gap --top 5
  fawp-scan --preset macro    --out macro.html

Options
-------
  --preset      crypto | equities | sectors | etfs | macro
  --period      yfinance period string (default: 2y)
  --timeframe   1d | 1wk | 1mo  (default: 1d)
  --window      rolling window in bars (default: preset default)
  --tau-max     max tau  (default: preset default)
  --rank-by     score | gap | entry | persistence | freshness
  --top         show top N  (default: 10)
  --out         save report (.html / .json / .csv)
  --no-alerts   suppress FAWP active markers
"""

import argparse
import sys
from pathlib import Path

from fawp_index import __version__ as _VERSION


def main():
    parser = argparse.ArgumentParser(
        prog="fawp-scan",
        description=(
            f"fawp-scan v{_VERSION} — Quick FAWP watchlist scanner\n"
            "Scan tickers or a preset and rank by FAWP signal."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  fawp-scan BTC-USD ETH-USD SOL-USD\n"
            "  fawp-scan SPY QQQ GLD --out scan.html\n"
            "  fawp-scan --preset crypto\n"
            "  fawp-scan --preset sectors --timeframe 1wk --rank-by gap\n"
            "  fawp-scan --preset equities --out equities.html\n"
        ),
    )

    parser.add_argument(
        "tickers", nargs="*",
        help="Ticker symbols to scan (e.g. BTC-USD SPY QQQ). "
             "Ignored if --preset is used.",
    )
    parser.add_argument(
        "--preset", choices=["crypto", "equities", "sectors", "etfs", "macro"],
        help="Use a built-in watchlist preset.",
    )
    parser.add_argument("--period",     default=None,    help="yfinance period (default: preset default or 2y)")
    parser.add_argument("--timeframe",  default=None,    help="1d | 1wk | 1mo (default: preset default or 1d)")
    parser.add_argument("--window",     type=int, default=None, help="Rolling window bars")
    parser.add_argument("--tau-max",    type=int, default=None, help="Max tau")
    parser.add_argument("--n-null",     type=int, default=0,    help="Null permutations (default: 0)")
    parser.add_argument("--rank-by",    default="score",
                        choices=["score", "gap", "entry", "persistence", "freshness"],
                        help="Ranking metric (default: score)")
    parser.add_argument("--top",        type=int, default=10, help="Show top N (default: 10)")
    parser.add_argument("--out",        default=None,    help="Save report (.html / .json / .csv)")
    parser.add_argument("--no-alerts",  action="store_true", help="Suppress FAWP active markers")
    parser.add_argument("--leaderboard", action="store_true",
                        help="Print ranked leaderboard categories after scan")
    parser.add_argument("--leaderboard-out", default=None,
                        help="Save leaderboard to file (.html / .json / .csv)")
    parser.add_argument("--explain",    action="store_true",
                        help="Print explain-score card for the top-ranked asset")
    parser.add_argument("--version",    action="version", version=f"fawp-scan {_VERSION}")

    args = parser.parse_args()

    # ── Validate ──────────────────────────────────────────────────────────────
    if not args.preset and not args.tickers:
        parser.print_help()
        print("\nERROR: provide ticker symbols or --preset <name>")
        sys.exit(1)

    print(f"\nfawp-scan v{_VERSION} | doi:10.5281/zenodo.18673949")
    print("=" * 60)

    # ── Run scan ──────────────────────────────────────────────────────────────
    kwargs = {}
    if args.window:  kwargs["window"]  = args.window
    if args.tau_max: kwargs["tau_max"] = args.tau_max
    kwargs["n_null"]      = args.n_null
    kwargs["max_workers"] = 4
    kwargs["verbose"]     = True

    tf = [args.timeframe] if args.timeframe else None

    if args.preset:
        from fawp_index.scanner.presets import _run_preset
        result = _run_preset(
            args.preset,
            period     = args.period,
            timeframes = tf,
            **kwargs,
        )
    else:
        from fawp_index.watchlist import scan_watchlist
        result = scan_watchlist(
            args.tickers,
            period     = args.period or "2y",
            timeframes = tf or ["1d"],
            **kwargs,
        )

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{result.summary(n=50)}")

    # ── Ranked top N ─────────────────────────────────────────────────────────
    ranked = result.rank_by(args.rank_by)
    n_show = min(args.top, len(ranked))
    print(f"\nTop {n_show} by {args.rank_by}:")
    print(f"  {'#':>3}  {'Ticker':<12} {'TF':<5} {'Score':>6} {'Gap(b)':>7} "
          f"{'Regime':<9} {'Days':>5} {'Age':>5} {'ODW':>8}")
    print("  " + "-" * 64)
    for i, a in enumerate(ranked[:n_show], 1):
        if a.error:
            print(f"  {i:>3}  {a.ticker:<12} {a.timeframe:<5} ERROR: {a.error[:30]}")
            continue
        fawp_tag = "🔴 FAWP" if (a.regime_active and not args.no_alerts) else (
                   "🟡 WATCH" if a.latest_score > 0.05 else "     —  ")
        odw = f"{a.peak_odw_start}–{a.peak_odw_end}" if a.peak_odw_start else "—"
        print(f"  {i:>3}  {a.ticker:<12} {a.timeframe:<5} "
              f"{a.latest_score:>6.4f} {a.peak_gap_bits:>7.4f} "
              f"{fawp_tag:<9} {a.days_in_regime:>5} {a.signal_age_days:>5} {odw:>8}")

    # ── FAWP Score for top asset ──────────────────────────────────────────────
    valid = [a for a in ranked if not a.error and a.scan is not None]
    if valid:
        top = valid[0]
        score = top.scan.latest.fawp_score
        print(f"\n── FAWP Score: {top.ticker} [{top.timeframe}] ──────────────")
        print(f"  Score      : {score['score']}/100")
        print(f"  Prediction : {score['prediction']}")
        print(f"  Control    : {score['control']}")
        print(f"  Regime     : {score['regime']}")
        print(f"  Gap        : {score['gap_bits']} bits")
        print(f"  ODW        : {score['odw']}")

    # ── Leaderboard ───────────────────────────────────────────────────────────
    if args.leaderboard:
        from fawp_index.leaderboard import Leaderboard
        lb = Leaderboard.from_watchlist(result)
        print("\n" + lb.summary())
        if args.leaderboard_out:
            p_lb = Path(args.leaderboard_out)
            ext = p_lb.suffix.lower().lstrip(".")
            if ext == "html":
                lb.to_html(p_lb)
            elif ext == "csv":
                lb.to_csv(p_lb)
            elif ext == "json":
                lb.to_json(p_lb)
            else:
                print(f"WARNING: unsupported leaderboard format '{p_lb.suffix}'")
            print(f"Leaderboard saved -> {p_lb}")

    # ── Explain top asset ─────────────────────────────────────────────────────
    if args.explain and valid:
        from fawp_index.explain import explain_asset
        print("\n" + explain_asset(valid[0]))

    # ── Save output ───────────────────────────────────────────────────────────
    if args.out:
        p = Path(args.out)
        ext = p.suffix.lower().lstrip(".")
        dispatch = {"html": result.to_html, "json": result.to_json, "csv": result.to_csv}
        fn = dispatch.get(ext)
        if fn:
            fn(p)
            print(f"\nSaved -> {p}")
        else:
            print(f"\nWARNING: unsupported format '{p.suffix}' — use .html .json .csv")

    # ── Active regime count ───────────────────────────────────────────────────
    if result.n_flagged and not args.no_alerts:
        print(f"\n{result.n_flagged} asset(s) currently in FAWP regime")
    else:
        print(f"\n{result.n_assets} asset(s) scanned — {result.n_flagged} flagged")


if __name__ == "__main__":
    main()
