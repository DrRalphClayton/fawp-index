"""
fawp-watchlist — Saved named watchlist manager
================================================

Create, scan, list, and delete named watchlists that persist between runs.

Watchlists are stored in ~/.fawp/watchlists.json (or $FAWP_STORE).

Commands
--------
  create <name> TICKER [TICKER ...]   Create a new named watchlist
  scan   <name>                       Scan a saved watchlist
  list                                List all saved watchlists
  show   <name>                       Show tickers and settings for a watchlist
  delete <name>                       Delete a saved watchlist

Examples
--------
  fawp-watchlist create tech AAPL MSFT NVDA AMD
  fawp-watchlist create crypto BTC-USD ETH-USD SOL-USD --period 1y
  fawp-watchlist scan tech
  fawp-watchlist scan tech --rank-by gap --top 5 --out tech.html
  fawp-watchlist list
  fawp-watchlist show tech
  fawp-watchlist delete tech
"""

import argparse
import sys

from fawp_index import __version__ as _VERSION
from fawp_index.watchlist_store import WatchlistStore


def _header():
    print(f"\nfawp-watchlist v{_VERSION} | doi:10.5281/zenodo.18673949")
    print("=" * 60)


def cmd_create(args):
    _header()
    store = WatchlistStore(args.store)
    tickers = [t.strip().upper() for t in args.tickers if t.strip()]
    if not tickers:
        print("ERROR: provide at least one ticker symbol")
        sys.exit(1)
    try:
        store.create(
            name       = args.name,
            tickers    = tickers,
            period     = args.period,
            timeframes = args.timeframes.split(",") if args.timeframes else None,
            window     = args.window,
            tau_max    = args.tau_max,
            overwrite  = args.overwrite,
        )
        print(f"Created watchlist '{args.name}': {tickers}")
        print(f"Stored at: {store._path}")
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)


def cmd_scan(args):
    _header()
    store = WatchlistStore(args.store)
    if not store.exists(args.name):
        print(f"ERROR: watchlist '{args.name}' not found. Run: fawp-watchlist list")
        sys.exit(1)

    result = store.scan(
        name        = args.name,
        timeframes  = args.timeframes.split(",") if args.timeframes else None,
        window      = args.window,
        tau_max     = args.tau_max,
        n_null      = args.n_null,
        verbose     = True,
    )
    print("\n" + result.summary(n=30))

    # Ranked table
    ranked = result.rank_by(args.rank_by)
    n_show = min(args.top, len(ranked))
    print(f"\nTop {n_show} by {args.rank_by}:")
    print(f"  {'#':>3}  {'Ticker':<10} {'TF':<5} {'Score':>6} {'Gap(b)':>7} {'Status':<10} {'ODW'}")
    print("  " + "-" * 60)
    for i, a in enumerate(ranked[:n_show], 1):
        if a.error:
            print(f"  {i:>3}  {a.ticker:<10} ERROR")
            continue
        status = "FAWP" if a.regime_active else ("watch" if a.latest_score > 0.05 else "—")
        odw = f"{a.peak_odw_start}-{a.peak_odw_end}" if a.peak_odw_start is not None else "—"
        print(f"  {i:>3}  {a.ticker:<10} {a.timeframe:<5} "
              f"{a.latest_score:>6.4f} {a.peak_gap_bits:>7.4f} {status:<10} {odw}")

    # Leaderboard
    if args.leaderboard:
        from fawp_index.leaderboard import Leaderboard
        lb = Leaderboard.from_watchlist(result)
        print("\n" + lb.summary())

    # Explain top asset
    if args.explain:
        from fawp_index.explain import explain_asset
        valid = [a for a in ranked if not a.error]
        if valid:
            print("\n" + explain_asset(valid[0]))

    # Save output
    if args.out:
        from pathlib import Path
        p = Path(args.out)
        ext = p.suffix.lower().lstrip(".")
        dispatch = {"html": result.to_html, "json": result.to_json, "csv": result.to_csv}
        fn = dispatch.get(ext)
        if fn:
            fn(p)
            print(f"\nSaved -> {p}")
        else:
            print(f"WARNING: unsupported format '{p.suffix}' — use .html .json .csv")

    n_flagged = result.n_flagged
    if n_flagged:
        print(f"\n{n_flagged} asset(s) currently in FAWP regime")
    else:
        print(f"\n{result.n_assets} asset(s) scanned — all clear")


def cmd_list(args):
    store = WatchlistStore(args.store)
    names = store.list()
    if not names:
        print("No saved watchlists. Create one with: fawp-watchlist create <name> TICKER ...")
        return
    print(f"\nfawp-watchlist v{_VERSION} — saved watchlists")
    print("=" * 60)
    print(store.summary())


def cmd_show(args):
    store = WatchlistStore(args.store)
    try:
        info = store.show(args.name)
    except KeyError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    print(f"\nWatchlist: {args.name}")
    print(f"  Tickers   : {' '.join(info['tickers'])}")
    print(f"  Period    : {info['period']}")
    print(f"  Timeframes: {','.join(info['timeframes'])}")
    if info.get("window"):
        print(f"  Window    : {info['window']}")
    if info.get("tau_max"):
        print(f"  Tau max   : {info['tau_max']}")
    print(f"  Created   : {info['created'][:10]}")
    last = info.get("last_scanned") or "never"
    if "T" in last:
        last = last[:10]
    print(f"  Last scan : {last}")


def cmd_delete(args):
    store = WatchlistStore(args.store)
    try:
        store.delete(args.name)
        print(f"Deleted watchlist '{args.name}'")
    except KeyError as e:
        print(f"ERROR: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        prog="fawp-watchlist",
        description="Manage named FAWP watchlists.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  fawp-watchlist create tech AAPL MSFT NVDA AMD\n"
            "  fawp-watchlist scan tech\n"
            "  fawp-watchlist scan tech --rank-by gap --out tech.html\n"
            "  fawp-watchlist list\n"
            "  fawp-watchlist show tech\n"
            "  fawp-watchlist delete tech\n"
        ),
    )
    parser.add_argument("--store", default=None,
                        help="Path to watchlist JSON store (default: ~/.fawp/watchlists.json)")
    parser.add_argument("--version", action="version", version=f"fawp-watchlist {_VERSION}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    # create
    p = sub.add_parser("create", help="Create a named watchlist")
    p.add_argument("name")
    p.add_argument("tickers", nargs="+")
    p.add_argument("--period",     default="2y")
    p.add_argument("--timeframes", default=None, help="Comma-separated: 1d,1wk")
    p.add_argument("--window",     type=int, default=None)
    p.add_argument("--tau-max",    type=int, default=None)
    p.add_argument("--overwrite",  action="store_true")
    p.set_defaults(func=cmd_create)

    # scan
    p = sub.add_parser("scan", help="Scan a saved watchlist")
    p.add_argument("name")
    p.add_argument("--timeframes", default=None)
    p.add_argument("--window",     type=int, default=None)
    p.add_argument("--tau-max",    type=int, default=None)
    p.add_argument("--n-null",     type=int, default=0)
    p.add_argument("--rank-by",    default="score",
                   choices=["score", "gap", "entry", "persistence", "freshness"])
    p.add_argument("--top",        type=int, default=10)
    p.add_argument("--out",        default=None)
    p.add_argument("--leaderboard", action="store_true",
                   help="Print leaderboard categories after scan")
    p.add_argument("--explain",    action="store_true",
                   help="Print explain-score card for top asset")
    p.set_defaults(func=cmd_scan)

    # list
    p = sub.add_parser("list", help="List all saved watchlists")
    p.set_defaults(func=cmd_list)

    # show
    p = sub.add_parser("show", help="Show tickers and settings for a watchlist")
    p.add_argument("name")
    p.set_defaults(func=cmd_show)

    # delete
    p = sub.add_parser("delete", help="Delete a saved watchlist")
    p.add_argument("name")
    p.set_defaults(func=cmd_delete)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
