"""
fawp_index.demo_launcher — One-command demo mode.

Launches the FAWP Dashboard pre-loaded with synthetic demo data
so first-time users can see it working with zero setup.

Usage::

    fawp-demo                          # launch with synthetic data
    fawp-demo --asset BTC-USD          # launch with a real ticker
    fawp-demo --port 8502              # use a different port
    fawp-demo --no-browser             # headless (don't open browser)

The demo mode sets FAWP_DEMO=1 in the environment so the dashboard
auto-selects "Demo data" on startup.

Ralph Clayton (2026) · doi:10.5281/zenodo.18673949
"""

import os
import sys
import subprocess
from pathlib import Path

from fawp_index import __version__ as _VERSION

_DASHBOARD = Path(__file__).parent.parent / "dashboard" / "app.py"


def _print_banner():
    print()
    print("=" * 58)
    print("  FAWP Scanner — Demo Mode")
    print(f"  fawp-index v{_VERSION}")
    print()
    print("  Detecting the Information-Control Exclusion Principle")
    print("  doi:10.5281/zenodo.18673949")
    print("=" * 58)
    print()


def launch_demo():
    """
    Launch the FAWP Dashboard in demo mode.

    Sets FAWP_DEMO=1 so the dashboard auto-selects the built-in
    synthetic dataset (SPY, QQQ, GLD, BTC, TLT — 600 bars each).

    Use --asset TICKER to pre-fill a real ticker instead.
    """
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print(
            "Streamlit is not installed.\n"
            "Install it with:  pip install 'fawp-index[dashboard]'\n"
            "or:               pip install streamlit"
        )
        sys.exit(1)

    if not _DASHBOARD.exists():
        print(
            f"Dashboard not found at {_DASHBOARD}\n"
            "Clone the repo and run from there:\n"
            "  git clone https://github.com/DrRalphClayton/fawp-index\n"
            "  cd fawp-index && pip install -e '.[dashboard]'\n"
            "  fawp-demo"
        )
        sys.exit(1)

    import argparse
    parser = argparse.ArgumentParser(
        prog="fawp-demo",
        description=(
            f"fawp-demo v{_VERSION} — Launch the FAWP Scanner with demo data.\n"
            "No API keys, no CSV files, no setup required."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  fawp-demo                        # synthetic data (instant)\n"
            "  fawp-demo --asset BTC-USD        # real ticker via yfinance\n"
            "  fawp-demo --asset SPY QQQ GLD    # multiple tickers\n"
            "  fawp-demo --port 8502            # different port\n"
        ),
    )
    parser.add_argument(
        "--asset", nargs="*", default=None,
        help="Ticker(s) to pre-load (requires yfinance). "
             "Default: built-in synthetic data.",
    )
    parser.add_argument("--port",       type=int, default=8501)
    parser.add_argument("--no-browser", action="store_true",
                        help="Don't auto-open browser")
    parser.add_argument("--version",    action="version",
                        version=f"fawp-demo {_VERSION}")
    args = parser.parse_args()

    _print_banner()

    # Set env vars for the dashboard to pick up
    env = os.environ.copy()
    env["FAWP_DEMO"] = "1"

    if args.asset:
        tickers = ",".join(t.strip().upper() for t in args.asset)
        env["FAWP_DEMO_TICKERS"] = tickers
        print(f"  Mode    : yfinance  ({tickers})")
    else:
        print("  Mode    : synthetic demo data")
        print("  Assets  : SPY QQQ GLD BTC TLT (600 bars each)")

    print(f"  URL     : http://localhost:{args.port}")
    print()
    print("  Press Ctrl+C to stop.")
    print()

    headless = "true" if args.no_browser else "false"
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(_DASHBOARD),
        "--server.port",     str(args.port),
        "--server.headless", headless,
    ]
    subprocess.run(cmd, env=env)


if __name__ == "__main__":
    launch_demo()
