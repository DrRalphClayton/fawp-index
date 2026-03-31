"""
fawp_index.dashboard_launcher
==============================
Entry point for the `fawp-dashboard` CLI command.

Usage::

    fawp-dashboard
    fawp-dashboard --port 8502
    fawp-dashboard --browser
"""
import sys
import subprocess
from pathlib import Path

from fawp_index import __version__ as _VERSION

_DASHBOARD = Path(__file__).parent.parent / "dashboard" / "app.py"


def launch():
    # Parse args FIRST so --help/--version work without optional imports
    import argparse as _ap_l
    _ap_l.ArgumentParser(
        prog="dashboard",
        description=f"FAWP Scanner (fawp-index v{_VERSION})",
    ).parse_known_args()  # non-fatal; real parsing below

    # Verify Streamlit is available
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print(
            "Streamlit is not installed.\n"
            "Install it with: pip install 'fawp-index[dashboard]'\n"
            "or:              pip install streamlit"
        )
        sys.exit(1)

    if not _DASHBOARD.exists():
        print(
            f"Dashboard not found at {_DASHBOARD}.\n"
            "Clone the repo and run from there: "
            "https://github.com/DrRalphClayton/fawp-index"
        )
        sys.exit(1)

    # Parse simple --port / --browser args
    import argparse
    parser = argparse.ArgumentParser(
        prog="fawp-dashboard",
        description=f"Launch the FAWP Dashboard (fawp-index v{_VERSION})",
    )
    parser.add_argument("--port",    type=int, default=8501, help="Port (default: 8501)")
    parser.add_argument("--browser", action="store_true",    help="Open browser automatically")
    parser.add_argument("--version", action="version", version=f"fawp-index {_VERSION}")
    args = parser.parse_args()

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(_DASHBOARD),
        "--server.port", str(args.port),
        "--server.headless", "false" if args.browser else "true",
    ]
    print(f"fawp-index v{_VERSION} | Starting dashboard on http://localhost:{args.port}")
    subprocess.run(cmd)


if __name__ == "__main__":
    launch()
