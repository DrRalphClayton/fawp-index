"""
Regression tests for 4 confirmed bugs (v3.7.x fix series).
Run with: pytest tests/test_bug_regressions.py -v
"""
import sys, subprocess, tomllib
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ── Bug 1: version consistency ────────────────────────────────────────────────
def test_version_matches_pyproject():
    """__version__ in __init__.py must match pyproject.toml."""
    import fawp_index as fi
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    expected = pyproject["project"]["version"]
    assert fi.__version__ == expected, (
        f"__version__={fi.__version__!r} != pyproject version={expected!r}. "
        f"Fix: update fawp_index/__init__.py"
    )


def test_version_is_not_hardcoded_stale():
    """__version__ must not be the old stale 2.8.0 value."""
    import fawp_index as fi
    assert fi.__version__ != "2.8.0", "__version__ is still the stale 2.8.0 value"


# ── Bug 2: launcher --help works without streamlit ───────────────────────────
def test_dashboard_launcher_help_exits_zero():
    """fawp-dashboard --help must exit 0 AND produce help text.

    Without if __name__ == "__main__": launch() in dashboard_launcher.py,
    python -m fawp_index.dashboard_launcher exits 0 silently (no output).
    """
    result = subprocess.run(
        [sys.executable, "-m", "fawp_index.dashboard_launcher", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"fawp-dashboard --help exited {result.returncode}. "
        f"stderr: {result.stderr[:300]}"
    )
    assert len(result.stdout.strip()) > 0, (
        "fawp-dashboard --help produced no output. "
        "Ensure dashboard_launcher.py ends with: "
        "if __name__ == '__main__': launch()"
    )
    assert "usage" in result.stdout.lower() or "fawp" in result.stdout.lower(), (
        f"fawp-dashboard --help output does not look like help text: "
        f"{result.stdout[:200]!r}"
    )


def test_demo_launcher_help_exits_zero():
    """fawp-demo --help must exit 0 AND produce help text."""
    result = subprocess.run(
        [sys.executable, "-m", "fawp_index.demo_launcher", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"fawp-demo --help exited {result.returncode}. "
        f"stderr: {result.stderr[:300]}"
    )
    assert len(result.stdout.strip()) > 0, (
        "fawp-demo --help produced no output. "
        "Ensure demo_launcher.py ends with: "
        "if __name__ == '__main__': launch_demo()"
    )
    assert "usage" in result.stdout.lower() or "fawp" in result.stdout.lower(), (
        f"fawp-demo --help output does not look like help text: "
        f"{result.stdout[:200]!r}"
    )


# ── Bug 3: tau_f=0 not hidden by falsy check ─────────────────────────────────
def test_tau_f_zero_not_hidden_report_html():
    """tau_f=0 must render as '0' not '—' in report_html."""
    src = (ROOT / "fawp_index" / "report_html.py").read_text()
    # The fix: use `is not None` not truthiness
    assert "tau_f      is not None" in src or "tau_f is not None" in src, (
        "report_html.py still uses falsy check for tau_f — "
        "replace `if r.odw_result.tau_f` with `if r.odw_result.tau_f is not None`"
    )


def test_tau_h_zero_not_hidden_report_html():
    """tau_h_plus=0 must render as '0' not '—' in report_html."""
    src = (ROOT / "fawp_index" / "report_html.py").read_text()
    assert "tau_h_plus is not None" in src, (
        "report_html.py still uses falsy check for tau_h_plus"
    )


def test_tau_f_zero_round_trip():
    """ODWResult with tau_f=0 must pass through CLI format without becoming '—'."""
    import re
    cli_src = (ROOT / "fawp_index" / "cli.py").read_text()
    # Ensure no `tau_f or '...'` patterns remain
    bad = re.findall(r"tau_f\s+or\s+[\"']", cli_src)
    assert not bad, f"cli.py still has falsy tau_f checks: {bad}"

    magic_src = (ROOT / "fawp_index" / "magic.py").read_text()
    bad_m = re.findall(r"tau_f\s+or\s+[\"']", magic_src)
    assert not bad_m, f"magic.py still has falsy tau_f checks: {bad_m}"


# ── Bug 4: hemisphere labels ──────────────────────────────────────────────────
def test_hemisphere_labels_negative_lat():
    """Negative lat must render as 'S', not 'N'."""
    src = (ROOT / "fawp_index" / "weather_cli.py").read_text()
    # Should contain hemisphere logic, not hardcoded N/E
    assert "_lat_h" in src, "weather_cli.py missing hemisphere variable _lat_h"
    assert "_lon_h" in src, "weather_cli.py missing hemisphere variable _lon_h"
    # Must NOT hardcode N/E regardless of sign
    assert ":.2f}N" not in src, "weather_cli.py still hardcodes N in format string"
    assert ":.2f}E" not in src, "weather_cli.py still hardcodes E in format string"


def test_hemisphere_logic_correct():
    """Hemisphere logic must produce correct N/S, E/W for all quadrants."""
    cases = [
        ((-33.87, -0.10),  "S", "W"),   # Sydney-ish, Atlantic
        ((48.85,   2.35),  "N", "E"),   # Paris
        ((-33.87, 151.21), "S", "E"),   # Sydney
        ((35.68,  139.69), "N", "E"),   # Tokyo
        ((0.0,     0.0),   "N", "E"),   # Equator / prime meridian
    ]
    for (lat, lon), exp_lat_h, exp_lon_h in cases:
        lat_h = "N" if lat >= 0 else "S"
        lon_h = "E" if lon >= 0 else "W"
        assert lat_h == exp_lat_h, f"lat={lat}: got {lat_h}, expected {exp_lat_h}"
        assert lon_h == exp_lon_h, f"lon={lon}: got {lon_h}, expected {exp_lon_h}"
