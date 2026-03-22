"""
fawp_index.weather_cli — CLI for FAWP weather detection.

Entry point: fawp-weather

Usage:
    fawp-weather scan --lat 51.5 --lon -0.1 --variable temperature_2m
    fawp-weather scan --lat 51.5 --lon -0.1 --start 2010-01-01 --end 2024-12-31
    fawp-weather grid --cities london paris newyork tokyo sydney
    fawp-weather list-variables

Ralph Clayton (2026) · doi:10.5281/zenodo.18673949
"""

import argparse
import json
import sys

from fawp_index import __version__ as _VERSION


# ── Known city presets ────────────────────────────────────────────────────────
_CITIES = {
    "london":     (51.50,  -0.10),
    "paris":      (48.86,   2.35),
    "newyork":    (40.71, -74.01),
    "tokyo":      (35.69, 139.69),
    "sydney":    (-33.87, 151.21),
    "dubai":      (25.20,  55.27),
    "berlin":     (52.52,  13.40),
    "chicago":    (41.88, -87.63),
    "beijing":    (39.91, 116.40),
    "saopaulo":  (-23.55, -46.63),
    "mumbai":     (19.08,  72.88),
    "cairo":      (30.04,  31.24),
    "moscow":     (55.75,  37.62),
    "toronto":    (43.65, -79.38),
}

_VARIABLES = [
    "temperature_2m", "precipitation_sum", "wind_speed_10m",
    "surface_pressure", "cloud_cover", "shortwave_radiation",
    "et0_fao_evapotranspiration",
]


def _print_banner():
    print(f"\nfawp-weather v{_VERSION} · doi:10.5281/zenodo.18673949")
    print("FAWP Weather & Climate Scanner\n")


def cmd_scan(args):
    """Single location scan."""
    _print_banner()
    try:
        from fawp_index.weather import fawp_from_open_meteo
    except ImportError as e:
        print(f"Error: {e}\nInstall with: pip install 'fawp-index[weather]'")
        sys.exit(1)

    # Resolve city preset if given
    if args.city:
        city = args.city.lower().replace(" ", "")
        if city not in _CITIES:
            print(f"Unknown city '{args.city}'. Use --lat/--lon or try: {list(_CITIES)[:8]}")
            sys.exit(1)
        lat, lon = _CITIES[city]
        location = args.city.title()
    else:
        lat, lon = args.lat, args.lon
        location = f"({lat:.2f}N, {lon:.2f}E)"

    print(f"Scanning: {location}")
    print(f"Variable: {args.variable}")
    print(f"Period  : {args.start} → {args.end}")
    print(f"Horizon : {args.horizon} day(s)")
    print()

    result = fawp_from_open_meteo(
        latitude           = lat,
        longitude          = lon,
        variable           = args.variable,
        start_date         = args.start,
        end_date           = args.end,
        horizon_days       = args.horizon,
        tau_max            = args.tau_max,
        epsilon            = args.epsilon,
        n_null             = args.n_null,
        remove_seasonality = args.remove_seasonality,
        estimator          = args.estimator,
    )

    print(result.summary())

    if args.out:
        path = args.out
        if path.endswith(".json"):
            with open(path, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
        elif path.endswith(".csv"):
            import pandas as pd
            import numpy as np
            pd.DataFrame({
                "tau": result.tau,
                "pred_mi": result.pred_mi,
                "steer_mi": result.steer_mi,
            }).to_csv(path, index=False)
        print(f"\nSaved → {path}")


def cmd_grid(args):
    """Multi-city grid scan."""
    _print_banner()
    try:
        from fawp_index.weather import scan_weather_grid
    except ImportError as e:
        print(f"Error: {e}\nInstall with: pip install 'fawp-index[weather]'")
        sys.exit(1)

    locations = []
    for city in args.cities:
        key = city.lower().replace(" ", "")
        if key in _CITIES:
            lat, lon = _CITIES[key]
            locations.append({"lat": lat, "lon": lon, "name": city.title()})
        else:
            print(f"Warning: unknown city '{city}' — skipping. "
                  f"Known: {list(_CITIES.keys())}")

    if not locations:
        print("No valid locations. Use city names or --lat/--lon with scan command.")
        sys.exit(1)

    print(f"Scanning {len(locations)} location(s): "
          f"{', '.join(l['name'] for l in locations)}\n")

    results = scan_weather_grid(
        locations    = locations,
        variable     = args.variable,
        start_date   = args.start,
        end_date     = args.end,
        horizon_days = args.horizon,
        tau_max      = args.tau_max,
        n_null       = args.n_null,
        verbose      = True,
    )

    print(f"\n{'Location':<20} {'FAWP':>6}  {'Gap (bits)':>10}  {'ODW':<12}  {'τ⁺ₕ':>4}")
    print("-" * 58)
    for r in results:
        flag  = "🔴 YES" if r.fawp_found else "—"
        odw   = f"τ {r.odw_start}–{r.odw_end}" if r.fawp_found else "—"
        tau_h = str(r.odw_result.tau_h_plus) if r.odw_result.tau_h_plus else "—"
        print(f"{r.location:<20} {flag:>6}  {r.peak_gap_bits:>10.4f}  {odw:<12}  {tau_h:>4}")

    if args.out:
        data = [r.to_dict() for r in results]
        with open(args.out, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nSaved → {args.out}")


def cmd_list_variables(args):
    print("\nSupported ERA5 variables:\n")
    descriptions = {
        "temperature_2m":             "2m air temperature (°C)",
        "precipitation_sum":          "Daily total precipitation (mm)",
        "wind_speed_10m":             "10m wind speed (m/s)",
        "surface_pressure":           "Surface pressure (hPa)",
        "cloud_cover":                "Total cloud cover (%)",
        "shortwave_radiation":        "Shortwave solar radiation (W/m²)",
        "et0_fao_evapotranspiration": "Reference evapotranspiration (mm)",
    }
    for var, desc in descriptions.items():
        print(f"  {var:<35} {desc}")
    print()


def main():
    parser = argparse.ArgumentParser(
        prog="fawp-weather",
        description=f"fawp-weather v{_VERSION} — FAWP weather & climate scanner",
    )
    parser.add_argument("--version", action="version", version=f"fawp-weather {_VERSION}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    # ── scan ──────────────────────────────────────────────────────────────────
    p_scan = sub.add_parser("scan", help="Scan a single location for FAWP")
    p_scan.add_argument("--city",     default=None, help="City name (e.g. london, tokyo)")
    p_scan.add_argument("--location", default=None, dest="city",
                        help="Alias for --city (e.g. \"London\", \"New York\")")
    p_scan.add_argument("--lat",      type=float, default=51.5)
    p_scan.add_argument("--lon",      type=float, default=-0.1)
    p_scan.add_argument("--variable", default="temperature_2m", choices=_VARIABLES)
    p_scan.add_argument("--start",    default="2010-01-01")
    p_scan.add_argument("--end",      default="2024-12-31")
    p_scan.add_argument("--horizon",  type=int,   default=7)
    p_scan.add_argument("--tau-max",  type=int,   default=30, dest="tau_max")
    p_scan.add_argument("--epsilon",  type=float, default=0.01)
    p_scan.add_argument("--n-null",   type=int,   default=50, dest="n_null")
    p_scan.add_argument("--out",      default=None, help="Save result to .json or .csv")
    p_scan.add_argument("--estimator", default="pearson", choices=["pearson", "knn"],
                        help="MI estimator: pearson (fast, default) or knn (non-Gaussian, requires sklearn)")
    p_scan.add_argument("--remove-seasonality", dest="remove_seasonality", action="store_true",
                        help="Remove annual seasonal cycle before detection")

    # ── grid ──────────────────────────────────────────────────────────────────
    p_grid = sub.add_parser("grid", help="Scan multiple cities")
    p_grid.add_argument("--cities",   nargs="+", default=["london","paris","newyork"],
                        help="City names to scan")
    p_grid.add_argument("--variable", default="temperature_2m", choices=_VARIABLES)
    p_grid.add_argument("--start",    default="2010-01-01")
    p_grid.add_argument("--end",      default="2024-12-31")
    p_grid.add_argument("--horizon",  type=int, default=7)
    p_grid.add_argument("--tau-max",  type=int, default=30, dest="tau_max")
    p_grid.add_argument("--n-null",   type=int, default=50, dest="n_null")
    p_grid.add_argument("--out",      default=None, help="Save results to .json")

    # ── list-variables ────────────────────────────────────────────────────────
    sub.add_parser("list-variables", help="List supported ERA5 variables")

    args = parser.parse_args()
    if args.command == "scan":
        cmd_scan(args)
    elif args.command == "grid":
        cmd_grid(args)
    elif args.command == "list-variables":
        cmd_list_variables(args)


if __name__ == "__main__":
    main()
