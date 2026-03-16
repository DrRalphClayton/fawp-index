"""
fawp-index v0.4.0
FAWP Alpha Index — Information-Control Exclusion Principle detector.
Includes full E1-E9 experimental suite data.
Ralph Clayton (2026) · doi:10.5281/zenodo.18673949
"""

__version__ = "0.4.0"
__author__ = "Ralph Clayton"
__doi__ = "https://doi.org/10.5281/zenodo.18673949"
__github__ = "https://github.com/DrRalphClayton/fawp-index"

from .validation import validate_signals, ValidationReport, HorizonStats
from .weather import (
    fawp_from_forecast, fawp_from_skill_series,
    fawp_from_open_meteo, scan_weather_grid,
    WeatherFAWPResult, fetch_openmeteo, to_fawp_dataframe,
    plot_weather_map, fawp_rolling_timeline,
    compare_locations, fawp_from_nwp_csvs,
)
from .report_html import generate_html_report
from .compare import compare_signals, CompareReport, compare_fawp, ComparisonResult
from .scan_history import ScanHistory
from .constants import (
    BETA_NULL_QUANTILE,
    EPSILON_STEERING_CORRECTED,
    EPSILON_STEERING_RAW,
    ETA_PRED_CORRECTED,
    PERSISTENCE_WINDOW_M,
    PERSISTENCE_RULE_M,
    PERSISTENCE_RULE_N,
    DELTA_LOG_SMOOTH,
    KAPPA_RESONANCE,
    TAU_MIN,
    FLAGSHIP_A,
    FLAGSHIP_K,
    FLAGSHIP_DELTA_PRED,
    FLAGSHIP_N_TRIALS,
    TAU_PLUS_H_FLAGSHIP,
    TAU_F_FLAGSHIP,
    PEAK_PRED_BITS,
    PRED_AT_CLIFF,
    TAU_PLUS_H_E9,
    TAU_F_E9,
    ODW_START_E9,
    ODW_END_E9,
    PEAK_GAP_BITS_E9_U, PEAK_GAP_BITS_E9_XI, PEAK_GAP_TAU_E9,
)
from .core.alpha_index import FAWPAlphaIndex, FAWPResult
from .core.alpha_v2 import FAWPAlphaIndexV2, AlphaV2Result
from .core.estimators import (
    mi_from_arrays, null_corrected_mi, conservative_null_floor, has_numba,
)
from .detection.odw import ODWDetector, ODWResult
from .io.csv_loader import load_csv, load_csv_simple, FAWPData
from .dataframe_api import fawp_from_dataframe, fawp_rolling
from .sklearn_api import FAWPTransformer
from .features import FAWPFeatureImportance
from .multivariate import MultivariateFAWP, MultivariateFAWPResult
from .simulate import FAWPSimulator, SimulationResult, ControlCliff, ControlCliffResult
from .explain import (
    explain, explain_fawp, explain_oats, explain_control_cliff,
    explain_asset, confidence_badge,
    attribute_gap, attribute_windows, attribution_report,
)
from .report import generate_report, FAWPReport
from .scanner import (
    scan_crypto, scan_equities, scan_sectors, scan_etfs, scan_macro,
    PRESETS,
)
from .watchlist import (
    WatchlistScanner, scan_watchlist,
    WatchlistResult, AssetResult,
)
from .alert_template_presets import TRADING_DESK, RESEARCH, MINIMAL, ALL_PRESETS
from .alerts import (
    AlertEngine, FAWPAlert, AlertType, AlertSeverity,
    WEATHER_ALERT_TEMPLATES, render_weather_alert,
)
from .leaderboard import Leaderboard, LeaderboardEntry
from .watchlist_store import WatchlistStore
from .market import (
    FAWPMarketScanner, scan_fawp_market,
    MarketScanConfig, MarketScanSeries, MarketWindowResult,
)
from .significance import fawp_significance, FAWPSignificance, SignificanceResult
from .benchmarks import (
    run_all as run_benchmarks,
    BenchmarkSuite, BenchmarkResult, BenchmarkFailure,
    clean_control, prediction_only, control_only,
    noisy_false_positive, delayed_collapse,
    gradual_fade, multi_regime, spiky_false_positive,
    hurricane_path, drought_persistence, extreme_precip_spike,
)
from .exports import _inject_exports
_inject_exports()

def _plot_result(self, **kwargs):
    from .viz.plots import plot_leverage_gap
    return plot_leverage_gap(self, **kwargs)

FAWPResult.plot = _plot_result

__all__ = [
    # Validation
    "validate_signals", "ValidationReport", "HorizonStats",
    # Weather / climate
    "fawp_from_forecast", "fawp_from_skill_series",
    "fawp_from_open_meteo", "scan_weather_grid", "WeatherFAWPResult",
    "fetch_openmeteo", "to_fawp_dataframe", "plot_weather_map",
    "fawp_rolling_timeline", "compare_locations", "fawp_from_nwp_csvs",
    "generate_html_report",
    "compare_signals", "CompareReport", "compare_fawp", "ComparisonResult",
    # Scan history
    "ScanHistory",
    # Calibration constants (paper-derived)
    "BETA_NULL_QUANTILE",
    "EPSILON_STEERING_CORRECTED", "EPSILON_STEERING_RAW",
    "ETA_PRED_CORRECTED",
    "PERSISTENCE_WINDOW_M", "PERSISTENCE_RULE_M", "PERSISTENCE_RULE_N",
    "DELTA_LOG_SMOOTH", "KAPPA_RESONANCE", "TAU_MIN",
    "FLAGSHIP_A", "FLAGSHIP_K", "FLAGSHIP_DELTA_PRED", "FLAGSHIP_N_TRIALS",
    "TAU_PLUS_H_FLAGSHIP", "TAU_F_FLAGSHIP",
    "PEAK_PRED_BITS", "PRED_AT_CLIFF",
    "TAU_PLUS_H_E9", "TAU_F_E9", "ODW_START_E9", "ODW_END_E9", "PEAK_GAP_BITS_E9_U", "PEAK_GAP_BITS_E9_XI", "PEAK_GAP_TAU_E9",
    # Core detector
    "FAWPAlphaIndex", "FAWPResult",
    "FAWPAlphaIndexV2", "AlphaV2Result",
    "ODWDetector", "ODWResult",
    # I/O
    "FAWPData", "load_csv", "load_csv_simple",
    # DataFrame / sklearn APIs
    "fawp_from_dataframe", "fawp_rolling",
    "FAWPTransformer",
    "FAWPFeatureImportance",
    # Multivariate / simulation
    "MultivariateFAWP", "MultivariateFAWPResult",
    "FAWPSimulator", "SimulationResult",
    "ControlCliff", "ControlCliffResult",
    # Explain / report
    "explain", "explain_fawp", "explain_oats", "explain_control_cliff",
    "explain_asset", "confidence_badge",
    "attribute_gap", "attribute_windows", "attribution_report",
    "generate_report", "FAWPReport",
    # Scanner presets
    "scan_crypto", "scan_equities", "scan_sectors", "scan_etfs", "scan_macro",
    "PRESETS",
    # Watchlist
    "WatchlistScanner", "scan_watchlist",
    "WatchlistResult", "AssetResult",
    # Alerts
    "AlertEngine", "FAWPAlert", "AlertType", "AlertSeverity",
    "WEATHER_ALERT_TEMPLATES", "render_weather_alert",
    "TRADING_DESK", "RESEARCH", "MINIMAL", "ALL_PRESETS",
    "AlertSeverity",  # re-export
    "gradual_fade", "multi_regime", "spiky_false_positive",
    # Leaderboard
    "Leaderboard", "LeaderboardEntry",
    # Saved watchlists
    "WatchlistStore",
    # Market scanner
    "FAWPMarketScanner", "scan_fawp_market",
    "MarketScanConfig", "MarketScanSeries", "MarketWindowResult",
    # Significance / compare
    "fawp_significance", "FAWPSignificance", "SignificanceResult",
    # Benchmarks
    "run_benchmarks", "BenchmarkSuite", "BenchmarkResult", "BenchmarkFailure",
    "clean_control", "prediction_only", "control_only",
    "noisy_false_positive", "delayed_collapse",
    "gradual_fade", "multi_regime", "spiky_false_positive",
    # MI estimators
    # (exports also injected onto result classes: .to_json / .to_markdown / .to_html / .to_dict)
    "mi_from_arrays", "null_corrected_mi", "conservative_null_floor", "has_numba",
]
