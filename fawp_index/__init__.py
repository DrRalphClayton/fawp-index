"""
fawp-index v0.10.0
FAWP Alpha Index — Information-Control Exclusion Principle detector.
Includes full E1-E9 experimental suite data.
Ralph Clayton (2026) · doi:10.5281/zenodo.18673949
"""

__version__ = "0.10.0"
__author__ = "Ralph Clayton"
__doi__ = "https://doi.org/10.5281/zenodo.18673949"
__github__ = "https://github.com/DrRalphClayton/fawp-index"

from .core.alpha_index import FAWPAlphaIndex, FAWPResult
from .core.alpha_v2 import FAWPAlphaIndexV2, AlphaV2Result
from .core.estimators import mi_from_arrays, null_corrected_mi, conservative_null_floor
from .detection.odw import ODWDetector, ODWResult
from .io.csv_loader import load_csv, load_csv_simple, FAWPData
from .dataframe_api import fawp_from_dataframe, fawp_rolling
from .sklearn_api import FAWPTransformer
from .features import FAWPFeatureImportance
from .multivariate import MultivariateFAWP, MultivariateFAWPResult
from .simulate import FAWPSimulator, SimulationResult, ControlCliff, ControlCliffResult
from .explain import explain, explain_fawp, explain_oats, explain_control_cliff
from .report import generate_report, FAWPReport
from .watchlist import (
    WatchlistScanner, scan_watchlist,
    WatchlistResult, AssetResult,
)
from .alerts import (
    AlertEngine, FAWPAlert, AlertType,
)
from .market import (
    FAWPMarketScanner, scan_fawp_market,
    MarketScanConfig, MarketScanSeries, MarketWindowResult,
)
from .significance import fawp_significance, FAWPSignificance, SignificanceResult
from .compare import compare_fawp, ComparisonResult
from .benchmarks import (
    run_all as run_benchmarks,
    BenchmarkSuite, BenchmarkResult, BenchmarkFailure,
    clean_control, prediction_only, control_only,
    noisy_false_positive, delayed_collapse,
)
from .exports import _inject_exports
_inject_exports()

def _plot_result(self, **kwargs):
    from .viz.plots import plot_leverage_gap
    return plot_leverage_gap(self, **kwargs)

FAWPResult.plot = _plot_result

__all__ = [
    "FAWPAlphaIndex", "FAWPResult",
    "FAWPAlphaIndexV2", "AlphaV2Result",
    "ODWDetector", "ODWResult",
        "FAWPData", "load_csv", "load_csv_simple",
    "fawp_from_dataframe", "fawp_rolling",
    "FAWPTransformer",
    "FAWPFeatureImportance",
    "MultivariateFAWP", "MultivariateFAWPResult",
    "FAWPSimulator", "SimulationResult",
    "ControlCliff", "ControlCliffResult",
    "explain", "explain_fawp", "explain_oats", "explain_control_cliff",
    "generate_report", "FAWPReport",
    "FAWPMarketScanner", "scan_fawp_market",
    "MarketScanConfig", "MarketScanSeries", "MarketWindowResult",
    "fawp_significance", "FAWPSignificance", "SignificanceResult",
    "compare_fawp", "ComparisonResult",
    "run_benchmarks", "BenchmarkSuite", "BenchmarkResult", "BenchmarkFailure",
    "clean_control", "prediction_only", "control_only",
    "noisy_false_positive", "delayed_collapse",
    # exports injected onto result classes: .to_json / .to_markdown / .to_html / .to_dict
    "mi_from_arrays", "null_corrected_mi", "conservative_null_floor",
]
