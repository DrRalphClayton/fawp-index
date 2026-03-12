"""
fawp-index v0.5.0
FAWP Alpha Index — Information-Control Exclusion Principle detector.
Includes full E1-E8 experimental suite data.
Ralph Clayton (2026) · doi:10.5281/zenodo.18673949
"""

__version__ = "0.5.0"
__author__ = "Ralph Clayton"
__doi__ = "https://doi.org/10.5281/zenodo.18673949"
__github__ = "https://github.com/DrRalphClayton/fawp-index"

from .core.alpha_index import FAWPAlphaIndex, FAWPResult
from .core.estimators import mi_from_arrays, null_corrected_mi, conservative_null_floor
from .stream.live import FAWPStreamDetector
from .io.csv_loader import load_csv, load_csv_simple, FAWPData
from .dataframe_api import fawp_from_dataframe, fawp_rolling
from .sklearn_api import FAWPTransformer
from .features import FAWPFeatureImportance
from .multivariate import MultivariateFAWP, MultivariateFAWPResult
from .simulate import FAWPSimulator, SimulationResult, ControlCliff, ControlCliffResult

def _plot_result(self, **kwargs):
    from .viz.plots import plot_leverage_gap
    return plot_leverage_gap(self, **kwargs)

FAWPResult.plot = _plot_result

__all__ = [
    "FAWPAlphaIndex", "FAWPResult",
    "FAWPStreamDetector",
    "FAWPData", "load_csv", "load_csv_simple",
    "fawp_from_dataframe", "fawp_rolling",
    "FAWPTransformer",
    "FAWPFeatureImportance",
    "MultivariateFAWP", "MultivariateFAWPResult",
    "FAWPSimulator", "SimulationResult",
    "ControlCliff", "ControlCliffResult",
    "mi_from_arrays", "null_corrected_mi", "conservative_null_floor",
]
