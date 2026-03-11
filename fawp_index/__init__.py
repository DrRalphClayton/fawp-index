"""
fawp-index v0.3.0
FAWP Alpha Index — Information-Control Exclusion Principle detector.

Ralph Clayton (2026)
DOI: https://doi.org/10.5281/zenodo.18673949
GitHub: https://github.com/DrRalphClayton/fawp-index

--- Core ---
    from fawp_index import FAWPAlphaIndex
    result = FAWPAlphaIndex().compute(pred, future, action, obs)
    result.plot()

--- DataFrame API ---
    from fawp_index import fawp_from_dataframe
    result = fawp_from_dataframe(df, pred_col='returns', action_col='volume')

--- Sklearn API ---
    from fawp_index.sklearn_api import FAWPTransformer
    fawp = FAWPTransformer(pred_col=0, action_col=1).fit(X)

--- Feature Importance ---
    from fawp_index.features import FAWPFeatureImportance
    result = FAWPFeatureImportance(action_col='volume').fit(df)

--- Quant Finance ---
    from fawp_index.quant import FAWPRegimeDetector, MomentumDecayDetector
    from fawp_index.quant import RiskParityWarning, EventStudyFAWP

--- CLI ---
    fawp-index mydata.csv --state price --action trade --plot
"""

__version__ = "0.3.0"
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
    "mi_from_arrays", "null_corrected_mi", "conservative_null_floor",
]
