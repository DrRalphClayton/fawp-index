"""
fawp-index v0.2.0
FAWP Alpha Index — Information-Control Exclusion Principle detector.

Ralph Clayton (2026)
DOI: https://doi.org/10.5281/zenodo.18673949
GitHub: https://github.com/DrRalphClayton/fawp-index

Quick start:
    # From CSV
    from fawp_index import FAWPAlphaIndex
    from fawp_index.io.csv_loader import load_csv_simple
    data = load_csv_simple("data.csv", state_col="price", action_col="trade")
    result = FAWPAlphaIndex().compute(
        data.pred_series, data.future_series,
        data.action_series, data.obs_series
    )
    print(result.summary())
    result.plot()  # leverage gap figure

    # From Yahoo Finance
    from fawp_index.io.feeds import load_yahoo_finance
    data = load_yahoo_finance('SPY', period='2y')
    result = FAWPAlphaIndex().compute(...)

    # Live stream
    from fawp_index import FAWPStreamDetector
    stream = FAWPStreamDetector(window=500, on_fawp=lambda r: print("FAWP!", r.peak_alpha))

    # CLI: fawp-index mydata.csv --state price --action trade --plot
"""

__version__ = "0.2.0"
__author__ = "Ralph Clayton"
__doi__ = "https://doi.org/10.5281/zenodo.18673949"
__github__ = "https://github.com/DrRalphClayton/fawp-index"

from .core.alpha_index import FAWPAlphaIndex, FAWPResult
from .core.estimators import mi_from_arrays, null_corrected_mi, conservative_null_floor
from .stream.live import FAWPStreamDetector
from .io.csv_loader import load_csv, load_csv_simple, FAWPData

def _plot_result(self, **kwargs):
    from .viz.plots import plot_leverage_gap
    return plot_leverage_gap(self, **kwargs)

# Attach .plot() directly to FAWPResult for convenience
FAWPResult.plot = _plot_result

__all__ = [
    "FAWPAlphaIndex",
    "FAWPResult",
    "FAWPStreamDetector",
    "FAWPData",
    "load_csv",
    "load_csv_simple",
    "mi_from_arrays",
    "null_corrected_mi",
    "conservative_null_floor",
]
