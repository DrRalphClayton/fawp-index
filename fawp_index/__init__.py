"""
fawp-index: FAWP Alpha Index v2.1
Information-Control Exclusion Principle detector.

Based on research by Ralph Clayton (2026):
  - DOI: https://doi.org/10.5281/zenodo.18663547
  - DOI: https://doi.org/10.5281/zenodo.18673949

Quick start:
    from fawp_index import FAWPAlphaIndex, FAWPStreamDetector
    from fawp_index.io.csv_loader import load_csv_simple

    # From CSV
    data = load_csv_simple("mydata.csv", state_col="price", action_col="trade")
    detector = FAWPAlphaIndex()
    result = detector.compute(
        pred_series=data.pred_series,
        future_series=data.future_series,
        action_series=data.action_series,
        obs_series=data.obs_series,
    )
    print(result.summary())

    # Live stream
    stream = FAWPStreamDetector(window=500, on_fawp=lambda r: print("FAWP!", r.peak_alpha))
    for state, action in live_feed:
        stream.update(state, action)
"""

__version__ = "0.1.0"
__author__ = "Ralph Clayton"
__doi__ = "https://doi.org/10.5281/zenodo.18673949"

from .core.alpha_index import FAWPAlphaIndex, FAWPResult
from .core.estimators import mi_from_arrays, null_corrected_mi, conservative_null_floor
from .stream.live import FAWPStreamDetector
from .io.csv_loader import load_csv, load_csv_simple, FAWPData

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
