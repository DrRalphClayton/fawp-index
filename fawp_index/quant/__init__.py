"""
fawp_index.quant — Quantitative Finance Extensions

Four tools for quant analysts and portfolio managers:

    FAWPRegimeDetector   — rolling-window market regime detection
    MomentumDecayDetector — crowded-trade / execution edge collapse
    RiskParityWarning    — vol-targeting strategy failure early warning
    EventStudyFAWP       — FAWP signature around earnings/announcements

All based on the Information-Control Exclusion Principle:
    Ralph Clayton (2026) — doi:10.5281/zenodo.18673949

Quick start:
    from fawp_index.quant import FAWPRegimeDetector
    detector = FAWPRegimeDetector(window=252, step=21)
    result = detector.detect(returns, volumes)
    print(result.summary())
"""

from .regime import FAWPRegimeDetector, RegimeResult
from .momentum import MomentumDecayDetector, MomentumDecayResult
from .risk import RiskParityWarning, RiskWarningResult
from .events import EventStudyFAWP, EventStudyResult

__all__ = [
    "FAWPRegimeDetector", "RegimeResult",
    "MomentumDecayDetector", "MomentumDecayResult",
    "RiskParityWarning", "RiskWarningResult",
    "EventStudyFAWP", "EventStudyResult",
]
