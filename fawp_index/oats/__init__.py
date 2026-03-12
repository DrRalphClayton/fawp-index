"""
fawp_index.oats — Analytic Agency Horizon (OATS model)

OATS = Operational Agency Time-Scaling

Closed-form Gaussian channel model for the Agency Horizon:
    I(τ) = 0.5 · log2(1 + P / σ²(τ))
    σ²(τ) = σ0² + α·τ   (linear noise growth with latency)

Agency horizon:
    τ_h = max(0, (P/(2^(2ε) - 1) - σ0²) / α)

This is the analytic foundation underlying E1-E4 of the experimental suite.
The FAWPSimulator (E5/E8) is the empirical validation of these predictions.

Ralph Clayton (2026) — doi:10.5281/zenodo.18663547
"""
from .model import AgencyHorizon, OATSResult, noise_variance, mutual_information
from .robustness import DistributionalRobustness, RobustnessResult
from .surfaces import CapacitySurface

__all__ = [
    "AgencyHorizon", "OATSResult",
    "noise_variance", "mutual_information",
    "DistributionalRobustness", "RobustnessResult",
    "CapacitySurface",
]
