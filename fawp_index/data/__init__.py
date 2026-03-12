"""
fawp_index.data — Bundled experimental data (E1-E8)

All paths are guaranteed to exist after pip install fawp-index.

E1 — Parameter sweep (horizon scaling)
E2 — Convergence (CLT scaling of MI estimator)
E3 — Noise-law discrimination (linear / quadratic / saturating)
E4 — Distributional robustness (Gaussian vs Uniform vs Student-t)
E5 — Control cliff (MI decay → stability failure)
E6 — Agency capacity surfaces (quantization + dropout)
E7 — Quantum checks (Bell/CHSH + no-signaling)
E8 — Full FAWP confirmation suite (flagship experiment)
"""
from pathlib import Path

_DATA = Path(__file__).parent

# E1
E1_HORIZONS_SWEEP   = _DATA / "e1_horizons_sweep.csv"

# E2
E2_CONVERGENCE      = _DATA / "e2_convergence.csv"

# E3
E3_CURVES_LINEAR    = _DATA / "e3_curves_linear.csv"
E3_CURVES_QUADRATIC = _DATA / "e3_curves_quadratic.csv"
E3_CURVES_SATURATING= _DATA / "e3_curves_saturating.csv"
E3_HORIZONS_SUMMARY = _DATA / "e3_horizons_summary.csv"

# E4
E4_ROBUSTNESS       = _DATA / "e4_robustness.csv"

# E5
E5_CONTROL_CLIFF    = _DATA / "e5_control_cliff.csv"

# E6
E6_SURFACE_BITS_MEAN    = _DATA / "e6_surface_bits_mean.csv"
E6_SURFACE_BITS_STD     = _DATA / "e6_surface_bits_std.csv"
E6_SURFACE_DROPOUT_MEAN = _DATA / "e6_surface_dropout_mean.csv"
E6_SURFACE_DROPOUT_STD  = _DATA / "e6_surface_dropout_std.csv"

# E7
E7_QUANTUM          = _DATA / "e7_quantum.csv"

# E8
E8_DATA             = _DATA / "e8_data.csv"
E8_CONFIRM_FULL     = _DATA / "e8_confirm_final_full.csv"
E8_SIGNIFICANCE     = _DATA / "e8_confirm_significance.csv"

__all__ = [
    "E1_HORIZONS_SWEEP",
    "E2_CONVERGENCE",
    "E3_CURVES_LINEAR", "E3_CURVES_QUADRATIC", "E3_CURVES_SATURATING", "E3_HORIZONS_SUMMARY",
    "E4_ROBUSTNESS",
    "E5_CONTROL_CLIFF",
    "E6_SURFACE_BITS_MEAN", "E6_SURFACE_BITS_STD",
    "E6_SURFACE_DROPOUT_MEAN", "E6_SURFACE_DROPOUT_STD",
    "E7_QUANTUM",
    "E8_DATA", "E8_CONFIRM_FULL", "E8_SIGNIFICANCE",
]
