"""
fawp_index.constants — Calibration anchors from published research.

All values are derived directly from the published papers:

  VTM   — doi:10.5281/zenodo.18634216
  E1–E7 — doi:10.5281/zenodo.18663547  (Agency Horizon / FORECASTING)
  E8    — doi:10.5281/zenodo.18673949  (Secret Formula / FAWP confirmation)
  E9    — SPHERE_14 confirmation suite

Do NOT change these without a corresponding paper update.
Use them as defaults throughout the package so every module
stays in sync with the research.

Ralph Clayton (2026)
"""

# ── Null-correction ────────────────────────────────────────────────────────
# Conservative null-quantile level (β). Used for both shuffle and shift controls.
# Source: SECRET Eq. 2–3, E9 standard config, SPHERE E9.4 portability sweep.
BETA_NULL_QUANTILE: float = 0.99

# Number of null permutations for floor estimation (shuffle + shift each).
# 200 balances accuracy vs speed; use 50 for fast mode.
N_NULL_DEFAULT: int = 200
N_NULL_FAST: int = 0        # 0 = skip null correction entirely (raw MI)

# ── Detectability thresholds ───────────────────────────────────────────────
# Post-null-correction steering near-null criterion.
# Source: SECRET Eq. 8, calibration note: "ε ~ 10⁻⁴ after floor subtraction"
EPSILON_STEERING_CORRECTED: float = 1e-4

# Raw steering detectability threshold (pre-correction, used by ODWDetector).
# Source: E8 flagship, E9 standard config: ε = 0.01 bits
EPSILON_STEERING_RAW: float = 0.01

# Predictive coupling floor (η) — post-null-correction.
# Source: SECRET calibration note: "η ~ 10⁻⁴ to 10⁻³ bits"
ETA_PRED_CORRECTED: float = 1e-4

# ── Persistence gating ─────────────────────────────────────────────────────
# Robust stability window width m for Sm(τ).
# Source: SECRET Eq. 6, default m=5. Use m=3 for noisier / shorter regimes.
PERSISTENCE_WINDOW_M: int = 5
PERSISTENCE_WINDOW_M_NOISY: int = 3   # for noisier regimes (from paper note)

# m-of-n persistence rule for ODW gate.
# Source: E9.1 ablation, SPHERE E9 standard: (m, n) = (3, 4)
PERSISTENCE_RULE_M: int = 3
PERSISTENCE_RULE_N: int = 4

# ── Alpha Index v2.1 ───────────────────────────────────────────────────────
# Log-slope regularizer δ (avoids log(0) in Rlog computation).
# Source: SECRET Eq. 7
DELTA_LOG_SMOOTH: float = 1e-6

# Resonance weight κ.  κ=1.0 is neutral (paper does not specify; 1.0 preserves
# the leverage gap signal without amplification or suppression).
KAPPA_RESONANCE: float = 1.0

# Minimum tau for causal interpretation — τ=0 excluded as shared-state diagnostic.
# Source: FORECASTING, E8, SECRET all enforce τ ≥ 1.
TAU_MIN: int = 1

# ── Flagship E8 parameters ─────────────────────────────────────────────────
# Source: FORECASTING Eq. 15–17, E8 table, SECRET
FLAGSHIP_A: float = 1.02          # unstable scalar drift (a > 1 → unstable)
FLAGSHIP_K: float = 0.8           # controller gain
FLAGSHIP_DELTA_PRED: int = 20     # forecast horizon Δ
FLAGSHIP_N_TRIALS: int = 400      # trials per τ in E8
FLAGSHIP_X_FAIL: float = 500.0    # failure threshold |x| > x_fail
FLAGSHIP_SIGMA_PROC: float = 1.0  # process noise std σ_w
FLAGSHIP_U_MAX: float = 10.0      # action clip bound

# ── Flagship E8 empirical anchors ─────────────────────────────────────────
# Source: E8 table in FAWP paper; E9 SPHERE confirmation
TAU_PLUS_H_FLAGSHIP: int = 4      # post-zero agency horizon τ⁺ₕ (E8)
TAU_F_FLAGSHIP: int = 35          # functional failure cliff (E8: fail ≥ 0.99)
PEAK_PRED_BITS: float = 2.2337    # peak stratified prediction (at τ=9)
PRED_AT_CLIFF: float = 1.0110     # prediction at cliff (τ=35)  ← NOTE: 1.0110 not 1.1010

# E9.2 / SPHERE confirmed values (20-seed aggregate, 400 trials/τ)
TAU_PLUS_H_E9: int = 31           # τ⁺ₕ for both u and ξ steering
TAU_F_E9: int = 36                # functional cliff in E9
ODW_START_E9: int = 31            # ODW start τ
ODW_END_E9: int = 33              # ODW end τ  (width = 3 steps)
PEAK_GAP_BITS_E9: float = 1.55    # peak leverage gap (E9.2 aggregate)
MEAN_GAP_U_E9: float = 1.1410     # mean gap inside ODW, u-steering
MEAN_GAP_XI_E9: float = 1.1256    # mean gap inside ODW, ξ-steering

# ── Gaussian channel / OATS defaults ──────────────────────────────────────
# Source: VTM Eq. 13–18, AgencyHorizon canonical example
OATS_P: float = 1.0               # signal (action) variance
OATS_SIGMA0_SQ: float = 0.01      # base observation noise variance σ²₀
OATS_ALPHA_NOISE: float = 0.001   # noise growth rate α (variance per unit τ)
OATS_EPSILON: float = 0.01        # MI threshold for horizon definition

# ── Market scanner defaults ────────────────────────────────────────────────
MARKET_WINDOW: int = 252          # rolling window (bars) — one trading year
MARKET_STEP: int = 5              # scan step
MARKET_TAU_MAX: int = 40          # max τ grid
MARKET_FORECAST_DELTA: int = 20   # Δ for predictive channel

# ── LERI photon-record horizon ─────────────────────────────────────────────
# Source: LERI Eq. 11–12
LERI_HORIZON_METERS: float = 14005.0   # ≈ 14.005 km
LERI_HORIZON_SECONDS: float = 4.6683e-5  # ≈ 46.68 µs
