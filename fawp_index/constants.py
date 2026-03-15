"""
fawp_index.constants — Calibration anchors from published research.

All values are derived directly from the published papers:

  VTM   — doi:10.5281/zenodo.18634216
  E1–E7 — doi:10.5281/zenodo.18663547  (Agency Horizon / FORECASTING)
  E8    — doi:10.5281/zenodo.18673949  (Secret Formula / FAWP confirmation)
  E9    — SPHERE_15 confirmation suite (doi:10.5281/zenodo.18693949 pending)

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
PEAK_PRED_BITS: float = 2.1964    # peak corrected stratified prediction (E9.2, τ=9)
PRED_AT_CLIFF: float = 1.0110     # prediction at cliff (τ=35)  ← NOTE: 1.0110 not 1.1010

# E9.2 / SPHERE confirmed values (20-seed aggregate, 400 trials/τ)
TAU_PLUS_H_E9: int = 31           # τ⁺ₕ for both u and ξ steering
TAU_F_E9: int = 36                # functional cliff in E9
ODW_START_E9: int = 31            # ODW start τ
ODW_END_E9: int = 33              # ODW end τ  (width = 3 steps)
PEAK_GAP_BITS_E9_U:  float = 1.5489  # peak leverage gap, u-steering  (E9.2, τ=34)
PEAK_GAP_BITS_E9_XI: float = 1.5524  # peak leverage gap, ξ-steering  (E9.2, τ=34)
PEAK_GAP_TAU_E9:     int   = 34      # τ at peak gap                   (E9.2)
MEAN_GAP_U_E9:   float = 1.1298   # mean gap inside ODW, u-steering  (E9.3 baseline)
MEAN_GAP_XI_E9:  float = 1.1486   # mean gap inside ODW, ξ-steering  (E9.3 baseline)

# ── E9 suite confirmed anchors (SPHERE_15) ────────────────────────────────
# Source: E9.1 ablation on E8 seed data
E91_TAU_PLUS_H: int    = 4       # post-zero horizon (E8 seed, E9.1 ablation)
E91_TAU_F: int         = 35      # failure cliff
E91_ODW_START: int     = 4       # ODW start τ
E91_ODW_END: int       = 34      # ODW end τ  (width = 31 steps)
E91_ODW_SIZE: int      = 31      # ODW width in delay steps
E91_MEAN_LEAD: float   = 15.8    # mean lead time to cliff (delay steps)
E91_FP_RATE: float     = 0.0     # false-positive rate (shuffle + shift nulls)

# E9.3 persistence sweep — baseline 3-of-4 rule (20 seeds)
E93_MEAN_ODW_START_U:  float = 30.9    # mean ODW start, u-steering
E93_MEAN_ODW_START_XI: float = 31.0    # mean ODW start, ξ-steering
E93_MEAN_ODW_END_U:    float = 33.1    # mean ODW end,   u-steering
E93_MEAN_ODW_END_XI:   float = 33.3    # mean ODW end,   ξ-steering
E93_MEAN_ODW_SIZE_U:   float = 3.2     # mean ODW width, u-steering
E93_MEAN_ODW_SIZE_XI:  float = 3.3     # mean ODW width, ξ-steering
E93_MEAN_TAU_H_U:      float = 30.15   # mean τ⁺ₕ, u-steering
E93_MEAN_TAU_H_XI:     float = 30.35   # mean τ⁺ₕ, ξ-steering

# E9.4 null-quantile portability — confirmed stable across β ∈ {0.90→0.995}
E94_BETA_GRID: tuple   = (0.90, 0.95, 0.975, 0.99, 0.995)
E94_FAWP_RATE: float   = 1.0    # detection rate across full beta grid
E94_TAU_F: float       = 36.0   # cliff fixed across all β

# E9.5 regime map — flagship basin (a=1.02, K=0.8)
E95_FAWP_CONFIGS_U:  int   = 10   # FAWP-positive configs (u),  out of 30
E95_FAWP_CONFIGS_XI: int   = 19   # FAWP-positive configs (ξ),  out of 30
E95_FLAGSHIP_ODW_U:  tuple = (30.625, 33.5)   # mean ODW, u-steering
E95_FLAGSHIP_ODW_XI: tuple = (31.0,   33.0)   # mean ODW, ξ-steering

# E9.6 timing — flagship (E9.2 aggregate)
E96_TAU_ALPHA: int    = 35    # detector peak
E96_TAU_RISE:  int    = 35    # steepest failure-rate rise
E96_TAU_F:     int    = 36    # failure cliff
E96_LEAD_RISE: int    = 0     # lead to steepest rise (τrise − τα)
E96_LEAD_CLIFF: int   = 1     # lead to cliff         (τf − τα)

# E9.6 perfect-basin timing (5 configs, a=1.02, K∈{0.4,0.6,0.8,1.0,1.2})
E96_BASIN_MEAN_LEAD_RISE_U:  float = 1.15    # mean lead to rise, u-steering
E96_BASIN_MEAN_LEAD_RISE_XI: float = 1.025   # mean lead to rise, ξ-steering
E96_BASIN_MEAN_LEAD_CLIFF_U: float = 2.15    # mean lead to cliff, u-steering
E96_BASIN_MEAN_LEAD_CLIFF_XI:float = 2.025   # mean lead to cliff, ξ-steering
E96_BASIN_BEFORE_RISE_RATE:  float = 1.0     # before-or-at-rise rate (both channels)
E96_BASIN_BEFORE_CLIFF_RATE: float = 1.0     # before-cliff rate      (both channels)

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
