"""
fawp_index.constants — Calibration anchors from published research.

All values are derived directly from the published papers:

  VTM   — doi:10.5281/zenodo.18634216
  E1–E7 — doi:10.5281/zenodo.18663547  (Agency Horizon / FORECASTING)
  E8    — doi:10.5281/zenodo.18673949  (FAWP confirmation suite)
  E9    — SPHERE_15 confirmation suite (doi:10.5281/zenodo.18693949 pending)
  E8v2  — SPHERE_16 recalibration (March 2026) — η=0, peak=2.233669 bits
  E9.7  — Comparative timing sweep (March 2026) — gap2 best predictor, α lags

Do NOT change these without a corresponding paper update.
Use them as defaults throughout the package so every module
stays in sync with the research.

Ralph Clayton (2026)
"""

# ── Null-correction ────────────────────────────────────────────────────────
# Conservative null-quantile level (β). Used for both shuffle and shift controls.
# Source: E9 standard config, E9 standard config, SPHERE E9.4 portability sweep.
BETA_NULL_QUANTILE: float = 0.99

# Number of null permutations for floor estimation (shuffle + shift each).
# 200 balances accuracy vs speed; use 50 for fast mode.
N_NULL_DEFAULT: int = 200
N_NULL_FAST: int = 0        # 0 = skip null correction entirely (raw MI)

# ── Detectability thresholds ───────────────────────────────────────────────
# Post-null-correction steering near-null criterion.
# Source: E8 flagship, calibration note: "ε ~ 10⁻⁴ after floor subtraction"
EPSILON_STEERING_CORRECTED: float = 1e-4

# Raw steering detectability threshold (pre-correction, used by ODWDetector).
# Source: E8 flagship, E9 standard config: ε = 0.01 bits
EPSILON_STEERING_RAW: float = 0.01

# Predictive coupling floor (η) — post-null-correction.
# SPHERE_16 Eq. 8: after null-correction, η=0 is already strict.
ETA_PRED_CORRECTED: float = 0.0

# ── Persistence gating ─────────────────────────────────────────────────────
# Robust stability window width m for Sm(τ).
# Source: E8 paper, default m=5. Use m=3 for noisier / shorter regimes.
PERSISTENCE_WINDOW_M: int = 5
PERSISTENCE_WINDOW_M_NOISY: int = 3   # for noisier regimes (from paper note)

# m-of-n persistence rule for ODW gate.
# Source: E9.1 ablation, SPHERE E9 standard: (m, n) = (3, 4)
PERSISTENCE_RULE_M: int = 3
PERSISTENCE_RULE_N: int = 4

# ── Alpha Index v2.2 (SPHERE_16 calibrated) ─────────────────────────────────
# Log-slope regularizer δ (avoids log(0) in Rlog computation).
# Source: E8 paper
DELTA_LOG_SMOOTH: float = 1e-6

# Resonance weight κ.  κ=1.0 is neutral (paper does not specify; 1.0 preserves
# the leverage gap signal without amplification or suppression).
KAPPA_RESONANCE: float = 1.0

# Minimum tau for causal interpretation — τ=0 excluded as shared-state diagnostic.
# Source: FORECASTING, E8, E8 all enforce τ ≥ 1.
TAU_MIN: int = 1

# ── Flagship E8 parameters ─────────────────────────────────────────────────
# Source: FORECASTING Eq. 15–17, E8 table
FLAGSHIP_A: float = 1.02          # unstable scalar drift (a > 1 → unstable)
FLAGSHIP_K: float = 0.8           # controller gain
FLAGSHIP_DELTA_PRED: int = 20     # forecast horizon Δ
FLAGSHIP_N_TRIALS: int = 400      # trials per τ in E8
FLAGSHIP_X_FAIL: float = 500.0    # failure threshold |x| > x_fail
FLAGSHIP_SIGMA_PROC: float = 1.0  # process noise std σ_w
FLAGSHIP_U_MAX: float = 10.0      # action clip bound

# ── Null control bounds (E8, SPHERE_16 Eq. 19) ───────────────────────────
# Conservative upper bounds on null MI for τ ≥ 1 — justify high-β subtraction.
NULL_MAX_SHUFFLE_E8: float = 0.00216  # max shuffle null pred MI (τ≥1)
NULL_MAX_SHIFT_E8:   float = 0.00421  # max shift   null pred MI (τ≥1)

# ── Flagship E8 empirical anchors ─────────────────────────────────────────
# Source: E8 table in FAWP paper; E9 SPHERE confirmation
TAU_PLUS_H_FLAGSHIP: int = 4      # post-zero agency horizon τ⁺ₕ (E8)
TAU_F_FLAGSHIP: int = 35          # functional failure cliff (E8: fail ≥ 0.99)
PEAK_PRED_BITS: float = 2.233669  # peak corrected stratified prediction (E8 flagship, τ=9) — SPHERE_16
PRED_AT_CLIFF: float = 1.01       # prediction MI at cliff τf=35 (SPHERE_16 Eq. 5: ~1.01 bits)

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


# ── E9.7 comparative timing sweep (SPHERE_17 / new) ──────────────────────
# Source: e9_7_out data — multi-seed sweep comparing alpha, alpha2, gap2 timing
# 4244 runs total (2122 per channel × 2 channels), seeds 51500–51621+
E97_N_RUNS: int        = 4244    # total runs across both channels
E97_N_RUNS_PER_CH: int = 2122    # runs per channel
E97_MEAN_TAU_F: float  = 52.5608 # mean failure cliff (broader than E9.2 fixed 36)
E97_MEAN_TAU_RISE: float = 28.297 # mean steepest rise τ

# ODW localisation
E97_MEAN_ODW_START_U:  float = 31.927  # mean ODW start, u-channel
E97_MEAN_ODW_START_XI: float = 31.327  # mean ODW start, ξ-channel
E97_MEAN_ODW_END_U:    float = 34.119  # mean ODW end,   u-channel
E97_MEAN_ODW_END_XI:   float = 33.315  # mean ODW end,   ξ-channel

# Gap2 peak (raw leverage gap peak) — best ODW localizer
E97_MEAN_TAU_GAP2_PEAK_U:  float = 37.105  # mean τ of gap2 peak, u
E97_MEAN_TAU_GAP2_PEAK_XI: float = 37.255  # mean τ of gap2 peak, ξ
E97_MEAN_LEAD_GAP2_TO_CLIFF_U:  float = 0.7552  # gap2 leads cliff, u
E97_MEAN_LEAD_GAP2_TO_CLIFF_XI: float = 0.4664  # gap2 leads cliff, ξ
E97_MEAN_ABS_ERR_GAP2_VS_ODW_START: float = 2.108  # best localizer: ≈2 delay error

# Alpha2 (SPHERE_16 formula) timing
E97_MEAN_TAU_ALPHA2_NEAREST_U:  float = 38.626  # mean nearest α₂ peak, u
E97_MEAN_TAU_ALPHA2_NEAREST_XI: float = 38.795  # mean nearest α₂ peak, ξ
E97_MEAN_LEAD_ALPHA2_TO_CLIFF_U:  float = 0.144  # α₂ barely leads cliff, u
E97_MEAN_LEAD_ALPHA2_TO_CLIFF_XI: float = 0.368  # α₂ barely leads cliff, ξ
E97_MEAN_ABS_ERR_ALPHA2_VS_ODW_START: float = 9.139  # α₂ moderate ODW localization

# Alpha (baseline/older) timing — LAGS cliff; inferior to alpha2
E97_MEAN_LEAD_ALPHA_TO_CLIFF_U:  float = -2.004  # alpha LAGS cliff by ~2 delays, u
E97_MEAN_LEAD_ALPHA_TO_CLIFF_XI: float = -1.964  # alpha LAGS cliff by ~2 delays, ξ
E97_MEAN_ABS_ERR_ALPHA_VS_ODW_START: float = 18.161  # alpha: worst ODW localization

# E9.7 verdict: gap2 peak → best cliff predictor (leads by ~0.6 delays, err ~2.1)
#               alpha2    → adequate early-warning (barely leads, err ~9.1)
#               alpha     → inferior (lags cliff, err ~18.2) — confirms alpha2 upgrade

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



# ── Triple Horizon Framework — SPHERE_23 (March 29, 2026) ────────────────────
# Source: Clayton (2026) "The Triple Horizon Framework"
# doi: pending — Institute for Operational Accessibility Studies
# Experiment 11 suite: readout, steering, and functional horizons measured jointly.

# Calibrated threshold pair (SPHERE_23 Eq. 12)
# αA  = practical steering wall (operational steerability boundary)
# α²A = deeper residual-readable floor (existence ≠ access)
ALPHA_A: float       = 0.007297              # practical steering wall αA
ALPHA_A_SQ: float    = 5.325135447834e-5     # residual-readable floor α²A

# E11-1 baseline Triple Horizon benchmark
# Configuration: a=1.02, K=0.8, delayed steering + degradable readout chain
E11_TAU_ALPHA: int         = 10   # τα — practical steering wall crossing
E11_TAU_PLUS_H: int        = 12   # τ⁺ₕ — post-zero steering horizon
E11_TAU_F: int             = 29   # τf  — functional horizon (viability cliff)
E11_TAU_ALPHA2: int        = 32   # τα² — deep residual-readable crossing
E11_TAU_READOUT: int       = 38   # τreadout — final readout horizon
# Observed ordering: τα < τ⁺ₕ < τf < τα² < τreadout
E11_FAWP_WINDOW_1: tuple   = (12, 19)    # first persistence-gated FAWP window
E11_FAWP_WINDOW_2: tuple   = (21, 26)    # second persistence-gated FAWP window
E11_PEAK_GAP_TAU: int      = 13          # τ of peak corrected leverage gap
E11_DOMINANT_ORDER: str    = "τα < τ⁺ₕ < τf < τα² < τreadout"

# E11-2 portability sweep (36 configurations)
E112_N_CONFIGS: int         = 36
E112_DOMINANT_RATE: float   = 34/36      # ≈ 0.9444 — dominant ordering rate
E112_PEAK_GAP: float        = 0.428671   # strongest corrected leverage gap
E112_PEAK_GAP_TAU: int      = 12         # τ of E11-2 peak gap
E112_WIDEST_FAWP_WIDTH: int = 18         # widest FAWP window (total width)

# E11-3 forced reordering (stress sweep, 18 configs)
E113_PEAK_GAP: float        = 0.437214   # strongest corrected gap (steering-suppression block)
E113_PEAK_GAP_TAU: int      = 9          # τ of E11-3 peak gap
E113_WIDEST_FAWP_WIDTH: int = 14         # widest FAWP window in E11-3

# E11-5 cross-family validation (27 configs: linear, cubic, coupled 2D)
E115_N_CONFIGS: int         = 27
E115_DOMINANT_RATE: float   = 1.0        # 27/27 dominant ordering
E115_CUBIC_PEAK_GAP: float  = 0.4421     # peak gap in cubic nonlinear family
E115_COUPLED_FAWP_WIDTH: int= 19         # widest FAWP in coupled 2D family

# E11-4 recovery: early upstream intervention (τlens = 5) best gains
E114_BEST_DELTA_TAU_H: int      = 14     # max Δτ⁺ₕ from early intervention
E114_BEST_DELTA_READOUT: int    = 17     # max Δτreadout
E114_BEST_DELTA_TAU_F: int      = 6      # max Δτf
E114_BEST_DELTA_ALPHA2: int     = 11     # max Δτα²
E114_BEST_DELTA_FAWP_W: int     = 12     # max ΔFAWP window width


# ── LERI photon-record horizon ─────────────────────────────────────────────
# Source: LERI Eq. 11–12
LERI_HORIZON_METERS: float = 14005.0   # ≈ 14.005 km
LERI_HORIZON_SECONDS: float = 4.6683e-5  # ≈ 46.68 µs
