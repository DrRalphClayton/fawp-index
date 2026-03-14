"""
FAWP Alpha Index v2.1
Information-Control Exclusion Principle detector.
Ralph Clayton (2026) — DOI: https://doi.org/10.5281/zenodo.18673949
"""

import numpy as np
from fawp_index.constants import (
    ETA_PRED_CORRECTED, EPSILON_STEERING_CORRECTED,
    PERSISTENCE_WINDOW_M, KAPPA_RESONANCE,
    BETA_NULL_QUANTILE, DELTA_LOG_SMOOTH, TAU_MIN,
    FLAGSHIP_DELTA_PRED,
)
from dataclasses import dataclass
from typing import Optional, List
from .estimators import null_corrected_mi, mi_from_arrays


@dataclass
class FAWPResult:
    """Results from a FAWP Alpha Index computation."""
    tau: np.ndarray                    # delay grid
    pred_mi_raw: np.ndarray           # raw predictive MI
    steer_mi_raw: np.ndarray          # raw steering MI
    pred_mi_corrected: np.ndarray     # null-corrected predictive MI
    steer_mi_corrected: np.ndarray    # null-corrected steering MI
    pred_null_floor: np.ndarray       # conservative null floor (pred)
    steer_null_floor: np.ndarray      # conservative null floor (steer)
    alpha_index: np.ndarray           # FAWP Alpha Index v2.1
    in_fawp: np.ndarray               # boolean: tau in FAWP regime
    tau_h: Optional[int]              # empirical agency horizon
    peak_alpha: float                 # max alpha index value
    peak_tau: Optional[int]           # tau at peak alpha

    def summary(self) -> str:
        lines = [
            "=" * 50,
            "FAWP Alpha Index v2.1 — Results Summary",
            "=" * 50,
            f"Agency Horizon (tau_h):  {self.tau_h}",
            f"Peak Alpha Index:        {self.peak_alpha:.4f}",
            f"Peak Alpha at tau:       {self.peak_tau}",
            f"FAWP regime detected:   {'YES' if self.in_fawp.any() else 'NO'}",
            f"FAWP tau range:          {list(self.tau[self.in_fawp]) if self.in_fawp.any() else 'None'}",
            "=" * 50,
        ]
        return "\n".join(lines)


class FAWPAlphaIndex:
    """
    FAWP Alpha Index v2.1 (Calibrated).

    Detects the Information-Control Exclusion Principle:
    the regime where predictive coupling persists while
    steering coupling collapses below detectability.

    Based on: Clayton, R. (2026). Future Access Without Presence.
    DOI: https://doi.org/10.5281/zenodo.18673949

    Parameters
    ----------
    eta : float
        Minimum null-corrected predictive MI to count as "present" (bits).
    epsilon : float
        Maximum null-corrected steering MI to count as "collapsed" (bits).
    m_persist : int
        Persistence window — FAWP requires m+1 consecutive tau to qualify.
    kappa : float
        Resonance weight in alpha index (log-slope term).
    beta : float
        Null quantile for conservative floor (0.99 recommended).
    n_null : int
        Number of null shuffles/shifts for floor estimation.
    delta : float
        Small constant for log-slope stability (avoids log(0)).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        eta: float = ETA_PRED_CORRECTED,
        epsilon: float = EPSILON_STEERING_CORRECTED,
        m_persist: int = PERSISTENCE_WINDOW_M,
        kappa: float = KAPPA_RESONANCE,
        beta: float = BETA_NULL_QUANTILE,
        n_null: int = 200,
        delta: float = DELTA_LOG_SMOOTH,
        seed: int = 42,
    ):
        self.eta = eta
        self.epsilon = epsilon
        self.m_persist = m_persist
        self.kappa = kappa
        self.beta = beta
        self.n_null = n_null
        self.delta = delta
        self.rng = np.random.default_rng(seed)

    def _robust_stability(self, corrected_pred: np.ndarray) -> np.ndarray:
        """Windowed min over last m+1 points (multi-step persistence)."""
        m = self.m_persist
        n = len(corrected_pred)
        sm = np.zeros(n)
        for i in range(n):
            window = corrected_pred[max(0, i - m): i + 1]
            sm[i] = float(np.min(window))
        return sm

    def _log_slope_resonance(self, sm: np.ndarray) -> np.ndarray:
        """Scale-invariant resonance: log-slope of robust predictability."""
        n = len(sm)
        rlog = np.zeros(n)
        for i in range(1, n):
            diff = np.log(self.delta + sm[i]) - np.log(self.delta + sm[i - 1])
            rlog[i] = max(0.0, diff)
        return rlog

    def compute(
        self,
        pred_series: np.ndarray,
        future_series: np.ndarray,
        action_series: np.ndarray,
        obs_series: np.ndarray,
        tau_grid: Optional[List[int]] = None,
        delta_pred: int = FLAGSHIP_DELTA_PRED,
        verbose: bool = False,
    ) -> FAWPResult:
        """
        Compute FAWP Alpha Index over a delay grid.

        Parameters
        ----------
        pred_series : array (T,)
            Monitoring stream D_t (predictor variable, e.g. current state).
        future_series : array (T,)
            Future target X_{t+delta} (pre-aligned to pred_series).
        action_series : array (T,)
            Action stream A_t.
        obs_series : array (T,)
            Delayed observation stream O_{t+tau+1} (pre-aligned per tau outside).
        tau_grid : list of int, optional
            Delay values to sweep. Defaults to [1..15].
        delta_pred : int
            Forecast horizon delta (used for labeling only).
        verbose : bool
            Print progress.
        """
        if tau_grid is None:
            tau_grid = list(range(1, 16))

        tau_arr = np.array(tau_grid, dtype=int)
        n_tau = len(tau_arr)

        pred_raw = np.zeros(n_tau)
        steer_raw = np.zeros(n_tau)
        pred_corr = np.zeros(n_tau)
        steer_corr = np.zeros(n_tau)
        pred_floor = np.zeros(n_tau)
        steer_floor = np.zeros(n_tau)

        pred_series = np.asarray(pred_series, dtype=float)
        future_series = np.asarray(future_series, dtype=float)
        action_series = np.asarray(action_series, dtype=float)
        obs_series = np.asarray(obs_series, dtype=float)

        for i, tau in enumerate(tau_arr):
            if verbose:
                print(f"  tau={tau} ...", end=" ", flush=True)

            # Predictive MI: D_t vs X_{t+delta} (alignment done externally)
            pred_raw[i] = mi_from_arrays(pred_series, future_series)
            pc, pf = null_corrected_mi(
                pred_series, future_series,
                n_null=self.n_null, beta=self.beta, rng=self.rng
            )
            pred_corr[i] = pc
            pred_floor[i] = pf

            # Steering MI: A_t vs O_{t+tau+1}
            # Align: shift obs by tau+1 relative to action
            shift = tau + 1
            if shift < len(action_series):
                a_aligned = action_series[:-shift]
                o_aligned = obs_series[shift:]
                min_len = min(len(a_aligned), len(o_aligned))
                a_aligned = a_aligned[:min_len]
                o_aligned = o_aligned[:min_len]
            else:
                a_aligned = np.array([])
                o_aligned = np.array([])

            steer_raw[i] = mi_from_arrays(a_aligned, o_aligned)
            sc, sf = null_corrected_mi(
                a_aligned, o_aligned,
                n_null=self.n_null, beta=self.beta, rng=self.rng
            )
            steer_corr[i] = sc
            steer_floor[i] = sf

            if verbose:
                print(f"pred={pred_corr[i]:.4f} steer={steer_corr[i]:.4f}")

        # Robust stability (multi-step persistence on corrected pred)
        sm = self._robust_stability(pred_corr)

        # Log-slope resonance
        rlog = self._log_slope_resonance(sm)

        # Gate: FAWP requires tau>=1, pred above eta, steer below epsilon
        gate = (tau_arr >= 1) & (sm > self.eta) & (steer_corr <= self.epsilon)

        # Alpha Index v2.1
        leverage_gap = np.maximum(0.0, sm - steer_corr)
        alpha = gate.astype(float) * leverage_gap * (1.0 + self.kappa * rlog)

        # Agency horizon: first tau>=1 where raw steer <= epsilon
        tau_h = None
        for i, tau in enumerate(tau_arr):
            if tau >= 1 and steer_raw[i] <= self.epsilon:
                tau_h = int(tau)
                break

        peak_idx = int(np.argmax(alpha))
        peak_alpha = float(alpha[peak_idx])
        peak_tau = int(tau_arr[peak_idx]) if peak_alpha > 0 else None

        return FAWPResult(
            tau=tau_arr,
            pred_mi_raw=pred_raw,
            steer_mi_raw=steer_raw,
            pred_mi_corrected=pred_corr,
            steer_mi_corrected=steer_corr,
            pred_null_floor=pred_floor,
            steer_null_floor=steer_floor,
            alpha_index=alpha,
            in_fawp=gate,
            tau_h=tau_h,
            peak_alpha=peak_alpha,
            peak_tau=peak_tau,
        )
