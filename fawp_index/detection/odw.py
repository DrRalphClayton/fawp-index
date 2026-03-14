"""
fawp_index.detection.odw — Operational Detection Window (ODW) Detector

Implements the E9 detection methodology: given null-corrected MI curves
and a failure-rate series, finds the Operational Detection Window — the
pre-cliff interval where prediction remains alive after steering has
already collapsed.

Key concepts (from Clayton 2026, Experiment 9):
  τ⁺ₕ  — post-zero agency horizon: first τ ≥ 1 where steering ≤ ε
  τf   — functional failure cliff: first τ where failure rate ≥ cliff threshold
  ODW  — Operational Detection Window: persistent pre-cliff FAWP interval

Ralph Clayton (2026) — doi:10.5281/zenodo.18663547
"""

import numpy as np
from fawp_index.constants import (
    EPSILON_STEERING_RAW, PERSISTENCE_RULE_M, PERSISTENCE_RULE_N,
)
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ODWResult:
    """
    Output of ODWDetector.detect().

    Attributes
    ----------
    tau_h_plus : int or None
        Post-zero agency horizon τ⁺ₕ (first τ≥1 where steering ≤ ε).
    tau_f : int or None
        Functional failure cliff (first τ where fail_rate ≥ cliff_threshold).
    odw_start : int or None
        ODW start τ.
    odw_end : int or None
        ODW end τ.
    odw_size : int
        Number of τ steps in ODW (0 if not found).
    fawp_found : bool
        True if a non-empty ODW was found before the cliff.
    mean_lead_to_cliff : float or None
        Mean τ distance from ODW to cliff.
    peak_gap_tau : int or None
        τ at maximum corrected leverage gap.
    peak_gap_bits : float
        Leverage gap at peak_gap_tau.
    """
    tau_h_plus: Optional[int]
    tau_f: Optional[int]
    odw_start: Optional[int]
    odw_end: Optional[int]
    odw_size: int
    fawp_found: bool
    mean_lead_to_cliff: Optional[float]
    peak_gap_tau: Optional[int]
    peak_gap_bits: float

    def summary(self) -> str:
        lines = [
            "=" * 55,
            "  Operational Detection Window (ODW) — E9 Method",
            "=" * 55,
            f"FAWP found         : {'YES' if self.fawp_found else 'NO'}",
            f"Post-zero horizon  : τ⁺ₕ = {self.tau_h_plus}",
            f"Failure cliff      : τf  = {self.tau_f}",
        ]
        if self.fawp_found:
            lines += [
                f"ODW                : τ = {self.odw_start} — {self.odw_end}  ({self.odw_size} steps)",
                f"Mean lead to cliff : {self.mean_lead_to_cliff:.1f} steps" if self.mean_lead_to_cliff else "",
                f"Peak leverage gap  : {self.peak_gap_bits:.4f} bits  at τ = {self.peak_gap_tau}",
            ]
        else:
            lines.append("No pre-cliff FAWP window detected.")
        lines.append("=" * 55)
        return "\n".join(lines)


class ODWDetector:
    """
    Operational Detection Window detector — implements the E9 methodology.

    Given null-corrected MI curves and a failure-rate series, finds the
    pre-cliff interval in which prediction remains above threshold while
    steering has already collapsed (the 'leverage gap' regime).

    Parameters
    ----------
    epsilon : float
        Steering near-null threshold (in corrected bits). Default 0.01.
        Use 1e-4 when paired with FAWPAlphaIndexV2 (strict null correction).
        Use 0.01 to match the E9.2 script default.
    fail_rate_cliff : float
        Failure-rate threshold for defining τf. Default 0.99.
    persistence_m : int
        Minimum number of True values in a window of size persistence_n
        for a candidate interval to count. Default 3.
    persistence_n : int
        Window size for persistence rule. Default 4.
    min_tau : int
        Minimum τ to consider for τ⁺ₕ (excludes τ=0). Default 1.

    Example
    -------
        import pandas as pd
        from fawp_index.data import E9_2_AGGREGATE_CURVES
        from fawp_index.detection.odw import ODWDetector

        df = pd.read_csv(E9_2_AGGREGATE_CURVES)
        det = ODWDetector(epsilon=0.01)
        result = det.detect(
            tau          = df['tau'].values,
            pred_corr    = df['pred_strat_corr'].values,
            steer_corr   = df['steer_u_corr'].values,
            fail_rate    = df['fail_rate'].values,
        )
        print(result.summary())
    """

    def __init__(
        self,
        epsilon: float = EPSILON_STEERING_RAW,
        fail_rate_cliff: float = 0.99,
        persistence_m: int = PERSISTENCE_RULE_M,
        persistence_n: int = PERSISTENCE_RULE_N,
        min_tau: int = 1,
    ):
        self.epsilon = epsilon
        self.fail_rate_cliff = fail_rate_cliff
        self.persistence_m = persistence_m
        self.persistence_n = persistence_n
        self.min_tau = min_tau

    def detect(
        self,
        tau: np.ndarray,
        pred_corr: np.ndarray,
        steer_corr: np.ndarray,
        fail_rate: np.ndarray,
    ) -> ODWResult:
        """
        Detect the Operational Detection Window.

        Parameters
        ----------
        tau : array of int
        pred_corr : array of float — null-corrected predictive MI
        steer_corr : array of float — null-corrected steering MI
        fail_rate : array of float — failure rate in [0,1] per τ

        Returns
        -------
        ODWResult
        """
        tau   = np.asarray(tau, dtype=int)
        pred  = np.asarray(pred_corr, dtype=float)
        steer = np.asarray(steer_corr, dtype=float)
        fail  = np.asarray(fail_rate, dtype=float)

        # τ⁺ₕ — first τ≥min_tau where steer ≤ ε
        cond_tau_h = (tau >= self.min_tau) & (steer <= self.epsilon)
        tau_h_plus = self._first_tau_where(tau, cond_tau_h)

        # τf — first τ where fail_rate ≥ cliff
        tau_f = self._first_tau_where(tau, fail >= self.fail_rate_cliff)

        # Base FAWP condition: pred > ε, steer ≤ ε
        base = (pred > self.epsilon) & (steer <= self.epsilon)

        # Apply post-zero gating and pre-cliff gating
        if tau_h_plus is not None:
            base = base & (tau >= tau_h_plus)
        if tau_f is not None:
            base = base & (tau < tau_f)

        # Persistence filter
        mask = self._persistent_mask(base, self.persistence_m, self.persistence_n)

        # ODW: first contiguous block
        odw_start, odw_end = self._first_contiguous_range(tau, mask)
        odw_size = int(odw_end - odw_start + 1) if (odw_start is not None) else 0
        fawp_found = odw_start is not None

        # Mean lead to cliff
        mean_lead = None
        if fawp_found and tau_f is not None:
            odw_taus = tau[mask]
            mean_lead = float(np.mean(tau_f - odw_taus)) if len(odw_taus) else None

        # Peak leverage gap
        gap = pred - steer
        peak_gap_idx = int(np.argmax(gap))
        peak_gap_tau  = int(tau[peak_gap_idx])
        peak_gap_bits = float(gap[peak_gap_idx])

        return ODWResult(
            tau_h_plus=tau_h_plus,
            tau_f=tau_f,
            odw_start=odw_start,
            odw_end=odw_end,
            odw_size=odw_size,
            fawp_found=fawp_found,
            mean_lead_to_cliff=mean_lead,
            peak_gap_tau=peak_gap_tau,
            peak_gap_bits=peak_gap_bits,
        )

    @classmethod
    def from_e9_2_data(
        cls,
        steering: str = 'u',
        epsilon: float = EPSILON_STEERING_RAW,
        persistence_m: int = PERSISTENCE_RULE_M,
        persistence_n: int = PERSISTENCE_RULE_N,
    ) -> ODWResult:
        """
        Convenience: run ODW detection on bundled E9.2 aggregate curves.

        Parameters
        ----------
        steering : 'u' or 'xi'
        epsilon : float — steering null threshold (default 0.01 matches E9.2)
        persistence_m, persistence_n : persistence rule

        Example
        -------
            from fawp_index.detection.odw import ODWDetector

            # Reproduce E9.2 results exactly
            result = ODWDetector.from_e9_2_data(steering='u')
            print(result.summary())
            # → ODW: τ = 31 — 33, τf = 36
        """
        from fawp_index.data import E9_2_AGGREGATE_CURVES
        df = pd.read_csv(E9_2_AGGREGATE_CURVES).sort_values('tau').reset_index(drop=True)
        steer_col = f'steer_{steering}_corr'
        det = cls(epsilon=epsilon, persistence_m=persistence_m, persistence_n=persistence_n)
        return det.detect(
            tau        = df['tau'].values,
            pred_corr  = df['pred_strat_corr'].values,
            steer_corr = df[steer_col].values,
            fail_rate  = df['fail_rate'].values,
        )

    # ── internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _first_tau_where(tau: np.ndarray, cond: np.ndarray) -> Optional[int]:
        idx = np.where(cond)[0]
        return int(tau[idx[0]]) if idx.size > 0 else None

    @staticmethod
    def _persistent_mask(mask: np.ndarray, m: int, n: int) -> np.ndarray:
        mask = np.asarray(mask, dtype=bool)
        out = np.zeros_like(mask, dtype=bool)
        if n <= 1:
            return mask.copy()
        for i in range(0, len(mask) - n + 1):
            window = mask[i:i + n]
            if int(window.sum()) >= m:
                out[i:i + n] |= window
        return out

    @staticmethod
    def _first_contiguous_range(
        tau: np.ndarray, mask: np.ndarray
    ) -> Tuple[Optional[int], Optional[int]]:
        idx = np.where(mask)[0]
        if idx.size == 0:
            return None, None
        start = idx[0]
        end = start
        for j in idx[1:]:
            if j == end + 1:
                end = j
            else:
                break
        return int(tau[start]), int(tau[end])
