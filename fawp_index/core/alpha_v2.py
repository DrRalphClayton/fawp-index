"""
fawp_index.core.alpha_v2 — Upgraded FAWP Alpha Index v2.1

Implements the full α₂(τ) formula from:
  Clayton (2026), "Future Access Without Presence:
  The Information–Control Exclusion Principle in Unstable Dynamics"
  doi:10.5281/zenodo.18673949

Formula:
  S_m(τ)   = min_{k=0,...,m}  Ĩ_pred(τ-k)           [robust stability window]
  R_log(τ) = max(0, log(δ+S_m(τ)) - log(δ+S_m(τ-1))) [scale-invariant resonance]
  g(τ)     = [τ≥1] · [S_m(τ) > η] · [Ĩ_steer(τ) ≤ ε]  [hard gate]
  α₂(τ)   = g(τ) · (S_m(τ) - Ĩ_steer(τ)) · (1 + κ·R_log(τ))

Where Ĩ_pred, Ĩ_steer are null-corrected MI curves (shuffle + shift, q_β=0.99).

Default calibration from independent record-chain validation:
  β = 0.99      — null quantile
  m = 5         — persistence window (use 3 for noisier regimes)
  η = 1e-4 bits — pred detectability buffer after floor subtraction
  ε = 1e-4 bits — steer near-null threshold after floor subtraction
  κ = 1.0       — resonance scaling (not specified in paper; 1.0 is neutral)
  δ = 1e-6      — log-domain regularizer
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AlphaV2Result:
    """
    Output of FAWPAlphaIndexV2.compute().

    Attributes
    ----------
    tau_grid : array[int]
        Latency values.
    pred_mi_corr : array[float]
        Null-corrected predictive MI, Ĩ_pred(τ).
    steer_mi_corr : array[float]
        Null-corrected steering MI, Ĩ_steer(τ).
    S_m : array[float]
        Robust stability window S_m(τ).
    R_log : array[float]
        Log-slope resonance R_log(τ).
    gate : array[bool]
        Hard gate g(τ).
    alpha2 : array[float]
        Upgraded alpha index α₂(τ).
    peak_alpha2 : float
        Maximum α₂(τ).
    peak_tau2 : int or None
        τ at which α₂ peaks.
    fawp_detected : bool
        True if any τ passes the gate.
    odw_start : int or None
        Start of first contiguous FAWP window.
    odw_end : int or None
        End of first contiguous FAWP window.
    params : dict
        Parameters used: m, eta, epsilon, kappa, delta.
    """
    tau_grid: np.ndarray
    pred_mi_corr: np.ndarray
    steer_mi_corr: np.ndarray
    S_m: np.ndarray
    R_log: np.ndarray
    gate: np.ndarray
    alpha2: np.ndarray
    peak_alpha2: float
    peak_tau2: Optional[int]
    fawp_detected: bool
    odw_start: Optional[int]
    odw_end: Optional[int]
    params: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "=" * 58,
            "  α₂(τ) v2.1 Result",
            "=" * 58,
            f"FAWP detected      : {'YES' if self.fawp_detected else 'NO'}",
            f"ODW                : "
            + (f"τ = {self.odw_start} — {self.odw_end}" if self.odw_start is not None else "none"),
            f"Peak α₂(τ)         : {self.peak_alpha2:.4f}"
            + (f"  at τ = {self.peak_tau2}" if self.peak_tau2 is not None else ""),
            f"Peak Ĩ_pred        : {self.pred_mi_corr.max():.4f} bits",
            f"Min Ĩ_steer (gated): "
            + f"{self.steer_mi_corr[self.gate].min():.6f} bits" if self.gate.any() else "  (no gated taus)",
            "",
            f"Params: m={self.params.get('m')}, η={self.params.get('eta')}, "
            f"ε={self.params.get('epsilon')}, κ={self.params.get('kappa')}",
            "=" * 58,
        ]
        return "\n".join(lines)

    def plot(self, show: bool = True, save_path: Optional[str] = None):
        """Plot α₂(τ) curves with gate regions highlighted."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("pip install fawp-index[plot]")

        tau = self.tau_grid
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

        # Top: MI curves
        ax1.plot(tau, self.pred_mi_corr, lw=2.2, label="Ĩ_pred (corr.)")
        ax1.plot(tau, self.steer_mi_corr, lw=2.0, ls='--', label="Ĩ_steer (corr.)")
        ax1.plot(tau, self.S_m, lw=1.6, ls=':', alpha=0.8, label=f"S_m (m={self.params.get('m',5)})")
        ax1.axhline(self.params.get('eta', 1e-4), ls='--', lw=1.0,
                    color='gray', alpha=0.6, label='η threshold')
        if self.odw_start is not None:
            ax1.axvspan(self.odw_start, self.odw_end, alpha=0.12,
                        color='green', label='ODW')
        ax1.set_ylabel("MI (bits)"); ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
        ax1.set_title("Upgraded FAWP Alpha Index v2.1 — null-corrected MI curves")

        # Bottom: α₂(τ)
        ax2.plot(tau, self.alpha2, lw=2.5, color='crimson', label='α₂(τ)')
        ax2.plot(tau, self.R_log, lw=1.4, ls=':', alpha=0.7,
                 color='orange', label='R_log (resonance)')
        ax2.fill_between(tau, self.alpha2, alpha=0.15, color='crimson')
        if self.odw_start is not None:
            ax2.axvspan(self.odw_start, self.odw_end, alpha=0.12,
                        color='green', label='ODW')
        ax2.axhline(0, lw=1.0, color='black')
        ax2.set_xlabel("Latency τ"); ax2.set_ylabel("α₂(τ)")
        ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
        ax2.set_title("α₂(τ) — upgraded FAWP index with log-slope resonance")

        fig.text(0.99, 0.01, 'fawp-index | Clayton (2026) doi:10.5281/zenodo.18673949',
                 ha='right', va='bottom', fontsize=7, color='gray', style='italic')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        return fig, (ax1, ax2)


class FAWPAlphaIndexV2:
    """
    Upgraded FAWP Alpha Index v2.1.

    Implements the full α₂(τ) formula from Clayton (2026),
    "Future Access Without Presence: The Information–Control Exclusion
    Principle in Unstable Dynamics" — doi:10.5281/zenodo.18673949

    Operates on null-corrected MI curves. Use with the pre-computed
    corrected curves from the E9.2 dataset or from your own MI pipeline.

    Parameters
    ----------
    m : int
        Robust stability window width. Default 5.
        Use 3 for noisier or shorter-window regimes.
    eta : float
        Predictive detectability buffer after null subtraction.
        Default 1e-4 bits (recommended when β=0.99 null correction applied).
    epsilon : float
        Steering near-null threshold after null subtraction.
        Default 1e-4 bits.
    kappa : float
        Resonance scaling. Default 1.0 (neutral).
        Increase to amplify log-slope resonance in α₂.
    delta : float
        Log-domain regularizer. Default 1e-6.

    Example — from bundled E9.2 data
    ---------------------------------
        import pandas as pd
        from fawp_index.data import E9_2_AGGREGATE_CURVES
        from fawp_index.core.alpha_v2 import FAWPAlphaIndexV2

        df = pd.read_csv(E9_2_AGGREGATE_CURVES)
        idx = FAWPAlphaIndexV2(m=5, epsilon=1e-4)
        result = idx.compute(
            tau_grid       = df['tau'].values,
            pred_mi_corr   = df['pred_strat_corr'].values,
            steer_mi_corr  = df['steer_u_corr'].values,
        )
        print(result.summary())

    Example — matching E9.2 paper parameters (ε=0.01)
    -------------------------------------------------
        idx = FAWPAlphaIndexV2(m=3, epsilon=0.01)
        result = idx.compute(tau_grid, pred_corr, steer_corr)
    """

    def __init__(
        self,
        m: int = 5,
        eta: float = 1e-4,
        epsilon: float = 1e-4,
        kappa: float = 1.0,
        delta: float = 1e-6,
    ):
        self.m = m
        self.eta = eta
        self.epsilon = epsilon
        self.kappa = kappa
        self.delta = delta

    def compute(
        self,
        tau_grid: np.ndarray,
        pred_mi_corr: np.ndarray,
        steer_mi_corr: np.ndarray,
    ) -> AlphaV2Result:
        """
        Compute α₂(τ) from null-corrected MI curves.

        Parameters
        ----------
        tau_grid : array-like of int
            Latency values τ.
        pred_mi_corr : array-like of float
            Null-corrected predictive MI, Ĩ_pred(τ).
            Must be ≥ 0 (floor already subtracted).
        steer_mi_corr : array-like of float
            Null-corrected steering MI, Ĩ_steer(τ).
            Must be ≥ 0.

        Returns
        -------
        AlphaV2Result
        """
        tau   = np.asarray(tau_grid, dtype=int)
        pred  = np.clip(np.asarray(pred_mi_corr, dtype=float), 0.0, None)
        steer = np.clip(np.asarray(steer_mi_corr, dtype=float), 0.0, None)
        n = len(tau)

        # S_m(τ) = min_{k=0,...,m} Ĩ_pred(τ-k)
        S_m = np.empty(n)
        for i in range(n):
            lo = max(0, i - self.m)
            S_m[i] = np.min(pred[lo:i + 1])

        # R_log(τ) = max(0, log(δ+S_m(τ)) - log(δ+S_m(τ-1)))
        R_log = np.zeros(n)
        for i in range(1, n):
            R_log[i] = max(
                0.0,
                np.log(self.delta + S_m[i]) - np.log(self.delta + S_m[i - 1]),
            )

        # Gate g(τ): τ≥1, S_m > η, Ĩ_steer ≤ ε
        gate = (tau >= 1) & (S_m > self.eta) & (steer <= self.epsilon)

        # α₂(τ)
        alpha2 = gate.astype(float) * (S_m - steer) * (1.0 + self.kappa * R_log)

        # Summary quantities
        peak_idx = int(np.argmax(alpha2)) if alpha2.max() > 0 else None
        peak_alpha2 = float(alpha2[peak_idx]) if peak_idx is not None else 0.0
        peak_tau2   = int(tau[peak_idx]) if peak_idx is not None else None
        fawp_detected = bool(gate.any())

        # First contiguous ODW
        odw_start, odw_end = self._first_contiguous(tau, gate)

        return AlphaV2Result(
            tau_grid=tau,
            pred_mi_corr=pred,
            steer_mi_corr=steer,
            S_m=S_m,
            R_log=R_log,
            gate=gate,
            alpha2=alpha2,
            peak_alpha2=peak_alpha2,
            peak_tau2=peak_tau2,
            fawp_detected=fawp_detected,
            odw_start=odw_start,
            odw_end=odw_end,
            params={
                'm': self.m,
                'eta': self.eta,
                'epsilon': self.epsilon,
                'kappa': self.kappa,
                'delta': self.delta,
            },
        )

    @classmethod
    def from_e9_2_data(
        cls,
        steering: str = 'u',
        m: int = 5,
        epsilon: float = 1e-4,
        kappa: float = 1.0,
    ) -> AlphaV2Result:
        """
        Convenience: compute α₂(τ) directly from bundled E9.2 aggregate curves.

        Parameters
        ----------
        steering : 'u' or 'xi'
            Which steering channel to use. Default 'u'.
        m, epsilon, kappa : see FAWPAlphaIndexV2 params.

        Returns
        -------
        AlphaV2Result

        Example
        -------
            from fawp_index.core.alpha_v2 import FAWPAlphaIndexV2

            result = FAWPAlphaIndexV2.from_e9_2_data(steering='u')
            print(result.summary())

            result_xi = FAWPAlphaIndexV2.from_e9_2_data(steering='xi', epsilon=0.01)
            print(result_xi.summary())
        """
        import pandas as pd
        from fawp_index.data import E9_2_AGGREGATE_CURVES

        df = pd.read_csv(E9_2_AGGREGATE_CURVES).sort_values('tau').reset_index(drop=True)
        steer_col = f'steer_{steering}_corr'
        if steer_col not in df.columns:
            raise ValueError(f"steering must be 'u' or 'xi'. Got: {steering!r}")

        idx = cls(m=m, epsilon=epsilon, kappa=kappa)
        return idx.compute(
            tau_grid      = df['tau'].values,
            pred_mi_corr  = df['pred_strat_corr'].values,
            steer_mi_corr = df[steer_col].values,
        )

    @staticmethod
    def _first_contiguous(tau, mask):
        """Return (start, end) of first contiguous True block."""
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
