"""
fawp_index.quant.momentum — Momentum Decay Detector

Detects when a momentum strategy enters FAWP:
  - Alpha signal (momentum factor → future return) is still present
  - But execution edge (trade → price impact) has collapsed
  → Strategy "knows" what will happen but can no longer profit from it

This is the crowded-trade signature: the factor works in backtests
but live execution can no longer capture it. FAWP quantifies exactly
when and how much of the edge is irrecoverable.

Ralph Clayton (2026) — doi:10.5281/zenodo.18673949
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class MomentumDecayResult:
    """Result from MomentumDecayDetector."""
    tau_grid: np.ndarray
    pred_mi: np.ndarray        # I(momentum_signal ; future_return)
    exec_mi: np.ndarray        # I(trade_size ; price_impact)
    alpha_signal_bits: float   # peak predictive MI
    exec_edge_bits: float      # peak execution MI
    leverage_gap: float        # signal - execution at peak signal tau
    in_fawp: bool              # overall FAWP flag
    decay_tau: Optional[int]   # tau where execution edge first collapses
    signal_tau: Optional[int]  # tau where signal peaks

    def summary(self) -> str:
        lines = [
            "=" * 55,
            "Momentum Decay — FAWP Analysis",
            "=" * 55,
            f"Alpha signal strength:  {self.alpha_signal_bits:.4f} bits",
            f"Execution edge:         {self.exec_edge_bits:.4f} bits",
            f"Leverage gap:           {self.leverage_gap:.4f} bits",
            f"Signal peaks at:        τ={self.signal_tau}",
            f"Execution decays at:    τ={self.decay_tau}",
            f"FAWP (crowded trade):   {'YES ⚠️' if self.in_fawp else 'No'}",
            "=" * 55,
        ]
        if self.in_fawp:
            lines.insert(-1,
                "→ Alpha survives but execution edge is gone."
            )
            lines.insert(-1,
                "  Strategy is predicting but cannot capture the edge."
            )
        return "\n".join(lines)


class MomentumDecayDetector:
    """
    Detects the FAWP signature in momentum strategies.

    Compares:
      - Predictive MI: how well the momentum signal predicts future returns
      - Execution MI:  how well trade size influences price (market impact)

    FAWP = signal persists but execution edge has collapsed = crowded trade.

    Parameters
    ----------
    tau_grid : list of int
        Delay values to sweep (default 1..15).
    delta : int
        Return forecast horizon in bars (default 21).
    eta : float
        Minimum signal MI to flag FAWP (default 1e-3).
    epsilon : float
        Maximum execution MI for FAWP (default 0.05).
    n_null : int
        Null samples for MI correction.

    Example
    -------
        import numpy as np
        from fawp_index.quant.momentum import MomentumDecayDetector

        # Simulate: signal predicts returns, but trades have no impact
        n = 3000
        signal = np.random.randn(n)
        future_ret = 0.3 * signal[:-21] + np.random.randn(n - 21) * 0.1
        trade_size = np.random.randn(n)
        price_impact = np.random.randn(n) * 0.001  # near zero = crowded

        detector = MomentumDecayDetector()
        result = detector.detect(signal[:-21], future_ret, trade_size[:-21], price_impact[:-21])
        print(result.summary())
    """

    def __init__(
        self,
        tau_grid: Optional[List[int]] = None,
        delta: int = 21,
        eta: float = 1e-3,
        epsilon: float = 0.05,
        n_null: int = 100,
        seed: int = 42,
    ):
        self.tau_grid = tau_grid or list(range(1, 16))
        self.delta = delta
        self.eta = eta
        self.epsilon = epsilon
        self.n_null = n_null
        self.seed = seed

    def detect(
        self,
        signal: np.ndarray,
        future_returns: np.ndarray,
        trade_size: np.ndarray,
        price_impact: np.ndarray,
    ) -> MomentumDecayResult:
        """
        Parameters
        ----------
        signal : array
            Momentum signal (e.g. 12-1 month return, factor score).
        future_returns : array
            Forward returns aligned to signal.
        trade_size : array
            Trade size / order flow.
        price_impact : array
            Resulting price impact / slippage.

        Returns
        -------
        MomentumDecayResult
        """
        from fawp_index.core.alpha_index import FAWPAlphaIndex

        n = min(len(signal), len(future_returns), len(trade_size), len(price_impact))
        signal = np.asarray(signal[:n], dtype=float)
        future_returns = np.asarray(future_returns[:n], dtype=float)
        trade_size = np.asarray(trade_size[:n], dtype=float)
        price_impact = np.asarray(price_impact[:n], dtype=float)

        detector = FAWPAlphaIndex(
            eta=self.eta,
            epsilon=self.epsilon,
            n_null=self.n_null,
            seed=self.seed,
        )

        result = detector.compute(
            pred_series=signal,
            future_series=future_returns,
            action_series=trade_size,
            obs_series=price_impact,
            tau_grid=self.tau_grid,
        )

        # Find execution decay point
        decay_tau = None
        for i, tau in enumerate(result.tau):
            if result.steer_mi_raw[i] <= self.epsilon:
                decay_tau = int(tau)
                break

        # Peak signal tau
        signal_tau = None
        if result.peak_tau is not None:
            signal_tau = int(result.peak_tau)

        # Leverage gap at signal peak
        if signal_tau is not None:
            idx = list(result.tau).index(signal_tau)
            gap = float(result.pred_mi_raw[idx] - result.steer_mi_raw[idx])
        else:
            gap = 0.0

        return MomentumDecayResult(
            tau_grid=result.tau,
            pred_mi=result.pred_mi_raw,
            exec_mi=result.steer_mi_raw,
            alpha_signal_bits=float(result.peak_alpha) if result.peak_alpha else 0.0,
            exec_edge_bits=float(result.steer_mi_raw.max()),
            leverage_gap=gap,
            in_fawp=bool(result.in_fawp.any()),
            decay_tau=decay_tau,
            signal_tau=signal_tau,
        )
