"""
fawp_index.quant.risk — Risk Parity / Vol Targeting Early Warning

FAWP as a leading indicator of strategy failure in risk parity
and volatility-targeting frameworks.

The insight: in these strategies, the "control" is the vol target
rebalance (scale up/down). FAWP fires when:
  - The vol forecast still predicts realized vol (pred MI high)
  - But the rebalance can no longer move portfolio risk (steer MI low)
  → Liquidity has dried up; the hedge is broken before you know it

Ralph Clayton (2026) — doi:10.5281/zenodo.18673949
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class RiskWarningResult:
    """Result from RiskParityWarning."""
    warning_active: bool
    pred_mi_bits: float       # vol forecast → realized vol MI
    steer_mi_bits: float      # rebalance action → portfolio risk MI
    leverage_gap_bits: float
    risk_horizon: Optional[int]
    message: str

    def summary(self) -> str:
        status = "⚠️  FAWP WARNING ACTIVE" if self.warning_active else "✓ No warning"
        lines = [
            "=" * 55,
            f"Risk Parity / Vol Target — FAWP Early Warning",
            "=" * 55,
            f"Status:               {status}",
            f"Vol forecast MI:      {self.pred_mi_bits:.4f} bits",
            f"Rebalance effect MI:  {self.steer_mi_bits:.4f} bits",
            f"Leverage gap:         {self.leverage_gap_bits:.4f} bits",
            f"Risk horizon:         τ={self.risk_horizon}",
            f"Message:              {self.message}",
            "=" * 55,
        ]
        return "\n".join(lines)


class RiskParityWarning:
    """
    FAWP-based early warning for risk parity and vol targeting strategies.

    Monitors whether the vol-targeting rebalance action still has traction
    on portfolio risk. When FAWP fires, the strategy is forecasting risk
    correctly but cannot act on it — a pre-crisis signature.

    Parameters
    ----------
    tau_grid : list of int
        Delay sweep for steering MI.
    delta : int
        Forward horizon for vol forecast validation (bars).
    epsilon : float
        Steering MI threshold for warning (default 0.02).
    eta : float
        Minimum pred MI to trigger warning (default 1e-3).

    Example
    -------
        import numpy as np
        from fawp_index.quant.risk import RiskParityWarning

        vol_forecast = np.abs(np.random.randn(1000)) * 0.01 + 0.015
        realized_vol = vol_forecast[21:] + np.random.randn(979) * 0.002
        rebalance    = np.random.randn(979) * 0.05
        port_risk    = np.random.randn(979) * 0.001  # rebalance stopped working

        warner = RiskParityWarning()
        result = warner.check(vol_forecast[:979], realized_vol, rebalance, port_risk)
        print(result.summary())
    """

    def __init__(
        self,
        tau_grid: Optional[List[int]] = None,
        delta: int = 21,
        epsilon: float = 0.02,
        eta: float = 1e-3,
        n_null: int = 100,
        seed: int = 42,
    ):
        self.tau_grid = tau_grid or list(range(1, 11))
        self.delta = delta
        self.epsilon = epsilon
        self.eta = eta
        self.n_null = n_null
        self.seed = seed

    def check(
        self,
        vol_forecast: np.ndarray,
        realized_vol: np.ndarray,
        rebalance_action: np.ndarray,
        portfolio_risk_response: np.ndarray,
    ) -> RiskWarningResult:
        """
        Parameters
        ----------
        vol_forecast : array
            Predicted volatility series.
        realized_vol : array
            Realized volatility (forward-aligned).
        rebalance_action : array
            Size of rebalancing trades.
        portfolio_risk_response : array
            Change in portfolio risk after rebalance.

        Returns
        -------
        RiskWarningResult
        """
        from fawp_index.core.alpha_index import FAWPAlphaIndex

        n = min(len(vol_forecast), len(realized_vol),
                len(rebalance_action), len(portfolio_risk_response))

        result = FAWPAlphaIndex(
            eta=self.eta, epsilon=self.epsilon, n_null=self.n_null, seed=self.seed
        ).compute(
            pred_series=np.asarray(vol_forecast[:n], dtype=float),
            future_series=np.asarray(realized_vol[:n], dtype=float),
            action_series=np.asarray(rebalance_action[:n], dtype=float),
            obs_series=np.asarray(portfolio_risk_response[:n], dtype=float),
            tau_grid=self.tau_grid,
        )

        warning = bool(result.in_fawp.any())
        pred_mi = float(result.pred_mi_raw.max())
        steer_mi = float(result.steer_mi_raw.min())
        gap = pred_mi - steer_mi

        if warning:
            msg = (
                "Vol forecast still valid but rebalance has lost traction. "
                "Risk cannot be managed — consider reducing exposure."
            )
        elif pred_mi < self.eta:
            msg = "Vol forecast MI too weak to assess regime."
        else:
            msg = "Rebalance is effective. No FAWP signature detected."

        return RiskWarningResult(
            warning_active=warning,
            pred_mi_bits=pred_mi,
            steer_mi_bits=steer_mi,
            leverage_gap_bits=gap,
            risk_horizon=result.tau_h,
            message=msg,
        )
