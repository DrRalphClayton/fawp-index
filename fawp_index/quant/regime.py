"""
fawp_index.quant.regime — Market Regime Detection
Flags when a market/strategy enters FAWP: model still predicts returns
but execution edge (steering) has collapsed.

The quant interpretation:
  - Predictive MI = alpha signal strength (factor → future return)
  - Steering MI   = execution edge (action → market response)
  - FAWP regime   = signal persists but market no longer responds to trades
                    → crowding, liquidity collapse, or market microstructure breakdown

Ralph Clayton (2026) — doi:10.5281/zenodo.18673949
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List
import warnings


@dataclass
class RegimeResult:
    """Result from FAWPRegimeDetector."""
    timestamps: np.ndarray          # time index (or integer positions)
    in_fawp: np.ndarray             # bool array — True = FAWP regime active
    pred_mi: np.ndarray             # rolling predictive MI
    steer_mi: np.ndarray            # rolling steering MI
    leverage_gap: np.ndarray        # pred_mi - steer_mi
    alpha_index: np.ndarray         # FAWP alpha score
    regime_changes: List[int]       # indices where regime flipped
    n_fawp_windows: int             # total windows in FAWP
    fawp_fraction: float            # fraction of time in FAWP

    def summary(self) -> str:
        lines = [
            "=" * 55,
            "FAWP Regime Detection — Market Breakdown Analysis",
            "=" * 55,
            f"Total windows:        {len(self.in_fawp)}",
            f"FAWP windows:         {self.n_fawp_windows} "
            f"({self.fawp_fraction*100:.1f}% of time)",
            f"Regime changes:       {len(self.regime_changes)}",
            f"Peak leverage gap:    {self.leverage_gap.max():.4f} bits",
            f"Peak alpha index:     {self.alpha_index.max():.4f}",
            "=" * 55,
        ]
        return "\n".join(lines)

    def plot(self, **kwargs):
        from fawp_index.quant._regime_plot import plot_regime
        return plot_regime(self, **kwargs)


class FAWPRegimeDetector:
    """
    Rolling-window FAWP regime detector for market data.

    Scans a time series with a sliding window, computing predictive MI
    and steering MI at each step. Flags windows where the leverage gap
    opens (prediction persists, execution edge collapses).

    Quant use cases:
      - Crowding detection: factor alpha survives, market impact has gone
      - Liquidity regimes: before/after liquidity events
      - Strategy monitoring: real-time flag when model breaks down

    Parameters
    ----------
    window : int
        Rolling window size in bars (default 252 = 1 trading year).
    step : int
        Step size between windows (default 21 = monthly).
    tau : int
        Delay for steering MI (default 5).
    delta : int
        Forecast horizon for predictive MI (default 21).
    eta : float
        Minimum predictive MI to gate FAWP (default 1e-3).
    epsilon : float
        Maximum steering MI for FAWP (default 0.05).
    n_null : int
        Null samples for MI correction (default 100).

    Example
    -------
        import numpy as np
        from fawp_index.quant.regime import FAWPRegimeDetector

        returns = np.random.randn(2000) * 0.01
        volumes = np.abs(np.random.randn(2000)) + 1

        detector = FAWPRegimeDetector(window=252, step=21, delta=21)
        result = detector.detect(returns, volumes)
        print(result.summary())
        result.plot()
    """

    def __init__(
        self,
        window: int = 252,
        step: int = 21,
        tau: int = 5,
        delta: int = 21,
        eta: float = 1e-3,
        epsilon: float = 0.05,
        n_null: int = 100,
        seed: int = 42,
    ):
        self.window = window
        self.step = step
        self.tau = tau
        self.delta = delta
        self.eta = eta
        self.epsilon = epsilon
        self.n_null = n_null
        self.seed = seed

    def detect(
        self,
        pred_series: np.ndarray,
        action_series: np.ndarray,
        future_series: Optional[np.ndarray] = None,
        obs_series: Optional[np.ndarray] = None,
        timestamps=None,
        verbose: bool = False,
    ) -> RegimeResult:
        """
        Run rolling FAWP regime detection.

        Parameters
        ----------
        pred_series : array
            Predictor (e.g. factor signal, log returns).
        action_series : array
            Action/control proxy (e.g. trade volume, order flow).
        future_series : array, optional
            Future target. If None, auto-constructed from pred_series + delta.
        obs_series : array, optional
            Observation. If None, uses pred_series.
        timestamps : array-like, optional
            Time index for output labelling.
        verbose : bool
            Print progress.

        Returns
        -------
        RegimeResult
        """
        from fawp_index.core.estimators import mi_from_arrays, conservative_null_floor

        pred = np.asarray(pred_series, dtype=float)
        action = np.asarray(action_series, dtype=float)
        n = len(pred)

        # Auto-construct future and obs if not provided
        if future_series is None:
            if n > self.delta:
                future = pred[self.delta:]
                pred = pred[:n - self.delta]
                action = action[:n - self.delta]
                n = len(pred)
            else:
                raise ValueError(f"Series too short ({n}) for delta={self.delta}")
        else:
            future = np.asarray(future_series, dtype=float)

        obs = np.asarray(obs_series, dtype=float) if obs_series is not None else pred.copy()

        min_len = min(len(pred), len(future), len(action), len(obs))
        pred, future, action, obs = (
            pred[:min_len], future[:min_len], action[:min_len], obs[:min_len]
        )

        if timestamps is not None:
            ts = np.asarray(timestamps)[:min_len]
        else:
            ts = np.arange(min_len)

        rng = np.random.default_rng(self.seed)

        # Rolling windows
        window_starts = list(range(0, min_len - self.window, self.step))
        if not window_starts:
            raise ValueError(
                f"Series length {min_len} too short for window={self.window}. "
                f"Use a smaller window."
            )

        pred_mi_arr = np.zeros(len(window_starts))
        steer_mi_arr = np.zeros(len(window_starts))
        alpha_arr = np.zeros(len(window_starts))
        in_fawp_arr = np.zeros(len(window_starts), dtype=bool)
        window_ts = np.array([ts[i + self.window // 2] for i in window_starts])

        for idx, start in enumerate(window_starts):
            end = start + self.window

            w_pred = pred[start:end]
            w_future = future[start:end]
            w_action = action[start:end]
            w_obs = obs[start:end]

            # Steering MI: I(action_t ; obs_{t+tau})
            tau_end = len(w_action) - self.tau
            if tau_end > 20:
                a_t = w_action[:tau_end]
                o_tau = w_obs[self.tau:self.tau + tau_end]
                raw_steer = mi_from_arrays(a_t, o_tau)
                null_steer = conservative_null_floor(
                    a_t, o_tau, n_null=self.n_null, rng=rng
                )
                steer_mi = max(0.0, raw_steer - null_steer)
            else:
                steer_mi = 0.0

            # Predictive MI: I(pred_t ; future_{t+delta})
            raw_pred = mi_from_arrays(w_pred, w_future)
            null_pred = conservative_null_floor(
                w_pred, w_future, n_null=self.n_null, rng=rng
            )
            pred_mi = max(0.0, raw_pred - null_pred)

            pred_mi_arr[idx] = pred_mi
            steer_mi_arr[idx] = steer_mi

            # FAWP condition
            in_fawp = (pred_mi >= self.eta) and (steer_mi <= self.epsilon)
            gap = pred_mi - steer_mi
            alpha = gap * float(in_fawp) * (1.0 + np.log1p(max(0, gap)))

            alpha_arr[idx] = alpha
            in_fawp_arr[idx] = in_fawp

            if verbose and idx % 10 == 0:
                print(
                    f"  Window {idx+1}/{len(window_starts)}: "
                    f"pred={pred_mi:.4f} steer={steer_mi:.4f} "
                    f"FAWP={'YES' if in_fawp else 'no'}"
                )

        # Regime change detection
        regime_changes = [
            i for i in range(1, len(in_fawp_arr))
            if in_fawp_arr[i] != in_fawp_arr[i - 1]
        ]

        return RegimeResult(
            timestamps=window_ts,
            in_fawp=in_fawp_arr,
            pred_mi=pred_mi_arr,
            steer_mi=steer_mi_arr,
            leverage_gap=pred_mi_arr - steer_mi_arr,
            alpha_index=alpha_arr,
            regime_changes=regime_changes,
            n_fawp_windows=int(in_fawp_arr.sum()),
            fawp_fraction=float(in_fawp_arr.mean()),
        )
