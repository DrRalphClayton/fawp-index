"""
fawp-index: Live / streaming data support
Supports rolling-window FAWP detection on live data feeds.
"""
import numpy as np
from collections import deque
from typing import Optional, Callable
from ..core.alpha_index import FAWPAlphaIndex, FAWPResult


class FAWPStreamDetector:
    """
    Rolling-window FAWP detector for live data streams.

    Maintains a sliding buffer of recent observations and recomputes
    the Alpha Index whenever enough data has accumulated.

    Usage
    -----
        detector = FAWPStreamDetector(window=500, delta_pred=20)

        # Feed data points one at a time:
        for state, action in live_feed:
            result = detector.update(state=state, action=action)
            if result and result.in_fawp.any():
                print("FAWP REGIME DETECTED", result.summary())

    Parameters
    ----------
    window : int
        Rolling buffer size (number of time steps to keep).
    delta_pred : int
        Forecast horizon for predictive MI.
    tau_grid : list of int, optional
        Delay values to sweep. Defaults to [1..15].
    min_samples : int
        Minimum samples before computing (avoids startup noise).
    on_fawp : callable, optional
        Callback triggered when FAWP regime is detected.
        Called with (FAWPResult,).
    fawp_kwargs : dict
        Arguments passed to FAWPAlphaIndex.
    """

    def __init__(
        self,
        window: int = 500,
        delta_pred: int = 20,
        tau_grid: Optional[list] = None,
        min_samples: int = 100,
        on_fawp: Optional[Callable] = None,
        **fawp_kwargs,
    ):
        self.window = window
        self.delta_pred = delta_pred
        self.tau_grid = tau_grid or list(range(1, 16))
        self.min_samples = min_samples
        self.on_fawp = on_fawp

        self._state_buf = deque(maxlen=window)
        self._action_buf = deque(maxlen=window)
        self._step = 0

        self._detector = FAWPAlphaIndex(**fawp_kwargs)
        self._last_result: Optional[FAWPResult] = None

    def update(self, state: float, action: float) -> Optional[FAWPResult]:
        """
        Feed one new data point.

        Parameters
        ----------
        state : float
            Current observed state value.
        action : float
            Current action/control input.

        Returns
        -------
        FAWPResult if enough data to compute, else None.
        """
        self._state_buf.append(float(state))
        self._action_buf.append(float(action))
        self._step += 1

        if len(self._state_buf) < max(self.min_samples, self.delta_pred + 10):
            return None

        state_arr = np.array(self._state_buf)
        action_arr = np.array(self._action_buf)
        n = len(state_arr) - self.delta_pred

        pred = state_arr[:n]
        future = state_arr[self.delta_pred:self.delta_pred + n]
        act = action_arr[:n]
        obs = state_arr[:n]

        result = self._detector.compute(
            pred_series=pred,
            future_series=future,
            action_series=act,
            obs_series=obs,
            tau_grid=self.tau_grid,
            delta_pred=self.delta_pred,
            verbose=False,
        )

        self._last_result = result

        if result.in_fawp.any() and self.on_fawp is not None:
            self.on_fawp(result)

        return result

    def update_batch(self, states, actions):
        """Feed multiple data points at once. Returns last result."""
        result = None
        for s, a in zip(states, actions):
            result = self.update(s, a)
        return result

    @property
    def last_result(self) -> Optional[FAWPResult]:
        return self._last_result

    @property
    def step(self) -> int:
        return self._step

    @property
    def buffer_size(self) -> int:
        return len(self._state_buf)
