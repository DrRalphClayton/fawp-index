"""
fawp_index.quant.events — Earnings / Event Study FAWP Analysis

Measures FAWP around scheduled announcements (earnings, FOMC, macro releases).

Pre-announcement: information begins accumulating in the state variable
but execution (trading ahead of the event) is constrained by regulations,
liquidity, or market awareness.

The FAWP signature around events:
  - Predictive MI rises as announcement approaches (information builds)
  - Steering MI falls (market won't move on your trades — everyone is waiting)
  - Post-event: FAWP collapses as information is released and tradeable

Ralph Clayton (2026) — doi:10.5281/zenodo.18673949
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class EventStudyResult:
    """Result from EventStudyFAWP."""
    window_labels: List[str]     # e.g. ['t-5', 't-4', ..., 't+5']
    pred_mi: np.ndarray          # predictive MI per event window
    steer_mi: np.ndarray         # steering MI per event window
    leverage_gap: np.ndarray     # gap per window
    in_fawp: np.ndarray          # bool per window
    pre_event_fawp: bool         # FAWP detected before event?
    post_event_fawp: bool        # FAWP detected after event?
    peak_gap_window: str         # window label with max gap
    n_events: int                # number of events averaged over

    def summary(self) -> str:
        lines = [
            "=" * 55,
            f"Event Study — FAWP Analysis ({self.n_events} events)",
            "=" * 55,
            f"Pre-event FAWP:   {'YES ⚠️' if self.pre_event_fawp else 'No'}",
            f"Post-event FAWP:  {'YES ⚠️' if self.post_event_fawp else 'No'}",
            f"Peak gap at:      {self.peak_gap_window}",
            "",
            f"{'Window':>8} {'Pred MI':>10} {'Steer MI':>10} {'Gap':>10} {'FAWP':>6}",
            "-" * 48,
        ]
        for i, label in enumerate(self.window_labels):
            fawp = "← ✓" if self.in_fawp[i] else ""
            lines.append(
                f"{label:>8} {self.pred_mi[i]:>10.4f} "
                f"{self.steer_mi[i]:>10.4f} "
                f"{self.leverage_gap[i]:>10.4f} {fawp}"
            )
        lines.append("=" * 55)
        return "\n".join(lines)


class EventStudyFAWP:
    """
    FAWP analysis around scheduled announcements.

    Slices a time series into pre/post event windows around each
    event date, computes MI in each window, and averages across events.

    Parameters
    ----------
    pre_window : int
        Bars before event to include (default 10).
    post_window : int
        Bars after event to include (default 10).
    min_window_size : int
        Minimum samples per window for MI computation (default 50).
    delta : int
        Forward horizon for predictive MI (default 5).
    tau : int
        Delay for steering MI (default 3).
    epsilon : float
        Steering MI threshold for FAWP.
    eta : float
        Predictive MI threshold for FAWP.

    Example
    -------
        import numpy as np
        from fawp_index.quant.events import EventStudyFAWP

        n = 5000
        prices = np.cumsum(np.random.randn(n) * 0.01)
        returns = np.diff(prices)
        volumes = np.abs(np.random.randn(n-1)) + 1
        # Quarterly earnings: every 63 bars
        event_indices = list(range(252, n-1, 63))

        study = EventStudyFAWP(pre_window=10, post_window=10)
        result = study.analyze(returns, volumes, event_indices)
        print(result.summary())
    """

    def __init__(
        self,
        pre_window: int = 10,
        post_window: int = 10,
        min_window_size: int = 30,
        delta: int = 5,
        tau: int = 3,
        epsilon: float = 0.05,
        eta: float = 1e-3,
        n_null: int = 50,
        seed: int = 42,
    ):
        self.pre_window = pre_window
        self.post_window = post_window
        self.min_window_size = min_window_size
        self.delta = delta
        self.tau = tau
        self.epsilon = epsilon
        self.eta = eta
        self.n_null = n_null
        self.seed = seed

    def analyze(
        self,
        returns: np.ndarray,
        volumes: np.ndarray,
        event_indices: List[int],
        future_returns: Optional[np.ndarray] = None,
    ) -> EventStudyResult:
        """
        Parameters
        ----------
        returns : array
            Return series.
        volumes : array
            Volume / order flow series.
        event_indices : list of int
            Integer positions of each event in the series.
        future_returns : array, optional
            Forward returns. If None, auto-shifted from returns.

        Returns
        -------
        EventStudyResult
        """
        from fawp_index.core.estimators import mi_from_arrays, conservative_null_floor

        returns = np.asarray(returns, dtype=float)
        volumes = np.asarray(volumes, dtype=float)
        n = len(returns)

        if future_returns is None:
            if n > self.delta:
                future = returns[self.delta:]
                base_ret = returns[:n - self.delta]
                base_vol = volumes[:n - self.delta]
            else:
                raise ValueError("Series too short for delta shift")
        else:
            future = np.asarray(future_returns, dtype=float)
            base_ret = returns
            base_vol = volumes

        n_base = min(len(base_ret), len(future), len(base_vol))
        base_ret = base_ret[:n_base]
        future = future[:n_base]
        base_vol = base_vol[:n_base]

        rng = np.random.default_rng(self.seed)
        total = self.pre_window + self.post_window + 1
        window_labels = (
            [f"t{i-self.pre_window}" for i in range(self.pre_window)] +
            ["t=0"] +
            [f"t+{i+1}" for i in range(self.post_window)]
        )

        # Accumulate MI per relative position across events
        pred_mi_accum = np.zeros(total)
        steer_mi_accum = np.zeros(total)
        counts = np.zeros(total, dtype=int)

        valid_events = [
            e for e in event_indices
            if (e - self.pre_window >= 0) and (e + self.post_window < n_base)
        ]

        if not valid_events:
            raise ValueError("No valid events found within series bounds.")

        for event_idx in valid_events:
            start = event_idx - self.pre_window

            for rel_pos in range(total):
                abs_pos = start + rel_pos
                # Small local window around this relative position
                w_start = max(0, abs_pos - self.min_window_size // 2)
                w_end = min(n_base, abs_pos + self.min_window_size // 2)

                if w_end - w_start < 20:
                    continue

                w_ret = base_ret[w_start:w_end]
                w_fut = future[w_start:w_end]
                w_vol = base_vol[w_start:w_end]

                # Predictive MI
                raw_pred = mi_from_arrays(w_ret, w_fut)
                null_pred = conservative_null_floor(w_ret, w_fut, n_null=self.n_null, rng=rng)
                pred_mi_accum[rel_pos] += max(0.0, raw_pred - null_pred)

                # Steering MI
                tau_end = len(w_vol) - self.tau
                if tau_end > 10:
                    raw_steer = mi_from_arrays(w_vol[:tau_end], w_ret[self.tau:self.tau + tau_end])
                    null_steer = conservative_null_floor(
                        w_vol[:tau_end], w_ret[self.tau:self.tau + tau_end],
                        n_null=self.n_null, rng=rng
                    )
                    steer_mi_accum[rel_pos] += max(0.0, raw_steer - null_steer)

                counts[rel_pos] += 1

        # Average across events
        safe_counts = np.where(counts > 0, counts, 1)
        pred_mi_avg = pred_mi_accum / safe_counts
        steer_mi_avg = steer_mi_accum / safe_counts
        gap_avg = pred_mi_avg - steer_mi_avg
        in_fawp = (pred_mi_avg >= self.eta) & (steer_mi_avg <= self.epsilon)

        pre_slice = slice(0, self.pre_window)
        post_slice = slice(self.pre_window + 1, total)

        peak_gap_idx = int(np.argmax(gap_avg))

        return EventStudyResult(
            window_labels=window_labels,
            pred_mi=pred_mi_avg,
            steer_mi=steer_mi_avg,
            leverage_gap=gap_avg,
            in_fawp=in_fawp,
            pre_event_fawp=bool(in_fawp[pre_slice].any()),
            post_event_fawp=bool(in_fawp[post_slice].any()),
            peak_gap_window=window_labels[peak_gap_idx],
            n_events=len(valid_events),
        )
