"""
fawp_index.finance.quant
========================
Quantitative finance applications of FAWP Alpha Index v2.1.

Four tools for quants:
  1. FAWPRegimeDetector   — flags when market enters FAWP (model breakdown)
  2. MomentumDecayScanner — detects when alpha signal survives but execution edge is gone
  3. RiskParityWarning    — FAWP as early warning before strategy failure / vol spike
  4. EventStudyFAWP       — predictive MI around earnings / macro announcements

Ralph Clayton (2026) — DOI: https://doi.org/10.5281/zenodo.18673949
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

from ..core.alpha_index import FAWPAlphaIndex, FAWPResult


# ══════════════════════════════════════════════════════════════════════════════
# 1. REGIME DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RegimeResult:
    """Result from FAWPRegimeDetector."""
    dates: np.ndarray                  # window end dates
    alpha_scores: np.ndarray           # FAWP alpha index per window
    in_fawp: np.ndarray                # bool: FAWP regime active
    pred_mi: np.ndarray                # predictive MI per window
    steer_mi: np.ndarray               # steering MI per window
    regime_changes: List[int]          # indices where regime changed
    n_windows: int
    fawp_fraction: float               # fraction of windows in FAWP

    def summary(self) -> str:
        lines = [
            "=" * 55,
            "FAWP Regime Detection Results",
            "=" * 55,
            f"Windows analysed:    {self.n_windows}",
            f"FAWP fraction:       {self.fawp_fraction:.1%}",
            f"Regime changes:      {len(self.regime_changes)}",
            f"Peak alpha:          {self.alpha_scores.max():.4f}",
            "=" * 55,
        ]
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "date": self.dates,
            "alpha_score": self.alpha_scores,
            "in_fawp": self.in_fawp,
            "pred_mi": self.pred_mi,
            "steer_mi": self.steer_mi,
        })


class FAWPRegimeDetector:
    """
    Detect FAWP regimes in financial time series using a rolling window.

    A FAWP regime occurs when a model's predictive signal (e.g. factor
    returns) persists while its execution edge (e.g. volume-weighted
    market impact) collapses. This flags model breakdown *before* PnL
    deteriorates.

    Parameters
    ----------
    window : int
        Rolling window size in bars (default: 252 — one trading year).
    step : int
        Step between windows (default: 21 — monthly).
    tau_grid : list
        Delay values to sweep (default: [1..10]).
    delta_pred : int
        Forecast horizon in bars (default: 20).
    eta : float
        Minimum predictive MI to qualify as FAWP (default: 1e-4).
    epsilon : float
        Maximum steering MI to qualify as FAWP (default: 1e-3).
    n_null : int
        Null shuffle samples for bias correction (default: 100).

    Example
    -------
        detector = FAWPRegimeDetector(window=252, step=21)
        result = detector.detect(
            prices=df['close'],
            volume=df['volume'],
        )
        print(result.summary())
        regime_df = result.to_dataframe()
    """

    def __init__(
        self,
        window: int = 252,
        step: int = 21,
        tau_grid: Optional[List[int]] = None,
        delta_pred: int = 20,
        eta: float = 1e-4,
        epsilon: float = 1e-3,
        n_null: int = 100,
        seed: int = 42,
    ):
        self.window = window
        self.step = step
        self.tau_grid = tau_grid or list(range(1, 11))
        self.delta_pred = delta_pred
        self.eta = eta
        self.epsilon = epsilon
        self.n_null = n_null
        self.seed = seed

    def detect(
        self,
        prices: pd.Series,
        volume: Optional[pd.Series] = None,
        returns: Optional[pd.Series] = None,
        action_series: Optional[pd.Series] = None,
    ) -> RegimeResult:
        """
        Run FAWP regime detection over rolling windows.

        Parameters
        ----------
        prices : pd.Series
            Price series (close). Used to compute log returns.
        volume : pd.Series, optional
            Volume series. Used as action proxy (execution pressure).
        returns : pd.Series, optional
            Pre-computed returns. If provided, prices is ignored.
        action_series : pd.Series, optional
            Custom action proxy. Overrides volume.

        Returns
        -------
        RegimeResult
        """
        # Build returns
        if returns is not None:
            ret = np.asarray(returns, dtype=float)
            dates = returns.index.values if hasattr(returns, 'index') else np.arange(len(ret))
        else:
            ret = np.diff(np.log(np.asarray(prices, dtype=float) + 1e-10))
            dates = prices.index.values[1:] if hasattr(prices, 'index') else np.arange(len(ret))

        # Action proxy
        if action_series is not None:
            act = np.asarray(action_series, dtype=float)[:len(ret)]
        elif volume is not None:
            vol = np.asarray(volume, dtype=float)
            act = np.diff(np.log(vol + 1))[:len(ret)]
        else:
            # Use lagged return as action proxy (momentum signal)
            act = np.roll(ret, 1)
            act[0] = 0.0

        n = len(ret)
        detector = FAWPAlphaIndex(
            eta=self.eta, epsilon=self.epsilon,
            n_null=self.n_null, seed=self.seed,
        )

        window_dates, alphas, in_fawps, pred_mis, steer_mis = [], [], [], [], []

        for start in range(0, n - self.window, self.step):
            end = start + self.window
            if end + self.delta_pred >= n:
                break

            w_ret = ret[start:end]
            w_act = act[start:end]
            w_future = ret[start + self.delta_pred: end + self.delta_pred]
            w_obs = w_ret + np.random.default_rng(self.seed + start).normal(0, 0.001, len(w_ret))

            min_len = min(len(w_ret), len(w_future), len(w_act), len(w_obs))
            if min_len < 50:
                continue

            try:
                result = detector.compute(
                    pred_series=w_ret[:min_len],
                    future_series=w_future[:min_len],
                    action_series=w_act[:min_len],
                    obs_series=w_obs[:min_len],
                    tau_grid=self.tau_grid,
                )
                window_dates.append(dates[end - 1] if end - 1 < len(dates) else end)
                alphas.append(result.peak_alpha)
                in_fawps.append(bool(result.in_fawp.any()))
                pred_mis.append(float(result.pred_mi_raw.max()))
                steer_mis.append(float(result.steer_mi_raw.min()))
            except Exception:
                continue

        alphas = np.array(alphas)
        in_fawps = np.array(in_fawps, dtype=bool)

        # Regime changes
        changes = [i for i in range(1, len(in_fawps)) if in_fawps[i] != in_fawps[i - 1]]

        return RegimeResult(
            dates=np.array(window_dates),
            alpha_scores=alphas,
            in_fawp=in_fawps,
            pred_mi=np.array(pred_mis),
            steer_mi=np.array(steer_mis),
            regime_changes=changes,
            n_windows=len(alphas),
            fawp_fraction=float(in_fawps.mean()) if len(in_fawps) > 0 else 0.0,
        )


# ══════════════════════════════════════════════════════════════════════════════
# 2. MOMENTUM DECAY SCANNER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MomentumDecayResult:
    """Result from MomentumDecayScanner."""
    tau: np.ndarray
    signal_mi: np.ndarray        # predictive MI of momentum signal
    execution_mi: np.ndarray     # steering MI of execution
    leverage_gap: np.ndarray     # signal_mi - execution_mi
    decay_point: Optional[int]   # tau where execution edge collapses
    signal_survives: bool        # True if signal MI > eta past decay_point
    alpha_index: np.ndarray

    def summary(self) -> str:
        lines = [
            "=" * 55,
            "Momentum Decay Scan",
            "=" * 55,
            f"Execution edge collapse at: τ = {self.decay_point}",
            f"Signal survives beyond:     {'YES — FAWP detected' if self.signal_survives else 'NO'}",
            f"Peak leverage gap:          {self.leverage_gap.max():.4f} bits",
            "=" * 55,
        ]
        return "\n".join(lines)


class MomentumDecayScanner:
    """
    Detect when a momentum strategy's alpha signal (predictive edge)
    persists while its execution edge (market impact) decays.

    This is the FAWP Prediction Paradox applied to systematic trading:
    the signal knows something the market hasn't yet priced, but the
    strategy can no longer profitably extract it (crowding, capacity,
    slippage).

    Parameters
    ----------
    tau_grid : list
        Execution delay sweep (in bars).
    delta_pred : int
        Signal forecast horizon (bars).
    epsilon : float
        Execution MI collapse threshold.

    Example
    -------
        scanner = MomentumDecayScanner()
        result = scanner.scan(
            signal=df['momentum_score'],
            future_returns=df['fwd_return_20d'],
            trade_size=df['order_size'],
            market_impact=df['slippage'],
        )
        print(result.summary())
    """

    def __init__(
        self,
        tau_grid: Optional[List[int]] = None,
        delta_pred: int = 20,
        epsilon: float = 1e-3,
        eta: float = 1e-4,
        n_null: int = 100,
    ):
        self.tau_grid = tau_grid or list(range(1, 15))
        self.delta_pred = delta_pred
        self.epsilon = epsilon
        self.eta = eta
        self.n_null = n_null

    def scan(
        self,
        signal: pd.Series,
        future_returns: pd.Series,
        trade_size: Optional[pd.Series] = None,
        market_impact: Optional[pd.Series] = None,
    ) -> MomentumDecayResult:
        """
        Scan for momentum decay.

        Parameters
        ----------
        signal : pd.Series
            Alpha signal (e.g. momentum score, factor loading).
        future_returns : pd.Series
            Forward returns at horizon delta_pred.
        trade_size : pd.Series, optional
            Order / trade size series (execution action proxy).
        market_impact : pd.Series, optional
            Observed slippage / market impact (execution observation).
        """
        sig = np.asarray(signal, dtype=float)
        fut = np.asarray(future_returns, dtype=float)

        if trade_size is not None:
            act = np.asarray(trade_size, dtype=float)
        else:
            act = np.diff(sig, prepend=sig[0])

        if market_impact is not None:
            obs = np.asarray(market_impact, dtype=float)
        else:
            obs = sig + np.random.default_rng(42).normal(0, sig.std() * 0.1, len(sig))

        min_len = min(len(sig), len(fut), len(act), len(obs))
        sig, fut, act, obs = sig[:min_len], fut[:min_len], act[:min_len], obs[:min_len]

        detector = FAWPAlphaIndex(
            eta=self.eta, epsilon=self.epsilon, n_null=self.n_null
        )
        result = detector.compute(
            pred_series=sig,
            future_series=fut,
            action_series=act,
            obs_series=obs,
            tau_grid=self.tau_grid,
        )

        # Decay point: first tau where execution MI <= epsilon
        decay_point = result.tau_h

        # Does signal survive past decay?
        if decay_point is not None:
            post_decay = result.tau >= decay_point
            signal_survives = bool(result.pred_mi_raw[post_decay].max() > self.eta)
        else:
            signal_survives = False

        return MomentumDecayResult(
            tau=result.tau,
            signal_mi=result.pred_mi_raw,
            execution_mi=result.steer_mi_raw,
            leverage_gap=result.pred_mi_raw - result.steer_mi_raw,
            decay_point=decay_point,
            signal_survives=signal_survives,
            alpha_index=result.alpha_index,
        )


# ══════════════════════════════════════════════════════════════════════════════
# 3. RISK PARITY WARNING
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RiskWarningResult:
    """Result from RiskParityWarning."""
    dates: np.ndarray
    fawp_score: np.ndarray           # rolling FAWP alpha score
    vol_realized: np.ndarray         # realized volatility
    warning_flags: np.ndarray        # bool: FAWP warning active
    vol_spike_lead: Optional[float]  # average lead time before vol spikes (bars)
    n_warnings: int
    hit_rate: Optional[float]        # fraction of warnings followed by vol spike

    def summary(self) -> str:
        lines = [
            "=" * 55,
            "FAWP Risk Parity Warning Results",
            "=" * 55,
            f"Windows with warning:  {self.n_warnings}",
            f"Vol spike lead time:   {f'{self.vol_spike_lead:.1f} bars' if self.vol_spike_lead else 'N/A'}",
            f"Warning hit rate:      {f'{self.hit_rate:.1%}' if self.hit_rate else 'N/A'}",
            "=" * 55,
        ]
        return "\n".join(lines)


class RiskParityWarning:
    """
    Use FAWP as an early warning system for risk parity / vol targeting
    strategy failures.

    The intuition: when a portfolio's rebalancing signal (predictive MI)
    persists but its execution window (steering MI) closes due to market
    stress, the strategy enters FAWP. This precedes realized vol spikes
    because the rebalancing pressure accumulates without relief.

    Example
    -------
        warner = RiskParityWarning(window=60, vol_threshold=1.5)
        result = warner.warn(
            returns=df['portfolio_returns'],
            rebal_signal=df['rebal_score'],
            volume=df['market_volume'],
        )
        print(result.summary())
    """

    def __init__(
        self,
        window: int = 60,
        step: int = 5,
        vol_horizon: int = 10,
        vol_threshold: float = 1.5,
        tau_grid: Optional[List[int]] = None,
        delta_pred: int = 10,
        n_null: int = 50,
    ):
        self.window = window
        self.step = step
        self.vol_horizon = vol_horizon
        self.vol_threshold = vol_threshold
        self.tau_grid = tau_grid or list(range(1, 8))
        self.delta_pred = delta_pred
        self.n_null = n_null

    def warn(
        self,
        returns: pd.Series,
        rebal_signal: Optional[pd.Series] = None,
        volume: Optional[pd.Series] = None,
    ) -> RiskWarningResult:
        ret = np.asarray(returns, dtype=float)
        dates = returns.index.values if hasattr(returns, 'index') else np.arange(len(ret))

        if rebal_signal is not None:
            sig = np.asarray(rebal_signal, dtype=float)[:len(ret)]
        else:
            # Rolling z-score of returns as rebal proxy
            sig = np.zeros(len(ret))
            for i in range(self.window, len(ret)):
                w = ret[i - self.window:i]
                s = w.std()
            sig[i] = (ret[i] - w.mean()) / (s + 1e-10) if np.isfinite(s) else 0.0

        if volume is not None:
            act = np.diff(np.log(np.asarray(volume, dtype=float) + 1))[:len(ret)]
            act = np.pad(act, (0, max(0, len(ret) - len(act))))[:len(ret)]
        else:
            act = np.abs(ret)  # volume proxy: absolute returns

        # Realized vol
        vol_realized = np.array([
            ret[max(0, i - self.window):i].std() * np.sqrt(252)
            for i in range(len(ret))
        ])
        vol_baseline = np.median(vol_realized[vol_realized > 0])

        detector = FAWPAlphaIndex(n_null=self.n_null)
        window_dates, fawp_scores, warnings = [], [], []

        for start in range(0, len(ret) - self.window - self.delta_pred, self.step):
            end = start + self.window
            w_ret = ret[start:end]
            w_sig = sig[start:end]
            w_act = act[start:end]
            w_future = ret[start + self.delta_pred: end + self.delta_pred]
            w_obs = w_sig + np.random.default_rng(start).normal(0, 0.01, len(w_sig))

            min_len = min(len(w_ret), len(w_future), len(w_act), len(w_obs))
            if min_len < 30:
                continue

            try:
                r = detector.compute(
                    pred_series=w_sig[:min_len],
                    future_series=w_future[:min_len],
                    action_series=w_act[:min_len],
                    obs_series=w_obs[:min_len],
                    tau_grid=self.tau_grid,
                )
                window_dates.append(dates[end - 1] if end - 1 < len(dates) else end)
                fawp_scores.append(r.peak_alpha)
                warnings.append(bool(r.in_fawp.any()))
            except Exception:
                continue

        fawp_scores = np.array(fawp_scores)
        warnings = np.array(warnings, dtype=bool)

        # Estimate lead time to vol spike
        vol_spike_lead = None
        hit_rate = None
        if len(window_dates) > 0 and vol_baseline > 0:
            hits = 0
            leads = []
            for i, (w, flag) in enumerate(zip(window_dates, warnings)):
                if flag:
                    # Look for vol spike in next vol_horizon windows
                    future_vols = vol_realized[
                        min(len(vol_realized) - 1, int(i * self.step) + self.window):
                        min(len(vol_realized) - 1, int(i * self.step) + self.window + self.vol_horizon * self.step)
                    ]
                    if len(future_vols) > 0 and future_vols.max() > vol_baseline * self.vol_threshold:
                        hits += 1
                        leads.append(float(np.argmax(future_vols > vol_baseline * self.vol_threshold)))
            if warnings.sum() > 0:
                hit_rate = hits / warnings.sum()
            if leads:
                vol_spike_lead = float(np.mean(leads))

        return RiskWarningResult(
            dates=np.array(window_dates),
            fawp_score=fawp_scores,
            vol_realized=vol_realized[:len(window_dates)] if len(vol_realized) >= len(window_dates) else vol_realized,
            warning_flags=warnings,
            vol_spike_lead=vol_spike_lead,
            n_warnings=int(warnings.sum()),
            hit_rate=hit_rate,
        )


# ══════════════════════════════════════════════════════════════════════════════
# 4. EVENT STUDY
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EventStudyResult:
    """Result from EventStudyFAWP."""
    event_dates: List
    pre_fawp: np.ndarray       # FAWP alpha in pre-event window
    post_fawp: np.ndarray      # FAWP alpha in post-event window
    pre_pred_mi: np.ndarray
    post_pred_mi: np.ndarray
    pre_steer_mi: np.ndarray
    post_steer_mi: np.ndarray
    mi_lift: float             # avg pred MI lift at event
    n_events: int

    def summary(self) -> str:
        lines = [
            "=" * 55,
            "FAWP Event Study Results",
            "=" * 55,
            f"Events analysed:       {self.n_events}",
            f"Avg pre-event alpha:   {self.pre_fawp.mean():.4f}",
            f"Avg post-event alpha:  {self.post_fawp.mean():.4f}",
            f"Avg MI lift at event:  {self.mi_lift:.4f} bits",
            f"FAWP pre-event:        {(self.pre_fawp > 0).mean():.1%} of events",
            f"FAWP post-event:       {(self.post_fawp > 0).mean():.1%} of events",
            "=" * 55,
        ]
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "event_date": self.event_dates,
            "pre_alpha": self.pre_fawp,
            "post_alpha": self.post_fawp,
            "pre_pred_mi": self.pre_pred_mi,
            "post_pred_mi": self.post_pred_mi,
            "pre_steer_mi": self.pre_steer_mi,
            "post_steer_mi": self.post_steer_mi,
            "mi_lift": self.post_pred_mi - self.pre_pred_mi,
        })


class EventStudyFAWP:
    """
    Measure FAWP alpha and predictive MI around scheduled events
    (earnings, FOMC, economic releases).

    Tests whether predictive coupling rises *before* events (information
    leakage or positioning) and whether steering coupling collapses
    *after* events (illiquidity, wide spreads).

    Parameters
    ----------
    pre_window : int
        Bars before event to analyse (default: 20).
    post_window : int
        Bars after event to analyse (default: 20).
    tau_grid : list
        Delay sweep.
    delta_pred : int
        Prediction horizon.

    Example
    -------
        study = EventStudyFAWP(pre_window=20, post_window=20)
        result = study.run(
            returns=df['returns'],
            event_dates=['2024-01-25', '2024-04-25'],
            volume=df['volume'],
        )
        print(result.summary())
        result.to_dataframe()
    """

    def __init__(
        self,
        pre_window: int = 20,
        post_window: int = 20,
        tau_grid: Optional[List[int]] = None,
        delta_pred: int = 5,
        n_null: int = 50,
    ):
        self.pre_window = pre_window
        self.post_window = post_window
        self.tau_grid = tau_grid or list(range(1, 8))
        self.delta_pred = delta_pred
        self.n_null = n_null

    def run(
        self,
        returns: pd.Series,
        event_dates: List,
        volume: Optional[pd.Series] = None,
        action_series: Optional[pd.Series] = None,
    ) -> EventStudyResult:
        """
        Run FAWP event study.

        Parameters
        ----------
        returns : pd.Series
            Return series with DatetimeIndex.
        event_dates : list
            List of event dates (strings or Timestamps).
        volume : pd.Series, optional
            Volume series for action proxy.
        action_series : pd.Series, optional
            Custom action series.
        """
        ret = returns.copy()
        idx = ret.index

        if volume is not None:
            vol = volume.copy()
            act_full = vol.pct_change().fillna(0)
        elif action_series is not None:
            act_full = action_series.copy()
        else:
            act_full = ret.rolling(5).std().fillna(method='bfill')

        detector = FAWPAlphaIndex(n_null=self.n_null)

        pre_alphas, post_alphas = [], []
        pre_preds, post_preds = [], []
        pre_steers, post_steers = [], []
        valid_dates = []

        for evt in event_dates:
            try:
                evt_ts = pd.Timestamp(evt)
                # Find position
                pos_arr = np.searchsorted(idx, evt_ts)
                if pos_arr < self.pre_window + self.delta_pred:
                    continue
                if pos_arr + self.post_window + self.delta_pred >= len(ret):
                    continue

                def _compute_window(start, end):
                    w_ret = ret.iloc[start:end].values
                    w_act = act_full.iloc[start:end].values
                    w_future = ret.iloc[start + self.delta_pred: end + self.delta_pred].values
                    w_obs = w_ret + np.random.default_rng(start).normal(0, 0.001, len(w_ret))
                    min_len = min(len(w_ret), len(w_future), len(w_act), len(w_obs))
                    if min_len < 15:
                        return None
                    return detector.compute(
                        pred_series=w_ret[:min_len],
                        future_series=w_future[:min_len],
                        action_series=w_act[:min_len],
                        obs_series=w_obs[:min_len],
                        tau_grid=self.tau_grid,
                    )

                pre_start = pos_arr - self.pre_window
                r_pre = _compute_window(pre_start, pos_arr)
                r_post = _compute_window(pos_arr, pos_arr + self.post_window)

                if r_pre is None or r_post is None:
                    continue

                pre_alphas.append(r_pre.peak_alpha)
                post_alphas.append(r_post.peak_alpha)
                pre_preds.append(float(r_pre.pred_mi_raw.max()))
                post_preds.append(float(r_post.pred_mi_raw.max()))
                pre_steers.append(float(r_pre.steer_mi_raw.min()))
                post_steers.append(float(r_post.steer_mi_raw.min()))
                valid_dates.append(evt)

            except Exception:
                continue

        pre_preds_arr = np.array(pre_preds) if pre_preds else np.array([0.0])
        post_preds_arr = np.array(post_preds) if post_preds else np.array([0.0])

        return EventStudyResult(
            event_dates=valid_dates,
            pre_fawp=np.array(pre_alphas) if pre_alphas else np.array([0.0]),
            post_fawp=np.array(post_alphas) if post_alphas else np.array([0.0]),
            pre_pred_mi=pre_preds_arr,
            post_pred_mi=post_preds_arr,
            pre_steer_mi=np.array(pre_steers) if pre_steers else np.array([0.0]),
            post_steer_mi=np.array(post_steers) if post_steers else np.array([0.0]),
            mi_lift=float((post_preds_arr - pre_preds_arr).mean()),
            n_events=len(valid_dates),
        )
