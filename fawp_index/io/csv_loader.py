"""
fawp-index: CSV data loader
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class FAWPData:
    pred_series: np.ndarray
    future_series: np.ndarray
    action_series: np.ndarray
    obs_series: np.ndarray
    timestamps: Optional[np.ndarray] = None
    metadata: Optional[dict] = None


def load_csv(filepath, pred_col, future_col, action_col, obs_col,
             timestamp_col=None, delta_pred=20, dropna=True):
    df = pd.read_csv(filepath)
    if dropna:
        cols = [pred_col, future_col, action_col, obs_col]
        df = df.dropna(subset=[c for c in cols if c in df.columns])
    timestamps = df[timestamp_col].to_numpy() if timestamp_col else None
    return FAWPData(
        pred_series=df[pred_col].to_numpy(dtype=float),
        future_series=df[future_col].to_numpy(dtype=float),
        action_series=df[action_col].to_numpy(dtype=float),
        obs_series=df[obs_col].to_numpy(dtype=float),
        timestamps=timestamps,
        metadata={"source": filepath, "delta_pred": delta_pred, "n_rows": len(df)},
    )


def load_csv_simple(filepath, state_col, action_col, delta_pred=20, timestamp_col=None):
    """
    Simplified loader — only needs state + action columns.
    Auto-builds future series by shifting state forward delta_pred steps.
    Great for quick analysis.
    """
    df = pd.read_csv(filepath)
    df = df.dropna(subset=[state_col, action_col])
    state = df[state_col].to_numpy(dtype=float)
    action = df[action_col].to_numpy(dtype=float)
    n = len(state) - delta_pred
    timestamps = df[timestamp_col].to_numpy()[:n] if timestamp_col else None
    return FAWPData(
        pred_series=state[:n],
        future_series=state[delta_pred:delta_pred + n],
        action_series=action[:n],
        obs_series=state[:n],
        timestamps=timestamps,
        metadata={"source": filepath, "delta_pred": delta_pred,
                  "n_rows": n, "mode": "simple_auto_align"},
    )
