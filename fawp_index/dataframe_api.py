"""
fawp_index.dataframe_api — Pandas DataFrame Native API

Pass a DataFrame directly. No numpy extraction needed.

Ralph Clayton (2026) — doi:10.5281/zenodo.18673949
"""

import numpy as np
from typing import Optional, List, Union
from fawp_index.core.alpha_index import FAWPAlphaIndex, FAWPResult


def fawp_from_dataframe(
    df,
    pred_col: str,
    action_col: str,
    future_col: Optional[str] = None,
    obs_col: Optional[str] = None,
    delta: int = 20,
    tau_grid: Optional[List[int]] = None,
    eta: float = 1e-4,
    epsilon: float = 1e-4,
    n_null: int = 200,
    dropna: bool = True,
    **kwargs,
) -> FAWPResult:
    """
    Run FAWP Alpha Index directly on a pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    pred_col : str
        Predictor column name (e.g. 'returns', 'factor_score').
    action_col : str
        Action/control column name (e.g. 'volume', 'trade_size').
    future_col : str, optional
        Future target column. If None, auto-shifts pred_col by delta.
    obs_col : str, optional
        Observation column. If None, uses pred_col.
    delta : int
        Forecast horizon if future_col not provided.
    tau_grid : list of int, optional
        Delay sweep. Default: 1..15.
    eta : float
        Predictive MI gate threshold.
    epsilon : float
        Steering MI collapse threshold.
    n_null : int
        Null samples for MI correction.
    dropna : bool
        Drop NaN rows before computing (default True).

    Returns
    -------
    FAWPResult
        Standard result with .summary(), .plot(), .to_dataframe() methods.

    Example
    -------
        import pandas as pd
        import numpy as np
        from fawp_index import fawp_from_dataframe

        df = pd.DataFrame({
            'returns': np.random.randn(2000) * 0.01,
            'volume':  np.abs(np.random.randn(2000)) + 1,
        })

        result = fawp_from_dataframe(df, pred_col='returns', action_col='volume')
        print(result.summary())
        result.plot()
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required: pip install pandas")

    if dropna:
        df = df.dropna(subset=[pred_col, action_col])

    pred = df[pred_col].values.astype(float)
    action = df[action_col].values.astype(float)

    if future_col is not None:
        future = df[future_col].values.astype(float)
    else:
        n = len(pred)
        if n <= delta:
            raise ValueError(f"DataFrame has {n} rows but delta={delta}. Need > {delta} rows.")
        future = pred[delta:]
        pred = pred[:n - delta]
        action = action[:n - delta]

    if obs_col is not None:
        obs = df[obs_col].values.astype(float)
    else:
        obs = pred.copy()

    n = min(len(pred), len(future), len(action), len(obs))
    pred, future, action, obs = pred[:n], future[:n], action[:n], obs[:n]

    tau_grid = tau_grid or list(range(1, 16))

    detector = FAWPAlphaIndex(eta=eta, epsilon=epsilon, n_null=n_null, **kwargs)
    result = detector.compute(pred, future, action, obs, tau_grid=tau_grid)

    return result


def fawp_rolling(
    df,
    pred_col: str,
    action_col: str,
    future_col: Optional[str] = None,
    window: int = 252,
    step: int = 21,
    tau: int = 5,
    delta: int = 21,
    epsilon: float = 0.05,
    eta: float = 1e-3,
    n_null: int = 50,
):
    """
    Add rolling FAWP columns to a DataFrame.

    Computes FAWP regime flags on a rolling window and returns
    the DataFrame with additional columns appended.

    New columns added:
      fawp_pred_mi    — rolling predictive MI
      fawp_steer_mi   — rolling steering MI
      fawp_gap        — leverage gap
      fawp_in_regime  — bool: FAWP active in this window

    Parameters
    ----------
    df : pd.DataFrame
    pred_col : str
    action_col : str
    future_col : str, optional
    window : int
    step : int
    tau : int
    delta : int
    epsilon : float
    eta : float
    n_null : int

    Returns
    -------
    pd.DataFrame with fawp_* columns added

    Example
    -------
        df_annotated = fawp_rolling(df, pred_col='returns', action_col='volume')
        df_annotated[df_annotated['fawp_in_regime']].head()
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pip install pandas")

    from fawp_index.quant.regime import FAWPRegimeDetector

    pred = df[pred_col].values.astype(float)
    action = df[action_col].values.astype(float)

    if future_col is not None:
        future = df[future_col].values.astype(float)
    else:
        future = None

    detector = FAWPRegimeDetector(
        window=window, step=step, tau=tau, delta=delta,
        epsilon=epsilon, eta=eta, n_null=n_null,
    )

    result = detector.detect(
        pred_series=pred,
        action_series=action,
        future_series=future,
        timestamps=df.index.values if hasattr(df.index, 'values') else None,
    )

    # Build output DataFrame aligned to window midpoints
    out = pd.DataFrame({
        'fawp_pred_mi': result.pred_mi,
        'fawp_steer_mi': result.steer_mi,
        'fawp_gap': result.leverage_gap,
        'fawp_in_regime': result.in_fawp,
    }, index=result.timestamps)

    return out
