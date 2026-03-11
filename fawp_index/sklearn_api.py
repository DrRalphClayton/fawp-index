"""
fawp_index.sklearn_api — Scikit-learn Style Interface

FAWPTransformer: fit() / transform() / score() compatible with
sklearn Pipelines and GridSearchCV.

Ralph Clayton (2026) — doi:10.5281/zenodo.18673949
"""

import numpy as np
from typing import Optional, List


class FAWPTransformer:
    """
    Sklearn-compatible FAWP transformer.

    Computes FAWP features from a (pred, action) array pair.
    Integrates with sklearn Pipelines.

    Parameters
    ----------
    pred_col : int or str
        Column index or name for predictor in X.
    action_col : int or str
        Column index or name for action in X.
    future_col : int or str, optional
        Column for future target. If None, auto-shifts pred by delta.
    obs_col : int or str, optional
        Column for observation. If None, uses pred.
    delta : int
        Forecast horizon (default 20).
    tau_grid : list of int
        Delay sweep.
    eta, epsilon : float
        FAWP thresholds.
    n_null : int
        Null samples for correction.

    Example
    -------
        import numpy as np
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from fawp_index.sklearn_api import FAWPTransformer

        X = np.column_stack([
            np.random.randn(2000),   # returns (pred)
            np.random.randn(2000),   # volume (action)
        ])

        fawp = FAWPTransformer(pred_col=0, action_col=1, delta=20)
        fawp.fit(X)
        score = fawp.score(X)
        print(f"FAWP score: {score:.4f}")
        print(f"In FAWP: {fawp.in_fawp_}")
    """

    def __init__(
        self,
        pred_col: int = 0,
        action_col: int = 1,
        future_col: Optional[int] = None,
        obs_col: Optional[int] = None,
        delta: int = 20,
        tau_grid: Optional[List[int]] = None,
        eta: float = 1e-4,
        epsilon: float = 1e-4,
        n_null: int = 200,
        seed: int = 42,
    ):
        self.pred_col = pred_col
        self.action_col = action_col
        self.future_col = future_col
        self.obs_col = obs_col
        self.delta = delta
        self.tau_grid = tau_grid or list(range(1, 16))
        self.eta = eta
        self.epsilon = epsilon
        self.n_null = n_null
        self.seed = seed

        # Fitted attributes (sklearn convention: trailing _)
        self.result_ = None
        self.in_fawp_ = False
        self.peak_alpha_ = 0.0
        self.tau_h_ = None
        self.is_fitted_ = False

    def _extract(self, X):
        """Extract arrays from X (numpy array or DataFrame)."""
        try:
            import pandas as pd
            if isinstance(X, pd.DataFrame):
                pred = X.iloc[:, self.pred_col].values.astype(float) \
                    if isinstance(self.pred_col, int) else X[self.pred_col].values.astype(float)
                action = X.iloc[:, self.action_col].values.astype(float) \
                    if isinstance(self.action_col, int) else X[self.action_col].values.astype(float)
                future = None if self.future_col is None else (
                    X.iloc[:, self.future_col].values.astype(float)
                    if isinstance(self.future_col, int) else X[self.future_col].values.astype(float)
                )
                obs = None if self.obs_col is None else (
                    X.iloc[:, self.obs_col].values.astype(float)
                    if isinstance(self.obs_col, int) else X[self.obs_col].values.astype(float)
                )
                return pred, action, future, obs
        except ImportError:
            pass

        X = np.asarray(X)
        pred = X[:, self.pred_col].astype(float)
        action = X[:, self.action_col].astype(float)
        future = None if self.future_col is None else X[:, self.future_col].astype(float)
        obs = None if self.obs_col is None else X[:, self.obs_col].astype(float)
        return pred, action, future, obs

    def fit(self, X, y=None):
        """
        Fit the FAWP transformer to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        self
        """
        from fawp_index.core.alpha_index import FAWPAlphaIndex

        pred, action, future, obs = self._extract(X)
        n = len(pred)

        if future is None:
            future = pred[self.delta:]
            pred = pred[:n - self.delta]
            action = action[:n - self.delta]

        if obs is None:
            obs = pred.copy()

        n = min(len(pred), len(future), len(action), len(obs))

        self.result_ = FAWPAlphaIndex(
            eta=self.eta, epsilon=self.epsilon,
            n_null=self.n_null, seed=self.seed,
        ).compute(
            pred[:n], future[:n], action[:n], obs[:n],
            tau_grid=self.tau_grid,
        )

        self.in_fawp_ = bool(self.result_.in_fawp.any())
        self.peak_alpha_ = float(self.result_.peak_alpha or 0.0)
        self.tau_h_ = self.result_.tau_h
        self.is_fitted_ = True
        return self

    def transform(self, X, y=None):
        """
        Transform X into FAWP feature array.

        Returns array of shape (n_tau, 4):
          [tau, pred_mi, steer_mi, alpha_index]
        """
        self._check_fitted()
        r = self.result_
        return np.column_stack([
            r.tau,
            r.pred_mi_corrected,
            r.steer_mi_corrected,
            r.alpha_index,
        ])

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)

    def score(self, X, y=None):
        """
        Returns peak FAWP alpha index score.
        Higher = stronger FAWP (more information-control exclusion).

        Used by sklearn GridSearchCV to compare parameter settings.
        """
        self.fit(X, y)
        return self.peak_alpha_

    def get_params(self, deep=True):
        return {
            'pred_col': self.pred_col,
            'action_col': self.action_col,
            'future_col': self.future_col,
            'obs_col': self.obs_col,
            'delta': self.delta,
            'tau_grid': self.tau_grid,
            'eta': self.eta,
            'epsilon': self.epsilon,
            'n_null': self.n_null,
            'seed': self.seed,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def _check_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before transform() or score().")

    def summary(self) -> str:
        self._check_fitted()
        return self.result_.summary()
