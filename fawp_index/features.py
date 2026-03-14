"""
fawp_index.features — Feature Importance
Which columns in your DataFrame carry the FAWP signal?

Ranks columns by their FAWP alpha index when used as the predictor,
helping identify which features are in the information-control exclusion regime.

Ralph Clayton (2026) — doi:10.5281/zenodo.18673949
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class FeatureImportanceResult:
    """Result from FAWPFeatureImportance."""
    feature_names: List[str]
    alpha_scores: np.ndarray      # FAWP alpha index per feature
    pred_mi: np.ndarray           # predictive MI per feature
    steer_mi: np.ndarray          # steering MI per feature
    in_fawp: np.ndarray           # bool: feature in FAWP regime
    ranked_features: List[str]    # features ranked by alpha score
    ranked_scores: np.ndarray     # corresponding scores

    def summary(self) -> str:
        lines = [
            "=" * 55,
            "FAWP Feature Importance",
            "=" * 55,
            f"{'Rank':>4} {'Feature':>20} {'Alpha':>8} {'Pred MI':>8} {'Steer MI':>9} {'FAWP':>6}",
            "-" * 55,
        ]
        for i, (feat, score) in enumerate(zip(self.ranked_features, self.ranked_scores)):
            idx = self.feature_names.index(feat)
            fawp = " ✓" if self.in_fawp[idx] else ""
            lines.append(
                f"{i+1:>4} {feat:>20} {score:>8.4f} "
                f"{self.pred_mi[idx]:>8.4f} {self.steer_mi[idx]:>9.4f}{fawp}"
            )
        lines.append("=" * 55)
        n_fawp = int(self.in_fawp.sum())
        lines.append(f"{n_fawp}/{len(self.feature_names)} features in FAWP regime")
        return "\n".join(lines)

    def plot(self, **kwargs):
        from fawp_index.features import _plot_importance
        return _plot_importance(self, **kwargs)


class FAWPFeatureImportance:
    """
    Rank DataFrame columns by FAWP signal strength.

    For each candidate feature column, treats it as the predictor and
    computes the FAWP alpha index against a fixed action column and
    target. Features with high alpha are in the information-control
    exclusion regime — they predict well but cannot be acted upon.

    Parameters
    ----------
    action_col : str or int
        Column to use as action/control proxy (fixed across all features).
    target_col : str or int, optional
        Future target column. If None, auto-shifts each feature by delta.
    delta : int
        Forecast horizon.
    tau_grid : list of int
        Delay sweep.
    eta, epsilon : float
        FAWP thresholds.
    n_null : int
        Null samples.

    Example
    -------
        import pandas as pd
        import numpy as np
        from fawp_index.features import FAWPFeatureImportance

        n = 2000
        df = pd.DataFrame({
            'momentum':  np.random.randn(n),
            'value':     np.random.randn(n),
            'quality':   np.random.randn(n),
            'vol':       np.abs(np.random.randn(n)),
            'trade_size': np.random.randn(n) * 0.001,
        })

        fi = FAWPFeatureImportance(action_col='trade_size', delta=21)
        result = fi.fit(df, feature_cols=['momentum', 'value', 'quality', 'vol'])
        print(result.summary())
        result.plot()
    """

    def __init__(
        self,
        action_col,
        target_col=None,
        delta: int = 20,
        tau_grid: Optional[List[int]] = None,
        eta: float = 1e-4,
        epsilon: float = 1e-4,
        n_null: int = 100,
        seed: int = 42,
    ):
        self.action_col = action_col
        self.target_col = target_col
        self.delta = delta
        self.tau_grid = tau_grid or list(range(1, 11))
        self.eta = eta
        self.epsilon = epsilon
        self.n_null = n_null
        self.seed = seed

    def fit(self, df, feature_cols: Optional[List] = None) -> FeatureImportanceResult:
        """
        Parameters
        ----------
        df : pd.DataFrame or np.ndarray
            Input data.
        feature_cols : list, optional
            Columns to evaluate. If None, uses all columns except action_col.

        Returns
        -------
        FeatureImportanceResult
        """
        from fawp_index.core.alpha_index import FAWPAlphaIndex

        try:
            import pandas as pd
            is_df = isinstance(df, pd.DataFrame)
        except ImportError:
            is_df = False

        if is_df:
            if feature_cols is None:
                feature_cols = [c for c in df.columns if c != self.action_col]
            action = df[self.action_col].values.astype(float)
            target = df[self.target_col].values.astype(float) if self.target_col else None

            def get_col(c):
                return df[c].values.astype(float)

            names = [str(c) for c in feature_cols]
        else:
            df = np.asarray(df)
            if feature_cols is None:
                feature_cols = [i for i in range(df.shape[1]) if i != self.action_col]
            action = df[:, self.action_col].astype(float)
            target = df[:, self.target_col].astype(float) if self.target_col is not None else None

            def get_col(c):  # type: ignore[misc]
                return df[:, c].astype(float)

            names = [f"col_{c}" for c in feature_cols]

        alpha_scores = []
        pred_mi_list = []
        steer_mi_list = []
        in_fawp_list = []

        detector = FAWPAlphaIndex(
            eta=self.eta, epsilon=self.epsilon,
            n_null=self.n_null, seed=self.seed,
        )

        for col in feature_cols:
            pred = get_col(col)
            n = len(pred)

            if target is not None:
                fut = target[:n]
            else:
                fut = pred[self.delta:]
                pred = pred[:n - self.delta]
                _action = action[:n - self.delta]
            
            if target is not None:
                _action = action

            obs = pred.copy()
            m = min(len(pred), len(fut), len(_action), len(obs))

            try:
                result = detector.compute(
                    pred[:m], fut[:m], _action[:m], obs[:m],
                    tau_grid=self.tau_grid,
                )
                alpha_scores.append(float(result.peak_alpha or 0.0))
                pred_mi_list.append(float(result.pred_mi_raw.max()))
                steer_mi_list.append(float(result.steer_mi_raw.min()))
                in_fawp_list.append(bool(result.in_fawp.any()))
            except Exception:
                alpha_scores.append(0.0)
                pred_mi_list.append(0.0)
                steer_mi_list.append(0.0)
                in_fawp_list.append(False)

        alpha_arr = np.array(alpha_scores)
        order = np.argsort(alpha_arr)[::-1]

        return FeatureImportanceResult(
            feature_names=names,
            alpha_scores=alpha_arr,
            pred_mi=np.array(pred_mi_list),
            steer_mi=np.array(steer_mi_list),
            in_fawp=np.array(in_fawp_list),
            ranked_features=[names[i] for i in order],
            ranked_scores=alpha_arr[order],
        )


def _plot_importance(result: FeatureImportanceResult, save_path=None, show=True):
    """Plot feature importance as horizontal bar chart."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        raise ImportError("pip install matplotlib")

    fig, ax = plt.subplots(figsize=(9, max(4, len(result.ranked_features) * 0.5 + 1)))

    y = np.arange(len(result.ranked_features))
    colors = [
        'darkorange' if result.in_fawp[result.feature_names.index(f)]
        else 'steelblue'
        for f in result.ranked_features
    ]

    ax.barh(y, result.ranked_scores[::-1], color=colors[::-1], edgecolor='white')
    ax.set_yticks(y)
    ax.set_yticklabels(result.ranked_features[::-1], fontsize=10)
    ax.set_xlabel("FAWP Alpha Index", fontsize=11)
    ax.set_title("FAWP Feature Importance\n"
                 "Orange = in FAWP regime (predicts but cannot be acted upon)",
                 fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')

    legend = [
        mpatches.Patch(color='darkorange', label='FAWP regime'),
        mpatches.Patch(color='steelblue', label='No FAWP'),
    ]
    ax.legend(handles=legend, fontsize=9)

    fig.text(0.99, 0.01, 'fawp-index | Clayton (2026)',
             ha='right', va='bottom', fontsize=7, color='gray', style='italic')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    return fig, ax
