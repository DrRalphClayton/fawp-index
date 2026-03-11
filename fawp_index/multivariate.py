"""
fawp_index.multivariate — Multivariate FAWP

Detects FAWP across multiple predictors simultaneously.
For each predictor, computes the Alpha Index and ranks by signal strength.
Also computes a joint FAWP score using information-theoretic combination.

Ralph Clayton (2026) — doi:10.5281/zenodo.18673949
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import warnings


@dataclass
class MultivariateFAWPResult:
    """Result from MultivariateFAWP."""
    feature_names: List[str]
    tau_grid: np.ndarray
    pred_mi: np.ndarray          # shape (n_features, n_tau)
    steer_mi: np.ndarray         # shape (n_features, n_tau)
    alpha_index: np.ndarray      # shape (n_features, n_tau)
    in_fawp: np.ndarray          # shape (n_features,) — bool per feature
    peak_alpha: np.ndarray       # shape (n_features,)
    peak_tau: np.ndarray         # shape (n_features,)
    tau_h: np.ndarray            # shape (n_features,)
    joint_pred_mi: np.ndarray    # combined predictive MI across features
    joint_in_fawp: bool          # any feature in FAWP?
    dominant_feature: str        # feature with highest peak alpha

    def summary(self) -> str:
        lines = [
            "=" * 65,
            "Multivariate FAWP — Information-Control Exclusion Analysis",
            "=" * 65,
            f"Features:         {len(self.feature_names)}",
            f"Dominant feature: {self.dominant_feature}",
            f"Joint FAWP:       {'YES ⚠️' if self.joint_in_fawp else 'No'}",
            "",
            f"{'Feature':>20} {'Peak α':>8} {'Peak τ':>7} {'τ_h':>5} {'FAWP':>6}",
            "-" * 50,
        ]
        order = np.argsort(self.peak_alpha)[::-1]
        for i in order:
            fawp = " ✓" if self.in_fawp[i] else ""
            tau_h_str = str(int(self.tau_h[i])) if self.tau_h[i] >= 0 else "—"
            peak_tau_str = str(int(self.peak_tau[i])) if self.peak_tau[i] >= 0 else "—"
            lines.append(
                f"{self.feature_names[i]:>20} {self.peak_alpha[i]:>8.4f} "
                f"{peak_tau_str:>7} {tau_h_str:>5}{fawp}"
            )
        lines.append("=" * 65)
        return "\n".join(lines)

    def plot(self, **kwargs):
        from fawp_index.multivariate import _plot_multivariate
        return _plot_multivariate(self, **kwargs)


class MultivariateFAWP:
    """
    Compute FAWP Alpha Index across multiple predictors simultaneously.

    Useful for:
      - Multi-factor models: which factors are in the FAWP regime?
      - Feature selection: ranking predictors by information-control exclusion
      - Portfolio management: identifying where alpha cannot be captured

    Parameters
    ----------
    tau_grid : list of int
        Delay sweep.
    delta : int
        Forecast horizon.
    eta, epsilon : float
        FAWP thresholds.
    n_null : int
        Null samples.
    combine : str
        How to combine across features: 'max', 'mean', or 'pca'.

    Example
    -------
        import numpy as np
        from fawp_index.multivariate import MultivariateFAWP

        n = 3000
        predictors = {
            'momentum': np.random.randn(n),
            'value':    np.random.randn(n),
            'carry':    np.random.randn(n),
        }
        action = np.random.randn(n) * 0.001
        future = np.random.randn(n)

        mv = MultivariateFAWP()
        result = mv.compute(predictors, action, future)
        print(result.summary())
        result.plot()
    """

    def __init__(
        self,
        tau_grid: Optional[List[int]] = None,
        delta: int = 20,
        eta: float = 1e-4,
        epsilon: float = 1e-4,
        n_null: int = 200,
        seed: int = 42,
        combine: str = 'max',
    ):
        self.tau_grid = tau_grid or list(range(1, 16))
        self.delta = delta
        self.eta = eta
        self.epsilon = epsilon
        self.n_null = n_null
        self.seed = seed
        self.combine = combine

    def compute(
        self,
        predictors: Dict[str, np.ndarray],
        action_series: np.ndarray,
        future_series: Optional[np.ndarray] = None,
        obs_series: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> MultivariateFAWPResult:
        """
        Parameters
        ----------
        predictors : dict of {name: array}
            Multiple predictor time series.
        action_series : array
            Action/control series (shared across all predictors).
        future_series : array, optional
            Future target. If None, auto-shifts each predictor by delta.
        obs_series : array, optional
            Observation. If None, uses action_series.
        verbose : bool

        Returns
        -------
        MultivariateFAWPResult
        """
        from fawp_index.core.alpha_index import FAWPAlphaIndex

        names = list(predictors.keys())
        action = np.asarray(action_series, dtype=float)
        obs = np.asarray(obs_series, dtype=float) if obs_series is not None else action.copy()

        n_features = len(names)
        n_tau = len(self.tau_grid)

        pred_mi_mat = np.zeros((n_features, n_tau))
        steer_mi_mat = np.zeros((n_features, n_tau))
        alpha_mat = np.zeros((n_features, n_tau))
        in_fawp_arr = np.zeros(n_features, dtype=bool)
        peak_alpha_arr = np.zeros(n_features)
        peak_tau_arr = np.full(n_features, -1, dtype=int)
        tau_h_arr = np.full(n_features, -1, dtype=int)

        detector = FAWPAlphaIndex(
            eta=self.eta, epsilon=self.epsilon,
            n_null=self.n_null, seed=self.seed,
        )

        for i, name in enumerate(names):
            if verbose:
                print(f"  Computing FAWP for feature '{name}' ({i+1}/{n_features})...")

            pred = np.asarray(predictors[name], dtype=float)
            n = len(pred)

            if future_series is not None:
                future = np.asarray(future_series, dtype=float)
            else:
                future = pred[self.delta:]
                pred = pred[:n - self.delta]
                _action = action[:n - self.delta]
                _obs = obs[:n - self.delta]
                n = len(pred)
            
            if future_series is not None:
                _action = action
                _obs = obs

            m = min(len(pred), len(future), len(_action), len(_obs))

            try:
                result = detector.compute(
                    pred[:m], future[:m], _action[:m], _obs[:m],
                    tau_grid=self.tau_grid,
                )
                pred_mi_mat[i] = result.pred_mi_raw[:n_tau]
                steer_mi_mat[i] = result.steer_mi_raw[:n_tau]
                alpha_mat[i] = result.alpha_index[:n_tau]
                in_fawp_arr[i] = bool(result.in_fawp.any())
                peak_alpha_arr[i] = float(result.peak_alpha or 0.0)
                if result.peak_tau is not None:
                    peak_tau_arr[i] = int(result.peak_tau)
                if result.tau_h is not None:
                    tau_h_arr[i] = int(result.tau_h)
            except Exception as e:
                warnings.warn(f"Feature '{name}' failed: {e}")

        # Joint predictive MI (max across features per tau)
        if self.combine == 'max':
            joint_pred_mi = pred_mi_mat.max(axis=0)
        elif self.combine == 'mean':
            joint_pred_mi = pred_mi_mat.mean(axis=0)
        else:
            joint_pred_mi = pred_mi_mat.max(axis=0)

        dominant_idx = int(np.argmax(peak_alpha_arr))

        return MultivariateFAWPResult(
            feature_names=names,
            tau_grid=np.array(self.tau_grid),
            pred_mi=pred_mi_mat,
            steer_mi=steer_mi_mat,
            alpha_index=alpha_mat,
            in_fawp=in_fawp_arr,
            peak_alpha=peak_alpha_arr,
            peak_tau=peak_tau_arr,
            tau_h=tau_h_arr,
            joint_pred_mi=joint_pred_mi,
            joint_in_fawp=bool(in_fawp_arr.any()),
            dominant_feature=names[dominant_idx],
        )


def _plot_multivariate(result: MultivariateFAWPResult, save_path=None, show=True):
    """Plot multivariate FAWP — one panel per feature."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        raise ImportError("pip install matplotlib")

    n = len(result.feature_names)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig = plt.figure(figsize=(5 * cols, 4 * rows + 1))
    fig.suptitle(
        f"Multivariate FAWP — {n} features\n"
        f"Dominant: {result.dominant_feature} | "
        f"Joint FAWP: {'YES ⚠️' if result.joint_in_fawp else 'No'}",
        fontsize=11, y=1.01,
    )

    tau = result.tau_grid

    for i, name in enumerate(result.feature_names):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.plot(tau, result.steer_mi[i], 'b--', linewidth=1.8, label='Steer MI')
        ax.plot(tau, result.pred_mi[i], color='darkorange', linewidth=2, label='Pred MI')
        ax.fill_between(tau, result.steer_mi[i], result.pred_mi[i],
                        where=(result.pred_mi[i] > result.steer_mi[i]),
                        alpha=0.15, color='darkorange')
        if result.tau_h[i] >= 0:
            ax.axvline(result.tau_h[i], color='gray', linestyle='--', linewidth=1.2)
        fawp_label = " ⚠️" if result.in_fawp[i] else ""
        ax.set_title(f"{name}{fawp_label}\nα={result.peak_alpha[i]:.3f}", fontsize=9)
        ax.set_xlabel('τ', fontsize=8)
        ax.set_ylabel('MI (bits)', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        if i == 0:
            ax.legend(fontsize=7)

    fig.text(0.99, 0.01, 'fawp-index | Clayton (2026)',
             ha='right', va='bottom', fontsize=7, color='gray', style='italic')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    return fig
