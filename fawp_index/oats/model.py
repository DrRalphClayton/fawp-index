"""
fawp_index.oats.model — Analytic Agency Horizon (Gaussian channel)

The closed-form model underlying E1-E4.

Ralph Clayton (2026) — doi:10.5281/zenodo.18663547
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Union


def noise_variance(tau: Union[float, np.ndarray],
                   sigma0_sq: float = 0.01,
                   alpha: float = 0.001) -> Union[float, np.ndarray]:
    """
    Linear noise-variance growth with latency.

    σ²(τ) = σ0² + α·τ

    Parameters
    ----------
    tau : float or array
        Latency value(s).
    sigma0_sq : float
        Base observation noise variance.
    alpha : float
        Noise growth rate per unit latency.

    Returns
    -------
    float or array
    """
    return sigma0_sq + alpha * np.asarray(tau)


def mutual_information(tau: Union[float, np.ndarray],
                       P: float = 1.0,
                       sigma0_sq: float = 0.01,
                       alpha: float = 0.001) -> Union[float, np.ndarray]:
    """
    Analytic Gaussian channel mutual information at latency τ.

    I(τ) = 0.5 · log2(1 + P / σ²(τ))

    Parameters
    ----------
    tau : float or array
        Latency value(s).
    P : float
        Signal power (action variance).
    sigma0_sq : float
        Base noise variance.
    alpha : float
        Noise growth rate.

    Returns
    -------
    float or array (bits)

    Example
    -------
        from fawp_index.oats import mutual_information
        import numpy as np
        tau = np.linspace(0, 1000, 500)
        mi = mutual_information(tau, P=1.0, sigma0_sq=0.01, alpha=0.001)
    """
    sigma_sq = noise_variance(tau, sigma0_sq, alpha)
    return 0.5 * np.log2(1.0 + P / sigma_sq)


@dataclass
class OATSResult:
    """Result from AgencyHorizon."""
    P: float
    sigma0_sq: float
    alpha: float
    epsilon: float
    tau_h: float                    # analytic horizon
    tau_grid: np.ndarray            # tau values
    mi: np.ndarray                  # I(τ) analytic curve
    noise_var: np.ndarray           # σ²(τ) curve
    mi_at_tau_h: float              # I at horizon (should ≈ epsilon)
    snr_initial: float              # P / σ0²

    def summary(self) -> str:
        lines = [
            "=" * 55,
            "Agency Horizon (OATS Analytic Model)",
            "=" * 55,
            f"Signal power P:       {self.P:.4g}",
            f"Base noise σ0²:       {self.sigma0_sq:.4g}",
            f"Noise growth α:       {self.alpha:.4g}",
            f"MI threshold ε:       {self.epsilon:.4g}",
            f"Initial SNR:          {self.snr_initial:.2f}",
            f"Initial MI:           {self.mi[0]:.4f} bits",
            f"Agency horizon τ_h:   {self.tau_h:.2f}",
            f"MI at τ_h:            {self.mi_at_tau_h:.4f} bits",
            "=" * 55,
        ]
        return "\n".join(lines)

    def plot(self, **kwargs):
        from fawp_index.oats._plots import plot_horizon
        return plot_horizon(self, **kwargs)


class AgencyHorizon:
    """
    Analytic Gaussian channel Agency Horizon calculator.

    Computes the closed-form horizon τ_h where mutual information
    I(τ) = 0.5·log2(1 + P/σ²(τ)) first falls to ε bits, under
    linear noise growth σ²(τ) = σ0² + α·τ.

    Also runs the E1 parameter sweep: compares analytic vs Monte Carlo
    horizon estimates across a grid of (P, α, ε) values.

    Parameters
    ----------
    P : float
        Signal power / action variance. Default 1.0.
    sigma0_sq : float
        Base observation noise variance. Default 0.01.
    alpha : float
        Noise growth rate with latency. Default 0.001.
    epsilon : float
        MI threshold defining the horizon. Default 0.1 bits.

    Example
    -------
        from fawp_index.oats import AgencyHorizon

        # Single horizon
        ah = AgencyHorizon(P=1.0, sigma0_sq=0.01, alpha=0.001, epsilon=0.1)
        result = ah.compute(tau_max=5000)
        print(result.summary())
        result.plot()

        # Sweep
        sweep = ah.sweep(
            P_values=[0.1, 1.0, 10.0],
            alpha_values=[0.0001, 0.001, 0.01],
            epsilon_values=[0.01, 0.1],
        )
    """

    def __init__(
        self,
        P: float = 1.0,
        sigma0_sq: float = 0.01,
        alpha: float = 0.001,
        epsilon: float = 0.1,
    ):
        self.P = P
        self.sigma0_sq = sigma0_sq
        self.alpha = alpha
        self.epsilon = epsilon

    def tau_h_analytic(
        self,
        P: Optional[float] = None,
        sigma0_sq: Optional[float] = None,
        alpha: Optional[float] = None,
        epsilon: Optional[float] = None,
    ) -> float:
        """
        Closed-form agency horizon.

        τ_h = max(0, (P/(2^(2ε) - 1) - σ0²) / α)

        Returns float (may be inf if α=0).
        """
        P = P if P is not None else self.P
        s = sigma0_sq if sigma0_sq is not None else self.sigma0_sq
        a = alpha if alpha is not None else self.alpha
        e = epsilon if epsilon is not None else self.epsilon

        if a <= 0:
            return float('inf')
        threshold_sigma_sq = P / (2 ** (2 * e) - 1)
        tau_h = (threshold_sigma_sq - s) / a
        return float(max(0.0, tau_h))

    def compute(
        self,
        tau_max: Optional[float] = None,
        n_points: int = 500,
    ) -> OATSResult:
        """
        Compute the full analytic MI curve and horizon.

        Parameters
        ----------
        tau_max : float, optional
            Maximum latency to plot. Default: 3 × τ_h (or 1000 if τ_h=0).
        n_points : int
            Number of points on the τ grid.

        Returns
        -------
        OATSResult
        """
        tau_h = self.tau_h_analytic()
        if tau_max is None:
            if np.isfinite(tau_h) and tau_h > 0:
                tau_max = 3.0 * tau_h
            else:
                tau_max = 1000.0

        tau = np.linspace(0, tau_max, n_points)
        mi = mutual_information(tau, self.P, self.sigma0_sq, self.alpha)
        nv = noise_variance(tau, self.sigma0_sq, self.alpha)
        mi_at_h = float(mutual_information(tau_h, self.P, self.sigma0_sq, self.alpha))

        return OATSResult(
            P=self.P,
            sigma0_sq=self.sigma0_sq,
            alpha=self.alpha,
            epsilon=self.epsilon,
            tau_h=tau_h,
            tau_grid=tau,
            mi=mi,
            noise_var=nv,
            mi_at_tau_h=mi_at_h,
            snr_initial=self.P / self.sigma0_sq,
        )

    def sweep(
        self,
        P_values: Optional[List[float]] = None,
        alpha_values: Optional[List[float]] = None,
        epsilon_values: Optional[List[float]] = None,
    ) -> "SweepResult":
        """
        Parameter sweep over (P, α, ε) — reproduces E1.

        Returns a SweepResult with a DataFrame of analytic horizons
        and convenience methods for plotting/filtering.

        Example
        -------
            sweep = AgencyHorizon().sweep(
                P_values=[0.1, 1.0, 10.0],
                alpha_values=[0.0001, 0.001, 0.01],
                epsilon_values=[0.01, 0.1],
            )
            print(sweep.dataframe())
            sweep.plot_scaling()
        """
        import pandas as pd

        P_vals = P_values or [0.1, 1.0, 10.0]
        a_vals = alpha_values or [0.0001, 0.001, 0.01]
        e_vals = epsilon_values or [0.01, 0.1, 0.5]

        rows = []
        for P in P_vals:
            for a in a_vals:
                for e in e_vals:
                    tau_h = self.tau_h_analytic(P=P, alpha=a, epsilon=e)
                    mi_0 = mutual_information(0, P, self.sigma0_sq, a)
                    rows.append({
                        'P': P, 'alpha': a, 'epsilon': e,
                        'tau_h_theory': tau_h,
                        'mi_initial': float(mi_0),
                        'snr': P / self.sigma0_sq,
                    })

        df = pd.DataFrame(rows)
        return SweepResult(df=df, sigma0_sq=self.sigma0_sq)

    def compare_e1(self) -> "SweepResult":
        """
        Load bundled E1 data and compare analytic vs MC horizons.

        Returns SweepResult enriched with Monte Carlo estimates from
        the published E1 sweep.
        """
        import pandas as pd
        from fawp_index.data import E1_HORIZONS_SWEEP

        df_mc = pd.read_csv(E1_HORIZONS_SWEEP)

        # Recompute theory with our formula for verification
        df_mc['tau_h_recomputed'] = df_mc.apply(
            lambda r: self.tau_h_analytic(
                P=r['P'], alpha=r['alpha'], epsilon=r['epsilon']
            ), axis=1
        )
        df_mc['error_recomputed'] = (
            (df_mc['tau_h_recomputed'] - df_mc['tau_h_theory']).abs()
            / df_mc['tau_h_theory'].replace(0, np.nan)
        )

        covered = df_mc[df_mc['status'] == 'COVERED'].copy()
        if len(covered) > 0:
            print(f"E1 sweep: {len(df_mc)} total runs | "
                  f"{len(covered)} COVERED | "
                  f"Mean rel. error: {covered['error_relative'].mean():.4f} | "
                  f"Max rel. error: {covered['error_relative'].max():.4f}")

        return SweepResult(df=df_mc, sigma0_sq=self.sigma0_sq, has_mc=True)


@dataclass
class SweepResult:
    """Result from AgencyHorizon.sweep() or compare_e1()."""
    df: object         # pd.DataFrame
    sigma0_sq: float
    has_mc: bool = False

    def dataframe(self):
        return self.df

    def plot_scaling(self, **kwargs):
        from fawp_index.oats._plots import plot_sweep_scaling
        return plot_sweep_scaling(self, **kwargs)

    def summary(self) -> str:
        df = self.df
        lines = [
            "=" * 55,
            f"Agency Horizon Sweep: {len(df)} parameter combinations",
            "=" * 55,
        ]
        if 'tau_h_theory' in df.columns:
            finite = df['tau_h_theory'].replace(float('inf'), np.nan).dropna()
            lines.append(f"Finite horizons: {len(finite)}/{len(df)}")
            if len(finite):
                lines.append(f"Horizon range:   {finite.min():.1f} — {finite.max():.1f}")
        if self.has_mc and 'error_relative' in df.columns:
            covered = df[df.get('status', '') == 'COVERED']['error_relative'].dropna()
            if len(covered):
                lines.append(f"MC vs theory error (COVERED): mean={covered.mean():.4f} max={covered.max():.4f}")
        lines.append("=" * 55)
        return "\n".join(lines)
