"""
fawp_index.simulate — Simulation Engine

Run your own E8-style FAWP experiments. Generate unstable AR(1) systems
with configurable growth factor, controller gain, observation delay,
and prediction horizon. Reproduce or extend Clayton (2026).

Ralph Clayton (2026) — doi:10.5281/zenodo.18673949
"""

import numpy as np
from fawp_index.constants import (
    FLAGSHIP_A, FLAGSHIP_K, FLAGSHIP_DELTA_PRED,
    FLAGSHIP_N_TRIALS, FLAGSHIP_X_FAIL, FLAGSHIP_SIGMA_PROC,
    FLAGSHIP_U_MAX, EPSILON_STEERING_RAW,
)
from dataclasses import dataclass
from typing import Optional, List, Dict


@dataclass
class SimulationResult:
    """Result from FAWPSimulator."""
    config: Dict
    tau_grid: np.ndarray
    mi_pred_pooled: np.ndarray
    mi_steer_pooled: np.ndarray
    mi_pred_strat: np.ndarray
    gap: np.ndarray
    fail_rate: np.ndarray
    tau_h: Optional[int]
    peak_strat_mi: float
    peak_strat_tau: Optional[int]

    def summary(self) -> str:
        cfg = self.config
        lines = [
            "=" * 60,
            "FAWP Simulation Results",
            "=" * 60,
            f"a={cfg['a']}  K={cfg['K']}  Δ={cfg['delta_pred']}  "
            f"trials={cfg['n_trials']}  x_fail={cfg['x_fail']}",
            f"Agency horizon τ_h:        {self.tau_h}",
            f"Peak stratified pred MI:   {self.peak_strat_mi:.4f} bits "
            f"(τ={self.peak_strat_tau})",
            "",
            f"{'τ':>4} {'Steer':>10} {'Pooled':>10} {'Strat':>10} {'Fail':>6}",
            "-" * 45,
        ]
        for i, tau in enumerate(self.tau_grid):
            lines.append(
                f"{int(tau):>4} {self.mi_steer_pooled[i]:>10.4f} "
                f"{self.mi_pred_pooled[i]:>10.4f} "
                f"{self.mi_pred_strat[i]:>10.4f} "
                f"{self.fail_rate[i]:>6.2f}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)

    def plot(self, **kwargs):
        from fawp_index.simulate import _plot_simulation
        return _plot_simulation(self, **kwargs)

    def to_dataframe(self):
        """Export results to pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pip install pandas")
        return pd.DataFrame({
            'tau': self.tau_grid,
            'mi_steer': self.mi_steer_pooled,
            'mi_pred_pooled': self.mi_pred_pooled,
            'mi_pred_strat': self.mi_pred_strat,
            'gap': self.gap,
            'fail_rate': self.fail_rate,
        })


class FAWPSimulator:
    """
    Run E8-style FAWP simulations from scratch.

    Simulates an unstable AR(1) system under delayed feedback control:
        x_{t+1} = a·x_t + u_t + w_t
        u_t = -K · y_{t-τ}   (delayed observation)
        y_t = x_t + noise

    Sweeps across a grid of delays τ, computing:
      - Pooled predictive MI: I(D_t ; X_{t+Δ})
      - Pooled steering MI:   I(A_t ; O_{t+τ+1})
      - Stratified pred MI:   time-confound-controlled version
      - Failure rate:         fraction of trials that crash

    Parameters
    ----------
    a : float
        AR(1) growth factor. a > 1 = unstable. Default 1.02.
    K : float
        Controller gain. Default 0.8.
    delta_pred : int
        Prediction horizon Δ. Default 20.
    n_trials : int
        Number of rollout trials per delay. Default 100.
    n_steps : int
        Steps per trial. Default 1000.
    x_fail : float
        Crash threshold. Default 500.
    sigma_proc : float
        Process noise std. Default 1.0.
    sigma_obs_base : float
        Base observation noise std. Default 0.1.
    alpha_obs : float
        Observation noise growth with delay. Default 0.001.
    u_max : float
        Control saturation. Default 10.0.
    epsilon : float
        Steering MI threshold for agency horizon. Default 0.01.
    seed : int
        Random seed. Default 42.

    Example
    -------
        from fawp_index.simulate import FAWPSimulator

        # Reproduce E8 flagship run
        sim = FAWPSimulator(a=1.02, K=0.8, delta_pred=20, n_trials=100)
        result = sim.run(tau_grid=list(range(0, 16)))
        print(result.summary())
        result.plot()

        # Explore parameter space
        for a in [1.01, 1.02, 1.05]:
            sim = FAWPSimulator(a=a, K=0.8)
            r = sim.run(tau_grid=list(range(0, 12)))
            print(f'a={a}: tau_h={r.tau_h}, peak={r.peak_strat_mi:.3f}')
    """

    def __init__(
        self,
        a: float = FLAGSHIP_A,
        K: float = FLAGSHIP_K,
        delta_pred: int = FLAGSHIP_DELTA_PRED,
        n_trials: int = FLAGSHIP_N_TRIALS,
        n_steps: int = 1000,
        x_fail: float = FLAGSHIP_X_FAIL,
        sigma_proc: float = FLAGSHIP_SIGMA_PROC,
        sigma_obs_base: float = 0.1,
        alpha_obs: float = 0.001,
        u_max: float = FLAGSHIP_U_MAX,
        epsilon: float = EPSILON_STEERING_RAW,
        seed: int = 42,
    ):
        self.a = a
        self.K = K
        self.delta_pred = delta_pred
        self.n_trials = n_trials
        self.n_steps = n_steps
        self.x_fail = x_fail
        self.sigma_proc = sigma_proc
        self.sigma_obs_base = sigma_obs_base
        self.alpha_obs = alpha_obs
        self.u_max = u_max
        self.epsilon = epsilon
        self.seed = seed

    def _obs_noise_std(self, delay: int) -> float:
        return float(np.sqrt(self.sigma_obs_base**2 + self.alpha_obs * delay))

    def _mi_bits(self, x: np.ndarray, y: np.ndarray, min_n: int = 30) -> float:
        """Gaussian MI from correlation (bits)."""
        x, y = np.asarray(x), np.asarray(y)
        if len(x) < min_n or len(y) < min_n:
            return 0.0
        rho = np.corrcoef(x, y)[0, 1]
        if not np.isfinite(rho):
            return 0.0
        rho = np.clip(rho, -0.9999, 0.9999)
        return float(-0.5 * np.log(1.0 - rho**2) / np.log(2.0))

    def _run_trial(self, delay: int, rng: np.random.Generator) -> dict:
        """Single rollout. Returns D, X, A_steer, O_steer, crash_time, t_end."""
        obs_std = self._obs_noise_std(delay)
        total = self.n_steps + self.delta_pred
        x_hist = np.zeros(total + 1)

        D = np.empty(self.n_steps)
        A = np.empty(self.n_steps)
        O = np.full(self.n_steps, np.nan)  # noqa: E741

        x = 0.0
        x_hist[0] = x
        crashed = False
        crash_time = None
        fail_step = self.n_steps

        for t in range(self.n_steps):
            D[t] = x
            td = t - delay
            if td < 0:
                u = 0.0
                y_del = np.nan
            else:
                y_del = x_hist[td] + rng.normal(0.0, obs_std)
                u = float(np.clip(-self.K * y_del, -self.u_max, self.u_max))
            A[t] = u
            O[t] = y_del
            x = self.a * x + u + rng.normal(0.0, self.sigma_proc)
            x_hist[t + 1] = x
            if abs(x) > self.x_fail:
                crashed = True
                crash_time = t
                fail_step = t + 1
                break

        t_end = fail_step if crashed else self.n_steps
        # Free evolution for future target
        x_curr = x_hist[t_end]
        for k in range(t_end, total):
            x_curr = self.a * x_curr + rng.normal(0.0, self.sigma_proc)
            x_hist[k + 1] = x_curr

        pred_arr = D[:t_end]
        fut_arr = x_hist[self.delta_pred: self.delta_pred + t_end]
        shift = delay + 1
        sl = max(0, t_end - shift)
        A_s = A[:sl]
        O_s = O[shift: shift + sl]
        mask = np.isfinite(O_s)
        return {
            "D": pred_arr, "X": fut_arr,
            "A": A_s[mask], "O": O_s[mask],
            "crash_time": crash_time, "t_end": t_end,
        }

    def _stratified_mi(self, trials, burn_in=100, min_trials=50, t_cap=600) -> float:
        t_cap_eff = min(t_cap, max(d["t_end"] for d in trials) - 1)
        if t_cap_eff <= 5:
            return 0.0
        mis = []
        for t in range(t_cap_eff):
            if t < burn_in:
                continue
            xs = [d["D"][t] for d in trials if d["t_end"] > t]
            ys = [d["X"][t] for d in trials if d["t_end"] > t]
            if len(xs) < min_trials:
                continue
            mi = self._mi_bits(np.array(xs), np.array(ys), min_n=min_trials)
            if np.isfinite(mi) and mi > 0:
                mis.append(mi)
        return float(np.mean(mis)) if mis else 0.0

    def run(
        self,
        tau_grid: Optional[List[int]] = None,
        verbose: bool = True,
        burn_in: int = 100,
        min_trials_strat: int = 50,
        t_cap: int = 600,
    ) -> SimulationResult:
        """
        Run the full delay sweep.

        Parameters
        ----------
        tau_grid : list of int
            Delays to sweep. Default: 0..15.
        verbose : bool
            Print progress.
        burn_in : int
            Burn-in steps for stratified MI.
        min_trials_strat : int
            Minimum trials per time step for stratified MI.
        t_cap : int
            Time cap for stratified MI computation.

        Returns
        -------
        SimulationResult
        """
        if tau_grid is None:
            tau_grid = list(range(0, 16))

        if verbose:
            print(f"FAWPSimulator: a={self.a}, K={self.K}, "
                  f"Δ={self.delta_pred}, {self.n_trials} trials")
            print(f"Delay sweep: τ={min(tau_grid)}..{max(tau_grid)}")

        master_rng = np.random.default_rng(self.seed)

        pred_pool_arr = []
        steer_pool_arr = []
        strat_arr = []
        fail_arr = []

        for d in tau_grid:
            trials = []
            pool_D, pool_X, pool_A, pool_O = [], [], [], []
            fail_count = 0

            for _ in range(self.n_trials):
                trial_seed = int(master_rng.integers(0, 2**32))
                td = self._run_trial(int(d), np.random.default_rng(trial_seed))
                trials.append(td)
                fail_count += int(td["crash_time"] is not None)
                pool_D.append(td["D"])
                pool_X.append(td["X"])
                if td["A"].size > 0:
                    pool_A.append(td["A"])
                    pool_O.append(td["O"])

            all_D = np.concatenate(pool_D) if pool_D else np.array([])
            all_X = np.concatenate(pool_X) if pool_X else np.array([])
            all_A = np.concatenate(pool_A) if pool_A else np.array([])
            all_O = np.concatenate(pool_O) if pool_O else np.array([])

            mi_pred = self._mi_bits(all_D, all_X)
            mi_steer = self._mi_bits(all_A, all_O) if all_A.size else 0.0
            mi_strat = self._stratified_mi(trials, burn_in=burn_in,
                                           min_trials=min_trials_strat, t_cap=t_cap)
            fail_rate = fail_count / self.n_trials

            pred_pool_arr.append(mi_pred)
            steer_pool_arr.append(mi_steer)
            strat_arr.append(mi_strat)
            fail_arr.append(fail_rate)

            if verbose:
                print(f"  τ={int(d):2d}: steer={mi_steer:.4f} "
                      f"pred={mi_pred:.4f} strat={mi_strat:.4f} "
                      f"fail={fail_rate:.2f}")

        tau_arr = np.array(tau_grid)
        steer_arr = np.array(steer_pool_arr)
        pred_arr = np.array(pred_pool_arr)
        strat_arr_np = np.array(strat_arr)
        fail_arr_np = np.array(fail_arr)

        # Agency horizon
        tau_h = None
        for i, (t, s) in enumerate(zip(tau_arr, steer_arr)):
            if s <= self.epsilon:
                tau_h = int(t)
                break

        # Peak stratified MI
        peak_idx = int(np.argmax(strat_arr_np))
        peak_strat = float(strat_arr_np[peak_idx])
        peak_tau = int(tau_arr[peak_idx]) if peak_strat > 0 else None

        config = {
            'a': self.a, 'K': self.K, 'delta_pred': self.delta_pred,
            'n_trials': self.n_trials, 'x_fail': self.x_fail,
            'sigma_proc': self.sigma_proc, 'seed': self.seed,
        }

        return SimulationResult(
            config=config,
            tau_grid=tau_arr,
            mi_pred_pooled=pred_arr,
            mi_steer_pooled=steer_arr,
            mi_pred_strat=strat_arr_np,
            gap=pred_arr - steer_arr,
            fail_rate=fail_arr_np,
            tau_h=tau_h,
            peak_strat_mi=peak_strat,
            peak_strat_tau=peak_tau,
        )

    def parameter_sweep(
        self,
        param: str,
        values: List[float],
        tau_grid: Optional[List[int]] = None,
        verbose: bool = False,
    ) -> List[SimulationResult]:
        """
        Sweep a single parameter, run simulation for each value.

        Parameters
        ----------
        param : str
            Parameter name: 'a', 'K', 'delta_pred', 'x_fail'
        values : list
            Values to sweep.
        tau_grid : list of int, optional
        verbose : bool

        Returns
        -------
        list of SimulationResult

        Example
        -------
            results = sim.parameter_sweep('a', [1.01, 1.02, 1.03, 1.05])
            for r, a in zip(results, [1.01, 1.02, 1.03, 1.05]):
                print(f'a={a}: tau_h={r.tau_h}, peak={r.peak_strat_mi:.3f}')
        """
        results = []
        for v in values:
            if not hasattr(self, param):
                raise ValueError(f"Unknown parameter '{param}'")
            orig = getattr(self, param)
            setattr(self, param, v)
            print(f"Running {param}={v}...")
            r = self.run(tau_grid=tau_grid, verbose=verbose)
            results.append(r)
            setattr(self, param, orig)
        return results


def _plot_simulation(result: SimulationResult, save_path=None, show=True):
    """Plot simulation result as leverage gap figure."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("pip install matplotlib")

    cfg = result.config
    fig, ax1 = plt.subplots(figsize=(11, 6))

    ax1.plot(result.tau_grid, result.mi_steer_pooled, 'b--', linewidth=2.5,
             label=r"Pooled steering $I(A_t\,;\,O_{t+\tau+1})$")
    ax1.plot(result.tau_grid, result.mi_pred_pooled, color='darkorange', linewidth=2.5,
             label=r"Pooled prediction $I(D_t\,;\,X_{t+\Delta})$")
    ax1.plot(result.tau_grid, result.mi_pred_strat, 'g-.', linewidth=2.5,
             label="Stratified prediction")

    ax1.fill_between(result.tau_grid, result.mi_steer_pooled, result.mi_pred_pooled,
                     where=(result.mi_pred_pooled > result.mi_steer_pooled),
                     alpha=0.15, color='darkorange', label="Leverage gap")

    if result.tau_h is not None:
        ax1.axvline(result.tau_h, color='gray', linestyle='--', linewidth=1.5,
                    label=f"τ_h={result.tau_h}")

    ax2 = ax1.twinx()
    ax2.plot(result.tau_grid, result.fail_rate, 'k:', linewidth=2, alpha=0.4,
             label="Failure rate")
    ax2.set_ylabel("Failure Probability", color='gray', fontsize=10)
    ax2.set_ylim(-0.05, 1.15)
    ax2.tick_params(axis='y', labelcolor='gray')

    ax1.set_xlabel(r"Latency τ", fontsize=11)
    ax1.set_ylabel("Mutual Information (bits)", fontsize=11)
    ax1.set_title(
        f"FAWP Simulation: a={cfg['a']}, K={cfg['K']}, "
        f"Δ={cfg['delta_pred']}, {cfg['n_trials']} trials\n"
        f"τ_h={result.tau_h} | Peak strat MI={result.peak_strat_mi:.4f} (τ={result.peak_strat_tau})",
        fontsize=10,
    )
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8.5, loc='center right')

    fig.text(0.99, 0.01, 'fawp-index | Clayton (2026) doi:10.5281/zenodo.18673949',
             ha='right', va='bottom', fontsize=7, color='gray', style='italic')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    return fig, ax1


class ControlCliff:
    """
    E5: The Control Cliff — delayed access → reduced coupling → instability.

    Links the OATS agency proxy (MI in bits) to an actual control outcome
    in an unstable AR(1) dynamical system with delayed noisy sensing.

    As delay d increases, the system exhibits a sharp transition:
      - Control cost rises dramatically (log-scale blow-up)
      - Failure rate jumps toward 1
      - MI proxy decreases smoothly

    The cliff is where MI decay becomes control collapse.

    Example
    -------
        from fawp_index.simulate import ControlCliff

        # Load published E5 data
        result = ControlCliff.from_e5_data()
        print(result.summary())
        result.plot()

        # Or run fresh with custom parameters
        cc = ControlCliff(a=1.02, K=0.8, n_trials=50)
        result = cc.run(delays=list(range(0, 50, 2)))
        result.plot()
    """

    def __init__(
        self,
        a: float = FLAGSHIP_A,
        K: float = FLAGSHIP_K,
        sigma_proc: float = FLAGSHIP_SIGMA_PROC,
        sigma0_sq: float = 0.01,
        alpha: float = 0.001,
        P: float = 10.0,
        u_max: float = 10.0,
        n_steps: int = 1000,
        n_trials: int = 50,
        x_fail: float = 1000.0,
        cost_cap: float = 10000.0,
        fail_rate_threshold: float = 0.5,
        seed: int = 123,
    ):
        self.a = a
        self.K = K
        self.sigma_proc = sigma_proc
        self.sigma0_sq = sigma0_sq
        self.alpha = alpha
        self.P = P
        self.u_max = u_max
        self.n_steps = n_steps
        self.n_trials = n_trials
        self.x_fail = x_fail
        self.cost_cap = cost_cap
        self.fail_rate_threshold = fail_rate_threshold
        self.seed = seed

    @classmethod
    def from_e5_data(cls) -> "ControlCliffResult":
        """Load the published E5 data directly."""
        import pandas as pd
        from fawp_index.data import E5_CONTROL_CLIFF
        df = pd.read_csv(E5_CONTROL_CLIFF)
        cliff_idx = (df['failure_rate'] >= 0.5).idxmax()
        cliff_delay = int(df.loc[cliff_idx, 'delay_steps']) if df['failure_rate'].max() >= 0.5 else None
        cliff_mi = float(df.loc[cliff_idx, 'mi_bits']) if cliff_delay is not None else None
        return ControlCliffResult(
            delays=df['delay_steps'].values,
            mean_cost=df['mean_cost'].values,
            failure_rate=df['failure_rate'].values,
            mi_bits=df['mi_bits'].values,
            ci_low=df['mean_cost_ci95_low'].values,
            ci_high=df['mean_cost_ci95_high'].values,
            cliff_delay=cliff_delay,
            cliff_mi=cliff_mi,
        )

    def run(self, delays: Optional[List[int]] = None, verbose: bool = False) -> "ControlCliffResult":
        """Run E5-style control cliff experiment."""
        if delays is None:
            delays = list(range(0, 50, 2))

        mean_costs, fail_rates, mi_list = [], [], []
        ci_lows, ci_highs = [], []

        for d in delays:
            obs_var = self.sigma0_sq + self.alpha * d
            obs_std = np.sqrt(obs_var)
            mi = 0.5 * np.log2(1.0 + self.P / obs_var)

            costs = []
            fails = 0
            for trial in range(self.n_trials):
                rng = np.random.default_rng(self.seed + d * 1000 + trial)
                x = 0.0
                history = [0.0] * (d + 1)
                cost = 0.0
                for t in range(self.n_steps):
                    obs = history[0] + rng.normal(0, obs_std)
                    u = float(np.clip(-self.K * obs, -self.u_max, self.u_max))
                    x = self.a * x + u + rng.normal(0, self.sigma_proc)
                    cost += min(x**2, self.cost_cap)
                    history = [x] + history[:-1]
                    if abs(x) > self.x_fail:
                        fails += 1
                        break
                costs.append(cost / self.n_steps)

            arr = np.array(costs)
            mean_costs.append(float(arr.mean()))
            fail_rates.append(fails / self.n_trials)
            mi_list.append(float(mi))
            se = arr.std() / np.sqrt(len(arr))
            ci_lows.append(float(arr.mean() - 1.96 * se))
            ci_highs.append(float(arr.mean() + 1.96 * se))

            if verbose:
                print(f"  d={d}: cost={mean_costs[-1]:.1f} fail={fail_rates[-1]:.2f} MI={mi:.3f}")

        delays_arr = np.array(delays)
        fr_arr = np.array(fail_rates)
        cliff_idx = np.where(fr_arr >= self.fail_rate_threshold)[0]
        cliff_delay = int(delays_arr[cliff_idx[0]]) if len(cliff_idx) else None
        cliff_mi = float(mi_list[cliff_idx[0]]) if len(cliff_idx) else None

        return ControlCliffResult(
            delays=delays_arr,
            mean_cost=np.array(mean_costs),
            failure_rate=fr_arr,
            mi_bits=np.array(mi_list),
            ci_low=np.array(ci_lows),
            ci_high=np.array(ci_highs),
            cliff_delay=cliff_delay,
            cliff_mi=cliff_mi,
        )


@dataclass
class ControlCliffResult:
    """Result from ControlCliff."""
    delays: np.ndarray
    mean_cost: np.ndarray
    failure_rate: np.ndarray
    mi_bits: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    cliff_delay: Optional[int]
    cliff_mi: Optional[float]

    def summary(self) -> str:
        lines = [
            "=" * 55,
            "E5: Control Cliff — MI Decay → Stability Failure",
            "=" * 55,
            f"Delay range:   {int(self.delays[0])} — {int(self.delays[-1])} steps",
            f"Cliff delay:   d={self.cliff_delay}",
            f"MI at cliff:   {self.cliff_mi:.4f} bits" if self.cliff_mi else "MI at cliff: N/A",
            f"Max fail rate: {self.failure_rate.max():.2f}",
            "=" * 55,
        ]
        return "\n".join(lines)

    def plot(self, save_path=None, show=True):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("pip install matplotlib")

        fig, ax1 = plt.subplots(figsize=(10, 5))

        # Cost on log scale
        ax1.semilogy(self.delays, self.mean_cost, 'darkorange', linewidth=2.5, label='Mean control cost')
        ax1.fill_between(self.delays, self.ci_low.clip(0.01), self.ci_high,
                         alpha=0.15, color='darkorange')
        ax1.set_xlabel("Delay d (steps)", fontsize=11)
        ax1.set_ylabel("Mean control cost (log scale)", fontsize=10, color='darkorange')
        ax1.tick_params(axis='y', labelcolor='darkorange')

        ax2 = ax1.twinx()
        ax2.plot(self.delays, self.mi_bits, 'b--', linewidth=2.2, label='MI proxy (bits)')
        ax2.plot(self.delays, self.failure_rate, 'k:', linewidth=2, alpha=0.6, label='Failure rate')
        ax2.set_ylabel("MI (bits) / Failure rate", fontsize=10)
        ax2.set_ylim(-0.1, max(self.mi_bits.max() * 1.1, 1.1))

        if self.cliff_delay is not None:
            ax1.axvline(self.cliff_delay, color='red', linestyle='--', linewidth=1.8,
                        label=f"Cliff d={self.cliff_delay}")
            if self.cliff_mi is not None:
                ax2.axhline(self.cliff_mi, color='steelblue', linestyle=':', linewidth=1.2,
                            label=f"MI at cliff = {self.cliff_mi:.3f}")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8.5)

        ax1.set_title("E5: Control Cliff\n"
                      "MI decreases smoothly; stability collapses abruptly", fontsize=10)
        ax1.grid(True, alpha=0.3)
        fig.text(0.99, 0.01, 'fawp-index | Clayton (2026) doi:10.5281/zenodo.18663547',
                 ha='right', va='bottom', fontsize=7, color='gray', style='italic')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        return fig, ax1
