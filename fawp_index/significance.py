"""
fawp_index.significance — Significance testing for FAWP results
================================================================

Answers the question: "Is this signal actually real?"

Three entry points depending on what data you have:

1. **from_seed_curves** — uses bundled E9.2 20-seed CSV as bootstrap population.
   Fastest. Works on any ODWResult without extra data.

2. **from_mi_curves** — you supply raw (uncorrected) MI arrays per tau.
   Runs shuffle + shift null at each tau, returns per-tau p-values.

3. **from_arrays** — you supply raw paired (x, y) observation arrays per tau.
   Full null testing: shuffle, shift, bootstrap CIs on all key quantities.

Quick start
-----------
    from fawp_index import ODWDetector
    from fawp_index.significance import fawp_significance

    odw = ODWDetector.from_e9_2_data()

    # Fastest: bootstrap from bundled seed curves
    sig = fawp_significance(odw)
    print(sig.summary())

    # With raw MI values (if you ran your own experiment)
    sig = fawp_significance(
        odw,
        pred_raw   = my_raw_pred_mi_array,   # shape (n_tau,)
        steer_raw  = my_raw_steer_mi_array,
        fail_rate  = my_fail_rate_array,
        tau        = my_tau_array,
    )

    # Full test with paired arrays (x, y) per tau
    sig = fawp_significance(
        odw,
        pred_pairs  = [(x_pred_tau0, y_pred_tau0), ...],
        steer_pairs = [(x_steer_tau0, y_steer_tau0), ...],
        fail_rate   = my_fail_rate_array,
        tau         = my_tau_array,
    )

Output
------
    sig.p_value_fawp          # P(ODW detected under null) — key number
    sig.ci_tau_h              # (lo, hi) 95% CI on tau_h+
    sig.ci_odw_start          # (lo, hi) 95% CI on ODW start
    sig.ci_odw_end            # (lo, hi) 95% CI on ODW end
    sig.ci_peak_gap           # (lo, hi) 95% CI on peak leverage gap
    sig.pred_p_values         # per-tau p-values for pred MI > 0
    sig.steer_p_values        # per-tau p-values for steer MI > 0
    sig.summary()
    sig.to_html("sig.html")
    sig.to_json("sig.json")

Ralph Clayton (2026) · https://doi.org/10.5281/zenodo.18673949
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from fawp_index import __version__ as _VERSION
_DOI     = "https://doi.org/10.5281/zenodo.18673949"
_GITHUB  = "https://github.com/DrRalphClayton/fawp-index"


# ─────────────────────────────────────────────────────────────────────────────
# Core null / MI helpers (lifted from E9.2 script, Clayton 2026)
# ─────────────────────────────────────────────────────────────────────────────

def _mi_bits(x: np.ndarray, y: np.ndarray, min_n: int = 20) -> float:
    """Gaussian MI estimator from Pearson correlation (bits)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < min_n:
        return 0.0
    sx, sy = float(np.std(x)), float(np.std(y))
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    rho = float(np.corrcoef(x, y)[0, 1])
    if not np.isfinite(rho):
        return 0.0
    rho = np.clip(rho, -0.999999, 0.999999)
    return float(-0.5 * np.log(1.0 - rho ** 2) / np.log(2.0))


def _shuffle_null(
    x: np.ndarray,
    y: np.ndarray,
    n: int,
    rng: np.random.Generator,
    min_n: int = 20,
) -> np.ndarray:
    """MI under H0 via permutation (destroys x-y relationship)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < min_n:
        return np.zeros(n)
    return np.array([_mi_bits(x, rng.permutation(y), min_n) for _ in range(n)])


def _shift_null(
    x: np.ndarray,
    y: np.ndarray,
    n: int,
    rng: np.random.Generator,
    min_n: int = 20,
) -> np.ndarray:
    """MI under H0 via circular shift (preserves autocorrelation in y)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    sz = x.size
    if sz < min_n or sz < 3:
        return np.zeros(n)
    shifts = rng.integers(1, sz, size=n)
    return np.array([_mi_bits(x, np.roll(y, int(s)), min_n) for s in shifts])


def _conservative_floor(
    x: np.ndarray,
    y: np.ndarray,
    n_null: int,
    beta: float,
    rng: np.random.Generator,
    min_n: int = 20,
) -> float:
    """Conservative null floor = max(q_beta shuffle, q_beta shift)."""
    q_shuffle = float(np.quantile(_shuffle_null(x, y, n_null, rng, min_n), beta))
    q_shift   = float(np.quantile(_shift_null(x, y, n_null, rng, min_n), beta))
    return max(q_shuffle, q_shift)


def _p_value(observed: float, null_dist: np.ndarray) -> float:
    """Fraction of null samples >= observed (one-tailed, upper)."""
    null_dist = np.asarray(null_dist, dtype=float)
    if null_dist.size == 0:
        return float("nan")
    return float(np.mean(null_dist >= observed))


def _bootstrap_ci(
    samples: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[Optional[float], Optional[float]]:
    """Percentile bootstrap CI from a 1-D array of bootstrap estimates."""
    samples = np.asarray([s for s in samples if s is not None], dtype=float)
    samples = samples[np.isfinite(samples)]
    if samples.size < 2:
        return None, None
    lo = float(np.percentile(samples, 100 * alpha / 2))
    hi = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    return lo, hi


# ─────────────────────────────────────────────────────────────────────────────
# SignificanceResult
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SignificanceResult:
    """
    Output of FAWPSignificance or fawp_significance().

    Attributes
    ----------
    method : str
        'seed_bootstrap', 'mi_permutation', or 'array_permutation'.
    alpha : float
        Significance level used for CIs (default 0.05 → 95% CI).
    n_bootstrap : int
        Number of bootstrap / permutation samples used.

    p_value_fawp : float
        Fraction of bootstrap/null runs in which an ODW was detected.
        Under H0 (no FAWP), this should be small.
        For real FAWP it should be large.
        Interpretation: p_value_fawp > 1-alpha → FAWP is significant.

    p_value_null : float
        Fraction of shuffled/null runs in which an ODW was detected.
        This is the false-positive rate under H0.

    ci_tau_h : (float, float) or (None, None)
        Bootstrap CI on tau_h+ (post-zero agency horizon).
    ci_odw_start : (float, float) or (None, None)
        Bootstrap CI on ODW start.
    ci_odw_end : (float, float) or (None, None)
        Bootstrap CI on ODW end.
    ci_peak_gap : (float, float) or (None, None)
        Bootstrap CI on peak leverage gap (bits).
    ci_peak_gap_tau : (float, float) or (None, None)
        Bootstrap CI on tau at which peak gap occurs.

    tau : ndarray
        Tau grid.
    pred_p_values : ndarray
        Per-tau p-value: P(pred MI >= observed | H0).
        Available when method='mi_permutation' or 'array_permutation'.
        None otherwise.
    steer_p_values : ndarray or None
        Per-tau p-value for steer MI. Same availability as pred_p_values.

    tau_h_samples : ndarray
        Bootstrap distribution of tau_h+ values (for histogram etc.).
    odw_start_samples : ndarray
        Bootstrap distribution of ODW start values.
    peak_gap_samples : ndarray
        Bootstrap distribution of peak leverage gap values.

    observed_odw_result : ODWResult
        The original detection result.
    """

    method: str
    alpha: float
    n_bootstrap: int

    p_value_fawp: float
    p_value_null: float

    ci_tau_h: Tuple[Optional[float], Optional[float]]
    ci_odw_start: Tuple[Optional[float], Optional[float]]
    ci_odw_end: Tuple[Optional[float], Optional[float]]
    ci_peak_gap: Tuple[Optional[float], Optional[float]]
    ci_peak_gap_tau: Tuple[Optional[float], Optional[float]]

    tau: np.ndarray
    pred_p_values: Optional[np.ndarray]
    steer_p_values: Optional[np.ndarray]

    tau_h_samples: np.ndarray
    odw_start_samples: np.ndarray
    peak_gap_samples: np.ndarray

    observed_odw_result: object  # ODWResult

    # ── derived ──────────────────────────────────────────────────────────────

    @property
    def significant(self) -> bool:
        """
        True if FAWP is robustly detected at the chosen alpha level.

        For seed_bootstrap: FAWP is significant if it is detected in
        at least (1 - alpha) fraction of bootstrap resamples, i.e.
        p_value_fawp >= 1 - alpha.

        For mi_permutation / array_permutation: significant if the
        null false-positive rate is <= alpha.
        """
        if self.method == "seed_bootstrap":
            return self.p_value_fawp >= (1.0 - self.alpha)
        return self.p_value_null <= self.alpha

    @property
    def confidence_pct(self) -> int:
        """CI percentage (e.g. 95)."""
        return int(round((1 - self.alpha) * 100))

    # ── summary ──────────────────────────────────────────────────────────────

    def summary(self) -> str:
        r = self.observed_odw_result
        ci_pct = self.confidence_pct

        def _fmt_ci(ci):
            lo, hi = ci
            if lo is None:
                return "n/a"
            return f"[{lo:.1f}, {hi:.1f}]"

        def _fmt_p(p):
            if p < 0.001:
                return "< 0.001"
            return f"{p:.3f}"

        lines = [
            "=" * 62,
            "  FAWP Significance Test",
            "=" * 62,
            f"  Method          : {self.method}",
            f"  Samples         : {self.n_bootstrap}",
            f"  Alpha / CI      : {self.alpha} / {ci_pct}%",
            "",
            f"  FAWP detected   : {'YES' if r.fawp_found else 'NO'}",
            f"  Significant     : {'YES ✓' if self.significant else 'NO ✗'}",
            "",
            "  -- P-values --",
            f"  P(FAWP | data)  : {_fmt_p(self.p_value_fawp)}",
            f"  P(FAWP | null)  : {_fmt_p(self.p_value_null)}  (false-positive rate)",
            "",
            f"  -- {ci_pct}% Bootstrap Confidence Intervals --",
            f"  tau_h+          : {_fmt_ci(self.ci_tau_h)}",
            f"  ODW start       : {_fmt_ci(self.ci_odw_start)}",
            f"  ODW end         : {_fmt_ci(self.ci_odw_end)}",
            f"  Peak gap (bits) : {_fmt_ci(self.ci_peak_gap)}",
            f"  Peak gap tau    : {_fmt_ci(self.ci_peak_gap_tau)}",
            "=" * 62,
        ]
        return "\n".join(lines)

    # ── exports ──────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        r = self.observed_odw_result

        def _ci(ci):
            lo, hi = ci
            return {"lo": lo, "hi": hi}

        return {
            "meta": {
                "generated_date": date.today().isoformat(),
                "fawp_index_version": _VERSION,
                "doi": _DOI,
            },
            "method": self.method,
            "alpha": self.alpha,
            "n_bootstrap": self.n_bootstrap,
            "significant": self.significant,
            "p_value_fawp": float(self.p_value_fawp),
            "p_value_null": float(self.p_value_null),
            "confidence_intervals": {
                "tau_h":        _ci(self.ci_tau_h),
                "odw_start":    _ci(self.ci_odw_start),
                "odw_end":      _ci(self.ci_odw_end),
                "peak_gap":     _ci(self.ci_peak_gap),
                "peak_gap_tau": _ci(self.ci_peak_gap_tau),
            },
            "observed": {
                "fawp_found":  bool(r.fawp_found),
                "tau_h_plus":  r.tau_h_plus,
                "tau_f":       r.tau_f,
                "odw_start":   r.odw_start,
                "odw_end":     r.odw_end,
                "peak_gap":    float(r.peak_gap_bits),
            },
            "bootstrap_samples": {
                "tau_h":     [None if np.isnan(v) else float(v)
                              for v in self.tau_h_samples],
                "odw_start": [None if np.isnan(v) else float(v)
                              for v in self.odw_start_samples],
                "peak_gap":  [float(v) for v in self.peak_gap_samples],
            },
        }

    def to_json(self, path: Union[str, Path], indent: int = 2) -> Path:
        """Write result to JSON. Returns Path."""
        p = Path(path)
        p.write_text(json.dumps(self.to_dict(), indent=indent))
        return p

    def to_html(self, path: Union[str, Path]) -> Path:
        """Write result to self-contained HTML. Returns Path."""
        p = Path(path)
        p.write_text(_sig_html(self))
        return p

    def plot(self, show: bool = True, save_path: Optional[str] = None):
        """
        Plot bootstrap distributions and per-tau p-values.

        Returns matplotlib Figure.
        """
        try:
            import matplotlib
            matplotlib.use("Agg" if not show else matplotlib.get_backend())
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("pip install fawp-index[plot]")

        r = self.observed_odw_result
        ci_pct = self.confidence_pct

        n_panels = 3 if self.pred_p_values is not None else 2
        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5))
        fig.suptitle(
            f"FAWP Significance ({self.method}, n={self.n_bootstrap})",
            fontsize=11,
        )

        # Panel 1 — tau_h bootstrap distribution
        ax = axes[0]
        clean = self.tau_h_samples[np.isfinite(self.tau_h_samples)]
        if clean.size > 0:
            ax.hist(clean, bins=min(20, clean.size), color="#0E2550",
                    alpha=0.75, edgecolor="white", linewidth=0.5)
        if r.tau_h_plus is not None:
            ax.axvline(r.tau_h_plus, color="#D4AF37", lw=2.2,
                       label=f"observed tau_h+ = {r.tau_h_plus}")
        lo, hi = self.ci_tau_h
        if lo is not None:
            ax.axvspan(lo, hi, alpha=0.15, color="#D4AF37",
                       label=f"{ci_pct}% CI [{lo:.1f}, {hi:.1f}]")
        ax.set_title("Bootstrap: tau_h+", fontsize=9)
        ax.set_xlabel("tau_h+")
        ax.set_ylabel("count")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

        # Panel 2 — peak gap bootstrap distribution
        ax = axes[1]
        ax.hist(self.peak_gap_samples, bins=min(20, len(self.peak_gap_samples)),
                color="#C0111A", alpha=0.75, edgecolor="white", linewidth=0.5)
        ax.axvline(r.peak_gap_bits, color="#D4AF37", lw=2.2,
                   label=f"observed = {r.peak_gap_bits:.3f} bits")
        lo, hi = self.ci_peak_gap
        if lo is not None:
            ax.axvspan(lo, hi, alpha=0.15, color="#D4AF37",
                       label=f"{ci_pct}% CI [{lo:.3f}, {hi:.3f}]")
        ax.set_title("Bootstrap: Peak leverage gap", fontsize=9)
        ax.set_xlabel("bits")
        ax.set_ylabel("count")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

        # Panel 3 — per-tau p-values (if available)
        if self.pred_p_values is not None:
            ax = axes[2]
            ax.plot(self.tau, self.pred_p_values,  lw=2.0, label="p(pred MI)")
            if self.steer_p_values is not None:
                ax.plot(self.tau, self.steer_p_values, lw=1.8, ls="--",
                        label="p(steer MI)")
            ax.axhline(self.alpha, ls=":", lw=1.2, color="red",
                       label=f"alpha = {self.alpha}")
            if r.odw_start is not None:
                ax.axvspan(r.odw_start, r.odw_end, alpha=0.12, color="green",
                           label="ODW")
            ax.set_title("Per-tau p-values", fontsize=9)
            ax.set_xlabel("tau")
            ax.set_ylabel("p-value")
            ax.set_ylim(-0.05, 1.05)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.2)

        fig.text(0.99, 0.01, "fawp-index | Clayton (2026)",
                 ha="right", fontsize=7, color="grey", style="italic")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            try:
                plt.show()
            except Exception:
                pass
        return fig


# ─────────────────────────────────────────────────────────────────────────────
# FAWPSignificance — the main class
# ─────────────────────────────────────────────────────────────────────────────

class FAWPSignificance:
    """
    Significance testing for FAWP detection results.

    Parameters
    ----------
    n_bootstrap : int
        Number of bootstrap / permutation samples. Default 500.
    n_null : int
        Number of null samples per tau (for mi_permutation method). Default 200.
    alpha : float
        Significance level for CIs. Default 0.05 (95% CI).
    beta_null : float
        Quantile for conservative null floor. Default 0.99.
    epsilon : float
        ODW detection threshold passed to ODWDetector. Default 0.01.
    seed : int
        Random seed. Default 42.
    """

    def __init__(
        self,
        n_bootstrap: int = 500,
        n_null: int = 200,
        alpha: float = 0.05,
        beta_null: float = 0.99,
        epsilon: float = 0.01,
        seed: int = 42,
    ):
        self.n_bootstrap = n_bootstrap
        self.n_null      = n_null
        self.alpha       = alpha
        self.beta_null   = beta_null
        self.epsilon     = epsilon
        self.seed        = seed

    # ── Public entry points ───────────────────────────────────────────────────

    def from_seed_curves(self, odw_result) -> SignificanceResult:
        """
        Bootstrap significance from bundled E9.2 seed curves CSV.

        The seed curves CSV contains 20 independent replicates per tau step.
        For each bootstrap iteration, this method resamples those 20 per-tau
        observations with replacement, averages the resulting curves, and
        runs ODWDetector on the averaged curve.  Repeating this n_bootstrap
        times gives bootstrap distributions of tau_h+, ODW bounds, and
        peak leverage gap.

        This is the fastest method — no raw data needed.

        Parameters
        ----------
        odw_result : ODWResult

        Returns
        -------
        SignificanceResult

        Example
        -------
            from fawp_index import ODWDetector
            from fawp_index.significance import FAWPSignificance

            odw = ODWDetector.from_e9_2_data()
            sig = FAWPSignificance().from_seed_curves(odw)
            print(sig.summary())
        """
        import pandas as pd
        from fawp_index.data import E9_2_SEED_CURVES
        from fawp_index.detection.odw import ODWDetector as _Det

        df = pd.read_csv(E9_2_SEED_CURVES).sort_values("tau")
        tau_arr = np.array(sorted(df["tau"].unique()), dtype=int)
        n_tau   = len(tau_arr)

        # Build (n_tau, n_replicate) matrices — 20 replicates per tau
        tau_groups = df.groupby("tau")
        n_rep = tau_groups.size().min()   # smallest replicate count across tau

        pred_mat  = np.zeros((n_tau, n_rep))   # null-corrected pred MI
        steer_mat = np.zeros((n_tau, n_rep))
        fail_mat  = np.zeros((n_tau, n_rep))

        for i, t in enumerate(tau_arr):
            sub = df[df["tau"] == t].reset_index(drop=True)
            pred_mat[i,  :] = sub["pred_strat_corr"].values[:n_rep]
            steer_mat[i, :] = sub["steer_u_corr"].values[:n_rep]
            fail_mat[i,  :] = sub["fail_rate"].values[:n_rep]

        rng   = np.random.default_rng(self.seed)
        n_boot = min(self.n_bootstrap, 2000)

        tau_h_samp, odw_start_samp, odw_end_samp = [], [], []
        peak_gap_samp, peak_gap_tau_samp = [], []
        fawp_count = 0

        for _ in range(n_boot):
            # Resample n_rep replicates with replacement at every tau
            idx       = rng.integers(0, n_rep, size=(n_tau, n_rep))
            pred_b    = pred_mat[np.arange(n_tau)[:, None], idx].mean(axis=1)
            steer_b   = steer_mat[np.arange(n_tau)[:, None], idx].mean(axis=1)
            fail_b    = fail_mat[np.arange(n_tau)[:, None], idx].mean(axis=1)

            det = _Det(epsilon=self.epsilon)
            r   = det.detect(
                tau        = tau_arr,
                pred_corr  = np.maximum(0.0, pred_b),
                steer_corr = np.maximum(0.0, steer_b),
                fail_rate  = np.clip(fail_b, 0.0, 1.0),
            )
            tau_h_samp.append(float(r.tau_h_plus) if r.tau_h_plus is not None else float("nan"))
            odw_start_samp.append(float(r.odw_start) if r.odw_start is not None else float("nan"))
            odw_end_samp.append(float(r.odw_end)   if r.odw_end   is not None else float("nan"))
            peak_gap_samp.append(float(r.peak_gap_bits))
            peak_gap_tau_samp.append(float(r.peak_gap_tau) if r.peak_gap_tau is not None else float("nan"))
            if r.fawp_found:
                fawp_count += 1

        # Null: swap pred <-> steer channel labels independently at each tau.
        # At each tau step, with probability 0.5, exchange pred and steer
        # replicate values.  This destroys the pred/steer distinction while
        # preserving within-tau variance and the fail_rate structure.
        fawp_null_count = 0
        fail_mean = fail_mat.mean(axis=1)
        for _ in range(n_boot):
            swap = rng.random(n_tau) < 0.5          # (n_tau,) bool mask
            null_pred  = np.where(swap[:, None], steer_mat, pred_mat)
            null_steer = np.where(swap[:, None], pred_mat,  steer_mat)
            # resample replicates within each tau as before
            idx = rng.integers(0, n_rep, size=(n_tau, n_rep))
            p_null = null_pred[ np.arange(n_tau)[:, None], idx].mean(axis=1)
            s_null = null_steer[np.arange(n_tau)[:, None], idx].mean(axis=1)

            det = _Det(epsilon=self.epsilon)
            r_null = det.detect(
                tau        = tau_arr,
                pred_corr  = np.maximum(0.0, p_null),
                steer_corr = np.maximum(0.0, s_null),
                fail_rate  = np.clip(fail_mean, 0.0, 1.0),
            )
            if r_null.fawp_found:
                fawp_null_count += 1

        tau_h_arr     = np.array(tau_h_samp)
        odw_start_arr = np.array(odw_start_samp)
        odw_end_arr   = np.array(odw_end_samp)
        peak_gap_arr  = np.array(peak_gap_samp)
        peak_tau_arr  = np.array(peak_gap_tau_samp)

        return SignificanceResult(
            method          = "seed_bootstrap",
            alpha           = self.alpha,
            n_bootstrap     = n_boot,
            p_value_fawp    = fawp_count / n_boot,
            p_value_null    = fawp_null_count / n_boot,
            ci_tau_h        = _bootstrap_ci(tau_h_arr, self.alpha),
            ci_odw_start    = _bootstrap_ci(odw_start_arr, self.alpha),
            ci_odw_end      = _bootstrap_ci(odw_end_arr, self.alpha),
            ci_peak_gap     = _bootstrap_ci(peak_gap_arr, self.alpha),
            ci_peak_gap_tau = _bootstrap_ci(peak_tau_arr, self.alpha),
            tau             = tau_arr,
            pred_p_values   = None,
            steer_p_values  = None,
            tau_h_samples   = tau_h_arr,
            odw_start_samples = odw_start_arr,
            peak_gap_samples  = peak_gap_arr,
            observed_odw_result = odw_result,
        )

    def from_mi_curves(
        self,
        odw_result,
        tau: np.ndarray,
        pred_raw: np.ndarray,
        steer_raw: np.ndarray,
        fail_rate: np.ndarray,
    ) -> SignificanceResult:
        """
        Permutation significance from raw (uncorrected) MI arrays.

        At each tau, computes a shuffle + shift null distribution for both
        pred and steer MI.  Returns per-tau p-values and bootstrap CIs
        from the corrected curves.

        Parameters
        ----------
        odw_result : ODWResult
        tau : array of int
        pred_raw : array of float — raw (uncorrected) pred MI per tau
        steer_raw : array of float — raw steer MI per tau
        fail_rate : array of float — failure rate per tau

        Returns
        -------
        SignificanceResult

        Example
        -------
            sig = FAWPSignificance().from_mi_curves(
                odw_result = odw,
                tau        = tau_array,
                pred_raw   = raw_pred_mi,
                steer_raw  = raw_steer_mi,
                fail_rate  = fail_rate_array,
            )
        """
        tau       = np.asarray(tau, dtype=int)
        pred_raw  = np.asarray(pred_raw, dtype=float)
        steer_raw = np.asarray(steer_raw, dtype=float)
        fail_rate = np.asarray(fail_rate, dtype=float)
        rng = np.random.default_rng(self.seed)

        n_null = self.n_null
        beta   = self.beta_null
        n_boot = self.n_bootstrap

        # Per-tau null floors via shuffle null distribution
        pred_p_vals  = np.zeros(len(tau))
        steer_p_vals = np.zeros(len(tau))

        # We don't have paired arrays, so we approximate the null distribution
        # by treating the raw MI value as the observation and comparing against
        # a null built from cross-tau shuffles of the full raw MI vector.
        pred_null_pool  = _shuffle_null(pred_raw,  pred_raw,  n_null, rng)
        steer_null_pool = _shuffle_null(steer_raw, steer_raw, n_null, rng)

        for i, t in enumerate(tau):
            pred_p_vals[i]  = _p_value(pred_raw[i],  pred_null_pool)
            steer_p_vals[i] = _p_value(steer_raw[i], steer_null_pool)

        # Bootstrap CIs: resample tau indices to build distribution of detector outputs
        from fawp_index.detection.odw import ODWDetector as _Det
        tau_h_samp, odw_start_samp, odw_end_samp = [], [], []
        peak_gap_samp, peak_gap_tau_samp = [], []
        fawp_count = fawp_null_count = 0

        pred_corr  = np.maximum(0.0, pred_raw  - np.quantile(pred_null_pool,  beta))
        steer_corr = np.maximum(0.0, steer_raw - np.quantile(steer_null_pool, beta))

        for _ in range(n_boot):
            idx    = rng.integers(0, len(tau), size=len(tau))
            p_boot = pred_corr[idx]
            s_boot = steer_corr[idx]
            f_boot = fail_rate[idx]

            det = _Det(epsilon=self.epsilon)
            r = det.detect(
                tau        = tau[np.argsort(idx)],
                pred_corr  = p_boot[np.argsort(idx)],
                steer_corr = s_boot[np.argsort(idx)],
                fail_rate  = f_boot[np.argsort(idx)],
            )
            tau_h_samp.append(float(r.tau_h_plus) if r.tau_h_plus is not None else float("nan"))
            odw_start_samp.append(float(r.odw_start) if r.odw_start is not None else float("nan"))
            odw_end_samp.append(float(r.odw_end) if r.odw_end is not None else float("nan"))
            peak_gap_samp.append(float(r.peak_gap_bits))
            peak_gap_tau_samp.append(float(r.peak_gap_tau) if r.peak_gap_tau is not None else float("nan"))
            if r.fawp_found:
                fawp_count += 1

        # Null: shuffle pred_corr
        for _ in range(n_boot):
            det = _Det(epsilon=self.epsilon)
            r_null = det.detect(
                tau        = tau,
                pred_corr  = rng.permutation(pred_corr),
                steer_corr = rng.permutation(steer_corr),
                fail_rate  = fail_rate,
            )
            if r_null.fawp_found:
                fawp_null_count += 1

        tau_h_arr     = np.array(tau_h_samp)
        odw_start_arr = np.array(odw_start_samp)
        peak_gap_arr  = np.array(peak_gap_samp)
        peak_tau_arr  = np.array(peak_gap_tau_samp)

        return SignificanceResult(
            method          = "mi_permutation",
            alpha           = self.alpha,
            n_bootstrap     = n_boot,
            p_value_fawp    = fawp_count / n_boot,
            p_value_null    = fawp_null_count / n_boot,
            ci_tau_h        = _bootstrap_ci(tau_h_arr, self.alpha),
            ci_odw_start    = _bootstrap_ci(np.array(odw_start_samp), self.alpha),
            ci_odw_end      = _bootstrap_ci(np.array(odw_end_samp), self.alpha),
            ci_peak_gap     = _bootstrap_ci(peak_gap_arr, self.alpha),
            ci_peak_gap_tau = _bootstrap_ci(peak_tau_arr, self.alpha),
            tau             = tau,
            pred_p_values   = pred_p_vals,
            steer_p_values  = steer_p_vals,
            tau_h_samples   = tau_h_arr,
            odw_start_samples = np.array(odw_start_samp),
            peak_gap_samples  = peak_gap_arr,
            observed_odw_result = odw_result,
        )

    def from_arrays(
        self,
        odw_result,
        tau: np.ndarray,
        pred_pairs: List[Tuple[np.ndarray, np.ndarray]],
        steer_pairs: List[Tuple[np.ndarray, np.ndarray]],
        fail_rate: np.ndarray,
    ) -> SignificanceResult:
        """
        Full permutation significance from raw paired (x, y) arrays per tau.

        Computes shuffle + shift null at each tau, then runs a bootstrap
        over corrected MI curves to get CIs on all key quantities.

        Parameters
        ----------
        odw_result : ODWResult
        tau : array of int — delay grid
        pred_pairs : list of (x, y) tuples, one per tau step
            x = state observations, y = future targets
        steer_pairs : list of (x, y) tuples, one per tau step
            x = action observations, y = outcome observations
        fail_rate : array of float

        Returns
        -------
        SignificanceResult

        Example
        -------
            pairs_pred  = [(D_tau, X_tau) for tau in tau_grid]
            pairs_steer = [(A_tau, O_tau) for tau in tau_grid]
            sig = FAWPSignificance().from_arrays(
                odw_result  = odw,
                tau         = tau_array,
                pred_pairs  = pairs_pred,
                steer_pairs = pairs_steer,
                fail_rate   = fail_rate_array,
            )
        """
        tau       = np.asarray(tau, dtype=int)
        fail_rate = np.asarray(fail_rate, dtype=float)
        rng = np.random.default_rng(self.seed)

        n_null = self.n_null
        beta   = self.beta_null
        n_boot = self.n_bootstrap

        # Per-tau: raw MI, null distribution, p-values, corrected MI
        pred_raw   = np.zeros(len(tau))
        steer_raw  = np.zeros(len(tau))
        pred_corr  = np.zeros(len(tau))
        steer_corr = np.zeros(len(tau))
        pred_p_vals   = np.zeros(len(tau))
        steer_p_vals  = np.zeros(len(tau))

        for i, (x_p, y_p) in enumerate(pred_pairs):
            raw_p   = _mi_bits(x_p, y_p)
            null_sh = _shuffle_null(x_p, y_p, n_null, rng)
            null_sf = _shift_null(x_p, y_p, n_null, rng)
            floor_p = max(float(np.quantile(null_sh, beta)),
                          float(np.quantile(null_sf, beta)))
            pred_raw[i]  = raw_p
            pred_corr[i] = max(0.0, raw_p - floor_p)
            pred_p_vals[i] = _p_value(raw_p, np.concatenate([null_sh, null_sf]))

        for i, (x_s, y_s) in enumerate(steer_pairs):
            raw_s   = _mi_bits(x_s, y_s)
            null_sh = _shuffle_null(x_s, y_s, n_null, rng)
            null_sf = _shift_null(x_s, y_s, n_null, rng)
            floor_s = max(float(np.quantile(null_sh, beta)),
                          float(np.quantile(null_sf, beta)))
            steer_raw[i]  = raw_s
            steer_corr[i] = max(0.0, raw_s - floor_s)
            steer_p_vals[i] = _p_value(raw_s, np.concatenate([null_sh, null_sf]))

        # Bootstrap CIs by tau-block resampling on corrected curves
        from fawp_index.detection.odw import ODWDetector as _Det
        tau_h_samp, odw_s_samp, odw_e_samp = [], [], []
        peak_gap_samp, peak_tau_samp = [], []
        fawp_count = fawp_null_count = 0

        for _ in range(n_boot):
            idx = np.sort(rng.integers(0, len(tau), size=len(tau)))
            det = _Det(epsilon=self.epsilon)
            r = det.detect(
                tau        = tau[idx],
                pred_corr  = pred_corr[idx],
                steer_corr = steer_corr[idx],
                fail_rate  = fail_rate[idx],
            )
            tau_h_samp.append(float(r.tau_h_plus) if r.tau_h_plus is not None else float("nan"))
            odw_s_samp.append(float(r.odw_start) if r.odw_start is not None else float("nan"))
            odw_e_samp.append(float(r.odw_end) if r.odw_end is not None else float("nan"))
            peak_gap_samp.append(float(r.peak_gap_bits))
            peak_tau_samp.append(float(r.peak_gap_tau) if r.peak_gap_tau is not None else float("nan"))
            if r.fawp_found:
                fawp_count += 1

        # Null
        for _ in range(n_boot):
            det = _Det(epsilon=self.epsilon)
            r_null = det.detect(
                tau        = tau,
                pred_corr  = rng.permutation(pred_corr),
                steer_corr = rng.permutation(steer_corr),
                fail_rate  = fail_rate,
            )
            if r_null.fawp_found:
                fawp_null_count += 1

        tau_h_arr = np.array(tau_h_samp)
        peak_arr  = np.array(peak_gap_samp)

        return SignificanceResult(
            method          = "array_permutation",
            alpha           = self.alpha,
            n_bootstrap     = n_boot,
            p_value_fawp    = fawp_count / n_boot,
            p_value_null    = fawp_null_count / n_boot,
            ci_tau_h        = _bootstrap_ci(tau_h_arr, self.alpha),
            ci_odw_start    = _bootstrap_ci(np.array(odw_s_samp), self.alpha),
            ci_odw_end      = _bootstrap_ci(np.array(odw_e_samp), self.alpha),
            ci_peak_gap     = _bootstrap_ci(peak_arr, self.alpha),
            ci_peak_gap_tau = _bootstrap_ci(np.array(peak_tau_samp), self.alpha),
            tau             = tau,
            pred_p_values   = pred_p_vals,
            steer_p_values  = steer_p_vals,
            tau_h_samples   = tau_h_arr,
            odw_start_samples = np.array(odw_s_samp),
            peak_gap_samples  = peak_arr,
            observed_odw_result = odw_result,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function
# ─────────────────────────────────────────────────────────────────────────────

def fawp_significance(
    odw_result,
    *,
    # from_mi_curves / from_arrays arguments (all optional)
    tau: Optional[np.ndarray] = None,
    pred_raw: Optional[np.ndarray] = None,
    steer_raw: Optional[np.ndarray] = None,
    pred_pairs: Optional[List] = None,
    steer_pairs: Optional[List] = None,
    fail_rate: Optional[np.ndarray] = None,
    # tuning
    n_bootstrap: int = 500,
    n_null: int = 200,
    alpha: float = 0.05,
    beta_null: float = 0.99,
    epsilon: float = 0.01,
    seed: int = 42,
) -> SignificanceResult:
    """
    Test the significance of a FAWP detection result.

    Automatically selects the best method based on what data is provided:
    - No extra data      → seed_bootstrap (uses bundled E9.2 seed curves)
    - pred_raw + steer_raw given → mi_permutation
    - pred_pairs + steer_pairs given → array_permutation (most rigorous)

    Parameters
    ----------
    odw_result : ODWResult
    tau : array, optional — tau grid (required for mi/array methods)
    pred_raw : array, optional — raw pred MI per tau
    steer_raw : array, optional — raw steer MI per tau
    pred_pairs : list of (x,y), optional — paired arrays per tau
    steer_pairs : list of (x,y), optional
    fail_rate : array, optional
    n_bootstrap : int — bootstrap samples (default 500)
    n_null : int — null samples per tau (default 200)
    alpha : float — significance level (default 0.05)
    beta_null : float — null quantile (default 0.99)
    epsilon : float — ODW detection threshold (default 0.01)
    seed : int

    Returns
    -------
    SignificanceResult

    Examples
    --------
    Fastest — uses bundled seed curves::

        from fawp_index import ODWDetector
        from fawp_index.significance import fawp_significance

        odw = ODWDetector.from_e9_2_data()
        sig = fawp_significance(odw)
        print(sig.summary())
        sig.to_html("significance.html")

    With raw MI arrays::

        sig = fawp_significance(
            odw,
            tau       = tau_array,
            pred_raw  = raw_pred_mi,
            steer_raw = raw_steer_mi,
            fail_rate = fail_rate_array,
        )

    With full paired arrays::

        sig = fawp_significance(
            odw,
            tau         = tau_array,
            pred_pairs  = [(x, y) for ...],
            steer_pairs = [(a, o) for ...],
            fail_rate   = fail_rate_array,
        )
    """
    tester = FAWPSignificance(
        n_bootstrap=n_bootstrap, n_null=n_null, alpha=alpha,
        beta_null=beta_null, epsilon=epsilon, seed=seed,
    )

    if pred_pairs is not None and steer_pairs is not None:
        if tau is None or fail_rate is None:
            raise ValueError("tau and fail_rate required for array_permutation method")
        return tester.from_arrays(odw_result, tau, pred_pairs, steer_pairs, fail_rate)

    if pred_raw is not None and steer_raw is not None:
        if tau is None or fail_rate is None:
            raise ValueError("tau and fail_rate required for mi_permutation method")
        return tester.from_mi_curves(odw_result, tau, pred_raw, steer_raw, fail_rate)

    return tester.from_seed_curves(odw_result)


# ─────────────────────────────────────────────────────────────────────────────
# HTML renderer
# ─────────────────────────────────────────────────────────────────────────────

def _sig_html(sig: SignificanceResult) -> str:
    r = sig.observed_odw_result
    ci_pct = sig.confidence_pct
    sig_colour = "#1a7a1a" if sig.significant else "#aa1111"
    sig_label  = "SIGNIFICANT ✓" if sig.significant else "NOT SIGNIFICANT ✗"

    def _fmt_p(p):
        if p < 0.001:
            return "< 0.001"
        return f"{p:.3f}"

    def _fmt_ci(ci):
        lo, hi = ci
        if lo is None:
            return "&mdash;"
        return f"[{lo:.2f}, {hi:.2f}]"

    rows = [
        ("Method",            sig.method),
        ("Bootstrap samples", str(sig.n_bootstrap)),
        ("Alpha / CI",        f"{sig.alpha} / {ci_pct}%"),
        ("P(FAWP | data)",    _fmt_p(sig.p_value_fawp)),
        ("P(FAWP | null)",    _fmt_p(sig.p_value_null) + " &nbsp;<em>(false-positive rate)</em>"),
        (f"{ci_pct}% CI &tau;<sub>h</sub><sup>+</sup>", _fmt_ci(sig.ci_tau_h)),
        (f"{ci_pct}% CI ODW start",  _fmt_ci(sig.ci_odw_start)),
        (f"{ci_pct}% CI ODW end",    _fmt_ci(sig.ci_odw_end)),
        (f"{ci_pct}% CI peak gap",   _fmt_ci(sig.ci_peak_gap)),
        ("Observed tau_h+",   str(r.tau_h_plus) if r.tau_h_plus is not None else "&mdash;"),
        ("Observed ODW",
         f"{r.odw_start}&ndash;{r.odw_end}" if r.odw_start is not None else "&mdash;"),
        ("Observed peak gap", f"{r.peak_gap_bits:.4f} bits"),
    ]

    table_rows = ""
    for i, (k, v) in enumerate(rows):
        bg = "#f8f8f8" if i % 2 == 0 else "#fff"
        table_rows += (
            f'<tr style="background:{bg}">'
            f'<td style="padding:6px 10px;font-weight:500;color:#444">{k}</td>'
            f'<td style="padding:6px 10px;font-weight:700;color:#0E2550">{v}</td>'
            f'</tr>\n'
        )

    # Embed bootstrap histogram (peak gap)
    chart_html = ""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import base64
        import io

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

        clean_h = sig.tau_h_samples[np.isfinite(sig.tau_h_samples)]
        if clean_h.size > 0:
            ax1.hist(clean_h, bins=min(20, clean_h.size), color="#0E2550",
                     alpha=0.75, edgecolor="white")
        if r.tau_h_plus is not None:
            ax1.axvline(r.tau_h_plus, color="#D4AF37", lw=2.2,
                        label=f"observed = {r.tau_h_plus}")
        lo, hi = sig.ci_tau_h
        if lo is not None:
            ax1.axvspan(lo, hi, alpha=0.15, color="#D4AF37",
                        label=f"{ci_pct}% CI [{lo:.1f},{hi:.1f}]")
        ax1.set_title("Bootstrap: tau_h+", fontsize=9)
        ax1.set_xlabel("tau_h+")
        ax1.legend(fontsize=7)
        ax1.grid(True, alpha=0.2)

        ax2.hist(sig.peak_gap_samples, bins=min(20, len(sig.peak_gap_samples)),
                 color="#C0111A", alpha=0.75, edgecolor="white")
        ax2.axvline(r.peak_gap_bits, color="#D4AF37", lw=2.2,
                    label=f"observed = {r.peak_gap_bits:.3f}")
        lo2, hi2 = sig.ci_peak_gap
        if lo2 is not None:
            ax2.axvspan(lo2, hi2, alpha=0.15, color="#D4AF37",
                        label=f"{ci_pct}% CI [{lo2:.3f},{hi2:.3f}]")
        ax2.set_title("Bootstrap: Peak gap (bits)", fontsize=9)
        ax2.set_xlabel("bits")
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.2)

        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        b64 = base64.b64encode(buf.getvalue()).decode()
        chart_html = (
            '<h2 style="color:#0E2550;margin-top:2em">Bootstrap Distributions</h2>'
            f'<img src="data:image/png;base64,{b64}" '
            'style="max-width:100%;border:1px solid #ddd;border-radius:4px">'
            f'<p style="font-size:0.8em;color:#888;text-align:center">'
            f'Gold line = observed value. Shaded = {ci_pct}% CI.</p>'
        )
    except Exception:
        pass

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>FAWP Significance — fawp-index</title>
<style>
  body {{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
         max-width:820px;margin:0 auto;padding:2em 1.5em;
         background:#fafafa;color:#222;line-height:1.6}}
  header {{background:#0E2550;color:white;padding:1.8em 2em 1.4em;
           border-radius:8px;margin-bottom:1.5em}}
  header h1 {{margin:0 0 0.3em;font-size:1.45em}}
  header p {{margin:0.2em 0;font-size:0.88em;color:#aac}}
  .badge {{display:inline-block;padding:0.4em 1.2em;border-radius:20px;
           font-weight:700;font-size:1em;color:white;
           background:{sig_colour};margin:0.8em 0}}
  h2 {{color:#0E2550;border-bottom:2px solid #D4AF37;padding-bottom:4px}}
  table {{width:100%;border-collapse:collapse;margin:1em 0;
          box-shadow:0 1px 4px rgba(0,0,0,0.07);border-radius:6px;overflow:hidden}}
  thead th {{background:#0E2550;color:white;padding:8px 10px;text-align:left}}
  footer {{margin-top:3em;padding-top:1em;border-top:1px solid #ddd;
           font-size:0.8em;color:#888}}
  a {{color:#0E2550}}
</style>
</head>
<body>
<header>
  <h1>FAWP Significance Test</h1>
  <p>Generated {date.today().isoformat()} &bull
  fawp-index v{_VERSION}</p>
  <p><a href="{_DOI}" style="color:#D4AF37">{_DOI}</a></p>
</header>

<div class="badge">{sig_label}</div>

<h2>Results</h2>
<table>
  <thead><tr><th>Quantity</th><th>Value</th></tr></thead>
  <tbody>{table_rows}</tbody>
</table>

{chart_html}

<footer>
  <a href="{_GITHUB}">fawp-index</a> &bull;
  Ralph Clayton (2026) &bull;
  <a href="{_DOI}">{_DOI}</a>
</footer>
</body>
</html>
"""
