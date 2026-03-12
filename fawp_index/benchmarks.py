"""
fawp_index.benchmarks — Synthetic Benchmark Suite
==================================================

Five canonical cases that cover the full detection landscape.
Each case has a known expected outcome so you can verify the
detector is working correctly on your installation.

Cases
-----
1. clean_control       — textbook FAWP: steering collapses, prediction survives
2. prediction_only     — prediction exists but no steering channel (stable, no cliff)
3. control_only        — steering exists but no predictive horizon (no leverage gap)
4. noisy_false_positive — noisy stable system designed to trap detectors
5. delayed_collapse    — fast-collapsing unstable system with a narrow ODW

All five run in < 1 second (analytic curves).  If you want the real
simulation behind each case, pass ``simulate=True`` (adds ~10–30 s).

Quick start
-----------
    from fawp_index.benchmarks import run_all

    suite = run_all()
    print(suite.summary())          # pass/fail table
    suite.to_html("bench.html")     # full self-contained report

    # Run one case
    from fawp_index.benchmarks import clean_control
    r = clean_control()
    r.verify()                      # raises BenchmarkFailure if detector is wrong
    r.plot()

    # Run with real simulation instead of analytic curves
    r = clean_control(simulate=True)

Ralph Clayton (2026) · https://doi.org/10.5281/zenodo.18673949
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import List, Optional, Union

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────────────────────────────────────

class BenchmarkFailure(AssertionError):
    """Raised by BenchmarkResult.verify() when the detector gives the wrong answer."""


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    """
    Result for a single benchmark case.

    Attributes
    ----------
    name : str
    description : str
    expected_fawp : bool
        Ground-truth: should FAWP be detected?
    tau : ndarray
    pred_mi : ndarray
        Predictive MI fed to the detector (corrected or analytic).
    steer_mi : ndarray
        Steering MI fed to the detector.
    fail_rate : ndarray
    odw_result : ODWResult
        Output of ODWDetector.detect() on these curves.
    passed : bool
        True if detector agreed with expected_fawp.
    verdict : str
        Human-readable one-liner.
    sim_result : object or None
        SimulationResult if simulate=True, else None.
    """
    name: str
    description: str
    expected_fawp: bool
    tau: np.ndarray
    pred_mi: np.ndarray
    steer_mi: np.ndarray
    fail_rate: np.ndarray
    odw_result: object          # ODWResult
    passed: bool
    verdict: str
    sim_result: Optional[object] = field(default=None, repr=False)

    # ── verification ─────────────────────────────────────────────────────────

    def verify(self) -> "BenchmarkResult":
        """
        Assert the detector gave the correct answer.

        Returns self (chainable).
        Raises BenchmarkFailure on mismatch.

        Example
        -------
            clean_control().verify()   # passes silently
        """
        got      = self.odw_result.fawp_found
        expected = self.expected_fawp
        if got != expected:
            raise BenchmarkFailure(
                f"[{self.name}] Expected FAWP={'YES' if expected else 'NO'}, "
                f"got {'YES' if got else 'NO'}.\n"
                f"ODW: start={self.odw_result.odw_start}, "
                f"end={self.odw_result.odw_end}, "
                f"tau_h+={self.odw_result.tau_h_plus}, "
                f"tau_f={self.odw_result.tau_f}"
            )
        return self

    # ── summary ──────────────────────────────────────────────────────────────

    def summary(self) -> str:
        r = self.odw_result
        ok = "PASS" if self.passed else "FAIL"
        det = "YES" if r.fawp_found else "NO"
        exp = "YES" if self.expected_fawp else "NO"
        odw = (f"tau={r.odw_start}-{r.odw_end}"
               if r.odw_start is not None else "none")
        return (
            f"[{ok}] {self.name}\n"
            f"       Expected FAWP={exp} | Detected={det} | {odw}\n"
            f"       {self.verdict}"
        )

    # ── plot ─────────────────────────────────────────────────────────────────

    def plot(self, show: bool = True, save_path: Optional[str] = None):
        """
        Plot the MI curves and leverage gap for this benchmark case.

        Example
        -------
            clean_control().plot()
        """
        try:
            import matplotlib
            matplotlib.use("Agg" if not show else matplotlib.get_backend())
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("pip install fawp-index[plot]")

        r = self.odw_result
        ok_str = "PASS ✓" if self.passed else "FAIL ✗"
        colour = "#1a7a1a" if self.passed else "#aa1111"

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        fig.suptitle(
            f"Benchmark: {self.name}  [{ok_str}]",
            fontsize=11, color=colour, y=1.01,
        )

        ax1.plot(self.tau, self.pred_mi,  lw=2.2, label="I_pred")
        ax1.plot(self.tau, self.steer_mi, lw=2.0, ls="--", label="I_steer")
        ax1.plot(self.tau, self.fail_rate, lw=1.4, ls=":", color="grey",
                 alpha=0.7, label="Fail rate")
        if r.tau_h_plus is not None:
            ax1.axvline(r.tau_h_plus, ls=":", lw=1.2, color="steelblue",
                        label=f"tau_h+ = {r.tau_h_plus}")
        if r.tau_f is not None:
            ax1.axvline(r.tau_f, ls=":", lw=1.2, color="firebrick",
                        label=f"tau_f = {r.tau_f}")
        if r.odw_start is not None:
            ax1.axvspan(r.odw_start, r.odw_end, alpha=0.15, color="green",
                        label="ODW")
        ax1.set_ylabel("MI (bits) / rate")
        ax1.legend(fontsize=7); ax1.grid(True, alpha=0.25)
        ax1.set_title(self.description, fontsize=8, style="italic")

        gap = self.pred_mi - self.steer_mi
        ax2.plot(self.tau, gap, lw=2.2, color="darkorange", label="Leverage gap")
        ax2.fill_between(self.tau, gap.clip(0), alpha=0.15, color="darkorange")
        ax2.axhline(0, lw=0.8, color="black")
        if r.odw_start is not None:
            ax2.axvspan(r.odw_start, r.odw_end, alpha=0.15, color="green")
        ax2.set_xlabel("Latency tau")
        ax2.set_ylabel("Gap (bits)")
        ax2.legend(fontsize=7); ax2.grid(True, alpha=0.25)

        exp = "YES" if self.expected_fawp else "NO"
        det = "YES" if r.fawp_found else "NO"
        fig.text(0.5, -0.01,
                 f"Expected FAWP={exp}  |  Detected={det}  |  "
                 f"fawp-index v0.7.0 | Clayton (2026)",
                 ha="center", fontsize=7, color="grey", style="italic")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            try:
                plt.show()
            except Exception:
                pass
        return fig

    # ── exports ──────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        r = self.odw_result
        return {
            "name": self.name,
            "description": self.description,
            "expected_fawp": self.expected_fawp,
            "detected_fawp": bool(r.fawp_found),
            "passed": self.passed,
            "verdict": self.verdict,
            "odw": {
                "tau_h_plus": r.tau_h_plus,
                "tau_f": r.tau_f,
                "odw_start": r.odw_start,
                "odw_end": r.odw_end,
                "odw_size": r.odw_size,
                "peak_gap_bits": float(r.peak_gap_bits),
                "mean_lead_to_cliff": (float(r.mean_lead_to_cliff)
                                       if r.mean_lead_to_cliff else None),
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# Suite dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchmarkSuite:
    """
    Collection of benchmark results from run_all().

    Attributes
    ----------
    results : list of BenchmarkResult
    n_passed : int
    n_failed : int
    generated_date : str
    """
    results: List[BenchmarkResult]
    generated_date: str = field(default_factory=lambda: date.today().isoformat())

    @property
    def n_passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def n_failed(self) -> int:
        return len(self.results) - self.n_passed

    def verify_all(self) -> "BenchmarkSuite":
        """
        Run verify() on every case.  Raises BenchmarkFailure on first failure.

        Returns self (chainable).
        """
        for r in self.results:
            r.verify()
        return self

    def summary(self) -> str:
        lines = [
            "=" * 62,
            "  fawp-index Benchmark Suite",
            "=" * 62,
            f"  {'Case':<28} {'Expected':>9} {'Detected':>9} {'Status':>7}",
            "  " + "-" * 56,
        ]
        for r in self.results:
            exp = "FAWP" if r.expected_fawp else "NONE"
            det = "FAWP" if r.odw_result.fawp_found else "NONE"
            ok  = "PASS ✓" if r.passed else "FAIL ✗"
            lines.append(f"  {r.name:<28} {exp:>9} {det:>9} {ok:>7}")
        lines += [
            "  " + "-" * 56,
            f"  Passed: {self.n_passed}/{len(self.results)}",
            "=" * 62,
        ]
        return "\n".join(lines)

    def to_json(self, path: Union[str, Path], indent: int = 2) -> Path:
        """Write full suite results to JSON."""
        data = {
            "generated_date": self.generated_date,
            "fawp_index_version": "0.7.0",
            "n_passed": self.n_passed,
            "n_failed": self.n_failed,
            "cases": [r.to_dict() for r in self.results],
        }
        p = Path(path)
        p.write_text(json.dumps(data, indent=indent))
        return p

    def to_html(self, path: Union[str, Path]) -> Path:
        """Write full suite to a self-contained HTML report."""
        p = Path(path)
        p.write_text(_suite_html(self))
        return p

    def to_pdf(self, path: Union[str, Path], **kwargs) -> Path:
        """Write suite as a PDF report via fawp_index.report."""
        from fawp_index.report import generate_report
        # Build a combined dict of all ODW results keyed by case name
        results_dict = {r.name: r.odw_result for r in self.results}
        return generate_report(
            results_dict,
            path,
            title="fawp-index Benchmark Suite",
            **kwargs,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Analytic curve generators
# ─────────────────────────────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray, centre: float, steepness: float = 1.0) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-steepness * (x - centre)))


def _exp_decay(tau: np.ndarray, peak: float, rate: float) -> np.ndarray:
    return np.maximum(0.0, peak * np.exp(-rate * tau))


def _run_detector(tau, pred, steer, fail, epsilon: float = 0.01):
    """Run ODWDetector on arrays, return ODWResult."""
    from fawp_index.detection.odw import ODWDetector
    det = ODWDetector(epsilon=epsilon)
    return det.detect(
        tau=tau,
        pred_corr=np.maximum(0.0, pred),
        steer_corr=np.maximum(0.0, steer),
        fail_rate=fail,
    )


# ─────────────────────────────────────────────────────────────────────────────
# The five canonical benchmark cases
# ─────────────────────────────────────────────────────────────────────────────

def clean_control(simulate: bool = False, seed: int = 42) -> BenchmarkResult:
    """
    Case 1 — Clean control (FAWP expected).

    Textbook scenario: steering MI decays to zero before the failure cliff;
    predictive MI stays healthy throughout.  A clear Operational Detection
    Window should be found.

    Expected: FAWP detected, ODW exists.

    Parameters
    ----------
    simulate : bool
        If True, run FAWPSimulator instead of analytic curves (slower).
    seed : int
        Random seed for analytic noise (ignored when simulate=True).

    Example
    -------
        from fawp_index.benchmarks import clean_control
        r = clean_control()
        r.verify()
        r.plot()
    """
    name        = "clean_control"
    description = ("Textbook FAWP: steering collapses before the failure cliff, "
                   "prediction survives.  ODW should be clearly detected.")

    if simulate:
        return _simulate_case(
            name=name, description=description, expected_fawp=True,
            sim_kwargs=dict(a=1.02, K=0.8, delta_pred=20, n_trials=40,
                            n_steps=600, seed=seed),
            tau_grid=list(range(0, 32)),
        )

    rng = np.random.default_rng(seed)
    tau = np.arange(0, 36)

    steer = _exp_decay(tau, peak=2.2, rate=0.18) + rng.normal(0, 0.02, len(tau))
    pred  = _exp_decay(tau, peak=1.8, rate=0.04) + rng.normal(0, 0.02, len(tau))
    fail  = _sigmoid(tau.astype(float), centre=28.0, steepness=0.6)

    odw = _run_detector(tau, pred, steer, fail)

    passed  = odw.fawp_found
    verdict = (
        f"ODW tau={odw.odw_start}-{odw.odw_end}, "
        f"tau_h+={odw.tau_h_plus}, tau_f={odw.tau_f}"
        if passed else
        "No ODW found — detector may be misconfigured"
    )

    return BenchmarkResult(
        name=name, description=description, expected_fawp=True,
        tau=tau, pred_mi=np.maximum(0, pred),
        steer_mi=np.maximum(0, steer), fail_rate=fail,
        odw_result=odw, passed=passed, verdict=verdict,
    )


def prediction_only(simulate: bool = False, seed: int = 42) -> BenchmarkResult:
    """
    Case 2 — Prediction without steering (FAWP NOT expected).

    The system is predictable but there is no active control channel —
    the steering MI is zero everywhere.  No leverage gap is possible,
    so no ODW should be detected.

    Expected: FAWP NOT detected.

    Example
    -------
        prediction_only().verify()
    """
    name        = "prediction_only"
    description = ("Controller remains effective at all observed delays: "
                   "steering MI never collapses before the failure cliff.  "
                   "No leverage gap opens — FAWP must NOT be detected.")

    if simulate:
        return _simulate_case(
            name=name, description=description, expected_fawp=False,
            sim_kwargs=dict(a=0.98, K=0.0, delta_pred=20, n_trials=40,
                            n_steps=600, seed=seed),
            tau_grid=list(range(0, 28)),
        )

    rng = np.random.default_rng(seed)
    tau = np.arange(0, 32)

    # Steer decays slowly (rate=0.04) — stays above epsilon throughout the
    # failure window, so tau_h+ is never reached before the cliff.
    # Scenario: controller remains effective at all observed delays.
    steer = _exp_decay(tau, peak=2.5, rate=0.04) + rng.normal(0, 0.01, len(tau))
    pred  = _exp_decay(tau, peak=1.8, rate=0.06) + rng.normal(0, 0.01, len(tau))
    fail  = _sigmoid(tau.astype(float), centre=12.0, steepness=0.8)

    odw = _run_detector(tau, pred, steer, fail)

    passed  = not odw.fawp_found
    verdict = (
        f"Correctly detected: no ODW (tau_h+={odw.tau_h_plus} — "
        "steering never collapses before cliff)"
        if passed else
        f"False positive: ODW={odw.odw_start}-{odw.odw_end}, "
        f"tau_h+={odw.tau_h_plus} tau_f={odw.tau_f} "
        f"— steer threshold too low?"
    )

    return BenchmarkResult(
        name=name, description=description, expected_fawp=False,
        tau=tau, pred_mi=np.maximum(0, pred),
        steer_mi=np.maximum(0, steer), fail_rate=fail,
        odw_result=odw, passed=passed, verdict=verdict,
    )


def control_only(simulate: bool = False, seed: int = 42) -> BenchmarkResult:
    """
    Case 3 — Control without predictive horizon (FAWP NOT expected).

    The controller is active (steering MI is healthy) but the prediction
    horizon is so short that predictive MI is negligible.  Without a
    leverage gap, no ODW is possible.

    Expected: FAWP NOT detected.

    Example
    -------
        control_only().verify()
    """
    name        = "control_only"
    description = ("Active controller (steering MI present) but trivial prediction "
                   "horizon — no leverage gap.  FAWP must NOT be detected.")

    if simulate:
        return _simulate_case(
            name=name, description=description, expected_fawp=False,
            sim_kwargs=dict(a=1.02, K=0.8, delta_pred=1, n_trials=40,
                            n_steps=600, seed=seed),
            tau_grid=list(range(0, 28)),
        )

    rng = np.random.default_rng(seed)
    tau = np.arange(0, 30)

    steer = _exp_decay(tau, peak=2.0, rate=0.18) + rng.normal(0, 0.02, len(tau))
    pred  = rng.normal(0, 0.015, len(tau))                         # no pred horizon
    fail  = _sigmoid(tau.astype(float), centre=22.0, steepness=0.6)

    odw = _run_detector(tau, pred, steer, fail)

    passed  = not odw.fawp_found
    verdict = (
        "Correctly detected: no ODW (prediction MI negligible)"
        if passed else
        f"False positive: ODW={odw.odw_start}-{odw.odw_end} "
        f"— pred MI floor too high?"
    )

    return BenchmarkResult(
        name=name, description=description, expected_fawp=False,
        tau=tau, pred_mi=np.maximum(0, pred),
        steer_mi=np.maximum(0, steer), fail_rate=fail,
        odw_result=odw, passed=passed, verdict=verdict,
    )


def noisy_false_positive(simulate: bool = False, seed: int = 42) -> BenchmarkResult:
    """
    Case 4 — Noisy stable system (FAWP NOT expected).

    A stable system with high noise, designed to produce small but
    non-zero MI values in both channels.  The epsilon threshold should
    prevent a false positive detection.

    Expected: FAWP NOT detected.

    Example
    -------
        noisy_false_positive().verify()
    """
    name        = "noisy_false_positive"
    description = ("Noisy stable system — both MI channels are small and noisy.  "
                   "The epsilon threshold must suppress a false positive.  "
                   "FAWP must NOT be detected.")

    if simulate:
        return _simulate_case(
            name=name, description=description, expected_fawp=False,
            sim_kwargs=dict(a=0.95, K=0.5, delta_pred=20, n_trials=40,
                            n_steps=600, sigma_proc=5.0, seed=seed),
            tau_grid=list(range(0, 28)),
        )

    rng = np.random.default_rng(seed)
    tau = np.arange(0, 30)

    # Both channels: tiny decaying signal buried in noise
    pred  = np.abs(rng.normal(0, 0.006, len(tau)))
    steer = np.abs(rng.normal(0, 0.005, len(tau)))
    fail  = np.zeros(len(tau))   # stable — no cliff

    odw = _run_detector(tau, pred, steer, fail)

    passed  = not odw.fawp_found
    verdict = (
        "Correctly rejected: noise below epsilon threshold"
        if passed else
        f"False positive: ODW={odw.odw_start}-{odw.odw_end} "
        f"— epsilon threshold too low?"
    )

    return BenchmarkResult(
        name=name, description=description, expected_fawp=False,
        tau=tau, pred_mi=pred, steer_mi=steer, fail_rate=fail,
        odw_result=odw, passed=passed, verdict=verdict,
    )


def delayed_collapse(simulate: bool = False, seed: int = 42) -> BenchmarkResult:
    """
    Case 5 — Fast-collapsing unstable system (FAWP expected, narrow ODW).

    A more aggressively unstable system (higher growth rate, weaker
    controller) that collapses earlier.  The ODW is narrow but the
    detector should still find it.  This case tests sensitivity on a
    tight detection problem.

    Expected: FAWP detected.

    Example
    -------
        delayed_collapse().verify()
    """
    name        = "delayed_collapse"
    description = ("Aggressive instability: fast-decaying steering, early failure cliff.  "
                   "ODW is narrow — tests detector sensitivity.  "
                   "FAWP MUST be detected.")

    if simulate:
        return _simulate_case(
            name=name, description=description, expected_fawp=True,
            sim_kwargs=dict(a=1.05, K=0.6, delta_pred=20, n_trials=40,
                            n_steps=600, seed=seed),
            tau_grid=list(range(0, 22)),
        )

    rng = np.random.default_rng(seed)
    tau = np.arange(0, 26)

    # steer rate=0.5 gives fast collapse (tau_h+~12), cliff at centre=18
    # → ODW window of ~3 steps before the cliff — tests sensitivity
    steer = _exp_decay(tau, peak=3.0, rate=0.50) + rng.normal(0, 0.02, len(tau))
    pred  = _exp_decay(tau, peak=2.0, rate=0.07) + rng.normal(0, 0.02, len(tau))
    fail  = _sigmoid(tau.astype(float), centre=18.0, steepness=0.9)

    odw = _run_detector(tau, pred, steer, fail)

    passed  = odw.fawp_found
    verdict = (
        f"ODW tau={odw.odw_start}-{odw.odw_end}, "
        f"lead={odw.mean_lead_to_cliff:.1f} steps to cliff"
        if passed else
        "Missed narrow ODW — consider lowering epsilon or persistence_m"
    )

    return BenchmarkResult(
        name=name, description=description, expected_fawp=True,
        tau=tau, pred_mi=np.maximum(0, pred),
        steer_mi=np.maximum(0, steer), fail_rate=fail,
        odw_result=odw, passed=passed, verdict=verdict,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Simulation-backed case builder (shared by all cases when simulate=True)
# ─────────────────────────────────────────────────────────────────────────────

def _simulate_case(
    name: str,
    description: str,
    expected_fawp: bool,
    sim_kwargs: dict,
    tau_grid: list,
) -> BenchmarkResult:
    """Run FAWPSimulator and feed its MI curves to ODWDetector."""
    from fawp_index.simulate import FAWPSimulator

    sim = FAWPSimulator(**sim_kwargs)
    sr  = sim.run(tau_grid=tau_grid, verbose=False)

    # Use pooled MI as corrected (synthetic ground truth — no null correction needed)
    tau   = sr.tau_grid
    pred  = np.maximum(0.0, sr.mi_pred_strat)
    steer = np.maximum(0.0, sr.mi_steer_pooled)
    fail  = sr.fail_rate

    odw = _run_detector(tau, pred, steer, fail)

    passed = odw.fawp_found == expected_fawp
    verdict = (
        f"ODW tau={odw.odw_start}-{odw.odw_end}"
        if odw.fawp_found else "No ODW detected"
    )
    return BenchmarkResult(
        name=name, description=description, expected_fawp=expected_fawp,
        tau=tau, pred_mi=pred, steer_mi=steer, fail_rate=fail,
        odw_result=odw, passed=passed, verdict=verdict,
        sim_result=sr,
    )


# ─────────────────────────────────────────────────────────────────────────────
# run_all()
# ─────────────────────────────────────────────────────────────────────────────

def run_all(simulate: bool = False, seed: int = 42) -> BenchmarkSuite:
    """
    Run all five benchmark cases and return a BenchmarkSuite.

    Parameters
    ----------
    simulate : bool
        If True, use real FAWPSimulator for each case (~10-30 s).
        If False (default), use analytic curves (< 1 s).
    seed : int
        Random seed for analytic noise.

    Returns
    -------
    BenchmarkSuite

    Example
    -------
        from fawp_index.benchmarks import run_all

        suite = run_all()
        print(suite.summary())
        suite.to_html("benchmarks.html")
        suite.verify_all()          # raises BenchmarkFailure if any case fails
    """
    cases = [
        clean_control(simulate=simulate, seed=seed),
        prediction_only(simulate=simulate, seed=seed),
        control_only(simulate=simulate, seed=seed),
        noisy_false_positive(simulate=simulate, seed=seed),
        delayed_collapse(simulate=simulate, seed=seed),
    ]
    return BenchmarkSuite(results=cases)


# ─────────────────────────────────────────────────────────────────────────────
# HTML report renderer
# ─────────────────────────────────────────────────────────────────────────────

def _case_chart_b64(br: BenchmarkResult) -> str:
    """Return base64 PNG of the case's MI curves, or '' if matplotlib absent."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import base64

        r   = br.odw_result
        fig = plt.figure(figsize=(8, 3))
        ax  = fig.add_subplot(111)
        ax.plot(br.tau, br.pred_mi,  lw=2.0, label="I_pred")
        ax.plot(br.tau, br.steer_mi, lw=1.8, ls="--", label="I_steer")
        ax.plot(br.tau, br.fail_rate, lw=1.2, ls=":", color="grey",
                alpha=0.7, label="Fail rate")
        gap = br.pred_mi - br.steer_mi
        ax.fill_between(br.tau, gap.clip(0), alpha=0.12, color="darkorange",
                        label="Gap")
        if r.odw_start is not None:
            ax.axvspan(r.odw_start, r.odw_end, alpha=0.18, color="green")
        if r.tau_h_plus is not None:
            ax.axvline(r.tau_h_plus, ls=":", lw=1, color="steelblue")
        if r.tau_f is not None:
            ax.axvline(r.tau_f, ls=":", lw=1, color="firebrick")
        ax.set_xlabel("tau"); ax.set_ylabel("bits / rate")
        ax.legend(fontsize=7, loc="upper right"); ax.grid(True, alpha=0.2)
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


def _suite_html(suite: BenchmarkSuite) -> str:
    all_passed = suite.n_failed == 0
    hdr_colour = "#1a7a1a" if all_passed else "#aa1111"
    summary_label = (
        f"ALL {suite.n_passed} CASES PASSED"
        if all_passed else
        f"{suite.n_passed}/{len(suite.results)} PASSED — {suite.n_failed} FAILED"
    )

    # Table rows
    table_rows = ""
    for r in suite.results:
        ok      = "PASS" if r.passed else "FAIL"
        ok_col  = "#1a7a1a" if r.passed else "#aa1111"
        exp     = "FAWP" if r.expected_fawp else "NONE"
        det     = "FAWP" if r.odw_result.fawp_found else "NONE"
        odw_str = (f"{r.odw_result.odw_start}&ndash;{r.odw_result.odw_end}"
                   if r.odw_result.odw_start is not None else "&mdash;")
        table_rows += (
            f'<tr>'
            f'<td style="font-weight:600">{r.name}</td>'
            f'<td>{exp}</td><td>{det}</td>'
            f'<td>{odw_str}</td>'
            f'<td style="color:{ok_col};font-weight:700">{ok}</td>'
            f'</tr>\n'
        )

    # Case cards
    cards = ""
    for r in suite.results:
        ok_col  = "#1a7a1a" if r.passed else "#aa1111"
        ok_lbl  = "PASS ✓" if r.passed else "FAIL ✗"
        b64     = _case_chart_b64(r)
        img_tag = (
            f'<img src="data:image/png;base64,{b64}" '
            f'style="width:100%;border:1px solid #ddd;border-radius:4px;margin-top:10px">'
            if b64 else ""
        )
        odw = r.odw_result
        cards += f"""
<div style="background:#fff;border:1px solid #ddd;border-radius:8px;
            padding:1.2em 1.4em;margin-bottom:1.5em;
            box-shadow:0 1px 4px rgba(0,0,0,0.07)">
  <h3 style="margin:0 0 4px;color:#0E2550">{r.name}
    <span style="font-size:0.75em;color:{ok_col};margin-left:8px">[{ok_lbl}]</span>
  </h3>
  <p style="font-size:0.88em;color:#555;margin:0 0 10px">{r.description}</p>
  <table style="font-size:0.85em;border-collapse:collapse;width:100%">
    <tr style="background:#f4f4f4">
      <td style="padding:4px 8px;font-weight:600">Expected FAWP</td>
      <td>{'YES' if r.expected_fawp else 'NO'}</td>
      <td style="font-weight:600">Detected FAWP</td>
      <td style="color:{ok_col};font-weight:700">{'YES' if odw.fawp_found else 'NO'}</td>
    </tr>
    <tr>
      <td style="padding:4px 8px;font-weight:600">tau_h+</td>
      <td>{odw.tau_h_plus if odw.tau_h_plus is not None else '&mdash;'}</td>
      <td style="font-weight:600">tau_f</td>
      <td>{odw.tau_f if odw.tau_f is not None else '&mdash;'}</td>
    </tr>
    <tr style="background:#f4f4f4">
      <td style="padding:4px 8px;font-weight:600">ODW</td>
      <td colspan="3">
        {f'tau={odw.odw_start}&ndash;{odw.odw_end} ({odw.odw_size} steps)'
         if odw.odw_start is not None else '&mdash;'}
      </td>
    </tr>
    <tr>
      <td style="padding:4px 8px;font-weight:600">Verdict</td>
      <td colspan="3" style="font-style:italic">{r.verdict}</td>
    </tr>
  </table>
  {img_tag}
</div>
"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>fawp-index Benchmark Suite</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    max-width: 900px; margin: 0 auto; padding: 2em 1.5em;
    background: #fafafa; color: #222; line-height: 1.6;
  }}
  header {{
    background: #0E2550; color: white; padding: 1.8em 2em 1.4em;
    border-radius: 8px; margin-bottom: 1.5em;
  }}
  header h1 {{ margin: 0 0 0.3em; font-size: 1.5em; }}
  header p  {{ margin: 0.2em 0; font-size: 0.88em; color: #aac; }}
  .badge {{
    display:inline-block; padding:0.4em 1.2em; border-radius:20px;
    font-weight:700; font-size:1em; color:white;
    background:{hdr_colour}; margin:0.8em 0;
  }}
  h2 {{ color:#0E2550; border-bottom:2px solid #D4AF37; padding-bottom:4px; }}
  table {{ width:100%; border-collapse:collapse; margin:1em 0;
           box-shadow:0 1px 4px rgba(0,0,0,0.07); border-radius:6px;
           overflow:hidden; }}
  thead th {{ background:#0E2550; color:white; padding:9px 12px; text-align:left; }}
  tbody tr:nth-child(even) {{ background:#f8f8f8; }}
  tbody td {{ padding:7px 12px; }}
  footer {{ margin-top:3em; padding-top:1em; border-top:1px solid #ddd;
            font-size:0.8em; color:#888; }}
  a {{ color:#0E2550; }}
  code {{ background:#f4f4f4; padding:2px 5px; border-radius:3px;
          font-size:0.9em; }}
</style>
</head>
<body>

<header>
  <h1>fawp-index Benchmark Suite</h1>
  <p>Generated {suite.generated_date} &bull; fawp-index v0.7.0</p>
  <p><a href="https://doi.org/10.5281/zenodo.18673949"
     style="color:#D4AF37">doi:10.5281/zenodo.18673949</a></p>
</header>

<div class="badge">{summary_label}</div>

<h2>Summary</h2>
<table>
  <thead>
    <tr><th>Case</th><th>Expected</th><th>Detected</th><th>ODW</th><th>Status</th></tr>
  </thead>
  <tbody>{table_rows}</tbody>
</table>

<p style="font-size:0.88em;color:#555">
  Run from Python: <code>from fawp_index.benchmarks import run_all; run_all().verify_all()</code>
</p>

<h2>Case Detail</h2>
{cards}

<footer>
  <a href="https://github.com/DrRalphClayton/fawp-index">fawp-index</a> &bull;
  Ralph Clayton (2026) &bull;
  <a href="https://doi.org/10.5281/zenodo.18673949">doi:10.5281/zenodo.18673949</a>
</footer>
</body>
</html>
"""
