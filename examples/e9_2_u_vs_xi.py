#!/usr/bin/env python3
"""
E9-2: u vs xi steering comparison for FAWP detection.

Standalone Colab-friendly script.

What it does
------------
1. Simulates an unstable delayed-feedback system.
2. Measures prediction MI and two steering channels:
   - u  : executed action
   - xi : intervention command / latent control signal
3. Builds conservative null floors using:
   - permutation null
   - autocorr-preserving circular shift null
4. Computes null-corrected curves.
5. Detects:
   - tau_h_plus_u
   - tau_h_plus_xi
   - tau_f
   - ODW_u
   - ODW_xi
6. Saves:
   - per-seed curves CSV
   - aggregate curves CSV
   - per-seed summary CSV
   - summary JSON
   - figures

Fast smoke test:
    python e9_2_u_vs_xi.py --out_dir e9_2_out --n_seeds 4 --n_trials 100 --tau_max 40

Heavier E9-style run:
    python e9_2_u_vs_xi.py --out_dir e9_2_out --n_seeds 20 --n_trials 400 --tau_max 80
"""
import sys, os as _os
# Allow running from repo root OR from examples/ directory
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class E92Config:
    a: float = 1.02
    K: float = 0.8
    delta_pred: int = 20
    n_trials: int = 100
    n_steps: int = 500
    tau_min: int = 0
    tau_max: int = 40
    x_fail: float = 500.0
    sigma_proc: float = 1.0
    sigma_obs_base: float = 0.1
    alpha_obs: float = 0.001
    sigma_exec: float = 0.50
    exec_gain: float = 1.0
    u_max: float = 10.0
    epsilon_bits: float = 0.01
    beta_null: float = 0.99
    n_null: int = 32
    burn_in: int = 50
    min_trials_strat: int = 30
    t_cap: int = 300
    n_seeds: int = 4
    base_seed: int = 42
    fail_rate_cliff: float = 0.99
    persistence_m: int = 3
    persistence_n: int = 4
    min_tau: int = 1


def mi_from_rho(rho: float) -> float:
    if not np.isfinite(rho):
        return 0.0
    rho = float(np.clip(rho, -0.999999, 0.999999))
    return float((-0.5 * np.log(1.0 - rho**2)) / np.log(2.0))


def mi_from_arrays(x: np.ndarray, y: np.ndarray, min_n: int = 30) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < min_n or y.size < min_n:
        return 0.0
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx <= 1e-12 or sy <= 1e-12:
        return 0.0
    rho = np.corrcoef(x, y)[0, 1]
    return mi_from_rho(rho)


def null_shuffle_quantile(
    x: np.ndarray,
    y: np.ndarray,
    n_null: int,
    beta: float,
    rng: np.random.Generator,
    min_n: int = 30,
) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < min_n:
        return 0.0
    vals = []
    for _ in range(n_null):
        vals.append(mi_from_arrays(x, rng.permutation(y), min_n=min_n))
    return float(np.quantile(vals, beta)) if vals else 0.0


def null_shift_quantile(
    x: np.ndarray,
    y: np.ndarray,
    n_null: int,
    beta: float,
    rng: np.random.Generator,
    min_n: int = 30,
) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = x.size
    if n < min_n or n < 3:
        return 0.0
    vals = []
    for _ in range(n_null):
        shift = int(rng.integers(1, n))
        vals.append(mi_from_arrays(x, np.roll(y, shift), min_n=min_n))
    return float(np.quantile(vals, beta)) if vals else 0.0


def conservative_null_floor(
    x: np.ndarray,
    y: np.ndarray,
    n_null: int,
    beta: float,
    rng: np.random.Generator,
    min_n: int = 30,
) -> float:
    q_shuffle = null_shuffle_quantile(x, y, n_null, beta, rng, min_n=min_n)
    q_shift = null_shift_quantile(x, y, n_null, beta, rng, min_n=min_n)
    return float(max(q_shuffle, q_shift))


def corrected_mi(
    x: np.ndarray,
    y: np.ndarray,
    n_null: int,
    beta: float,
    rng: np.random.Generator,
    min_n: int = 30,
) -> Tuple[float, float, float]:
    raw = mi_from_arrays(x, y, min_n=min_n)
    floor = conservative_null_floor(x, y, n_null=n_null, beta=beta, rng=rng, min_n=min_n)
    corr = max(0.0, raw - floor)
    return raw, floor, corr


def persistent_mask(mask: np.ndarray, m: int, n: int) -> np.ndarray:
    """
    Mark indices that participate in any window of length n with at least m True values.
    """
    mask = np.asarray(mask, dtype=bool)
    out = np.zeros_like(mask, dtype=bool)
    if n <= 1:
        return mask.copy()
    for i in range(0, len(mask) - n + 1):
        window = mask[i:i + n]
        if int(window.sum()) >= m:
            out[i:i + n] |= window
    return out


def first_tau_where(tau: np.ndarray, cond: np.ndarray) -> Optional[int]:
    idx = np.where(cond)[0]
    if idx.size == 0:
        return None
    return int(tau[int(idx[0])])


def first_contiguous_range(tau: np.ndarray, mask: np.ndarray) -> Tuple[Optional[int], Optional[int]]:
    mask = np.asarray(mask, dtype=bool)
    idx = np.where(mask)[0]
    if idx.size == 0:
        return None, None
    start = idx[0]
    end = start
    for j in idx[1:]:
        if j == end + 1:
            end = j
        else:
            break
    return int(tau[start]), int(tau[end])


def obs_noise_std(cfg: E92Config, delay: int) -> float:
    return float(np.sqrt(cfg.sigma_obs_base**2 + cfg.alpha_obs * delay))


def run_trial(delay: int, cfg: E92Config, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """
    x_{t+1} = a * x_t + u_t + w_t
    xi_t = clipped delayed controller command
    u_t  = executed action = exec_gain * xi_t + execution noise, clipped
    """
    total = cfg.n_steps + cfg.delta_pred + delay + 5
    x_hist = np.zeros(total + 1, dtype=float)

    D = np.zeros(cfg.n_steps, dtype=float)
    U = np.zeros(cfg.n_steps, dtype=float)
    XI = np.zeros(cfg.n_steps, dtype=float)
    YD = np.full(cfg.n_steps, np.nan, dtype=float)

    x = 0.0
    x_hist[0] = x
    crashed = False
    crash_time = None
    obs_std = obs_noise_std(cfg, delay)

    for t in range(cfg.n_steps):
        D[t] = x

        td = t - delay
        if td < 0:
            y_del = np.nan
            xi = 0.0
        else:
            y_del = x_hist[td] + rng.normal(0.0, obs_std)
            xi = float(np.clip(-cfg.K * y_del, -cfg.u_max, cfg.u_max))

        u = float(np.clip(cfg.exec_gain * xi + rng.normal(0.0, cfg.sigma_exec), -cfg.u_max, cfg.u_max))

        XI[t] = xi
        U[t] = u
        YD[t] = y_del

        x = cfg.a * x + u + rng.normal(0.0, cfg.sigma_proc)
        x_hist[t + 1] = x

        if abs(x) > cfg.x_fail:
            crashed = True
            crash_time = t
            break

    t_end = int(crash_time + 1) if crashed else cfg.n_steps

    x_curr = x_hist[t_end]
    for k in range(t_end, total):
        x_curr = cfg.a * x_curr + rng.normal(0.0, cfg.sigma_proc)
        x_hist[k + 1] = x_curr

    pred_arr = D[:t_end]
    fut_arr = x_hist[cfg.delta_pred:cfg.delta_pred + t_end]

    shift = delay + 1
    sl = max(0, t_end - shift)
    U_s = U[:sl]
    XI_s = XI[:sl]
    O_s = x_hist[shift:shift + sl] + rng.normal(0.0, obs_std, size=sl)

    return {
        "D": pred_arr,
        "X": fut_arr,
        "U": U_s,
        "XI": XI_s,
        "O": O_s,
        "t_end": t_end,
        "crashed": int(crashed),
        "crash_time": -1 if crash_time is None else int(crash_time),
    }


def stratified_prediction_and_null(
    trials: List[Dict[str, np.ndarray]],
    cfg: E92Config,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    """
    Raw stratified predictive MI:
        mean_t I(D_t ; X_{t+Δ}) across alive-trial strata.
    Null:
        stratified shuffle within each time stratum, averaged across strata.
    """
    if not trials:
        return 0.0, 0.0, 0.0

    max_t_end = max(int(tr["t_end"]) for tr in trials)
    t_cap_eff = min(cfg.t_cap, max_t_end - 1)
    if t_cap_eff <= cfg.burn_in + 2:
        return 0.0, 0.0, 0.0

    strata = []
    raw_per_t = []

    for t in range(cfg.burn_in, t_cap_eff):
        xs = []
        ys = []
        for tr in trials:
            if tr["t_end"] > t:
                xs.append(tr["D"][t])
                ys.append(tr["X"][t])
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        if xs.size < cfg.min_trials_strat:
            continue
        strata.append((xs, ys))
        raw_per_t.append(mi_from_arrays(xs, ys, min_n=cfg.min_trials_strat))

    if not raw_per_t:
        return 0.0, 0.0, 0.0

    raw = float(np.mean(raw_per_t))

    null_vals = []
    for _ in range(cfg.n_null):
        vals = []
        for xs, ys in strata:
            vals.append(mi_from_arrays(xs, rng.permutation(ys), min_n=cfg.min_trials_strat))
        null_vals.append(float(np.mean(vals)) if vals else 0.0)

    floor = float(np.quantile(null_vals, cfg.beta_null)) if null_vals else 0.0
    corr = max(0.0, raw - floor)
    return raw, floor, corr


def analyze_delay_for_seed(
    delay: int,
    cfg: E92Config,
    seed: int,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    trials: List[Dict[str, np.ndarray]] = []

    pool_D = []
    pool_X = []
    pool_U = []
    pool_XI = []
    pool_O = []
    fail_count = 0

    for _ in range(cfg.n_trials):
        tr_seed = int(rng.integers(0, 2**32 - 1))
        tr = run_trial(delay, cfg, np.random.default_rng(tr_seed))
        trials.append(tr)
        fail_count += int(tr["crashed"])

        if tr["D"].size:
            pool_D.append(tr["D"])
            pool_X.append(tr["X"])
        if tr["U"].size:
            pool_U.append(tr["U"])
            pool_XI.append(tr["XI"])
            pool_O.append(tr["O"])

    all_D = np.concatenate(pool_D) if pool_D else np.array([])
    all_X = np.concatenate(pool_X) if pool_X else np.array([])
    all_U = np.concatenate(pool_U) if pool_U else np.array([])
    all_XI = np.concatenate(pool_XI) if pool_XI else np.array([])
    all_O = np.concatenate(pool_O) if pool_O else np.array([])

    local_rng = np.random.default_rng(seed + 100000 + 17 * delay)

    pred_pooled_raw, pred_pooled_floor, pred_pooled_corr = corrected_mi(
        all_D, all_X,
        n_null=cfg.n_null,
        beta=cfg.beta_null,
        rng=local_rng,
        min_n=cfg.min_trials_strat,
    )

    pred_strat_raw, pred_strat_floor, pred_strat_corr = stratified_prediction_and_null(
        trials=trials,
        cfg=cfg,
        rng=np.random.default_rng(seed + 200000 + 29 * delay),
    )

    steer_u_raw, steer_u_floor, steer_u_corr = corrected_mi(
        all_U, all_O,
        n_null=cfg.n_null,
        beta=cfg.beta_null,
        rng=np.random.default_rng(seed + 300000 + 31 * delay),
        min_n=cfg.min_trials_strat,
    )

    steer_xi_raw, steer_xi_floor, steer_xi_corr = corrected_mi(
        all_XI, all_O,
        n_null=cfg.n_null,
        beta=cfg.beta_null,
        rng=np.random.default_rng(seed + 400000 + 37 * delay),
        min_n=cfg.min_trials_strat,
    )

    fail_rate = fail_count / max(1, cfg.n_trials)

    return {
        "seed": int(seed),
        "tau": int(delay),
        "pred_pooled_raw": float(pred_pooled_raw),
        "pred_pooled_floor": float(pred_pooled_floor),
        "pred_pooled_corr": float(pred_pooled_corr),
        "pred_strat_raw": float(pred_strat_raw),
        "pred_strat_floor": float(pred_strat_floor),
        "pred_strat_corr": float(pred_strat_corr),
        "steer_u_raw": float(steer_u_raw),
        "steer_u_floor": float(steer_u_floor),
        "steer_u_corr": float(steer_u_corr),
        "steer_xi_raw": float(steer_xi_raw),
        "steer_xi_floor": float(steer_xi_floor),
        "steer_xi_corr": float(steer_xi_corr),
        "gap_u_corr": float(pred_strat_corr - steer_u_corr),
        "gap_xi_corr": float(pred_strat_corr - steer_xi_corr),
        "fail_rate": float(fail_rate),
    }


def summarize_seed(seed_df: pd.DataFrame, cfg: E92Config) -> Dict[str, Optional[float]]:
    tau = seed_df["tau"].to_numpy(dtype=int)
    pred = seed_df["pred_strat_corr"].to_numpy(dtype=float)
    su = seed_df["steer_u_corr"].to_numpy(dtype=float)
    sxi = seed_df["steer_xi_corr"].to_numpy(dtype=float)
    fail = seed_df["fail_rate"].to_numpy(dtype=float)

    cond_u = (tau >= cfg.min_tau) & (su <= cfg.epsilon_bits)
    cond_xi = (tau >= cfg.min_tau) & (sxi <= cfg.epsilon_bits)
    tau_h_u = first_tau_where(tau, cond_u)
    tau_h_xi = first_tau_where(tau, cond_xi)
    tau_f = first_tau_where(tau, fail >= cfg.fail_rate_cliff)

    base_u = (pred > cfg.epsilon_bits) & (su <= cfg.epsilon_bits)
    base_xi = (pred > cfg.epsilon_bits) & (sxi <= cfg.epsilon_bits)

    if tau_h_u is not None:
        base_u &= (tau >= tau_h_u)
    if tau_h_xi is not None:
        base_xi &= (tau >= tau_h_xi)
    if tau_f is not None:
        base_u &= (tau < tau_f)
        base_xi &= (tau < tau_f)

    mask_u = persistent_mask(base_u, cfg.persistence_m, cfg.persistence_n)
    mask_xi = persistent_mask(base_xi, cfg.persistence_m, cfg.persistence_n)

    odw_u_start, odw_u_end = first_contiguous_range(tau, mask_u)
    odw_xi_start, odw_xi_end = first_contiguous_range(tau, mask_xi)

    peak_gap_u_idx = int(np.argmax(seed_df["gap_u_corr"].to_numpy(dtype=float)))
    peak_gap_xi_idx = int(np.argmax(seed_df["gap_xi_corr"].to_numpy(dtype=float)))

    return {
        "seed": int(seed_df["seed"].iloc[0]),
        "tau_h_plus_u": None if tau_h_u is None else int(tau_h_u),
        "tau_h_plus_xi": None if tau_h_xi is None else int(tau_h_xi),
        "tau_f": None if tau_f is None else int(tau_f),
        "odw_u_start": None if odw_u_start is None else int(odw_u_start),
        "odw_u_end": None if odw_u_end is None else int(odw_u_end),
        "odw_xi_start": None if odw_xi_start is None else int(odw_xi_start),
        "odw_xi_end": None if odw_xi_end is None else int(odw_xi_end),
        "peak_gap_u_tau": int(seed_df["tau"].iloc[peak_gap_u_idx]),
        "peak_gap_u_bits": float(seed_df["gap_u_corr"].iloc[peak_gap_u_idx]),
        "peak_gap_xi_tau": int(seed_df["tau"].iloc[peak_gap_xi_idx]),
        "peak_gap_xi_bits": float(seed_df["gap_xi_corr"].iloc[peak_gap_xi_idx]),
    }


def summarize_aggregate(curves_df: pd.DataFrame, cfg: E92Config) -> Dict[str, Optional[float]]:
    tau = curves_df["tau"].to_numpy(dtype=int)
    pred = curves_df["pred_strat_corr"].to_numpy(dtype=float)
    su = curves_df["steer_u_corr"].to_numpy(dtype=float)
    sxi = curves_df["steer_xi_corr"].to_numpy(dtype=float)
    fail = curves_df["fail_rate"].to_numpy(dtype=float)

    tau_h_u = first_tau_where(tau, (tau >= cfg.min_tau) & (su <= cfg.epsilon_bits))
    tau_h_xi = first_tau_where(tau, (tau >= cfg.min_tau) & (sxi <= cfg.epsilon_bits))
    tau_f = first_tau_where(tau, fail >= cfg.fail_rate_cliff)

    base_u = (pred > cfg.epsilon_bits) & (su <= cfg.epsilon_bits)
    base_xi = (pred > cfg.epsilon_bits) & (sxi <= cfg.epsilon_bits)

    if tau_h_u is not None:
        base_u &= (tau >= tau_h_u)
    if tau_h_xi is not None:
        base_xi &= (tau >= tau_h_xi)
    if tau_f is not None:
        base_u &= (tau < tau_f)
        base_xi &= (tau < tau_f)

    mask_u = persistent_mask(base_u, cfg.persistence_m, cfg.persistence_n)
    mask_xi = persistent_mask(base_xi, cfg.persistence_m, cfg.persistence_n)

    odw_u_start, odw_u_end = first_contiguous_range(tau, mask_u)
    odw_xi_start, odw_xi_end = first_contiguous_range(tau, mask_xi)

    peak_gap_u_idx = int(np.argmax(curves_df["gap_u_corr"].to_numpy(dtype=float)))
    peak_gap_xi_idx = int(np.argmax(curves_df["gap_xi_corr"].to_numpy(dtype=float)))
    peak_pred_idx = int(np.argmax(curves_df["pred_strat_corr"].to_numpy(dtype=float)))

    return {
        "tau_h_plus_u": None if tau_h_u is None else int(tau_h_u),
        "tau_h_plus_xi": None if tau_h_xi is None else int(tau_h_xi),
        "tau_f": None if tau_f is None else int(tau_f),
        "odw_u_start": None if odw_u_start is None else int(odw_u_start),
        "odw_u_end": None if odw_u_end is None else int(odw_u_end),
        "odw_xi_start": None if odw_xi_start is None else int(odw_xi_start),
        "odw_xi_end": None if odw_xi_end is None else int(odw_xi_end),
        "peak_gap_u_tau": int(curves_df["tau"].iloc[peak_gap_u_idx]),
        "peak_gap_u_bits": float(curves_df["gap_u_corr"].iloc[peak_gap_u_idx]),
        "peak_gap_xi_tau": int(curves_df["tau"].iloc[peak_gap_xi_idx]),
        "peak_gap_xi_bits": float(curves_df["gap_xi_corr"].iloc[peak_gap_xi_idx]),
        "peak_pred_tau": int(curves_df["tau"].iloc[peak_pred_idx]),
        "peak_pred_bits": float(curves_df["pred_strat_corr"].iloc[peak_pred_idx]),
        "fawp_found_u": bool(odw_u_start is not None),
        "fawp_found_xi": bool(odw_xi_start is not None),
    }


def plot_main(curves_df: pd.DataFrame, summary: Dict[str, Optional[float]], out_path: str) -> None:
    tau = curves_df["tau"].to_numpy()
    fig, ax1 = plt.subplots(figsize=(12, 6.5))

    ax1.plot(tau, curves_df["pred_strat_corr"].to_numpy(), linewidth=2.5, label="Prediction, stratified, corrected")
    ax1.plot(tau, curves_df["steer_u_corr"].to_numpy(), linewidth=2.2, linestyle="--", label="Steering via u, corrected")
    ax1.plot(tau, curves_df["steer_xi_corr"].to_numpy(), linewidth=2.2, linestyle="-.", label="Steering via xi, corrected")
    ax1.plot(tau, curves_df["pred_strat_raw"].to_numpy(), linewidth=1.6, alpha=0.6, label="Prediction, stratified, raw")

    if summary.get("tau_h_plus_u") is not None:
        ax1.axvline(summary["tau_h_plus_u"], linestyle=":", linewidth=1.5, label=f"tau_h_plus_u = {summary['tau_h_plus_u']}")
    if summary.get("tau_h_plus_xi") is not None:
        ax1.axvline(summary["tau_h_plus_xi"], linestyle=":", linewidth=1.5, label=f"tau_h_plus_xi = {summary['tau_h_plus_xi']}")

    ax2 = ax1.twinx()
    ax2.plot(tau, curves_df["fail_rate"].to_numpy(), linewidth=2.0, linestyle=":", alpha=0.7, label="Failure rate")
    ax2.set_ylabel("Failure rate")
    ax2.set_ylim(-0.02, 1.02)

    ax1.set_xlabel("Latency tau")
    ax1.set_ylabel("Mutual information, bits")
    ax1.set_title("E9-2 u vs xi steering comparison")
    ax1.grid(True, alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_gaps(curves_df: pd.DataFrame, summary: Dict[str, Optional[float]], out_path: str) -> None:
    tau = curves_df["tau"].to_numpy()
    gap_u = curves_df["gap_u_corr"].to_numpy()
    gap_xi = curves_df["gap_xi_corr"].to_numpy()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(tau, gap_u, linewidth=2.5, label="Leverage gap using u")
    ax.plot(tau, gap_xi, linewidth=2.5, linestyle="--", label="Leverage gap using xi")
    ax.axhline(0.0, linewidth=1.2)

    if summary.get("odw_u_start") is not None and summary.get("odw_u_end") is not None:
        ax.axvspan(summary["odw_u_start"], summary["odw_u_end"], alpha=0.15, label="ODW, u")
    if summary.get("odw_xi_start") is not None and summary.get("odw_xi_end") is not None:
        ax.axvspan(summary["odw_xi_start"], summary["odw_xi_end"], alpha=0.10, label="ODW, xi")

    if summary.get("tau_f") is not None:
        ax.axvline(summary["tau_f"], linestyle=":", linewidth=1.5, label=f"tau_f = {summary['tau_f']}")

    ax.set_xlabel("Latency tau")
    ax.set_ylabel("Prediction minus steering, corrected bits")
    ax.set_title("E9-2 leverage-gap comparison")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_nulls(curves_df: pd.DataFrame, out_path: str) -> None:
    tau = curves_df["tau"].to_numpy()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(tau, curves_df["pred_strat_raw"].to_numpy(), linewidth=2.2, label="Prediction raw")
    ax.plot(tau, curves_df["pred_strat_floor"].to_numpy(), linewidth=2.0, linestyle="--", label="Prediction null floor")
    ax.plot(tau, curves_df["steer_u_raw"].to_numpy(), linewidth=2.0, label="Steering u raw")
    ax.plot(tau, curves_df["steer_u_floor"].to_numpy(), linewidth=2.0, linestyle="--", label="Steering u null floor")
    ax.plot(tau, curves_df["steer_xi_raw"].to_numpy(), linewidth=2.0, label="Steering xi raw")
    ax.plot(tau, curves_df["steer_xi_floor"].to_numpy(), linewidth=2.0, linestyle="--", label="Steering xi null floor")

    ax.set_xlabel("Latency tau")
    ax.set_ylabel("Mutual information, bits")
    ax.set_title("E9-2 raw curves vs null floors")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_suite(cfg: E92Config, out_dir: str) -> Dict[str, object]:
    os.makedirs(out_dir, exist_ok=True)

    tau_grid = list(range(cfg.tau_min, cfg.tau_max + 1))
    rows = []

    seed_values = [cfg.base_seed + i for i in range(cfg.n_seeds)]

    for seed in seed_values:
        print(f"\nRunning seed {seed}")
        for tau in tau_grid:
            row = analyze_delay_for_seed(delay=tau, cfg=cfg, seed=seed * 1000 + tau)
            rows.append(row)
            print(
                f"  tau={tau:>3d}  "
                f"pred_strat_corr={row['pred_strat_corr']:.4f}  "
                f"steer_u_corr={row['steer_u_corr']:.4f}  "
                f"steer_xi_corr={row['steer_xi_corr']:.4f}  "
                f"fail={row['fail_rate']:.3f}"
            )

    seed_curves_df = pd.DataFrame(rows)
    seed_curves_path = os.path.join(out_dir, "e9_2_seed_curves.csv")
    seed_curves_df.to_csv(seed_curves_path, index=False)

    per_seed_summary = []
    for _, sdf in seed_curves_df.groupby("seed", sort=True):
        per_seed_summary.append(summarize_seed(sdf.sort_values("tau"), cfg))
    per_seed_summary_df = pd.DataFrame(per_seed_summary)
    per_seed_summary_path = os.path.join(out_dir, "e9_2_seed_summary.csv")
    per_seed_summary_df.to_csv(per_seed_summary_path, index=False)

    curves_df = (
        seed_curves_df
        .groupby("tau", as_index=False)
        .mean(numeric_only=True)
        .sort_values("tau")
        .reset_index(drop=True)
    )
    curves_df["gap_u_corr"] = curves_df["pred_strat_corr"] - curves_df["steer_u_corr"]
    curves_df["gap_xi_corr"] = curves_df["pred_strat_corr"] - curves_df["steer_xi_corr"]
    curves_path = os.path.join(out_dir, "e9_2_aggregate_curves.csv")
    curves_df.to_csv(curves_path, index=False)

    agg_summary = summarize_aggregate(curves_df, cfg)
    summary = {
        "experiment": "E9-2",
        "description": "u vs xi steering comparison for FAWP detection",
        "config": asdict(cfg),
        "aggregate_summary": agg_summary,
        "seed_summary_medians": per_seed_summary_df.median(numeric_only=True).to_dict() if not per_seed_summary_df.empty else {},
        "seed_summary_means": per_seed_summary_df.mean(numeric_only=True).to_dict() if not per_seed_summary_df.empty else {},
        "files": {
            "seed_curves_csv": seed_curves_path,
            "seed_summary_csv": per_seed_summary_path,
            "aggregate_curves_csv": curves_path,
        },
    }

    summary_path = os.path.join(out_dir, "e9_2_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    summary["files"]["summary_json"] = summary_path

    fig_main = os.path.join(out_dir, "E9_2_Main_Compare.png")
    fig_gaps = os.path.join(out_dir, "E9_2_Gaps.png")
    fig_nulls = os.path.join(out_dir, "E9_2_Nulls.png")
    plot_main(curves_df, agg_summary, fig_main)
    plot_gaps(curves_df, agg_summary, fig_gaps)
    plot_nulls(curves_df, fig_nulls)

    summary["files"]["fig_main"] = fig_main
    summary["files"]["fig_gaps"] = fig_gaps
    summary["files"]["fig_nulls"] = fig_nulls

    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="E9-2 u vs xi steering comparison")
    p.add_argument("--out_dir", type=str, default="e9_2_out")
    p.add_argument("--a", type=float, default=1.02)
    p.add_argument("--K", type=float, default=0.8)
    p.add_argument("--delta_pred", type=int, default=20)
    p.add_argument("--n_trials", type=int, default=100)
    p.add_argument("--n_steps", type=int, default=500)
    p.add_argument("--tau_min", type=int, default=0)
    p.add_argument("--tau_max", type=int, default=40)
    p.add_argument("--x_fail", type=float, default=500.0)
    p.add_argument("--sigma_proc", type=float, default=1.0)
    p.add_argument("--sigma_obs_base", type=float, default=0.1)
    p.add_argument("--alpha_obs", type=float, default=0.001)
    p.add_argument("--sigma_exec", type=float, default=0.5)
    p.add_argument("--exec_gain", type=float, default=1.0)
    p.add_argument("--u_max", type=float, default=10.0)
    p.add_argument("--epsilon_bits", type=float, default=0.01)
    p.add_argument("--beta_null", type=float, default=0.99)
    p.add_argument("--n_null", type=int, default=32)
    p.add_argument("--burn_in", type=int, default=50)
    p.add_argument("--min_trials_strat", type=int, default=30)
    p.add_argument("--t_cap", type=int, default=300)
    p.add_argument("--n_seeds", type=int, default=4)
    p.add_argument("--base_seed", type=int, default=42)
    p.add_argument("--fail_rate_cliff", type=float, default=0.99)
    p.add_argument("--persistence_m", type=int, default=3)
    p.add_argument("--persistence_n", type=int, default=4)
    p.add_argument("--min_tau", type=int, default=1)
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = E92Config(
        a=args.a,
        K=args.K,
        delta_pred=args.delta_pred,
        n_trials=args.n_trials,
        n_steps=args.n_steps,
        tau_min=args.tau_min,
        tau_max=args.tau_max,
        x_fail=args.x_fail,
        sigma_proc=args.sigma_proc,
        sigma_obs_base=args.sigma_obs_base,
        alpha_obs=args.alpha_obs,
        sigma_exec=args.sigma_exec,
        exec_gain=args.exec_gain,
        u_max=args.u_max,
        epsilon_bits=args.epsilon_bits,
        beta_null=args.beta_null,
        n_null=args.n_null,
        burn_in=args.burn_in,
        min_trials_strat=args.min_trials_strat,
        t_cap=args.t_cap,
        n_seeds=args.n_seeds,
        base_seed=args.base_seed,
        fail_rate_cliff=args.fail_rate_cliff,
        persistence_m=args.persistence_m,
        persistence_n=args.persistence_n,
        min_tau=args.min_tau,
    )

    summary = run_suite(cfg=cfg, out_dir=args.out_dir)

    print("\nDone.\n")
    print(json.dumps(summary["aggregate_summary"], indent=2))


if __name__ == "__main__":
    main()