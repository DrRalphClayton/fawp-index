"""
fawp_index.explain — Plain-English Result Interpreter

Translates any fawp-index result object into a clear, human-readable
narrative: what the numbers mean, whether the system is in trouble,
and what to do about it.

Works with:
  - FAWPResult       (core alpha index)
  - OATSResult       (analytic agency horizon)
  - ControlCliffResult (E5 control cliff)
  - SimulationResult (E8-style simulation)
  - dict / arbitrary kwargs (free-form explain)

Ralph Clayton (2026) — doi:10.5281/zenodo.18673949
"""

import numpy as np



# ── Severity levels ────────────────────────────────────────────────────────

SEVERITY_LABELS = {
    0: "✅ HEALTHY",
    1: "⚠️  WATCH",
    2: "🔴 WARNING",
    3: "🚨 CRITICAL",
}


def _severity_color(level: int) -> str:
    return ["green", "yellow", "orange", "red"][min(level, 3)]


# ── Core explain functions ──────────────────────────────────────────────────

def explain_fawp(result, verbose: bool = True) -> str:
    """
    Plain-English explanation of a FAWPResult.

    Parameters
    ----------
    result : FAWPResult
        Output of FAWPAlphaIndex().compute().
    verbose : bool
        If True, include interpretation guide.

    Returns
    -------
    str
    """
    in_fawp = result.in_fawp
    if hasattr(in_fawp, '__len__'):
        fawp_count = int(in_fawp.sum())
        fawp_total = len(in_fawp)
        in_fawp_any = fawp_count > 0
    else:
        in_fawp_any = bool(in_fawp)
        fawp_count = 1 if in_fawp_any else 0
        fawp_total = 1

    peak_alpha = float(result.peak_alpha)
    tau_h = result.tau_h
    peak_tau = result.peak_tau

    pred_mi = result.pred_mi_corrected
    steer_mi = result.steer_mi_corrected
    mean_pred = float(np.mean(pred_mi)) if hasattr(pred_mi, '__len__') else float(pred_mi)
    mean_steer = float(np.mean(steer_mi)) if hasattr(steer_mi, '__len__') else float(steer_mi)

    # Severity
    if not in_fawp_any:
        severity = 0
    elif peak_alpha < 0.5:
        severity = 1
    elif peak_alpha < 1.5:
        severity = 2
    else:
        severity = 3

    lines = [
        "=" * 62,
        f"  FAWP DIAGNOSIS  —  {SEVERITY_LABELS[severity]}",
        "=" * 62,
        "",
    ]

    # What is happening
    lines.append("WHAT IS HAPPENING:")
    if not in_fawp_any:
        lines.append(
            "  Your system is operating normally. Predictive information\n"
            "  and control authority are coupled — your model can still\n"
            "  act on what it knows."
        )
    else:
        lines.append(
            f"  FAWP regime detected at {fawp_count}/{fawp_total} horizon(s).\n"
            f"  Your system has entered the Information-Control Exclusion\n"
            f"  zone: the model still predicts (MI = {mean_pred:.3f} bits),\n"
            f"  but the ability to act on those predictions has collapsed\n"
            f"  (steering MI = {mean_steer:.4f} bits ≈ zero)."
        )

    lines.append("")

    # Key numbers
    lines.append("KEY NUMBERS:")
    lines.append(f"  Peak FAWP Alpha Index : {peak_alpha:.4f}")
    lines.append(f"  Mean predictive MI    : {mean_pred:.4f} bits")
    lines.append(f"  Mean steering MI      : {mean_steer:.4f} bits")
    if tau_h is not None:
        lines.append(f"  Agency horizon τ_h    : {tau_h}  (control collapses here)")
    if peak_tau is not None:
        lines.append(f"  Peak alpha at τ       : {peak_tau}")

    lines.append("")

    # What it means
    lines.append("WHAT THIS MEANS:")
    if severity == 0:
        lines.append(
            "  Your predictive model retains meaningful coupling to\n"
            "  outcomes. Normal operations — no structural decoupling\n"
            "  detected."
        )
    elif severity == 1:
        lines.append(
            "  Mild FAWP signal. Your model is beginning to decouple\n"
            "  from outcomes. Worth monitoring — not yet actionable."
        )
    elif severity == 2:
        lines.append(
            "  Moderate FAWP regime. Your predictions remain informative\n"
            "  but your actions are losing impact. This is the crowding /\n"
            "  latency / saturation zone. Review execution assumptions."
        )
    else:
        lines.append(
            "  Strong FAWP regime. Predictive MI is high but steering MI\n"
            "  has essentially collapsed. Acting on model outputs in this\n"
            "  state is likely futile or counterproductive. The model is\n"
            "  'seeing' something it can no longer 'touch'."
        )

    lines.append("")

    # What to do
    lines.append("SUGGESTED ACTIONS:")
    if severity == 0:
        lines.append("  • Continue normal operations.")
        lines.append("  • Re-run FAWP check periodically as conditions change.")
    elif severity == 1:
        lines.append("  • Investigate whether latency or execution costs have increased.")
        lines.append("  • Check if signal crowding is developing.")
        lines.append("  • Set a monitoring threshold and alert if alpha index rises.")
    elif severity == 2:
        lines.append("  • Reduce position sizing or execution aggression.")
        lines.append("  • Consider whether the prediction horizon needs adjustment.")
        lines.append("  • Look for alternative action channels with lower steering MI decay.")
    else:
        lines.append("  • Suspend or severely reduce actions based on this model.")
        lines.append("  • The information-control link is broken — acting amplifies noise.")
        lines.append("  • Investigate root cause: latency? crowding? regime shift?")
        lines.append("  • Consider a fundamentally different action mechanism.")

    if verbose:
        lines.append("")
        lines.append("─" * 62)
        lines.append("INTERPRETATION GUIDE:")
        lines.append("  Alpha Index > 0   → FAWP regime (pred MI >> steer MI)")
        lines.append("  Alpha Index = 0   → Normal coupling")
        lines.append("  τ_h               → Horizon where control first collapses")
        lines.append("  High pred MI      → Model still 'sees' the future")
        lines.append("  Near-zero steer MI → Model can no longer 'touch' the future")
        lines.append("  doi:10.5281/zenodo.18663547")

    lines.append("=" * 62)
    return "\n".join(lines)


def explain_oats(result, verbose: bool = True) -> str:
    """
    Plain-English explanation of an OATSResult (analytic agency horizon).

    Parameters
    ----------
    result : OATSResult
        Output of AgencyHorizon().compute().
    verbose : bool
        Include interpretation guide.
    """
    tau_h = result.tau_h
    snr = result.snr_initial
    epsilon = result.epsilon
    mi_0 = float(result.mi[0])
    alpha = result.alpha
    P = result.P

    # How fast does it decay? Time to halve MI
    half_mi = mi_0 / 2.0
    half_idx = np.where(result.mi <= half_mi)[0]
    tau_half = float(result.tau_grid[half_idx[0]]) if len(half_idx) else None

    if not np.isfinite(tau_h) or tau_h <= 0:
        severity = 0
        verdict = "NO FINITE HORIZON — system retains agency indefinitely under these parameters"
    elif tau_h < 10:
        severity = 3
        verdict = "VERY SHORT HORIZON — agency collapses almost immediately"
    elif tau_h < 100:
        severity = 2
        verdict = "SHORT HORIZON — limited operational window before control loss"
    elif tau_h < 1000:
        severity = 1
        verdict = "MODERATE HORIZON — meaningful but finite agency window"
    else:
        severity = 0
        verdict = "LONG HORIZON — extensive agency window under current parameters"

    lines = [
        "=" * 62,
        f"  OATS AGENCY HORIZON  —  {SEVERITY_LABELS[severity]}",
        "=" * 62,
        "",
        f"VERDICT: {verdict}",
        "",
        "WHAT IS HAPPENING:",
        f"  Under the analytic Gaussian channel model, your system\n"
        f"  starts with {mi_0:.3f} bits of mutual information at zero latency\n"
        f"  (SNR = {snr:.1f}). As latency grows, noise accumulates at rate\n"
        f"  α = {alpha:.4g} per unit time, eroding the information link.",
        "",
        "KEY NUMBERS:",
        f"  Initial MI (τ=0)    : {mi_0:.4f} bits",
        f"  MI threshold ε      : {epsilon} bits",
        f"  Agency horizon τ_h  : {tau_h:.2f}",
    ]

    if tau_half is not None:
        lines.append(f"  MI halving point    : τ = {tau_half:.2f}")

    lines += [
        f"  Noise growth rate α : {alpha:.4g}",
        f"  Signal power P      : {P:.4g}",
        f"  Initial SNR         : {snr:.1f}",
        "",
        "WHAT THIS MEANS:",
    ]

    if severity == 0 and np.isfinite(tau_h):
        lines.append(
            f"  Your system maintains meaningful agency (MI > {epsilon} bits)\n"
            f"  for τ up to {tau_h:.1f}. This is a long operational window.\n"
            f"  Increasing noise rate α or reducing P would shorten it."
        )
    elif severity == 0:
        lines.append(
            "  No finite horizon detected — MI never drops to threshold\n"
            "  within the modelled range. The system retains agency\n"
            "  indefinitely under these parameters."
        )
    elif severity == 1:
        lines.append(
            f"  Agency persists to τ = {tau_h:.1f} — a moderate window.\n"
            f"  Plan interventions well within this range for reliable\n"
            f"  coupling between prediction and action."
        )
    elif severity == 2:
        lines.append(
            f"  Agency collapses by τ = {tau_h:.1f}. Short operational window.\n"
            f"  Reduce latency, increase signal power, or tighten the\n"
            f"  action-observation loop to extend the horizon."
        )
    else:
        lines.append(
            f"  Agency collapses almost immediately (τ_h = {tau_h:.2f}).\n"
            f"  The noise-to-signal ratio is too high for meaningful\n"
            f"  coupling at almost any latency. This system is effectively\n"
            f"  operating blind beyond τ ≈ {tau_h:.1f}."
        )

    lines += [
        "",
        "LEVERS TO EXTEND THE HORIZON:",
        f"  ↑ Increase P (signal power)    — current: {P:.3g}",
        f"  ↓ Reduce α (noise growth rate) — current: {alpha:.3g}",
        f"  ↓ Reduce ε (MI threshold)      — current: {epsilon}",
        "  ↓ Reduce latency directly      — shorten action-observation loop",
    ]

    if verbose:
        lines += [
            "",
            "─" * 62,
            "MODEL: σ²(τ) = σ0² + α·τ,  I(τ) = 0.5·log₂(1 + P/σ²(τ))",
            "τ_h = max(0, (P/(2^(2ε)−1) − σ0²) / α)",
            "doi:10.5281/zenodo.18663547",
        ]

    lines.append("=" * 62)
    return "\n".join(lines)


def explain_control_cliff(result, verbose: bool = True) -> str:
    """
    Plain-English explanation of a ControlCliffResult (E5).
    """
    cliff = result.cliff_delay
    cliff_mi = result.cliff_mi
    max_fail = float(result.failure_rate.max())
    min_mi = float(result.mi_bits.min())
    max_mi = float(result.mi_bits.max())

    severity = 3 if (cliff is not None and cliff < 20) else \
               2 if (cliff is not None and cliff < 40) else \
               1 if cliff is not None else 0

    lines = [
        "=" * 62,
        f"  CONTROL CLIFF DIAGNOSIS  —  {SEVERITY_LABELS[severity]}",
        "=" * 62,
        "",
        "WHAT IS HAPPENING:",
        "  As observation delay increases, the controller's mutual\n"
        "  information proxy (agency) decreases smoothly — but\n"
        "  system stability collapses abruptly at a specific cliff.",
        "",
        "KEY NUMBERS:",
    ]

    if cliff is not None:
        lines.append(f"  Control cliff at delay : d = {cliff} steps")
        lines.append(f"  MI at cliff            : {cliff_mi:.4f} bits")
    else:
        lines.append("  Control cliff          : not reached in sweep range")

    lines += [
        f"  Max failure rate       : {max_fail:.0%}",
        f"  MI range               : {min_mi:.3f} — {max_mi:.3f} bits",
        "",
        "WHAT THIS MEANS:",
    ]

    if cliff is None:
        lines.append(
            "  No control cliff detected in the simulated delay range.\n"
            "  The system remains stable throughout."
        )
    elif severity <= 1:
        lines.append(
            f"  The system tolerates moderate delay (cliff at d={cliff}).\n"
            f"  MI is still {cliff_mi:.3f} bits at the cliff — the threshold\n"
            f"  is relatively low. Keep delays below d={cliff} for stability."
        )
    elif severity == 2:
        lines.append(
            f"  Control cliff at d={cliff}. The system loses stability\n"
            f"  when delay reaches {cliff} steps, at which point MI has\n"
            f"  decayed to {cliff_mi:.3f} bits. This is a meaningful but\n"
            f"  reachable failure point — design conservatively."
        )
    else:
        lines.append(
            f"  Early control cliff at d={cliff}. Stability fails quickly.\n"
            f"  At MI = {cliff_mi:.3f} bits (delay = {cliff}), the controller\n"
            f"  can no longer stabilise the system. Very low tolerance\n"
            f"  for latency — real-time operation is essential."
        )

    if verbose:
        lines += [
            "",
            "─" * 62,
            "MODEL: x_{t+1} = a·x_t + u_t + w_t  (unstable AR(1))",
            "Agency proxy: I(d) = 0.5·log₂(1 + P/σ²(d))",
            "doi:10.5281/zenodo.18663547",
        ]

    lines.append("=" * 62)
    return "\n".join(lines)



def explain_asset(asset, verbose: bool = True) -> str:
    """
    Plain-English "Why flagged?" explanation for an AssetResult.

    Produces a self-contained card showing:
      - FAWP Score (0–100), status, days in regime
      - Prediction coupling tier
      - Steering coupling tier and collapse status
      - Leverage gap assessment
      - ODW presence and extent
      - Bullet-point reasons the alert fired
      - Recommended action

    Parameters
    ----------
    asset : AssetResult
        Output of WatchlistScanner or scan_watchlist.
    verbose : bool
        Include extended interpretation notes.

    Returns
    -------
    str
    """
    # ── Pull numbers from the latest scan window ──────────────────────────
    score_0_100 = int(round(asset.latest_score * 100))
    odw_str     = (
        f"detected (τ = {asset.peak_odw_start}–{asset.peak_odw_end})"
        if asset.peak_odw_start is not None
        else "not detected"
    )

    # pred/steer MI from latest window (if scan available)
    mean_pred  = 0.0
    mean_steer = 0.0
    n_fawp_windows = 0
    n_windows_total = 0
    steer_below_eps_taus = 0

    if asset.scan is not None:
        latest = asset.scan.latest
        pred_arr  = np.asarray(latest.pred_mi,  dtype=float)
        steer_arr = np.asarray(latest.steer_mi, dtype=float)
        if pred_arr.size > 0:
            mean_pred  = float(np.mean(pred_arr))
            mean_steer = float(np.mean(steer_arr))
            steer_below_eps_taus = int(np.sum(steer_arr < 0.01))

        windows = asset.scan.windows
        n_windows_total = len(windows)
        n_fawp_windows  = sum(1 for w in windows if w.fawp_found)

    # ── Coupling tiers ────────────────────────────────────────────────────
    def _pred_tier(mi: float) -> str:
        if mi > 1.0:   return f"very high ({mi:.3f} bits)"
        if mi > 0.3:   return f"elevated ({mi:.3f} bits)"
        if mi > 0.05:  return f"moderate ({mi:.3f} bits)"
        return f"low ({mi:.3f} bits)"

    def _steer_tier(mi: float) -> str:
        if mi < 0.005: return f"collapsed ({mi:.4f} bits ≈ zero)"
        if mi < 0.05:  return f"very weak ({mi:.4f} bits)"
        if mi < 0.2:   return f"weak ({mi:.3f} bits)"
        return f"present ({mi:.3f} bits)"

    gap_bits = asset.peak_gap_bits

    def _gap_tier(g: float) -> str:
        if g > 1.0:   return f"very large ({g:.3f} bits)"
        if g > 0.3:   return f"large ({g:.3f} bits)"
        if g > 0.05:  return f"moderate ({g:.3f} bits)"
        return f"small ({g:.3f} bits)"

    # ── Severity ──────────────────────────────────────────────────────────
    s = asset.latest_score
    if s >= 0.75:
        severity = "🚨 CRITICAL"
    elif s >= 0.50:
        severity = "🔴 HIGH"
    elif s >= 0.25:
        severity = "⚠️  MEDIUM"
    elif s > 0.01:
        severity = "🟡 LOW"
    else:
        severity = "✅ CLEAR"

    # ── Status line ───────────────────────────────────────────────────────
    if asset.regime_active:
        status_line = f"{severity}  —  FAWP ACTIVE  ({asset.days_in_regime} days)"
    elif asset.signal_age_days < 30:
        status_line = f"🟡 RECENTLY ACTIVE  (signal age {asset.signal_age_days}d)"
    else:
        status_line = "✅ CLEAR"

    # ── Why-flagged bullets ───────────────────────────────────────────────
    reasons: list[str] = []

    if asset.regime_active:
        reasons.append(
            f"Regime active: {asset.days_in_regime} day(s) since onset"
        )
    if n_fawp_windows > 0 and n_windows_total > 0:
        pct = int(round(100 * n_fawp_windows / n_windows_total))
        reasons.append(
            f"{n_fawp_windows}/{n_windows_total} scan windows flagged FAWP ({pct}%)"
        )
    if steer_below_eps_taus > 0:
        reasons.append(
            f"Steering MI below ε at {steer_below_eps_taus} tau value(s) "
            f"in latest window"
        )
    if gap_bits > 0.05:
        reasons.append(f"Leverage gap = {gap_bits:.4f} bits (pred MI exceeds steer MI)")
    if asset.peak_odw_start is not None:
        width = (asset.peak_odw_end or 0) - asset.peak_odw_start + 1
        reasons.append(
            f"ODW spans {width} tau step(s) "
            f"(τ {asset.peak_odw_start}–{asset.peak_odw_end})"
        )
    if mean_pred > 0.05 and mean_steer < 0.01:
        reasons.append(
            "Prediction MI significantly exceeds steering MI — "
            "Information-Control Exclusion Principle signature"
        )
    if not reasons:
        reasons.append("Score threshold crossed (mild signal, no dominant driver)")

    # ── Recommendation ────────────────────────────────────────────────────
    if s >= 0.50:
        rec = (
            "Prediction persists but execution edge has collapsed.\n"
            "  Your signals are still informative but the channel to act on\n"
            "  them is outside the agency horizon. Reduce execution aggression\n"
            "  or investigate latency / crowding conditions."
        )
    elif s >= 0.25:
        rec = (
            "Moderate FAWP signal. Monitor for escalation.\n"
            "  Investigate whether execution costs or signal crowding are rising.\n"
            "  Set a tighter alert threshold if this position is size-sensitive."
        )
    elif asset.regime_active:
        rec = (
            "Mild regime — worth watching.\n"
            "  No immediate action required, but schedule a rescan."
        )
    else:
        rec = "No active regime. System operating normally."

    # ── Assemble card ─────────────────────────────────────────────────────
    w = 60
    lines = [
        "=" * w,
        f"  {asset.ticker}  [{asset.timeframe}]",
        "=" * w,
        f"  FAWP Score   : {score_0_100}/100",
        f"  Status       : {status_line}",
        "-" * w,
        f"  Prediction   : {_pred_tier(mean_pred)}",
        f"  Steering     : {_steer_tier(mean_steer)}",
        f"  Leverage gap : {_gap_tier(gap_bits)}",
        f"  ODW          : {odw_str}",
        "-" * w,
        "  Why flagged:",
    ]
    for r in reasons:
        lines.append(f"    • {r}")
    lines += [
        "-" * w,
        "  Recommendation:",
        f"  {rec}",
    ]
    if verbose:
        lines += [
            "-" * w,
            "  Definitions:",
            "    Prediction MI  — I(return_t; return_{t+Δ}): is the market forecastable?",
            "    Steering MI    — I(flow_t; return_{t+τ}): do your orders move price?",
            "    Leverage gap   — pred MI − steer MI: the FAWP signal width",
            "    ODW            — Operational Detection Window: tau range of FAWP",
            "    doi:10.5281/zenodo.18673949",
        ]
    lines.append("=" * w)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Top-level dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def explain(result, verbose: bool = True) -> str:
    """
    Auto-dispatch explain to the correct function based on result type.

    Handles: FAWPResult, OATSResult, ControlCliffResult, SimulationResult,
    AssetResult, dict.
    """
    cls_name = type(result).__name__
    if cls_name == "FAWPResult":
        return explain_fawp(result, verbose=verbose)
    if cls_name == "OATSResult":
        return explain_oats(result, verbose=verbose)
    if cls_name == "ControlCliffResult":
        return explain_control_cliff(result, verbose=verbose)
    if cls_name in ("SimulationResult", "FAWPSimulationResult"):
        return explain_fawp(result, verbose=verbose)
    if cls_name == "AssetResult":
        return explain_asset(result, verbose=verbose)
    if isinstance(result, dict):
        return "\n".join(f"  {k}: {v}" for k, v in result.items())
    return f"No explain handler for type '{cls_name}'."


# ─────────────────────────────────────────────────────────────────────────────
# Attribution — per-tau and per-window contribution analysis
# ─────────────────────────────────────────────────────────────────────────────

def attribute_gap(window, top_n: int = 5) -> dict:
    """
    Per-tau attribution of the leverage gap for a single scan window.

    For each tau value, computes:
      - gap(τ)      = pred_MI(τ) − steer_MI(τ)
      - share(τ)    = gap(τ) / sum(gap) — fractional contribution to total gap
      - pred_share  = pred_MI(τ) / sum(pred_MI) — pred contribution weight
      - steer_share = steer_MI(τ) / sum(steer_MI) — steer contribution weight
      - inside_odw  = bool — whether this tau is inside the ODW

    The ``top_tau`` list identifies which specific lag values are driving
    the divergence — the FAWP signal "hotspots".

    Parameters
    ----------
    window : MarketWindowResult
        A single scan window (e.g. ``asset.scan.latest``).
    top_n : int
        Number of top tau values to highlight.

    Returns
    -------
    dict with keys:
        tau           : list[int]
        gap           : list[float]
        share         : list[float]   — fractional contribution (sums to 1)
        pred_share    : list[float]
        steer_share   : list[float]
        inside_odw    : list[bool]
        top_tau       : list[int]     — top_n tau values by gap share
        top_shares    : list[float]   — their share values
        total_gap     : float
        odw_share     : float         — fraction of gap inside ODW
        peak_tau      : int           — single tau with largest gap

    Example
    -------
    ::

        from fawp_index.watchlist import scan_watchlist
        from fawp_index.explain import attribute_gap

        result = scan_watchlist(["SPY"], period="2y")
        top_asset = result.rank_by("score")[0]
        attr = attribute_gap(top_asset.scan.latest)

        print(f"Peak gap at tau={attr['peak_tau']}")
        print(f"ODW captures {attr['odw_share']*100:.1f}% of total gap")
        for tau, share in zip(attr['top_tau'], attr['top_shares']):
            print(f"  tau={tau:>3}  share={share*100:.1f}%")
    """
    import numpy as np

    tau      = np.asarray(window.tau,      dtype=float)
    pred_mi  = np.asarray(window.pred_mi,  dtype=float)
    steer_mi = np.asarray(window.steer_mi, dtype=float)
    gap      = np.maximum(0.0, pred_mi - steer_mi)

    total_gap  = float(gap.sum())
    total_pred = float(pred_mi.sum())
    total_steer= float(steer_mi.sum())

    eps = 1e-12
    share       = gap       / (total_gap   + eps)
    pred_share  = pred_mi   / (total_pred  + eps)
    steer_share = steer_mi  / (total_steer + eps)

    odw = window.odw_result
    inside_odw = [
        bool(odw.odw_start is not None
             and odw.odw_start <= int(t) <= odw.odw_end)
        for t in tau
    ]
    odw_share = float(gap[inside_odw].sum()) / (total_gap + eps) \
                if any(inside_odw) else 0.0

    # Top-N by gap share
    order     = np.argsort(gap)[::-1]
    top_idx   = order[:top_n]
    top_tau   = [int(tau[i]) for i in top_idx]
    top_shares= [float(share[i]) for i in top_idx]

    peak_tau  = int(tau[int(np.argmax(gap))])

    return {
        "tau":         [int(t) for t in tau],
        "gap":         [round(float(g), 6) for g in gap],
        "share":       [round(float(s), 4) for s in share],
        "pred_share":  [round(float(s), 4) for s in pred_share],
        "steer_share": [round(float(s), 4) for s in steer_share],
        "inside_odw":  inside_odw,
        "top_tau":     top_tau,
        "top_shares":  [round(s, 4) for s in top_shares],
        "total_gap":   round(total_gap, 6),
        "odw_share":   round(odw_share, 4),
        "peak_tau":    peak_tau,
    }


def attribute_windows(asset, top_n: int = 5) -> dict:
    """
    Per-window attribution of the FAWP regime for an AssetResult.

    Ranks every scan window by its regime score and measures how much
    each window contributes to the overall FAWP signal.  Identifies
    the onset window (first FAWP), peak window (highest score), and the
    most recent window.

    Parameters
    ----------
    asset : AssetResult
    top_n : int
        Number of top windows to return.

    Returns
    -------
    dict with keys:
        dates           : list[str]
        scores          : list[float]
        fawp_flags      : list[bool]
        gap_bits        : list[float]
        top_windows     : list[dict]  — top_n windows by score
        onset_date      : str or None — first window where FAWP fired
        peak_date       : str
        peak_score      : float
        n_fawp_windows  : int
        n_total_windows : int
        fawp_fraction   : float
        score_slope     : float       — linear trend of last 10 windows

    Example
    -------
    ::

        from fawp_index.explain import attribute_windows

        result = scan_watchlist(["SPY"], period="2y")
        top = result.rank_by("score")[0]
        attr = attribute_windows(top)

        print(f"FAWP in {attr['n_fawp_windows']}/{attr['n_total_windows']} windows")
        print(f"Onset: {attr['onset_date']}  Peak: {attr['peak_date']}")
        print(f"Score trend: {attr['score_slope']:+.4f}/window")
    """
    import numpy as np

    if asset.scan is None:
        return {}

    windows = asset.scan.windows
    dates   = [str(w.date.date()) for w in windows]
    scores  = [float(w.regime_score) for w in windows]
    flags   = [bool(w.fawp_found)   for w in windows]
    gaps    = [float(w.odw_result.peak_gap_bits) for w in windows]

    n_total  = len(windows)
    n_fawp   = sum(flags)
    peak_idx = int(np.argmax(scores))

    # Onset: first window where FAWP fired
    onset_date = None
    for w in windows:
        if w.fawp_found:
            onset_date = str(w.date.date())
            break

    # Score slope over last 10 windows
    recent_scores = np.array(scores[-10:])
    n_r = len(recent_scores)
    if n_r >= 3:
        x = np.arange(n_r, dtype=float)
        score_slope = float(np.polyfit(x, recent_scores, 1)[0])
    else:
        score_slope = 0.0

    # Top-N windows by score
    order = np.argsort(scores)[::-1][:top_n]
    top_windows = [
        {
            "date":  dates[i],
            "score": round(scores[i], 4),
            "fawp":  flags[i],
            "gap":   round(gaps[i], 4),
        }
        for i in order
    ]

    return {
        "dates":           dates,
        "scores":          [round(s, 4) for s in scores],
        "fawp_flags":      flags,
        "gap_bits":        [round(g, 4) for g in gaps],
        "top_windows":     top_windows,
        "onset_date":      onset_date,
        "peak_date":       dates[peak_idx],
        "peak_score":      round(scores[peak_idx], 4),
        "n_fawp_windows":  n_fawp,
        "n_total_windows": n_total,
        "fawp_fraction":   round(n_fawp / n_total, 4) if n_total else 0.0,
        "score_slope":     round(score_slope, 6),
    }


def attribution_report(asset, top_n: int = 5) -> str:
    """
    Combined attribution report: per-tau gap + per-window timeline.

    Builds a plain-English narrative of *why* a gap appeared:
    which tau lags drove the divergence, which windows were most active,
    when the regime started, and whether the signal is accelerating.

    Parameters
    ----------
    asset : AssetResult
    top_n : int

    Returns
    -------
    str

    Example
    -------
    ::

        from fawp_index.explain import attribution_report

        result = scan_watchlist(["SPY"], period="2y")
        top = result.rank_by("score")[0]
        print(attribution_report(top))
    """
    w = 62
    lines = [
        "=" * w,
        f"  Attribution Report — {asset.ticker} [{asset.timeframe}]",
        "=" * w,
    ]

    # ── Per-tau attribution (latest window) ──────────────────────────────
    if asset.scan is not None:
        attr_tau = attribute_gap(asset.scan.latest, top_n=top_n)
        lines += [
            "",
            "TAU-LEVEL ATTRIBUTION  (latest window)",
            f"  Total leverage gap : {attr_tau['total_gap']:.4f} bits",
            f"  Peak tau           : τ = {attr_tau['peak_tau']}",
            f"  ODW captures       : {attr_tau['odw_share']*100:.1f}% of total gap",
            "",
            f"  Top {top_n} tau values by gap contribution:",
            f"  {'τ':>4}  {'Gap share':>9}  {'Pred share':>10}  {'Steer share':>11}  {'In ODW':>6}",
            "  " + "-" * 50,
        ]
        tau_list   = attr_tau["tau"]
        gap_list   = attr_tau["gap"]
        share_list = attr_tau["share"]
        ps_list    = attr_tau["pred_share"]
        ss_list    = attr_tau["steer_share"]
        odw_list   = attr_tau["inside_odw"]

        for t, sh in zip(attr_tau["top_tau"], attr_tau["top_shares"]):
            idx    = tau_list.index(t)
            in_odw = "✓" if odw_list[idx] else "—"
            bar    = "█" * max(1, int(sh * 30))
            lines.append(
                f"  {t:>4}  {sh*100:>8.1f}%  "
                f"{ps_list[idx]*100:>9.1f}%  "
                f"{ss_list[idx]*100:>10.1f}%  "
                f"{in_odw:>6}  {bar}"
            )

        # Narrative
        top_t = attr_tau["top_tau"][0]
        top_s = attr_tau["top_shares"][0]
        odw_s = attr_tau["odw_share"]
        lines += [
            "",
            "  Narrative:",
        ]
        if odw_s > 0.5:
            lines.append(
                f"  The gap is concentrated inside the ODW "
                f"({odw_s*100:.0f}% of signal). "
                f"Tau={top_t} is the dominant lag."
            )
        elif odw_s > 0.2:
            lines.append(
                f"  The ODW captures {odw_s*100:.0f}% of the gap — "
                f"moderate concentration. "
                f"Signal spreads across multiple lags, peak at τ={top_t}."
            )
        else:
            lines.append(
                f"  Gap is diffuse across tau range. "
                f"No single ODW lag dominates. "
                f"Peak contribution at τ={top_t} ({top_s*100:.0f}%)."
            )

    # ── Per-window attribution ────────────────────────────────────────────
    attr_win = attribute_windows(asset, top_n=top_n)
    if attr_win:
        slope     = attr_win["score_slope"]
        slope_str = f"+{slope:.4f}/win ▲ accelerating" if slope > 1e-4 else (
                    f"{slope:.4f}/win ▼ decelerating" if slope < -1e-4 else
                    "~0 flat")
        lines += [
            "",
            "WINDOW-LEVEL ATTRIBUTION",
            f"  FAWP windows  : {attr_win['n_fawp_windows']} / {attr_win['n_total_windows']} "
            f"({attr_win['fawp_fraction']*100:.0f}%)",
            f"  Onset date    : {attr_win['onset_date'] or '—'}",
            f"  Peak date     : {attr_win['peak_date']}  (score {attr_win['peak_score']:.4f})",
            f"  Score trend   : {slope_str}",
            "",
            f"  Top {top_n} windows by score:",
            f"  {'Date':<12} {'Score':>6}  {'Gap':>7}  FAWP",
            "  " + "-" * 35,
        ]
        for win in attr_win["top_windows"]:
            flag = "YES" if win["fawp"] else "—"
            lines.append(
                f"  {win['date']:<12} {win['score']:>6.4f}  "
                f"{win['gap']:>7.4f}  {flag}"
            )

    lines += ["", "=" * w]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Confidence badge — null-calibrated signal quality tier
# ─────────────────────────────────────────────────────────────────────────────

def confidence_badge(asset, n_bootstrap: int = 50, seed: int = 42) -> dict:
    """
    Compute a null-calibrated confidence tier for an AssetResult.

    Does NOT run a full bootstrap (too slow for a dashboard scan).
    Instead uses three fast heuristics derived from the same foundations
    as the bootstrap significance test:

    1. **Gap persistence** — fraction of the last 10 windows where
       pred MI > steer MI above the null floor.  High persistence →
       signal is not a one-off spike.

    2. **ODW concentration** — share of the total leverage gap
       that falls inside the Operational Detection Window.
       High concentration → signal is structurally focused, not diffuse.

    3. **Score stability** — coefficient of variation (CV) of regime
       scores across the last 10 windows.  Low CV → signal is stable.

    The three components are combined into a single confidence score
    (0–1) which maps to LOW / MEDIUM / HIGH.

    Parameters
    ----------
    asset : AssetResult
    n_bootstrap : int
        Unused — kept for API compatibility with future full bootstrap.
    seed : int
        Unused — kept for API compatibility.

    Returns
    -------
    dict with keys:
        tier         : str   — "HIGH" | "MEDIUM" | "LOW" | "INSUFFICIENT"
        score        : float — composite confidence score (0–1)
        persistence  : float — gap persistence fraction (0–1)
        concentration: float — ODW gap concentration (0–1)
        stability    : float — score stability (0–1, higher = more stable)
        n_windows    : int   — number of windows used

    Example
    -------
    ::

        from fawp_index.explain import confidence_badge

        result = scan_watchlist(["SPY", "QQQ"], period="2y")
        for a in result.rank_by("score")[:5]:
            badge = confidence_badge(a)
            print(f"{a.ticker}  {badge['tier']}  ({badge['score']:.2f})")
    """
    # Fast path: no scan data
    if asset.scan is None or len(asset.scan.windows) < 3:
        return {
            "tier": "INSUFFICIENT", "score": 0.0,
            "persistence": 0.0, "concentration": 0.0,
            "stability": 0.0, "n_windows": 0,
        }

    windows = asset.scan.windows[-10:]  # last 10 windows
    n = len(windows)

    # ── 1. Gap persistence ────────────────────────────────────────────────
    # Fraction of windows where mean(pred_mi) > mean(steer_mi) by > epsilon
    eps = 1e-4
    persist_count = 0
    for w in windows:
        pm = float(np.mean(w.pred_mi))  if len(w.pred_mi)  > 0 else 0.0
        sm = float(np.mean(w.steer_mi)) if len(w.steer_mi) > 0 else 0.0
        if pm - sm > eps:
            persist_count += 1
    persistence = persist_count / n

    # ── 2. ODW concentration ──────────────────────────────────────────────
    # How focused is the gap inside the ODW in the latest window?
    concentration = 0.0
    latest = asset.scan.latest
    if asset.peak_odw_start is not None:
        gap = np.maximum(0, latest.pred_mi - latest.steer_mi)
        total = float(gap.sum())
        if total > 1e-12:
            odw_mask = np.array([
                asset.peak_odw_start <= int(t) <= (asset.peak_odw_end or 0)
                for t in latest.tau
            ])
            concentration = float(gap[odw_mask].sum()) / total

    # ── 3. Score stability ────────────────────────────────────────────────
    # 1 - CV (coefficient of variation) of regime scores; clipped to [0,1]
    scores = np.array([float(w.regime_score) for w in windows])
    mean_s = float(scores.mean())
    if mean_s > 1e-8:
        cv = float(scores.std()) / mean_s
        stability = float(np.clip(1.0 - cv, 0.0, 1.0))
    else:
        stability = 0.0

    # ── Composite score ───────────────────────────────────────────────────
    # Weighted: persistence matters most, then concentration, then stability
    composite = 0.50 * persistence + 0.30 * concentration + 0.20 * stability

    # ── Tier mapping ──────────────────────────────────────────────────────
    # Require regime_active for HIGH; non-active assets capped at MEDIUM
    if composite >= 0.65 and asset.regime_active:
        tier = "HIGH"
    elif composite >= 0.35:
        tier = "MEDIUM"
    else:
        tier = "LOW"

    return {
        "tier":          tier,
        "score":         round(composite, 3),
        "persistence":   round(persistence, 3),
        "concentration": round(concentration, 3),
        "stability":     round(stability, 3),
        "n_windows":     n,
    }
