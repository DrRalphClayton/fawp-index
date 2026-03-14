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
