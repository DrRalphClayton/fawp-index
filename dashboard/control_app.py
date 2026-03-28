"""
FAWP Dynamic Systems Scanner
Upload any state + action time series and detect the Information-Control
Exclusion Principle — when prediction persists but steering authority collapses.

Supports:
  - ML training logs (loss + gradient norm)
  - Control systems (plant output + actuator signal)
  - State estimation (filter residual + correction)
  - Custom time series (any two columns)
"""

import streamlit as st
import pandas as pd
import numpy as np
import io

# ── Domain presets ─────────────────────────────────────────────────────────────
DOMAINS = {
    "⚙️ Control System": {
        "state_label":  "Plant output (state variable)",
        "action_label": "Actuator / controller signal",
        "desc": "Detects when the plant output remains predictable but your controller has lost causal grip on it — the operational definition of control authority collapse.",
        "state_col":    "state",
        "action_col":   "action",
        "example_csv":  "step,state,action\n0,0.0,0.0\n1,0.12,0.10\n2,0.28,0.22\n",
    },
    "🤖 ML Training": {
        "state_label":  "Validation loss (output to predict)",
        "action_label": "Gradient norm (intervention signal)",
        "desc": "Detects when val loss is still forecastable but gradient updates have lost steering authority — early warning of overfitting or training collapse, before it's visible in the loss curve.",
        "state_col":    "val_loss",
        "action_col":   "grad_norm",
        "example_csv":  "epoch,val_loss,grad_norm\n0,2.31,1.20\n1,1.85,0.98\n2,1.42,0.71\n",
    },
    "📡 State Estimation": {
        "state_label":  "Filter residual / innovation",
        "action_label": "Correction magnitude (Kalman gain × residual)",
        "desc": "Detects when residuals remain structured and forecastable but the filter corrections are no longer reducing uncertainty — estimation divergence precursor.",
        "state_col":    "residual",
        "action_col":   "correction",
        "example_csv":  "t,residual,correction\n0,0.05,0.04\n1,0.08,0.06\n2,0.11,0.07\n",
    },
    "📊 Custom": {
        "state_label":  "State / output column",
        "action_label": "Action / intervention column",
        "desc": "Upload any two-column time series. The FAWP detector measures whether prediction MI persists after steering MI collapses.",
        "state_col":    "",
        "action_col":   "",
        "example_csv":  "t,state,action\n0,1.0,0.5\n1,1.2,0.6\n2,1.5,0.4\n",
    },
}

# ── Page header ────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;gap:.6em;margin-bottom:.2em">
  <span style="font-size:1.6em">⚙️</span>
  <span style="font-family:'Syne',sans-serif;font-size:1.5em;font-weight:800;
               color:#D4AF37">FAWP Dynamic Systems</span>
</div>
<div style="color:#5A8ABA;font-size:.88em;margin-bottom:1.2em">
  Information-Control Exclusion Principle · Upload any state + action time series
</div>
""", unsafe_allow_html=True)

# ── Sidebar config ─────────────────────────────────────────────────────────────
st.sidebar.markdown("### ⚙️ Dynamic Systems Config")

domain_name = st.sidebar.selectbox("Domain", list(DOMAINS.keys()), key="ctrl_domain")
domain = DOMAINS[domain_name]

st.sidebar.markdown(f"<div style='font-size:.8em;color:#5A8ABA;margin-bottom:.6em'>{domain['desc']}</div>",
                    unsafe_allow_html=True)

tau_max  = st.sidebar.slider("τ max (lag steps)", 5, 80, 40, key="ctrl_tau_max")
window   = st.sidebar.slider("Rolling window", 20, 500, 100, key="ctrl_window")
step     = st.sidebar.slider("Step size", 1, 50, 10, key="ctrl_step")
epsilon  = st.sidebar.number_input("ε (null threshold, bits)", 0.001, 0.1, 0.01,
                                    step=0.001, format="%.3f", key="ctrl_eps")
n_null   = st.sidebar.slider("Null permutations", 0, 200, 50, key="ctrl_n_null")
horizon  = st.sidebar.slider("Horizon Δ (steps ahead)", 1, 50, 10, key="ctrl_horizon")

# ── Data source ────────────────────────────────────────────────────────────────
st.markdown("#### 📂 Load data")
data_src = st.radio("Source",
                    ["Upload CSV", "Batch upload (multiple CSVs)",
                     "Simulate AR(1) system (E8 demo)",
                     "Example: ML training log",
                     "Example: Double pendulum"],
                    horizontal=True, key="ctrl_src")

df = None

if data_src == "Batch upload (multiple CSVs)":
    batch_files = st.file_uploader("Upload multiple CSV files (each = one system)",
                                   type=["csv"], accept_multiple_files=True, key="ctrl_batch")
    if batch_files:
        st.info(f"Loaded {len(batch_files)} file(s). Scanning each with current settings.")
        if st.button("▶ Run Batch FAWP Scan", type="primary", key="ctrl_batch_run"):
            from fawp_index.core.estimators import mi_from_arrays, conservative_null_floor
            from fawp_index.detection.odw import ODWDetector
            from fawp_index.constants import PERSISTENCE_RULE_M, PERSISTENCE_RULE_N
            import numpy as _np_b
            _batch_rows = []
            _prog_b = st.progress(0.0, "Scanning…")
            for _bi, _bf in enumerate(batch_files):
                try:
                    _df_b = pd.read_csv(_bf)
                    _cols_b = list(_df_b.columns)
                    _sc = domain["state_col"]  if domain["state_col"]  in _cols_b else _cols_b[0]
                    _ac = domain["action_col"] if domain["action_col"] in _cols_b else (_cols_b[1] if len(_cols_b)>1 else _cols_b[0])
                    _sa = _df_b[_sc].values.astype(float)
                    _aa = _df_b[_ac].values.astype(float)
                    _mask_b = _np_b.isfinite(_sa) & _np_b.isfinite(_aa)
                    _sa = _sa[_mask_b]; _aa = _aa[_mask_b]
                    _tau_b = _np_b.arange(1, tau_max + 1)
                    _pm_b = _np_b.zeros(len(_tau_b)); _sm_b = _np_b.zeros(len(_tau_b))
                    for _ti, _tau in enumerate(_tau_b):
                        _xp = _sa[:-_tau]; _yp = _sa[_tau:]
                        _pm_b[_ti] = max(0.0, mi_from_arrays(_xp,_yp) - conservative_null_floor(_xp,_yp,n_null,0.99))
                        _xs = _aa[:-_tau]; _ys = _sa[_tau:]
                        _sm_b[_ti] = max(0.0, mi_from_arrays(_xs,_ys) - conservative_null_floor(_xs,_ys,n_null,0.99))
                    _odw_b = ODWDetector(epsilon=epsilon, persistence_m=PERSISTENCE_RULE_M,
                                        persistence_n=PERSISTENCE_RULE_N).detect(
                        tau=_tau_b, pred_corr=_pm_b, steer_corr=_sm_b,
                        fail_rate=_np_b.zeros(len(_tau_b)))
                    _batch_rows.append({
                        "File": _bf.name, "FAWP": "🔴 YES" if _odw_b.fawp_found else "—",
                        "Peak Gap (bits)": round(float(_odw_b.peak_gap_bits or 0),4),
                        "τ⁺ₕ": _odw_b.tau_h_plus or "—", "τf": _odw_b.tau_f or "—",
                        "ODW": f"{_odw_b.odw_start}–{_odw_b.odw_end}" if _odw_b.odw_start else "—",
                        "n_obs": int(_mask_b.sum()),
                    })
                except Exception as _be:
                    _batch_rows.append({"File": _bf.name, "FAWP": "❌ Error", "Peak Gap (bits)": 0,
                                        "τ⁺ₕ": "—", "τf": "—", "ODW": str(_be)[:40], "n_obs": 0})
                _prog_b.progress((_bi+1)/len(batch_files), f"Scanned {_bi+1}/{len(batch_files)}")
            _prog_b.empty()
            _bdf = pd.DataFrame(_batch_rows)
            st.dataframe(_bdf, use_container_width=True, hide_index=True)
            _n_fawp_b = (_bdf["FAWP"] == "🔴 YES").sum()
            if _n_fawp_b:
                st.error(f"🔴 {_n_fawp_b}/{len(batch_files)} systems in FAWP")
            else:
                st.success(f"✅ No FAWP detected across {len(batch_files)} systems")
            st.download_button("⬇ Export batch results CSV",
                               data=_bdf.to_csv(index=False).encode(),
                               file_name="fawp_batch_results.csv", mime="text/csv")

elif data_src == "Upload CSV":
    col_up1, col_up2 = st.columns([2, 1])
    with col_up1:
        uploaded = st.file_uploader("Upload CSV with state + action columns",
                                    type=["csv"], key="ctrl_upload")
    with col_up2:
        st.caption("Expected format:")
        st.code(domain["example_csv"], language="text")

    if uploaded:
        df = pd.read_csv(uploaded)
        st.success(f"✅ Loaded: {len(df)} rows × {len(df.columns)} columns")
        st.dataframe(df.head(5), use_container_width=True)

    # Column mapping
    if df is not None:
        cols = list(df.columns)
        default_state  = domain["state_col"]  if domain["state_col"]  in cols else cols[0]
        default_action = domain["action_col"] if domain["action_col"] in cols else (cols[1] if len(cols) > 1 else cols[0])
        c1, c2 = st.columns(2)
        with c1:
            state_col  = st.selectbox(domain["state_label"],  cols,
                                       index=cols.index(default_state),  key="ctrl_state_col")
        with c2:
            action_col = st.selectbox(domain["action_label"], cols,
                                       index=cols.index(default_action), key="ctrl_action_col")

elif data_src == "Example: ML training log":
    # Synthetic ML training: val loss decays, grad norm collapses after epoch 60
    import numpy as _np_ml
    _rng_ml = _np_ml.random.default_rng(0)
    _n_ep = 200
    _ep = _np_ml.arange(_n_ep)
    _val_loss = 2.3 * _np_ml.exp(-_ep / 40) + 0.3 + _rng_ml.normal(0, 0.05, _n_ep)
    _grad_norm = _np_ml.where(_ep < 60,
                  1.5 * _np_ml.exp(-_ep / 30) + 0.3 + _rng_ml.normal(0, 0.08, _n_ep),
                  0.02 + _rng_ml.normal(0, 0.01, _n_ep).clip(0))
    df = pd.DataFrame({"epoch": _ep, "val_loss": _val_loss, "grad_norm": _grad_norm})
    state_col = "val_loss"; action_col = "grad_norm"
    st.success("✅ Synthetic ML training log: val_loss (state) + grad_norm (action), 200 epochs")
    st.caption("Gradient norm collapses at epoch ~60 while val_loss remains structured — classic FAWP pattern for training collapse.")

elif data_src == "Example: Double pendulum":
    # Double pendulum: angle θ₁ (state) + torque input (action), chaotic after t~150
    import numpy as _np_dp
    _rng_dp = _np_dp.random.default_rng(7)
    _n_dp = 400
    _t_dp = _np_dp.linspace(0, 20, _n_dp)
    _theta = 0.3 * _np_dp.sin(2.5 * _t_dp) + 0.15 * _np_dp.sin(5.1 * _t_dp)
    _theta += _np_dp.where(_t_dp > 10,
                           _rng_dp.normal(0, 0.4, _n_dp) * (_t_dp - 10) / 10, 0)
    _torque = -0.8 * _theta + _rng_dp.normal(0, 0.05, _n_dp)
    _torque[_t_dp > 10] *= _np_dp.exp(-(_t_dp[_t_dp > 10] - 10) / 3)
    df = pd.DataFrame({"t": _t_dp, "theta1": _theta, "torque": _torque})
    state_col = "theta1"; action_col = "torque"
    st.success("✅ Synthetic double pendulum: θ₁ angle (state) + torque input (action), 400 steps")
    st.caption("Torque coupling collapses into chaotic regime (~step 200) while θ₁ remains predictable — FAWP in nonlinear dynamics.")

else:
    # Simulate AR(1) delayed-feedback system (E8)
    st.info("Simulating the E8 flagship system: a=1.02, K=0.8, Δ=20, n=1000 steps")
    sim_col1, sim_col2, sim_col3 = st.columns(3)
    with sim_col1: a_gain = st.number_input("AR(1) gain a", 0.9, 1.2, 1.02, 0.01, key="ctrl_a")
    with sim_col2: K_gain = st.number_input("Controller gain K", 0.1, 2.0, 0.8, 0.1, key="ctrl_K")
    with sim_col3: delay  = st.number_input("Observation delay Δ", 1, 50, 20, key="ctrl_delay")

    rng = np.random.default_rng(42)
    n_steps = 1000
    x = np.zeros(n_steps)
    u = np.zeros(n_steps)
    for t in range(1, n_steps):
        obs_t = x[max(0, t - int(delay))]
        u[t]  = -K_gain * obs_t
        x[t]  = a_gain * x[t-1] + u[t] + rng.normal(0, 0.1)
        if abs(x[t]) > 500: x[t] = np.sign(x[t]) * 500

    df = pd.DataFrame({"step": np.arange(n_steps), "state": x, "action": u})
    state_col  = "state"
    action_col = "action"
    st.success(f"✅ Simulated {n_steps} steps  ·  a={a_gain}  K={K_gain}  Δ={delay}")

    fig_sim, ax_sim = None, None
    try:
        import matplotlib.pyplot as _plt_sim
        _fig_sim, (_ax_x, _ax_u) = _plt_sim.subplots(2, 1, figsize=(9, 3),
                                                       facecolor="#0D1729", sharex=True)
        for _ax in (_ax_x, _ax_u):
            _ax.set_facecolor("#07101E")
            for _sp in _ax.spines.values(): _sp.set_edgecolor("#3A4E70")
            _ax.tick_params(colors="#7A90B8", labelsize=7)
        _ax_x.plot(df["state"],  color="#D4AF37", lw=0.8, alpha=0.8)
        _ax_u.plot(df["action"], color="#4A7FCC", lw=0.8, alpha=0.8)
        _ax_x.set_ylabel("State x(t)", fontsize=8, color="#D4AF37")
        _ax_u.set_ylabel("Action u(t)", fontsize=8, color="#4A7FCC")
        _ax_u.set_xlabel("Step", fontsize=8, color="#7A90B8")
        _fig_sim.tight_layout(pad=0.5)
        st.pyplot(_fig_sim, use_container_width=True)
        _plt_sim.close(_fig_sim)
    except Exception:
        pass

# ── Run scan ───────────────────────────────────────────────────────────────────
run_btn = st.button("▶ Run FAWP Scan", type="primary",
                    use_container_width=True, key="ctrl_run")

if run_btn and df is not None:
    try:
        from fawp_index import FAWPAlphaIndex
        from fawp_index.detection.odw import ODWDetector
        from fawp_index.constants import PERSISTENCE_RULE_M, PERSISTENCE_RULE_N

        state_arr  = df[state_col].values.astype(float)
        action_arr = df[action_col].values.astype(float)

        # Remove NaN
        mask = np.isfinite(state_arr) & np.isfinite(action_arr)
        state_arr  = state_arr[mask]
        action_arr = action_arr[mask]

        if len(state_arr) < window + tau_max:
            st.error(f"Need at least {window + tau_max} valid rows. Got {len(state_arr)}.")
            st.stop()

        with st.spinner("Computing MI curves…"):
            # Compute pred MI (state predicts future state) and steer MI (action → state)
            tau_arr   = np.arange(1, tau_max + 1)
            pred_mi   = np.zeros(len(tau_arr))
            steer_mi  = np.zeros(len(tau_arr))
            null_pred = np.zeros(len(tau_arr))
            null_steer= np.zeros(len(tau_arr))

            from fawp_index.core.estimators import mi_from_arrays, conservative_null_floor

            for ti, tau in enumerate(tau_arr):
                # Prediction: state[t] → state[t+tau]
                x_p = state_arr[:-tau]
                y_p = state_arr[tau:]
                raw_p = mi_from_arrays(x_p, y_p)
                floor_p = conservative_null_floor(x_p, y_p, n_null, quantile=0.99)
                pred_mi[ti] = max(0.0, raw_p - floor_p)

                # Steering: action[t] → state[t+tau]
                x_s = action_arr[:-tau]
                y_s = state_arr[tau:]
                raw_s = mi_from_arrays(x_s, y_s)
                floor_s = conservative_null_floor(x_s, y_s, n_null, quantile=0.99)
                steer_mi[ti] = max(0.0, raw_s - floor_s)

            # ODW detection
            det = ODWDetector(epsilon=epsilon,
                              persistence_m=PERSISTENCE_RULE_M,
                              persistence_n=PERSISTENCE_RULE_N)
            odw = det.detect(tau=tau_arr, pred_corr=pred_mi,
                             steer_corr=steer_mi,
                             fail_rate=np.zeros(len(tau_arr)))

        st.session_state["dynamo_result"] = {
            "odw": odw, "tau": tau_arr,
            "pred_mi": pred_mi, "steer_mi": steer_mi,
            "epsilon": epsilon, "domain": domain_name,
            "n_obs": len(state_arr),
        }
        st.rerun()

    except Exception as e:
        st.error(f"Scan failed: {e}")
        import traceback; st.code(traceback.format_exc())

# ── Results ────────────────────────────────────────────────────────────────────
if "dynamo_result" in st.session_state:
    r = st.session_state["dynamo_result"]
    odw = r["odw"]

    # KPI row
    fawp_found = bool(odw.fawp_found)
    peak_gap   = float(odw.peak_gap_bits) if odw.peak_gap_bits else 0.0
    tau_h      = odw.tau_h_plus or "—"
    tau_f      = odw.tau_f      or "—"
    odw_start  = odw.odw_start  or "—"
    odw_end    = odw.odw_end    or "—"

    regime_color = "#C0111A" if fawp_found else "#1DB954"
    regime_label = "🔴 FAWP DETECTED" if fawp_found else "✅ No FAWP"

    st.markdown(f"""
<div style="background:{'rgba(192,17,26,0.12)' if fawp_found else 'rgba(29,185,84,0.10)'};
border:1px solid {'#C0111A' if fawp_found else '#1DB954'};border-radius:8px;
padding:.8em 1.2em;margin-bottom:1em;display:flex;align-items:center;gap:.8em">
  <span style="font-size:1.3em;font-weight:800;color:{regime_color}">{regime_label}</span>
  <span style="color:#7A90B8;font-size:.85em">
    · Peak gap {peak_gap:.4f} bits · τ⁺ₕ = {tau_h} · τf = {tau_f}
    · ODW τ = {odw_start}–{odw_end} · {r['n_obs']} observations
  </span>
</div>
""", unsafe_allow_html=True)

    # KPI cards
    kc1, kc2, kc3, kc4 = st.columns(4)
    def _kpi(col, val, lbl, color="#D4AF37"):
        col.markdown(f"""<div style="background:#0D1729;border:1px solid #1E3050;border-radius:6px;
padding:.5em;text-align:center"><div style="font-size:1.4em;font-weight:800;color:{color}">{val}</div>
<div style="font-size:.72em;color:#5A8ABA;text-transform:uppercase">{lbl}</div></div>""",
                     unsafe_allow_html=True)
    _kpi(kc1, f"{peak_gap:.4f}", "Peak Gap (bits)", regime_color)
    _kpi(kc2, str(tau_h), "τ⁺ₕ horizon")
    _kpi(kc3, str(tau_f), "τf cliff")
    _kpi(kc4, f"τ{odw_start}–{odw_end}", "ODW")

    st.markdown("")

    # MI chart
    try:
        import matplotlib.pyplot as _plt_r
        _fig_r, _ax_r = _plt_r.subplots(figsize=(9, 3.5), facecolor="#0D1729")
        _ax_r.set_facecolor("#07101E")
        for _sp in _ax_r.spines.values(): _sp.set_edgecolor("#3A4E70")
        _ax_r.tick_params(colors="#7A90B8", labelsize=8)

        tau_arr = r["tau"]; pred_mi = r["pred_mi"]; steer_mi = r["steer_mi"]
        _ax_r.plot(tau_arr, pred_mi,  color="#D4AF37", lw=2,   label="Prediction MI")
        _ax_r.plot(tau_arr, steer_mi, color="#4A7FCC", lw=1.5, ls="--", label="Steering MI")
        _ax_r.axhline(epsilon, color="#3A4E70", ls=":", lw=1, label=f"ε = {epsilon:.3f}")

        if odw.odw_start and odw.odw_end:
            _ax_r.axvspan(odw.odw_start, odw.odw_end, alpha=0.12, color="#C0111A",
                         label=f"ODW τ={odw.odw_start}–{odw.odw_end}")
        if odw.tau_h_plus:
            _ax_r.axvline(odw.tau_h_plus, color="#C0111A", ls="--", lw=1.2, alpha=0.7,
                         label=f"τ⁺ₕ = {odw.tau_h_plus}")
        if odw.tau_f:
            _ax_r.axvline(odw.tau_f, color="#FF8C00", ls=":", lw=1.2, alpha=0.7,
                         label=f"τf = {odw.tau_f}")

        _ax_r.set_xlabel("τ (lag, steps)", fontsize=8, color="#7A90B8")
        _ax_r.set_ylabel("MI (bits)", fontsize=8, color="#7A90B8")
        _ax_r.set_title(f"FAWP Dynamic Systems — {r['domain']}",
                        color="#D4AF37", fontsize=10, fontweight="bold")
        # Gaussian channel fit overlay (VTM paper formula)
        try:
            import scipy.optimize as _sco
            def _gauss_mi(tau_arr, P, alpha, sigma0):
                return 0.5 * np.log2(1 + P / (sigma0 + alpha * tau_arr))
            _pm_pos = pred_mi[pred_mi > 0]
            _tau_pos = tau_arr[pred_mi > 0]
            if len(_pm_pos) >= 3:
                _p0 = [float(pred_mi.max()), 0.01, 0.01]
                _popt, _ = _sco.curve_fit(_gauss_mi, _tau_pos, _pm_pos,
                                          p0=_p0, maxfev=2000,
                                          bounds=([0,1e-6,1e-6],[10,10,10]))
                _fit_curve = _gauss_mi(tau_arr, *_popt)
                _ax_r.plot(tau_arr, _fit_curve, color="#1DB954", lw=1.2, ls="-.",
                          alpha=0.7, label=f"Gaussian fit (P={_popt[0]:.2f} α={_popt[1]:.4f})")
        except Exception:
            pass
        _ax_r.legend(fontsize=7, framealpha=0.2, loc="upper right")
        _fig_r.tight_layout()
        st.pyplot(_fig_r, use_container_width=True)
        _plt_r.close(_fig_r)
    except Exception as _me:
        st.caption(f"Chart: {_me}")

    # E9.7 callout
    try:
        import fawp_index as _fi
        if fawp_found:
            st.info(
                f"📐 **E9.7 calibration:** gap2 peak leads cliff by "
                f"+{_fi.E97_MEAN_LEAD_GAP2_TO_CLIFF_U:.4f} delays · "
                f"ODW localisation error ~{_fi.E97_MEAN_ABS_ERR_GAP2_VS_ODW_START:.1f} steps"
            )
    except Exception:
        pass

    # Colab launch button
    st.markdown("---")
    _colab_url = (
        "https://colab.research.google.com/github/DrRalphClayton/"
        "fawp-index/blob/main/notebooks/E9_full_replication.ipynb"
    )
    st.markdown(
        f'''<a href="{_colab_url}" target="_blank">
<img src="https://colab.research.google.com/assets/colab-badge.svg"
     alt="Open in Colab" style="height:28px;vertical-align:middle;margin-right:.5em">
</a>
<span style="color:#5A8ABA;font-size:.85em">Open the E9 replication notebook in Colab
— pre-loads the same AR(1) simulation used here</span>''',
        unsafe_allow_html=True
    )
    st.markdown("")

    # Export
    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        _exp = {
            "fawp_found": fawp_found, "peak_gap_bits": round(peak_gap, 6),
            "tau_h_plus": odw.tau_h_plus, "tau_f": odw.tau_f,
            "odw_start": odw.odw_start, "odw_end": odw.odw_end,
            "epsilon": epsilon, "domain": r["domain"], "n_obs": r["n_obs"],
            "pred_mi": r["pred_mi"].tolist(), "steer_mi": r["steer_mi"].tolist(),
            "tau": r["tau"].tolist(),
        }
        import json
        st.download_button("⬇ Export JSON", data=json.dumps(_exp, indent=2),
                           file_name="fawp_dynamo_result.json", mime="application/json")
    with exp_col2:
        _csv_out = pd.DataFrame({"tau": r["tau"],
                                  "pred_mi": r["pred_mi"],
                                  "steer_mi": r["steer_mi"]})
        st.download_button("⬇ Export CSV", data=_csv_out.to_csv(index=False).encode(),
                           file_name="fawp_dynamo_mi.csv", mime="text/csv")

    # Agency horizon calculator
    st.markdown("---")
    st.markdown("#### 🔬 Agency Horizon Calculator (Gaussian channel model)")
    st.caption("From the VTM paper: τₕ = P/α(2²ᵉ − 1) − σ²₀/α")
    ah_c1, ah_c2, ah_c3, ah_c4 = st.columns(4)
    with ah_c1: _P   = st.number_input("P (action variance)", 0.01, 100.0, 1.0, key="ah_P")
    with ah_c2: _alp = st.number_input("α (noise growth rate)", 0.0001, 1.0, 0.001,
                                        format="%.4f", key="ah_alpha")
    with ah_c3: _s0  = st.number_input("σ²₀ (baseline noise)", 0.0, 1.0, 0.0001,
                                        format="%.4f", key="ah_s0")
    with ah_c4: _eps = st.number_input("ε (threshold bits)", 0.001, 0.1, 0.01,
                                        format="%.3f", key="ah_eps")
    _tau_h_calc = _P / (_alp * (2 ** (2 * _eps) - 1)) - _s0 / _alp
    _color_ah = "#1DB954" if _tau_h_calc > 10 else "#D4AF37" if _tau_h_calc > 3 else "#C0111A"
    st.markdown(f"""<div style="text-align:center;padding:.6em;background:#0D1729;
border:1px solid #1E3050;border-radius:6px;margin-top:.4em">
<span style="font-size:1.4em;font-weight:800;color:{_color_ah}">τₕ = {_tau_h_calc:.2f} steps</span>
<span style="color:#5A8ABA;font-size:.82em;margin-left:.8em">agency horizon</span>
</div>""", unsafe_allow_html=True)

