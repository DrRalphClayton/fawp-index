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

# ── Inherit dark/light theme from app.py via query params ────────────────────
_ctrl_theme = st.query_params.get("theme", "dark")
if _ctrl_theme == "light":
    st.markdown(
        "<script>document.documentElement.setAttribute('data-theme','light')</script>",
        unsafe_allow_html=True,
    )
# Dark mode CSS baseline for Dynamic Systems scanner
st.markdown("""
<style>
:root { --bg-deep:#07101E; --bg-card:#0D1729; --border:#1E3050;
        --gold:#D4AF37; --blue:#4A7FCC; --text:#EDF0F8; --sub:#7A90B8; }
[data-theme="light"] { --bg-deep:#F8FAFB; --bg-card:#FFFFFF;
    --border:#D0DCE8; --gold:#A07800; --blue:#2860AA;
    --text:#0A1020; --sub:#4A5E7A; }
.stApp { background: var(--bg-deep); color: var(--text); }
.stButton>button { background: var(--gold); color: #07101E; font-weight:700; border:none; }
.stButton>button:hover { opacity:.88; }
.stSelectbox label, .stSlider label, .stRadio label,
.stNumberInput label, .stFileUploader label { color: var(--sub) !important; }
</style>
""", unsafe_allow_html=True)


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
                        "τ⁺ₕ": _odw_b.tau_h_plus if _odw_b.tau_h_plus is not None else "—", "τf": _odw_b.tau_f if _odw_b.tau_f is not None else "—",
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
    tau_h      = odw.tau_h_plus if odw.tau_h_plus is not None else "—"
    tau_f      = odw.tau_f      if odw.tau_f      is not None else "—"
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
        if odw.tau_f is not None:
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

    # Triple Horizon visualiser — SPHERE_23
    try:
        import matplotlib.pyplot as _plt_th, matplotlib.patches as _mp_th
        import fawp_index as _fi_th
        _fig_th, _ax_th = _plt_th.subplots(figsize=(9, 1.8), facecolor="#0D1729")
        _ax_th.set_facecolor("#07101E")
        for _sp in _ax_th.spines.values(): _sp.set_edgecolor("#3A4E70")
        _ax_th.tick_params(colors="#7A90B8", labelsize=8)
        _ax_th.set_xlim(0, max(tau_arr[-1], _fi_th.E11_TAU_READOUT + 5))
        _ax_th.set_ylim(0, 1); _ax_th.set_yticks([])
        # FAWP window shading (pred > eps, steer < eps)
        _fawp_mask = (pred_mi > epsilon) & (steer_mi <= epsilon)
        _in_fawp = False; _fawp_start = None
        for _ti, _tau in enumerate(tau_arr):
            if _fawp_mask[_ti] and not _in_fawp:
                _fawp_start = _tau; _in_fawp = True
            elif not _fawp_mask[_ti] and _in_fawp:
                _ax_th.axvspan(_fawp_start, _tau, alpha=0.18, color="#C0111A", zorder=1)
                _in_fawp = False
        if _in_fawp: _ax_th.axvspan(_fawp_start, tau_arr[-1], alpha=0.18, color="#C0111A", zorder=1)
        # Empirical horizons from scan
        if odw.tau_h_plus:
            _ax_th.axvline(odw.tau_h_plus, color="#FF6B2B", lw=2, zorder=3,
                          label=f"τ⁺ₕ={odw.tau_h_plus} (steering horizon)")
        if odw.tau_f is not None:
            _ax_th.axvline(odw.tau_f, color="#C0111A", lw=2, ls="--", zorder=3,
                          label=f"τf={odw.tau_f} (functional horizon)")
        # Reference constants from SPHERE_23 (E11-1 baseline)
        _ax_th.axvline(_fi_th.E11_TAU_ALPHA,   color="#D4AF37", lw=1.2, ls=":",
                      label=f"τα={_fi_th.E11_TAU_ALPHA} (steer wall)")
        _ax_th.axvline(_fi_th.E11_TAU_ALPHA2,  color="#4A7FCC", lw=1.2, ls="-.",
                      label=f"τα²={_fi_th.E11_TAU_ALPHA2} (residual floor)")
        _ax_th.axvline(_fi_th.E11_TAU_READOUT, color="#1DB954", lw=1.2, ls=":",
                      label=f"τread={_fi_th.E11_TAU_READOUT} (readout horizon)")
        _ax_th.set_xlabel("τ (delay)", fontsize=8, color="#7A90B8")
        _ax_th.set_title("Triple Horizon Framework — SPHERE_23 reference boundaries",
                        color="#D4AF37", fontsize=9, fontweight="bold")
        _ax_th.legend(fontsize=6.5, framealpha=0.25, loc="upper right",
                     ncol=3, facecolor="#0D1729", edgecolor="#3A4E70")
        _fig_th.tight_layout(pad=0.4)
        st.pyplot(_fig_th, use_container_width=True)
        _plt_th.close(_fig_th)
        st.caption("🔴 Shaded = FAWP zone · Reference lines from E11-1 baseline (SPHERE_23)")
    except Exception as _the:
        st.caption(f"Triple Horizon visualiser: {_the}")

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

    # FAWP Regime Breadth Map (E9.5 interactive)
    with st.expander("🗺 Regime Breadth Map — sweep (a, K) grid", expanded=False):
        st.caption("Reproduce E9.5 coarse regime map. Sweeps unstable growth rate a × controller gain K.")
        _rb_c1, _rb_c2 = st.columns(2)
        with _rb_c1:
            _rb_a = st.multiselect("Growth rates a",
                [1.005,1.01,1.015,1.02,1.03,1.04],
                default=[1.01,1.02,1.03], key="rb_a")
            _rb_K = st.multiselect("Controller gains K",
                [0.4,0.6,0.8,1.0,1.2],
                default=[0.4,0.8,1.2], key="rb_K")
        with _rb_c2:
            _rb_eps   = st.number_input("ε", 0.001, 0.1, 0.01, format="%.3f", key="rb_eps")
            _rb_seeds = st.slider("Seeds/config", 1, 8, 4, key="rb_seeds")
            _rb_tau   = st.slider("τ max", 20, 80, 40, key="rb_tau")

        if st.button("▶ Run regime map", key="rb_run", type="primary"):
            from fawp_index.core.estimators import mi_from_arrays, conservative_null_floor
            from fawp_index.detection.odw import ODWDetector
            from fawp_index.constants import PERSISTENCE_RULE_M, PERSISTENCE_RULE_N
            _rb_rows = []
            _rb_prog = st.progress(0.0, "Scanning regime grid…")
            _rb_total = len(_rb_a) * len(_rb_K) * _rb_seeds
            _rb_done  = 0
            for _a_v in (_rb_a or [1.02]):
                for _K_v in (_rb_K or [0.8]):
                    _fawp_count = 0
                    for _seed in range(_rb_seeds):
                        try:
                            _rng_rb = np.random.default_rng(_seed + 100)
                            _n_rb = 500; _tau_rb = np.arange(1, _rb_tau+1)
                            _x_rb = np.zeros(_n_rb); _u_rb = np.zeros(_n_rb)
                            for _t in range(1, _n_rb):
                                _obs = _x_rb[max(0,_t-20)]
                                _u_rb[_t] = np.clip(-_K_v*_obs, -10, 10)
                                _x_rb[_t] = _a_v*_x_rb[_t-1]+_u_rb[_t]+_rng_rb.normal(0,.1)
                                if abs(_x_rb[_t])>500: _x_rb[_t]=np.sign(_x_rb[_t])*500
                            _pm_rb = np.zeros(len(_tau_rb)); _sm_rb = np.zeros(len(_tau_rb))
                            for _ti,_tau in enumerate(_tau_rb):
                                _xp=_x_rb[:-_tau]; _yp=_x_rb[_tau:]
                                _pm_rb[_ti]=max(0.,mi_from_arrays(_xp,_yp)-conservative_null_floor(_xp,_yp,20,.99))
                                _xs=_u_rb[:-_tau]; _ys=_x_rb[_tau:]
                                _sm_rb[_ti]=max(0.,mi_from_arrays(_xs,_ys)-conservative_null_floor(_xs,_ys,20,.99))
                            _odw_rb = ODWDetector(epsilon=_rb_eps,
                                persistence_m=PERSISTENCE_RULE_M,
                                persistence_n=PERSISTENCE_RULE_N).detect(
                                tau=_tau_rb, pred_corr=_pm_rb,
                                steer_corr=_sm_rb,
                                fail_rate=np.zeros(len(_tau_rb)))
                            if _odw_rb.fawp_found: _fawp_count += 1
                        except Exception:
                            pass
                        _rb_done += 1
                        _rb_prog.progress(_rb_done/_rb_total)
                    _rb_rows.append({"a":_a_v,"K":_K_v,
                                     "FAWP rate":_fawp_count/_rb_seeds,
                                     "Seeds":_rb_seeds})
            _rb_prog.empty()
            st.session_state["rb_results"] = _rb_rows

        if "rb_results" in st.session_state and st.session_state["rb_results"]:
            if HAS_MPL:
                _rb_df = pd.DataFrame(st.session_state["rb_results"])
                _a_u = sorted(_rb_df["a"].unique())
                _K_u = sorted(_rb_df["K"].unique())
                _mat = [[_rb_df[(_rb_df["a"]==_av)&(_rb_df["K"]==_kv)]["FAWP rate"].mean()
                         for _kv in _K_u] for _av in _a_u]
                _fig_rb,_ax_rb=_plt.subplots(figsize=(max(4,len(_K_u)*1.2),
                                                       max(3,len(_a_u)*0.8)),
                                             facecolor="#0D1729")
                _ax_rb.set_facecolor("#07101E")
                import numpy as _np_rb
                _im = _ax_rb.imshow(_np_rb.array(_mat), cmap="RdYlGn_r",
                                   vmin=0, vmax=1, aspect="auto")
                _plt.colorbar(_im, ax=_ax_rb, label="FAWP detection rate")
                _ax_rb.set_xticks(range(len(_K_u))); _ax_rb.set_yticks(range(len(_a_u)))
                _ax_rb.set_xticklabels([str(k) for k in _K_u],
                                       fontsize=8, color="#EDF0F8")
                _ax_rb.set_yticklabels([str(a) for a in _a_u],
                                       fontsize=8, color="#EDF0F8")
                _ax_rb.set_xlabel("K (controller gain)", fontsize=8, color="#7A90B8")
                _ax_rb.set_ylabel("a (growth rate)", fontsize=8, color="#7A90B8")
                _ax_rb.set_title("E9.5 Regime Breadth Map — FAWP detection rate",
                                color="#D4AF37", fontsize=9)
                for _sp in _ax_rb.spines.values(): _sp.set_edgecolor("#3A4E70")
                _fig_rb.tight_layout(); st.pyplot(_fig_rb, use_container_width=True)
                _plt.close(_fig_rb)



    # LERI Readout Chain Visualiser
    with st.expander("🔬 LERI Readout Chain (X→R→Y_τ→D_τ)", expanded=False):
        st.caption("Estimates how much of the original signal survives at each delay. "
                   "Based on LERI paper: Clayton (2026).")
        _lr_c1, _lr_c2 = st.columns(2)
        with _lr_c1:
            _lr_sig0 = st.number_input("Baseline noise σ²₀", 0.001, 2.0, 1.0, format="%.3f", key="lr_sig0")
            _lr_alpha = st.number_input("Noise growth α",    0.001, 2.0, 0.25, format="%.3f", key="lr_alpha")
        with _lr_c2:
            _lr_P     = st.number_input("Signal power P",    0.01,  10., 1.0,  format="%.2f", key="lr_P")
            _lr_eps   = st.number_input("Threshold ε (bits)",0.001, 0.1, 0.01, format="%.3f", key="lr_eps")
            _lr_tau   = st.slider("τ max", 50, 500, 300, key="lr_tau")

        if st.button("▶ Compute readout chain", key="lr_run", type="primary"):
            import numpy as _np_lr
            _tau_lr = _np_lr.arange(0, _lr_tau + 1)
            _sigma2 = _lr_sig0 + _lr_alpha * _tau_lr
            _I_tau  = 0.5 * _np_lr.log2(1 + _lr_P / _sigma2)
            _I_tau  = _np_lr.maximum(0, _I_tau)
            # Analytic horizon (LERI Eq. 15-16)
            _denom = 2**(2*_lr_eps) - 1
            _tau_h = max(0.0, (_lr_P/_denom - _lr_sig0) / _lr_alpha) if _lr_alpha > 0 else float('inf')

            if HAS_MPL:
                _fig_lr, _ax_lr = _plt.subplots(figsize=(8, 3.5), facecolor="#0D1729")
                _ax_lr.set_facecolor("#07101E")
                _ax_lr.plot(_tau_lr, _I_tau, color="#4A7FCC", lw=1.5, label="I(X;D_τ) Gaussian channel")
                _ax_lr.axhline(_lr_eps, color="#C0111A", ls="--", lw=1, label=f"ε={_lr_eps:.3f}")
                if _tau_h < _lr_tau:
                    _ax_lr.axvline(_tau_h, color="#D4AF37", ls=":", lw=1.5,
                                  label=f"τ⁺ₕ≈{_tau_h:.1f} (analytic)")
                # Shade below threshold
                _ax_lr.fill_between(_tau_lr, _I_tau, 0,
                                   where=_I_tau <= _lr_eps, alpha=0.15, color="#C0111A",
                                   label="Below detectability")
                for _sp in _ax_lr.spines.values(): _sp.set_edgecolor("#3A4E70")
                _ax_lr.tick_params(colors="#7A90B8", labelsize=8)
                _ax_lr.set_xlabel("τ (delay)", fontsize=8, color="#7A90B8")
                _ax_lr.set_ylabel("Mutual information (bits)", fontsize=8, color="#7A90B8")
                _ax_lr.set_title("LERI Readout Chain — Operational Access Horizon",
                                color="#D4AF37", fontsize=9)
                _ax_lr.legend(fontsize=7, framealpha=0.2, facecolor="#0D1729")
                _fig_lr.tight_layout()
                st.pyplot(_fig_lr, use_container_width=True)
                _plt.close(_fig_lr)

            _alive = _I_tau[_I_tau > _lr_eps]
            st.metric("Analytic τ⁺ₕ", f"{_tau_h:.1f}" if _tau_h < _lr_tau else ">τ_max")
            st.caption(f"Signal readable for {len(_alive)}/{len(_tau_lr)} delays · "
                       f"σ²(τₕ) ≈ {(_lr_sig0 + _lr_alpha*_tau_h):.3f}")

    # Agency Capacity Surface — P × α → τₕ
    with st.expander("📡 Agency Capacity Surface (SPHERE_23)", expanded=False):
        st.caption("Heatmap of agency horizon τₕ = max(0, (P/(2^(2ε)−1) − σ²₀) / α) over P × α space.")
        _ag_c1, _ag_c2, _ag_c3 = st.columns(3)
        with _ag_c1:
            _ag_eps   = st.number_input("ε (bits)", 0.001, 0.1, 0.01, format="%.3f", key="ag_eps")
            _ag_sig0  = st.number_input("σ²₀", 0.0, 1.0, 0.0001, format="%.4f", key="ag_sig0")
        with _ag_c2:
            _ag_pmin  = st.number_input("P min", 0.01, 10.0, 0.1,  key="ag_pmin")
            _ag_pmax  = st.number_input("P max", 0.1,  100., 10.0, key="ag_pmax")
        with _ag_c3:
            _ag_amin  = st.number_input("α min", 1e-4, 0.5, 0.001, format="%.4f", key="ag_amin")
            _ag_amax  = st.number_input("α max", 1e-3, 1.0, 0.1,   format="%.3f", key="ag_amax")
            _ag_steps = st.slider("Grid steps", 10, 60, 30, key="ag_steps")

        if st.button("▶ Compute surface", key="ag_run", type="primary"):
            import numpy as _np_ag
            _P_v  = _np_ag.logspace(_np_ag.log10(_ag_pmin), _np_ag.log10(_ag_pmax), _ag_steps)
            _A_v  = _np_ag.logspace(_np_ag.log10(_ag_amin), _np_ag.log10(_ag_amax), _ag_steps)
            _PP, _AA = _np_ag.meshgrid(_P_v, _A_v)
            _denom = 2**(2*_ag_eps) - 1
            _TH = _np_ag.maximum(0, (_PP/_denom - _ag_sig0) / _AA)
            st.session_state["ag_surface"] = (_P_v, _A_v, _TH)

        if "ag_surface" in st.session_state:
            _P_v, _A_v, _TH = st.session_state["ag_surface"]
            if HAS_MPL:
                _fig_ag, _ax_ag = _plt.subplots(figsize=(7, 5), facecolor="#0D1729")
                _ax_ag.set_facecolor("#07101E")
                import numpy as _np_ag2
                _im_ag = _ax_ag.contourf(_np_ag2.log10(_P_v), _np_ag2.log10(_A_v),
                                         _TH, levels=25, cmap="RdYlGn")
                _plt.colorbar(_im_ag, ax=_ax_ag, label="Agency horizon τₕ (steps)")
                _ax_ag.set_xlabel("log₁₀ P (signal power)", fontsize=8, color="#7A90B8")
                _ax_ag.set_ylabel("log₁₀ α (noise growth)", fontsize=8, color="#7A90B8")
                _ax_ag.set_title("Agency Capacity Surface — VTM Eq. 15–16",
                                color="#D4AF37", fontsize=9)
                for _sp in _ax_ag.spines.values(): _sp.set_edgecolor("#3A4E70")
                _ax_ag.tick_params(colors="#7A90B8", labelsize=7)
                _fig_ag.tight_layout()
                st.pyplot(_fig_ag, use_container_width=True)
                _plt.close(_fig_ag)
                st.caption(f"Max τₕ = {_TH.max():.0f}  ·  "
                           f"at P={_P_v[_np_ag2.unravel_index(_TH.argmax(),_TH.shape)[1]]:.2f}, "
                           f"α={_A_v[_np_ag2.unravel_index(_TH.argmax(),_TH.shape)[0]]:.4f}")

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

