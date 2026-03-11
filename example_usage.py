"""
fawp-index: Example usage with synthetic E8-style data
"""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/home/claude/fawp_index')

from fawp_index import FAWPAlphaIndex, FAWPStreamDetector
from fawp_index.io.csv_loader import load_csv_simple

# ── 1. Generate synthetic E8-style data (pooled across trials) ────────────────
print("Generating synthetic E8-style data (400 trials)...")
np.random.seed(42)

a, K, sigma_proc = 1.02, 0.8, 1.0
n_steps, n_trials = 600, 400
delta_pred = 20
tau_true = 4  # true delay

all_states, all_futures, all_actions, all_obs = [], [], [], []

for trial in range(n_trials):
    rng = np.random.default_rng(trial)
    x_hist = np.zeros(n_steps + delta_pred + tau_true + 5)
    x = 0.0
    x_hist[0] = x
    states_t, actions_t = [], []
    failed = False

    for t in range(n_steps):
        td = t - tau_true
        if td < 0:
            y_del = 0.0
        else:
            y_del = x_hist[td] + rng.normal(0, 0.1)
        u = float(np.clip(-K * y_del, -10, 10))
        states_t.append(x)
        actions_t.append(u)
        x = a * x + u + rng.normal(0, sigma_proc)
        x_hist[t + 1] = x
        if abs(x) > 500:
            failed = True
            break

    t_end = len(states_t)
    # extend x_hist for future targets
    x_curr = x_hist[t_end]
    for k in range(t_end, n_steps + delta_pred):
        x_curr = a * x_curr + rng.normal(0, sigma_proc)
        x_hist[k + 1] = x_curr

    # build aligned arrays
    for t in range(min(t_end, n_steps - delta_pred)):
        all_states.append(states_t[t])
        all_futures.append(x_hist[t + delta_pred])
        all_actions.append(actions_t[t])
        # obs for steering: O_{t+tau+1} = x_{t+1} measured with noise
        all_obs.append(x_hist[t + 1] + rng.normal(0, 0.1))

# Save to CSV
df = pd.DataFrame({
    "state":   all_states,
    "future":  all_futures,
    "action":  all_actions,
    "obs":     all_obs,
})
df.to_csv("/tmp/e8_style_data.csv", index=False)
print(f"Saved {len(df)} rows to /tmp/e8_style_data.csv")

# ── 2. Run Alpha Index ────────────────────────────────────────────────────────
print("\nComputing FAWP Alpha Index...")
from fawp_index.io.csv_loader import load_csv

data = load_csv(
    "/tmp/e8_style_data.csv",
    pred_col="state",
    future_col="future",
    action_col="action",
    obs_col="obs",
    delta_pred=delta_pred,
)

detector = FAWPAlphaIndex(
    eta=1e-4,
    epsilon=1e-4,
    m_persist=3,
    kappa=1.0,
    n_null=100,
)

result = detector.compute(
    pred_series=data.pred_series,
    future_series=data.future_series,
    action_series=data.action_series,
    obs_series=data.obs_series,
    tau_grid=list(range(1, 16)),
    verbose=True,
)

print("\n" + result.summary())
print(f"\nRaw pred MI:  {np.round(result.pred_mi_raw, 3)}")
print(f"Raw steer MI: {np.round(result.steer_mi_raw, 3)}")
print(f"Alpha Index:  {np.round(result.alpha_index, 4)}")
print(f"In FAWP:      {result.in_fawp}")

# ── 3. Live stream ────────────────────────────────────────────────────────────
print("\n--- Live stream demo ---")
fawp_count = 0

def on_fawp(r):
    global fawp_count
    fawp_count += 1
    if fawp_count <= 3:
        print(f"  ⚠️  FAWP detected at step {stream.step}: tau_h={r.tau_h}, peak_alpha={r.peak_alpha:.4f}")

stream = FAWPStreamDetector(
    window=400,
    delta_pred=delta_pred,
    tau_grid=list(range(1, 10)),
    min_samples=150,
    n_null=50,
    on_fawp=on_fawp,
)

stream.update_batch(all_states[:600], all_actions[:600])
print(f"Stream complete. Steps: {stream.step}, FAWP detections: {fawp_count}")
print("\n✅ fawp-index working correctly!")
